"""
LLM/embedding token ve maliyet raporu. Her çağrı cache/llm_log.jsonl'e yazılır.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/12_cost_report.py

Girdi:  cache/llm_log.jsonl
        cache/embed_log.jsonl
Çıktı:  data/eval/cost_report.md
Önkoşul: 05/06/07a/10 çağrıları bu logger'ı kullanır (aşağıdaki fonksiyonlar).

Neden jsonl: satır satır eklenir; bir koşu bozulsa öncekiler durur.
Neden sabit fiyat tablosu: OpenAI faturası buradan tahmin (gpt-4o-mini / 3-small).

Fiyat (USD / 1M token, 2026-09):
  gpt-4o-mini input $0.15  output $0.60
  text-embedding-3-small $0.02
Cache HIT maliyeti $0. Cache'siz sütun: aynı token'ı her seferinde ödemiş olsak.

recall@k / faithfulness: bu script retrieval ölçmez; n/a.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_LOG = PROJECT_ROOT / "cache" / "llm_log.jsonl"
EMBED_LOG = PROJECT_ROOT / "cache" / "embed_log.jsonl"
REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "cost_report.md"

# USD / 1_000_000 token
PRICES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

STAGE_LABELS = {
    "entity_extraction": "Entity extraction",
    "query": "Query",
    "metadata": "Metadata filter",
    "evaluate": "Evaluate / judge",
    "hybrid": "Hybrid RAG",
    "embed_index": "Embedding index",
    "embed_query": "Embedding query",
}


def estimate_tokens(text: str) -> int:
    """tiktoken yok; ~4 karakter ≈ 1 token (öğrenci tahmini)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def usd_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    table = PRICES.get(model) or PRICES["gpt-4o-mini"]
    return (input_tokens * table["input"] + output_tokens * table["output"]) / 1_000_000.0


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_llm_call(
    *,
    stage: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
) -> dict[str, Any]:
    """Tek kelime: her chat completion buraya düşer."""
    paid = 0.0 if cache_hit else usd_cost(model, input_tokens, output_tokens)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost": round(paid, 8),
        "cache_hit": bool(cache_hit),
    }
    try:
        _append_jsonl(LLM_LOG, record)
    except Exception as exc:
        print(f"WARN  llm_log yazılamadı: {exc}")
    return record


def log_embed_call(
    *,
    stage: str,
    model: str,
    input_tokens: int,
    cache_hit: bool,
    n_items: int = 1,
) -> dict[str, Any]:
    paid = 0.0 if cache_hit else usd_cost(model, input_tokens, 0)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": 0,
        "cost": round(paid, 8),
        "cache_hit": bool(cache_hit),
        "n_items": int(n_items),
    }
    try:
        _append_jsonl(EMBED_LOG, record)
    except Exception as exc:
        print(f"WARN  embed_log yazılamadı: {exc}")
    return record


def logged_chat(
    client: Any,
    cache: Any,
    prompt: str,
    model: str,
    cache_key: str,
    stage: str,
) -> str:
    """diskcache + jsonl. HIT'de API yok, maliyet 0; token tahmini."""
    cached = cache.get(cache_key)
    if cached is not None:
        text = str(cached)
        log_llm_call(
            stage=stage,
            model=model,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            cache_hit=True,
        )
        return text

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    if usage is not None:
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    else:
        in_tok = estimate_tokens(prompt)
        out_tok = estimate_tokens(text)
    cache.set(cache_key, text)
    log_llm_call(
        stage=stage,
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_hit=False,
    )
    return text


def logged_embed(
    client: Any,
    cache: Any,
    text: str,
    model: str,
    cache_key: str,
    stage: str,
) -> list[float]:
    cached = cache.get(cache_key)
    if cached is not None:
        log_embed_call(
            stage=stage,
            model=model,
            input_tokens=estimate_tokens(text),
            cache_hit=True,
        )
        return cached

    response = client.embeddings.create(model=model, input=text)
    vector = response.data[0].embedding
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
    if tokens <= 0:
        tokens = estimate_tokens(text)
    cache.set(cache_key, vector)
    log_embed_call(stage=stage, model=model, input_tokens=tokens, cache_hit=False)
    return vector


def logged_embed_batch(
    client: Any,
    texts: list[str],
    model: str,
    stage: str = "embed_index",
) -> tuple[list[list[float]], int]:
    """04_embed batch: cache dışı. usage.total_tokens varsa onu yaz."""
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = [item.embedding for item in ordered]
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if tokens <= 0:
        tokens = sum(estimate_tokens(text) for text in texts)
    log_embed_call(
        stage=stage,
        model=model,
        input_tokens=tokens,
        cache_hit=False,
        n_items=len(texts),
    )
    return vectors, tokens


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    by_stage: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "tokens": 0,
            "cost": 0.0,
            "cost_no_cache": 0.0,
            "hits": 0,
            "misses": 0,
        }
    )
    for row in rows:
        stage = str(row.get("stage") or "unknown")
        bucket = by_stage[stage]
        in_tok = _num(row.get("input_tokens"))
        out_tok = _num(row.get("output_tokens"))
        tokens = in_tok + out_tok
        model = str(row.get("model") or "gpt-4o-mini")
        hit = bool(row.get("cache_hit"))
        bucket["calls"] += 1
        bucket["tokens"] += tokens
        bucket["cost"] += _num(row.get("cost"))
        bucket["cost_no_cache"] += usd_cost(model, int(in_tok), int(out_tok))
        if hit:
            bucket["hits"] += 1
        else:
            bucket["misses"] += 1
    return by_stage


def fmt_money(value: float) -> str:
    if value < 0.0001:
        return f"${value:.6f}"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def write_report(llm_rows: list[dict], embed_rows: list[dict]) -> str:
    llm = aggregate(llm_rows)
    embed = aggregate(embed_rows)

    lines = [
        "# Maliyet raporu",
        "",
        f"Üretilme: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Fiyat varsayımı (1M token): `gpt-4o-mini` $0.15 in / $0.60 out; "
        "`text-embedding-3-small` $0.02. Cache HIT = $0.",
        "",
        "## LLM",
        "",
        "| Aşama | LLM çağrı | Token | Maliyet (cache ile) |",
        "|-------|-----------|-------|---------------------|",
    ]

    total_calls = 0
    total_tokens = 0.0
    total_cost = 0.0
    total_no_cache = 0.0
    for stage in sorted(llm):
        bucket = llm[stage]
        label = STAGE_LABELS.get(stage, stage)
        total_calls += int(bucket["calls"])
        total_tokens += bucket["tokens"]
        total_cost += bucket["cost"]
        total_no_cache += bucket["cost_no_cache"]
        lines.append(
            f"| {label} | {int(bucket['calls'])} | {int(bucket['tokens'])} | "
            f"{fmt_money(bucket['cost'])} |"
        )
    if not llm:
        lines.append("| (log boş) | 0 | 0 | $0.00 |")
    lines.append(
        f"| Toplam | {total_calls} | {int(total_tokens)} | {fmt_money(total_cost)} |"
    )

    embed_calls = sum(int(b["calls"]) for b in embed.values())
    embed_hits = sum(int(b["hits"]) for b in embed.values())
    embed_misses = sum(int(b["misses"]) for b in embed.values())
    embed_tokens = sum(b["tokens"] for b in embed.values())
    embed_cost = sum(b["cost"] for b in embed.values())
    embed_no_cache = sum(b["cost_no_cache"] for b in embed.values())
    hit_den = embed_hits + embed_misses
    hit_pct = (100.0 * embed_hits / hit_den) if hit_den else 0.0

    lines.extend(
        [
            "",
            "## Embedding cache",
            "",
            f"- Çağrı: {embed_calls}  hit={embed_hits}  miss={embed_misses}  "
            f"hit oranı={hit_pct:.1f}%",
            f"- Token: {int(embed_tokens)}  maliyet (cache ile): {fmt_money(embed_cost)}",
            "",
            "## Cache yok vs cache var",
            "",
            "| | Cache ile (ödenen) | Cache yok (tahmin) | Tasarruf |",
            "|---|---|---|---|",
            f"| LLM | {fmt_money(total_cost)} | {fmt_money(total_no_cache)} | "
            f"{fmt_money(total_no_cache - total_cost)} |",
            f"| Embedding | {fmt_money(embed_cost)} | {fmt_money(embed_no_cache)} | "
            f"{fmt_money(embed_no_cache - embed_cost)} |",
            f"| Toplam | {fmt_money(total_cost + embed_cost)} | "
            f"{fmt_money(total_no_cache + embed_no_cache)} | "
            f"{fmt_money((total_no_cache + embed_no_cache) - (total_cost + embed_cost))} |",
            "",
            "Cache yok sütunu: logdaki her satırı MISS sayıp aynı token'a fiyat uygular. "
            "İndeks embedding'i bir kez yazıldıysa tekrar koşuda hit oranı yükselir.",
            "",
            f"Log: `{LLM_LOG.relative_to(PROJECT_ROOT)}`, "
            f"`{EMBED_LOG.relative_to(PROJECT_ROOT)}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    llm_rows = read_jsonl(LLM_LOG)
    embed_rows = read_jsonl(EMBED_LOG)
    print(f"LLM log: {len(llm_rows)} satır  ({LLM_LOG})")
    print(f"Embed log: {len(embed_rows)} satır  ({EMBED_LOG})")
    if not llm_rows and not embed_rows:
        print(
            "WARN  log boş. Önce 07a / 05 / 10 çalıştır "
            "(çağrılar 12_cost_report logger'ına yazıyor)."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = write_report(llm_rows, embed_rows)
    try:
        REPORT_PATH.write_text(text, encoding="utf-8")
        print(f"Kaydedildi: {REPORT_PATH}")
    except Exception as exc:
        print(f"HATA: rapor yazılamadı: {exc}")
        return

    llm = aggregate(llm_rows)
    embed = aggregate(embed_rows)
    print("\nLLM aşama:")
    for stage, bucket in sorted(llm.items()):
        print(
            f"  {STAGE_LABELS.get(stage, stage)}  "
            f"çağrı={int(bucket['calls'])}  token={int(bucket['tokens'])}  "
            f"ödenen={fmt_money(bucket['cost'])}  "
            f"cache'siz={fmt_money(bucket['cost_no_cache'])}"
        )
    emb_h = sum(int(b["hits"]) for b in embed.values())
    emb_m = sum(int(b["misses"]) for b in embed.values())
    den = emb_h + emb_m
    print(
        f"Embedding hit oranı: "
        f"{(100.0 * emb_h / den) if den else 0:.1f}%  ({emb_h} hit / {den} çağrı)"
    )
    print("Evaluation: recall@k=n/a  faithfulness=n/a  (maliyet raporu).")


if __name__ == "__main__":
    main()
