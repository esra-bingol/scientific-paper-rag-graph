"""
Aşama aşama RAG metrikleri: recall@k, faithfulness, relevance, latency.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/11_evaluate.py
    python3 src/11_evaluate.py --stages naive,hybrid_rrf --max-questions 3

Girdi:  tests/ground_truth.json
Çıktı:  data/eval/results.csv
        data/eval/summary.md
Önkoşul: 04_embed (naive+grobid), 08_build_graph, 10_hybrid_rag
         ground_truth içinde relevant_* dolu olmalı (şimdilik şablon).

Neden pandas: aşama satırlarını CSV/ortalama olarak yazmak.
Neden openai: cevap + LLM-as-judge (faithfulness/relevance).
Neden diskcache: aynı soru/judge tekrar ödenmesin.
Neden chromadb/networkx: naive/grobid/metadata/graph/hybrid retrieval.

recall@k: |gold ∩ top-k| / |gold|. Gold boşsa o soru ortalamaya girmez (n/a).
Faithfulness: iddia context'te var mı? Relevance: cevap soruya mı?
Latency: retrieval + cevap (judge hariç), ms.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chromadb
import pandas as pd
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GT_PATH = PROJECT_ROOT / "tests" / "ground_truth.json"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CSV_PATH = EVAL_DIR / "results.csv"
MD_PATH = EVAL_DIR / "summary.md"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBED_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"
LLM_CACHE_DIR = PROJECT_ROOT / "cache" / "llm"

CHAT_MODEL = "gpt-4o-mini"
RECALL_KS = (5, 10)
ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)

STAGES = ("naive", "grobid", "metadata", "graph", "hybrid_rrf")

SIMPLE_ANSWER = """Verilen parçalara dayanarak cevapla.
Cevap yoksa "Bu bilgi verilen makalelerde yok" de. Uydurma.

Parçalar:
{context}

Soru: {question}
"""

GRAPH_ANSWER = """Graf makale listesine dayanarak cevapla.
Yazar yok / liste boşsa "Bu bilgi verilen makalelerde yok" de. Uydurma.

Makaleler: {papers}
Path: {paths}

Soru: {question}
"""

FAITHFULNESS_PROMPT = """Sen sadakat hakemisin. Cevaptaki iddialar SADECE context'te var mı?
Uydurma iddia varsa düşük skor. "Bu bilgi yok" + boş/alakasız context → 1.0.
Sadece JSON: {{"score": 0.0 ile 1.0 arası sayı, "reason": "kısa"}}

Context:
{context}

Cevap:
{answer}
"""

RELEVANCE_PROMPT = """Sen alaka hakemisin. Cevap soruyu yanıtlıyor mu?
Konu dışıysa düşük. "yok" doğru abstain ise 1.0.
Sadece JSON: {{"score": 0.0 ile 1.0 arası sayı, "reason": "kısa"}}

Soru:
{question}

Cevap:
{answer}
"""


def load_src(filename: str, alias: str) -> ModuleType:
    path = PROJECT_ROOT / "src" / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} yüklenemedi")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ground_truth(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for key, value in raw.items():
        if key.startswith("_") or not isinstance(value, dict):
            continue
        question = str(value.get("question") or "").strip()
        if not question:
            continue
        rows.append(
            {
                "id": key,
                "question": question,
                "relevant_chunks": [
                    str(item).strip()
                    for item in (value.get("relevant_chunks") or [])
                    if str(item).strip()
                ],
                "relevant_papers": [
                    str(item).strip()
                    for item in (value.get("relevant_papers") or [])
                    if str(item).strip()
                ],
                "expected_answer_keywords": [
                    str(item).strip()
                    for item in (value.get("expected_answer_keywords") or [])
                    if str(item).strip()
                ],
            }
        )
    return rows


def bare_arxiv(token: str) -> str:
    text = (token or "").strip()
    if "#" in text:
        text = text.split("#", 1)[0]
    if re.match(r"^\d{4}\.", text) and "_" in text:
        text = text.split("_", 1)[0]
    return ARXIV_VERSION.sub("", text)


def gold_matches(gold_item: str, hits: list[dict]) -> bool:
    gold = gold_item.strip()
    gold_paper = bare_arxiv(gold)
    for hit in hits:
        chunk_id = str(hit.get("chunk_id") or "")
        arxiv_id = str(hit.get("arxiv_id") or "")
        if gold in {chunk_id, arxiv_id}:
            return True
        if gold_paper and gold_paper in {bare_arxiv(chunk_id), bare_arxiv(arxiv_id)}:
            return True
    return False


def recall_at_k(record: dict, hits: list[dict], k: int) -> float | None:
    golds = record["relevant_chunks"] + record["relevant_papers"]
    if not golds:
        return None
    top = hits[:k]
    found = sum(1 for item in golds if gold_matches(item, top))
    return found / len(golds)


def format_hits(hits: list[dict]) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits[:10], start=1):
        parts.append(
            f"[{index}] arxiv_id={hit.get('arxiv_id')} chunk_id={hit.get('chunk_id')}\n"
            f"{hit.get('text') or ''}"
        )
    return "\n\n".join(parts)


def mean_or_none(values: list[float | None]) -> float | None:
    filled = [item for item in values if item is not None]
    if not filled:
        return None
    return float(statistics.mean(filled))


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def get_collection(name: str) -> chromadb.Collection | None:
    if not CHROMA_DIR.exists():
        return None
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=name, embedding_function=None)
    except Exception:
        return None


def vector_stage(
    h10: ModuleType,
    collection: chromadb.Collection,
    client: OpenAI,
    embed_cache: Cache,
    llm_cache: Cache,
    question: str,
    k: int,
) -> tuple[list[dict], str]:
    raw = h10.vector_retrieve(collection, client, embed_cache, question, k)
    hits = raw["chunks"]
    prompt = SIMPLE_ANSWER.format(context=format_hits(hits[:5]) or "(boş)", question=question)
    answer = h10.chat_cached(client, llm_cache, prompt, stage="query")
    return hits, answer


def metadata_stage(
    h06: ModuleType,
    h10: ModuleType,
    collection: chromadb.Collection,
    client: OpenAI,
    embed_cache: Cache,
    llm_cache: Cache,
    question: str,
    k: int,
) -> tuple[list[dict], str]:
    filt = h06.normalize_filter(
        h06.extract_json_object(
            h10.chat_cached(
                client,
                llm_cache,
                h06.FILTER_PROMPT.format(question=question),
                stage="metadata",
            )
        )
    )
    where = h06.build_where(filt)
    query_vector = h10.embed_question(client, embed_cache, question)
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": min(max(k, 40), max(collection.count(), 1)),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    raw = collection.query(**kwargs)
    documents = (raw.get("documents") or [[]])[0] or []
    metadatas = (raw.get("metadatas") or [[]])[0] or []
    distances = (raw.get("distances") or [[]])[0] or []
    ids = (raw.get("ids") or [[]])[0] or []
    hits = h06.rows_to_hits(documents, metadatas, distances, ids)
    if filt["authors"]:
        hits = [hit for hit in hits if h06.author_matches(hit["authors"], filt["authors"])]
    catalog = h06.catalog_papers(collection, where, filt["authors"])
    catalog_text = ", ".join(
        f"{row['arxiv_id']} (year={row.get('year')})" for row in catalog
    ) or "(boş)"
    prompt = h06.ANSWER_PROMPT.format(
        filt=json.dumps(filt, ensure_ascii=False),
        catalog=catalog_text,
        context=h06.format_context(hits[:5]) or "(boş)",
        question=question,
    )
    answer = h10.chat_cached(client, llm_cache, prompt, stage="query")
    return hits[:k], answer


def graph_stage(
    h10: ModuleType,
    graph,
    client: OpenAI,
    llm_cache: Cache,
    question: str,
    k: int,
) -> tuple[list[dict], str]:
    entities = h10.normalize_entities(
        h10.extract_json_object(
            h10.chat_cached(
                client,
                llm_cache,
                h10.ENTITY_PROMPT.format(question=question),
                stage="entity_extraction",
            )
        )
    )
    result = h10.graph_retrieve(graph, entities)
    hits: list[dict] = []
    for row in result["papers"][:k]:
        hits.append(
            {
                "arxiv_id": row["arxiv_id"],
                "chunk_id": f"{row['arxiv_id']}#graph",
                "text": row.get("title") or "",
                "year": row.get("year"),
            }
        )
    papers = ", ".join(
        f"{row['arxiv_id']} (year={row.get('year')}) {row.get('title')}"
        for row in result["papers"][:k]
    ) or "(boş)"
    prompt = GRAPH_ANSWER.format(
        papers=papers,
        paths="; ".join(result["paths"][:12]) or "(yok)",
        question=question,
    )
    if result.get("notes"):
        prompt += "\nGraf not: " + "; ".join(result["notes"])
    answer = h10.chat_cached(client, llm_cache, prompt, stage="query")
    return hits, answer


def hybrid_stage(
    h10: ModuleType,
    graph,
    collection: chromadb.Collection,
    client: OpenAI,
    embed_cache: Cache,
    llm_cache: Cache,
    question: str,
    k: int,
) -> tuple[list[dict], str]:
    entities = h10.normalize_entities(
        h10.extract_json_object(
            h10.chat_cached(
                client,
                llm_cache,
                h10.ENTITY_PROMPT.format(question=question),
                stage="entity_extraction",
            )
        )
    )
    graph_hit = h10.graph_retrieve(graph, entities)
    vector_hit = h10.vector_retrieve(collection, client, embed_cache, question, max(k, 15))
    fused = h10.fuse_papers("rrf", graph_hit["ranked_ids"], vector_hit["ranked_ids"])
    graph_meta = {row["arxiv_id"]: row for row in graph_hit["papers"]}
    hits = h10.build_context_hits(fused, vector_hit["chunks"], graph_meta, k)
    graph_catalog = (
        "; ".join(
            f"{row['arxiv_id']} (year={row.get('year')}) {row.get('title') or ''}"
            for row in graph_hit["papers"]
        )
        or "(boş)"
    )
    prompt = h10.ANSWER_PROMPT.format(
        entities=json.dumps(entities, ensure_ascii=False),
        fusion="rrf",
        graph_papers=graph_catalog,
        graph_paths="; ".join(graph_hit["paths"][:12]) or "(boş)",
        context=h10.format_context(hits[:5]) or "(parça yok)",
        question=question,
    )
    answer = h10.chat_cached(client, llm_cache, prompt, stage="query")
    return hits, answer


def judge_score(
    h10: ModuleType,
    client: OpenAI,
    cache: Cache,
    prompt: str,
) -> float:
    raw = h10.chat_cached(client, cache, prompt, stage="evaluate")
    data = h10.extract_json_object(raw)
    score = data.get("score")
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score)))
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aşama aşama RAG eval.")
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help="virgülle: naive,grobid,metadata,graph,hybrid_rrf",
    )
    parser.add_argument("--max-questions", type=int, default=0, help="0 = hepsi")
    parser.add_argument("--skip-judge", action="store_true", help="faithfulness/relevance atla")
    return parser.parse_args()


def write_summary(frame: pd.DataFrame, gold_filled: bool) -> str:
    lines = [
        "# Eval özeti",
        "",
        "Aşamalar: naive → grobid → metadata → graph → hybrid_rrf (RRF, weighted sum yok).",
        "",
        "| stage | recall@5 | recall@10 | faithfulness | relevance | latency_ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['stage']} | {row['recall@5']} | {row['recall@10']} | "
            f"{row['faithfulness']} | {row['relevance']} | {row['latency_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Yorum",
            "",
        ]
    )
    if not gold_filled:
        lines.append(
            "- `recall@k` şu an **n/a**: `tests/ground_truth.json` içinde "
            "`relevant_chunks` / `relevant_papers` boş. Doldurunca aynı script'i tekrar çalıştır."
        )
    else:
        lines.append(
            "- Recall: gold id top-k'da mı? Chunk `arxiv#n` veya paper `arxiv_id` (sürüm toleranslı)."
        )
    lines.extend(
        [
            "- Faithfulness: LLM-as-judge; iddia context'te yoksa düşük. Relevance: cevap soruya mı bakıyor.",
            "- Naive yıl/yazar/ilişki kaçırır (baseline). GROBID kaynakçayı keser, cosine yine yıl bilmez.",
            "- Metadata `where` yıl listesini düzeltir; soyad hâlâ string.",
            "- Graph `author_id` hop'u yazar sorusunu çözer; gövde cümlesi yoktur.",
            "- Hybrid RRF iki sırayı birleştirir (`1/(60+rank)`). Weighted sum yok: ölçekler farklı.",
            "- Latency judge'ı içermez (sorgu = retrieval + cevap).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    wanted = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in STAGES]
    if unknown:
        print(f"HATA: bilinmeyen stage {unknown}. Seçenekler: {', '.join(STAGES)}")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("HATA: OPENAI_API_KEY yok.")
        return
    if not GT_PATH.exists():
        print(f"HATA: {GT_PATH} yok.")
        return

    try:
        questions = load_ground_truth(GT_PATH)
    except Exception as exc:
        print(f"HATA: ground_truth okunamadı: {exc}")
        return
    if args.max_questions > 0:
        questions = questions[: args.max_questions]
    if not questions:
        print("HATA: soru yok.")
        return

    gold_filled = any(row["relevant_chunks"] or row["relevant_papers"] for row in questions)
    print(f"Soru: {len(questions)}  gold_dolu={gold_filled}  stage={wanted}")
    if not gold_filled:
        print("WARN  relevant_* boş; recall@k n/a olacak. Şablonu doldur.")

    try:
        h10 = load_src("10_hybrid_rag.py", "hybrid_rag")
        h06 = load_src("06_metadata_filter.py", "metadata_filter")
    except Exception as exc:
        print(f"HATA: src yüklenemedi: {exc}")
        return

    client = OpenAI(api_key=api_key)
    naive_col = get_collection("papers_naive")
    grobid_col = get_collection("papers_grobid")
    graph = None
    try:
        graph = h10.load_graph()
        print(f"graf: {graph.number_of_nodes()} node")
    except Exception as exc:
        print(f"WARN  graf yok: {exc}")

    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    with Cache(str(EMBED_CACHE_DIR)) as embed_cache, Cache(str(LLM_CACHE_DIR)) as llm_cache:
        for stage in wanted:
            print(f"\n=== stage={stage} ===")
            rec5: list[float | None] = []
            rec10: list[float | None] = []
            faiths: list[float | None] = []
            rels: list[float | None] = []
            latencies: list[float] = []

            for record in tqdm(questions, desc=stage):
                qid = record["id"]
                question = record["question"]
                hits: list[dict] = []
                answer = ""
                started = time.perf_counter()
                try:
                    if stage == "naive":
                        if naive_col is None:
                            raise RuntimeError("papers_naive yok")
                        hits, answer = vector_stage(
                            h10, naive_col, client, embed_cache, llm_cache, question, 10
                        )
                    elif stage == "grobid":
                        if grobid_col is None:
                            raise RuntimeError("papers_grobid yok")
                        hits, answer = vector_stage(
                            h10, grobid_col, client, embed_cache, llm_cache, question, 10
                        )
                    elif stage == "metadata":
                        if grobid_col is None:
                            raise RuntimeError("papers_grobid yok")
                        hits, answer = metadata_stage(
                            h06, h10, grobid_col, client, embed_cache, llm_cache, question, 10
                        )
                    elif stage == "graph":
                        if graph is None:
                            raise RuntimeError("graf yok")
                        hits, answer = graph_stage(
                            h10, graph, client, llm_cache, question, 10
                        )
                    elif stage == "hybrid_rrf":
                        if graph is None or grobid_col is None:
                            raise RuntimeError("graf veya papers_grobid yok")
                        hits, answer = hybrid_stage(
                            h10, graph, grobid_col, client, embed_cache, llm_cache, question, 10
                        )
                    else:
                        raise RuntimeError(stage)
                except Exception as exc:
                    print(f"SKIP  {stage} {qid}: {exc}")
                    continue

                latency_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(latency_ms)
                r5 = recall_at_k(record, hits, 5)
                r10 = recall_at_k(record, hits, 10)
                rec5.append(r5)
                rec10.append(r10)

                faith = None
                rel = None
                if not args.skip_judge:
                    try:
                        context = format_hits(hits[:5]) or "(boş)"
                        faith = judge_score(
                            h10,
                            client,
                            llm_cache,
                            FAITHFULNESS_PROMPT.format(context=context, answer=answer),
                        )
                        rel = judge_score(
                            h10,
                            client,
                            llm_cache,
                            RELEVANCE_PROMPT.format(question=question, answer=answer),
                        )
                    except Exception as exc:
                        print(f"WARN  judge {stage} {qid}: {exc}")
                faiths.append(faith)
                rels.append(rel)

                print(
                    f"  {qid}  r@5={fmt_metric(r5)}  r@10={fmt_metric(r10)}  "
                    f"faith={fmt_metric(faith)}  rel={fmt_metric(rel)}  "
                    f"{latency_ms:.0f}ms  hits={len(hits)}"
                )

            row = {
                "stage": stage,
                "recall@5": fmt_metric(mean_or_none(rec5)),
                "recall@10": fmt_metric(mean_or_none(rec10)),
                "faithfulness": fmt_metric(mean_or_none(faiths)),
                "relevance": fmt_metric(mean_or_none(rels)),
                "latency_ms": f"{statistics.mean(latencies):.0f}" if latencies else "n/a",
            }
            summary_rows.append(row)
            print(
                f"ORT   {stage}  r@5={row['recall@5']}  r@10={row['recall@10']}  "
                f"faith={row['faithfulness']}  rel={row['relevance']}  "
                f"{row['latency_ms']}ms"
            )

    if not summary_rows:
        print("HATA: hiç stage ölçülmedi.")
        return

    frame = pd.DataFrame(summary_rows)
    try:
        frame.to_csv(CSV_PATH, index=False)
        MD_PATH.write_text(write_summary(frame, gold_filled), encoding="utf-8")
        print(f"Kaydedildi: {CSV_PATH}")
        print(f"Kaydedildi: {MD_PATH}")
    except Exception as exc:
        print(f"HATA: eval yazılamadı: {exc}")
        return

    print(frame.to_string(index=False))
    print(
        "Evaluation: recall@k="
        + ("n/a (gold boş)" if not gold_filled else "yukarıdaki tablo")
        + "  faithfulness="
        + ("n/a (judge atlandı)" if args.skip_judge else "LLM-as-judge")
        + "."
    )


if __name__ == "__main__":
    main()
