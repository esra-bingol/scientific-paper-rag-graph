"""
GROBID section JSON'larından LLM ile kaba entity çıkarır (henüz kanonik değil).

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/07a_extract_entities.py

Girdi:  data/processed_text/{arxiv_id}_sections.json
Çıktı:  data/entities_raw.json
Önkoşul: python3 src/02b_parse_grobid.py

Neden openai: gpt-4o-mini ile yöntem/eleştiri/topic çıkarımı; regex yetmez.
Neden diskcache: aynı makale metni tekrar ödenmesin (cache/llm).
Neden python-dotenv: OPENAI_API_KEY koda gömülmez.
Neden tqdm: 20 çağrıda hangi kağıtta olduğumuzu görmek için.

OpenAlex author_id burada yok; 07b entity resolution kanonikleştirecek.
recall@k / faithfulness: retrieval yok; n/a.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"
OUT_PATH = PROJECT_ROOT / "data" / "entities_raw.json"
LLM_CACHE_DIR = PROJECT_ROOT / "cache" / "llm"

CHAT_MODEL = "gpt-4o-mini"
# Bölüm başı tavan: bir çağrıda tüm kağıt, kaynakça yok.
MAX_SECTION_CHARS = 2500

EXTRACT_PROMPT = """You extract entities from ONE scientific paper. Output JSON only.

Schema:
{{
  "title": "string",
  "authors": ["Given Family", "..."],
  "methods_proposed": ["short method/model names this paper introduces or claims"],
  "methods_criticized": ["methods the paper says are insufficient, limited, or replaces"],
  "topics": ["short topic tags"]
}}

Rules:
- authors = this paper's authors from the text/header, not bibliography names.
- methods_proposed: what THEY propose (e.g. mvPE, EchoPT). Not generic "Transformer" unless they introduce it.
- methods_criticized: baselines they call weak or replace (e.g. RNN, vanilla Transformer). Empty list if none.
- topics: 3-8 short tags (NLP, attention, time-series, ...).
- Do not invent names or methods that are not in the text.
- No markdown, no extra keys.

Paper:
{paper}
"""


def llm_cache_key(prompt: str) -> str:
    payload = f"{CHAT_MODEL}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def grobid_author_names(record: dict) -> list[str]:
    names: list[str] = []
    for author in record.get("authors") or []:
        if isinstance(author, dict):
            name = (author.get("name") or "").strip()
        else:
            name = str(author).strip()
        if name:
            names.append(name)
    return names


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[...truncated]"


def paper_blob(arxiv_id: str, record: dict) -> str:
    """Kaynakça yok: listBibl entity'ye yazar/yöntem diye sızmasın."""
    sections = record.get("sections") if isinstance(record.get("sections"), dict) else {}
    header_authors = grobid_author_names(record)
    parts = [
        f"arxiv_id: {arxiv_id}",
        f"title: {record.get('title') or ''}",
        f"header_authors: {', '.join(header_authors) or '(none)'}",
        f"abstract:\n{clip(str(record.get('abstract') or ''), MAX_SECTION_CHARS)}",
    ]
    for key in ("introduction", "method", "results", "conclusion"):
        body = clip(str(sections.get(key) or ""), MAX_SECTION_CHARS)
        if body:
            parts.append(f"{key}:\n{body}")
    return "\n\n".join(parts)


def extract_json_object(raw: str) -> dict:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("JSON nesne yok")
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON nesne değil")
    return data


def as_str_list(value: object) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        token = str(item).strip()
        if token:
            items.append(token)
    return items


def normalize_record(arxiv_id: str, fallback_title: str, raw: dict) -> dict:
    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        title = fallback_title
    return {
        "arxiv_id": arxiv_id,
        "title": title.strip(),
        "authors": as_str_list(raw.get("authors")),
        "methods_proposed": as_str_list(raw.get("methods_proposed")),
        "methods_criticized": as_str_list(raw.get("methods_criticized")),
        "topics": as_str_list(raw.get("topics")),
    }


def chat_cached(client: OpenAI, cache: Cache, prompt: str) -> str:
    key = llm_cache_key(prompt)
    cached = cache.get(key)
    if cached is not None:
        return str(cached)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()
    cache.set(key, text)
    return text


def extract_one(
    path: Path,
    client: OpenAI,
    cache: Cache,
) -> tuple[str, dict | None]:
    """Döner: OK + kayıt, veya SKIP + None (hata print edildi)."""
    arxiv_id = path.name.removesuffix("_sections.json")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA (okuma): {exc}")
        return "SKIP", None

    if not isinstance(record, dict):
        print(f"SKIP  {arxiv_id}  HATA: JSON nesne değil")
        return "SKIP", None

    prompt = EXTRACT_PROMPT.format(paper=paper_blob(arxiv_id, record))
    cache_key = llm_cache_key(prompt)
    was_cached = cache.get(cache_key) is not None

    try:
        raw_text = chat_cached(client, cache, prompt)
        parsed = extract_json_object(raw_text)
        entity = normalize_record(arxiv_id, str(record.get("title") or ""), parsed)
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA (LLM/JSON): {exc}")
        return "SKIP", None

    mark = "cache" if was_cached else "api"
    print(
        f"OK    {arxiv_id}  [{mark}]  "
        f"authors={len(entity['authors'])}  "
        f"proposed={len(entity['methods_proposed'])}  "
        f"criticized={len(entity['methods_criticized'])}  "
        f"topics={len(entity['topics'])}"
    )
    return "OK", entity


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("HATA: OPENAI_API_KEY yok. cp .env.example .env yapıp anahtarı yaz.")
        print("0 makale işlendi")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (entity extract).")
        return

    section_files = sorted(TEXT_DIR.glob("*_sections.json"))
    if not section_files:
        print(f"HATA: {TEXT_DIR} içinde *_sections.json yok. Önce: python3 src/02b_parse_grobid.py")
        print("0 makale işlendi")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (entity extract).")
        return

    print(f"Makale: {len(section_files)}")
    print(f"model={CHAT_MODEL}  1 çağrı/makale  cache={LLM_CACHE_DIR}")
    print("kaynakça gönderilmiyor (yazar/yöntem sızıntısı olmasın)")

    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key)
    entities: list[dict] = []
    ok_count = 0
    skip_count = 0

    with Cache(str(LLM_CACHE_DIR)) as llm_cache:
        for path in tqdm(section_files, desc="Entity extract"):
            status, entity = extract_one(path, client, llm_cache)
            if status == "OK" and entity is not None:
                entities.append(entity)
                ok_count += 1
            else:
                skip_count += 1

    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            json.dumps(entities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Kaydedildi: {OUT_PATH}")
    except Exception as exc:
        print(f"HATA: JSON yazılamadı: {exc}")
        print("0 makale işlendi")
        return

    print(f"{ok_count} makale işlendi  SKIP={skip_count}")
    print(
        "Evaluation: recall@k=n/a  faithfulness=n/a  "
        "(ham entity; kanonik id yok, retrieval yok)."
    )


if __name__ == "__main__":
    main()
