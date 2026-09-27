"""
Sorudan metadata filtresi çıkarır, Chroma where ile keser, sonra cevap üretir.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/04_embed.py --source grobid
    python3 src/06_metadata_filter.py
    python3 src/06_metadata_filter.py --question "2023'ten sonra yayımlanan makaleler?"

Girdi:  chroma_db/ collection=papers_grobid  +  soru
Önkoşul: python3 src/04_embed.py --source grobid
         (naive koleksiyonda year/authors/section yok)

Neden openai: filtre JSON'u ve cevap aynı resmi SDK.
Neden chromadb: year/section where ile cosine öncesi kesilir.
Neden diskcache: aynı soru/prompt tekrar ödenmesin.
Neden python-dotenv: OPENAI_API_KEY koda gömülmez.

authors alanı Chroma'da "Ad; Ad" string; $eq yetmez, yazar kesimi Python'da.
recall@5 / faithfulness: gold yok; n/a.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBED_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"
LLM_CACHE_DIR = PROJECT_ROOT / "cache" / "llm"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
# Yazar sonradan kesilince top-5 boş kalmasın diye fazla çek.
CANDIDATE_K = 40

ALLOWED_SECTIONS = ("abstract", "introduction", "method", "results")

COLLECTION_ALIASES = {
    "naive": "papers_naive",
    "papers_naive": "papers_naive",
    "grobid": "papers_grobid",
    "papers_grobid": "papers_grobid",
}

FILTER_PROMPT = """Kullanıcı sorusundan Chroma metadata filtresi çıkar.
Sadece JSON yaz, başka metin yok.

Şema:
{{
  "year_min": int veya null,
  "year_max": int veya null,
  "authors": [soyad veya ad] veya [],
  "section": "abstract" | "introduction" | "method" | "results" | null
}}

Kurallar:
- "2023'ten sonra" / "after 2023" → year_min=2024 (o yıl hariç).
- "2023 ve sonrası" / "since 2023" → year_min=2023.
- "2020'den önce" → year_max=2019.
- "sadece 2025" → year_min=2025 ve year_max=2025.
- Yazar/soyad geçiyorsa authors'a ekle (ör. "Li", "Zhao Song", "Esra").
- "method/yöntem bölümü" → section=method. Belirsizse null.
- Filtre yoksa hepsi null / [].

Soru: {question}
"""

ANSWER_PROMPT = """Verilen makale parçalarına ve filtre kataloğuna dayanarak cevapla.
Cevap bunlarda yoksa "Bu bilgi verilen makalelerde yok" de.
Uydurma.
Liste sorusunda katalogdaki arxiv_id'leri kullan; top-5 parçayla sınırlama.

Uygulanan filtre: {filt}
Filtreye uyan makaleler (katalog): {catalog}

Parçalar:
{context}

Soru: {question}
"""


def embedding_cache_key(text: str) -> str:
    payload = f"{EMBEDDING_MODEL}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def llm_cache_key(prompt: str) -> str:
    payload = f"{CHAT_MODEL}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_collection(name: str) -> str:
    key = name.strip()
    if key not in COLLECTION_ALIASES:
        allowed = ", ".join(sorted(COLLECTION_ALIASES))
        raise ValueError(f"bilinmeyen collection {name!r}; seçenekler: {allowed}")
    return COLLECTION_ALIASES[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metadata filtreli RAG sorgusu.")
    parser.add_argument(
        "--collection",
        default="papers_grobid",
        help="varsayılan papers_grobid (year/authors/section burada)",
    )
    parser.add_argument("--question", default="", help="Boşsa klavyeden sorulur.")
    return parser.parse_args()


_COST_MOD = None


def _cost_mod():
    global _COST_MOD
    if _COST_MOD is None:
        path = Path(__file__).resolve().parent / "12_cost_report.py"
        spec = importlib.util.spec_from_file_location("cost_report", path)
        if spec is None or spec.loader is None:
            raise ImportError("12_cost_report.py yüklenemedi")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _COST_MOD = module
    return _COST_MOD


def chat_cached(client: OpenAI, cache: Cache, prompt: str, stage: str = "metadata") -> str:
    key = llm_cache_key(prompt)
    cached = cache.get(key)
    print("LLM cache: HIT" if cached is not None else "LLM cache: MISS (OpenAI)")
    return _cost_mod().logged_chat(client, cache, prompt, CHAT_MODEL, key, stage)


def embed_question(client: OpenAI, cache: Cache, question: str) -> list[float]:
    key = embedding_cache_key(question)
    cached = cache.get(key)
    print("Embedding cache: HIT" if cached is not None else "Embedding cache: MISS (OpenAI)")
    return _cost_mod().logged_embed(
        client, cache, question, EMBEDDING_MODEL, key, "embed_query"
    )


def extract_json_object(raw: str) -> dict:
    """LLM bazen ```json ...``` sarmalar."""
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


def normalize_filter(raw: dict) -> dict[str, Any]:
    """Beklenen anahtarlar; çöp değerler None/[]."""
    year_min = raw.get("year_min")
    year_max = raw.get("year_max")
    if not isinstance(year_min, int):
        year_min = None
    if not isinstance(year_max, int):
        year_max = None

    authors_raw = raw.get("authors") or []
    authors: list[str] = []
    if isinstance(authors_raw, str) and authors_raw.strip():
        authors = [authors_raw.strip()]
    elif isinstance(authors_raw, list):
        authors = [str(item).strip() for item in authors_raw if str(item).strip()]

    section = raw.get("section")
    if isinstance(section, str):
        section = section.strip().lower() or None
        if section not in ALLOWED_SECTIONS:
            section = None
    else:
        section = None

    return {
        "year_min": year_min,
        "year_max": year_max,
        "authors": authors,
        "section": section,
    }


def build_where(filt: dict[str, Any]) -> dict | None:
    """Chroma where: year + section. authors string olduğu için burada yok."""
    clauses: list[dict] = []
    if filt["year_min"] is not None:
        clauses.append({"year": {"$gte": filt["year_min"]}})
    if filt["year_max"] is not None:
        clauses.append({"year": {"$lte": filt["year_max"]}})
    if filt["section"]:
        clauses.append({"section": {"$eq": filt["section"]}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def author_matches(authors_field: str, needles: list[str]) -> bool:
    haystack = authors_field.casefold()
    return any(needle.casefold() in haystack for needle in needles)


def format_context(hits: list[dict]) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        extra = ""
        if hit.get("section"):
            extra += f" section={hit['section']}"
        if hit.get("year") not in (None, ""):
            extra += f" year={hit['year']}"
        if hit.get("authors"):
            extra += f" authors={hit['authors']}"
        parts.append(
            f"[{index}] arxiv_id={hit['arxiv_id']} chunk_id={hit['chunk_id']}{extra}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(parts)


def rows_to_hits(
    documents: list,
    metadatas: list,
    distances: list,
    ids: list,
) -> list[dict]:
    hits: list[dict] = []
    for index, doc in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        chunk_id = str(meta.get("chunk_id") or (ids[index] if index < len(ids) else ""))
        hits.append(
            {
                "text": doc or "",
                "arxiv_id": str(meta.get("arxiv_id") or ""),
                "chunk_id": chunk_id,
                "distance": distances[index] if index < len(distances) else None,
                "section": str(meta.get("section") or ""),
                "year": meta.get("year"),
                "authors": str(meta.get("authors") or ""),
            }
        )
    return hits


def catalog_papers(
    collection: chromadb.Collection,
    where: dict | None,
    author_needles: list[str],
) -> list[dict]:
    """where (+ yazar) uyan benzersiz makaleler; top-5 cosine tüm id'leri getirmez."""
    kwargs: dict[str, Any] = {"include": ["metadatas"]}
    if where is not None:
        kwargs["where"] = where
    try:
        raw = collection.get(**kwargs)
    except Exception as exc:
        print(f"WARN  catalog get başarısız: {exc}")
        return []

    seen: dict[str, dict] = {}
    metadatas = raw.get("metadatas") or []
    for meta in metadatas:
        if not meta:
            continue
        authors = str(meta.get("authors") or "")
        if author_needles and not author_matches(authors, author_needles):
            continue
        arxiv_id = str(meta.get("arxiv_id") or "").strip()
        if not arxiv_id or arxiv_id in seen:
            continue
        seen[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "year": meta.get("year"),
            "authors": authors,
        }
    rows = list(seen.values())
    rows.sort(key=lambda row: (row.get("year") is None, row.get("year") or 0, row["arxiv_id"]))
    return rows


def print_hits(hits: list[dict]) -> None:
    for rank, hit in enumerate(hits, start=1):
        dist = hit["distance"]
        dist_text = f"{dist:.4f}" if isinstance(dist, (int, float)) else "?"
        print(
            f"  {rank}. arxiv_id={hit['arxiv_id']}  chunk_id={hit['chunk_id']}  "
            f"section={hit.get('section') or '-'}  year={hit.get('year')!r}  "
            f"authors={hit.get('authors') or '-'}  cosine_distance={dist_text}"
        )


def main() -> None:
    args = parse_args()
    try:
        collection_name = resolve_collection(args.collection)
    except ValueError as exc:
        print(f"HATA: {exc}")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("HATA: OPENAI_API_KEY yok. cp .env.example .env yapıp anahtarı yaz.")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    if not CHROMA_DIR.exists():
        print(f"HATA: {CHROMA_DIR} yok. Önce: python3 src/04_embed.py --source grobid")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    if args.question.strip():
        question = args.question.strip()
    else:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHATA: soru alınamadı.")
            print("Evaluation: recall@5=n/a  faithfulness=n/a")
            return

    if not question:
        print("HATA: boş soru.")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return
    print(f"Alınan: {question}")

    if collection_name != "papers_grobid":
        print(
            "WARN  year/authors/section papers_grobid'de dolu; "
            f"{collection_name!r} filtresi boş dönebilir."
        )

    openai_client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception as exc:
        print(
            f"HATA: collection {collection_name!r} yok ({exc}). "
            "Önce: python3 src/04_embed.py --source grobid"
        )
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    if collection.count() == 0:
        print("HATA: collection boş.")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    print(f"collection={collection_name!r}  n={collection.count()}  top_k={TOP_K}")
    print(f"embed={EMBEDDING_MODEL}  chat={CHAT_MODEL}")

    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with Cache(str(EMBED_CACHE_DIR)) as embed_cache, Cache(str(LLM_CACHE_DIR)) as llm_cache:
        try:
            filter_raw = chat_cached(
                openai_client, llm_cache, FILTER_PROMPT.format(question=question)
            )
            filt = normalize_filter(extract_json_object(filter_raw))
        except Exception as exc:
            print(f"HATA: filtre çıkarılamadı: {exc}")
            print("Evaluation: recall@5=n/a  faithfulness=n/a")
            return

        where = build_where(filt)
        print(f"çıkarılan filtre: {json.dumps(filt, ensure_ascii=False)}")
        print(f"chroma where: {json.dumps(where, ensure_ascii=False)}")
        if filt["authors"]:
            print(
                "authors Chroma string olduğu için where'de yok; "
                f"Python post-filter={filt['authors']}"
            )

        try:
            retrieval_started = time.perf_counter()
            query_vector = embed_question(openai_client, embed_cache, question)
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": min(CANDIDATE_K, collection.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where
            raw = collection.query(**query_kwargs)
            retrieval_seconds = time.perf_counter() - retrieval_started
        except Exception as exc:
            print(f"HATA: retrieval başarısız: {exc}")
            print("Evaluation: recall@5=n/a  faithfulness=n/a")
            return

        documents = (raw.get("documents") or [[]])[0] or []
        metadatas = (raw.get("metadatas") or [[]])[0] or []
        distances = (raw.get("distances") or [[]])[0] or []
        ids = (raw.get("ids") or [[]])[0] or []
        hits = rows_to_hits(documents, metadatas, distances, ids)
        before_author = len(hits)
        if filt["authors"]:
            hits = [hit for hit in hits if author_matches(hit["authors"], filt["authors"])]
        hits = hits[:TOP_K]
        catalog = catalog_papers(collection, where, filt["authors"])

        print(
            f"retrieval: {len(hits)} chunk  "
            f"(where sonrası aday={before_author}, {retrieval_seconds:.3f} s)"
        )
        print_hits(hits)
        print(f"filtre katalog: {len(catalog)} makale")
        for row in catalog:
            print(
                f"  - {row['arxiv_id']}  year={row.get('year')!r}  "
                f"authors={row.get('authors') or '-'}"
            )

        if not hits and not catalog:
            print("\n--- Cevap ---")
            print("Bu bilgi verilen makalelerde yok")
            print("\n--- Kaynaklar ---")
            print("  (filtreye uyan chunk/makale yok)")
            print("Evaluation: recall@5=n/a  faithfulness=n/a  (boş retrieval).")
            return

        catalog_text = (
            ", ".join(
                f"{row['arxiv_id']} (year={row.get('year')})"
                for row in catalog
            )
            or "(boş)"
        )
        prompt = ANSWER_PROMPT.format(
            filt=json.dumps(filt, ensure_ascii=False),
            catalog=catalog_text,
            context=format_context(hits) or "(filtreye uyan parça yok)",
            question=question,
        )
        try:
            llm_started = time.perf_counter()
            answer = chat_cached(openai_client, llm_cache, prompt, stage="query")
            llm_seconds = time.perf_counter() - llm_started
        except Exception as exc:
            print(f"HATA: LLM çağrısı başarısız: {exc}")
            print("Evaluation: recall@5=n/a  faithfulness=n/a")
            return

        print(f"LLM süresi: {llm_seconds:.3f} s")
        print("\n--- Cevap ---")
        print(answer)
        print("\n--- Kaynaklar ---")
        for hit in hits:
            print(
                f"  arxiv_id={hit['arxiv_id']}  chunk_id={hit['chunk_id']}  "
                f"year={hit.get('year')!r}  section={hit.get('section') or '-'}"
            )
        print(
            "Evaluation: recall@5=n/a (gold etiket yok)  "
            "faithfulness=n/a (ayrı judge yok; kaynaklara elle bak)."
        )


if __name__ == "__main__":
    main()
