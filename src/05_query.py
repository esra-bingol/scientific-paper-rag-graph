"""
Naive Chroma indeksinde top-5 chunk bulur, GPT-4o-mini ile cevap üretir.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/05_query.py

Girdi:  chroma_db/ collection=papers_naive  +  klavyeden soru
Önkoşul: python3 src/04_embed.py

Neden openai: soru embedding'i ve gpt-4o-mini aynı resmi SDK.
Neden chromadb: cosine ile en yakın 5 chunk, harici servis yok.
Neden diskcache: aynı soru/prompt'u tekrar ödememek.
Neden python-dotenv: OPENAI_API_KEY koda gömülmez.
recall@5 gold etiket olmadan ölçülmez; faithfulness için ayrı judge yok.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

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
COLLECTION_NAME = "papers_naive"
TOP_K = 5

PROMPT_TEMPLATE = """Verilen makale parçalarına dayanarak cevapla. 
Cevap parçalarda yoksa "Bu bilgi verilen makalelerde yok" de.
Uydurma.

Parçalar:
{context}

Soru: {question}
"""


def embedding_cache_key(text: str) -> str:
    """Model adını anahtara katarız; 3-small vektörü 3-large sanılmasın."""
    payload = f"{EMBEDDING_MODEL}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def llm_cache_key(prompt: str) -> str:
    payload = f"{CHAT_MODEL}\n{prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def format_context(hits: list[dict]) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        parts.append(
            f"[{index}] arxiv_id={hit['arxiv_id']} chunk_id={hit['chunk_id']}\n{hit['text']}"
        )
    return "\n\n".join(parts)


def embed_question(client: OpenAI, cache: Cache, question: str) -> list[float]:
    key = embedding_cache_key(question)
    cached = cache.get(key)
    if cached is not None:
        print("Embedding cache: HIT")
        return cached
    print("Embedding cache: MISS (OpenAI)")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    vector = response.data[0].embedding
    cache.set(key, vector)
    return vector


def chat_answer(client: OpenAI, cache: Cache, prompt: str) -> str:
    key = llm_cache_key(prompt)
    cached = cache.get(key)
    if cached is not None:
        print("LLM cache: HIT")
        return str(cached)
    print("LLM cache: MISS (OpenAI)")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (response.choices[0].message.content or "").strip()
    cache.set(key, text)
    return text


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("HATA: OPENAI_API_KEY yok. cp .env.example .env yapıp anahtarı yaz.")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    if not CHROMA_DIR.exists():
        print(f"HATA: {CHROMA_DIR} yok. Önce: python3 src/04_embed.py")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

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

    openai_client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )
    except Exception as exc:
        print(f"HATA: collection {COLLECTION_NAME!r} yok ({exc}). Önce: python3 src/04_embed.py")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    if collection.count() == 0:
        print("HATA: collection boş. Önce: python3 src/04_embed.py")
        print("Evaluation: recall@5=n/a  faithfulness=n/a")
        return

    print(f"collection={COLLECTION_NAME!r}  n={collection.count()}  top_k={TOP_K}")
    print(f"embed={EMBEDDING_MODEL}  chat={CHAT_MODEL}")

    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with Cache(str(EMBED_CACHE_DIR)) as embed_cache, Cache(str(LLM_CACHE_DIR)) as llm_cache:
        try:
            retrieval_started = time.perf_counter()
            query_vector = embed_question(openai_client, embed_cache, question)
            raw = collection.query(
                query_embeddings=[query_vector],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"],
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
        except Exception as exc:
            print(f"HATA: retrieval başarısız: {exc}")
            print("Evaluation: recall@5=n/a  faithfulness=n/a")
            return

        documents = (raw.get("documents") or [[]])[0] or []
        metadatas = (raw.get("metadatas") or [[]])[0] or []
        distances = (raw.get("distances") or [[]])[0] or []
        ids = (raw.get("ids") or [[]])[0] or []

        hits: list[dict] = []
        for index, doc in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            chunk_id = str(meta.get("chunk_id") or (ids[index] if index < len(ids) else ""))
            arxiv_id = str(meta.get("arxiv_id") or "")
            distance = distances[index] if index < len(distances) else None
            hits.append(
                {
                    "text": doc or "",
                    "arxiv_id": arxiv_id,
                    "chunk_id": chunk_id,
                    "distance": distance,
                }
            )

        print(f"retrieval süresi: {retrieval_seconds:.3f} s  (top-{len(hits)})")
        for rank, hit in enumerate(hits, start=1):
            dist = hit["distance"]
            dist_text = f"{dist:.4f}" if isinstance(dist, (int, float)) else "?"
            print(
                f"  {rank}. arxiv_id={hit['arxiv_id']}  chunk_id={hit['chunk_id']}  "
                f"cosine_distance={dist_text}"
            )

        if not hits:
            print("HATA: hiç chunk dönmedi.")
            print("Evaluation: recall@5=n/a  faithfulness=n/a")
            return

        prompt = PROMPT_TEMPLATE.format(context=format_context(hits), question=question)
        try:
            llm_started = time.perf_counter()
            answer = chat_answer(openai_client, llm_cache, prompt)
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
            print(f"  arxiv_id={hit['arxiv_id']}  chunk_id={hit['chunk_id']}")

        print(
            "Evaluation: recall@5=n/a (gold etiket yok)  "
            "faithfulness=n/a (ayrı judge yok; kaynaklara elle bak)."
        )


if __name__ == "__main__":
    main()
