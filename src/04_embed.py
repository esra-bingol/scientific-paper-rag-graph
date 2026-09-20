from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import chromadb
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks_naive.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "papers_naive"
BATCH_SIZE = 100


def cache_key(text: str) -> str:
    """Model adını anahtara katarız; 3-small vektörü 3-large sanılmasın."""
    payload = f"{EMBEDDING_MODEL}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_chunks(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("chunks_naive.json bir liste olmalı")
    return raw


def batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """OpenAI bir batch'te birden fazla metin alır; sıra index ile korunur."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("HATA: OPENAI_API_KEY yok. cp .env.example .env yapıp anahtarı yaz.")
        print("0 chunk embed edildi (0 cached)")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (henüz retrieval yok).")
        return

    if not CHUNKS_PATH.exists():
        print(f"HATA: {CHUNKS_PATH} yok. Önce: python3 src/03a_chunk_naive.py")
        print("0 chunk embed edildi (0 cached)")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (henüz retrieval yok).")
        return

    try:
        chunks = load_chunks(CHUNKS_PATH)
    except Exception as exc:
        print(f"HATA: chunk JSON okunamadı: {exc}")
        print("0 chunk embed edildi (0 cached)")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (henüz retrieval yok).")
        return

    print(f"Chunk sayısı: {len(chunks)}")
    print(f"model={EMBEDDING_MODEL}  batch={BATCH_SIZE}  collection={COLLECTION_NAME!r}")
    print(f"Chroma: {CHROMA_DIR}")
    print(f"Cache:  {CACHE_DIR}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    openai_client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # embedding_function=None: vektörü biz veriyoruz; Chroma MiniLM indirmesin.
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )

    written = 0
    cached_hits = 0
    skipped = 0

    with Cache(str(CACHE_DIR)) as embedding_cache:
        batches = batched(chunks, BATCH_SIZE)
        for batch in tqdm(batches, desc="Embedding"):
            ready: list[tuple[dict, list[float]]] = []
            missing_items: list[dict] = []

            for item in batch:
                text = str(item.get("text") or "")
                chunk_id = str(item.get("chunk_id") or "").strip()
                if not text.strip() or not chunk_id:
                    skipped += 1
                    print(f"SKIP  boş text/chunk_id  {chunk_id or '(id yok)'}")
                    continue

                vector = embedding_cache.get(cache_key(text))
                if vector is not None:
                    cached_hits += 1
                    ready.append((item, vector))
                else:
                    missing_items.append(item)

            if missing_items:
                texts = [str(item["text"]) for item in missing_items]
                try:
                    vectors = embed_texts(openai_client, texts)
                except Exception as exc:
                    print(f"HATA: OpenAI embedding başarısız ({len(texts)} metin): {exc}")
                    continue
                if len(vectors) != len(missing_items):
                    print(
                        f"HATA: beklenen {len(missing_items)} vektör, gelen {len(vectors)}"
                    )
                    continue
                for item, vector in zip(missing_items, vectors, strict=True):
                    embedding_cache.set(cache_key(str(item["text"])), vector)
                    ready.append((item, vector))

            if not ready:
                continue

            ids = [str(item["chunk_id"]) for item, _ in ready]
            embeddings = [vector for _, vector in ready]
            documents = [str(item["text"]) for item, _ in ready]
            metadatas = [
                {
                    "arxiv_id": str(item.get("arxiv_id") or ""),
                    "chunk_id": str(item["chunk_id"]),
                    "source_file": str(item.get("source_file") or ""),
                    "method": str(item.get("method") or "pypdf"),
                }
                for item, _ in ready
            ]
            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                written += len(ids)
            except Exception as exc:
                print(f"HATA: Chroma yazılamadı ({len(ids)} kayıt): {exc}")

    print(f"{written} chunk embed edildi ({cached_hits} cached)")
    print(f"SKIP={skipped}  collection.count={collection.count()}")
    print("Evaluation: recall@k=n/a  faithfulness=n/a  (henüz retrieval/sorgu yok).")


if __name__ == "__main__":
    main()
