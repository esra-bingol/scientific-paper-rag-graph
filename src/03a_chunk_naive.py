from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks_naive.json"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
METHOD = "pypdf"


def arxiv_id_from_raw(path: Path) -> str:
    """1910.13634v1_raw.txt -> 1910.13634v1"""
    name = path.stem
    suffix = "_raw"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def chunk_one(path: Path, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    """Bir ham metni chunk listesine çevirir. Hata olursa boş liste."""
    arxiv_id = arxiv_id_from_raw(path)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA (okuma): {exc}")
        return []

    if not text.strip():
        print(f"SKIP  {arxiv_id}  boş dosya")
        return []

    try:
        pieces = splitter.split_text(text)
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA (split): {exc}")
        return []

    records: list[dict] = []
    for index, piece in enumerate(pieces):
        records.append(
            {
                "text": piece,
                "arxiv_id": arxiv_id,
                "chunk_id": f"{arxiv_id}_{index:04d}",
                "source_file": path.name,
                "method": METHOD,
            }
        )
    print(f"OK    {arxiv_id}  {len(records)} chunk")
    return records


def main() -> None:
    text_files = sorted(TEXT_DIR.glob("*_raw.txt"))
    if not text_files:
        print(f"HATA: {TEXT_DIR} içinde *_raw.txt yok. Önce: python src/02a_extract_pypdf.py")
        print("0 chunk üretildi, ortalama 0 karakter")
        print("Evaluation: chunking'te recall@k ve faithfulness yok (retrieval/LLM yok).")
        return

    print(f"Metin dosyası: {len(text_files)}")
    print(f"chunk_size={CHUNK_SIZE}  chunk_overlap={CHUNK_OVERLAP}  method={METHOD!r}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks: list[dict] = []
    for path in tqdm(text_files, desc="Chunking"):
        all_chunks.extend(chunk_one(path, splitter))

    try:
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHUNKS_PATH.write_text(
            json.dumps(all_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Kaydedildi: {CHUNKS_PATH}")
    except Exception as exc:
        print(f"HATA: JSON yazılamadı: {exc}")
        print("0 chunk üretildi, ortalama 0 karakter")
        return

    lengths = [len(chunk["text"]) for chunk in all_chunks]
    mean_chars = round(sum(lengths) / len(lengths)) if lengths else 0
    print(f"{len(all_chunks)} chunk üretildi, ortalama {mean_chars} karakter")
    print("Evaluation: recall@k=n/a  faithfulness=n/a  (henüz retrieval/LLM yok).")


if __name__ == "__main__":
    main()
