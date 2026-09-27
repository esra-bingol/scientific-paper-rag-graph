"""
GROBID section JSON'larından anlamlı chunk üretir. Kaynakça RAG'a girmez.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/03b_chunk_section.py

Girdi:  data/processed_text/{arxiv_id}_sections.json
Çıktı:  data/chunks/chunks_grobid.json
Önkoşul: python3 src/02b_parse_grobid.py

Neden langchain_text_splitters: 03a ile aynı 512/50 kesici; naive vs section karşılaştırması adil olsun.
recall@k / faithfulness: bu aşamada retrieval/LLM yok; n/a.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks_grobid.json"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
METHOD = "grobid"

# Hedef kova sayıları. references kasıtlı yok — listBibl indekse girmez.
SECTION_PLAN = {
    "introduction": (2, 3),
    "method": (3, 5),
    "results": (2, 4),
}


def load_year_index(path: Path) -> dict[str, int | None]:
    """arxiv_id -> year. Yıl metadata'da; GROBID JSON'da kağıt yılı yok."""
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN  metadata okunamadı ({exc}); year=None yazılacak")
        return {}
    index: dict[str, int | None] = {}
    if not isinstance(rows, list):
        return index
    for row in rows:
        arxiv_id = str(row.get("arxiv_id") or "")
        if arxiv_id:
            index[arxiv_id] = row.get("year")
    return index


def author_names(record: dict) -> list[str]:
    names: list[str] = []
    for author in record.get("authors") or []:
        if isinstance(author, dict):
            name = (author.get("name") or "").strip()
        else:
            name = str(author).strip()
        if name:
            names.append(name)
    return names


def merge_to_n(pieces: list[str], target: int) -> list[str]:
    """Fazla parçayı komşu gruplara yığ; cümle sınırını splitter zaten seçti."""
    if target <= 0 or not pieces:
        return []
    if len(pieces) <= target:
        return pieces
    merged: list[str] = []
    total = len(pieces)
    for index in range(target):
        start = index * total // target
        end = (index + 1) * total // target
        merged.append("\n\n".join(pieces[start:end]))
    return merged


def split_in_range(text: str, min_n: int, max_n: int) -> list[str]:
    """512/50 ile böl, sonra [min_n, max_n] aralığına sıkıştır. Kısa metni şişirme."""
    text = text.strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    pieces = [part.strip() for part in splitter.split_text(text) if part.strip()]
    if not pieces:
        return [text]

    if len(pieces) > max_n:
        return merge_to_n(pieces, max_n)

    if len(pieces) < min_n and len(text) > CHUNK_SIZE:
        finer_size = max(128, len(text) // min_n)
        finer = RecursiveCharacterTextSplitter(
            chunk_size=finer_size,
            chunk_overlap=CHUNK_OVERLAP,
        )
        pieces = [part.strip() for part in finer.split_text(text) if part.strip()]
        if len(pieces) > max_n:
            return merge_to_n(pieces, max_n)
    return pieces


def make_chunk(
    text: str,
    arxiv_id: str,
    section: str,
    index: int,
    authors: list[str],
    year: int | None,
) -> dict:
    return {
        "text": text,
        "arxiv_id": arxiv_id,
        "section": section,
        "chunk_id": f"{arxiv_id}_{section}_{index:02d}",
        "authors": authors,
        "year": year,
        "method": METHOD,
    }


def chunk_record(record: dict, arxiv_id: str, year: int | None) -> tuple[list[dict], Counter]:
    """
    Bir makalenin section JSON'ını chunk listesine çevirir.

    abstract = tek chunk (özel tag). references atlanır.
    conclusion şemada yok; varsa loglanır, indekse girmez.
    """
    authors = author_names(record)
    counts: Counter = Counter()
    chunks: list[dict] = []

    abstract = (record.get("abstract") or "").strip()
    if abstract:
        chunks.append(make_chunk(abstract, arxiv_id, "abstract", 0, authors, year))
        counts["abstract"] += 1

    sections = record.get("sections") or {}
    if not isinstance(sections, dict):
        sections = {}

    conclusion = (sections.get("conclusion") or "").strip()
    if conclusion:
        print(f"      conclusion atlandı ({len(conclusion)} karakter; planda yok)")

    for section, (min_n, max_n) in SECTION_PLAN.items():
        text = (sections.get(section) or "").strip()
        if not text:
            continue
        pieces = split_in_range(text, min_n, max_n)
        for index, piece in enumerate(pieces):
            chunks.append(make_chunk(piece, arxiv_id, section, index, authors, year))
        counts[section] += len(pieces)

    return chunks, counts


def format_counts(counts: Counter) -> str:
    parts = []
    for key in ("abstract", "introduction", "method", "results"):
        parts.append(f"{key}={counts.get(key, 0)}")
    return "  ".join(parts)


def chunk_one(path: Path, year_index: dict[str, int | None]) -> tuple[list[dict], Counter]:
    arxiv_id = path.name.removesuffix("_sections.json")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA (okuma): {exc}")
        return [], Counter()

    if not isinstance(record, dict):
        print(f"SKIP  {arxiv_id}  HATA: JSON nesne değil")
        return [], Counter()

    try:
        chunks, counts = chunk_record(record, arxiv_id, year_index.get(arxiv_id))
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA (split): {exc}")
        return [], Counter()

    if not chunks:
        print(f"SKIP  {arxiv_id}  boş gövde/abstract")
        return [], Counter()

    print(f"OK    {arxiv_id}  {format_counts(counts)}")
    return chunks, counts


def main() -> None:
    section_files = sorted(TEXT_DIR.glob("*_sections.json"))
    if not section_files:
        print(f"HATA: {TEXT_DIR} içinde *_sections.json yok. Önce: python3 src/02b_parse_grobid.py")
        print("0 chunk üretildi")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (retrieval/LLM yok).")
        return

    year_index = load_year_index(METADATA_PATH)
    print(f"Section dosyası: {len(section_files)}")
    print(
        f"chunk_size={CHUNK_SIZE}  chunk_overlap={CHUNK_OVERLAP}  "
        f"method={METHOD!r}  references=ATLANDI"
    )

    all_chunks: list[dict] = []
    totals: Counter = Counter()
    for path in tqdm(section_files, desc="Section chunking"):
        chunks, counts = chunk_one(path, year_index)
        all_chunks.extend(chunks)
        totals.update(counts)

    try:
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHUNKS_PATH.write_text(
            json.dumps(all_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Kaydedildi: {CHUNKS_PATH}")
    except Exception as exc:
        print(f"HATA: JSON yazılamadı: {exc}")
        print("0 chunk üretildi")
        return

    print("section başına chunk sayısı:")
    for key in ("abstract", "introduction", "method", "results"):
        print(f"  {key}: {totals.get(key, 0)}")
    extra = set(totals) - {"abstract", "introduction", "method", "results"}
    if extra:
        print(f"  (beklenmeyen: {sorted(extra)})")

    lengths = [len(chunk["text"]) for chunk in all_chunks]
    mean_chars = round(sum(lengths) / len(lengths)) if lengths else 0
    print(f"{len(all_chunks)} chunk üretildi, ortalama {mean_chars} karakter")
    print("Evaluation: recall@k=n/a  faithfulness=n/a  (henüz retrieval/LLM yok).")


if __name__ == "__main__":
    main()
