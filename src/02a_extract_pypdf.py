from __future__ import annotations

from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"


def pdf_to_text(pdf_path: Path) -> str:
    """Tüm sayfaların metnini birleştirir. Layout analizi yok."""
    parts: list[str] = []
    with pymupdf.open(pdf_path) as document:
        for page in document:
            parts.append(page.get_text())
    return "\n".join(parts)


def extract_one(pdf_path: Path) -> tuple[str, int]:
    """
    Bir PDF işler.

    Döner: ("OK" | "SKIP", karakter sayısı)
    SKIP = çıktı zaten var veya hata (hata mesajı print edilir).
    """
    arxiv_id = pdf_path.stem
    out_path = TEXT_DIR / f"{arxiv_id}_raw.txt"

    if out_path.exists() and out_path.stat().st_size > 0:
        char_count = len(out_path.read_text(encoding="utf-8"))
        print(f"SKIP  {arxiv_id}  {char_count} karakter (zaten var)")
        return "SKIP", char_count

    try:
        text = pdf_to_text(pdf_path)
        out_path.write_text(text, encoding="utf-8")
        char_count = len(text)
        print(f"OK    {arxiv_id}  {char_count} karakter")
        return "OK", char_count
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  0 karakter  HATA: {exc}")
        if out_path.exists():
            out_path.unlink()
        return "SKIP", 0


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"HATA: {PDF_DIR} içinde PDF yok. Önce çalıştır: python src/01_ingest.py")
        print("0 dosya işlendi")
        print("Evaluation: extract'te recall@k ve faithfulness yok (retrieval/LLM yok).")
        return

    print(f"PDF sayısı: {len(pdf_files)}")
    print(f"Girdi: {PDF_DIR}")
    print(f"Çıktı: {TEXT_DIR}")

    ok_count = 0
    skip_count = 0
    empty_count = 0

    for pdf_path in pdf_files:
        status, char_count = extract_one(pdf_path)
        if status == "OK":
            ok_count += 1
        else:
            skip_count += 1
        if char_count == 0:
            empty_count += 1

    print(f"{ok_count} OK, {skip_count} SKIP")
    print(
        "Evaluation: recall@k=n/a  faithfulness=n/a  "
        f"(henüz retrieval/LLM yok; boş çıktı={empty_count}/{len(pdf_files)})"
    )


if __name__ == "__main__":
    main()
