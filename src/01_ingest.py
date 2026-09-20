from __future__ import annotations

import json
import time
from pathlib import Path

import arxiv
import requests
from tqdm import tqdm

# Proje kökü: src/01_ingest.py -> bir üst klasör
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"

# Düz "transformer architecture" CPU/yazılım mimarisi kağıtlarını da çekiyor.
# cs.LG / cs.CL / cs.AI ile nöral Transformer makalelerine kısıtlıyoruz.
QUERY = '"transformer architecture" AND (cat:cs.LG OR cat:cs.CL OR cat:cs.AI)'
MAX_PAPERS = 20
DOWNLOAD_DELAY_SECONDS = 3.0
HTTP_TIMEOUT_SECONDS = 60
USER_AGENT = "scientific-paper-rag-graph/0.1 (learning project)"


def to_arxiv_id(result: arxiv.Result) -> str:
    """Dosya adı için güvenli arXiv id üretir (eski id'lerdeki / karakterini temizler)."""
    return result.get_short_id().replace("/", "_")


def result_to_metadata(result: arxiv.Result, arxiv_id: str) -> dict:
    return {
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "year": result.published.year if result.published else None,
        "arxiv_id": arxiv_id,
        "abstract": result.summary,
    }


def download_pdf(pdf_url: str, pdf_path: Path) -> None:
    """PDF'i HTTP ile indirir. Bozuk/HTML cevap gelirse hata fırlatır."""
    headers = {"User-Agent": USER_AGENT}
    with requests.get(
        pdf_url,
        stream=True,
        timeout=HTTP_TIMEOUT_SECONDS,
        headers=headers,
    ) as response:
        response.raise_for_status()
        chunks = response.iter_content(chunk_size=8192)
        first_chunk = next(chunks, b"")
        if not first_chunk.startswith(b"%PDF"):
            raise ValueError(
                f"Beklenen PDF değil (content-type={response.headers.get('Content-Type')})"
            )
        with pdf_path.open("wb") as file:
            file.write(first_chunk)
            for chunk in chunks:
                if chunk:
                    file.write(chunk)


def download_one(result: arxiv.Result) -> dict | None:
    """Tek makaleyi indirir. Hata olursa None döner, çağıran devam eder."""
    arxiv_id = to_arxiv_id(result)
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

    print(f"\n→ {arxiv_id}: {result.title}")

    try:
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"  PDF zaten var, indirme atlandı: {pdf_path.name}")
            return result_to_metadata(result, arxiv_id)

        if not result.pdf_url:
            raise ValueError("pdf_url yok")

        download_pdf(result.pdf_url, pdf_path)
        print(f"  PDF kaydedildi: {pdf_path}")
        time.sleep(DOWNLOAD_DELAY_SECONDS)
        return result_to_metadata(result, arxiv_id)
    except Exception as exc:
        print(f"  HATA, bu makale atlandı ({arxiv_id}): {exc}")
        if pdf_path.exists():
            pdf_path.unlink()
        return None


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Sorgu: {QUERY!r}")
    print(f"Hedef: {MAX_PAPERS} makale")
    print(f"PDF klasörü: {PDF_DIR}")

    client = arxiv.Client(
        page_size=MAX_PAPERS,
        delay_seconds=3.0,  # arXiv rate limit'e takılmamak için
        num_retries=3,
    )
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    try:
        results = list(client.results(search))
    except Exception as exc:
        print(f"HATA: arXiv araması başarısız: {exc}")
        print("0 makale indirildi")
        print("Evaluation: ingest'te recall@k ve faithfulness yok (retrieval/LLM yok).")
        return

    print(f"arXiv {len(results)} sonuç döndü.")

    records: list[dict] = []
    for result in tqdm(results, total=len(results), desc="PDF indiriliyor"):
        record = download_one(result)
        if record is not None:
            records.append(record)

    try:
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Metadata yazıldı: {METADATA_PATH}")
    except Exception as exc:
        print(f"HATA: metadata.json yazılamadı: {exc}")

    print(f"{len(records)} makale indirildi")
    print("Evaluation: ingest'te recall@k ve faithfulness yok (retrieval/LLM yok).")


if __name__ == "__main__":
    main()
