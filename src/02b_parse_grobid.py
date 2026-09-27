"""
GROBID ile PDF'leri section-based parse eder (gövde vs kaynakça ayrı).

Nasıl çalıştırılır:
    # başka terminalde GROBID açık olmalı:
    # docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.1
    source .venv/bin/activate
    python3 src/02b_parse_grobid.py

Girdi:  data/raw_pdfs/*.pdf
Çıktı:  data/processed_text/{arxiv_id}_sections.json
Önkoşul: python3 src/01_ingest.py  ve  GROBID :8070

Neden requests: GROBID REST API'sine PDF POST etmek için (requirements.txt).
Neden xml.etree: TEI XML'i ek kütüphane olmadan okumak için.
recall@k / faithfulness: bu aşamada retrieval/LLM yok; n/a.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
TEXT_DIR = PROJECT_ROOT / "data" / "processed_text"

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
GROBID_ALIVE_URL = "http://localhost:8070/api/isalive"
# M1'de amd64 emülasyon ~3 dk/PDF sürebilir.
HTTP_TIMEOUT_SECONDS = 300
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Kullanıcının istediği dört kova; başlık anahtar kelimeyle eşlenir.
SECTION_KEYWORDS = {
    "introduction": ("introduction", "intro"),
    "method": (
        "method",
        "methods",
        "methodology",
        "approach",
        "architecture",
        "algorithm",
        "model",
        "proposed",
    ),
    "results": (
        "result",
        "results",
        "experiment",
        "experiments",
        "evaluation",
        "experimental",
    ),
    "conclusion": (
        "conclusion",
        "conclusions",
        "concluding",
        "discussion",
        "summary",
    ),
}

# I. II. A. 1. gibi numaraları düşür.
_HEAD_PREFIX = re.compile(
    r"^(?:[IVXLCDM]+|[A-Z]|\d+)(?:\.\d+)*\.?\s+",
    re.IGNORECASE,
)
_SUBSECTION_PREFIX = re.compile(r"^[A-Z]\.\s+")
# Dört kovaya girmesin; kaynakça zaten listBibl'de.
_SKIP_HEAD_KEYWORDS = (
    "related work",
    "related works",
    "background",
    "literature",
    "appendix",
    "acknowledg",
    "reference",
    "references",
    "bibliography",
)


def local_name(element: ET.Element) -> str:
    """{namespace}tag -> tag"""
    return element.tag.rsplit("}", 1)[-1]


def collapsed_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def person_name(pers_name: ET.Element | None) -> str:
    """forename + surname; itertext bitişik 'HailiangLi' üretir, çocukları ayrı al."""
    if pers_name is None:
        return ""
    parts: list[str] = []
    for child in pers_name:
        if local_name(child) in {"forename", "surname", "genName"}:
            token = (child.text or "").strip()
            if token:
                parts.append(token)
    return " ".join(parts)


def normalize_head(head: str) -> str:
    return _HEAD_PREFIX.sub("", head).strip().lower()


def classify_section(head: str) -> str | None:
    """Başlığı dört kovadan birine koy. Eşleşmezse None (loglanır)."""
    cleaned = normalize_head(head)
    if not cleaned:
        return None
    for bucket, keywords in SECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in cleaned:
                return bucket
    return None


def is_skipped_head(head: str) -> bool:
    cleaned = normalize_head(head)
    return any(keyword in cleaned for keyword in _SKIP_HEAD_KEYWORDS)


def div_body_text(div: ET.Element) -> str:
    """head hariç paragraf metni; kaynakça buraya girmez (body div)."""
    parts: list[str] = []
    for child in div:
        if local_name(child) == "head":
            continue
        text = collapsed_text(child)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def parse_authors(header: ET.Element) -> list[dict]:
    authors: list[dict] = []
    # sourceDesc: makale yazarları. listBibl'deki isimler burada yok.
    for author in header.findall(".//tei:sourceDesc//tei:author", TEI_NS):
        name = person_name(author.find("tei:persName", TEI_NS))
        if not name:
            continue
        affiliations = [
            collapsed_text(aff)
            for aff in author.findall("tei:affiliation", TEI_NS)
            if collapsed_text(aff)
        ]
        authors.append(
            {
                "name": name,
                "affiliation": "; ".join(affiliations),
            }
        )
    return authors


def parse_references(root: ET.Element) -> list[dict]:
    """listBibl = kaynakça; gövdeye karışmasın diye ayrı tutulur."""
    references: list[dict] = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", TEI_NS):
        xml_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id") or ""
        title_el = bibl.find(".//tei:title[@type='main']", TEI_NS)
        if title_el is None:
            title_el = bibl.find(".//tei:title", TEI_NS)
        authors = [
            person_name(author.find("tei:persName", TEI_NS))
            for author in bibl.findall(".//tei:author", TEI_NS)
        ]
        authors = [name for name in authors if name]
        date_el = bibl.find(".//tei:date", TEI_NS)
        year = ""
        if date_el is not None:
            when = (date_el.get("when") or "").strip()
            year = when[:4] if when else collapsed_text(date_el)
        references.append(
            {
                "id": xml_id,
                "title": collapsed_text(title_el),
                "authors": authors,
                "year": year,
                "raw": collapsed_text(bibl),
            }
        )
    return references


def parse_sections(root: ET.Element) -> tuple[dict[str, str], list[str]]:
    """
    body/div başlıklarını introduction/method/results/conclusion'a yığar.

    GROBID alt başlıkları kardeş div yapar; A./B. bir önceki kovaya yapışır.
    Eşleşmeyen majör başlık (Related Work) loglanır, şemaya eklenmez.
    """
    buckets = {key: [] for key in SECTION_KEYWORDS}
    unmapped: list[str] = []
    last_bucket: str | None = None
    seen_intro = False

    body = root.find(".//tei:text/tei:body", TEI_NS)
    if body is None:
        return {key: "" for key in SECTION_KEYWORDS}, unmapped

    for div in body.findall("tei:div", TEI_NS):
        head_el = div.find("tei:head", TEI_NS)
        head = collapsed_text(head_el) if head_el is not None else ""
        text = div_body_text(div)

        is_subsection = bool(_SUBSECTION_PREFIX.match(head))
        bucket = classify_section(head) if head else None
        if bucket is None and last_bucket and is_subsection:
            bucket = last_bucket
        if bucket is None and head and is_skipped_head(head):
            if text:
                unmapped.append(head)
            last_bucket = None
            continue
        # Intro'dan sonra, results öncesi eşleşmeyen majör bölüm ≈ method
        # (ör. "III. A Refined Positional Encoding").
        if (
            bucket is None
            and not is_subsection
            and seen_intro
            and last_bucket != "results"
        ):
            bucket = "method"

        if bucket is None:
            if text:
                unmapped.append(head or "(başlıksız)")
            if not is_subsection:
                last_bucket = None
            continue

        # Boş div de kovayı günceller; sonraki A./B. doğru yere yapışır.
        if text:
            buckets[bucket].append(text)
        last_bucket = bucket
        if bucket == "introduction":
            seen_intro = True

    return {key: "\n\n".join(parts) for key, parts in buckets.items()}, unmapped


def parse_tei(tei_xml: str) -> dict:
    root = ET.fromstring(tei_xml)
    header = root.find("tei:teiHeader", TEI_NS)
    if header is None:
        raise ValueError("TEI içinde teiHeader yok")

    title_el = header.find(".//tei:titleStmt/tei:title", TEI_NS)
    abstract_el = header.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    sections, unmapped = parse_sections(root)

    return {
        "title": collapsed_text(title_el),
        "authors": parse_authors(header),
        "abstract": collapsed_text(abstract_el),
        "sections": sections,
        "references": parse_references(root),
        "_unmapped_heads": unmapped,
    }


def grobid_is_alive() -> bool:
    try:
        response = requests.get(GROBID_ALIVE_URL, timeout=5)
        return response.ok and response.text.strip().lower() == "true"
    except requests.RequestException:
        return False


def post_pdf(pdf_path: Path) -> str:
    with pdf_path.open("rb") as handle:
        response = requests.post(
            GROBID_URL,
            files={"input": (pdf_path.name, handle, "application/pdf")},
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    if not response.text.strip():
        raise ValueError("GROBID boş TEI döndü")
    return response.text


def parse_one(pdf_path: Path) -> str:
    """
    Bir PDF işler.

    Döner: "OK" | "SKIP"
    SKIP = çıktı zaten var veya hata (mesaj print edilir).
    """
    arxiv_id = pdf_path.stem
    out_path = TEXT_DIR / f"{arxiv_id}_sections.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"SKIP  {arxiv_id}  zaten var")
        return "SKIP"

    try:
        tei_xml = post_pdf(pdf_path)
        record = parse_tei(tei_xml)
        unmapped = record.pop("_unmapped_heads")
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        filled = [key for key, value in record["sections"].items() if value]
        print(
            f"OK    {arxiv_id}  "
            f"yazar={len(record['authors'])}  "
            f"bölüm={'+'.join(filled) or 'yok'}  "
            f"ref={len(record['references'])}"
        )
        if unmapped:
            print(f"      eşleşmeyen başlık: {', '.join(unmapped)}")
        return "OK"
    except requests.Timeout as exc:
        print(f"SKIP  {arxiv_id}  HATA (GROBID timeout): {exc}")
        if out_path.exists():
            out_path.unlink()
        return "TIMEOUT"
    except requests.RequestException as exc:
        print(f"SKIP  {arxiv_id}  HATA (GROBID HTTP): {exc}")
        if out_path.exists():
            out_path.unlink()
        return "SKIP"
    except Exception as exc:
        print(f"SKIP  {arxiv_id}  HATA: {exc}")
        if out_path.exists():
            out_path.unlink()
        return "SKIP"


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    if not grobid_is_alive():
        print(
            "HATA: GROBID kapalı (http://localhost:8070/api/isalive). "
            "Önce çalıştır: docker run --rm --init --ulimit core=0 "
            "-p 8070:8070 lfoppiano/grobid:0.8.1"
        )
        print("0 makale parse edildi")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (retrieval/LLM yok)")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"HATA: {PDF_DIR} içinde PDF yok. Önce çalıştır: python src/01_ingest.py")
        print("0 makale parse edildi")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (retrieval/LLM yok)")
        return

    print(f"PDF sayısı: {len(pdf_files)}")
    print(f"GROBID: {GROBID_URL}")
    print(f"Çıktı: {TEXT_DIR}/{{arxiv_id}}_sections.json")

    ok_count = 0
    skip_count = 0
    consecutive_timeouts = 0
    for pdf_path in pdf_files:
        status = parse_one(pdf_path)
        if status == "OK":
            ok_count += 1
            consecutive_timeouts = 0
        elif status == "TIMEOUT":
            skip_count += 1
            consecutive_timeouts += 1
            # M1 emülasyonunda GROBID kilitlenince kalanı 5'er dk bekletme.
            if consecutive_timeouts >= 2:
                print(
                    "GROBID art arda 2 timeout verdi; kilitlenmiş olabilir. "
                    "Kalan PDF'ler atlandı — container'ı restart edip scripti tekrar çalıştır."
                )
                break
        else:
            skip_count += 1

    print(f"{ok_count} makale parse edildi")
    print(f"{skip_count} SKIP")
    print(
        "Evaluation: recall@k=n/a  faithfulness=n/a  "
        "(henüz retrieval/LLM yok; TEI yapı karşılaştırması elle)"
    )


if __name__ == "__main__":
    main()
