"""
entities_raw.json yazar adlarını OpenAlex'te arar; kanonik author_id adaylarını üretir.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/07b_enrich_openalex.py

Girdi:  data/entities_raw.json
Çıktı:  data/authors_candidates.json
Önkoşul: python3 src/07a_extract_entities.py

Neden requests: OpenAlex REST; resmi Python SDK yok.
Neden diskcache: aynı ad / aynı A-id tekrar 1 sn beklenmesin (cache/openalex).
Neden rapidfuzz: skor 0-1 ad benzerliği; OpenAlex relevance_score binli ölçek.
Neden python-dotenv: isteğe bağlı OPENALEX_MAILTO (nezaket havuzu).
Neden tqdm: yazar başına hangi isimde olduğumuzu görmek için.

Kanonik kimlik = OpenAlex author_id (A...). Bu script seçmez; aday listesi üretir.
recall@k / faithfulness: retrieval yok; n/a.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from diskcache import Cache
from dotenv import load_dotenv
from rapidfuzz import fuzz
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "data" / "entities_raw.json"
OUT_PATH = PROJECT_ROOT / "data" / "authors_candidates.json"
CACHE_DIR = PROJECT_ROOT / "cache" / "openalex"

OPENALEX_BASE = "https://api.openalex.org"
AUTHORS_SEARCH_PATH = "/authors"
WORKS_PATH = "/works"
PER_PAGE = 5
SLEEP_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3

# OpenAlex 429 verirse biraz daha bekle.
RETRY_SLEEP_SECONDS = 3.0


def cache_key(kind: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def short_author_id(openalex_id: str) -> str:
    """https://openalex.org/A5103024730 -> A5103024730 (kanonik)."""
    token = (openalex_id or "").rstrip("/").split("/")[-1].strip()
    if token.startswith("A") and token[1:].isdigit():
        return token
    return token


def unique_author_names(records: list[dict]) -> list[str]:
    """İlk görülen yazımı koru; büyük/küçük harf tekrarını at."""
    seen: set[str] = set()
    names: list[str] = []
    for record in records:
        for raw in record.get("authors") or []:
            name = str(raw).strip()
            key = " ".join(name.split()).casefold()
            if not name or key in seen:
                continue
            seen.add(key)
            names.append(name)
    return names


def institution_name(author: dict) -> str:
    """last_known_institutions sık null; en yeni affiliation yedek."""
    last_known = author.get("last_known_institutions")
    if isinstance(last_known, list):
        for inst in last_known:
            if not isinstance(inst, dict):
                continue
            name = str(inst.get("display_name") or "").strip()
            if name:
                return name

    legacy = author.get("last_known_institution")
    if isinstance(legacy, dict):
        name = str(legacy.get("display_name") or "").strip()
        if name:
            return name

    best_name = ""
    best_year = -1
    for aff in author.get("affiliations") or []:
        if not isinstance(aff, dict):
            continue
        inst = aff.get("institution") if isinstance(aff.get("institution"), dict) else {}
        name = str(inst.get("display_name") or "").strip()
        years = aff.get("years") or []
        year = max(years) if isinstance(years, list) and years else -1
        if name and year >= best_year:
            best_year = year
            best_name = name
    return best_name


def name_score(query_name: str, display_name: str) -> float:
    """0-1; token sırası farkını yut (Vaswani, Ashish)."""
    if not query_name.strip() or not display_name.strip():
        return 0.0
    return round(fuzz.token_sort_ratio(query_name, display_name) / 100.0, 4)


def build_session() -> requests.Session:
    email = os.getenv("OPENALEX_MAILTO", "").strip()
    user_agent = "scientific-paper-rag-graph/07b"
    if email:
        user_agent = f"{user_agent} (mailto:{email})"
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


def polite_params(params: dict[str, Any]) -> dict[str, Any]:
    email = os.getenv("OPENALEX_MAILTO", "").strip()
    if email:
        return {**params, "mailto": email}
    return params


def openalex_get(
    session: requests.Session,
    cache: Cache,
    path: str,
    params: dict[str, Any],
) -> tuple[dict, str]:
    """JSON döner. mark = cache | api | SKIP benzeri hata fırlatır."""
    query = polite_params(params)
    key = cache_key("get", f"{path}?{urlencode(sorted(query.items()))}")
    cached = cache.get(key)
    if cached is not None:
        if isinstance(cached, dict):
            return cached, "cache"
        raise ValueError("cache bozuk")

    url = f"{OPENALEX_BASE}{path}"
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=query, timeout=HTTP_TIMEOUT_SECONDS)
            if response.status_code == 429:
                print(f"  429 rate limit  deneme={attempt}/{MAX_RETRIES}  sleep={RETRY_SLEEP_SECONDS}s")
                time.sleep(RETRY_SLEEP_SECONDS)
                last_error = RuntimeError("OpenAlex 429")
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("OpenAlex JSON nesne değil")
            cache.set(key, payload)
            time.sleep(SLEEP_SECONDS)
            return payload, "api"
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"  HATA  {path}  deneme={attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(SLEEP_SECONDS)
    raise RuntimeError(str(last_error) if last_error else "OpenAlex istek başarısız")


def fetch_top_works(
    session: requests.Session,
    cache: Cache,
    author_id: str,
) -> list[str]:
    """Atıf sırasıyla en fazla 5 başlık. Hata olursa boş liste (aday düşmez)."""
    if not author_id:
        return []
    try:
        payload, _mark = openalex_get(
            session,
            cache,
            WORKS_PATH,
            {
                "filter": f"author.id:{author_id}",
                "sort": "cited_by_count:desc",
                "per_page": PER_PAGE,
                "select": "id,display_name,publication_year,cited_by_count",
            },
        )
    except Exception as exc:
        print(f"  SKIP works  {author_id}: {exc}")
        return []

    titles: list[str] = []
    for work in payload.get("results") or []:
        if not isinstance(work, dict):
            continue
        title = str(work.get("display_name") or "").strip()
        if title:
            titles.append(title)
    return titles


def search_author_candidates(
    session: requests.Session,
    cache: Cache,
    query_name: str,
) -> tuple[list[dict], str]:
    payload, mark = openalex_get(
        session,
        cache,
        AUTHORS_SEARCH_PATH,
        {"search": query_name, "per_page": PER_PAGE},
    )
    candidates: list[dict] = []
    for raw in payload.get("results") or []:
        if not isinstance(raw, dict):
            continue
        author_id = short_author_id(str(raw.get("id") or ""))
        if not author_id:
            continue
        display_name = str(raw.get("display_name") or "").strip()
        works_count = raw.get("works_count")
        try:
            works_int = int(works_count)
        except (TypeError, ValueError):
            works_int = 0
        candidates.append(
            {
                "author_id": author_id,
                "display_name": display_name,
                "institution": institution_name(raw),
                "works": works_int,
                "score": name_score(query_name, display_name),
                "top_5_works": fetch_top_works(session, cache, author_id),
            }
        )
    candidates.sort(key=lambda row: (row["score"], row["works"]), reverse=True)
    return candidates, mark


def load_entities() -> list[dict] | None:
    if not IN_PATH.exists():
        print(f"HATA: {IN_PATH} yok. Önce: python3 src/07a_extract_entities.py")
        return None
    try:
        data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"HATA: entities_raw okunamadı: {exc}")
        return None
    if not isinstance(data, list):
        print("HATA: entities_raw.json liste değil")
        return None
    return [row for row in data if isinstance(row, dict)]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    records = load_entities()
    if records is None:
        print("0 yazar işlendi")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (OpenAlex aday).")
        return

    names = unique_author_names(records)
    print(f"Tekil yazar adı: {len(names)}")
    print(f"OpenAlex search  per_page={PER_PAGE}  sleep={SLEEP_SECONDS}s")
    print(f"cache={CACHE_DIR}")
    print("seçim yok: her ad için en fazla 5 aday (author_id henüz bağlanmaz)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    mapping: dict[str, list[dict]] = {}
    ok_count = 0
    skip_count = 0
    empty_count = 0
    ambiguous: list[str] = []

    with Cache(str(CACHE_DIR)) as oa_cache:
        for name in tqdm(names, desc="OpenAlex authors"):
            try:
                candidates, mark = search_author_candidates(session, oa_cache, name)
            except Exception as exc:
                print(f"SKIP  {name}  HATA: {exc}")
                mapping[name] = []
                skip_count += 1
                continue

            mapping[name] = candidates
            ok_count += 1
            if not candidates:
                empty_count += 1
                print(f"OK    {name}  [{mark}]  aday=0")
                continue

            top = candidates[0]
            print(
                f"OK    {name}  [{mark}]  aday={len(candidates)}  "
                f"top={top['author_id']}  inst={top['institution'] or '-'}  "
                f"works={top['works']}  score={top['score']}"
            )
            if len(candidates) >= 2 and candidates[1]["score"] >= 0.9 and top["score"] >= 0.9:
                ambiguous.append(name)
                print(
                    f"  AMBIG  aynı ada yakın: {top['author_id']} vs {candidates[1]['author_id']}"
                )

    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Kaydedildi: {OUT_PATH}")
    except Exception as exc:
        print(f"HATA: JSON yazılamadı: {exc}")
        print("0 yazar işlendi")
        return

    print(f"{ok_count} yazar işlendi  SKIP={skip_count}  boş_aday={empty_count}")
    if ambiguous:
        print(f"Belirsiz ad (homonym): {len(ambiguous)}  ör. {ambiguous[:5]}")
    print(
        "Evaluation: recall@k=n/a  faithfulness=n/a  "
        "(aday listesi; kanonik author_id henüz seçilmedi)."
    )


if __name__ == "__main__":
    main()
