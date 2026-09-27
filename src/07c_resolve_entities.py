"""
LLM yazar adlarını OpenAlex author_id'ye bağlar (kanonik kimlik).

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/07c_resolve_entities.py

Girdi:  data/entities_raw.json
        data/authors_candidates.json
Çıktı:  data/author_mapping.json
        data/unresolved_authors.json
Önkoşul: python3 src/07b_enrich_openalex.py

Neden rapidfuzz: ad / kurum / başlık benzerliği 0-1; regex "Yılmaz" vs "Yilmaz" kaçırır.
Neden requests: makalenin OpenAlex kaydından gerçek byline id'leri (ortak yazar kanıtı).
Neden diskcache: aynı DOI / ortak-makale sorgusu tekrar 1 sn beklenmesin (cache/openalex).
Neden python-dotenv: isteğe bağlı OPENALEX_MAILTO.
Neden tqdm: makale + yazar ilerleme.

Seçim: aynı OpenAlex makaledeki author_id > ortak makale > kurum tutarlılığı > ad skoru.
Retrieval fusion değil; RRF yok. Eşik 0.7 altı belirsiz.
recall@k / faithfulness: retrieval yok; n/a.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
ENTITIES_PATH = PROJECT_ROOT / "data" / "entities_raw.json"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "authors_candidates.json"
MAPPING_PATH = PROJECT_ROOT / "data" / "author_mapping.json"
UNRESOLVED_PATH = PROJECT_ROOT / "data" / "unresolved_authors.json"
CACHE_DIR = PROJECT_ROOT / "cache" / "openalex"

OPENALEX_BASE = "https://api.openalex.org"
WORKS_PATH = "/works"
SLEEP_SECONDS = 1.0
HTTP_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 3.0
CONFIDENCE_THRESHOLD = 0.7
TITLE_HIT = 0.75
INSTITUTION_HIT = 0.85

ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def cache_key(kind: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def short_openalex_id(openalex_id: str) -> str:
    return (openalex_id or "").rstrip("/").split("/")[-1].strip()


def name_score(left: str, right: str) -> float:
    """0-1; 'Mahoor, Mohammad H.' ile 'Mohammad H Mahoor' aynı kişi."""
    if not (left or "").strip() or not (right or "").strip():
        return 0.0
    return round(fuzz.token_sort_ratio(left, right) / 100.0, 4)


def title_score(left: str, right: str) -> float:
    if not (left or "").strip() or not (right or "").strip():
        return 0.0
    return round(fuzz.token_set_ratio(left, right) / 100.0, 4)


def institution_score(left: str, right: str) -> float:
    return title_score(left, right)


def make_aliases(query_name: str, display_name: str) -> list[str]:
    aliases: set[str] = set()
    parts = [p for p in query_name.replace(".", " ").split() if p]
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        aliases.add(f"{first[0]}. {last}")
        aliases.add(f"{first} {last[0]}.")
        aliases.add(f"{last}, {first}")
    display = (display_name or "").strip()
    if display and display.casefold() != query_name.casefold():
        aliases.add(display)
    return sorted(aliases)


def unique_author_names(records: list[dict]) -> list[str]:
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


def papers_by_author(records: list[dict]) -> dict[str, list[dict]]:
    """İsim (orijinal yazım) -> o ismin geçtiği makaleler."""
    index: dict[str, list[dict]] = {}
    for record in records:
        for raw in record.get("authors") or []:
            name = str(raw).strip()
            if not name:
                continue
            index.setdefault(name, []).append(record)
    return index


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_session() -> requests.Session:
    email = os.getenv("OPENALEX_MAILTO", "").strip()
    user_agent = "scientific-paper-rag-graph/07c"
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
                print(f"  429 rate limit  deneme={attempt}/{MAX_RETRIES}")
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


def parse_authorships(work: dict) -> list[dict]:
    rows: list[dict] = []
    for auth in work.get("authorships") or []:
        if not isinstance(auth, dict):
            continue
        author = auth.get("author") if isinstance(auth.get("author"), dict) else {}
        author_id = short_openalex_id(str(author.get("id") or ""))
        if not author_id.startswith("A"):
            continue
        inst = ""
        for block in auth.get("institutions") or []:
            if isinstance(block, dict) and block.get("display_name"):
                inst = str(block["display_name"]).strip()
                break
        rows.append(
            {
                "author_id": author_id,
                "display_name": str(author.get("display_name") or "").strip(),
                "raw_author_name": str(auth.get("raw_author_name") or "").strip(),
                "institution": inst,
            }
        )
    return rows


def find_paper_work(
    session: requests.Session,
    cache: Cache,
    arxiv_id: str,
    title: str,
) -> tuple[dict | None, str]:
    """Önce arXiv DOI, olmazsa başlık araması. Yoksa (None, neden)."""
    bare = ARXIV_VERSION.sub("", arxiv_id)
    doi = f"10.48550/arXiv.{bare}"
    try:
        payload, mark = openalex_get(
            session,
            cache,
            WORKS_PATH,
            {
                "filter": f"doi:{doi}",
                "per_page": 1,
                "select": "id,display_name,authorships,ids",
            },
        )
        results = [row for row in (payload.get("results") or []) if isinstance(row, dict)]
        if results:
            return results[0], mark
    except Exception as exc:
        print(f"  SKIP doi  {arxiv_id}: {exc}")

    if not title.strip():
        return None, "no_title"
    try:
        payload, mark = openalex_get(
            session,
            cache,
            WORKS_PATH,
            {"search": title, "per_page": 5, "select": "id,display_name,authorships,ids"},
        )
    except Exception as exc:
        print(f"  SKIP title  {arxiv_id}: {exc}")
        return None, "error"

    best: dict | None = None
    best_sim = 0.0
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        sim = title_score(title, str(row.get("display_name") or ""))
        if sim > best_sim:
            best_sim = sim
            best = row
    if best is not None and best_sim >= TITLE_HIT:
        return best, mark
    return None, "not_found"


def have_joint_work(
    session: requests.Session,
    cache: Cache,
    left_id: str,
    right_id: str,
) -> bool:
    if not left_id or not right_id or left_id == right_id:
        return False
    first, second = sorted((left_id, right_id))
    try:
        payload, _mark = openalex_get(
            session,
            cache,
            WORKS_PATH,
            {
                "filter": f"author.id:{first},author.id:{second}",
                "per_page": 1,
                "select": "id",
            },
        )
    except Exception as exc:
        print(f"  SKIP joint  {first}+{second}: {exc}")
        return False
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    try:
        return int(meta.get("count") or 0) > 0
    except (TypeError, ValueError):
        return bool(payload.get("results"))


def byline_match(query_name: str, authorships: list[dict]) -> tuple[dict | None, float]:
    best: dict | None = None
    best_score = 0.0
    for row in authorships:
        score = max(
            name_score(query_name, row.get("display_name") or ""),
            name_score(query_name, row.get("raw_author_name") or ""),
        )
        if score > best_score:
            best_score = score
            best = row
    return best, best_score


def shared_work_title(candidate: dict, peer_candidates: list[dict]) -> bool:
    titles = [str(t).casefold() for t in (candidate.get("top_5_works") or []) if t]
    if not titles:
        return False
    for peer in peer_candidates:
        for title in peer.get("top_5_works") or []:
            needle = str(title).casefold()
            if needle and needle in titles:
                return True
    return False


def score_candidate(
    query_name: str,
    candidate: dict,
    paper_author_ids: set[str],
    peer_institutions: list[str],
    joint: bool,
    title_overlap: bool,
) -> dict:
    display = str(candidate.get("display_name") or "")
    institution = str(candidate.get("institution") or "").strip()
    author_id = str(candidate.get("author_id") or "").strip()
    base = name_score(query_name, display)
    inst_agree = 0.0
    if institution and peer_institutions:
        inst_agree = max(institution_score(institution, peer) for peer in peer_institutions)

    evidence: list[str] = ["name"]
    confidence = base
    on_paper = author_id in paper_author_ids
    if on_paper:
        evidence.append("same_paper")
        confidence = max(confidence, 0.92)
    if title_overlap:
        evidence.append("title_overlap")
        confidence = max(confidence, 0.8)
    if joint:
        evidence.append("coauthor")
        confidence = max(confidence, 0.85)
    if inst_agree >= INSTITUTION_HIT:
        evidence.append("institution")
        confidence = min(1.0, round(confidence + 0.05, 4))

    return {
        "author_id": author_id,
        "display_name": display,
        "institution": institution,
        "confidence": round(confidence, 4),
        "name_score": base,
        "institution_agree": round(inst_agree, 4),
        "evidence": evidence,
        "hard": on_paper or joint or title_overlap,
    }


def pick_winner(scored: list[dict]) -> tuple[dict | None, dict | None]:
    if not scored:
        return None, None
    ranked = sorted(
        scored,
        key=lambda row: (
            1 if row.get("hard") else 0,
            row.get("confidence") or 0.0,
            row.get("institution_agree") or 0.0,
        ),
        reverse=True,
    )
    return ranked[0], ranked[1] if len(ranked) > 1 else None


def is_ambiguous(winner: dict, second: dict | None) -> bool:
    if winner["confidence"] < CONFIDENCE_THRESHOLD:
        return True
    if winner.get("hard"):
        return False
    if second is None:
        return False
    # İki homonym, ikisi de eşik üstü, kağıt kanıtı yok → belirsiz.
    gap = winner["confidence"] - second["confidence"]
    return second["confidence"] >= CONFIDENCE_THRESHOLD and gap < 0.05


def mapping_record(query_name: str, winner: dict) -> dict:
    return {
        "author_id": winner["author_id"],
        "confidence": winner["confidence"],
        "institution": winner.get("institution") or "",
        "aliases": make_aliases(query_name, str(winner.get("display_name") or "")),
    }


def unresolved_record(query_name: str, winner: dict | None, reason: str) -> dict:
    row = {
        "author_id": winner["author_id"] if winner else None,
        "confidence": winner["confidence"] if winner else 0.0,
        "institution": (winner.get("institution") if winner else "") or "",
        "aliases": make_aliases(query_name, str((winner or {}).get("display_name") or "")),
        "reason": reason,
        "evidence": (winner or {}).get("evidence") or [],
    }
    return row


def load_inputs() -> tuple[list[dict], dict[str, list]] | None:
    if not ENTITIES_PATH.exists():
        print(f"HATA: {ENTITIES_PATH} yok. Önce: python3 src/07a_extract_entities.py")
        return None
    if not CANDIDATES_PATH.exists():
        print(f"HATA: {CANDIDATES_PATH} yok. Önce: python3 src/07b_enrich_openalex.py")
        return None
    try:
        entities = load_json(ENTITIES_PATH)
        candidates = load_json(CANDIDATES_PATH)
    except Exception as exc:
        print(f"HATA: JSON okunamadı: {exc}")
        return None
    if not isinstance(entities, list) or not isinstance(candidates, dict):
        print("HATA: entities liste, candidates nesne olmalı")
        return None
    return [row for row in entities if isinstance(row, dict)], candidates


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    loaded = load_inputs()
    if loaded is None:
        print("0 eşleşti, 0 belirsiz")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (entity resolve).")
        return

    entities, candidates_map = loaded
    names = unique_author_names(entities)
    by_author = papers_by_author(entities)
    print(f"Tekil yazar: {len(names)}  eşik={CONFIDENCE_THRESHOLD}")
    print("kanonik id = OpenAlex author_id; seçim aday + aynı makale byline")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()

    paper_works: dict[str, dict] = {}
    with Cache(str(CACHE_DIR)) as oa_cache:
        paper_records = [
            row for row in entities if str(row.get("arxiv_id") or "").strip()
        ]
        for record in tqdm(paper_records, desc="OpenAlex works"):
            arxiv_id = str(record.get("arxiv_id") or "")
            title = str(record.get("title") or "")
            try:
                work, mark = find_paper_work(session, oa_cache, arxiv_id, title)
            except Exception as exc:
                print(f"SKIP  work  {arxiv_id}: {exc}")
                continue
            if work is None:
                print(f"MISS  {arxiv_id}  OpenAlex makale yok")
                continue
            authorships = parse_authorships(work)
            paper_works[arxiv_id] = {
                "title": str(work.get("display_name") or title),
                "authorships": authorships,
                "author_ids": {row["author_id"] for row in authorships},
                "mark": mark,
            }
            print(f"WORK  {arxiv_id}  [{mark}]  yazar={len(authorships)}")

        mapping: dict[str, dict] = {}
        unresolved: dict[str, dict] = {}

        for name in tqdm(names, desc="Resolve authors"):
            papers = by_author.get(name) or []
            raw_candidates = candidates_map.get(name) or []
            if not isinstance(raw_candidates, list):
                raw_candidates = []

            paper_ids: set[str] = set()
            all_authorships: list[dict] = []
            peer_institutions: list[str] = []
            for paper in papers:
                meta = paper_works.get(str(paper.get("arxiv_id") or ""))
                if not meta:
                    continue
                paper_ids |= meta["author_ids"]
                all_authorships.extend(meta["authorships"])
                for auth in meta["authorships"]:
                    if auth.get("institution"):
                        peer_institutions.append(auth["institution"])

            # Byline: OpenAlex makaledeki isimle eşle (aday listesinde olmasa da).
            line, line_score = byline_match(name, all_authorships)
            extra: list[dict] = []
            if line is not None and line_score >= CONFIDENCE_THRESHOLD:
                extra.append(
                    {
                        "author_id": line["author_id"],
                        "display_name": line["display_name"] or line["raw_author_name"],
                        "institution": line["institution"],
                        "top_5_works": [],
                    }
                )

            # Kurum sinyali: aynı makaledeki diğer adların 07b 1. adayı.
            for paper in papers:
                for raw in paper.get("authors") or []:
                    peer = str(raw).strip()
                    if not peer or peer == name:
                        continue
                    peers = candidates_map.get(peer) or []
                    if peers and isinstance(peers, list) and isinstance(peers[0], dict):
                        inst = str(peers[0].get("institution") or "").strip()
                        if inst:
                            peer_institutions.append(inst)

            seen_ids: set[str] = set()
            pool: list[dict] = []
            for cand in extra + [c for c in raw_candidates if isinstance(c, dict)]:
                author_id = str(cand.get("author_id") or "")
                if not author_id or author_id in seen_ids:
                    continue
                seen_ids.add(author_id)
                pool.append(cand)

            if not pool:
                unresolved[name] = unresolved_record(name, None, "no_candidates")
                print(f"AMBIG {name}  aday yok")
                continue

            scored: list[dict] = []
            for cand in pool:
                author_id = str(cand.get("author_id") or "")
                title_overlap = False
                for paper in papers:
                    paper_title = str(paper.get("title") or "")
                    for work_title in cand.get("top_5_works") or []:
                        if title_score(paper_title, str(work_title)) >= TITLE_HIT:
                            title_overlap = True
                            break
                    if title_overlap:
                        break

                joint = False
                # Önce ucuz sinyal: top_5 başlık kesişimi. Yetmezse OpenAlex ortak makale.
                peer_cands: list[dict] = []
                resolved_peer_ids: list[str] = []
                for paper in papers:
                    for raw in paper.get("authors") or []:
                        peer = str(raw).strip()
                        if not peer or peer == name:
                            continue
                        if peer in mapping:
                            resolved_peer_ids.append(mapping[peer]["author_id"])
                        peer_cands.extend(
                            c for c in (candidates_map.get(peer) or [])[:2] if isinstance(c, dict)
                        )
                if shared_work_title(cand, peer_cands):
                    joint = True
                elif author_id not in paper_ids:
                    for peer_id in resolved_peer_ids[:2]:
                        if have_joint_work(session, oa_cache, author_id, peer_id):
                            joint = True
                            break

                scored.append(
                    score_candidate(
                        name,
                        cand,
                        paper_ids,
                        peer_institutions,
                        joint,
                        title_overlap,
                    )
                )

            winner, second = pick_winner(scored)
            if winner is None:
                unresolved[name] = unresolved_record(name, None, "no_candidates")
                print(f"AMBIG {name}  skor yok")
                continue

            if is_ambiguous(winner, second):
                reason = (
                    "confidence < 0.7"
                    if winner["confidence"] < CONFIDENCE_THRESHOLD
                    else "homonym_no_paper_evidence"
                )
                unresolved[name] = unresolved_record(name, winner, reason)
                print(
                    f"AMBIG {name}  conf={winner['confidence']}  "
                    f"{winner['author_id']}  {reason}"
                )
                continue

            mapping[name] = mapping_record(name, winner)
            print(
                f"OK    {name}  {winner['author_id']}  conf={winner['confidence']}  "
                f"via={'+'.join(winner['evidence'])}  inst={winner['institution'] or '-'}"
            )

    try:
        MAPPING_PATH.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        UNRESOLVED_PATH.write_text(
            json.dumps(unresolved, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Kaydedildi: {MAPPING_PATH}")
        print(f"Kaydedildi: {UNRESOLVED_PATH}")
    except Exception as exc:
        print(f"HATA: JSON yazılamadı: {exc}")
        print("0 eşleşti, 0 belirsiz")
        return

    matched = len(mapping)
    unsure = len(unresolved)
    print(f"{matched} eşleşti, {unsure} belirsiz")
    print(
        "Evaluation: recall@k=n/a  faithfulness=n/a  "
        "(entity resolve; gold author_id yok)."
    )


if __name__ == "__main__":
    main()
