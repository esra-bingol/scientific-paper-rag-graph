"""
Graph hop + vektör chunk listelerini birleştirir; varsayılan fusion RRF.

Nasıl çalıştırılır:
    source .venv/bin/activate
    python3 src/10_hybrid_rag.py --question "Zhao Song'un makaleleri neler?"
    python3 src/10_hybrid_rag.py --fusion weighted --question "RNN'i kim eleştiriyor?"

Girdi:  data/graph/knowledge_graph.gpickle  +  chroma_db/papers_grobid
Önkoşul: python3 src/08_build_graph.py
         python3 src/04_embed.py --source grobid

Neden networkx: yazar/yöntem hop; cosine yıl-yazar kaçırır.
Neden chromadb: gövde parçası (tanım, iddia) grafte yok.
Neden openai: entity JSON + cevap.
Neden diskcache: aynı soru/embedding tekrar ödenmesin.

RRF (k=60): rrf = Σ 1/(60 + rank_i).
Neden RRF, weighted sum değil:
  Graf hop sayısı ile cosine distance aynı ölçekte değil.
  Weighted sum önce min-max/z-score ister; ölçek seçimi cevabı değiştirir.
  RRF yalnız sıra kullanır, normalizasyon gerekmez.
  --fusion weighted yalnızca bu farkı göstermek için var; varsayılan rrf.

recall@k / faithfulness: gold yok; n/a.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from diskcache import Cache
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "knowledge_graph.gpickle"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EMBED_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"
LLM_CACHE_DIR = PROJECT_ROOT / "cache" / "llm"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
# Cormack et al. RRF; k=60 sıra gürültüsünü yumuşatır.
RRF_K = 60
TOP_N = 5
VECTOR_K = 15
NAME_THRESHOLD = 0.85
# weighted karşılaştırma — ölçek uydurması şart; üretimde kullanma.
WEIGHTED_GRAPH = 0.5
WEIGHTED_VECTOR = 0.5

COLLECTION_ALIASES = {
    "naive": "papers_naive",
    "papers_naive": "papers_naive",
    "grobid": "papers_grobid",
    "papers_grobid": "papers_grobid",
}

ENTITY_PROMPT = """Kullanıcı sorusundan varlık çıkar. Sadece JSON yaz.

Şema:
{{
  "author_name": string veya null,
  "topic": string veya null,
  "year_min": int veya null,
  "method": string veya null
}}

Kurallar:
- Yazar/soyad geçiyorsa author_name (ör. "Zhao Song", "Esra", "Li").
- Eleştirilen/önerilen model-yöntem → method (ör. "RNN", "Transformer").
- Konu etiketi → topic (ör. "attention", "time-series"). Yoksa null.
- "2023'ten sonra" → year_min=2024. "2023 ve sonrası" → year_min=2023.
- Belirsiz alan null.

Soru: {question}
"""

ANSWER_PROMPT = """İki kanıtın var: graf (kim yazdı / neyi eleştirdi) ve gövde parçaları.

Kurallar:
- Yazar/yöntem/yıl sorusunda graf path kaynak gerçeğidir. YAZDI path varsa o makale o yazara aittir; parçada isim geçmese de listele.
- Graf "yazar yok" / boş liste ise vektör parçasına kişi uydurma; "Bu bilgi verilen makalelerde yok" de.
- Gövde iddiası (nasıl çalışır, sonuç) için parçaları kullan. Uydurma.

Çıkarılan varlıklar: {entities}
Fusion: {fusion}
Graf makaleleri (sıralı): {graph_papers}
Graf path: {graph_paths}

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
    parser = argparse.ArgumentParser(description="Graph + vektör hibrit RAG (RRF).")
    parser.add_argument(
        "--fusion",
        choices=("rrf", "weighted"),
        default="rrf",
        help="rrf varsayılan. weighted = naive toplam (karşılaştırma).",
    )
    parser.add_argument("--question", default="", help="Boşsa klavyeden sorulur.")
    parser.add_argument("--collection", default="papers_grobid")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    return parser.parse_args()


def chat_cached(client: OpenAI, cache: Cache, prompt: str) -> str:
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


def extract_json_object(raw: str) -> dict:
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


def normalize_entities(raw: dict) -> dict[str, Any]:
    author_name = raw.get("author_name")
    if isinstance(author_name, str):
        author_name = author_name.strip() or None
    else:
        author_name = None

    topic = raw.get("topic")
    if isinstance(topic, str):
        topic = topic.strip() or None
    else:
        topic = None

    method = raw.get("method")
    if isinstance(method, str):
        method = method.strip() or None
    else:
        method = None

    year_min = raw.get("year_min")
    if not isinstance(year_min, int):
        year_min = None

    return {
        "author_name": author_name,
        "topic": topic,
        "year_min": year_min,
        "method": method,
    }


def load_graph(path: Path = GRAPH_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Önce: python3 src/08_build_graph.py")
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if graph is None:
        raise ValueError("graf boş")
    return graph


def node_kind(node: tuple) -> str:
    return str(node[0]) if node else ""


def node_id(node: tuple) -> Any:
    return node[1] if len(node) > 1 else None


def format_node(node: tuple) -> str:
    return f"{node_kind(node)}:{node_id(node)}"


def format_path(parts: list[str]) -> str:
    return " — ".join(parts)


def name_score(query: str, label: str) -> float:
    needle = " ".join(query.split()).casefold()
    hay = " ".join(label.split()).casefold()
    if not needle or not hay:
        return 0.0
    if needle == hay:
        return 1.0
    tokens = [tok for tok in hay.replace(",", " ").replace(".", " ").split() if tok]
    if needle in tokens:
        return 0.98
    if len(needle) >= 3 and any(tok.startswith(needle) for tok in tokens):
        return 0.92
    return round(fuzz.token_sort_ratio(needle, hay) / 100.0, 4)


def title_score(left: str, right: str) -> float:
    if not (left or "").strip() or not (right or "").strip():
        return 0.0
    return round(fuzz.token_set_ratio(left, right) / 100.0, 4)


def match_authors(graph, query: str) -> list[tuple]:
    hits: list[tuple] = []
    for node, data in graph.nodes(data=True):
        if node_kind(node) != "author":
            continue
        labels = [str(data.get("name") or "")]
        labels.extend(str(item) for item in (data.get("aliases") or []))
        labels.append(str(node_id(node)))
        best = max((name_score(query, label) for label in labels if label), default=0.0)
        if best >= NAME_THRESHOLD:
            hits.append((node, best))
    hits.sort(key=lambda item: item[1], reverse=True)
    return [node for node, _score in hits]


def match_methods(graph, query: str) -> list[tuple]:
    hits: list[tuple] = []
    for node, _data in graph.nodes(data=True):
        if node_kind(node) != "method":
            continue
        if name_score(query, str(node_id(node))) >= NAME_THRESHOLD:
            hits.append(node)
    return hits


def paper_record(graph, paper_node: tuple) -> dict:
    data = graph.nodes[paper_node]
    return {
        "arxiv_id": str(node_id(paper_node)),
        "title": str(data.get("title") or ""),
        "year": data.get("year"),
    }


def year_ok(year: object, year_min: int | None) -> bool:
    if year_min is None:
        return True
    if not isinstance(year, int):
        return False
    return year >= year_min


def graph_retrieve(graph, entities: dict[str, Any]) -> dict:
    """Sinyal sayısına göre paper sıralar (tek liste; henüz RRF değil)."""
    scores: dict[str, int] = {}
    meta: dict[str, dict] = {}
    paths: list[str] = []
    notes: list[str] = []

    def add_paper(paper_node: tuple, points: int, path: str) -> None:
        rec = paper_record(graph, paper_node)
        if not year_ok(rec.get("year"), entities["year_min"]):
            return
        arxiv_id = rec["arxiv_id"]
        scores[arxiv_id] = scores.get(arxiv_id, 0) + points
        meta[arxiv_id] = rec
        paths.append(path)

    if entities["author_name"]:
        authors = match_authors(graph, entities["author_name"])
        if not authors:
            notes.append(f"yazar yok: {entities['author_name']!r}")
            print(f"GRAPH  yazar yok: {entities['author_name']!r}")
        for author_node in authors:
            for _src, dst, key, data in graph.out_edges(author_node, keys=True, data=True):
                relation = str(data.get("relation") or key)
                if relation != "YAZDI" or node_kind(dst) != "paper":
                    continue
                add_paper(
                    dst,
                    100,
                    format_path([format_node(author_node), "YAZDI", format_node(dst)]),
                )

    method_query = entities["method"] or entities["topic"]
    if entities["method"]:
        method_nodes = match_methods(graph, entities["method"])
        if not method_nodes:
            notes.append(f"yöntem yok: {entities['method']!r}")
            print(f"GRAPH  yöntem yok: {entities['method']!r}")
        for method_node in method_nodes:
            for src, _dst, key, data in graph.in_edges(method_node, keys=True, data=True):
                relation = str(data.get("relation") or key)
                if node_kind(src) != "paper":
                    continue
                if relation == "ELEŞTİRİYOR":
                    add_paper(
                        src,
                        50,
                        format_path([format_node(src), "ELEŞTİRİYOR", format_node(method_node)]),
                    )
                elif relation == "ÖNERİYOR":
                    add_paper(
                        src,
                        40,
                        format_path([format_node(src), "ÖNERİYOR", format_node(method_node)]),
                    )

    if entities["topic"] and entities["topic"] != entities["method"]:
        for method_node in match_methods(graph, entities["topic"]):
            for src, _dst, key, data in graph.in_edges(method_node, keys=True, data=True):
                relation = str(data.get("relation") or key)
                if relation in {"ÖNERİYOR", "ELEŞTİRİYOR"} and node_kind(src) == "paper":
                    add_paper(
                        src,
                        20,
                        format_path(
                            [format_node(src), relation, format_node(method_node), "(topic)"]
                        ),
                    )
        for node, data in graph.nodes(data=True):
            if node_kind(node) != "paper":
                continue
            title = str(data.get("title") or "")
            if title_score(entities["topic"], title) >= 0.75:
                add_paper(
                    node,
                    10,
                    format_path([f"topic '{entities['topic']}'", "title≈", format_node(node)]),
                )

    # Yıl tek başına: o yıldan itibaren tüm kağıtlar (zayıf sinyal).
    if entities["year_min"] is not None and not scores and method_query is None and not entities["author_name"]:
        for node, data in graph.nodes(data=True):
            if node_kind(node) != "paper":
                continue
            if year_ok(data.get("year"), entities["year_min"]):
                add_paper(
                    node,
                    5,
                    format_path([format_node(node), "YAYINLANDI", f"year>={entities['year_min']}"]),
                )

    ranked_ids = sorted(
        scores,
        key=lambda arxiv_id: (
            -scores[arxiv_id],
            meta[arxiv_id].get("year") is None,
            -(meta[arxiv_id].get("year") or 0),
            arxiv_id,
        ),
    )
    papers = [meta[arxiv_id] for arxiv_id in ranked_ids]
    print(f"GRAPH  {len(papers)} makale  (sinyal sıralı, henüz fusion yok)")
    for rank, arxiv_id in enumerate(ranked_ids, start=1):
        rec = meta[arxiv_id]
        print(f"  g{rank}. {arxiv_id}  year={rec.get('year')!r}  sinyal={scores[arxiv_id]}")
    return {
        "ranked_ids": ranked_ids,
        "papers": papers,
        "scores": scores,
        "paths": paths,
        "notes": notes,
    }


def vector_retrieve(
    collection: chromadb.Collection,
    client: OpenAI,
    cache: Cache,
    question: str,
    n_results: int,
) -> dict:
    query_vector = embed_question(client, cache, question)
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_results, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    documents = (raw.get("documents") or [[]])[0] or []
    metadatas = (raw.get("metadatas") or [[]])[0] or []
    distances = (raw.get("distances") or [[]])[0] or []
    ids = (raw.get("ids") or [[]])[0] or []

    chunks: list[dict] = []
    paper_first_rank: dict[str, int] = {}
    paper_ids: list[str] = []
    for index, doc in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        arxiv_id = str(meta.get("arxiv_id") or "").strip()
        chunk_id = str(meta.get("chunk_id") or (ids[index] if index < len(ids) else ""))
        hit = {
            "text": doc or "",
            "arxiv_id": arxiv_id,
            "chunk_id": chunk_id,
            "distance": distances[index] if index < len(distances) else None,
            "section": str(meta.get("section") or ""),
            "year": meta.get("year"),
            "authors": str(meta.get("authors") or ""),
            "vector_rank": index + 1,
        }
        chunks.append(hit)
        if arxiv_id and arxiv_id not in paper_first_rank:
            paper_first_rank[arxiv_id] = index + 1
            paper_ids.append(arxiv_id)

    print(f"VECTOR {len(chunks)} chunk  →  {len(paper_ids)} makale (ilk görünme sırası)")
    for rank, arxiv_id in enumerate(paper_ids, start=1):
        print(f"  v{rank}. {arxiv_id}  first_chunk_rank={paper_first_rank[arxiv_id]}")
    return {
        "chunks": chunks,
        "ranked_ids": paper_ids,
        "paper_first_rank": paper_first_rank,
    }


def rrf_merge(graph_ids: list[str], vector_ids: list[str]) -> list[tuple[str, float]]:
    """
    rrf(doc) = Σ 1/(k + rank_i). k=60.
    Skorlar toplanmaz; her listede kaçıncı olduğun toplanır.
    Normalizasyon yok: cosine 0.2 ile hop 100 aynı formüle girmez, sıra girer.
    """
    lists = [graph_ids, vector_ids]
    universe: list[str] = []
    for ranking in lists:
        for doc_id in ranking:
            if doc_id and doc_id not in universe:
                universe.append(doc_id)

    scored: list[tuple[str, float]] = []
    for doc_id in universe:
        total = 0.0
        for ranking in lists:
            if doc_id not in ranking:
                continue
            rank = ranking.index(doc_id) + 1
            total += 1.0 / (RRF_K + rank)
        scored.append((doc_id, total))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def min_max(values: dict[str, float]) -> dict[str, float]:
    """weighted sum için zorunlu ölçek. RRF bunu yapmaz."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {key: 1.0 for key in values}
    return {key: (val - lo) / (hi - lo) for key, val in values.items()}


def weighted_merge(graph_ids: list[str], vector_ids: list[str]) -> list[tuple[str, float]]:
    """
    Naive weighted sum — karşılaştırma. Üretimde kullanma.
    1/rank sonra min-max; ağırlık 0.5/0.5 keyfî. Ölçek değişince sıra değişir.
    """
    graph_raw = {doc_id: 1.0 / (index + 1) for index, doc_id in enumerate(graph_ids)}
    vector_raw = {doc_id: 1.0 / (index + 1) for index, doc_id in enumerate(vector_ids)}
    graph_n = min_max(graph_raw)
    vector_n = min_max(vector_raw)
    universe = []
    for doc_id in list(graph_ids) + list(vector_ids):
        if doc_id and doc_id not in universe:
            universe.append(doc_id)
    scored: list[tuple[str, float]] = []
    for doc_id in universe:
        total = (
            WEIGHTED_GRAPH * graph_n.get(doc_id, 0.0)
            + WEIGHTED_VECTOR * vector_n.get(doc_id, 0.0)
        )
        scored.append((doc_id, total))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def fuse_papers(
    fusion: str,
    graph_ids: list[str],
    vector_ids: list[str],
) -> list[tuple[str, float]]:
    if fusion == "weighted":
        print(
            f"FUSION weighted  w_g={WEIGHTED_GRAPH} w_v={WEIGHTED_VECTOR}  "
            "(min-max şart; bu yüzden varsayılan değil)"
        )
        return weighted_merge(graph_ids, vector_ids)
    print(f"FUSION RRF  k={RRF_K}  score=Σ 1/({RRF_K}+rank_i)  normalizasyon yok")
    return rrf_merge(graph_ids, vector_ids)


def build_context_hits(
    fused: list[tuple[str, float]],
    vector_chunks: list[dict],
    graph_meta: dict[str, dict],
    top_n: int,
) -> list[dict]:
    """Top-N makale için vektör parçası; yoksa graf başlık kartı."""
    by_paper: dict[str, list[dict]] = {}
    for chunk in vector_chunks:
        arxiv_id = chunk.get("arxiv_id") or ""
        if not arxiv_id:
            continue
        by_paper.setdefault(arxiv_id, []).append(chunk)

    hits: list[dict] = []
    for arxiv_id, score in fused[:top_n]:
        chunks = by_paper.get(arxiv_id) or []
        if chunks:
            best = chunks[0]
            row = dict(best)
            row["fusion_score"] = score
            hits.append(row)
            continue
        rec = graph_meta.get(arxiv_id) or {"title": "", "year": None}
        hits.append(
            {
                "text": f"(graf) {rec.get('title') or arxiv_id}",
                "arxiv_id": arxiv_id,
                "chunk_id": f"{arxiv_id}_graph",
                "distance": None,
                "section": "graph",
                "year": rec.get("year"),
                "authors": "",
                "fusion_score": score,
            }
        )
    return hits


def format_context(hits: list[dict]) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits, start=1):
        extra = ""
        if hit.get("section"):
            extra += f" section={hit['section']}"
        if hit.get("year") not in (None, ""):
            extra += f" year={hit['year']}"
        parts.append(
            f"[{index}] arxiv_id={hit['arxiv_id']} chunk_id={hit['chunk_id']}{extra}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    try:
        collection_name = resolve_collection(args.collection)
    except ValueError as exc:
        print(f"HATA: {exc}")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return

    top_n = args.top_n if args.top_n > 0 else TOP_N
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("HATA: OPENAI_API_KEY yok. cp .env.example .env yapıp anahtarı yaz.")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return

    if args.question.strip():
        question = args.question.strip()
    else:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHATA: soru alınamadı.")
            print("Evaluation: recall@k=n/a  faithfulness=n/a")
            return
    if not question:
        print("HATA: boş soru.")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return
    print(f"Alınan: {question}")
    print(f"fusion={args.fusion}  top_n={top_n}  collection={collection_name}")

    try:
        graph = load_graph()
    except Exception as exc:
        print(f"HATA: graf yüklenemedi: {exc}")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return

    if not CHROMA_DIR.exists():
        print(f"HATA: {CHROMA_DIR} yok. Önce: python3 src/04_embed.py --source grobid")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return

    openai_client = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except Exception as exc:
        print(f"HATA: collection {collection_name!r} yok ({exc}).")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return
    if collection.count() == 0:
        print("HATA: collection boş.")
        print("Evaluation: recall@k=n/a  faithfulness=n/a")
        return

    print(f"graf={graph.number_of_nodes()} node  chroma n={collection.count()}")

    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with Cache(str(EMBED_CACHE_DIR)) as embed_cache, Cache(str(LLM_CACHE_DIR)) as llm_cache:
        try:
            raw_entities = chat_cached(
                openai_client, llm_cache, ENTITY_PROMPT.format(question=question)
            )
            entities = normalize_entities(extract_json_object(raw_entities))
        except Exception as exc:
            print(f"HATA: entity çıkarılamadı: {exc}")
            print("Evaluation: recall@k=n/a  faithfulness=n/a")
            return
        print(f"entity: {json.dumps(entities, ensure_ascii=False)}")

        started = time.perf_counter()
        try:
            graph_hit = graph_retrieve(graph, entities)
            vector_hit = vector_retrieve(
                collection, openai_client, embed_cache, question, VECTOR_K
            )
        except Exception as exc:
            print(f"HATA: retrieval: {exc}")
            print("Evaluation: recall@k=n/a  faithfulness=n/a")
            return
        retrieval_seconds = time.perf_counter() - started
        print(f"retrieval: {retrieval_seconds:.3f} s")

        fused = fuse_papers(args.fusion, graph_hit["ranked_ids"], vector_hit["ranked_ids"])
        print("FUSION sıra:")
        for rank, (arxiv_id, score) in enumerate(fused[: max(top_n, 8)], start=1):
            in_g = "G" if arxiv_id in graph_hit["ranked_ids"] else "-"
            in_v = "V" if arxiv_id in vector_hit["ranked_ids"] else "-"
            print(f"  {rank}. {arxiv_id}  score={score:.5f}  [{in_g}{in_v}]")

        graph_meta = {row["arxiv_id"]: row for row in graph_hit["papers"]}
        hits = build_context_hits(fused, vector_hit["chunks"], graph_meta, top_n)
        graph_paths = graph_hit["paths"][:12]
        if graph_hit["notes"]:
            print("GRAPH not:")
            for note in graph_hit["notes"]:
                print(f"  - {note}")

        if not hits and not graph_hit["ranked_ids"]:
            print("\n--- Cevap ---")
            print("Bu bilgi verilen makalelerde yok")
            print("\n--- Kaynaklar ---")
            print("  (graf + vektör boş)")
            print("\n--- Graph path ---")
            print("  (yok)")
            print("Evaluation: recall@k=n/a  faithfulness=n/a  (boş retrieval).")
            return

        graph_catalog = (
            "; ".join(
                f"{row['arxiv_id']} (year={row.get('year')}) {row.get('title') or ''}"
                for row in graph_hit["papers"]
            )
            or "(boş)"
        )
        prompt = ANSWER_PROMPT.format(
            entities=json.dumps(entities, ensure_ascii=False),
            fusion=args.fusion,
            graph_papers=graph_catalog,
            graph_paths="; ".join(graph_paths) or "(boş)",
            context=format_context(hits) or "(parça yok)",
            question=question,
        )
        try:
            llm_started = time.perf_counter()
            answer = chat_cached(openai_client, llm_cache, prompt)
            llm_seconds = time.perf_counter() - llm_started
        except Exception as exc:
            print(f"HATA: LLM: {exc}")
            print("Evaluation: recall@k=n/a  faithfulness=n/a")
            return

        print(f"LLM süresi: {llm_seconds:.3f} s")
        print("\n--- Cevap ---")
        print(answer)
        print("\n--- Kaynaklar ---")
        for hit in hits:
            print(
                f"  arxiv_id={hit['arxiv_id']}  chunk_id={hit['chunk_id']}  "
                f"section={hit.get('section') or '-'}  "
                f"fusion={hit.get('fusion_score')}"
            )
        print("\n--- Graph path ---")
        if graph_paths:
            for line in graph_paths:
                print(f"  {line}")
        else:
            print("  (yok)")
        print(
            "Evaluation: recall@k=n/a (gold yok)  "
            "faithfulness=n/a (judge yok; path + kaynağa bak)."
        )


if __name__ == "__main__":
    main()
