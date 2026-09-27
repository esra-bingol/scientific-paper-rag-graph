from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTITIES_PATH = PROJECT_ROOT / "data" / "entities_raw.json"
MAPPING_PATH = PROJECT_ROOT / "data" / "author_mapping.json"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
OUT_PATH = PROJECT_ROOT / "data" / "graph" / "knowledge_graph.gpickle"

ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def year_from_arxiv_id(arxiv_id: str) -> int | None:
    """YYMM.xxxxx → 20YY. metadata yoksa yedek."""
    bare = ARXIV_VERSION.sub("", arxiv_id)
    prefix = bare.split(".")[0]
    if len(prefix) >= 2 and prefix[:2].isdigit():
        return 2000 + int(prefix[:2])
    return None


def load_year_index(path: Path) -> dict[str, int | None]:
    index: dict[str, int | None] = {}
    if not path.exists():
        print(f"WARN  {path} yok; yıl arXiv id'den tahmin edilecek")
        return index
    try:
        rows = load_json(path)
    except Exception as exc:
        print(f"WARN  metadata okunamadı ({exc}); yıl arXiv id'den")
        return index
    if not isinstance(rows, list):
        return index
    for row in rows:
        if not isinstance(row, dict):
            continue
        arxiv_id = str(row.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        year = row.get("year")
        if isinstance(year, int):
            index[arxiv_id] = year
        elif isinstance(year, str) and year.isdigit():
            index[arxiv_id] = int(year)
        else:
            index[arxiv_id] = None
    return index


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def unique_strings(values: list[object]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for raw in values:
        text = clean_text(raw)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def display_name(llm_name: str, aliases: list[str]) -> str:
    """Etiket; kimlik değil. En uzun yazımı göster."""
    pool = [name for name in [llm_name, *aliases] if name]
    if not pool:
        return llm_name
    return max(pool, key=len)


def add_author_node(
    graph: nx.MultiDiGraph,
    author_id: str,
    llm_name: str,
    aliases: list[str],
    institution: str,
) -> None:
    key = ("author", author_id)
    all_aliases = unique_strings([*aliases, llm_name])
    label = display_name(llm_name, all_aliases)
    if key in graph:
        old = graph.nodes[key]
        merged = unique_strings([*(old.get("aliases") or []), *all_aliases])
        graph.nodes[key]["aliases"] = merged
        if not old.get("institution") and institution:
            graph.nodes[key]["institution"] = institution
        if len(label) > len(str(old.get("name") or "")):
            graph.nodes[key]["name"] = label
        return
    graph.add_node(
        key,
        kind="author",
        name=label,
        aliases=all_aliases,
        institution=institution,
    )


def add_simple_node(graph: nx.MultiDiGraph, kind: str, ident: str | int) -> tuple:
    key = (kind, ident)
    if key not in graph:
        graph.add_node(key, kind=kind)
    return key


def add_relation(
    graph: nx.MultiDiGraph,
    src: tuple,
    dst: tuple,
    relation: str,
) -> None:
    if graph.has_edge(src, dst, key=relation):
        return
    graph.add_edge(src, dst, key=relation, relation=relation)


def load_inputs() -> tuple[list[dict], dict[str, dict]] | None:
    if not ENTITIES_PATH.exists():
        print(f"HATA: {ENTITIES_PATH} yok. Önce: python3 src/07a_extract_entities.py")
        return None
    if not MAPPING_PATH.exists():
        print(f"HATA: {MAPPING_PATH} yok. Önce: python3 src/07c_resolve_entities.py")
        return None
    try:
        entities = load_json(ENTITIES_PATH)
        mapping = load_json(MAPPING_PATH)
    except Exception as exc:
        print(f"HATA: JSON okunamadı: {exc}")
        return None
    if not isinstance(entities, list) or not isinstance(mapping, dict):
        print("HATA: entities liste, mapping nesne olmalı")
        return None
    clean_entities = [row for row in entities if isinstance(row, dict)]
    clean_mapping = {
        str(name): row
        for name, row in mapping.items()
        if isinstance(row, dict) and str(row.get("author_id") or "").startswith("A")
    }
    return clean_entities, clean_mapping


def build_graph(
    entities: list[dict],
    mapping: dict[str, dict],
    year_index: dict[str, int | None],
) -> nx.MultiDiGraph:
    graph: nx.MultiDiGraph = nx.MultiDiGraph()
    skip_authors = 0
    linked_authors = 0

    for record in entities:
        arxiv_id = clean_text(record.get("arxiv_id"))
        if not arxiv_id:
            print("SKIP  makale  arxiv_id yok")
            continue

        title = clean_text(record.get("title"))
        year = year_index.get(arxiv_id)
        if not isinstance(year, int):
            year = year_from_arxiv_id(arxiv_id)

        paper_key = ("paper", arxiv_id)
        graph.add_node(paper_key, kind="paper", title=title, year=year)
        print(f"PAPER {arxiv_id}  year={year!r}")

        if isinstance(year, int):
            year_key = add_simple_node(graph, "year", year)
            add_relation(graph, paper_key, year_key, "YAYINLANDI")

        for method in unique_strings(list(record.get("methods_proposed") or [])):
            method_key = add_simple_node(graph, "method", method)
            add_relation(graph, paper_key, method_key, "ÖNERİYOR")

        for method in unique_strings(list(record.get("methods_criticized") or [])):
            method_key = add_simple_node(graph, "method", method)
            add_relation(graph, paper_key, method_key, "ELEŞTİRİYOR")

        for raw_name in record.get("authors") or []:
            llm_name = clean_text(raw_name)
            if not llm_name:
                continue
            resolved = mapping.get(llm_name)
            if resolved is None:
                skip_authors += 1
                print(f"SKIP  {llm_name}  kanonik author_id yok (07c belirsiz)")
                continue

            author_id = str(resolved.get("author_id") or "").strip()
            institution = clean_text(resolved.get("institution"))
            aliases = [
                clean_text(item) for item in (resolved.get("aliases") or []) if clean_text(item)
            ]
            add_author_node(graph, author_id, llm_name, aliases, institution)
            author_key = ("author", author_id)
            add_relation(graph, author_key, paper_key, "YAZDI")
            linked_authors += 1
            print(f"YAZDI {author_id}  {llm_name} → {arxiv_id}")

            if institution:
                inst_key = add_simple_node(graph, "institution", institution)
                add_relation(graph, author_key, inst_key, "ÇALIŞIYOR")

    print(f"YAZDI bağlandı={linked_authors}  belirsiz atlandı={skip_authors}")
    return graph


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Yuvarlak kontrol: dosya açılabiliyor mu?
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if loaded.number_of_nodes() != graph.number_of_nodes():
        raise ValueError("gpickle yazıldı ama node sayısı uyuşmuyor")


def summarize(graph: nx.MultiDiGraph) -> None:
    kinds = Counter(str(data.get("kind") or "?") for _, data in graph.nodes(data=True))
    relations = Counter(
        str(data.get("relation") or "?") for *_, data in graph.edges(keys=True, data=True)
    )
    for kind, count in sorted(kinds.items()):
        print(f"  node {kind}={count}")
    for relation, count in sorted(relations.items()):
        print(f"  edge {relation}={count}")


def main() -> None:
    loaded = load_inputs()
    if loaded is None:
        print("0 node, 0 edge")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (graph build).")
        return

    entities, mapping = loaded
    year_index = load_year_index(METADATA_PATH)
    print(f"Makale: {len(entities)}  kanonik yazar: {len(mapping)}")
    print("düğüm kimliği=OpenAlex author_id  LLM ad=aliases")

    try:
        graph = build_graph(entities, mapping, year_index)
    except Exception as exc:
        print(f"HATA: graf kurulamadı: {exc}")
        print("0 node, 0 edge")
        return

    try:
        save_graph(graph, OUT_PATH)
        print(f"Kaydedildi: {OUT_PATH}")
    except Exception as exc:
        print(f"HATA: gpickle yazılamadı: {exc}")
        print("0 node, 0 edge")
        return

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    print(f"{node_count} node, {edge_count} edge")
    summarize(graph)
    print(
        "Evaluation: recall@k=n/a  faithfulness=n/a  "
        "(graf kuruldu; retrieval/hop henüz yok)."
    )


if __name__ == "__main__":
    main()
