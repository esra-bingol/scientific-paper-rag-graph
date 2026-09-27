from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "knowledge_graph.gpickle"

NAME_THRESHOLD = 0.85


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


def format_path(nodes_and_rels: list[str]) -> str:
    return " — ".join(nodes_and_rels)


def author_key(author_id: str) -> tuple[str, str]:
    return ("author", author_id)


def paper_payload(graph, paper_node: tuple) -> dict:
    data = graph.nodes[paper_node]
    return {
        "arxiv_id": str(node_id(paper_node)),
        "title": str(data.get("title") or ""),
        "year": data.get("year"),
    }


def author_payload(graph, author_node: tuple) -> dict:
    data = graph.nodes[author_node]
    return {
        "author_id": str(node_id(author_node)),
        "name": str(data.get("name") or ""),
        "aliases": list(data.get("aliases") or []),
        "institution": str(data.get("institution") or ""),
    }


def name_labels(data: dict) -> list[str]:
    labels = [str(data.get("name") or "")]
    labels.extend(str(item) for item in (data.get("aliases") or []))
    return [item.strip() for item in labels if item and str(item).strip()]


def name_score(query: str, label: str) -> float:
    """0-1. Kısa sorgu ('Esra') rastgele ada yapışmasın."""
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


def get_author_by_name(graph, name: str) -> dict:
    """Alias + name + author_id taraması. Kimlik hâlâ author_id."""
    query = " ".join((name or "").split())
    matches: list[dict] = []
    paths: list[str] = []
    if not query:
        return {"matches": [], "paths": [], "query": name}

    if query.startswith("A") and author_key(query) in graph:
        node = author_key(query)
        payload = author_payload(graph, node)
        payload["score"] = 1.0
        payload["matched_label"] = query
        matches.append(payload)
        paths.append(format_path([f"id {query}", "author_id", format_node(node)]))
        return {"matches": matches, "paths": paths, "query": query}

    for node, data in graph.nodes(data=True):
        if node_kind(node) != "author":
            continue
        labels = name_labels(data) + [str(node_id(node))]
        best_label = ""
        best = 0.0
        for label in labels:
            score = name_score(query, label)
            if score > best:
                best = score
                best_label = label
        if best < NAME_THRESHOLD:
            continue
        payload = author_payload(graph, node)
        payload["score"] = best
        payload["matched_label"] = best_label
        matches.append(payload)
        paths.append(
            format_path(
                [
                    f"alias/name '{best_label}'",
                    f"score={best}",
                    format_node(node),
                ]
            )
        )
    matches.sort(key=lambda row: row.get("score") or 0.0, reverse=True)
    return {"matches": matches, "paths": paths, "query": query}


def get_author_papers(graph, author_id: str) -> dict:
    node = author_key(author_id)
    if node not in graph:
        return {
            "author_id": author_id,
            "papers": [],
            "paths": [format_path([f"author:{author_id}", "YOK", "düğüm yok"])],
        }
    papers: list[dict] = []
    paths: list[str] = []
    for _src, dst, key, data in graph.out_edges(node, keys=True, data=True):
        relation = str(data.get("relation") or key)
        if relation != "YAZDI" or node_kind(dst) != "paper":
            continue
        papers.append(paper_payload(graph, dst))
        paths.append(format_path([format_node(node), "YAZDI", format_node(dst)]))
    papers.sort(key=lambda row: (row.get("year") is None, row.get("year") or 0, row["arxiv_id"]))
    return {"author_id": author_id, "papers": papers, "paths": paths}


def get_author_institution(graph, author_id: str) -> dict:
    node = author_key(author_id)
    if node not in graph:
        return {
            "author_id": author_id,
            "institutions": [],
            "paths": [format_path([f"author:{author_id}", "YOK", "düğüm yok"])],
        }
    institutions: list[str] = []
    paths: list[str] = []
    for _src, dst, key, data in graph.out_edges(node, keys=True, data=True):
        relation = str(data.get("relation") or key)
        if relation != "ÇALIŞIYOR" or node_kind(dst) != "institution":
            continue
        name = str(node_id(dst))
        institutions.append(name)
        paths.append(format_path([format_node(node), "ÇALIŞIYOR", format_node(dst)]))
    attr = str(graph.nodes[node].get("institution") or "").strip()
    if attr and attr not in institutions:
        institutions.append(attr)
        paths.append(
            format_path([format_node(node), "attr.institution", f"institution:{attr}"])
        )
    return {"author_id": author_id, "institutions": institutions, "paths": paths}


def get_papers_criticizing(graph, method: str) -> dict:
    query = " ".join((method or "").split())
    method_nodes: list[tuple] = []
    for node, _data in graph.nodes(data=True):
        if node_kind(node) != "method":
            continue
        label = str(node_id(node))
        if name_score(query, label) >= NAME_THRESHOLD:
            method_nodes.append(node)

    papers: list[dict] = []
    paths: list[str] = []
    seen: set[str] = set()
    for method_node in method_nodes:
        for src, _dst, key, data in graph.in_edges(method_node, keys=True, data=True):
            relation = str(data.get("relation") or key)
            if relation != "ELEŞTİRİYOR" or node_kind(src) != "paper":
                continue
            arxiv_id = str(node_id(src))
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            papers.append(paper_payload(graph, src))
            paths.append(
                format_path([format_node(src), "ELEŞTİRİYOR", format_node(method_node)])
            )
    papers.sort(key=lambda row: (row.get("year") is None, row.get("year") or 0, row["arxiv_id"]))
    return {"method": query, "method_nodes": [str(node_id(n)) for n in method_nodes], "papers": papers, "paths": paths}


def get_coauthors(graph, author_id: str) -> dict:
    node = author_key(author_id)
    if node not in graph:
        return {
            "author_id": author_id,
            "coauthors": [],
            "paths": [format_path([f"author:{author_id}", "YOK", "düğüm yok"])],
        }
    own = get_author_papers(graph, author_id)
    by_id: dict[str, dict] = {}
    paths: list[str] = []
    for paper in own["papers"]:
        paper_node = ("paper", paper["arxiv_id"])
        for src, _dst, key, data in graph.in_edges(paper_node, keys=True, data=True):
            relation = str(data.get("relation") or key)
            if relation != "YAZDI" or node_kind(src) != "author":
                continue
            other_id = str(node_id(src))
            if other_id == author_id:
                continue
            if other_id not in by_id:
                payload = author_payload(graph, src)
                payload["shared_papers"] = []
                by_id[other_id] = payload
            by_id[other_id]["shared_papers"].append(paper["arxiv_id"])
            paths.append(
                format_path(
                    [
                        format_node(node),
                        "YAZDI",
                        format_node(paper_node),
                        "YAZDI⁻¹",
                        format_node(src),
                    ]
                )
            )
    coauthors = sorted(by_id.values(), key=lambda row: row.get("name") or row["author_id"])
    return {"author_id": author_id, "coauthors": coauthors, "paths": paths}


def print_paths(paths: list[str]) -> None:
    if not paths:
        print("  path: (yok)")
        return
    print("  path:")
    for line in paths:
        print(f"    {line}")


def print_author_bundle(graph, author_id: str, heading: str) -> None:
    print(f"\n=== {heading} ===")
    papers = get_author_papers(graph, author_id)
    inst = get_author_institution(graph, author_id)
    peers = get_coauthors(graph, author_id)

    if papers["papers"]:
        print("Makaleler:")
        for row in papers["papers"]:
            print(f"  - {row['arxiv_id']}  year={row['year']!r}  {row['title']}")
    else:
        print("Makaleler: yok")
    print_paths(papers["paths"])

    if inst["institutions"]:
        print("Kurum:")
        for name in inst["institutions"]:
            print(f"  - {name}")
    else:
        print("Kurum: yok")
    print_paths(inst["paths"])

    if peers["coauthors"]:
        print("Coauthor:")
        for row in peers["coauthors"]:
            shared = ", ".join(row.get("shared_papers") or [])
            print(f"  - {row['author_id']}  {row['name']}  ortak={shared}")
    else:
        print("Coauthor: yok")
    print_paths(peers["paths"])


def run_author_query(graph, name: str) -> None:
    print(f"Sorgu ad: {name!r}  (alias → author_id)")
    found = get_author_by_name(graph, name)
    print_paths(found["paths"])
    if not found["matches"]:
        print(f"Yazar yok: {name!r}  (grafte alias/name eşleşmedi)")
        print("Makaleler: yok")
        print("Kurum: yok")
        print("Coauthor: yok")
        return
    if len(found["matches"]) > 1:
        print(f"{len(found['matches'])} aday (homonym); hepsi listelenir, tek id uydurulmaz:")
        for row in found["matches"]:
            print(
                f"  - {row['author_id']}  {row['name']}  "
                f"score={row['score']}  label={row['matched_label']!r}"
            )
    for row in found["matches"]:
        print_author_bundle(graph, row["author_id"], f"{row['name']} ({row['author_id']})")


def run_method_query(graph, method: str) -> None:
    print(f"Sorgu yöntem: {method!r}  (ELEŞTİRİYOR)")
    result = get_papers_criticizing(graph, method)
    if result["method_nodes"]:
        print(f"method düğüm: {result['method_nodes']}")
    if result["papers"]:
        print("Eleştiren makaleler:")
        for row in result["papers"]:
            print(f"  - {row['arxiv_id']}  year={row['year']!r}  {row['title']}")
    else:
        print("Eleştiren makale: yok")
    print_paths(result["paths"])


def run_menu(graph) -> None:
    print("Graf sorgu menüsü. Cosine yok; kenar yürüyüşü.")
    print("1 get_author_papers")
    print("2 get_author_institution")
    print("3 get_papers_criticizing")
    print("4 get_coauthors")
    print("5 get_author_by_name")
    print("0 çıkış")
    while True:
        try:
            choice = input("seçim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in {"0", "q", "çık", "exit"}:
            return
        try:
            if choice == "1":
                author_id = input("author_id> ").strip()
                result = get_author_papers(graph, author_id)
                print(result["papers"] or "yok")
                print_paths(result["paths"])
            elif choice == "2":
                author_id = input("author_id> ").strip()
                result = get_author_institution(graph, author_id)
                print(result["institutions"] or "yok")
                print_paths(result["paths"])
            elif choice == "3":
                method = input("method> ").strip()
                run_method_query(graph, method)
            elif choice == "4":
                author_id = input("author_id> ").strip()
                result = get_coauthors(graph, author_id)
                print(result["coauthors"] or "yok")
                print_paths(result["paths"])
            elif choice == "5":
                name = input("name> ").strip()
                run_author_query(graph, name)
            else:
                print("1-5 veya 0")
        except Exception as exc:
            print(f"HATA: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NetworkX graf hop sorgusu.")
    parser.add_argument("--author", default="", help="İsim veya alias (ör. Esra, Zhao Song)")
    parser.add_argument("--method", default="", help="Eleştirilen yöntem (ör. RNN)")
    parser.add_argument("--menu", action="store_true", help="İnteraktif menü")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        graph = load_graph()
    except Exception as exc:
        print(f"HATA: graf yüklenemedi: {exc}")
        print("Evaluation: recall@k=n/a  faithfulness=n/a  (graph query).")
        return

    print(f"Graf: {graph.number_of_nodes()} node, {graph.number_of_edges()} edge")

    ran = False
    if args.author.strip():
        run_author_query(graph, args.author)
        ran = True
    if args.method.strip():
        run_method_query(graph, args.method)
        ran = True
    if args.menu or not ran:
        run_menu(graph)

    print("Evaluation: recall@k=n/a  faithfulness=n/a  (graf hop; gold yok).")


if __name__ == "__main__":
    main()
