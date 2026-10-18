"""Turn results/all_results.json into Markdown tables (docs/RESULTS.md).

    python scripts/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METHOD = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + rerank"}
CHUNK_ORDER = ["fixed-200-o0", "fixed-200-o15", "fixed-500-o0", "fixed-500-o15",
               "fixed-1000-o0", "fixed-1000-o15", "heading", "paragraph"]


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def best(rows, key="mrr"):
    return max(rows, key=lambda r: r[key]) if rows else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = json.loads((ROOT / "results" / "all_results.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["embedder"] != "tfidf" or r["method"] == "bm25"] or rows
    n_q = rows[0]["n_questions"]
    parts = [f"# Results\n\n{len(rows)} configurations, {n_q} questions, 14 documents. "
             "Recall@k and MRR use the top-k list; MRR is MRR@k. Answer metrics use the offline "
             f"generator/judge unless noted (generator: `{rows[0]['generator']}`, judge: `{rows[0]['judge']}`).\n"]

    # 1. best per method at k=5
    lines = []
    for m in METHOD:
        b = best(pick(rows, method=m, k=5))
        if b:
            lines.append([METHOD[m], b["chunker"], b["embedder"], f3(b["recall@k"]), f3(b["mrr"]), f3(b["recall@1"]),
                          f3(b["correctness"]), f"{b['latency_ms']:.1f}", f"{b['index_time_s']:.1f}"])
    parts.append("## Best configuration per retrieval method (k = 5)\n\n" + table(
        ["Retrieval", "Chunking", "Embedding", "Recall@5", "MRR", "Recall@1", "Correctness", "Latency (ms)", "Index (s)"], lines))

    # 2. chunking, minilm hybrid k=5 + bm25
    lines = []
    for c in CHUNK_ORDER:
        h = pick(rows, chunker=c, embedder="minilm", method="hybrid", k=5)
        s = pick(rows, chunker=c, method="bm25", k=5)
        if h or s:
            h, s = (h or [None])[0], (s or [None])[0]
            ref = h or s
            lines.append([c, ref["n_chunks"], ref["avg_chunk_words"], f3(s and s["recall@k"]), f3(s and s["mrr"]),
                          f3(h and h["recall@k"]), f3(h and h["mrr"]),
                          f3(h and h["recall_by_type"].get("factoid")), f3(h and h["recall_by_type"].get("explanatory")),
                          f3(h and h["correctness"])])
    parts.append("## Chunking strategy (k = 5)\n\n" + table(
        ["Chunking", "Chunks", "Avg words", "BM25 R@5", "BM25 MRR", "Hybrid R@5", "Hybrid MRR",
         "Hybrid R@5 factoid", "Hybrid R@5 explanatory", "Correctness"], lines))

    # 3. embeddings on fixed-200-o15
    lines = []
    for e in ["tfidf", "minilm", "bge-small", "e5-base", "api"]:
        all_rows = json.loads((ROOT / "results" / "all_results.json").read_text(encoding="utf-8"))
        d = pick(all_rows, chunker="fixed-200-o15", embedder=e, method="dense", k=5)
        h = pick(all_rows, chunker="fixed-200-o15", embedder=e, method="hybrid", k=5)
        if d or h:
            d, h = (d or [None])[0], (h or [None])[0]
            ref = d or h
            lines.append([e, f3(d and d["recall@k"]), f3(d and d["mrr"]), f3(h and h["recall@k"]), f3(h and h["mrr"]),
                          f"{ref['index_time_s']:.1f}", f"{(d or h)['retrieval_latency_ms']:.1f}", f"${ref['cost_per_1k_usd']}"])
    parts.append("## Embedding model (fixed-200-o15, k = 5)\n\n" + table(
        ["Embedding", "Dense R@5", "Dense MRR", "Hybrid R@5", "Hybrid MRR", "Index (s)", "Query (ms)", "$ / 1k q"], lines))

    # 4. top-k sweep for the overall best (chunker, embedder, method)
    top = best(rows)
    lines = []
    for k in (3, 5, 10):
        r = pick(rows, chunker=top["chunker"], embedder=top["embedder"], method=top["method"], k=k)
        if r:
            r = r[0]
            lines.append([k, f3(r["recall@k"]), f3(r["mrr"]), f3(r["correctness"]), f"{r['latency_ms']:.1f}"])
    parts.append(f"## Top-k ({top['chunker']} · {top['embedder']} · {METHOD[top['method']]})\n\n" + table(
        ["k", "Recall@k", "MRR", "Correctness", "Latency (ms)"], lines))

    # 5. reranker effect
    lines = []
    for r in pick(rows, method="hybrid_rerank", k=5):
        base = pick(rows, chunker=r["chunker"], embedder=r["embedder"], method="hybrid", k=5)
        if base:
            b = base[0]
            lines.append([r["chunker"], r["embedder"], f3(b["mrr"]), f3(r["mrr"]), f"{r['mrr'] - b['mrr']:+.3f}",
                          f3(b["recall@1"]), f3(r["recall@1"]), f"{b['latency_ms']:.0f}", f"{r['latency_ms']:.0f}"])
    lines.sort(key=lambda x: (x[0], x[1]))
    parts.append("## Does the reranker help? (hybrid vs hybrid + rerank, k = 5)\n\n" + table(
        ["Chunking", "Embedding", "Hybrid MRR", "+Rerank MRR", "Δ MRR", "Hybrid R@1", "+Rerank R@1",
         "Hybrid ms", "+Rerank ms"], lines))

    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
