import json

from src.data import EvalItem
from src.evaluate import RunConfig, expand_grid, run_group, run_suite

DOCS = {
    "a": "Okapi BM25 is a ranking function. The parameter b is usually set to 0.75 in practice.",
    "b": "Cosine similarity is the dot product divided by the product of the vector lengths.",
    "c": "HNSW builds a navigable small world graph for approximate nearest neighbour search.",
}
ITEMS = [
    EvalItem("q1", "What is b usually set to in BM25?", "0.75", "usually set to 0.75", "a", "generated", "factoid"),
    EvalItem("q2", "How is cosine similarity computed?", "dot product over lengths",
             "the dot product divided by the product of the vector lengths", "b", "generated", "explanatory"),
    EvalItem("q3", "Which graph does HNSW build?", "navigable small world graph", "navigable small world graph", "c", "generated", "factoid"),
    EvalItem("q4", "What kind of function is BM25?", "a ranking function", "a ranking function", "a", "generated", "factoid"),
]


def test_expand_grid_counts():
    cfg = {"chunkers": [{"type": "fixed", "size": 200}, {"type": "paragraph"}],
           "embedders": ["tfidf"], "methods": ["dense", "bm25", "hybrid"], "ks": [3, 5]}
    runs = expand_grid(cfg)
    assert len(runs) == 12
    assert all(isinstance(r, RunConfig) for r in runs)
    # BM25 does not use an embedder
    assert {r.embedder for r in runs if r.method == "bm25"} == {"none"}
    assert len({r.run_id for r in runs}) == 12


def test_expand_grid_dedupes_bm25_across_embedders():
    cfg = {"chunkers": [{"type": "heading"}], "embedders": ["tfidf", "minilm"], "methods": ["bm25"], "ks": [5]}
    assert len(expand_grid(cfg)) == 1


def test_run_group_tiny():
    runs = [RunConfig({"type": "fixed", "size": 8}, "tfidf", m, k) for m in ("dense", "bm25") for k in (1, 3)]
    rows = run_group(runs, DOCS, ITEMS, with_answers=True)
    assert len(rows) == 4
    for r in rows:
        assert 0.0 <= r["recall@k"] <= 1.0 and 0.0 <= r["mrr"] <= 1.0
        assert len(r["per_question"]) == 4
        assert r["n_chunks"] > 3 and r["faithfulness"] is not None
        assert r["generator"] == "extractive" and r["judge"] == "heuristic"
        assert set(r["recall_by_type"]) == {"factoid", "explanatory"}
    by = {(r["method"], r["k"]): r for r in rows}
    assert by[("bm25", 3)]["recall@k"] >= by[("bm25", 1)]["recall@k"]


def test_rerun_overwrites_by_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate.load_documents", lambda: DOCS)
    monkeypatch.setattr("src.evaluate.load_eval", lambda: ITEMS)
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text("name: tiny\nchunkers: [{type: fixed, size: 8}]\nembedders: [tfidf]\n"
                   "methods: [bm25]\nks: [1, 3]\n", encoding="utf-8")
    out = tmp_path / "results"
    run_suite(cfg, results_dir=out)
    run_suite(cfg, results_dir=out)
    all_rows = json.loads((out / "all_results.json").read_text(encoding="utf-8"))
    assert len(all_rows) == 2
    assert (out / "tiny.json").exists()
