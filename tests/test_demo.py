from demo import run_demo


def test_run_demo_grid():
    rows = run_demo()
    assert len(rows) == 9
    assert {r["method"] for r in rows} == {"tfidf", "bm25", "hybrid"}
    assert {r["chunk_size"] for r in rows} == {20, 40, 80}
    for r in rows:
        for m in ("recall@1", "recall@3", "mrr"):
            assert 0.0 <= r[m] <= 1.0
    assert max(r["mrr"] for r in rows) > 0.5
