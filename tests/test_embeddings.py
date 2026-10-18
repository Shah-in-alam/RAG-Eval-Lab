import numpy as np
import pytest

from src.embeddings import EMBEDDERS, EmbedderUnavailable, cached_embed, get_embedder

DOCS = ["the cat sat on the mat", "stock markets fell sharply today", "a kitten is a young cat"]


def test_tfidf_shapes_and_norm():
    emb = get_embedder("tfidf")
    m = emb.embed_documents(DOCS)
    assert m.shape[0] == 3 and m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    q = emb.embed_query("cat")
    assert q.shape == (m.shape[1],)
    assert int(np.argmax(m @ q)) in (0, 2)


class CountingEmbedder:
    name = "counting"
    cacheable = True
    dim = 2

    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return np.ones((len(texts), 2), dtype=np.float32)


def test_cache_hit_skips_model(tmp_path):
    e = CountingEmbedder()
    a = cached_embed(e, DOCS, cache_dir=tmp_path)
    b = cached_embed(e, DOCS, cache_dir=tmp_path)
    assert e.calls == 1 and np.array_equal(a, b)
    cached_embed(e, DOCS + ["new text"], cache_dir=tmp_path)
    assert e.calls == 2


def test_api_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(EmbedderUnavailable):
        get_embedder("api")


def test_unknown_embedder():
    with pytest.raises(ValueError):
        get_embedder("nope")
    assert {"tfidf", "minilm", "bge-small", "e5-base", "api"} <= set(EMBEDDERS)


@pytest.mark.slow
def test_minilm_similarity():
    emb = get_embedder("minilm")
    m = emb.embed_documents(["kitten", "stock market"])
    q = emb.embed_query("cat")
    assert m[0] @ q > m[1] @ q


class SlowEmbedder(CountingEmbedder):
    name = "slow"

    def embed_documents(self, texts):
        import time
        time.sleep(0.2)
        return super().embed_documents(texts)


def test_cache_remembers_original_embedding_time(tmp_path):
    from src.embeddings import cached_embed_timed
    e = SlowEmbedder()
    _, t1, hit1 = cached_embed_timed(e, DOCS, cache_dir=tmp_path)
    _, t2, hit2 = cached_embed_timed(e, DOCS, cache_dir=tmp_path)
    assert (hit1, hit2) == (False, True)
    assert t1 >= 0.2 and abs(t2 - t1) < 1e-6  # cache hit reports the original cost


def test_cache_without_timing_is_recomputed(tmp_path):
    from src.embeddings import cached_embed_timed
    e = SlowEmbedder()
    cached_embed_timed(e, DOCS, cache_dir=tmp_path)
    for f in tmp_path.glob("*.time"):
        f.unlink()  # simulate a cache file written by an older version
    _, seconds, hit = cached_embed_timed(e, DOCS, cache_dir=tmp_path)
    assert hit is False and seconds >= 0.2 and e.calls == 2


@pytest.mark.slow
def test_concurrent_model_loads_share_one_instance():
    from concurrent.futures import ThreadPoolExecutor
    import src.embeddings as emb
    emb._ST_MODELS.clear()
    embedders = [get_embedder("minilm") for _ in range(4)]
    with ThreadPoolExecutor(4) as pool:
        models = list(pool.map(lambda e: e.model, embedders))
    assert all(m is models[0] for m in models)


def test_server_builds_each_retriever_once_under_concurrency(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time as _t
    import api.server as server
    calls = []

    def fake_build(method, chunks, emb):
        calls.append(method)
        _t.sleep(0.2)
        return object()

    monkeypatch.setattr(server, "build_retriever", fake_build)
    server._RETRIEVERS.clear()
    spec = {"type": "fixed", "size": 300, "overlap": 0}
    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(lambda _: server._retriever(spec, "tfidf", "dense"), range(4)))
    assert len(calls) == 1 and all(o is out[0] for o in out)
