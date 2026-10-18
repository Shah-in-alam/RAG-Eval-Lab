import pytest

from src.chunking import Chunk
from src.embeddings import get_embedder
from src.retrievers import (BM25Retriever, DenseRetriever, Hit, HybridRetriever, METHODS,
                            build_retriever)


def _chunks(texts):
    return [Chunk(f"d{i}#0", f"d{i}", t, 0, len(t)) for i, t in enumerate(texts)]


CHUNKS = _chunks([
    "Okapi BM25 is a ranking function with parameters k1 and b.",
    "Sentence embeddings map text to dense vectors for semantic search.",
    "The HNSW graph index finds approximate nearest neighbours quickly.",
    "Zebras live in Africa and eat grass.",
])


def test_bm25_finds_exact_term():
    hits = BM25Retriever(CHUNKS).search("what are the BM25 parameters", k=2)
    assert hits[0].chunk.doc_id == "d0"
    assert isinstance(hits[0], Hit)


def test_dense_tfidf_finds_doc():
    hits = DenseRetriever(CHUNKS, get_embedder("tfidf")).search("nearest neighbours graph", k=1)
    assert hits[0].chunk.doc_id == "d2"


def test_hybrid_combines():
    dense = DenseRetriever(CHUNKS, get_embedder("tfidf"))
    sparse = BM25Retriever(CHUNKS)
    hybrid = HybridRetriever(dense, sparse, alpha=0.5)
    hits = hybrid.search("zebras grass", k=2)
    assert hits[0].chunk.doc_id == "d3"
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_k_larger_than_corpus():
    for method in ("dense", "bm25", "hybrid"):
        r = build_retriever(method, CHUNKS, get_embedder("tfidf"))
        hits = r.search("ranking", k=50)
        assert len(hits) == len(CHUNKS)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


def test_build_retriever_rejects_unknown():
    assert set(METHODS) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    with pytest.raises(ValueError):
        build_retriever("magic", CHUNKS, get_embedder("tfidf"))


@pytest.mark.slow
def test_reranker_reorders():
    r = build_retriever("hybrid_rerank", CHUNKS, get_embedder("tfidf"))
    hits = r.search("Which animal eats grass in Africa?", k=2)
    assert hits[0].chunk.doc_id == "d3"


def test_bm25_works_on_tiny_collections():
    chunks = _chunks(["Quarterly report: revenue grew twelve percent.",
                      "The launch codename is Bluebird and ships in March."])
    hits = BM25Retriever(chunks).search("What is the launch codename?", k=2)
    assert hits[0].chunk.doc_id == "d1" and hits[0].score > hits[1].score
