"""Retrieval methods: dense (vector), sparse (BM25), hybrid (weighted fusion), hybrid + reranker.

Every retriever exposes `search(query, k) -> list[Hit]` sorted by score (best first).
Dense and BM25 also expose `scores(query)` over the whole collection, which the hybrid
retriever fuses after min-max normalisation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.chunking import Chunk
from src.embeddings import _MODEL_LOCK, cached_embed_timed

METHODS = ("dense", "bm25", "hybrid", "hybrid_rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    index_time_s: float

    def search(self, query: str, k: int) -> list[Hit]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    k = min(k, len(chunks))
    idx = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i])) for i in idx]


def _minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min()) if len(x) else 0.0
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


class DenseRetriever:
    """Cosine similarity over normalised embeddings, stored in a FAISS inner-product index."""

    def __init__(self, chunks: list[Chunk], embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        matrix, embed_s, self.cache_hit = cached_embed_timed(embedder, [c.text for c in chunks])
        self.matrix = matrix.astype(np.float32)
        t0 = time.perf_counter() - embed_s  # index time = original embedding time + index build
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)
        except ImportError:  # numpy fallback gives identical results
            self.index = None
        self.index_time_s = time.perf_counter() - t0

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.embed_query(query)

    def search(self, query: str, k: int) -> list[Hit]:
        k = min(k, len(self.chunks))
        if self.index is None:
            return _top_k(self.chunks, self.scores(query), k)
        q = self.embedder.embed_query(query).astype(np.float32)[None, :]
        scores, idx = self.index.search(q, k)
        return [Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


class BM25Retriever:
    """Okapi BM25 (k1=1.5, b=0.75) with the Lucene IDF, log(1 + (N - n + 0.5) / (n + 0.5)).

    That IDF is always positive, so it still ranks sensibly on tiny collections
    (rank-bm25's Okapi IDF is 0 or negative for terms in half or more of the chunks).
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        t0 = time.perf_counter()
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        lengths = []
        for i, c in enumerate(chunks):
            toks = tokenize(c.text)
            lengths.append(len(toks))
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                self.postings.setdefault(t, []).append((i, tf))
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.avgdl = float(self.lengths.mean()) if len(chunks) else 0.0
        n = len(chunks)
        self.idf = {t: float(np.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))) for t, p in self.postings.items()}
        self.index_time_s = time.perf_counter() - t0

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1))
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[i])
        return scores

    def search(self, query: str, k: int) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


class HybridRetriever:
    """score = alpha * minmax(dense) + (1 - alpha) * minmax(bm25), over the whole collection."""

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever, alpha: float = 0.5) -> None:
        self.dense, self.sparse, self.alpha = dense, sparse, alpha
        self.chunks = dense.chunks
        self.index_time_s = dense.index_time_s + sparse.index_time_s

    def scores(self, query: str) -> np.ndarray:
        return self.alpha * _minmax(self.dense.scores(query)) + (1 - self.alpha) * _minmax(self.sparse.scores(query))

    def search(self, query: str, k: int) -> list[Hit]:
        return _top_k(self.chunks, self.scores(query), k)


_CROSS_ENCODERS: dict[str, object] = {}


class RerankRetriever:
    """Takes the top `pool` hits from a base retriever and reorders them with a cross-encoder."""

    def __init__(self, base, model: str = RERANK_MODEL, pool: int = 20) -> None:
        self.base, self.model_id, self.pool = base, model, pool
        self.index_time_s = base.index_time_s

    @property
    def model(self):
        with _MODEL_LOCK:
            if self.model_id not in _CROSS_ENCODERS:
                from sentence_transformers import CrossEncoder

                _CROSS_ENCODERS[self.model_id] = CrossEncoder(self.model_id, device="cpu")
            return _CROSS_ENCODERS[self.model_id]

    def search(self, query: str, k: int) -> list[Hit]:
        candidates = self.base.search(query, max(k, self.pool))
        if not candidates:
            return []
        logits = self.model.predict([(query, h.chunk.text) for h in candidates], show_progress_bar=False)
        order = np.argsort(-np.asarray(logits), kind="stable")[:k]
        return [Hit(candidates[i].chunk, float(logits[i])) for i in order]


def build_retriever(method: str, chunks: list[Chunk], embedder, alpha: float = 0.5) -> Retriever:
    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "dense":
        return DenseRetriever(chunks, embedder)
    if method in ("hybrid", "hybrid_rerank"):
        hybrid = HybridRetriever(DenseRetriever(chunks, embedder), BM25Retriever(chunks), alpha)
        return hybrid if method == "hybrid" else RerankRetriever(hybrid)
    raise ValueError(f"unknown retrieval method {method!r}; choose from {METHODS}")
