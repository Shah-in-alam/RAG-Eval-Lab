"""Retrieval metrics.

Relevance is defined by character overlap with the question's evidence span, so it
works the same way for every chunking strategy: a chunk is relevant when it holds
at least half of the evidence, or when at least half of the chunk is evidence
(a small chunk sitting inside a long evidence passage).
"""
from __future__ import annotations

from typing import Sequence

from src.chunking import Chunk


def is_relevant(chunk: Chunk, span: tuple[int, int], doc_id: str, min_cover: float = 0.5) -> bool:
    if chunk.doc_id != doc_id:
        return False
    overlap = min(chunk.end, span[1]) - max(chunk.start, span[0])
    if overlap <= 0:
        return False
    evidence_len = span[1] - span[0]
    chunk_len = chunk.end - chunk.start
    return overlap >= min_cover * evidence_len or overlap >= min_cover * chunk_len


def recall_at_k(ranked_relevance: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant, else 0.0 (one relevant passage per question)."""
    return 1.0 if any(ranked_relevance[:k]) else 0.0


def reciprocal_rank(ranked_relevance: Sequence[bool]) -> float:
    for rank, rel in enumerate(ranked_relevance, 1):
        if rel:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
