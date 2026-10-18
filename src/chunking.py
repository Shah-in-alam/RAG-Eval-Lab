"""Chunking strategies.

A "token" here is a whitespace-separated word. That keeps chunk sizes
comparable across embedding models (each has its own subword tokenizer) and
needs no extra dependency. English Wikipedia text averages ~1.3 subword tokens
per word, so a 500-word chunk is roughly 650 model tokens.

Every chunk keeps character offsets into its source document, so
`text[chunk.start:chunk.end] == chunk.text`. Evaluation uses these offsets to
decide whether a chunk contains a question's evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

HEADING_RE = re.compile(r"^(=+ .+? =+|#+ .+)\s*$", re.MULTILINE)
PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    start: int
    end: int


def _word_spans(text: str, offset: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    return [(m.start(), m.end()) for m in re.compile(r"\S+").finditer(text, offset, end)]


def _make(doc_id: str, text: str, spans: list[tuple[int, int]], idx: int) -> Chunk:
    start, end = spans[0][0], spans[-1][1]
    return Chunk(id=f"{doc_id}#{idx}", doc_id=doc_id, text=text[start:end], start=start, end=end)


def _windows(spans: list[tuple[int, int]], size: int, overlap: float) -> list[list[tuple[int, int]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(1, size - round(size * overlap))
    out = []
    for i in range(0, len(spans), step):
        out.append(spans[i : i + size])
        if i + size >= len(spans):
            break
    return out


def chunk_fixed(doc_id: str, text: str, size: int, overlap: float = 0.0) -> list[Chunk]:
    """Cut every `size` words; consecutive chunks share `overlap * size` words."""
    windows = _windows(_word_spans(text), size, overlap)
    return [_make(doc_id, text, w, i) for i, w in enumerate(windows) if w]


def _split_sections(text: str, boundaries: list[int], max_tokens: int, doc_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    edges = sorted(set([0, *boundaries, len(text)]))
    for a, b in zip(edges, edges[1:]):
        spans = _word_spans(text, a, b)
        for w in _windows(spans, max_tokens, 0.0) if spans else []:
            chunks.append(_make(doc_id, text, w, len(chunks)))
    return chunks


def chunk_by_heading(doc_id: str, text: str, max_tokens: int = 400) -> list[Chunk]:
    """One chunk per section (heading line + body). Long sections are split at `max_tokens` words."""
    return _split_sections(text, [m.start() for m in HEADING_RE.finditer(text)], max_tokens, doc_id)


def chunk_by_paragraph(doc_id: str, text: str, max_tokens: int = 300) -> list[Chunk]:
    """Split at blank lines, then greedily merge neighbouring paragraphs up to `max_tokens` words."""
    paragraphs: list[list[tuple[int, int]]] = []
    pos = 0
    for m in [*PARAGRAPH_BREAK_RE.finditer(text), None]:
        end = m.start() if m else len(text)
        spans = _word_spans(text, pos, end)
        if spans:
            paragraphs.extend(_windows(spans, max_tokens, 0.0))  # oversized paragraphs get split
        pos = m.end() if m else len(text)

    chunks: list[Chunk] = []
    current: list[tuple[int, int]] = []
    for para in paragraphs:
        if current and len(current) + len(para) > max_tokens:
            chunks.append(_make(doc_id, text, current, len(chunks)))
            current = []
        current = current + para
    if current:
        chunks.append(_make(doc_id, text, current, len(chunks)))
    return chunks


def make_chunker(spec: dict) -> Callable[[str, str], list[Chunk]]:
    """Build a chunker from a config dict, e.g. {"type": "fixed", "size": 500, "overlap": 0.15}."""
    kind = spec["type"]
    if kind == "fixed":
        return lambda doc_id, text: chunk_fixed(doc_id, text, int(spec["size"]), float(spec.get("overlap", 0.0)))
    if kind == "heading":
        return lambda doc_id, text: chunk_by_heading(doc_id, text, int(spec.get("max_tokens", 400)))
    if kind == "paragraph":
        return lambda doc_id, text: chunk_by_paragraph(doc_id, text, int(spec.get("max_tokens", 300)))
    raise ValueError(f"unknown chunker type: {kind!r}")


def chunker_name(spec: dict) -> str:
    if spec["type"] == "fixed":
        return f"fixed-{int(spec['size'])}-o{round(float(spec.get('overlap', 0)) * 100)}"
    return spec["type"]


def chunk_documents(docs: dict[str, str], spec: dict) -> list[Chunk]:
    chunker = make_chunker(spec)
    return [c for doc_id, text in docs.items() for c in chunker(doc_id, text)]
