"""Answer generation from retrieved chunks.

With Claude API credentials, Claude answers from the numbered context and cites it
as [1], [2], ...; otherwise an extractive fallback returns the context sentence that
best matches the question. Every Answer records which generator produced it.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field

from src.retrievers import Hit
from src.textutil import content_tokens, sentences

LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "claude-opus-5-5")
# USD per 1M tokens for claude-opus-5-5 (input, output). Update if you change RAG_LLM_MODEL.
PRICE_PER_1M = {"input": 4.00, "output": 20.00}

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages you are given. "
    "Cite the passages you use inline as [n]. Keep the answer to one to three sentences. "
    "If the passages do not contain the answer, reply exactly: I don't know."
)


@dataclass
class Answer:
    text: str
    citations: list[int]
    generator: str  # "llm" | "extractive"
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def llm_available() -> bool:
    if os.environ.get("RAG_DISABLE_LLM"):
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def llm_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_PER_1M["input"] + output_tokens * PRICE_PER_1M["output"]) / 1_000_000


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] (source: {h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def _extractive(question: str, hits: list[Hit]) -> tuple[str, list[int]]:
    q = set(content_tokens(question))
    best, best_key = None, (0, 0.0)
    for rank, hit in enumerate(hits, 1):
        for sent in sentences(hit.chunk.text):
            overlap = len(q & set(content_tokens(sent)))
            key = (overlap, -rank)  # more overlap first, then higher-ranked chunk
            if overlap and (best is None or key > best_key):
                best, best_key = (sent, rank), key
    if best is None:
        return "I don't know.", []
    return best[0], [best[1]]


def _llm(question: str, hits: list[Hit], model: str) -> Answer:
    import anthropic

    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    response = client.beta.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        # re-run on a fallback model if the request is declined; sent via extra_body
        # because older SDK versions have no `fallbacks` keyword
        extra_body={"fallbacks": "default"},
        messages=[{"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"}],
    )
    latency = (time.perf_counter() - t0) * 1000
    notes = []
    if response.stop_reason == "refusal":
        text = "I don't know."
        notes.append("model declined the request")
    else:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
    cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    return Answer(text, cites, "llm", latency, response.usage.input_tokens, response.usage.output_tokens,
                  model=response.model, notes=notes)


def generate_answer(question: str, hits: list[Hit], model: str | None = None, force_offline: bool = False) -> Answer:
    """Answer `question` from `hits`. Falls back to extractive if no LLM is configured or the call fails."""
    notes: list[str] = []
    if hits and not force_offline and llm_available():
        try:
            return _llm(question, hits, model or LLM_MODEL)
        except Exception as e:  # network/auth/rate errors: degrade gracefully, say why
            notes.append(f"LLM call failed, used extractive fallback: {type(e).__name__}: {e}")
    t0 = time.perf_counter()
    text, cites = _extractive(question, hits)
    return Answer(text, cites, "extractive", (time.perf_counter() - t0) * 1000, notes=notes)
