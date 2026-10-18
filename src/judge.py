"""Answer-quality scoring: faithfulness (is the answer supported by the context?) and
correctness (does it match the reference answer?).

With Claude API credentials an LLM judge scores both on a 0-1 scale using structured
output; otherwise a heuristic is used and labelled `"judge": "heuristic"`:
  faithfulness = share of the answer's content words that appear in the context
  correctness  = token F1 between the answer and the reference answer
"""
from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from src.generate import LLM_MODEL, llm_available
from src.textutil import content_tokens

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}

Retrieved context:
{context}

Reference answer: {reference}

System answer: {answer}

Score two things from 0.0 to 1.0:
- faithfulness: how much of the system answer is supported by the retrieved context (1.0 = every claim is supported; an honest "I don't know" counts as 1.0).
- correctness: how well the system answer matches the reference answer in meaning (1.0 = same facts, 0.0 = wrong or missing).
Give one short sentence of reasoning."""


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def heuristic_scores(answer: str, reference: str, contexts: list[str]) -> dict:
    ans = content_tokens(answer)
    ctx = set(content_tokens(" ".join(contexts)))
    if answer.strip().lower().startswith("i don't know"):
        faith = 1.0
    else:
        faith = sum(t in ctx for t in ans) / len(ans) if ans else 0.0
    return {"faithfulness": round(faith, 4), "correctness": round(_f1(ans, content_tokens(reference)), 4),
            "judge": "heuristic"}


def _llm_scores(question: str, answer: str, reference: str, contexts: list[str], model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    response = client.messages.parse(
        model=model,
        max_tokens=2048,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, context=context, reference=reference, answer=answer)}],
        output_format=JudgeScores,
    )
    s = response.parsed_output
    return {"faithfulness": s.faithfulness, "correctness": s.correctness, "judge": "llm",
            "reasoning": s.reasoning, "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens}


def judge_answer(question: str, answer: str, reference: str, contexts: list[str],
                 force_offline: bool = False, model: str | None = None) -> dict:
    if not force_offline and llm_available():
        try:
            return _llm_scores(question, answer, reference, contexts, model or LLM_MODEL)
        except Exception as e:
            out = heuristic_scores(answer, reference, contexts)
            out["note"] = f"LLM judge failed, used heuristic: {type(e).__name__}"
            return out
    return heuristic_scores(answer, reference, contexts)
