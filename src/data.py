"""Load the document collection and the labeled evaluation set."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EVAL_PATH = ROOT / "data" / "eval" / "questions.jsonl"


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    answer: str
    evidence: str  # copied verbatim from the document
    doc_id: str
    source: str  # "manual" | "generated"
    type: str  # "factoid" | "explanatory"

    def to_dict(self) -> dict:
        return asdict(self)


def load_documents(path: str | Path = RAW_DIR) -> dict[str, str]:
    """doc_id (file stem) -> full text."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path(path).glob("*.txt"))}


def load_eval(path: str | Path = EVAL_PATH) -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def evidence_span(docs: dict[str, str], item: EvalItem) -> tuple[int, int]:
    """Character span of the item's evidence inside its document."""
    start = docs[item.doc_id].index(item.evidence)
    return start, start + len(item.evidence)


def doc_title(doc_id: str) -> str:
    return doc_id.replace("_", " ").capitalize()
