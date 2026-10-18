"""Merge data/eval/parts/*.jsonl into data/eval/questions.jsonl and validate every item.

Each item must have a question, a reference answer, and an `evidence` string that
appears verbatim in data/raw/<doc_id>.txt. Items get stable ids (q001, q002, ...).

`source` records who wrote the item:
  - "generated": drafted by an LLM and checked automatically (evidence is verbatim)
  - "manual":    written or reviewed by a person
Add hand-written items to data/eval/parts/manual.jsonl with "source": "manual".

    python scripts/build_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = ("question", "answer", "evidence", "doc_id")


def build(parts_dir: str = "data/eval/parts", raw_dir: str = "data/raw",
          out_path: str = "data/eval/questions.jsonl") -> list[dict]:
    docs = {p.stem: p.read_text(encoding="utf-8") for p in Path(raw_dir).glob("*.txt")}
    items, errors = [], []
    for part in sorted(Path(parts_dir).glob("*.jsonl")):
        for line_no, line in enumerate(part.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            where = f"{part.name}:{line_no}"
            missing = [k for k in REQUIRED if not item.get(k)]
            if missing:
                errors.append(f"{where}: missing {missing}")
                continue
            if item["doc_id"] not in docs:
                errors.append(f"{where}: unknown doc_id {item['doc_id']!r}")
            elif item["evidence"] not in docs[item["doc_id"]]:
                errors.append(f"{where}: evidence not found verbatim in {item['doc_id']}: {item['evidence'][:60]!r}")
            item.setdefault("source", "manual" if part.stem == "manual" else "generated")
            item.setdefault("type", "factoid")
            items.append(item)
    if errors:
        raise SystemExit("Eval set has errors:\n  " + "\n  ".join(errors))
    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            if item["question"] in seen:
                raise SystemExit(f"duplicate question: {item['question']}")
            seen.add(item["question"])
            row = {"id": f"q{i:03d}", **{k: item[k] for k in ("question", "answer", "evidence", "doc_id", "source", "type")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return items


if __name__ == "__main__":
    items = build(*sys.argv[1:])
    by_doc: dict[str, int] = {}
    for it in items:
        by_doc[it["doc_id"]] = by_doc.get(it["doc_id"], 0) + 1
    print(f"{len(items)} items written to data/eval/questions.jsonl")
    for doc, n in sorted(by_doc.items()):
        print(f"  {doc:32} {n}")
