"""Run one or more experiment configs and print a leaderboard.

    python scripts/run_experiments.py quick
    python scripts/run_experiments.py experiments/configs/chunking.yaml retrieval
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import run_suite, summary_table  # noqa: E402


def main(args: list[str]) -> None:
    for cfg in args or ["quick"]:
        t0 = time.perf_counter()
        print(f"\n=== {cfg} ===")
        rows = run_suite(cfg, progress=lambda d, t: print(f"  {d}/{t} runs", end="\r", flush=True))
        print(f"\n{summary_table(rows)}\n({len(rows)} runs in {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
