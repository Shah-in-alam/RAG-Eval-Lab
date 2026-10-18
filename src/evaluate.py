"""Experiment runner.

A YAML config lists chunkers, embedders, retrieval methods and k values; `expand_grid`
turns it into RunConfigs. Runs that share a (chunker, embedder) share one index, and
each method searches once with the largest k so every smaller k is scored from the same
ranking. Results go to results/<suite>.json and are merged by run_id into
results/all_results.json.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from src.chunking import chunk_documents, chunker_name
from src.data import EvalItem, evidence_span, load_documents, load_eval
from src.embeddings import approx_tokens, available_embedders, get_embedder
from src.generate import generate_answer, llm_cost_usd
from src.judge import judge_answer
from src.metrics import is_relevant, mean, recall_at_k, reciprocal_rank
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "experiments" / "configs"
SPARSE_ONLY = {"bm25"}


@dataclass(frozen=True)
class RunConfig:
    chunker: dict
    embedder: str
    method: str
    k: int

    @property
    def run_id(self) -> str:
        return f"{chunker_name(self.chunker)}|{self.embedder}|{self.method}|k{self.k}"

    def __hash__(self) -> int:
        return hash(self.run_id)


def expand_grid(config: dict) -> list[RunConfig]:
    runs: dict[str, RunConfig] = {}
    for chunker, embedder, method, k in itertools.product(
            config["chunkers"], config["embedders"], config["methods"], config["ks"]):
        rc = RunConfig(dict(chunker), "none" if method in SPARSE_ONLY else embedder, method, int(k))
        runs.setdefault(rc.run_id, rc)
    return list(runs.values())


def _build(method: str, chunks, cache: dict, embedder_name: str):
    """Build (or reuse) the retriever for `method`; dense and BM25 indexes are shared."""
    if method in ("dense", "hybrid", "hybrid_rerank") and "dense" not in cache:
        cache["dense"] = DenseRetriever(chunks, get_embedder(embedder_name))
    if method in ("bm25", "hybrid", "hybrid_rerank") and "bm25" not in cache:
        cache["bm25"] = BM25Retriever(chunks)
    if method == "dense":
        return cache["dense"]
    if method == "bm25":
        return cache["bm25"]
    hybrid = HybridRetriever(cache["dense"], cache["bm25"])
    return hybrid if method == "hybrid" else RerankRetriever(hybrid)


def run_group(runs: list[RunConfig], docs: dict[str, str], items: list[EvalItem],
              with_answers: bool = False, progress: Callable[[int], None] | None = None) -> list[dict]:
    """Score runs that share one chunker. Returns one result row per RunConfig."""
    chunker = runs[0].chunker
    assert all(r.chunker == chunker for r in runs), "run_group expects a single chunker"
    t0 = time.perf_counter()
    chunks = chunk_documents(docs, chunker)
    chunk_time = time.perf_counter() - t0
    spans = {it.id: evidence_span(docs, it) for it in items}
    rows: list[dict] = []

    by_embedder: dict[str, list[RunConfig]] = {}
    for r in runs:
        by_embedder.setdefault(r.embedder, []).append(r)

    for embedder_name, emb_runs in by_embedder.items():
        cache: dict = {}
        by_method: dict[str, list[RunConfig]] = {}
        for r in emb_runs:
            by_method.setdefault(r.method, []).append(r)
        for method, m_runs in by_method.items():
            retriever = _build(method, chunks, cache, embedder_name if embedder_name != "none" else "tfidf")
            kmax = max(r.k for r in m_runs)
            rankings, latencies = {}, []
            for it in items:
                ts = time.perf_counter()
                hits = retriever.search(it.question, kmax)
                latencies.append((time.perf_counter() - ts) * 1000)
                rankings[it.id] = hits
            emb_price = cache["dense"].embedder.cost_per_1m_tokens if method != "bm25" else 0.0
            for rc in sorted(m_runs, key=lambda r: r.k):
                rows.append(_score(rc, items, rankings, spans, chunks, chunk_time, retriever,
                                   latencies, emb_price, with_answers))
                if progress:
                    progress(1)
    return rows


def _score(rc, items, rankings, spans, chunks, chunk_time, retriever, latencies, emb_price, with_answers) -> dict:
    per_q, gen_lat, faith, corr, llm_cost = [], [], [], [], 0.0
    generator = judge = None
    for it in items:
        hits = rankings[it.id][: rc.k]
        rel = [is_relevant(h.chunk, spans[it.id], it.doc_id) for h in hits]
        first = next((i + 1 for i, r in enumerate(rel) if r), None)
        q = {"id": it.id, "type": it.type, "recall": recall_at_k(rel, rc.k), "rr": reciprocal_rank(rel),
             "first_relevant_rank": first, "hits": [h.chunk.id for h in hits]}
        if with_answers:
            ans = generate_answer(it.question, hits)
            s = judge_answer(it.question, ans.text, it.answer, [h.chunk.text for h in hits])
            gen_lat.append(ans.latency_ms)
            llm_cost += llm_cost_usd(ans.input_tokens, ans.output_tokens)
            generator, judge = ans.generator, s["judge"]
            faith.append(s["faithfulness"])
            corr.append(s["correctness"])
            q.update(answer=ans.text, faithfulness=s["faithfulness"], correctness=s["correctness"])
        per_q.append(q)

    n = len(items)
    by_type = {t: mean([q["recall"] for q in per_q if q["type"] == t]) for t in sorted({it.type for it in items})}
    query_emb_cost = emb_price * approx_tokens([it.question for it in items]) / 1e6
    index_emb_cost = emb_price * approx_tokens([c.text for c in chunks]) / 1e6
    return {
        "run_id": rc.run_id,
        "chunker": chunker_name(rc.chunker),
        "chunker_spec": rc.chunker,
        "embedder": rc.embedder,
        "method": rc.method,
        "k": rc.k,
        "n_questions": n,
        "n_chunks": len(chunks),
        "avg_chunk_words": round(mean([len(c.text.split()) for c in chunks]), 1),
        "recall@k": round(mean([q["recall"] for q in per_q]), 4),
        "mrr": round(mean([q["rr"] for q in per_q]), 4),
        "recall@1": round(mean([1.0 if q["first_relevant_rank"] == 1 else 0.0 for q in per_q]), 4),
        "recall_by_type": {t: round(v, 4) for t, v in by_type.items()},
        "faithfulness": round(mean(faith), 4) if with_answers else None,
        "correctness": round(mean(corr), 4) if with_answers else None,
        "generator": generator,
        "judge": judge,
        "index_time_s": round(chunk_time + retriever.index_time_s, 3),
        "latency_ms": round(mean(latencies) + (mean(gen_lat) if gen_lat else 0.0), 2),
        "retrieval_latency_ms": round(mean(latencies), 2),
        "cost_per_1k_usd": round((query_emb_cost + llm_cost) / n * 1000, 4),
        "index_cost_usd": round(index_emb_cost, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_question": per_q,
    }


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists() and (CONFIG_DIR / f"{p.stem}.yaml").exists():
        p = CONFIG_DIR / f"{p.stem}.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg.setdefault("name", p.stem)
    return cfg


def merge_results(rows: list[dict], path: Path) -> list[dict]:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = {r["run_id"]: r for r in existing}
    merged.update({r["run_id"]: r for r in rows})
    out = list(merged.values())
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _log_mlflow(suite: str, rows: list[dict]) -> None:
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; skipping tracking (pip install mlflow)")
        return
    mlflow.set_experiment(f"rag-eval-lab/{suite}")
    for r in rows:
        with mlflow.start_run(run_name=r["run_id"]):
            mlflow.log_params({"chunker": r["chunker"], "embedder": r["embedder"], "method": r["method"], "k": r["k"]})
            mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})


def run_suite(config_path: str | Path, results_dir: str | Path = RESULTS_DIR,
              progress: Callable[[int, int], None] | None = None) -> list[dict]:
    cfg = load_config(config_path)
    docs, items = load_documents(), load_eval()
    if cfg.get("limit_questions"):
        items = items[: int(cfg["limit_questions"])]
    usable = set(available_embedders())
    skipped = [e for e in cfg["embedders"] if e not in usable]
    if skipped:
        print(f"skipping unavailable embedders: {skipped} (missing API key or package)")
    cfg = {**cfg, "embedders": [e for e in cfg["embedders"] if e in usable]}
    runs = expand_grid(cfg)
    total, done = len(runs), 0

    def tick(n: int) -> None:
        nonlocal done
        done += n
        if progress:
            progress(done, total)

    by_chunker: dict[str, list[RunConfig]] = {}
    for r in runs:
        by_chunker.setdefault(chunker_name(r.chunker), []).append(r)
    rows: list[dict] = []
    for group in by_chunker.values():
        rows.extend(run_group(group, docs, items, bool(cfg.get("with_answers")), tick))

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg['name']}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    merge_results(rows, out / "all_results.json")
    if cfg.get("tracking") == "mlflow":
        _log_mlflow(cfg["name"], rows)
    return rows


def summary_table(rows: list[dict]) -> str:
    head = f"{'chunker':<16}{'embedder':<11}{'method':<15}{'k':>3}{'R@k':>8}{'MRR':>8}{'faith':>8}{'lat ms':>9}"
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: (-r["mrr"], r["run_id"])):
        faith = f"{r['faithfulness']:.3f}" if r["faithfulness"] is not None else "-"
        lines.append(f"{r['chunker']:<16}{r['embedder']:<11}{r['method']:<15}{r['k']:>3}"
                     f"{r['recall@k']:>8.3f}{r['mrr']:>8.3f}{faith:>8}{r['latency_ms']:>9.1f}")
    return "\n".join(lines)


def to_public(row: dict) -> dict:
    """Row without the bulky per-question details (for listings)."""
    return {k: v for k, v in row.items() if k != "per_question"}

