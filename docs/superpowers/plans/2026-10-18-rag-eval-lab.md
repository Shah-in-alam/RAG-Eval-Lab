# RAG Eval Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible RAG evaluation pipeline (chunking × embeddings × retrieval × top-k) over a real document set, plus a TypeScript web interface where a user can ask questions and inspect experiment results.

**Architecture:** A Python package `src/` holds small, single-purpose modules (chunking, embeddings, retrievers, generate, metrics, evaluate). An experiment runner expands a YAML grid, scores every combination on a labeled eval set, and writes JSON/CSV results. A FastAPI server (`api/server.py`) exposes the pipeline and results; a Vite + React + TypeScript app (`web/`) is the user-facing interface.

**Tech Stack:** Python 3.13, numpy, scikit-learn, sentence-transformers, faiss-cpu, rank-bm25, anthropic SDK (optional), PyYAML, FastAPI + uvicorn, pytest; Node 20, Vite, React 18, TypeScript, Vitest. MLflow and Streamlit are optional extras.

**Spec:** `README.md`

## Global Constraints

- `demo.py` must run with only `numpy` and `scikit-learn` installed: "three chunk sizes × three retrieval methods, scored with Recall@1, Recall@3 and MRR on a built-in mini corpus".
- Chunking variants: fixed size 200 / 500 / 1000 tokens, overlap 0 / 15%, semantic by heading / paragraph.
- Embeddings: `all-MiniLM-L6-v2`, `bge-small-en` (`BAAI/bge-small-en-v1.5`), `e5-base` (`intfloat/e5-base-v2`), one API embedding model (optional, skipped without a key).
- Retrieval: dense (vector), sparse (BM25), hybrid (weighted fusion), hybrid + reranker. Top-k: 3, 5, 10.
- Metrics: Recall@k, MRR, faithfulness, answer correctness, indexing time, query latency, cost per 1,000 queries.
- Eval set: ~100 question / answer / evidence triples in `data/eval/questions.jsonl`; every `evidence` string must appear verbatim in its document.
- No LLM API key is configured on this machine: every LLM-dependent step (generation, judge, API embeddings) needs a working offline fallback and must label its output as such (`"judge": "heuristic"`, `"generator": "extractive"`).
- Node is v20: pin Vite 5.x (Vite 6+/7 may require newer Node features).
- Heavy optional imports (sentence-transformers, faiss, anthropic, mlflow) are imported lazily inside functions so the core package imports fast.

## Review Focus

- A question whose evidence spans a chunk boundary: relevance must count a chunk that covers most of the evidence, not require the full span (test in Task 3).
- Empty or whitespace-only query sent to the API / UI: return HTTP 422 with a message, not a crash (test in Task 8).
- Requesting k larger than the number of chunks: retrievers return all chunks, no index error (test in Task 5).
- Re-running an experiment with the same config: embeddings come from the disk cache, results overwrite cleanly (test in Task 4 / Task 7).
- Web UI opened before any experiment has run: results page shows an empty state with a "run" action instead of a blank table (test in Task 9).

---

### Task 1: Scaffold + self-contained `demo.py`

**Files:**
- Create: `requirements.txt`, `pyproject.toml` (pytest config, `pythonpath = ["."]`), `.gitignore`, `src/__init__.py`, `demo.py`
- Test: `tests/test_demo.py`

**Interfaces:**
- Produces: `demo.run_demo() -> list[dict]` — one row per (chunk_size, method) with keys `chunk_size`, `method`, `recall@1`, `recall@3`, `mrr`.

- [ ] **Step 1: Write the failing test** — `test_run_demo_grid`: `rows = run_demo()`; `len(rows) == 9`; methods == `{"tfidf", "bm25", "hybrid"}`; chunk sizes == `{20, 40, 80}` (words — the mini corpus is small); all metrics in `[0, 1]`; best `mrr` > 0.5.
- [ ] **Step 2: Run** `pytest tests/test_demo.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement `demo.py`** — built-in corpus of ~6 short documents and ~12 questions with evidence strings; inline word chunker, TF-IDF cosine (sklearn), BM25 (numpy, k1=1.5, b=0.75), hybrid = 0.5·minmax(tfidf) + 0.5·minmax(bm25); prints a table when run as `__main__`.
- [ ] **Step 4: Run** `pytest tests/test_demo.py -v` and `python demo.py` → PASS, table printed.

### Task 2: Chunking module

**Files:**
- Create: `src/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces: `@dataclass Chunk(id: str, doc_id: str, text: str, start: int, end: int)` (char offsets into the source document); `chunk_fixed(doc_id: str, text: str, size: int, overlap: float = 0.0) -> list[Chunk]`; `chunk_by_heading(doc_id, text) -> list[Chunk]`; `chunk_by_paragraph(doc_id, text, max_tokens: int = 300) -> list[Chunk]`; `make_chunker(spec: dict) -> Callable[[str, str], list[Chunk]]` where spec is `{"type": "fixed", "size": 500, "overlap": 0.15}` / `{"type": "heading"}` / `{"type": "paragraph"}`; `chunker_name(spec) -> str` (e.g. `fixed-500-o15`).

Tokens = whitespace-separated words (documented in docstring). Headings are lines matching `^=+ .+ =+$` (Wikipedia plaintext) or `^#+ ` (Markdown).

- [ ] **Step 1: Tests** — `test_fixed_sizes_and_offsets` (each chunk ≤ size words; `text[c.start:c.end] == c.text`); `test_overlap_repeats_words` (overlap 0.15, size 100 → consecutive chunks share 15 words); `test_heading_split` (3 headings → 3+ chunks, each chunk text starts at a heading or doc start); `test_paragraph_split_merges_short_paragraphs`; `test_empty_text_returns_no_chunks`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/test_chunking.py -v` → PASS.

### Task 3: Dataset + metrics

**Files:**
- Create: `scripts/fetch_corpus.py`, `data/raw/*.txt`, `data/eval/questions.jsonl`, `src/data.py`, `src/metrics.py`
- Test: `tests/test_data.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `load_documents(path="data/raw") -> dict[str, str]` (doc_id = file stem); `@dataclass EvalItem(id, question, answer, evidence, doc_id, source)` with `source ∈ {"manual", "generated"}`; `load_eval(path="data/eval/questions.jsonl") -> list[EvalItem]`; `evidence_span(docs, item) -> tuple[int, int]`; `is_relevant(chunk: Chunk, span: tuple[int,int], doc_id: str, min_cover: float = 0.5) -> bool` (chunk covers ≥ 50% of the evidence chars, or contains the whole chunk inside evidence); `recall_at_k(ranked_relevance: list[bool], k) -> float`; `reciprocal_rank(ranked_relevance) -> float`.

Corpus: ~14 English Wikipedia articles on information retrieval / NLP (Information retrieval, Okapi BM25, tf–idf, Inverted index, Word embedding, Sentence embedding, Transformer (deep learning architecture), BERT (language model), Vector database, Nearest neighbor search, Retrieval-augmented generation, Large language model, Precision and recall, Cosine similarity), fetched as plaintext with section headings via the MediaWiki `extracts` API (CC BY-SA, attribution in `data/raw/SOURCES.md`).

- [ ] **Step 1: Tests** — `test_every_evidence_is_verbatim` (for all items, `evidence in docs[doc_id]`); `test_eval_set_size` (≥ 90 items, both sources present, unique ids); `test_recall_and_mrr` (`[F,T,F]` → R@1=0, R@3=1, RR=0.5; all False → 0); `test_relevance_boundary` (evidence split 60/40 across two chunks → first relevant, second not).
- [ ] **Step 2: Run** → FAIL. **Step 3:** Write fetch script, run it, write the eval set (manual items written from reading the documents; "generated" items are drafted by the model and each checked against the source by `scripts/validate_eval.py`), implement `data.py` + `metrics.py`. **Step 4: Run** `pytest tests/test_data.py tests/test_metrics.py -v` → PASS.

### Task 4: Embeddings with disk cache

**Files:**
- Create: `src/embeddings.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Produces: `class Embedder(Protocol): name: str; dim: int; def embed_documents(texts: list[str]) -> np.ndarray; def embed_query(text: str) -> np.ndarray` (L2-normalised float32 rows); `get_embedder(name: str) -> Embedder` with names `tfidf`, `minilm` (`sentence-transformers/all-MiniLM-L6-v2`), `bge-small` (`BAAI/bge-small-en-v1.5`, query prefix `"Represent this sentence for searching relevant passages: "`), `e5-base` (`intfloat/e5-base-v2`, prefixes `"query: "` / `"passage: "`), `api` (Voyage `voyage-3.5-lite` if `VOYAGE_API_KEY` set, else raises `EmbedderUnavailable`); `cached_embed(embedder, texts, cache_dir=".cache/emb") -> np.ndarray` keyed by sha1(name + texts).

- [ ] **Step 1: Tests** — `test_tfidf_shapes_and_norm`; `test_cache_hit_skips_model` (counting fake embedder called once across two `cached_embed` calls); `test_api_unavailable_without_key`; `test_minilm_similarity` (marked `slow`; "cat" closer to "kitten" than to "stock market").
- [ ] **Step 2–4:** FAIL → implement → `pytest tests/test_embeddings.py -v` PASS.

### Task 5: Retrievers + reranker

**Files:**
- Create: `src/retrievers.py`
- Test: `tests/test_retrievers.py`

**Interfaces:**
- Consumes: `Chunk`, `Embedder`, `cached_embed`.
- Produces: `@dataclass Hit(chunk: Chunk, score: float)`; `class Retriever(Protocol): def search(query: str, k: int) -> list[Hit]`; `DenseRetriever(chunks, embedder)` (faiss `IndexFlatIP`); `BM25Retriever(chunks)` (rank-bm25 `BM25Okapi`, lowercase word tokens); `HybridRetriever(dense, sparse, alpha=0.5, pool=50)` (min-max normalise each list over the pooled candidates, score = α·dense + (1−α)·sparse); `RerankRetriever(base, model="cross-encoder/ms-marco-MiniLM-L-6-v2", pool=20)`; `build_retriever(method: str, chunks, embedder) -> Retriever` for `dense | bm25 | hybrid | hybrid_rerank`.

- [ ] **Step 1: Tests** — `test_bm25_finds_exact_term`; `test_dense_tfidf_finds_doc`; `test_hybrid_combines` (a doc found only by one method still appears in hybrid top-k); `test_k_larger_than_corpus` (k=50 on 3 chunks → 3 hits, sorted desc); `test_reranker_reorders` (`slow`).
- [ ] **Step 2–4:** FAIL → implement → PASS.

### Task 6: Answer generation + answer-quality judge

**Files:**
- Create: `src/generate.py`, `src/judge.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `Hit`.
- Produces: `@dataclass Answer(text: str, citations: list[int], generator: str, latency_ms: float, input_tokens: int, output_tokens: int)`; `generate_answer(question, hits, model=None) -> Answer` — Claude via `anthropic` when `ANTHROPIC_API_KEY` is set (prompt numbers the context `[1]..[n]`, asks for cited answer or "I don't know"), else extractive: best-overlapping sentence(s) from hits, `generator="extractive"`; `judge_answer(question, answer, reference, contexts) -> dict(faithfulness: float, correctness: float, judge: "llm"|"heuristic")` — heuristic faithfulness = share of answer content-words present in context, correctness = token F1 vs reference.

- [ ] **Step 1: Tests** — `test_extractive_answer_cites_source`; `test_heuristic_judge_scores` (answer copied from context → faithfulness 1.0; unrelated answer → < 0.3; F1 exact match → 1.0); `test_llm_path_selected_only_with_key` (monkeypatch env).
- [ ] **Step 2–4:** FAIL → implement → PASS.

### Task 7: Experiment runner + configs

**Files:**
- Create: `src/evaluate.py`, `experiments/configs/{quick,chunking,embeddings,retrieval,full}.yaml`, `scripts/run_experiments.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `expand_grid(config: dict) -> list[RunConfig]` where `RunConfig(chunker: dict, embedder: str, method: str, k: int)`; `run_experiment(rc, docs, items, with_answers=False) -> dict` with keys `run_id, chunker, embedder, method, k, n_chunks, recall@k, mrr, faithfulness, correctness, index_time_s, latency_ms, cost_per_1k_usd, per_question: list`; `run_suite(config_path) -> list[dict]` writing `results/<suite>.json` and appending/overwriting rows (by `run_id`) in `results/all_results.json`; optional MLflow logging when `mlflow` is importable and `tracking: mlflow` is in the config. Index (chunks + embeddings) is built once per (chunker, embedder) and reused across methods and k. Cost per 1k = embedding API cost (0 for local) + LLM token cost (0 for extractive).

- [ ] **Step 1: Tests** — `test_expand_grid_counts` (2 chunkers × 1 embedder × 3 methods × 2 k → 12); `test_run_experiment_tiny` (tfidf, 3 docs, 4 items → recall ∈ [0,1], `per_question` length 4); `test_rerun_overwrites_by_run_id`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** run `python scripts/run_experiments.py experiments/configs/quick.yaml` and then the larger configs; check `results/all_results.json`.

### Task 8: FastAPI server

**Files:**
- Create: `api/__init__.py`, `api/server.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces (all JSON, CORS for `http://localhost:5173`):
  - `GET /api/health` → `{status, llm_available, embedders}`
  - `GET /api/options` → `{chunkers: [spec], embedders: [str], methods: [str], ks: [int]}`
  - `GET /api/documents` → `[{doc_id, title, n_chars}]`; `GET /api/documents/{doc_id}` → `{doc_id, text}`
  - `GET /api/eval` → `[EvalItem]`
  - `POST /api/query` body `{question, chunker, embedder, method, k, generate: bool}` → `{hits: [{chunk_id, doc_id, text, score, start, end}], answer?: Answer, timings}`; empty question → 422.
  - `GET /api/results` → rows of `all_results.json` without `per_question`; `GET /api/results/{run_id}` → full row.
  - `POST /api/experiments/run` body `{config: "quick"}` → starts the suite in a background thread; `GET /api/experiments/status` → `{running, suite, done, total, error}`.
- Retriever instances cached in an LRU keyed by (chunker, embedder, method).

- [ ] **Step 1: Tests** (FastAPI `TestClient`, tfidf only) — health, options, documents, query returns k hits, empty question → 422, results empty → `[]`.
- [ ] **Step 2–4:** FAIL → implement → PASS.

### Task 9: TypeScript web interface

**Files:**
- Create: `web/` (Vite 5 + React 18 + TS): `src/api.ts`, `src/types.ts`, `src/App.tsx`, `src/pages/{Ask,Results,Dataset}.tsx`, `src/components/{BarChart,ResultsTable,HitCard}.tsx`, `src/lib/{sort,format}.ts`, `src/styles.css`; Vite dev proxy `/api → http://127.0.0.1:8000`.
- Test: `web/src/lib/*.test.ts` (Vitest), `npm run build` (tsc type-check).

**Interfaces:**
- Consumes: Task 8 endpoints; `types.ts` mirrors their shapes.
- Pages: **Ask** — question box, pickers for chunker / embedder / method / k, "generate answer" toggle; shows answer with citations and ranked chunk cards (score, doc, highlighted text). **Results** — leaderboard table (sortable by any metric, filter by method/embedder/chunker), bar chart of the selected metric per configuration, per-question drill-down, "run suite" button with progress, empty state when there are no results. **Dataset** — documents list and eval questions with evidence.

- [ ] **Step 1: Tests** — `sortRows` sorts numerically desc/asc and puts missing values last; `formatMetric` (0.8123 → "0.812", null → "—"); `groupBest` returns the best row per method.
- [ ] **Step 2–4:** FAIL → implement → `npm test` PASS, `npm run build` PASS. **Step 5:** run API + `npm run dev`, open the UI in a browser, ask a question and view results.

### Task 10: Results write-up + Streamlit demo + docs

**Files:**
- Modify: `README.md` (results table, key findings, quickstart for API + web UI, roadmap checkboxes)
- Create: `app/streamlit_app.py` (minimal ask + results view reusing `src/`), `docs/WRITEUP.md`, `scripts/start.ps1` / `scripts/start.sh` (launch API + web)

- [ ] **Step 1:** Fill the README results table from `results/all_results.json` (best config per method + chunking + embedding comparisons) and write findings backed by the numbers.
- [ ] **Step 2:** `python -m pytest -q` (all fast tests) and `npm --prefix web run build` both pass.
- [ ] **Step 3:** `streamlit run app/streamlit_app.py` starts without error.
