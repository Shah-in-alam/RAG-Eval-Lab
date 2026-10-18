# RAG Eval Lab

**What actually makes retrieval-augmented generation work?** This project answers that with experiments, not guesses.

RAG Eval Lab is a question-answering system over a real document collection, built around a reproducible evaluation pipeline. Instead of picking a chunk size or embedding model by intuition, every design choice is tested against a labeled question set, and the results decide the final configuration.

## Key question

> For a given document set, which combination of **chunking strategy**, **embedding model** and **retrieval method** gives the most accurate answers, and at what cost?

## Experiment grid

| Dimension | Variants tested |
|---|---|
| Chunking | Fixed size (200 / 500 / 1000 tokens), overlap (0 / 15%), semantic (by heading / paragraph) |
| Embeddings | `all-MiniLM-L6-v2`, `bge-small-en`, `e5-base`, one API embedding model |
| Retrieval | Dense (vector), sparse (BM25), hybrid (weighted fusion), hybrid + reranker |
| Top-k | 3, 5, 10 |

## Metrics

**Retrieval quality**
- Recall@k: is the chunk containing the answer in the top k results?
- MRR (mean reciprocal rank): how high does the right chunk rank?

**Answer quality**
- Faithfulness: is the answer supported by the retrieved context?
- Answer correctness: compared against the reference answer (LLM-as-judge, spot-checked by hand)

**Cost and speed**
- Indexing time, query latency, and cost per 1,000 queries

## Dataset

- **Documents:** 14 English Wikipedia articles on information retrieval and NLP (BM25, tf–idf, inverted index, word and sentence embeddings, Transformer, BERT, vector databases, nearest-neighbour search, RAG, LLMs, precision/recall, cosine similarity): about 36,000 words, CC BY-SA 4.0. Each article is pinned to a revision in `data/raw/SOURCES.md`.
- **Evaluation set:** 114 question / answer / evidence triples (70 factoid, 44 explanatory). Every evidence quote is checked to appear verbatim in its document. **Currently all 114 were drafted by an LLM** (`source: "generated"`); the planned hand-written half goes in `data/eval/parts/manual.jsonl`.
- **Relevance labels:** a retrieved chunk counts as relevant when it covers at least half of the evidence span (or at least half of the chunk is evidence). Because labels are character spans rather than chunk IDs, the same eval set scores every chunking strategy fairly.

## Architecture

```
documents ──► chunker ──► embedder ──► vector store (ChromaDB / FAISS)
                                            │
question ──► retriever (dense / BM25 / hybrid) ──► reranker ──► LLM ──► answer + citations
                                            │
                     evaluation runner ──► metrics ──► results table + charts
```

## Tech stack

Python, sentence-transformers, FAISS, BM25 (built in, Lucene IDF), the Claude API (optional, `claude-opus-5-5`), a custom judge, MLflow (optional) for experiment tracking, FastAPI for the API, and a **TypeScript web interface** (Vite, React, Tailwind CSS, shadcn/ui, Recharts). A minimal Streamlit demo is included too.

## Project structure

```
RAG Eval Lab/
├── data/
│   ├── raw/                    # 14 Wikipedia articles (+ SOURCES.md with revision links)
│   ├── uploads/                # your uploaded documents (extracted text + metadata)
│   └── eval/
│       ├── parts/*.jsonl       # question sources; add hand-written items to parts/manual.jsonl
│       └── questions.jsonl     # built + validated by scripts/build_eval.py
├── src/
│   ├── chunking.py             # fixed (size, overlap), by heading, by paragraph
│   ├── embeddings.py           # tfidf, MiniLM, bge-small, e5-base, Voyage API; disk cache
│   ├── retrievers.py           # dense (FAISS), BM25, hybrid fusion, cross-encoder reranker
│   ├── generate.py             # Claude answer with citations, or extractive fallback
│   ├── judge.py                # faithfulness + correctness (LLM judge or heuristic)
│   ├── metrics.py              # evidence-overlap relevance, Recall@k, MRR
│   ├── data.py                 # document + eval-set loading
│   ├── ingest.py               # file/link import: PDF, Office, HTML, EPUB, …; cloud share links
│   └── evaluate.py             # grid expansion, experiment runner, result storage
├── experiments/configs/        # quick, chunking, embeddings, retrieval, full
├── api/server.py               # FastAPI backend for the web UI
├── web/                        # TypeScript web interface (Vite + React)
├── app/streamlit_app.py        # minimal Streamlit demo
├── scripts/                    # fetch_corpus, build_eval, run_experiments, report, start.*
├── results/                    # experiment outputs (all_results.json + one file per suite)
├── docs/                       # RESULTS.md (generated), WRITEUP.md, implementation plan
├── tests/                      # pytest suite
├── demo.py                     # minimal, dependency-light version of the pipeline
└── README.md
```

## Quickstart

**Minimal demo** (no models, a few seconds):

```bash
pip install numpy scikit-learn
python demo.py
```

`demo.py` runs a small self-contained version of the experiment: three chunk sizes × three retrieval methods, scored with Recall@1, Recall@3 and MRR on a built-in mini corpus.

**Full project:**

```bash
pip install -r requirements.txt
cd web && npm install && npm run build && cd ..     # build the web interface once

python scripts/run_experiments.py quick            # ~15 s; also: chunking, embeddings, retrieval, full
python -m uvicorn api.server:app --port 8000       # API + built web UI
```

Open **http://localhost:8000**. The web interface has three pages:

- **Ask**: type a question (or pick one from the eval set), choose chunking, embedding, retrieval method and top-k, and see the answer, the retrieved passages with scores, and whether the evidence was found.
- **Experiments**: a leaderboard of all runs with filters, a ranked bar chart and a metric-vs-k chart, per-question details for any run, and a button to run a suite from the browser.
- **Documents**: add your own documents, browse everything that can be searched, and see the evaluation questions (clicking one highlights its evidence in the text).

**Your own documents.** On the Documents page, drag and drop files or browse for them (several at once, up to 25 MB each), or paste a link:

- File types: PDF, Word (.docx), OpenDocument (.odt), PowerPoint (.pptx), Excel (.xlsx), CSV/TSV, TXT, Markdown, HTML, RTF, EPUB, JSON. Headings, slides, sheets and PDF pages become sections, so the heading chunker works on them.
- Links: any direct file or web-page URL, plus Google Drive, Docs, Sheets and Slides, OneDrive, SharePoint and Dropbox share links set to "Anyone with the link" (converted to direct downloads automatically). Files in synced cloud folders on your computer can also be picked with the file browser.
- On the Ask page, **Search in** chooses All documents, My documents, or the Wikipedia set. Uploads are stored as extracted text in `data/uploads/` (git-ignored) and can be deleted from the table.
- Not supported: scanned PDFs without a text layer (no OCR), legacy .doc/.ppt/.xls (save as the newer format), and private cloud files that need a sign-in. Links to local or private network addresses are refused.
- Experiments still score on the labelled Wikipedia set, because uploads have no answer key.

For frontend development with hot reload, run `npm run dev` in `web/` (http://localhost:5173, proxies `/api` to port 8000), or use `scripts/start.ps1` / `scripts/start.sh` to start both.

**Optional keys:**

- `ANTHROPIC_API_KEY`: generated answers with citations, plus the LLM judge (otherwise: extractive answers + heuristic judge, clearly labelled).
- `VOYAGE_API_KEY`: the API embedding model (`api` in the configs; skipped otherwise).

**Rebuild the data:** `python scripts/fetch_corpus.py` (downloads the articles) and `python scripts/build_eval.py` (validates that every evidence quote appears verbatim in its document).

**Tests:** `python -m pytest` (fast) or `python -m pytest -m ""` (also loads the real models); `npm test` in `web/`.

## Results

114 questions, 14 documents, everything run locally on CPU (no API cost). Best configuration of each retrieval method at k = 5; full tables in [`docs/RESULTS.md`](docs/RESULTS.md), discussion in [`docs/WRITEUP.md`](docs/WRITEUP.md).

| Chunking | Embedding | Retrieval | Recall@5 | MRR | Recall@1 | Latency (ms) |
|---|---|---|---|---|---|---|
| fixed 200 words, 15% overlap | bge-small | **Hybrid** | **0.991** | **0.919** | **0.868** | 44 |
| fixed 1000 words, no overlap | — | BM25 | 0.991 | 0.906 | 0.842 | 9 |
| fixed 200 words, 15% overlap | bge-small | Hybrid + rerank | 0.965 | 0.901 | 0.851 | 2,298 |
| paragraph | bge-small | Dense | 0.912 | 0.789 | 0.702 | 47 |

**Chosen configuration:** 200-word chunks with 15% overlap, `bge-small` embeddings, hybrid retrieval (0.5 · dense + 0.5 · BM25), top-k = 5.

**Key findings:**

- **Hybrid retrieval was best.** It beat dense-only retrieval by +0.13 MRR (0.919 vs 0.789) and +8 points of Recall@5, and edged out BM25 alone (0.906). Dense and BM25 fail on different questions, so fusing them recovers both.
- **This eval set favours keyword matching.** BM25 alone came within 0.013 MRR of the best system, and plain TF-IDF beat every neural model when used alone (dense MRR 0.852 vs 0.774 for bge-small). The questions were written from the evidence passages and reuse their words. Paraphrased, hand-written questions are the most important next step.
- **Overlap helps small chunks.** At 200 words, 15% overlap raised hybrid MRR from 0.873 to 0.910 and BM25 MRR from 0.859 to 0.903, because facts cut at a chunk boundary are kept whole in one chunk. At 500 and 1000 words the effect was small.
- **Small chunks suit factual questions, big chunks suit explanations.** With 200-word chunks plus overlap, factoid Recall@5 was 1.000; explanatory questions reached 1.000 only with 500- or 1000-word chunks. The bigger chunks send 2–4× more text to the answer model per query.
- **bge-small was the best embedding model.** Dense MRR was 0.774, against 0.704 for MiniLM and 0.718 for e5-base. e5-base was 3× slower to index and query with no gain. Inside hybrid retrieval the gap shrinks to about 0.01.
- **The reranker didn't pay off.** The ms-marco MiniLM cross-encoder lowered MRR by 0.018 and made queries about 50× slower (2.3 s vs 44 ms on CPU). It was trained on web search queries, and the first-stage ranking was already strong.
- **Top-k:** Recall rises from 0.974 at k = 3 to 1.000 at k = 10, while MRR barely moves, so k = 5 is the balance point.
- **Answer quality is not measured properly yet.** No Claude API key was configured, so answers are extracted sentences scored with a heuristic (token-F1 correctness about 0.34, faithfulness 1.0 by construction). Set `ANTHROPIC_API_KEY` and re-run to get real generation and LLM-judge scores.

## Roadmap

- [x] Minimal demo: chunking × retrieval evaluation
- [x] Real document set + 100-question eval set (114 questions; hand-written half still to add)
- [x] Dense embeddings + vector store (FAISS)
- [x] Hybrid retrieval + reranker
- [x] Answer generation + faithfulness scoring (Claude when a key is set; offline fallback otherwise)
- [x] Results charts (web UI) and optional MLflow tracking (`tracking: mlflow` in a config)
- [x] TypeScript web interface (React, Tailwind CSS, shadcn/ui) + minimal Streamlit demo
- [x] Upload your own documents (files and cloud share links)
- [x] Write-up: what worked, what didn't, and why ([`docs/WRITEUP.md`](docs/WRITEUP.md))
- [ ] Hand-written, paraphrased questions to reduce keyword bias
- [ ] Re-run answer metrics with an LLM generator and judge
- [ ] Deploy the demo (e.g. Hugging Face Spaces)
