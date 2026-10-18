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

Python, sentence-transformers, FAISS, BM25 (built in, Lucene IDF), the Claude API (optional, `claude-opus-5-5`), a custom judge, MLflow (optional) for experiment tracking, FastAPI for the API, and a **TypeScript web interface** (Vite + React). A minimal Streamlit demo is included too.

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

*(Filled in as experiments run.)*

| Chunking | Embedding | Retrieval | Recall@5 | MRR | Faithfulness | Latency (ms) |
|---|---|---|---|---|---|---|
| | | | | | | |

**Key findings:** *(e.g. "Hybrid retrieval beat dense-only by X points on Recall@5; smaller chunks helped factual questions but hurt multi-sentence answers.")*

## Roadmap

- [x] Minimal demo: chunking × retrieval evaluation
- [ ] Real document set + 100-question eval set
- [ ] Dense embeddings + vector store
- [ ] Hybrid retrieval + reranker
- [ ] Answer generation + faithfulness scoring
- [ ] MLflow tracking + results charts
- [ ] Streamlit demo, deployed on Hugging Face Spaces
- [ ] Write-up: what worked, what didn't, and why
