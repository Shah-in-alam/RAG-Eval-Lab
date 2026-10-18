# What makes retrieval work here: write-up

## Setup

- **Documents:** 14 English Wikipedia articles on retrieval and NLP, about 36,000 words after cleaning (references and rendered maths removed, LaTeX kept inline).
- **Questions:** 114 question / reference answer / evidence triples (70 factoid, 44 explanatory). Every evidence quote is checked to appear verbatim in its article.
- **Relevance:** a chunk counts as relevant when it holds at least half of the evidence span, or when at least half of the chunk is evidence. Labels are character spans, not chunk IDs, so all chunking strategies are judged against the same answer key.
- **Grid:** 8 chunkers (fixed 200/500/1000 words × 0/15% overlap, by heading, by paragraph) × 4 local embedding models (TF-IDF, all-MiniLM-L6-v2, bge-small-en-v1.5, e5-base-v2) × 4 retrieval methods (dense with FAISS, BM25, hybrid min-max fusion with α = 0.5, hybrid + ms-marco MiniLM cross-encoder) × k ∈ {3, 5, 10}.
- **Hardware:** a laptop CPU. All numbers come from `results/all_results.json`; `python scripts/report.py` regenerates the tables in `docs/RESULTS.md`.

## What worked

**Hybrid retrieval.** It was best on every metric: MRR 0.919, Recall@5 0.991, Recall@1 0.868. Dense-only retrieval with the best neural model reached 0.789 MRR. Dense retrieval tends to miss exact names, numbers and rare terms, while BM25 misses paraphrases. Normalising both scores and adding them keeps both kinds of match, so the hybrid is never much worse than its better half, and often better than both.

**Overlap on small chunks.** Going from 0 to 15% overlap at 200 words raised hybrid MRR by 0.037 and BM25 MRR by 0.044. Many evidence sentences sat across a chunk boundary, and overlap keeps them whole in at least one chunk. At 500 and 1000 words, boundaries are rare relative to chunk size, so overlap barely mattered.

**bge-small among neural embeddings.** It had the best dense MRR (0.774), ahead of e5-base (0.718) and MiniLM (0.704), at a third of e5-base's compute. Its query instruction prefix ("Represent this sentence for searching relevant passages:") is part of the reason; e5 also uses its `query:` / `passage:` prefixes here.

## What didn't

**The cross-encoder reranker.** On both chunkings it lowered MRR by about 0.02 and Recall@1 by about 2 points, while making each query about 50× slower on CPU (2.3 s vs 44 ms). The hybrid first stage already put the evidence at rank 1 for 87% of questions, so there was little left to fix. The ms-marco model was trained on short web queries and passages, and it sometimes preferred a passage that discusses the topic over the one that states the exact fact. A reranker trained on this kind of text, or one that only reorders the top 5, might behave differently.

**e5-base.** It was the slowest model (121 s to index, 114 ms per query) and no better than bge-small, in either dense or hybrid retrieval.

**Very large chunks for factoid questions.** 1000-word chunks give near-perfect recall, but only because each chunk is a big slice of an article. Precision suffers: the answer model has to find one sentence in about 850 words, and correctness dropped slightly (0.323–0.327 vs 0.347 for 200-word chunks with overlap).

## Why BM25 and TF-IDF look so strong

BM25 alone reached 0.906 MRR, and TF-IDF used as a "dense" model beat every neural model on its own. That is not a general result about these methods. The questions were drafted from their evidence passages and often reuse their words, which suits keyword matching. Two things would make the comparison fairer:

1. Hand-written questions that paraphrase the evidence (the planned "manual" half of the eval set; see `data/eval/parts/manual.jsonl`).
2. Questions whose answer spans two passages, where neither passage shares most words with the question.

Expect neural and hybrid retrieval to pull further ahead of BM25 on that set.

## Answer quality

No LLM key was configured during these runs, so the generator returned the best-matching sentence from the retrieved passages, and the judge used token overlap:

- Faithfulness is 1.0 by construction, because the answer is copied from the context.
- Correctness (token F1 against the reference) sits around 0.33–0.35 for every configuration and tracks Recall@1 loosely. It mainly measures how much a single extracted sentence overlaps a paraphrased reference answer.

Real generation and LLM-judge scores need `ANTHROPIC_API_KEY`. With a key, `generate.py` uses `claude-opus-5-5` with citations and `judge.py` scores both metrics with structured output; re-running `python scripts/run_experiments.py retrieval` fills them in.

## Recommendation

For this collection: **200-word chunks with 15% overlap, bge-small embeddings, hybrid retrieval, top-k = 5.** This configuration has the best MRR, finds the evidence in the top 5 for 99% of questions, answers in about 45 ms on a CPU, and costs nothing per query beyond the answer model.

## Limitations

- The eval set was drafted by an LLM and checked automatically (evidence must be verbatim), not reviewed by a person.
- One small, single-domain collection; differences under about 0.01 MRR are within noise for 114 questions.
- Index times include embedding the whole collection once; the cache stores the original time, so repeated runs report it consistently.
- Latency was measured on one laptop CPU while other jobs were running at times; treat it as relative, not absolute.
