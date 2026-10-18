"""Minimal, dependency-light version of the RAG Eval Lab experiment.

Runs three chunk sizes x three retrieval methods on a built-in mini corpus and
scores each combination with Recall@1, Recall@3 and MRR.

    pip install numpy scikit-learn
    python demo.py
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS = {
    "bm25": (
        "Okapi BM25 is a ranking function used by search engines to estimate the relevance "
        "of documents to a given search query. It is based on the probabilistic retrieval "
        "framework developed in the 1970s and 1980s by Stephen Robertson and Karen Sparck Jones. "
        "BM25 has two free parameters. The parameter k1 controls term frequency saturation and is "
        "usually chosen between 1.2 and 2.0. The parameter b controls how strongly document length "
        "normalisation is applied and is usually set to 0.75. BM25 is a bag-of-words model, so it "
        "ignores the order in which query terms appear in a document."
    ),
    "embeddings": (
        "A sentence embedding maps a piece of text to a dense vector of real numbers. Texts with "
        "similar meaning end up close together in the vector space. Similarity between two "
        "embeddings is usually measured with cosine similarity, which is the dot product of the "
        "two vectors divided by the product of their lengths. Popular open models include "
        "all-MiniLM-L6-v2, which produces 384-dimensional vectors and is fast enough to run on a "
        "laptop CPU. Dense retrieval uses these vectors to find passages that are semantically "
        "related to a question even when they share no words with it."
    ),
    "chunking": (
        "Before documents are indexed for retrieval they are split into chunks. Fixed-size "
        "chunking cuts the text every N tokens regardless of its structure. Overlapping chunks "
        "repeat a few tokens from the end of the previous chunk so that a sentence cut at a "
        "boundary still appears whole in one chunk. Semantic chunking instead splits at natural "
        "boundaries such as headings or paragraphs. Small chunks give precise matches for factual "
        "questions, while large chunks keep more context for questions whose answer spans several "
        "sentences."
    ),
    "vector_db": (
        "A vector database stores embeddings and answers nearest neighbour queries. Exact search "
        "compares the query with every stored vector, which becomes slow for millions of items. "
        "Approximate nearest neighbour indexes such as HNSW build a navigable graph that finds "
        "close vectors in logarithmic time at the cost of occasionally missing the true nearest "
        "neighbour. FAISS is a library from Meta for efficient similarity search, and ChromaDB is "
        "an open source embedding database designed for AI applications."
    ),
    "rag": (
        "Retrieval-augmented generation combines a retriever with a large language model. The "
        "retriever finds passages relevant to the user's question, and the passages are inserted "
        "into the prompt so that the model can ground its answer in them. This reduces "
        "hallucination and lets the system answer questions about documents the model was never "
        "trained on. The term was introduced in a 2020 paper by Patrick Lewis and colleagues at "
        "Facebook AI Research. A reranker can be added after retrieval to reorder the candidates "
        "with a more expensive cross-encoder model."
    ),
    "metrics": (
        "Retrieval systems are evaluated with labelled questions whose answers are known. Recall "
        "at k measures the fraction of questions for which a relevant passage appears among the "
        "top k results. Mean reciprocal rank averages one divided by the rank of the first "
        "relevant result, so a system that puts the right passage first scores 1.0. Faithfulness "
        "measures whether a generated answer is supported by the retrieved context rather than "
        "invented by the model."
    ),
}

# (question, doc_id, evidence) - evidence is copied verbatim from the document.
QUESTIONS = [
    ("What value is the BM25 parameter b usually set to?", "bm25", "usually set to 0.75"),
    ("Who developed the probabilistic framework behind BM25?", "bm25", "Stephen Robertson and Karen Sparck Jones"),
    ("Does BM25 care about the order of query terms?", "bm25", "ignores the order in which query terms appear"),
    ("How many dimensions do all-MiniLM-L6-v2 vectors have?", "embeddings", "produces 384-dimensional vectors"),
    ("How is the similarity of two embeddings usually measured?", "embeddings", "usually measured with cosine similarity"),
    ("Why do overlapping chunks repeat tokens?", "chunking", "a sentence cut at a boundary still appears whole in one chunk"),
    ("Where does semantic chunking split text?", "chunking", "natural boundaries such as headings or paragraphs"),
    ("Which index builds a navigable graph for approximate search?", "vector_db", "HNSW build a navigable graph"),
    ("Which company created FAISS?", "vector_db", "FAISS is a library from Meta"),
    ("When was retrieval-augmented generation introduced?", "rag", "introduced in a 2020 paper by Patrick Lewis"),
    ("What model type does a reranker use?", "rag", "more expensive cross-encoder model"),
    ("What does mean reciprocal rank average?", "metrics", "averages one divided by the rank of the first relevant result"),
]

CHUNK_SIZES = (20, 40, 80)  # words; the mini corpus is small
METHODS = ("tfidf", "bm25", "hybrid")


def chunk_words(doc_id: str, text: str, size: int) -> list[dict]:
    """Fixed-size chunks of `size` words, keeping character offsets."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    chunks = []
    for i in range(0, len(spans), size):
        group = spans[i : i + size]
        start, end = group[0][0], group[-1][1]
        chunks.append({"doc_id": doc_id, "text": text[start:end], "start": start, "end": end})
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(chunks: list[dict], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    docs = [tokenize(c["text"]) for c in chunks]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    scores = np.zeros(n)
    for term in set(tokenize(query)):
        df = sum(term in d for d in docs)
        if df == 0:
            continue
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        for i, d in enumerate(docs):
            tf = d.count(term)
            scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
    return scores


def minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def is_relevant(chunk: dict, doc_id: str, span: tuple[int, int]) -> bool:
    """A chunk is relevant if it covers at least half of the evidence span."""
    if chunk["doc_id"] != doc_id:
        return False
    overlap = min(chunk["end"], span[1]) - max(chunk["start"], span[0])
    return overlap >= 0.5 * (span[1] - span[0])


def run_demo() -> list[dict]:
    rows = []
    for size in CHUNK_SIZES:
        chunks = [c for doc_id, text in CORPUS.items() for c in chunk_words(doc_id, text, size)]
        vec = TfidfVectorizer(stop_words="english").fit([c["text"] for c in chunks])
        matrix = vec.transform([c["text"] for c in chunks])
        for method in METHODS:
            r1 = r3 = rr = 0.0
            for question, doc_id, evidence in QUESTIONS:
                start = CORPUS[doc_id].index(evidence)
                span = (start, start + len(evidence))
                tfidf = (matrix @ vec.transform([question]).T).toarray().ravel()
                bm25 = bm25_scores(chunks, question)
                scores = {"tfidf": tfidf, "bm25": bm25, "hybrid": 0.5 * minmax(tfidf) + 0.5 * minmax(bm25)}[method]
                ranked = np.argsort(-scores, kind="stable")
                relevant = [is_relevant(chunks[i], doc_id, span) for i in ranked]
                r1 += any(relevant[:1])
                r3 += any(relevant[:3])
                rr += next((1.0 / (rank + 1) for rank, rel in enumerate(relevant) if rel), 0.0)
            n = len(QUESTIONS)
            rows.append({"chunk_size": size, "method": method, "n_chunks": len(chunks),
                         "recall@1": r1 / n, "recall@3": r3 / n, "mrr": rr / n})
    return rows


def main() -> None:
    rows = run_demo()
    print(f"Mini corpus: {len(CORPUS)} documents, {len(QUESTIONS)} questions\n")
    print(f"{'chunk':>6} {'method':<8} {'chunks':>6} {'R@1':>6} {'R@3':>6} {'MRR':>6}")
    for r in rows:
        print(f"{r['chunk_size']:>6} {r['method']:<8} {r['n_chunks']:>6} "
              f"{r['recall@1']:>6.3f} {r['recall@3']:>6.3f} {r['mrr']:>6.3f}")
    best = max(rows, key=lambda r: r["mrr"])
    print(f"\nBest by MRR: chunk_size={best['chunk_size']}, method={best['method']} (MRR {best['mrr']:.3f})")


if __name__ == "__main__":
    main()
