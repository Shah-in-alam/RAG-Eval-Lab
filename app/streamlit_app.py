"""Minimal Streamlit demo (the main UI is the TypeScript app in web/).

    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.chunking import chunk_documents  # noqa: E402
from src.data import load_documents  # noqa: E402
from src.embeddings import available_embedders, get_embedder  # noqa: E402
from src.generate import generate_answer  # noqa: E402
from src.retrievers import METHODS, build_retriever  # noqa: E402

CHUNKERS = {"fixed-200-o15": {"type": "fixed", "size": 200, "overlap": 0.15},
            "fixed-500-o0": {"type": "fixed", "size": 500, "overlap": 0},
            "heading": {"type": "heading"}, "paragraph": {"type": "paragraph"}}

st.set_page_config(page_title="RAG Eval Lab", layout="wide")
st.title("RAG Eval Lab")


@st.cache_resource(show_spinner="Building index…")
def retriever(chunker: str, embedder: str, method: str):
    chunks = chunk_documents(load_documents(), CHUNKERS[chunker])
    return build_retriever(method, chunks, get_embedder("tfidf" if method == "bm25" else embedder))


ask, results = st.tabs(["Ask", "Results"])
with ask:
    c1, c2, c3, c4 = st.columns(4)
    chunker = c1.selectbox("Chunking", list(CHUNKERS))
    method = c2.selectbox("Retrieval", METHODS, index=2)
    embedder = c3.selectbox("Embedding", available_embedders(), index=1)
    k = c4.selectbox("Top-k", [3, 5, 10], index=1)
    q = st.text_input("Question", "What value is the BM25 parameter b usually set to?")
    if q.strip():
        hits = retriever(chunker, embedder, method).search(q, k)
        ans = generate_answer(q, hits)
        st.subheader("Answer")
        st.write(ans.text)
        st.caption(f"generator: {ans.generator}")
        for i, h in enumerate(hits, 1):
            with st.expander(f"[{i}] {h.chunk.doc_id} · score {h.score:.3f}"):
                st.write(h.chunk.text)

with results:
    path = ROOT / "results" / "all_results.json"
    if not path.exists():
        st.info("No results yet. Run: python scripts/run_experiments.py quick")
    else:
        rows = [{k: v for k, v in r.items() if k not in ("per_question", "chunker_spec", "recall_by_type")}
                for r in json.loads(path.read_text(encoding="utf-8"))]
        st.dataframe(sorted(rows, key=lambda r: -r["mrr"]), use_container_width=True)
