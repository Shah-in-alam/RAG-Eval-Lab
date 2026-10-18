"""Embedding models behind one small interface.

Every embedder returns L2-normalised float32 rows, so a dot product is cosine
similarity. Local sentence-transformers models are loaded lazily and shared;
document embeddings are cached on disk keyed by model name + texts.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "emb"


class EmbedderUnavailable(RuntimeError):
    """Raised when an embedder needs something that is not installed or configured."""


class Embedder(Protocol):
    name: str
    cacheable: bool
    cost_per_1m_tokens: float  # USD; 0 for local models

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


def approx_tokens(texts: list[str]) -> int:
    """Rough subword token count (~1.3 tokens per English word) for cost estimates."""
    return int(sum(len(t.split()) for t in texts) * 1.3)


class TfidfEmbedder:
    """Sparse lexical vectors (as dense rows). Fitted on the documents it embeds, so not cached."""

    name = "tfidf"
    cacheable = False
    cost_per_1m_tokens = 0.0

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        m = self._vec.fit_transform(texts)
        self._fitted = True
        return _normalise(m.toarray())

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call embed_documents before embed_query")
        return _normalise(self._vec.transform([text]).toarray())[0]


_ST_MODELS: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()  # concurrent loads of one model crash torch ("meta tensor")


class SentenceTransformerEmbedder:
    cacheable = True
    cost_per_1m_tokens = 0.0

    def __init__(self, name: str, model_id: str, query_prefix: str = "", doc_prefix: str = "") -> None:
        self.name = name
        self.model_id = model_id
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    @property
    def model(self):
        with _MODEL_LOCK:
            if self.model_id not in _ST_MODELS:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:  # pragma: no cover
                    raise EmbedderUnavailable("pip install sentence-transformers") from e
                _ST_MODELS[self.model_id] = SentenceTransformer(self.model_id, device="cpu")
            return _ST_MODELS[self.model_id]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [self.doc_prefix + t for t in texts]
        return _normalise(self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self.model.encode([self.query_prefix + text], normalize_embeddings=True, show_progress_bar=False))[0]


class VoyageEmbedder:
    """Voyage AI API embeddings (Anthropic's recommended embedding provider). Needs VOYAGE_API_KEY."""

    name = "api"
    cacheable = True
    model_id = "voyage-3.5-lite"
    cost_per_1m_tokens = 0.02  # USD list price for voyage-3.5-lite; check current pricing

    def __init__(self) -> None:
        self.key = os.environ.get("VOYAGE_API_KEY")
        if not self.key:
            raise EmbedderUnavailable("set VOYAGE_API_KEY to use the API embedding model")

    def _call(self, texts: list[str], input_type: str) -> np.ndarray:
        import httpx

        out = []
        for i in range(0, len(texts), 128):
            for attempt in range(5):
                r = httpx.post("https://api.voyageai.com/v1/embeddings", timeout=60,
                               headers={"Authorization": f"Bearer {self.key}"},
                               json={"input": texts[i : i + 128], "model": self.model_id, "input_type": input_type})
                if r.status_code != 429:
                    break
                time.sleep(2 ** attempt)
            r.raise_for_status()
            out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        return _normalise(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._call(texts, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "query")[0]


EMBEDDERS = {
    "tfidf": lambda: TfidfEmbedder(),
    "minilm": lambda: SentenceTransformerEmbedder("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    "bge-small": lambda: SentenceTransformerEmbedder(
        "bge-small", "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: "),
    "e5-base": lambda: SentenceTransformerEmbedder(
        "e5-base", "intfloat/e5-base-v2", query_prefix="query: ", doc_prefix="passage: "),
    "api": lambda: VoyageEmbedder(),
}


def get_embedder(name: str) -> Embedder:
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
    return EMBEDDERS[name]()


def available_embedders() -> list[str]:
    names = []
    for name in EMBEDDERS:
        try:
            get_embedder(name)
            names.append(name)
        except EmbedderUnavailable:
            pass
    return names


def cached_embed_timed(embedder, texts: list[str], cache_dir: str | Path = CACHE_DIR) -> tuple[np.ndarray, float, bool]:
    """embed_documents with an on-disk cache keyed by embedder name + exact texts.

    Returns (matrix, seconds, cache_hit). On a cache hit, `seconds` is the time the
    original embedding took, so reported indexing time reflects the real cost.
    """
    if not getattr(embedder, "cacheable", False):
        t0 = time.perf_counter()
        m = embedder.embed_documents(texts)
        return m, time.perf_counter() - t0, False
    h = hashlib.sha1(embedder.name.encode())
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    path = Path(cache_dir) / f"{embedder.name}-{h.hexdigest()[:16]}.npy"
    meta = path.with_suffix(".time")
    if path.exists() and meta.exists():  # entries without timing are recomputed
        return np.load(path), float(meta.read_text()), True
    t0 = time.perf_counter()
    m = embedder.embed_documents(texts)
    seconds = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, m)
    meta.write_text(f"{seconds:.6f}")
    return m, seconds, False


def cached_embed(embedder, texts: list[str], cache_dir: str | Path = CACHE_DIR) -> np.ndarray:
    return cached_embed_timed(embedder, texts, cache_dir)[0]
