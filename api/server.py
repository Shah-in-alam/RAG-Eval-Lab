"""HTTP API for the RAG Eval Lab web interface.

    uvicorn api.server:app --reload --port 8000

If the web app has been built (web/dist), it is served from / as well.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.chunking import chunk_documents, chunker_name
from src.data import doc_title, evidence_span, load_documents, load_eval
from src.embeddings import EMBEDDERS, EmbedderUnavailable, available_embedders, get_embedder
from src.evaluate import CONFIG_DIR, RESULTS_DIR, load_config, run_suite, to_public
from src.generate import LLM_MODEL, generate_answer, llm_available
from src.ingest import MAX_BYTES, SUPPORTED, EmptyDocument, UnsafeUrl, UnsupportedFile, UploadStore, fetch_url
from src.metrics import is_relevant
from src.retrievers import METHODS, build_retriever

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = RESULTS_DIR / "all_results.json"
WEB_DIST = ROOT / "web" / "dist"

CHUNKERS = [
    {"type": "fixed", "size": 200, "overlap": 0},
    {"type": "fixed", "size": 200, "overlap": 0.15},
    {"type": "fixed", "size": 500, "overlap": 0},
    {"type": "fixed", "size": 500, "overlap": 0.15},
    {"type": "fixed", "size": 1000, "overlap": 0},
    {"type": "fixed", "size": 1000, "overlap": 0.15},
    {"type": "heading"},
    {"type": "paragraph"},
]
KS = [3, 5, 10]

app = FastAPI(title="RAG Eval Lab")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

DOCS = load_documents()
EVAL_BY_QUESTION = {it.question.strip().lower(): it for it in load_eval()}
_RETRIEVERS: OrderedDict[tuple, object] = OrderedDict()
_CHUNKS: dict[tuple, list] = {}
STORE = UploadStore()
Collection = Literal["all", "builtin", "uploads"]
_LOCK = threading.Lock()
_BUILD_LOCK = threading.Lock()  # one index build at a time; later callers reuse it
MAX_CACHED_RETRIEVERS = 12


class ChunkerSpec(BaseModel):
    type: Literal["fixed", "heading", "paragraph"]
    size: int | None = Field(default=None, ge=20, le=4000)
    overlap: float | None = Field(default=None, ge=0, le=0.5)

    def as_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.type == "fixed":
            d["size"] = self.size or 500
            d["overlap"] = self.overlap or 0
        return d


class QueryRequest(BaseModel):
    question: str = Field(max_length=2000)
    chunker: ChunkerSpec
    embedder: str = "minilm"
    method: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid"
    k: int = Field(default=5, ge=1, le=50)
    generate: bool = True
    collection: Collection = "all"

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()

    @field_validator("embedder")
    @classmethod
    def known_embedder(cls, v: str) -> str:
        if v not in EMBEDDERS:
            raise ValueError(f"unknown embedder; choose from {sorted(EMBEDDERS)}")
        return v


class RunRequest(BaseModel):
    config: str = "quick"


class UrlImport(BaseModel):
    url: str = Field(max_length=4000)


def _collection_docs(collection: str) -> tuple[str, dict[str, str]]:
    """(version key, doc_id -> text) for a search collection."""
    if collection == "builtin":
        return "builtin", DOCS
    uploads = STORE.texts()
    if collection == "uploads":
        if not uploads:
            raise HTTPException(400, "No uploaded documents yet. Upload a file on the Documents page first.")
        return f"uploads-{STORE.version}", uploads
    return f"all-{STORE.version}", {**DOCS, **uploads}


def _title(doc_id: str) -> str:
    if doc_id.startswith("up_"):
        meta = STORE.get(doc_id)
        return meta["title"] if meta else doc_id
    return doc_title(doc_id)


def _retriever(spec: dict, embedder: str, method: str, collection: str = "builtin"):
    name = chunker_name(spec)
    version, docs = _collection_docs(collection)
    key = (version, name, "none" if method == "bm25" else embedder, method)
    with _LOCK:
        if key in _RETRIEVERS:
            _RETRIEVERS.move_to_end(key)
            return _RETRIEVERS[key]
    with _BUILD_LOCK:
        with _LOCK:  # another request may have built it while we waited
            if key in _RETRIEVERS:
                return _RETRIEVERS[key]
            if (version, name) not in _CHUNKS:
                _CHUNKS[(version, name)] = chunk_documents(docs, spec)
        try:
            emb = get_embedder("tfidf" if method == "bm25" else embedder)
        except EmbedderUnavailable as e:
            raise HTTPException(400, str(e)) from e
        r = build_retriever(method, _CHUNKS[(version, name)], emb)
        with _LOCK:
            _RETRIEVERS[key] = r
            while len(_RETRIEVERS) > MAX_CACHED_RETRIEVERS:
                _RETRIEVERS.popitem(last=False)
        return r


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_available": llm_available(), "llm_model": LLM_MODEL if llm_available() else None,
            "embedders": available_embedders(), "n_documents": len(DOCS), "n_uploads": len(STORE.list()),
            "upload_types": sorted(SUPPORTED), "max_upload_mb": MAX_BYTES // (1024 * 1024)}


@app.get("/api/options")
def options() -> dict:
    return {"chunkers": [{**c, "name": chunker_name(c)} for c in CHUNKERS], "embedders": available_embedders(),
            "methods": list(METHODS), "ks": KS}


@app.get("/api/documents")
def documents() -> list[dict]:
    builtin = [{"doc_id": d, "title": doc_title(d), "n_words": len(t.split()), "n_chars": len(t), "collection": "builtin"}
               for d, t in DOCS.items()]
    uploads = [{"doc_id": m["doc_id"], "title": m["title"], "n_words": m["n_words"], "n_chars": m["n_chars"],
                "collection": "uploads", "kind": m["kind"], "filename": m["filename"]} for m in STORE.list()]
    return uploads + builtin


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict:
    if doc_id in DOCS:
        return {"doc_id": doc_id, "title": doc_title(doc_id), "text": DOCS[doc_id], "collection": "builtin"}
    meta = STORE.get(doc_id)
    if not meta:
        raise HTTPException(404, "document not found")
    return {"doc_id": doc_id, "title": meta["title"], "text": STORE.texts()[doc_id], "collection": "uploads", "meta": meta}


@app.get("/api/uploads")
def list_uploads() -> list[dict]:
    return STORE.list()


@app.post("/api/uploads")
async def upload_files(files: list[UploadFile] = File(...)) -> dict:
    added, errors = [], []
    for f in files:
        name = Path(f.filename or "upload.txt").name
        data = await f.read(MAX_BYTES + 1)
        try:
            added.append(STORE.add(name, data, source="upload"))
        except (UnsupportedFile, EmptyDocument, ValueError) as e:
            errors.append({"filename": name, "error": str(e)})
    return {"added": added, "errors": errors}


@app.post("/api/uploads/url")
def import_url(req: UrlImport) -> dict:
    try:
        name, data = fetch_url(req.url)
        return STORE.add(name, data, source="link", url=req.url)
    except (UnsafeUrl, UnsupportedFile, EmptyDocument, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # network errors
        raise HTTPException(400, f"Could not download the link: {type(e).__name__}") from e


@app.delete("/api/uploads/{doc_id}", status_code=204)
def delete_upload(doc_id: str) -> Response:
    if not STORE.delete(doc_id):
        raise HTTPException(404, "document not found")
    return Response(status_code=204)


@app.get("/api/eval")
def eval_items() -> list[dict]:
    return [it.to_dict() for it in load_eval()]


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    t0 = time.perf_counter()
    retriever = _retriever(req.chunker.as_dict(), req.embedder, req.method, req.collection)
    t1 = time.perf_counter()
    hits = retriever.search(req.question, req.k)
    t2 = time.perf_counter()
    answer = generate_answer(req.question, hits).to_dict() if req.generate else None
    t3 = time.perf_counter()
    item = EVAL_BY_QUESTION.get(req.question.lower()) if req.collection != "uploads" else None
    span = evidence_span(DOCS, item) if item else None
    return {
        "question": req.question,
        "config": {"chunker": chunker_name(req.chunker.as_dict()), "embedder": req.embedder, "method": req.method, "k": req.k,
                   "collection": req.collection},
        "hits": [{"rank": i, "chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "title": _title(h.chunk.doc_id),
                  "text": h.chunk.text, "score": h.score, "start": h.chunk.start, "end": h.chunk.end,
                  "relevant": is_relevant(h.chunk, span, item.doc_id) if item else None}
                 for i, h in enumerate(hits, 1)],
        "answer": answer,
        "eval_item": item.to_dict() if item else None,
        "timings": {"index_ms": round((t1 - t0) * 1000, 1), "retrieval_ms": round((t2 - t1) * 1000, 1),
                    "generation_ms": round((t3 - t2) * 1000, 1)},
    }


def _read_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@app.get("/api/results")
def results() -> list[dict]:
    return [to_public(r) for r in _read_results()]


@app.get("/api/results/{run_id:path}")
def result(run_id: str) -> dict:
    for r in _read_results():
        if r["run_id"] == run_id:
            return r
    raise HTTPException(404, "run not found")


_STATUS = {"running": False, "suite": None, "done": 0, "total": 0, "error": None, "finished_at": None}


@app.get("/api/experiments/configs")
def configs() -> list[dict]:
    out = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(p)
        desc = next((ln.lstrip("# ").strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("#")), "")
        out.append({"name": p.stem, "description": desc, "chunkers": len(cfg["chunkers"]),
                    "embedders": cfg["embedders"], "methods": cfg["methods"], "ks": cfg["ks"]})
    return out


@app.get("/api/experiments/status")
def status() -> dict:
    return dict(_STATUS)


@app.post("/api/experiments/run", status_code=202)
def run_experiments(req: RunRequest) -> dict:
    path = CONFIG_DIR / f"{Path(req.config).stem}.yaml"
    if not path.exists():
        raise HTTPException(404, f"no config named {req.config!r}")
    with _LOCK:
        if _STATUS["running"]:
            raise HTTPException(409, f"suite {_STATUS['suite']!r} is already running")
        _STATUS.update(running=True, suite=path.stem, done=0, total=0, error=None, finished_at=None)

    def work() -> None:
        try:
            run_suite(path, results_dir=RESULTS_PATH.parent,
                      progress=lambda d, t: _STATUS.update(done=d, total=t))
        except Exception as e:  # surface to the UI
            _STATUS["error"] = f"{type(e).__name__}: {e}"
        finally:
            _STATUS.update(running=False, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    threading.Thread(target=work, daemon=True).start()
    return dict(_STATUS)


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
