import pytest
from fastapi.testclient import TestClient

import api.server as server
from src.ingest import UploadStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RESULTS_PATH", tmp_path / "all_results.json")
    monkeypatch.setattr(server, "STORE", UploadStore(tmp_path / "uploads"))  # never touch real uploads
    return TestClient(server.app)


def test_health_and_options(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "tfidf" in h["embedders"]
    o = client.get("/api/options").json()
    assert {"dense", "bm25", "hybrid", "hybrid_rerank"} <= set(o["methods"])
    assert o["ks"] == [3, 5, 10] and any(c["type"] == "heading" for c in o["chunkers"])


def test_documents_and_eval(client):
    docs = client.get("/api/documents").json()
    assert len(docs) == 14 and {"doc_id", "title", "n_words"} <= set(docs[0])
    one = client.get("/api/documents/okapi_bm25").json()
    assert "BM25" in one["text"]
    assert client.get("/api/documents/nope").status_code == 404
    items = client.get("/api/eval").json()
    assert len(items) >= 90 and "evidence" in items[0]


def test_query_returns_k_hits(client):
    body = {"question": "What is the BM25 parameter b usually set to?", "chunker": {"type": "fixed", "size": 200},
            "embedder": "tfidf", "method": "hybrid", "k": 5, "generate": True}
    r = client.post("/api/query", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["hits"]) == 5
    assert data["hits"][0]["doc_id"] == "okapi_bm25"
    assert data["answer"]["generator"] in ("extractive", "llm")
    assert "retrieval_ms" in data["timings"]


def test_empty_question_is_422(client):
    r = client.post("/api/query", json={"question": "   ", "chunker": {"type": "heading"},
                                         "embedder": "tfidf", "method": "bm25", "k": 3})
    assert r.status_code == 422


def test_bad_method_and_k_are_422(client):
    base = {"question": "x", "chunker": {"type": "heading"}, "embedder": "tfidf"}
    assert client.post("/api/query", json={**base, "method": "magic", "k": 3}).status_code == 422
    assert client.post("/api/query", json={**base, "method": "bm25", "k": 0}).status_code == 422


def test_results_empty_then_404(client):
    assert client.get("/api/results").json() == []
    assert client.get("/api/results/nope").status_code == 404


def test_experiment_status_idle(client):
    s = client.get("/api/experiments/status").json()
    assert s["running"] is False
    assert client.post("/api/experiments/run", json={"config": "does-not-exist"}).status_code == 404


def test_query_marks_relevant_hits_for_eval_questions(client):
    item = client.get("/api/eval").json()[0]
    body = {"question": item["question"], "chunker": {"type": "fixed", "size": 200},
            "embedder": "tfidf", "method": "bm25", "k": 5, "generate": False}
    data = client.post("/api/query", json=body).json()
    assert data["eval_item"]["id"] == item["id"]
    assert all("relevant" in h for h in data["hits"])
    assert any(h["relevant"] for h in data["hits"])
    other = client.post("/api/query", json={**body, "question": "something not in the eval set"}).json()
    assert other["eval_item"] is None and all(h["relevant"] is None for h in other["hits"])
