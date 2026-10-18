import pytest
from fastapi.testclient import TestClient

import api.server as server
from src.ingest import UploadStore
from tests.test_ingest import make_docx


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", UploadStore(tmp_path / "uploads"))
    monkeypatch.setattr(server, "RESULTS_PATH", tmp_path / "all_results.json")
    server._RETRIEVERS.clear()
    server._CHUNKS.clear()
    return TestClient(server.app)


def test_upload_list_query_delete(client):
    files = [
        ("files", ("plan.docx", make_docx(), "application/octet-stream")),
        ("files", ("notes.txt", b"The launch codename is Bluebird and ships in March.", "text/plain")),
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
    ]
    r = client.post("/api/uploads", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["added"]) == 2 and body["errors"][0]["filename"] == "bad.exe"

    listed = client.get("/api/uploads").json()
    assert {m["filename"] for m in listed} == {"plan.docx", "notes.txt"}
    docs = client.get("/api/documents").json()
    assert sum(d["collection"] == "uploads" for d in docs) == 2 and sum(d["collection"] == "builtin" for d in docs) == 14

    q = {"question": "What is the launch codename?", "chunker": {"type": "paragraph"}, "embedder": "tfidf",
         "method": "bm25", "k": 3, "generate": False, "collection": "uploads"}
    hits = client.post("/api/query", json=q).json()["hits"]
    assert hits[0]["title"] == "notes" and all(h["doc_id"].startswith("up_") for h in hits)
    allhits = client.post("/api/query", json={**q, "collection": "all", "k": 20}).json()["hits"]
    assert any(not h["doc_id"].startswith("up_") for h in allhits)

    doc_id = next(m["doc_id"] for m in listed if m["filename"] == "notes.txt")
    assert "Bluebird" in client.get(f"/api/documents/{doc_id}").json()["text"]
    assert client.delete(f"/api/uploads/{doc_id}").status_code == 204
    assert client.delete(f"/api/uploads/{doc_id}").status_code == 404
    hits = client.post("/api/query", json=q).json()["hits"]
    assert all(h["doc_id"] != doc_id for h in hits)  # index rebuilt after delete


def test_query_uploads_when_empty_is_400(client):
    q = {"question": "x", "chunker": {"type": "heading"}, "embedder": "tfidf", "method": "bm25", "k": 3,
         "collection": "uploads"}
    r = client.post("/api/query", json=q)
    assert r.status_code == 400 and "upload" in r.json()["detail"].lower()


def test_url_import_rejects_unsafe_and_bad(client):
    assert client.post("/api/uploads/url", json={"url": "http://127.0.0.1:8000/api/health"}).status_code == 400
    assert client.post("/api/uploads/url", json={"url": "notaurl"}).status_code == 400


def test_url_import_uses_fetcher(client, monkeypatch):
    monkeypatch.setattr(server, "fetch_url", lambda url: ("shared.txt", b"Shared drive file about embeddings."))
    r = client.post("/api/uploads/url", json={"url": "https://drive.google.com/file/d/ABC/view"})
    assert r.status_code == 200 and r.json()["source"] == "link" and r.json()["filename"] == "shared.txt"
