from src.data import EvalItem, evidence_span, load_documents, load_eval


def test_every_evidence_is_verbatim():
    docs = load_documents()
    items = load_eval()
    for it in items:
        assert it.evidence in docs[it.doc_id], it.id
        a, b = evidence_span(docs, it)
        assert docs[it.doc_id][a:b] == it.evidence


def test_eval_set_size_and_fields():
    items = load_eval()
    assert len(items) >= 90
    assert len({it.id for it in items}) == len(items)
    assert all(isinstance(it, EvalItem) for it in items)
    assert {it.source for it in items} <= {"manual", "generated"}
    assert {it.type for it in items} == {"factoid", "explanatory"}


def test_documents_loaded():
    docs = load_documents()
    assert len(docs) == 14
    assert "okapi_bm25" in docs and len(docs["transformer"]) > 10_000
