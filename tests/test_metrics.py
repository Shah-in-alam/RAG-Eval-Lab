from src.chunking import Chunk
from src.metrics import is_relevant, mean, recall_at_k, reciprocal_rank


def test_recall_and_mrr():
    ranked = [False, True, False]
    assert recall_at_k(ranked, 1) == 0.0
    assert recall_at_k(ranked, 3) == 1.0
    assert reciprocal_rank(ranked) == 0.5
    assert reciprocal_rank([False, False]) == 0.0
    assert recall_at_k([], 5) == 0.0
    assert mean([]) == 0.0 and mean([1.0, 0.0]) == 0.5


def test_relevance_boundary():
    # evidence covers chars 100..200; chunk A holds 60% of it, chunk B 40%
    span = (100, 200)
    a = Chunk("d#0", "d", "x", 0, 160)
    b = Chunk("d#1", "d", "x", 160, 400)
    assert is_relevant(a, span, "d")
    assert not is_relevant(b, span, "d")


def test_relevance_wrong_doc_and_small_chunk_inside_evidence():
    span = (100, 400)
    assert not is_relevant(Chunk("e#0", "e", "x", 0, 500), span, "d")
    # a chunk entirely inside a long evidence span counts as relevant
    assert is_relevant(Chunk("d#3", "d", "x", 150, 250), span, "d")
