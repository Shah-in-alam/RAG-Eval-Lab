from src.chunking import Chunk
from src.generate import Answer, generate_answer, llm_available, llm_cost_usd
from src.judge import judge_answer
from src.retrievers import Hit

CTX = ("Okapi BM25 is a ranking function used by search engines. "
       "The parameter b is usually set to 0.75. BM25 ignores word order.")
HITS = [
    Hit(Chunk("a#0", "a", "Zebras eat grass in Africa.", 0, 27), 0.9),
    Hit(Chunk("b#0", "b", CTX, 0, len(CTX)), 0.8),
]


def test_extractive_answer_cites_source(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.generate.llm_available", lambda: False)
    ans = generate_answer("What is the BM25 parameter b usually set to?", HITS)
    assert isinstance(ans, Answer)
    assert ans.generator == "extractive"
    assert "0.75" in ans.text
    assert ans.citations == [2]
    assert ans.input_tokens == 0 and ans.output_tokens == 0


def test_extractive_no_hits():
    ans = generate_answer("anything", [], force_offline=True)
    assert ans.text == "I don't know." and ans.citations == []


def test_heuristic_judge_scores():
    s = judge_answer("q", "The parameter b is usually set to 0.75.", "b is usually 0.75", [CTX], force_offline=True)
    assert s["judge"] == "heuristic"
    assert s["faithfulness"] == 1.0
    assert 0.3 < s["correctness"] <= 1.0
    bad = judge_answer("q", "Penguins enjoy cold weather near Antarctica.", "b is usually 0.75", [CTX], force_offline=True)
    assert bad["faithfulness"] < 0.3 and bad["correctness"] == 0.0
    exact = judge_answer("q", "b is 0.75", "b is 0.75", [CTX], force_offline=True)
    assert exact["correctness"] == 1.0


def test_llm_path_selected_only_with_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("RAG_DISABLE_LLM", "")
    assert llm_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm_available() is True
    monkeypatch.setenv("RAG_DISABLE_LLM", "1")
    assert llm_available() is False


def test_llm_cost():
    assert llm_cost_usd(1_000_000, 0) == 4.0
    assert llm_cost_usd(0, 1_000_000) == 20.0


def test_extractive_matches_plurals():
    text = ("BM25+ only has one additional free parameter delta. "
            "The k1 and b are free parameters, usually chosen as k1 in [1.2, 2.0] and b=0.75.")
    hits = [Hit(Chunk("c#0", "c", text, 0, len(text)), 1.0)]
    ans = generate_answer("What value is the BM25 parameter b usually set to?", hits, force_offline=True)
    assert "0.75" in ans.text
