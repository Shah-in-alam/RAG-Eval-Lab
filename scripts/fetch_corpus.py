"""Download the document collection: English Wikipedia articles on information retrieval and NLP.

Text is fetched as plain text with section headings (`== Heading ==`) through the
MediaWiki extracts API and saved to data/raw/<doc_id>.txt. Content is CC BY-SA 4.0;
see data/raw/SOURCES.md. Raw API responses are cached in .cache/wiki/ so the
cleaning step can be re-run without hitting Wikipedia again.

    python scripts/fetch_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ARTICLES = {
    "information_retrieval": "Information retrieval",
    "okapi_bm25": "Okapi BM25",
    "tf_idf": "Tf–idf",
    "inverted_index": "Inverted index",
    "word_embedding": "Word embedding",
    "sentence_embedding": "Sentence embedding",
    "transformer": "Transformer (deep learning architecture)",
    "bert": "BERT (language model)",
    "vector_database": "Vector database",
    "nearest_neighbor_search": "Nearest neighbor search",
    "retrieval_augmented_generation": "Retrieval-augmented generation",
    "large_language_model": "Large language model",
    "precision_and_recall": "Precision and recall",
    "cosine_similarity": "Cosine similarity",
}

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAGEvalLab/0.1 (educational research project)"}
DROP_SECTIONS = {"See also", "References", "External links", "Further reading", "Notes",
                 "Sources", "Citations", "Bibliography", "General references"}
CACHE = Path(".cache/wiki")
DISPLAY = "{\\displaystyle"


def fetch(title: str) -> tuple[str, str, int]:
    params = {"action": "query", "format": "json", "prop": "extracts|revisions", "rvprop": "ids",
              "explaintext": 1, "exsectionformat": "wiki", "redirects": 1, "titles": title}
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                page = next(iter(json.load(resp)["query"]["pages"].values()))
            return page["title"], page["extract"], page["revisions"][0]["revid"]
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(5 * 2 ** attempt)  # rate limited: back off
    raise RuntimeError("unreachable")


def collapse_math(text: str) -> str:
    """MathML in the extract comes out as many indented one-token lines followed by a
    `{\\displaystyle ...}` line with the LaTeX source. Keep only the LaTeX, inline as $...$."""
    out: list[str] = []
    in_math = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(DISPLAY) and stripped.endswith("}"):
            latex = stripped[len(DISPLAY):-1].strip()
            if out:
                out[-1] = f"{out[-1].rstrip()} ${latex}$"
            else:
                out.append(f"${latex}$")
            in_math = True
        elif (line[:1] in (" ", "\t") and " " not in stripped) or (in_math and not stripped):
            in_math = True  # rendered-math noise: indented single tokens
        elif in_math and out and (stripped[:1] in ",.;:)" or stripped[:1].islower()):
            sep = "" if stripped[:1] in ",.;:)" else " "
            out[-1] = out[-1] + sep + stripped
            in_math = False
        else:
            out.append(line)
            in_math = False
    return "\n".join(out)


def clean(text: str) -> str:
    """Collapse math markup, drop reference-style and empty sections, normalise whitespace."""
    text = collapse_math(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", text)
    parts = re.split(r"(?m)^(=+ .+? =+)\s*$", text)
    out = [parts[0].strip()]
    for heading, body in zip(parts[1::2], parts[2::2]):
        name = heading.strip("= ").strip()
        if name in DROP_SECTIONS:
            continue
        body = body.strip()
        if body:
            out.append(f"{heading}\n{body}")
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(out)).strip() + "\n"


def main(out_dir: str = "data/raw") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = ["# Sources\n", "All documents are from English Wikipedia, licensed CC BY-SA 4.0.\n",
               "| doc_id | Article | Revision |", "|---|---|---|"]
    for doc_id, title in ARTICLES.items():
        cached = CACHE / f"{doc_id}.json"
        if cached.exists():
            real_title, text, revid = json.loads(cached.read_text(encoding="utf-8"))
        else:
            real_title, text, revid = fetch(title)
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps([real_title, text, revid]), encoding="utf-8")
            time.sleep(2)
        text = clean(text)
        (out / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        url = f"https://en.wikipedia.org/w/index.php?oldid={revid}"
        sources.append(f"| {doc_id} | [{real_title}]({url}) | {revid} |")
        print(f"{doc_id:32} {len(text.split()):6} words")
    (out / "SOURCES.md").write_text("\n".join(sources) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
