"""Small text helpers shared by the extractive generator and the heuristic judge."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has have
having he her here hers him his how i if in into is it its itself just me more most my no nor
not of off on once only or other our ours out over own same she should so some such than that
the their theirs them then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your yours also using used
called known one two
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[$])|\n+")


def _stem(t: str) -> str:
    """Very light plural folding so 'parameters' matches 'parameter'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 1]
