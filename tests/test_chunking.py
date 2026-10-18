from src.chunking import (Chunk, chunk_by_heading, chunk_by_paragraph, chunk_fixed,
                          chunker_name, make_chunker)

TEXT = " ".join(f"w{i}" for i in range(250))

DOC = """Intro sentence one. Intro sentence two.

== History ==
Started in 1970. It grew.

More history here.

== Usage ==
Used widely.
"""


def _offsets_ok(text, chunks):
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_fixed_sizes_and_offsets():
    chunks = chunk_fixed("d", TEXT, size=100)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 50]
    _offsets_ok(TEXT, chunks)
    assert len({c.id for c in chunks}) == 3


def test_overlap_repeats_words():
    chunks = chunk_fixed("d", TEXT, size=100, overlap=0.15)
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-15:] == b[:15]
    _offsets_ok(TEXT, chunks)


def test_heading_split():
    chunks = chunk_by_heading("d", DOC)
    assert len(chunks) == 3
    assert chunks[0].text.startswith("Intro")
    assert chunks[1].text.startswith("== History ==")
    assert chunks[2].text.startswith("== Usage ==")
    _offsets_ok(DOC, chunks)


def test_heading_split_caps_long_sections():
    long_doc = "== A ==\n" + TEXT
    chunks = chunk_by_heading("d", long_doc, max_tokens=100)
    assert all(len(c.text.split()) <= 100 for c in chunks)
    _offsets_ok(long_doc, chunks)


def test_paragraph_split_merges_short_paragraphs():
    chunks = chunk_by_paragraph("d", DOC, max_tokens=12)
    _offsets_ok(DOC, chunks)
    assert all(len(c.text.split()) <= 12 for c in chunks)
    # short paragraphs are merged, so fewer chunks than paragraphs (6)
    assert 1 < len(chunks) < 6


def test_empty_text_returns_no_chunks():
    assert chunk_fixed("d", "   ", 100) == []
    assert chunk_by_heading("d", "") == []
    assert chunk_by_paragraph("d", "\n\n") == []


def test_make_chunker_and_name():
    spec = {"type": "fixed", "size": 500, "overlap": 0.15}
    assert chunker_name(spec) == "fixed-500-o15"
    assert chunker_name({"type": "fixed", "size": 200}) == "fixed-200-o0"
    assert chunker_name({"type": "heading"}) == "heading"
    out = make_chunker({"type": "fixed", "size": 100})("d", TEXT)
    assert isinstance(out[0], Chunk) and len(out) == 3
