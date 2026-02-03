
from cuecoach.rag.chunking import DocMeta, chunk_text


def test_chunk_text_returns_chunks_with_ids():
    text = "Para1 line.\n\nPara2 line.\n\nPara3 line."
    meta = DocMeta(doc_id="doc1", source="Test", title="T", topic="aiming", skill_level="beginner")
    chunks = chunk_text(text, meta, max_chars=20, overlap_chars=5)

    assert len(chunks) >= 2
    assert all(c.chunk_id.startswith("doc1::") for c in chunks)
    assert all(c.text for c in chunks)


def test_chunk_overlap_present():
    text = ("A " * 80) + "\n\n" + ("B " * 80)  # includes whitespace
    meta = DocMeta(doc_id="doc2", source="Test", title="T")
    chunks = chunk_text(text, meta, max_chars=120, overlap_chars=20)

    assert len(chunks) >= 2
    tail = chunks[0].text[-20:]
    assert tail in chunks[1].text

def test_chunk_does_not_start_midword():
    text = ("word " * 120) + "automatic opening break " + ("word " * 120)
    meta = DocMeta(doc_id="doc3", source="Test", title="T")
    chunks = chunk_text(text, meta, max_chars=520, overlap_chars=50)

    assert len(chunks) >= 2
    # second chunk should not start with a chopped fragment like "atic" or "ening"
    bad_prefixes = ("atic", "ening", "utom", "pen")
    assert not chunks[1].text.lstrip().startswith(bad_prefixes)