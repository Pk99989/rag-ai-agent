from ingest import chunk_text, load_documents, BASE_DIR


def test_chunk_text_respects_size():
    text = "A" * 2500
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) >= 3


def test_chunk_text_short_text_single_chunk():
    text = "short document"
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert chunks == ["short document"]


def test_chunk_text_rejects_bad_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_load_documents_parses_department_from_filename():
    docs = load_documents(BASE_DIR)
    departments = {d["department"] for d in docs}
    # These departments should exist given the shipped docs_*.md sample corpus.
    assert {"finance", "hr", "general", "executive"} <= departments
    assert all(d["text"] for d in docs)
