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


def test_load_documents_parses_department_from_frontmatter():
    # Olist-only pivot: load_documents() no longer reads legacy
    # docs_<department>_<slug>.md files (department-from-filename); it only
    # loads the generated Olist RAG_DOCS_DIR corpus, department parsed from
    # each file's frontmatter. Requires RAG_DOCS_DIR to have been populated
    # by scripts/generate_olist_rag_docs.py -- if that hasn't been run yet
    # in this environment, docs will be empty and this test is skipped
    # rather than asserting a false negative.
    import pytest
    from config import RAG_DOCS_DIR, ALL_DEPARTMENTS

    docs = load_documents(BASE_DIR)
    if not docs:
        pytest.skip("RAG_DOCS_DIR has no generated documents yet -- run "
                     "scripts/generate_olist_rag_docs.py first")
    departments = {d["department"] for d in docs}
    # Every department seen must be one of the real Olist departments
    # config.py's RBAC dictionaries know about -- no leftover AtliQ tags.
    assert departments <= ALL_DEPARTMENTS
    assert all(d["text"] for d in docs)
