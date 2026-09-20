"""Document ingestion: load Olist business-report docs, chunk, embed, and store in
ChromaDB with RBAC metadata.

Olist-only pivot: this used to also load a legacy set of flat fictional-company
files named `docs_<department>_<slug>.md` (the AtliQ documents). Those files were
deleted and that loading branch removed -- the only document source now is the
generated Olist business-report set in RAG_DOCS_DIR (Phase 5), department + richer
metadata parsed from each file's frontmatter (see rbac.py for how department maps
to RBAC access).
"""
import sys
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

from config import BASE_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, RAG_DOCS_DIR

sys.path.insert(0, str(BASE_DIR / "src"))
from rag_agent.retrieval.bm25_index import build_bm25_index


def _parse_frontmatter(text: str) -> tuple:
    """Parses a simple '---\\nkey: value\\n---\\nbody' block, as written by
    scripts/generate_olist_rag_docs.py. Returns (metadata_dict, body_text).
    If the file has no frontmatter block, returns ({}, text) unchanged --
    so this is safe to call on any .md file, not just generated ones."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    _, fm_block, body = parts
    meta = {}
    for line in fm_block.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def load_documents(base_dir: Path = BASE_DIR) -> list:
    """Loads the generated Olist business-report docs in RAG_DOCS_DIR (Phase 5),
    department + richer metadata parsed from each file's frontmatter. Each doc
    has the shape {"department", "source", "text", "document_type",
    "access_level", "record_id"}, which build_index() and RBAC filtering
    downstream consume directly.
    """
    docs = []
    if RAG_DOCS_DIR.exists():
        for file_path in sorted(RAG_DOCS_DIR.glob("*.md")):
            raw = file_path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(raw)
            docs.append({
                "department": meta.get("department", "general"),
                "source": file_path.name,
                "text": body,
                "document_type": meta.get("document_type"),
                "access_level": meta.get("access_level"),
                "record_id": meta.get("record_id"),
            })
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Simple character-based sliding-window chunker (no extra NLP dependency required)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def get_chroma_client(persist_dir: Path = CHROMA_DIR):
    persist_dir.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_embedding_function():
    # Lightweight ONNX MiniLM embedding bundled with chromadb — no torch/sentence-transformers needed.
    return embedding_functions.DefaultEmbeddingFunction()


def build_index(base_dir: Path = BASE_DIR, persist_dir: Path = CHROMA_DIR, reset: bool = True) -> int:
    """Ingest all documents into the Chroma collection. Returns the number of chunks indexed."""
    client = get_chroma_client(persist_dir)
    ef = get_embedding_function()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

    docs = load_documents(base_dir)
    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['department']}::{doc['source']}::{i}")
            texts.append(chunk)
            meta = {"department": doc["department"], "source": doc["source"], "chunk_index": i}
            for extra_key in ("document_type", "access_level", "record_id"):
                if doc.get(extra_key):
                    meta[extra_key] = doc[extra_key]
            metadatas.append(meta)

    if texts:
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        build_bm25_index(ids, texts, metadatas)  # Phase 6: keyword index over the same corpus
    return len(texts)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
