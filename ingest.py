"""Document ingestion: load company docs, chunk, embed, and store in ChromaDB with RBAC metadata.

Documents are flat files named `docs_<department>_<slug>.md` in this same directory
(e.g. docs_finance_financial-report-2025.md). The department segment becomes retrieval
metadata used to enforce RBAC filtering at query time (see rbac.py).
"""
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

from config import BASE_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME, DOCS_GLOB


def load_documents(base_dir: Path = BASE_DIR) -> list:
    docs = []
    for file_path in sorted(base_dir.glob(DOCS_GLOB)):
        parts = file_path.stem.split("_", 2)  # ['docs', department, slug]
        department = parts[1] if len(parts) > 1 else "general"
        text = file_path.read_text(encoding="utf-8")
        docs.append({"department": department, "source": file_path.name, "text": text})
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
            metadatas.append({"department": doc["department"], "source": doc["source"], "chunk_index": i})

    if texts:
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return len(texts)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
