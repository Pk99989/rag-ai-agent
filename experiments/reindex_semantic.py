"""Build an ISOLATED semantic-chunked index for the Phase 10 experiments/
comparison. Deliberately writes to separate paths from the production
index so running this can never corrupt the live app's chroma_store or
BM25 index:

    production (untouched by this script):
        config.CHROMA_DIR            (chroma_store/)
        config.BM25_INDEX_PATH       (bm25_index.pkl)
    experimental (written here):
        experiments/chroma_store_experiments/   (collection "company_docs_semantic")
        experiments/bm25_index_experiments.pkl

This reuses ingest.load_documents() (read-only) to source the exact same
documents the production index uses, so the ONLY variable that changes
between the "hybrid_rerank" and "semantic_chunking" configs in
run_experiments.py is the chunking strategy -- not the underlying corpus.

Run:
    python experiments/reindex_semantic.py
"""
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb  # noqa: E402
from chromadb.utils import embedding_functions  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from ingest import load_documents  # noqa: E402
from semantic_chunker import semantic_chunk_text  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent
CHROMA_DIR_EXPERIMENTS = EXPERIMENTS_DIR / "chroma_store_experiments"
BM25_PATH_EXPERIMENTS = EXPERIMENTS_DIR / "bm25_index_experiments.pkl"
COLLECTION_NAME_SEMANTIC = "company_docs_semantic"


def _tokenize(text: str) -> list:
    return text.lower().split()


def build_semantic_index() -> int:
    CHROMA_DIR_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR_EXPERIMENTS))
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        client.delete_collection(COLLECTION_NAME_SEMANTIC)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME_SEMANTIC, embedding_function=ef)

    docs = load_documents(REPO_ROOT)
    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = semantic_chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['department']}::{doc['source']}::sem{i}")
            texts.append(chunk)
            meta = {"department": doc["department"], "source": doc["source"], "chunk_index": i}
            for extra_key in ("document_type", "access_level", "record_id"):
                if doc.get(extra_key):
                    meta[extra_key] = doc[extra_key]
            metadatas.append(meta)

    if texts:
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        tokenized = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(BM25_PATH_EXPERIMENTS, "wb") as f:
            pickle.dump({"bm25": bm25, "ids": ids, "texts": texts, "metadatas": metadatas}, f)

    return len(texts)


if __name__ == "__main__":
    n = build_semantic_index()
    print(f"Indexed {n} semantic chunks into '{COLLECTION_NAME_SEMANTIC}' at {CHROMA_DIR_EXPERIMENTS}")
    print(f"BM25 index written to {BM25_PATH_EXPERIMENTS}")
    print("Production chroma_store/ and bm25_index.pkl were NOT touched.")
