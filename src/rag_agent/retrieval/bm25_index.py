"""BM25 keyword index over the same corpus embedded into ChromaDB.

Built once (by ingest.py's build_index(), extended in Phase 6) and
persisted to disk so queries don't rebuild it from scratch every time.

RBAC note, stated plainly rather than left implicit: Chroma's vector search
is filtered by department at the database-engine level (a `where` clause),
so unauthorized embeddings are never even scored. rank_bm25 has no
metadata-filtering concept -- it is a single in-memory statistical index
over the whole corpus, so scoring necessarily touches every document's
text. The security guarantee that actually matters (unauthorized document
text never reaches the LLM or the user) still holds, because
bm25_search_all() returns the full ranked list and the CALLER (hybrid.py)
filters by the role's authorized departments before any of it is used
further -- but this is filtering-after-scoring, not filtering-before-scoring
like the vector path. If that distinction matters for a stricter compliance
requirement later, the fix is a separate BM25 index per department rather
than one global index.
"""
import pickle

from rank_bm25 import BM25Okapi

from config import BM25_INDEX_PATH


def _tokenize(text: str) -> list:
    return text.lower().split()


def build_bm25_index(ids: list, texts: list, metadatas: list) -> None:
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "texts": texts, "metadatas": metadatas}, f)


_CACHE = None


def _load_bm25_index() -> dict:
    global _CACHE
    if _CACHE is None:
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"{BM25_INDEX_PATH} not found. Run ingest.py to build it.")
        with open(BM25_INDEX_PATH, "rb") as f:
            _CACHE = pickle.load(f)
    return _CACHE


def bm25_search_all(query: str) -> list:
    """Returns EVERY document scored against the query, sorted descending
    by raw BM25 score, as (text, metadata, score) -- unfiltered by RBAC.
    Callers MUST filter by authorized department before using results."""
    data = _load_bm25_index()
    scores = data["bm25"].get_scores(_tokenize(query))
    ranked = sorted(zip(scores, data["texts"], data["metadatas"]), key=lambda x: x[0], reverse=True)
    return [(text, meta, float(score)) for score, text, meta in ranked]
