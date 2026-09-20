"""Four retrieval configurations compared by run_experiments.py, all
callable as fn(query, role, top_k) -> list[(text, metadata, score)].

    baseline_vector_only  -- plain single-vector search, production index/chunking.
    hybrid_no_rerank      -- vector+BM25 fusion, no cross-encoder, production index/chunking.
    hybrid_rerank         -- fusion + cross-encoder rerank, production index/chunking
                             (this is the current live rag_chain.retrieve()).
    semantic_chunking     -- fusion + cross-encoder rerank, but against the
                             ISOLATED semantic-chunked index from
                             reindex_semantic.py, so this isolates chunking
                             strategy as the only variable relative to
                             hybrid_rerank.

baseline/hybrid_no_rerank/hybrid_rerank all read the SAME production
chroma_store/ + bm25_index.pkl -- they only vary the retrieval algorithm.
Only semantic_chunking points at a different index. This is what makes the
comparison mean "algorithm A vs algorithm B on the same data" and "chunking
A vs chunking B holding the algorithm fixed" rather than conflating both.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb  # noqa: E402
from chromadb.utils import embedding_functions  # noqa: E402

from config import (COLLECTION_NAME, CHROMA_DIR, HYBRID_CANDIDATES, VECTOR_WEIGHT,  # noqa: E402
                     BM25_WEIGHT, SIMILARITY_THRESHOLD, RERANK_TOP_K)
from rbac import build_chroma_filter, is_authorized  # noqa: E402
from rag_agent.retrieval.bm25_index import bm25_search_all  # noqa: E402
from rag_agent.retrieval.reranker import rerank as cross_encoder_rerank  # noqa: E402

from reindex_semantic import CHROMA_DIR_EXPERIMENTS, BM25_PATH_EXPERIMENTS, COLLECTION_NAME_SEMANTIC  # noqa: E402

_ef = embedding_functions.DefaultEmbeddingFunction()


def _get_collection(persist_dir: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=collection_name, embedding_function=_ef)


def _vector_search(query: str, role: str, top_k: int, persist_dir: Path, collection_name: str) -> list:
    collection = _get_collection(persist_dir, collection_name)
    where = build_chroma_filter(role)
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    sims = [1.0 / (1.0 + d) for d in dists]
    return list(zip(docs, metas, sims))


def _normalize(scores: list) -> list:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _pool_key(meta: dict, text: str) -> tuple:
    return (meta.get("source"), meta.get("chunk_index"), text[:50])


def _bm25_search_experimental(query: str) -> list:
    """Same shape/behavior as rag_agent.retrieval.bm25_index.bm25_search_all,
    but reads the ISOLATED experimental BM25 pickle instead of the
    production one -- kept as a separate tiny function rather than adding a
    path parameter to the production module, to avoid touching a file the
    live app depends on for the sake of an experiment script."""
    import pickle
    with open(BM25_PATH_EXPERIMENTS, "rb") as f:
        data = pickle.load(f)
    scores = data["bm25"].get_scores(query.lower().split())
    ranked = sorted(zip(scores, data["texts"], data["metadatas"]), key=lambda x: x[0], reverse=True)
    return [(text, meta, float(score)) for score, text, meta in ranked]


def _fuse(query: str, role: str, persist_dir: Path, collection_name: str, bm25_fn) -> list:
    vector_hits = _vector_search(query, role, HYBRID_CANDIDATES, persist_dir, collection_name)
    bm25_hits_all = bm25_fn(query)
    bm25_hits = [
        (text, meta, score) for text, meta, score in bm25_hits_all
        if is_authorized(role, meta.get("department", ""))
    ][:HYBRID_CANDIDATES]

    pool: dict = {}
    if vector_hits:
        v_norm = _normalize([h[2] for h in vector_hits])
        for (text, meta, _), norm in zip(vector_hits, v_norm):
            pool[_pool_key(meta, text)] = {"text": text, "meta": meta, "vector": norm, "bm25": 0.0}
    if bm25_hits:
        b_norm = _normalize([h[2] for h in bm25_hits])
        for (text, meta, _), norm in zip(bm25_hits, b_norm):
            key = _pool_key(meta, text)
            if key in pool:
                pool[key]["bm25"] = norm
            else:
                pool[key] = {"text": text, "meta": meta, "vector": 0.0, "bm25": norm}

    fused = []
    for entry in pool.values():
        fused_score = VECTOR_WEIGHT * entry["vector"] + BM25_WEIGHT * entry["bm25"]
        if fused_score >= SIMILARITY_THRESHOLD:
            fused.append((entry["text"], entry["meta"], fused_score))
    fused.sort(key=lambda x: x[2], reverse=True)
    return fused[:HYBRID_CANDIDATES]


def baseline_vector_only(query: str, role: str, top_k: int = RERANK_TOP_K) -> list:
    return _vector_search(query, role, top_k, CHROMA_DIR, COLLECTION_NAME)


def hybrid_no_rerank(query: str, role: str, top_k: int = RERANK_TOP_K) -> list:
    fused = _fuse(query, role, CHROMA_DIR, COLLECTION_NAME, bm25_search_all)
    return fused[:top_k]


def hybrid_rerank(query: str, role: str, top_k: int = RERANK_TOP_K) -> list:
    fused = _fuse(query, role, CHROMA_DIR, COLLECTION_NAME, bm25_search_all)
    return cross_encoder_rerank(query, fused, top_k)


def semantic_chunking(query: str, role: str, top_k: int = RERANK_TOP_K) -> list:
    if not CHROMA_DIR_EXPERIMENTS.exists() or not BM25_PATH_EXPERIMENTS.exists():
        raise FileNotFoundError(
            "Experimental semantic index not found -- run "
            "experiments/reindex_semantic.py first."
        )
    fused = _fuse(query, role, CHROMA_DIR_EXPERIMENTS, COLLECTION_NAME_SEMANTIC, _bm25_search_experimental)
    return cross_encoder_rerank(query, fused, top_k)


CONFIGS = {
    "baseline_vector_only": baseline_vector_only,
    "hybrid_no_rerank": hybrid_no_rerank,
    "hybrid_rerank": hybrid_rerank,
    "semantic_chunking": semantic_chunking,
}
