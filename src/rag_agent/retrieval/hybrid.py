"""Hybrid retrieval orchestrator: vector search + BM25, score-fused by
configurable weights, then cross-encoder reranked. RBAC is enforced on
both paths before anything is returned -- see bm25_index.py's docstring
for the one asymmetry between the two paths (vector is filtered before
scoring at the DB engine level; BM25 is filtered after scoring, before
returning) and why the actual leak-prevention guarantee still holds either way.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for config/ingest/rbac

from config import (COLLECTION_NAME, CHROMA_DIR, HYBRID_CANDIDATES, VECTOR_WEIGHT,
                     BM25_WEIGHT, SIMILARITY_THRESHOLD, RERANK_TOP_K)
from ingest import get_chroma_client, get_embedding_function
from rbac import build_chroma_filter, is_authorized
from rag_agent.retrieval.bm25_index import bm25_search_all
from rag_agent.retrieval.reranker import rerank as cross_encoder_rerank


def _vector_search(query: str, role: str, top_k: int) -> list:
    client = get_chroma_client(CHROMA_DIR)
    ef = get_embedding_function()
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    where = build_chroma_filter(role)
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    # Chroma returns a distance (lower = closer, unbounded above) -- turn it
    # into a similarity-like score in (0,1] for fusion with BM25.
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


def _fuse_candidates(query: str, role: str) -> list:
    """Vector + BM25 fusion only, no reranking -- factored out so
    hybrid_retrieve_with_timing (Phase 11) can measure fusion and reranking
    as two separate real durations instead of one opaque call."""
    vector_hits = _vector_search(query, role, HYBRID_CANDIDATES)

    bm25_hits_all = bm25_search_all(query)
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


def hybrid_retrieve_with_timing(query: str, role: str, top_k: int = RERANK_TOP_K) -> tuple:
    """Same result as hybrid_retrieve(), plus a real timing breakdown:
    {"retrieval_ms": <vector+BM25+fusion>, "reranking_ms": <cross-encoder>}.
    Added for Phase 11 structured monitoring, which asks for retrieval and
    reranking latency as separate fields -- measured at the one real
    boundary that already existed in this pipeline (fusion output ->
    reranker input), not estimated or split arbitrarily."""
    t0 = time.time()
    candidates = _fuse_candidates(query, role)
    t1 = time.time()
    results = cross_encoder_rerank(query, candidates, top_k)
    t2 = time.time()
    timing = {"retrieval_ms": round((t1 - t0) * 1000, 2), "reranking_ms": round((t2 - t1) * 1000, 2)}
    return results, timing


def hybrid_retrieve(query: str, role: str, top_k: int = RERANK_TOP_K) -> list:
    """Returns up to top_k (text, metadata, rerank_score) tuples, RBAC-filtered
    and cross-encoder reranked. Unchanged contract -- existing callers
    (rag_chain.retrieve, eval_run.py, evaluation/, experiments/) are
    unaffected by hybrid_retrieve_with_timing's addition above; this is now
    a thin wrapper over it that discards the timing info."""
    results, _timing = hybrid_retrieve_with_timing(query, role, top_k)
    return results
