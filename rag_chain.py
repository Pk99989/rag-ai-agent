"""Core RAG chain: RBAC-filtered retrieval -> guardrails -> LLM generation -> monitoring.

Phase 9 change: retrieve() now calls the Phase 6 hybrid (vector+BM25) +
cross-encoder-reranked pipeline (rag_agent.retrieval.hybrid.hybrid_retrieve)
instead of plain single-vector search. That pipeline's own module docstring
said explicitly "this is the function rag_chain.py's retrieve() should call
once this phase is wired in" -- it had been built and smoke-tested (Phase 6)
but never actually used by the live chat path or eval_run.py until now. This
is also the first time the ONNX cross-encoder is exercised from these code
paths specifically (smoke_test_hybrid.py exercised it directly before), so
the first real query after this change will trigger a Hugging Face Hub
download of the reranker model if it isn't already cached locally.

retrieve()'s return shape changed from (text, metadata) pairs to
(text, metadata, rerank_score) triples -- eval_run.py's two call sites were
updated to match (see that file); anything else importing retrieve()
directly needs the same third-value update.
"""
import time
from pathlib import Path
import sys

from config import GROQ_API_KEY, GROQ_MODEL, RERANK_TOP_K
from guardrails import (
    apply_input_guardrails, apply_output_guardrails, looks_out_of_scope,
    scan_context_for_injection,
)
from monitoring import log_interaction

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag_agent.retrieval.hybrid import hybrid_retrieve, hybrid_retrieve_with_timing

SYSTEM_PROMPT = (
    "You are an e-commerce analytics assistant for Olist, a Brazilian online "
    "marketplace. Answer the user's question ONLY using the provided context "
    "extracted from Olist business-report documents (aggregated from real order, "
    "payment, seller, product, and review data). If the answer is not contained in "
    "the context, say you don't have that information in the knowledge base -- do not use "
    "outside knowledge. Be concise and cite the source file names you used."
)

# Cross-encoder relevance-score cutoffs used to bucket RAG confidence.
# These are heuristic, not calibrated against a labeled eval set -- chosen
# from the score spread actually observed in Phase 6's own smoke test
# (0.9993 for a genuinely relevant passage vs 0.0000 for an irrelevant one).
# Phase 10's real eval dataset is what will validate or retune these; treat
# them as a starting point stated honestly, not a tuned threshold.
RAG_CONFIDENCE_HIGH = 0.7
RAG_CONFIDENCE_MEDIUM = 0.35


def retrieve(query: str, role: str, top_k: int = RERANK_TOP_K):
    """RBAC-filtered hybrid retrieval + cross-encoder rerank. Returns up to
    top_k (text, metadata, rerank_score) triples, highest score first."""
    return hybrid_retrieve(query, role, top_k=top_k)


def _confidence_for_chunks(chunks: list) -> str:
    """High/Medium/Low bucket from the best reranker score among the chunks
    actually used for the answer -- see RAG_CONFIDENCE_* comment above for
    what these cutoffs are and aren't."""
    if not chunks:
        return "low"
    top_score = max(score for _, _, score in chunks)
    if top_score >= RAG_CONFIDENCE_HIGH:
        return "high"
    if top_score >= RAG_CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def _citations_for_chunks(chunks: list) -> list:
    """One citation per unique source, keeping that source's best-scoring
    chunk (a document can contribute more than one chunk to the same
    answer). Sorted by score descending so the UI can show the strongest
    evidence first."""
    best_by_source = {}
    for text, meta, score in chunks:
        source = meta.get("source", "unknown")
        if source not in best_by_source or score > best_by_source[source]["score"]:
            best_by_source[source] = {
                "source": source,
                "department": meta.get("department"),
                "score": round(float(score), 4),
                "chunk_index": meta.get("chunk_index"),
                "snippet": text[:200],
            }
    return sorted(best_by_source.values(), key=lambda c: c["score"], reverse=True)


def _get_llm():
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def answer_query(query: str, user, role: str) -> dict:
    """Full pipeline: guardrails -> RBAC retrieval -> LLM -> guardrails -> monitoring log."""
    start = time.time()

    input_check = apply_input_guardrails(query)
    if input_check["blocked"]:
        result = {
            "answer": "This request was blocked by guardrails and cannot be processed.",
            "sources": [],
            "citations": [],
            "confidence": "low",
            "blocked": True,
            "reason": input_check["reason"],
            "retrieval_latency_ms": None,
            "reranking_latency_ms": None,
            "generation_latency_ms": None,
        }
        log_interaction(user, role, query, result, (time.time() - start) * 1000, 0, 0)
        return result

    chunks, retrieval_timing = hybrid_retrieve_with_timing(query, role)

    # Phase 8: a retrieved document can itself carry an injected instruction
    # aimed at the LLM ("ignore previous instructions...") rather than at
    # the user's query, which the old input-only guardrail never saw.
    # Exclude any flagged chunk from context rather than trusting it; if
    # everything gets excluded, fall through to the same out-of-scope path
    # used for "nothing retrieved" rather than answering from nothing.
    chunks, flagged_sources = scan_context_for_injection(chunks)

    if looks_out_of_scope(query, chunks, score_threshold=RAG_CONFIDENCE_MEDIUM):
        result = {
            "answer": "I don't have information about that in the company knowledge base you're "
                      "authorized to access.",
            "sources": [],
            "citations": [],
            "confidence": "low",
            "blocked": False,
            "reason": "context_flagged_as_injection" if flagged_sources else "out_of_scope_or_no_context",
            "retrieval_latency_ms": retrieval_timing["retrieval_ms"],
            "reranking_latency_ms": retrieval_timing["reranking_ms"],
            "generation_latency_ms": None,
        }
        log_interaction(user, role, query, result, (time.time() - start) * 1000, 0, 0)
        return result

    context_text = "\n\n".join(f"[Source: {m['source']}]\n{d}" for d, m, _ in chunks)
    llm = _get_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context_text}\n\nQuestion: {query}"),
    ]
    gen_start = time.time()
    response = llm.invoke(messages)
    generation_latency_ms = round((time.time() - gen_start) * 1000, 2)
    raw_answer = response.content

    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    output_check = apply_output_guardrails(raw_answer, system_prompt=SYSTEM_PROMPT)

    if output_check["blocked"]:
        reason = "output_guardrail_" + "_and_".join(output_check["flags"])
    elif output_check["redacted"]:
        reason = "pii_redacted_in_output"
    elif flagged_sources:
        reason = "context_partially_excluded_as_injection"
    else:
        reason = None

    citations = _citations_for_chunks(chunks)
    result = {
        "answer": output_check["answer"],
        "sources": [] if output_check["blocked"] else sorted({m["source"] for _, m, _ in chunks}),
        "citations": [] if output_check["blocked"] else citations,
        "confidence": "low" if output_check["blocked"] else _confidence_for_chunks(chunks),
        "blocked": output_check["blocked"],
        "reason": reason,
        "retrieval_latency_ms": retrieval_timing["retrieval_ms"],
        "reranking_latency_ms": retrieval_timing["reranking_ms"],
        "generation_latency_ms": generation_latency_ms,
    }

    log_interaction(user, role, query, result, (time.time() - start) * 1000, tokens_in, tokens_out)
    return result
