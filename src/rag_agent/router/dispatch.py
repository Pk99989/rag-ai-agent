"""Phase 11: the single real entrypoint that wires the Phase 4 router
(classify()) to the RAG (rag_chain.answer_query) and SQL
(text_to_sql.run_text_to_sql) pipelines, and writes one structured
monitoring record per request.

Before this module existed, classify() was built and smoke-tested (Phase 4)
but nothing in the live app ever called it -- app.py hardcoded
answer_query() regardless of what the question actually was. That gap was
found and named explicitly during Phase 11 planning, not discovered by
accident; this module is the fix. handle_query() below is what app.py
should call from now on instead of importing answer_query directly.

Route handling:
    RAG          -> rag_chain.answer_query()
    SQL          -> text_to_sql.run_text_to_sql()
    OUT_OF_SCOPE -> immediate refusal, no LLM call for the answer itself
                    (still runs input guardrails -- an out-of-scope-
                    classified question can still be a security probe)
    HYBRID       -> stated interpretation, not an obvious spec answer: try
                    RAG first; only if RAG's own result says it found
                    nothing relevant (reason is one of the out-of-scope
                    reasons) does this fall back to attempting SQL. This
                    avoids two LLM calls on the common case where RAG's
                    router-independent out-of-scope detection (Phase 10 fix)
                    already agrees the question is answerable from documents.

Every route returns a common envelope (see _normalize_*) so a UI or API
layer doesn't need to know which pipeline actually ran.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src

from config import GROQ_MODEL  # noqa: E402
from guardrails import apply_input_guardrails  # noqa: E402
from rag_chain import answer_query  # noqa: E402
from rag_agent.sql.text_to_sql import run_text_to_sql  # noqa: E402
from rag_agent.router.classifier import classify, RAG, SQL, HYBRID, OUT_OF_SCOPE  # noqa: E402
from rag_agent.monitoring.structured_logger import log_structured_interaction  # noqa: E402

_OUT_OF_SCOPE_REASONS = {"out_of_scope_or_no_context", "context_flagged_as_injection"}


def _normalize_rag_result(result: dict) -> dict:
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "citations": result.get("citations", []),
        "sql": None,
        "row_count": None,
        "confidence": result.get("confidence"),
        "blocked": result.get("blocked", False),
        "reason": result.get("reason"),
        "retrieval_latency_ms": result.get("retrieval_latency_ms"),
        "reranking_latency_ms": result.get("reranking_latency_ms"),
        "generation_latency_ms": result.get("generation_latency_ms"),
        "sql_execution_time_ms": None,
    }


def _normalize_sql_result(result: dict) -> dict:
    return {
        "answer": result["answer"],
        "sources": [],
        "citations": [],
        "sql": result.get("sql"),
        "row_count": result.get("row_count"),
        "confidence": result.get("confidence"),
        "blocked": result.get("blocked", False),
        "reason": result.get("reason"),
        "retrieval_latency_ms": None,
        "reranking_latency_ms": None,
        "generation_latency_ms": result.get("generation_latency_ms"),
        "sql_execution_time_ms": result.get("execution_time_ms"),
    }


def handle_query(question: str, user, role: str) -> dict:
    start = time.time()
    route_result = classify(question)
    route = route_result.route

    if route == OUT_OF_SCOPE:
        input_check = apply_input_guardrails(question)
        envelope = {
            "answer": "This request was blocked by guardrails and cannot be processed."
                      if input_check["blocked"] else
                      "This doesn't look like a question about company documents or data -- "
                      "I can only help with those.",
            "sources": [], "citations": [], "sql": None, "row_count": None,
            "confidence": "low", "blocked": input_check["blocked"],
            "reason": input_check["reason"] if input_check["blocked"] else "router_out_of_scope",
            "retrieval_latency_ms": None, "reranking_latency_ms": None,
            "generation_latency_ms": None, "sql_execution_time_ms": None,
        }
    elif route == SQL:
        envelope = _normalize_sql_result(run_text_to_sql(question, user=user, role=role))
    elif route == RAG:
        envelope = _normalize_rag_result(answer_query(question, user=user, role=role))
    else:  # HYBRID
        rag_result = answer_query(question, user=user, role=role)
        if rag_result.get("reason") in _OUT_OF_SCOPE_REASONS:
            sql_result = run_text_to_sql(question, user=user, role=role)
            envelope = _normalize_sql_result(sql_result)
            envelope["reason"] = envelope["reason"] or "hybrid_fell_back_to_sql"
        else:
            envelope = _normalize_rag_result(rag_result)

    total_latency_ms = (time.time() - start) * 1000
    username = getattr(user, "username", str(user) if user else "unknown")
    guardrail_result = {"blocked": envelope["blocked"], "reason": envelope["reason"]}
    retrieved_documents = envelope["sources"] or [c.get("source") for c in envelope["citations"]]

    log_structured_interaction(
        user_role=role,
        query_route=route,
        route_confidence=route_result.confidence,
        question=question,
        answer=envelope["answer"],
        model=GROQ_MODEL,
        total_latency_ms=total_latency_ms,
        retrieval_latency_ms=envelope["retrieval_latency_ms"],
        reranking_latency_ms=envelope["reranking_latency_ms"],
        generation_latency_ms=envelope["generation_latency_ms"],
        sql_execution_time_ms=envelope["sql_execution_time_ms"],
        retrieved_documents=retrieved_documents,
        guardrail_result=guardrail_result,
        confidence=envelope["confidence"],
    )

    envelope["route"] = route
    envelope["route_confidence"] = route_result.confidence
    envelope["username"] = username
    return envelope
