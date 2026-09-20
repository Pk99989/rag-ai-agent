"""JSONL structured request log -- one real record per request, written by
src/rag_agent/router/dispatch.py (the single funnel point for both RAG and
SQL requests as of Phase 11).

Field list (matches the Phase 11 spec):
    request_id, timestamp, user_role, query_route, route_confidence,
    retrieval_latency_ms, reranking_latency_ms, generation_latency_ms,
    sql_execution_time_ms, total_latency_ms, input_tokens, output_tokens,
    estimated_cost_usd, model, retrieved_documents, guardrail_result,
    confidence

Fields that don't apply to a given route are logged as null rather than
omitted, so every line has the same schema and a dashboard/consumer never
has to guess whether a missing key means "zero" or "not applicable."

Privacy note: `question` and `answer_preview` are stored PII-redacted
(guardrails.redact_pii) -- this is an operational log meant to be read by
whoever runs the dashboard, not a place to accumulate raw user PII
indefinitely.
"""
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from config import LOG_DIR  # noqa: E402
from guardrails import redact_pii  # noqa: E402

STRUCTURED_LOG_PATH = LOG_DIR / "structured_log.jsonl"


def log_structured_interaction(
    *,
    user_role: str,
    query_route: str,
    route_confidence: float,
    question: str,
    answer: str,
    model: str,
    total_latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    retrieval_latency_ms: float = None,
    reranking_latency_ms: float = None,
    generation_latency_ms: float = None,
    sql_execution_time_ms: float = None,
    vision_latency_ms: float = None,
    retrieved_documents: list = None,
    guardrail_result: dict = None,
    confidence: str = None,
) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "request_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_role": user_role,
        "query_route": query_route,
        "route_confidence": route_confidence,
        "question": redact_pii(question),
        "answer_preview": redact_pii(answer)[:200] if answer else "",
        "model": model,
        "retrieval_latency_ms": retrieval_latency_ms,
        "reranking_latency_ms": reranking_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "sql_execution_time_ms": sql_execution_time_ms,
        # Phase 16: time spent in the vision-LLM field-extraction call for
        # document Q&A requests. Null for every RAG/SQL route, same "null
        # means not applicable, not omitted" convention as the other
        # per-route latency fields above.
        "vision_latency_ms": vision_latency_ms,
        "total_latency_ms": round(total_latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "retrieved_documents": retrieved_documents or [],
        "guardrail_result": guardrail_result or {"blocked": False, "reason": None},
        "confidence": confidence,
    }
    with open(STRUCTURED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_structured_log(limit: int = None) -> list:
    """Returns records oldest-first. Used by dashboard.py. Tolerates a
    missing file (no requests logged yet) and skips any line that fails to
    parse as JSON rather than crashing the whole read (a partially-written
    last line from a killed process is a real possibility for an
    append-only log, not a hypothetical one)."""
    if not STRUCTURED_LOG_PATH.exists():
        return []
    records = []
    with open(STRUCTURED_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit:
        records = records[-limit:]
    return records
