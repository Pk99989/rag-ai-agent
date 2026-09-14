"""Orchestrates: question -> LLM generates SQL -> validate -> execute (read-only,
timed) -> natural-language answer, grounded strictly in the returned rows.

RBAC note (updated -- this was stale): role-based table access IS enforced
here, via validate_sql(safe_sql, role=role) below, which was added in
Phase 7 and checks the role against config.SQL_TABLE_ACCESS. The original
version of this docstring said filtering was "not yet enforced" -- true
when it was written, false once Phase 7 landed, and left uncorrected until
now. Column-level RBAC is wired but inert (config.SQL_COLUMN_BLOCKLIST is
empty) -- see validator.py for that distinction.
"""
import re
import time

from config import GROQ_API_KEY, GROQ_MODEL
from guardrails import apply_input_guardrails
from rag_agent.sql.schema import render_schema_for_prompt
from rag_agent.sql.validator import validate_sql, SQLValidationError
from rag_agent.sql.executor import execute_sql, log_sql_query, SQLExecutionError, SQLResult, OLIST_DB_LABEL

GROUNDED_REFUSAL = (
    "I couldn't find sufficient evidence in the available company data to answer that reliably."
)

SQL_SYSTEM_PROMPT = (
    "You write exactly one read-only SQLite SELECT statement (CTEs with WITH are allowed) "
    "that answers the user's question, using ONLY the tables and columns listed below. "
    "Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, or PRAGMA. "
    "Never write more than one statement. Output ONLY the raw SQL -- no explanation, "
    "no markdown code fences, no trailing commentary.\n\n" + render_schema_for_prompt()
)

ANSWER_SYSTEM_PROMPT = (
    "You answer a business question in 1-2 sentences using ONLY the query result rows given "
    "below. State numbers exactly as given -- do not round, estimate, or add any fact not "
    "present in the rows. If the rows are empty, say the data doesn't show a matching result."
)

_CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _get_llm(temperature: float = 0.0):
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=temperature)


def generate_sql(question: str, retry_feedback: str | None = None) -> str:
    llm = _get_llm()
    human = f"Question: {question}"
    if retry_feedback:
        human += (
            f"\n\nYour previous SQL was rejected for this reason: {retry_feedback}\n"
            "Write a corrected single SELECT statement."
        )
    response = llm.invoke([("system", SQL_SYSTEM_PROMPT), ("human", human)])
    raw = response.content.strip()
    return _CODE_FENCE_RE.sub("", raw).strip()


def _format_rows_for_prompt(result: SQLResult) -> str:
    if not result.rows:
        return "(no rows returned)"
    header = " | ".join(result.columns)
    lines = [header] + [" | ".join(str(v) for v in row) for row in result.rows[:20]]
    return "\n".join(lines)


def _confidence_for_result(result: SQLResult) -> str:
    if result.row_count == 0:
        return "medium"  # query ran fine; the data just doesn't have a match
    return "high"


def _generate_nl_answer(question: str, result: SQLResult) -> str:
    llm = _get_llm()
    human = f"Question: {question}\n\nQuery result:\n{_format_rows_for_prompt(result)}"
    response = llm.invoke([("system", ANSWER_SYSTEM_PROMPT), ("human", human)])
    return response.content.strip()


def _refusal(reason: str, generation_latency_ms: float = None) -> dict:
    return {
        "answer": GROUNDED_REFUSAL,
        "route": "SQL",
        "confidence": "low",
        "data_source": OLIST_DB_LABEL,
        "sql": None,
        "row_count": 0,
        "blocked": False,
        "reason": reason,
        "generation_latency_ms": generation_latency_ms,
        "execution_time_ms": None,
    }


def run_text_to_sql(question: str, user=None, role: str = "employee") -> dict:
    """Full pipeline. `user`/`role` select which tables validate_sql() will
    authorize (see module docstring -- this IS enforced, not a placeholder)."""
    username = getattr(user, "username", str(user) if user else "unknown")

    input_check = apply_input_guardrails(question)
    if input_check["blocked"]:
        return {
            "answer": "This request was blocked by guardrails and cannot be processed.",
            "route": "SQL", "confidence": "low", "data_source": OLIST_DB_LABEL,
            "sql": None, "row_count": 0, "blocked": True, "reason": input_check["reason"],
            "generation_latency_ms": None, "execution_time_ms": None,
        }

    # generation_latency_ms tracks ONLY time spent inside LLM calls (SQL
    # generation, any retry, and the final NL-answer call) -- deliberately
    # excludes validate_sql()'s CPU-only checks and execute_sql()'s DB time,
    # which is already reported separately as execution_time_ms. An earlier
    # version of this timing wrapped the whole function body, which would
    # have double-counted DB execution time inside "generation" latency.
    llm_time = 0.0

    t0 = time.time()
    raw_sql = generate_sql(question)
    llm_time += time.time() - t0
    try:
        safe_sql = validate_sql(raw_sql, role=role)
    except SQLValidationError as exc:
        try:
            t0 = time.time()
            raw_sql_retry = generate_sql(question, retry_feedback=str(exc))
            llm_time += time.time() - t0
            safe_sql = validate_sql(raw_sql_retry, role=role)
        except SQLValidationError as exc2:
            log_sql_query(username, role, raw_sql, None, str(exc2))
            return _refusal("sql_validation_failed", round(llm_time * 1000, 2))

    try:
        result = execute_sql(safe_sql)
    except SQLExecutionError as exc:
        log_sql_query(username, role, safe_sql, None, str(exc))
        return _refusal("sql_execution_failed", round(llm_time * 1000, 2))

    log_sql_query(username, role, safe_sql, result, None)
    t0 = time.time()
    nl_answer = _generate_nl_answer(question, result)
    llm_time += time.time() - t0
    generation_latency_ms = round(llm_time * 1000, 2)

    return {
        "answer": nl_answer,
        "route": "SQL",
        "confidence": _confidence_for_result(result),
        "data_source": OLIST_DB_LABEL,
        "sql": safe_sql,  # UI layer decides whether this role may see it (Phase 7/13)
        "row_count": result.row_count,
        "execution_time_ms": result.execution_time_ms,
        "generation_latency_ms": generation_latency_ms,
        "blocked": False,
        "reason": None,
    }
