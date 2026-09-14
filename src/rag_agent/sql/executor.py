"""Read-only, timed execution of already-validated SQL against data/olist.db.

Callers MUST pass SQL that has already been through
rag_agent.sql.validator.validate_sql() -- this module does not re-validate,
it only adds execution-time defenses: a read-only connection (so even a
validator bug cannot write to the database), a wall-clock query timeout via
SQLite's progress handler, and a friendly database label so the real
filesystem path never reaches an end user.
"""
import csv
import datetime as dt
import sqlite3
import time
from dataclasses import dataclass, field

from config import OLIST_DB_PATH, SQL_MAX_ROWS, SQL_TIMEOUT_SECONDS, SQL_QUERY_LOG_PATH
from guardrails import redact_pii

OLIST_DB_LABEL = "Olist Analytics Database"


class SQLExecutionError(Exception):
    """Raised on execution failure. Message is safe to log, NOT safe to show
    an end user verbatim (may reference internal schema/error details)."""


@dataclass
class SQLResult:
    columns: list[str]
    rows: list[tuple]
    row_count: int
    execution_time_ms: float
    truncated: bool = field(default=False)


def _make_timeout_handler(deadline: float):
    def handler():
        return time.monotonic() > deadline
    return handler


def execute_sql(sql: str, timeout_seconds: float = SQL_TIMEOUT_SECONDS) -> SQLResult:
    if not OLIST_DB_PATH.exists():
        raise SQLExecutionError(
            f"{OLIST_DB_LABEL} has not been built yet. Run scripts/prepare_olist.py first."
        )

    uri = f"file:{OLIST_DB_PATH.as_posix()}?mode=ro"
    start = time.monotonic()
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    except sqlite3.OperationalError as exc:
        raise SQLExecutionError(f"could not open {OLIST_DB_LABEL} read-only") from exc

    try:
        conn.execute("PRAGMA query_only = ON;")  # second guard beyond the read-only URI
        deadline = time.monotonic() + timeout_seconds
        conn.set_progress_handler(_make_timeout_handler(deadline), 1000)

        cursor = conn.execute(sql)
        rows = cursor.fetchmany(SQL_MAX_ROWS + 1)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        truncated = len(rows) > SQL_MAX_ROWS
        if truncated:
            rows = rows[:SQL_MAX_ROWS]
        elapsed_ms = (time.monotonic() - start) * 1000
        return SQLResult(columns=columns, rows=rows, row_count=len(rows),
                          execution_time_ms=round(elapsed_ms, 1), truncated=truncated)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise SQLExecutionError(f"query exceeded the {timeout_seconds}s timeout") from exc
        raise SQLExecutionError("query failed to execute against the schema") from exc
    finally:
        conn.close()


def _ensure_log_file() -> None:
    if not SQL_QUERY_LOG_PATH.exists():
        with open(SQL_QUERY_LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "username", "role", "sql", "success", "row_count",
                 "execution_time_ms", "error"]
            )


def log_sql_query(username: str, role: str, sql: str, result: SQLResult | None,
                   error: str | None) -> None:
    """Logs the query safely: PII-redacted, and never includes the database
    file path or a raw traceback -- only the already-sanitized error string
    text callers pass in (see text_to_sql.py)."""
    _ensure_log_file()
    with open(SQL_QUERY_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            dt.datetime.utcnow().isoformat(), username, role, redact_pii(sql),
            error is None,
            result.row_count if result else 0,
            result.execution_time_ms if result else 0,
            error or "",
        ])
