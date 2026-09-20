"""Security gate for LLM-generated SQL. Nothing reaches the database without
passing through validate_sql() first -- rag_chain-style, this is enforced in
code, not by asking the LLM nicely.

Honest scope statement: the table/column allowlist checks below are a
sanity check ("does this reference our real schema"), not a from-scratch SQL
parser with full alias resolution -- a query that aliases a table to a
single letter and references a column through it will usually resolve
correctly here, but this is not guaranteed for arbitrarily nested queries.
The properties that ARE guaranteed regardless of parsing edge cases:
  - exactly one statement, and it starts with SELECT or WITH (CTE)
  - none of the blocked DDL/DML keywords appear anywhere in the text
  - execution happens on a read-only SQLite connection (see executor.py),
    so even a validator bypass cannot write to the database
  - every result set is capped at SQL_MAX_ROWS
Defense in depth: several of these overlap on purpose.
"""
import re

import sqlparse

from config import SQL_MAX_ROWS, SQL_TABLE_ACCESS, SQL_COLUMN_BLOCKLIST
from rag_agent.sql.schema import ALLOWED_TABLES, ALLOWED_COLUMNS


class SQLValidationError(Exception):
    """Raised when generated SQL fails a security or schema check. The
    message is safe to log; callers should show the user a generic refusal,
    not this message verbatim, to avoid leaking schema/query internals to
    an unauthorized user (see rag_agent.sql.text_to_sql)."""


FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH",
    "DETACH", "REPLACE", "PRAGMA", "VACUUM", "REINDEX", "TRIGGER",
    "GRANT", "REVOKE", "TRANSACTION", "COMMIT", "ROLLBACK",
]
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)

# Function/keyword names that are legitimately bare identifiers in a SELECT
# and must not be mistaken for an unknown column reference.
_SQL_SAFE_WORDS = {
    "select", "from", "where", "group", "by", "order", "limit", "as", "and",
    "or", "not", "in", "is", "null", "like", "between", "asc", "desc",
    "join", "left", "right", "inner", "outer", "on", "count", "sum", "avg",
    "min", "max", "distinct", "having", "case", "when", "then", "else",
    "end", "with", "cast", "coalesce", "round", "strftime", "date", "text",
    "real", "integer", "over", "partition", "star",
}

_TABLE_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)


def _check_single_select_statement(sql: str) -> str:
    statements = [s for s in sqlparse.parse(sql) if s.token_first(skip_cm=True) is not None]
    if len(statements) != 1:
        raise SQLValidationError(f"expected exactly 1 SQL statement, found {len(statements)}")
    stmt = statements[0]

    stmt_type = stmt.get_type()
    first_token = stmt.token_first(skip_cm=True)
    starts_with_with = first_token is not None and first_token.ttype is not None and \
        first_token.normalized.upper() == "WITH"

    if stmt_type != "SELECT" and not starts_with_with:
        raise SQLValidationError(
            f"only SELECT statements are allowed (parsed statement type: {stmt_type})"
        )
    return str(stmt).strip()


def _check_no_forbidden_keywords(sql: str) -> None:
    match = _FORBIDDEN_RE.search(sql)
    if match:
        raise SQLValidationError(f"forbidden keyword in generated SQL: {match.group(1).upper()}")


def _check_no_comments_or_stacked_statements(sql: str) -> None:
    if "--" in sql or "/*" in sql:
        raise SQLValidationError("SQL comments are not allowed (may hide stacked statements)")
    body = sql.strip()
    if body.endswith(";"):
        body = body[:-1]
    if ";" in body:
        raise SQLValidationError("multiple SQL statements are not allowed")


def _referenced_tables(sql: str) -> set:
    return {m.group(1).lower() for m in _TABLE_PATTERN.finditer(sql)}


def _check_referenced_tables(sql: str) -> None:
    referenced = _referenced_tables(sql)
    unknown = referenced - ALLOWED_TABLES
    if unknown:
        raise SQLValidationError(f"query references unknown table(s): {sorted(unknown)}")


def _check_role_authorized_tables(sql: str, role: str) -> None:
    """RBAC for SQL (Phase 7): a table existing in the schema (checked above)
    is not the same as this ROLE being allowed to query it. Fails safe --
    an unrecognized role gets an empty allowed-set, i.e. denied by default,
    never allowed by default."""
    referenced = _referenced_tables(sql)
    allowed = SQL_TABLE_ACCESS.get(role, set())
    unauthorized = referenced - allowed
    if unauthorized:
        raise SQLValidationError(
            f"role {role!r} is not authorized to query table(s): {sorted(unauthorized)}"
        )


def _check_role_column_blocklist(sql: str, role: str) -> None:
    """Column-level RBAC hook (Phase 7). SQL_COLUMN_BLOCKLIST is empty by
    default (see config.py) -- this only does something once a real
    table->blocked-columns rule is added there. A wildcard SELECT * against
    a table that has ANY blocked column for this role is rejected outright,
    since we cannot otherwise guarantee a blocked column isn't included."""
    if not SQL_COLUMN_BLOCKLIST:
        return
    referenced_tables = _referenced_tables(sql)
    has_star = bool(re.search(r"SELECT\s+\*", sql, re.IGNORECASE))
    for table in referenced_tables:
        blocked = SQL_COLUMN_BLOCKLIST.get(table, set())
        if not blocked:
            continue
        if has_star:
            raise SQLValidationError(
                f"SELECT * on {table!r} is not allowed for role {role!r} "
                f"(table has role-restricted column(s))"
            )
        for col in blocked:
            if re.search(rf"\b{re.escape(col)}\b", sql, re.IGNORECASE):
                raise SQLValidationError(
                    f"role {role!r} is not authorized to select column {col!r} on {table!r}"
                )


_AS_ALIAS_PATTERN = re.compile(r"\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_TABLE_ALIAS_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+[a-zA-Z_][a-zA-Z0-9_]*\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)


def _collect_query_defined_aliases(sql: str) -> set[str]:
    """Column/table aliases the query itself DEFINES (`AS revenue`, or an
    implicit table alias like `FROM orders o`) are not schema references and
    must not be flagged as unrecognized identifiers -- they only exist
    because the query just created them."""
    aliases = {m.group(1).lower() for m in _AS_ALIAS_PATTERN.finditer(sql)}
    aliases |= {m.group(1).lower() for m in _TABLE_ALIAS_PATTERN.finditer(sql)}
    # a table-alias regex match can accidentally capture the next keyword
    # (e.g. "FROM orders WHERE") -- keywords are already in _SQL_SAFE_WORDS
    # so harmless to include them here too.
    return aliases


_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'")


def _mask_string_literals(sql: str) -> str:
    """Replace single-quoted string literal contents with a neutral, empty
    literal before scanning for identifiers. A filter VALUE like
    'delivered', 'credit_card', or 'on_time' is data, not a column
    reference, and must not be tokenized as one.

    Found by execution, not by hand-tracing: 5 of the 20 real Phase 10 SQL
    eval questions were rejected by validate_sql even though the generated
    SQL was correct, because _check_referenced_columns_best_effort scanned
    the raw SQL text including the inside of string literals. Two
    superficially similar queries in the same eval run (WHERE seller_state
    = 'SP', WHERE customer_state = 'MG') happened to pass only because
    their literal values were <=2 characters and hit the short-token
    skip below by accident -- 'delivered'/'canceled'/'credit_card'/'boleto'
    /'on_time' are all realistic values a natural-language question would
    produce and are not short enough to dodge the bug that way."""
    return _STRING_LITERAL_RE.sub("''", sql)


def _check_referenced_columns_best_effort(sql: str) -> None:
    """Best-effort check -- see module docstring for what this does and does
    not guarantee. Strips table-qualifiers (t.col -> col) before matching,
    ignores any alias the query defines for itself (see above), and masks
    out string literal contents first (see _mask_string_literals) so filter
    values aren't mistaken for column references."""
    masked_sql = _mask_string_literals(sql)
    defined_aliases = _collect_query_defined_aliases(masked_sql)
    for raw_token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?", masked_sql):
        token = raw_token.split(".")[-1].lower()
        if (token in _SQL_SAFE_WORDS or token in ALLOWED_TABLES
                or token in ALLOWED_COLUMNS or token in defined_aliases):
            continue
        if token.isdigit() or len(token) <= 2:
            continue  # short tokens are almost always aliases, not columns
        raise SQLValidationError(f"query references unrecognized column or identifier: {token!r}")


def _enforce_row_limit(sql: str, max_rows: int = SQL_MAX_ROWS) -> str:
    sql = sql.strip().rstrip(";")
    match = _LIMIT_PATTERN.search(sql)
    if match:
        requested = int(match.group(1))
        if requested > max_rows:
            sql = _LIMIT_PATTERN.sub(f"LIMIT {max_rows}", sql)
        return sql
    return f"{sql} LIMIT {max_rows}"


def validate_sql(sql: str, role: str = "employee") -> str:
    """Validate LLM-generated SQL against every rule above. Returns the
    (LIMIT-enforced) safe SQL string to execute. Raises SQLValidationError
    on any violation -- callers must not execute the original input on
    failure, and must not show SQLValidationError's message to an
    unauthorized end user (see text_to_sql.py for the refusal wrapper).

    `role` defaults to "employee" -- the most restrictive role in
    SQL_TABLE_ACCESS -- so that a caller which forgets to pass a role fails
    SAFE (denies the payments table) rather than failing open. Always pass
    the real authenticated role explicitly; do not rely on this default."""
    if not sql or not sql.strip():
        raise SQLValidationError("empty SQL")

    _check_no_comments_or_stacked_statements(sql)
    _check_no_forbidden_keywords(sql)
    clean_sql = _check_single_select_statement(sql)
    _check_referenced_tables(clean_sql)
    _check_role_authorized_tables(clean_sql, role)
    _check_role_column_blocklist(clean_sql, role)
    _check_referenced_columns_best_effort(clean_sql)
    return _enforce_row_limit(clean_sql)
