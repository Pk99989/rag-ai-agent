"""Tests for src/rag_agent/sql/validator.py.

The string-literal cases below exist because Phase 10's real eval dataset
caught a real bug: validate_sql rejected 5 of 20 legitimate SQL questions
because the best-effort column checker scanned string literal VALUES
('delivered', 'credit_card', 'on_time', ...) as if they were column
references. Two similar queries in the same run only passed by accident
because their literal values ('SP', 'MG') were <=2 characters and hit the
short-token skip. These tests pin the fix so it can't silently regress.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_agent.sql.validator import validate_sql, SQLValidationError


def test_string_literal_value_not_treated_as_column():
    sql = "SELECT COUNT(*) AS num_delivered FROM orders WHERE order_status = 'delivered'"
    # Must not raise -- 'delivered' is a filter value, not a column reference.
    validate_sql(sql, role="employee")


def test_string_literal_with_underscore_not_treated_as_column():
    sql = "SELECT SUM(payment_value) AS total FROM payments WHERE payment_type = 'credit_card'"
    validate_sql(sql, role="finance")


def test_string_literal_on_time_value_not_treated_as_column():
    sql = "SELECT AVG(delivery_days) AS avg_days FROM orders WHERE delivery_status = 'on_time'"
    validate_sql(sql, role="employee")


def test_two_char_literal_still_worked_before_and_after_fix():
    sql = "SELECT COUNT(*) AS num_sellers FROM sellers WHERE seller_state = 'SP'"
    validate_sql(sql, role="employee")


def test_genuinely_unrecognized_column_still_rejected():
    # The fix must not make the checker toothless -- a real bogus column
    # reference (not inside a string literal) should still be rejected.
    sql = "SELECT made_up_column FROM orders"
    try:
        validate_sql(sql, role="employee")
        assert False, "expected SQLValidationError for a genuinely unknown column"
    except SQLValidationError:
        pass


def test_forbidden_keyword_still_rejected():
    sql = "DROP TABLE orders"
    try:
        validate_sql(sql, role="admin")
        assert False, "expected SQLValidationError for a forbidden keyword"
    except SQLValidationError:
        pass


def test_role_unauthorized_table_still_rejected():
    sql = "SELECT * FROM payments"
    try:
        validate_sql(sql, role="employee")
        assert False, "expected SQLValidationError for employee querying payments"
    except SQLValidationError:
        pass
