"""Manual smoke test for Phase 3 (Text-to-SQL + SQL security).

Usage:
    python scripts/smoke_test_sql.py

Part 1 needs no API key and no network -- it only exercises the validator's
pure security logic. Part 2 needs GROQ_API_KEY set (and network access) and
actually generates + executes SQL against data/olist.db.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GROQ_API_KEY
from rag_agent.sql.validator import validate_sql, SQLValidationError


def run_validator_checks() -> bool:
    print("=== Part 1: validator security checks (no API key needed) ===")
    cases = [
        ("SELECT product_category_name, SUM(price) AS revenue FROM order_items "
         "GROUP BY product_category_name ORDER BY revenue DESC", True),
        ("SELECT * FROM orders", True),
        ("DROP TABLE orders", False),
        ("SELECT * FROM orders; DROP TABLE orders;", False),
        ("SELECT * FROM orders WHERE 1=1 -- comment", False),
        ("SELECT * FROM secret_admin_table", False),
        ("PRAGMA table_info(orders)", False),
        ("ATTACH DATABASE 'x.db' AS x", False),
    ]
    all_ok = True
    for sql, should_pass in cases:
        try:
            result = validate_sql(sql)
            passed = should_pass
            detail = f"-> ACCEPTED: {result}"
        except SQLValidationError as exc:
            passed = not should_pass
            detail = f"-> REJECTED: {exc}"
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"[{status}] {sql!r} {detail}")
    return all_ok


def run_full_pipeline_check() -> None:
    print("\n=== Part 2: full pipeline (needs GROQ_API_KEY + network) ===")
    if not GROQ_API_KEY:
        print("Skipping: GROQ_API_KEY not set in .env")
        return
    from rag_agent.sql.text_to_sql import run_text_to_sql

    question = "Which product category generated the highest revenue?"
    print(f"Question: {question}")
    result = run_text_to_sql(question, role="finance")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    validator_ok = run_validator_checks()
    run_full_pipeline_check()
    print(f"\nValidator checks: {'ALL PASSED' if validator_ok else 'SOME FAILED -- see above'}")
    sys.exit(0 if validator_ok else 1)
