"""Manual smoke test for Phase 7 (RBAC upgrade: multi-role, doc + SQL tables).

Usage:
    python scripts/smoke_test_rbac_phase7.py

No API key needed -- pure Python checks against rbac.py and the SQL
validator. Reproduces the spec's example pattern, updated for the
Olist-only pivot's RBAC redesign (the "hr" role was removed entirely, and
document departments are now sales/products/payments/revenue/sellers/
delivery/orders -- see config.py's comments for why):
    Employee  -> payment-restricted data -> DENIED
    Sales     -> payment documents       -> DENIED
    Finance   -> payment data            -> ALLOWED
    Executive -> authorized business data -> ALLOWED
...applied to BOTH the document-RBAC path and the SQL-table-RBAC path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rbac import is_authorized
from rag_agent.sql.validator import validate_sql, SQLValidationError

DOC_CASES = [
    ("employee", "payments", False),
    ("sales", "payments", False),
    ("finance", "payments", True),
    ("executive", "delivery", True),
    ("admin", "sellers", True),
    ("manager", "payments", True),      # manager = union incl. finance
    ("operations", "payments", False),
    ("operations", "delivery", True),   # operations' own department
    ("employee", "sales", False),       # employee has zero document departments
]

SQL_CASES = [
    ("employee", "SELECT * FROM payments", False),
    ("finance", "SELECT * FROM payments", True),
    ("sales", "SELECT * FROM orders", True),
    ("employee", "SELECT * FROM orders", True),
    ("admin", "SELECT * FROM payments", True),
    ("operations", "SELECT * FROM payments", False),
]


def run_doc_rbac_checks() -> bool:
    print("=== Part 1: document RBAC (is_authorized) ===")
    all_ok = True
    for role, department, expected in DOC_CASES:
        result = is_authorized(role, department)
        ok = result == expected
        all_ok &= ok
        allowed_word = "ALLOWED" if expected else "DENIED"
        print(f"[{'PASS' if ok else 'FAIL'}] {role} -> {department}: "
              f"got {'ALLOWED' if result else 'DENIED'} (expected {allowed_word})")
    return all_ok


def run_sql_rbac_checks() -> bool:
    print("\n=== Part 2: SQL table RBAC (validate_sql role check) ===")
    all_ok = True
    for role, sql, expected_allowed in SQL_CASES:
        try:
            validate_sql(sql, role=role)
            actual_allowed = True
            detail = "validated OK"
        except SQLValidationError as exc:
            actual_allowed = False
            detail = str(exc)
        ok = actual_allowed == expected_allowed
        all_ok &= ok
        expected_word = "ALLOWED" if expected_allowed else "DENIED"
        print(f"[{'PASS' if ok else 'FAIL'}] {role} -> {sql!r}: "
              f"got {'ALLOWED' if actual_allowed else 'DENIED'} (expected {expected_word}) "
              f"-- {detail}")
    return all_ok


if __name__ == "__main__":
    ok1 = run_doc_rbac_checks()
    ok2 = run_sql_rbac_checks()
    all_ok = ok1 and ok2
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED -- see above'}")
    sys.exit(0 if all_ok else 1)
