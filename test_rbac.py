"""Tests for rbac.py against the current (Olist-only) config.ROLE_ACCESS.

This file was found stale during the AtliQ-removal pivot: it asserted
department sets from before Phase 7's RBAC redesign (e.g. finance ==
{"finance", "general"}), which hadn't matched config.py's real values in
a while, independent of the Olist pivot. Rewritten here to match the
actual current ROLE_ACCESS, including the "hr" role's removal and the
disappearance of the "general" department (it only ever held AtliQ's
company-overview/product-FAQ docs).
"""
from rbac import allowed_departments, is_authorized, build_chroma_filter


def test_finance_role_access():
    # Phase 16 added expense_docs (the synthetic expense-policy/vendor-
    # profile corpus backing the multimodal document Q&A feature) to
    # finance's departments alongside payments/revenue.
    assert allowed_departments("finance") == {"payments", "revenue", "expense_docs"}
    assert is_authorized("finance", "payments")
    assert is_authorized("finance", "revenue")
    assert is_authorized("finance", "expense_docs")
    assert not is_authorized("finance", "sales")
    assert not is_authorized("finance", "executive")


def test_executive_full_access():
    depts = allowed_departments("executive")
    assert {"sales", "products", "payments", "revenue", "sellers", "delivery",
            "orders", "expense_docs"} <= depts


def test_expense_docs_restricted_to_finance_tier_roles():
    # Phase 16 RBAC decision: expense_docs is finance/manager/executive/admin
    # only -- sales, operations, and employee must not see it.
    for role in ("finance", "manager", "executive", "admin"):
        assert is_authorized(role, "expense_docs"), f"{role} should have expense_docs access"
    for role in ("sales", "operations", "employee"):
        assert not is_authorized(role, "expense_docs"), f"{role} should NOT have expense_docs access"


def test_admin_matches_executive_access():
    # Document-RBAC level: admin has no distinguishing document category of
    # its own yet (see config.py's comment on this), so it mirrors executive.
    assert allowed_departments("admin") == allowed_departments("executive")


def test_employee_has_no_document_departments():
    # Deliberate, not an oversight: there is no "general" department left
    # once AtliQ's docs are gone, so a baseline employee has zero RAG
    # document access (they can still query general SQL tables -- see
    # config.SQL_TABLE_ACCESS, which is a separate RBAC layer).
    assert allowed_departments("employee") == set()
    assert not is_authorized("employee", "sales")
    assert not is_authorized("employee", "payments")


def test_hr_role_no_longer_exists():
    # "hr" was removed entirely in the Olist-only pivot -- there is nothing
    # left for it to guard once the employee handbook/payroll docs are gone.
    assert allowed_departments("hr") == set()


def test_unknown_role_has_no_access():
    assert allowed_departments("unknown_role") == set()
    assert build_chroma_filter("unknown_role") == {"department": "___none___"}


def test_filter_shape_for_multi_department_role():
    f = build_chroma_filter("finance")
    assert f == {"department": {"$in": ["expense_docs", "payments", "revenue"]}}


def test_filter_shape_for_single_department_role():
    f = build_chroma_filter("sales")
    assert f == {"department": {"$in": ["products", "sales"]}}


def test_filter_shape_for_zero_department_role():
    # employee's empty department set must produce the same zero-access
    # filter as an unknown role, not a filter that accidentally matches
    # everything.
    f = build_chroma_filter("employee")
    assert f == {"department": "___none___"}
