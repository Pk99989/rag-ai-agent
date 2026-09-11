from rbac import allowed_departments, is_authorized, build_chroma_filter


def test_finance_role_access():
    assert allowed_departments("finance") == {"finance", "general"}
    assert is_authorized("finance", "finance")
    assert not is_authorized("finance", "hr")
    assert not is_authorized("finance", "executive")


def test_executive_full_access():
    depts = allowed_departments("executive")
    assert {"finance", "hr", "general", "executive"} <= depts


def test_employee_general_only():
    assert allowed_departments("employee") == {"general"}
    assert not is_authorized("employee", "hr")


def test_unknown_role_has_no_access():
    assert allowed_departments("unknown_role") == set()
    assert build_chroma_filter("unknown_role") == {"department": "___none___"}


def test_filter_shape_for_multi_department_role():
    f = build_chroma_filter("finance")
    assert f == {"department": {"$in": ["finance", "general"]}}


def test_filter_shape_for_single_department_role():
    f = build_chroma_filter("employee")
    assert f == {"department": "general"}
