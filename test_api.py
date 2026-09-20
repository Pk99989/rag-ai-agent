"""Tests for the Phase 12 FastAPI backend (api/main.py).

These deliberately do NOT test /query's actual answer content -- that needs
a live Groq call and a built vector index, already covered by
evaluation/run_evaluation.py and eval_run.py. What's tested here is the
security boundary this phase adds: does auth actually gate access, is the
role taken from the verified token (not the request body), and does a
lower-privilege role get refused where it should.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # slowapi's in-memory limiter keys on remote address, and every
    # TestClient request in this module shares one fake address -- without
    # resetting between tests, /auth/login's 10/minute limit accumulates
    # across every test that calls _login() (there are now more than 10
    # across this file) and later tests start getting 429s instead of real
    # logins. Found the hard way twice now (first with the dedicated rate-
    # limit test, then again when the Phase 16 document tests pushed the
    # cumulative count over the edge) -- autouse so no future test has to
    # remember to do this itself.
    app.state.limiter.reset()
    yield


def _login(username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    return r


def test_health_requires_no_auth():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_me_without_token_is_401():
    r = client.get("/me")
    assert r.status_code == 401


def test_login_wrong_password_is_401():
    r = _login("alice.finance", "wrong-password")
    assert r.status_code == 401


def test_login_unknown_user_is_401_same_as_wrong_password():
    # Same status/detail as a wrong password for a real user -- the API
    # must not let a caller distinguish "no such user" from "wrong
    # password", or it becomes a username enumeration oracle.
    r_unknown = _login("nobody.at.all", "whatever")
    r_wrong = _login("alice.finance", "wrong-password")
    assert r_unknown.status_code == r_wrong.status_code == 401
    assert r_unknown.json()["detail"] == r_wrong.json()["detail"]


def test_login_success_returns_token_and_role():
    r = _login("alice.finance", "finance123")
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "finance"
    assert body["username"] == "alice.finance"
    assert body["access_token"]


def test_me_with_valid_token_reflects_role():
    token = _login("erin.sales", "sales123").json()["access_token"]
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["role"] == "sales"


def test_me_with_garbage_token_is_401():
    r = client.get("/me", headers={"Authorization": "Bearer not.a.real.token"})
    assert r.status_code == 401


def test_monitoring_summary_denied_for_non_management_role():
    token = _login("alice.finance", "finance123").json()["access_token"]
    r = client.get("/monitoring/summary", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403


def test_monitoring_summary_allowed_for_admin():
    token = _login("henry.admin", "admin123").json()["access_token"]
    r = client.get("/monitoring/summary", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "queries" in r.json()


def test_monitoring_summary_allowed_for_manager():
    token = _login("grace.mgr", "manager123").json()["access_token"]
    r = client.get("/monitoring/summary", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_query_without_token_is_401():
    r = client.post("/query", json={"question": "anything"})
    assert r.status_code == 401


def test_query_request_has_no_role_field():
    # Structural guarantee, not just a runtime check: QueryRequest must not
    # even have a field a client could (ab)use to claim a role. If someone
    # adds one later, this test breaks loudly instead of silently opening a
    # privilege-escalation path.
    from api.schemas import QueryRequest
    assert "role" not in QueryRequest.model_fields


def test_query_empty_question_is_422():
    token = _login("dave.eng", "employee123").json()["access_token"]
    r = client.post("/query", json={"question": ""}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 422


def _login_token(username, password):
    return _login(username, password).json()["access_token"]


def test_query_document_open_to_all_authenticated_roles():
    # The endpoint itself is NOT role-gated (changed from an earlier
    # version that required finance/manager/executive/admin before
    # accepting a request at all) -- reading a document a user just
    # uploaded needs no RBAC of its own. What's still role-restricted is
    # which expense-policy/vendor documents ground the answer, enforced
    # inside analyze_document()'s hybrid_retrieve() call via
    # config.ROLE_ACCESS, not at this endpoint -- see
    # test_rbac.py::test_expense_docs_restricted_to_finance_tier_roles for
    # that layer. Using an unsupported content-type here (not a real
    # image) so every role reaches the SAME 400 content-type check without
    # needing a live Groq call -- if any role were still RBAC-denied at
    # the endpoint, it would get 403 here instead of 400.
    for username, password in [
        ("dave.eng", "employee123"), ("erin.sales", "sales123"), ("frank.ops", "ops123"),
        ("alice.finance", "finance123"), ("grace.mgr", "manager123"),
        ("carol.ceo", "exec123"), ("henry.admin", "admin123"),
    ]:
        token = _login_token(username, password)
        r = client.post(
            "/query/document", data={"question": "why was this charged?"},
            files={"file": ("doc.txt", b"hello", "text/plain")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 400, f"{username} should reach content-type validation, not be RBAC-denied"
        assert "Unsupported file type" in r.json()["detail"]


def test_query_document_allowed_role_but_unsupported_content_type():
    token = _login_token("alice.finance", "finance123")
    r = client.post(
        "/query/document", data={"question": "why?"},
        files={"file": ("doc.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_query_document_empty_file_rejected():
    token = _login_token("alice.finance", "finance123")
    r = client.post(
        "/query/document", data={"question": "why?"},
        files={"file": ("receipt.png", b"", "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_query_document_oversized_file_rejected_before_calling_groq():
    from config import MAX_DOCUMENT_IMAGE_MB
    token = _login_token("alice.finance", "finance123")
    oversized = b"x" * ((MAX_DOCUMENT_IMAGE_MB + 1) * 1024 * 1024)
    r = client.post(
        "/query/document", data={"question": "why?"},
        files={"file": ("receipt.png", oversized, "image/png")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "too large" in r.json()["detail"].lower()


def test_query_document_no_token_is_401():
    r = client.post(
        "/query/document", data={"question": "why?"},
        files={"file": ("receipt.png", b"abc", "image/png")},
    )
    assert r.status_code == 401


def test_login_rate_limit_kicks_in():
    # config.API_RATE_LIMIT_PER_MINUTE governs /query; /auth/login has its
    # own tighter literal limit (10/minute, see api/main.py) since
    # unauthenticated login attempts are the more attractive brute-force
    # target. The _reset_rate_limiter autouse fixture above already gave
    # this test a clean slate -- see that fixture's comment for why that's
    # required at all, not optional.
    codes = [
        _login("alice.finance", "wrong-password").status_code
        for _ in range(14)
    ]
    assert codes[:10] == [401] * 10
    assert 429 in codes[10:]
