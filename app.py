"""Streamlit frontend (Phase 13) -- pure API client.

Post-Phase-12, this file has NO import of rag_agent/rag_chain/auth/rbac
internals. Every action (login, ask a question, view usage) goes over HTTP
to the FastAPI backend in api/main.py, exactly the way any other client
would have to. This is a deliberate architectural change, not a cosmetic
one: RBAC, guardrails, and the router all live and are enforced in the API
process now, so this file cannot accidentally bypass them even if a future
edit here gets sloppy -- there's nothing left to bypass, since this process
never sees a role it wasn't handed inside a verified token by the backend.
"""
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_DEMO_CREDENTIALS = [
    ("alice.finance", "finance123", "finance"),
    ("carol.ceo", "exec123", "executive"),
    ("dave.eng", "employee123", "employee"),
    ("erin.sales", "sales123", "sales"),
    ("frank.ops", "ops123", "operations"),
    ("grace.mgr", "manager123", "manager"),
    ("henry.admin", "admin123", "admin"),
]

st.set_page_config(page_title="Role-Based Access Controlled RAG AI Agent", page_icon="🔒", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def _api_get(path, **kwargs):
    headers = kwargs.pop("headers", {})
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return requests.get(f"{API_BASE_URL}{path}", headers=headers, timeout=30, **kwargs)


def _api_post(path, **kwargs):
    headers = kwargs.pop("headers", {})
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return requests.post(f"{API_BASE_URL}{path}", headers=headers, timeout=60, **kwargs)


def _backend_reachable() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def login_view():
    st.title("🔒 Role-Based Access Controlled RAG AI Agent — Sign In")

    if not _backend_reachable():
        st.error(
            f"Can't reach the API backend at {API_BASE_URL}. Start it first: "
            f"`uvicorn api.main:app --host 0.0.0.0 --port 8000`"
        )

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            r = _api_post("/auth/login", json={"username": username, "password": password})
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API backend: {exc}")
        else:
            if r.status_code == 200:
                body = r.json()
                st.session_state.token = body["access_token"]
                st.session_state.user = {
                    "username": body["username"], "name": body["name"], "role": body["role"],
                }
                st.rerun()
            else:
                st.error(r.json().get("detail", "Invalid username or password."))

    with st.expander("Demo credentials"):
        for username, password, role in _DEMO_CREDENTIALS:
            st.code(f"{username} / {password}  ->  role: {role}")


def _chat_tab():
    st.caption("Answers are grounded only in Olist business documents and data you're authorized to access.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role_ui"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

    if prompt := st.chat_input("Ask about orders, sellers, payments, deliveries..."):
        st.session_state.messages.append({"role_ui": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing and answering..."):
                try:
                    r = _api_post("/query", json={"question": prompt})
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach the API backend: {exc}")
                    st.stop()

            if r.status_code == 401:
                st.error("Your session expired. Please sign out and sign in again.")
                st.stop()
            elif r.status_code != 200:
                st.error(r.json().get("detail", f"Request failed ({r.status_code})."))
                st.stop()

            result = r.json()
            st.markdown(result["answer"])
            if result.get("sources"):
                st.caption("Sources: " + ", ".join(result["sources"]))
            if result.get("citations"):
                with st.expander(f"Citations ({len(result['citations'])})"):
                    for c in result["citations"]:
                        st.json(c)
            if result.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(result["sql"], language="sql")
                    if result.get("row_count") is not None:
                        st.caption(f"{result['row_count']} row(s) returned")
            st.caption(f"route: {result.get('route')} · confidence: {result.get('confidence')}")
            latency_bits = [
                (label, result.get(key)) for label, key in [
                    ("retrieval", "retrieval_latency_ms"),
                    ("reranking", "reranking_latency_ms"),
                    ("generation", "generation_latency_ms"),
                    ("SQL exec", "sql_execution_time_ms"),
                ] if result.get(key) is not None
            ]
            if latency_bits:
                st.caption(" · ".join(f"{label}: {value:.0f}ms" for label, value in latency_bits))
            if result.get("blocked"):
                st.warning(f"Blocked: {result['reason']}")
        st.session_state.messages.append({
            "role_ui": "assistant", "content": result["answer"], "sources": result.get("sources"),
        })


def _document_tab():
    role = st.session_state.user["role"]
    st.caption(
        "Upload a credit card statement, invoice, receipt, or expense report and ask why a "
        "charge was made. Answers are grounded in the document itself plus retrieved expense "
        "policy/vendor context -- Phase 16, see rag_agent/vision/document_qa.py."
    )
    # Display-only note, not an access-control boundary (that's enforced
    # server-side via config.ROLE_ACCESS): tells non-finance-tier roles
    # up front why their answer may lack policy citations, instead of
    # leaving them to wonder why finance's answers look more detailed.
    if role not in {"finance", "manager", "executive", "admin"}:
        st.caption(
            f"Note: as a `{role}` user, you'll get an answer based on what's extracted from "
            "the document itself, but not expense-policy/vendor citations -- those are "
            "restricted to finance-tier roles."
        )
    uploaded = st.file_uploader(
        "Document (image or PDF)", type=["png", "jpg", "jpeg", "webp", "pdf"],
    )
    question = st.text_input("Question", value="Why was this charge deducted?")
    submitted = st.button("Explain this charge", disabled=uploaded is None)

    if not submitted:
        return
    if uploaded is None:
        st.warning("Upload a document first.")
        return

    with st.spinner("Reading the document and looking up supporting context..."):
        try:
            r = requests.post(
                f"{API_BASE_URL}/query/document",
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                data={"question": question},
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=90,
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the API backend: {exc}")
            return

    if r.status_code == 401:
        st.error("Your session expired. Please sign out and sign in again.")
        return
    if r.status_code == 403:
        st.error("Your role doesn't have access to this feature.")
        return
    if r.status_code != 200:
        st.error(r.json().get("detail", f"Request failed ({r.status_code})."))
        return

    result = r.json()
    st.markdown(result["answer"])
    if result.get("extracted_fields"):
        with st.expander("Extracted fields"):
            st.json(result["extracted_fields"])
    if result.get("citations"):
        st.caption("Sources: " + ", ".join(result["citations"]))
    st.caption(f"confidence: {result.get('confidence')}")
    latency_bits = [
        (label, result.get(key)) for label, key in [
            ("vision", "vision_latency_ms"),
            ("retrieval", "retrieval_latency_ms"),
            ("reranking", "reranking_latency_ms"),
            ("generation", "generation_latency_ms"),
        ] if result.get(key) is not None
    ]
    if latency_bits:
        st.caption(" · ".join(f"{label}: {value:.0f}ms" for label, value in latency_bits))
    if result.get("blocked"):
        st.warning(f"Blocked: {result['reason']}")


def chat_view():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**{user['name']}**")
        st.caption(f"Role: `{user['role']}`")
        if st.button("Sign out"):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.subheader("Usage today")
        r = _api_get("/monitoring/summary")
        if r.status_code == 200:
            summary = r.json()
            st.metric("Queries", summary["queries"])
            st.metric("Est. cost (USD)", f"${summary['total_cost_usd']}")
            st.metric("Avg latency (ms)", summary["avg_latency_ms"])
            st.metric("Blocked", summary["blocked"])
        elif r.status_code == 403:
            st.caption("Usage summary is restricted to manager/executive/admin roles.")
        else:
            st.caption("Usage summary unavailable.")

    st.title("💬 Role-Based Access Controlled RAG AI Agent")

    # Every role gets both tabs now: uploading and reading a document needs
    # no RBAC of its own. What's still role-restricted is which supporting
    # expense-policy/vendor documents ground the answer -- that's enforced
    # server-side in hybrid_retrieve() via config.ROLE_ACCESS, not here.
    tab_chat, tab_document = st.tabs(["💬 Chat", "📄 Explain a Charge"])
    with tab_chat:
        _chat_tab()
    with tab_document:
        _document_tab()


if st.session_state.token is None:
    login_view()
else:
    chat_view()
