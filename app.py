import streamlit as st
from auth import authenticate, demo_credentials
from rag_chain import answer_query
from monitoring import today_usage_summary

st.set_page_config(page_title="AtliQ Internal RAG Chatbot", page_icon="🔒", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def login_view():
    st.title("🔒 AtliQ Internal Assistant — Sign In")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid username or password.")
    with st.expander("Demo credentials"):
        for c in demo_credentials():
            st.code(f"{c['username']} / {c['password']}  ->  role: {c['role']} ({c['name']})")


def chat_view():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"**{user.name}**")
        st.caption(f"Role: `{user.role}`")
        if st.button("Sign out"):
            st.session_state.user = None
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.subheader("Usage today")
        summary = today_usage_summary()
        st.metric("Queries", summary["queries"])
        st.metric("Est. cost (USD)", f"${summary['total_cost_usd']}")
        st.metric("Avg latency (ms)", summary["avg_latency_ms"])
        st.metric("Blocked", summary["blocked"])

    st.title("💬 AtliQ Internal Assistant")
    st.caption("Answers are grounded only in company documents you're authorized to access.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role_ui"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))

    if prompt := st.chat_input("Ask about company policy, finance, HR..."):
        st.session_state.messages.append({"role_ui": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                result = answer_query(prompt, user=user, role=user.role)
            st.markdown(result["answer"])
            if result.get("sources"):
                st.caption("Sources: " + ", ".join(result["sources"]))
            if result.get("blocked"):
                st.warning(f"Blocked: {result['reason']}")
        st.session_state.messages.append({
            "role_ui": "assistant", "content": result["answer"], "sources": result.get("sources"),
        })


if st.session_state.user is None:
    login_view()
else:
    chat_view()
