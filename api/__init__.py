"""Phase 12: FastAPI backend.

Wraps the existing RBAC/RAG/Text-to-SQL/guardrails pipeline
(src/rag_agent/router/dispatch.handle_query) behind a real HTTP API, so the
Streamlit UI (app.py) becomes a pure API client (Phase 13) instead of
importing pipeline internals directly, and so any other client (a future
non-Streamlit frontend, a CLI, another service) can integrate the same way.

Nothing about RBAC changes by introducing this layer: the role used for
every downstream authorization check is the role embedded in the verified
JWT claim (issued server-side from auth.authenticate()'s real user record),
never a value the client can freely supply. See api/dependencies.py.
"""
