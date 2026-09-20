"""FastAPI backend (Phase 12).

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

This process is the ONLY thing that should ever import rag_agent/rag_chain/
auth/rbac internals going forward -- app.py (Phase 13) talks to this over
HTTP instead, the same way any other client would. That separation is the
actual point of this phase, not just "put a REST wrapper on it": it means
the pipeline can be scaled, deployed, and secured independently of whatever
frontend happens to be calling it.
"""
import os
import sys
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auth import authenticate, User  # noqa: E402
from config import (  # noqa: E402
    GROQ_API_KEY, GROQ_MODEL, GROQ_VISION_MODEL, API_RATE_LIMIT_PER_MINUTE,
    MAX_DOCUMENT_PAGES, MAX_DOCUMENT_IMAGE_MB,
)
from monitoring import today_usage_summary  # noqa: E402
from rag_agent.router.dispatch import handle_query  # noqa: E402
from rag_agent.vision.document_qa import (  # noqa: E402
    analyze_document, DocumentQAError, SUPPORTED_IMAGE_CONTENT_TYPES, SUPPORTED_PDF_CONTENT_TYPE,
)
from rag_agent.monitoring.structured_logger import log_structured_interaction  # noqa: E402

from api.dependencies import get_current_user, require_roles  # noqa: E402
from api.schemas import (  # noqa: E402
    DocumentQueryResponse, HealthResponse, LoginRequest, QueryRequest, QueryResponse,
    TokenResponse, UsageSummary, UserInfo,
)
from api.security import create_access_token  # noqa: E402

_MAX_DOCUMENT_BYTES = MAX_DOCUMENT_IMAGE_MB * 1024 * 1024

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{API_RATE_LIMIT_PER_MINUTE}/minute"])

app = FastAPI(
    title="Role-Based Access Controlled RAG AI Agent API",
    description=(
        "RBAC-enforced RAG + Text-to-SQL over the real Olist e-commerce "
        "dataset. Every /query request is authorized using the role "
        "embedded in the caller's JWT, verified server-side -- never a "
        "client-supplied value."
    ),
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_allowed_origins = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """No auth required -- used by container orchestrators (Docker/Azure
    Container Apps) for liveness/readiness probes. Deliberately reports
    whether GROQ_API_KEY is configured rather than silently failing later,
    since a misconfigured deployment should be visible immediately."""
    return HealthResponse(status="ok", groq_configured=bool(GROQ_API_KEY))


@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
    user = authenticate(body.username, body.password)
    if not user:
        # Same error for "unknown username" and "wrong password" -- do not
        # let the API be used to enumerate valid usernames.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token, expires_in = create_access_token(user.username, user.role, user.name)
    return TokenResponse(
        access_token=token, expires_in_minutes=expires_in,
        role=user.role, name=user.name, username=user.username,
    )


@app.get("/me", response_model=UserInfo)
def me(user: User = Depends(get_current_user)):
    return UserInfo(username=user.username, name=user.name, role=user.role)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(f"{API_RATE_LIMIT_PER_MINUTE}/minute")
def query(request: Request, body: QueryRequest, user: User = Depends(get_current_user)):
    """The role passed to handle_query is user.role -- taken from the
    verified JWT, never from `body`. QueryRequest intentionally has no role
    field; a client cannot escalate privilege by sending one."""
    start = time.time()
    try:
        result = handle_query(body.question, user=user, role=user.role)
    except Exception as exc:
        # Don't leak internal stack traces/exception text to the client --
        # log server-side (stdout, captured by the container runtime) and
        # return a sanitized 500.
        print(f"[API ERROR] user={user.username} role={user.role} "
              f"elapsed_ms={(time.time() - start) * 1000:.1f} error={exc!r}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The query could not be processed. Please try again.",
        )
    return QueryResponse(**result)


@app.post("/query/document", response_model=DocumentQueryResponse)
@limiter.limit(f"{API_RATE_LIMIT_PER_MINUTE}/minute")
def query_document(
    request: Request,
    question: str = Form(..., min_length=1, max_length=2000),
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Multimodal "why was this charge deducted?" endpoint (Phase 16).

    Open to every authenticated role -- reading an image the user just
    uploaded and extracting its fields (vendor, line items, total) needs no
    RBAC of its own, so gating the whole endpoint to a role subset would
    have blocked, say, a sales or ops employee from even asking "what does
    this receipt say?" That's the change from the earlier version of this
    endpoint, which required finance/manager/executive/admin before
    accepting a request at all.

    What's still role-restricted -- unconditionally, and NOT by anything in
    this function -- is which supporting expense-policy/vendor documents
    come back to ground the answer: analyze_document() calls the exact same
    hybrid_retrieve(query, role=user.role, ...) every other route uses, and
    config.ROLE_ACCESS only maps the expense_docs department to
    finance/manager/executive/admin (see config.py's comment and
    test_rbac.py's test_expense_docs_restricted_to_finance_tier_roles).
    A sales/ops/employee caller therefore still gets a real, useful answer
    grounded in whatever the vision model read off the document itself --
    it just won't cite expense-policy documents it isn't authorized to see,
    because hybrid_retrieve() returns zero expense_docs chunks for that
    role, not because anything here special-cases the response. Same
    "identical behavior whether unauthorized or nonexistent" property the
    RAG route already relies on -- no new code path to get wrong.
    """
    start = time.time()

    content_type = file.content_type
    allowed_types = SUPPORTED_IMAGE_CONTENT_TYPES | {SUPPORTED_PDF_CONTENT_TYPE}
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{content_type}'. Upload a PNG/JPEG/WebP image or a PDF.",
        )

    file_bytes = file.file.read()
    if len(file_bytes) > _MAX_DOCUMENT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large ({len(file_bytes) / 1024 / 1024:.1f}MB). "
                   f"Maximum is {MAX_DOCUMENT_IMAGE_MB}MB.",
        )
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        result = analyze_document(
            file_bytes, content_type, question, user=user, role=user.role,
            max_pages=MAX_DOCUMENT_PAGES,
        )
    except DocumentQAError as exc:
        # A real, user-fixable problem with the upload itself (corrupt PDF,
        # password-protected, etc.) -- 400, and safe to show exc's message
        # since DocumentQAError messages are already written to be
        # user-facing (see its docstring), unlike a bare Exception below.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        print(f"[API ERROR] user={user.username} role={user.role} endpoint=/query/document "
              f"elapsed_ms={(time.time() - start) * 1000:.1f} error={exc!r}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be processed. Please try again.",
        )

    total_latency_ms = (time.time() - start) * 1000
    log_structured_interaction(
        user_role=user.role,
        query_route="DOCUMENT",
        route_confidence=1.0,
        question=question,
        answer=result["answer"],
        model=f"{GROQ_VISION_MODEL}+{GROQ_MODEL}",
        total_latency_ms=total_latency_ms,
        vision_latency_ms=result.get("vision_latency_ms"),
        retrieval_latency_ms=result.get("retrieval_latency_ms"),
        reranking_latency_ms=result.get("reranking_latency_ms"),
        generation_latency_ms=result.get("generation_latency_ms"),
        retrieved_documents=result.get("citations", []),
        guardrail_result={"blocked": result["blocked"], "reason": result["reason"]},
        confidence=result.get("confidence"),
    )
    return DocumentQueryResponse(**result)


@app.get("/monitoring/summary", response_model=UsageSummary)
def monitoring_summary(user: User = Depends(require_roles("manager", "executive", "admin"))):
    """Restricted to management-tier roles -- usage/cost data is itself a
    business-sensitive aggregate, not something every authenticated role
    should see by default."""
    return UsageSummary(**today_usage_summary())
