"""Pydantic request/response models for the FastAPI backend."""
from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: str
    name: str
    username: str


class UserInfo(BaseModel):
    username: str
    name: str
    role: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    citations: list = []
    sql: Optional[str] = None
    row_count: Optional[int] = None
    confidence: Optional[str] = None
    blocked: bool = False
    reason: Optional[str] = None
    route: str
    route_confidence: float
    retrieval_latency_ms: Optional[float] = None
    reranking_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None
    sql_execution_time_ms: Optional[float] = None


class DocumentQueryResponse(BaseModel):
    answer: str
    extracted_fields: Optional[dict] = None
    citations: list = []
    confidence: Optional[str] = None
    blocked: bool = False
    reason: Optional[str] = None
    vision_latency_ms: Optional[float] = None
    retrieval_latency_ms: Optional[float] = None
    reranking_latency_ms: Optional[float] = None
    generation_latency_ms: Optional[float] = None


class UsageSummary(BaseModel):
    queries: int
    total_cost_usd: float
    avg_latency_ms: float
    blocked: int


class HealthResponse(BaseModel):
    status: str
    groq_configured: bool
