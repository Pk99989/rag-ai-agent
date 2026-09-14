"""Central configuration for the RBAC RAG Chatbot."""
import os
import secrets
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_store"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# --- Vision LLM (Phase 16: multimodal document Q&A) ---
# qwen/qwen3.6-27b, not a Llama vision model -- verified against Groq's own
# /docs/vision page at build time (Sept 2026), not assumed from training
# data or a generic web search result (which surfaced now-superseded
# Llama-4-Scout/Maverick names). Supports up to 5 images/request (multi-page
# statements), JSON mode combined with vision, and a 20MB per-request image
# size cap -- MAX_DOCUMENT_PAGES/MAX_DOCUMENT_IMAGE_MB below mirror those
# real API limits so we reject an oversized upload before ever calling Groq,
# not after paying for a guaranteed-to-fail request.
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
MAX_DOCUMENT_PAGES = int(os.getenv("MAX_DOCUMENT_PAGES", 5))
MAX_DOCUMENT_IMAGE_MB = int(os.getenv("MAX_DOCUMENT_IMAGE_MB", 20))
# Real limit hit in testing, not a guess: Groq's free/on-demand tier caps
# this vision model at 1000 output tokens/minute (OTPM). Left unset,
# ChatGroq requests enough headroom for a large response and Groq rejects
# the request outright (429 "Requested 1342 ... exceed the enforced
# limit") before it ever runs -- it's a preflight rejection, not a mid-
# generation truncation. The extraction schema is a small JSON object, so
# capping well under the per-minute budget is safe; if a real deployment
# is on a paid tier with a higher OTPM limit, raise this via env var.
GROQ_VISION_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_VISION_MAX_OUTPUT_TOKENS", 512))

# --- FastAPI backend auth (Phase 12) ---
# JWTs issued by api/auth.py so the Streamlit UI (or any other client) can
# call the API as a separate process/host instead of importing rag_agent
# internals directly. Deliberately does NOT hardcode a secret in source --
# that would be exactly the "don't hardcode API keys/secrets" mistake this
# project has avoided everywhere else. If JWT_SECRET_KEY isn't set, a
# random one is generated per-process: this is fine for local dev (login
# still works within that run) but means every restart invalidates
# existing tokens, and it is NOT safe for a real multi-instance deployment
# (each instance would mint tokens the others can't verify) -- the loud
# warning below is intentional, not decorative. Set JWT_SECRET_KEY in .env
# (or as an Azure Container Apps secret, see deploy_azure.sh) before
# deploying anywhere real.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not JWT_SECRET_KEY:
    JWT_SECRET_KEY = secrets.token_hex(32)
    warnings.warn(
        "JWT_SECRET_KEY is not set -- generated a random per-process secret. "
        "Tokens will stop validating on restart, and this is UNSAFE for any "
        "deployment with more than one running instance. Set JWT_SECRET_KEY "
        "in your .env before deploying.",
        stacklevel=2,
    )
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))
API_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", 30))
# --- Retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("TOP_K", 4))
COLLECTION_NAME = "company_docs"

# --- RBAC: role -> set of document departments it may retrieve from ---
# Olist-only redesign: the project originally mixed a fictional AtliQ
# company's internal docs (finance/hr/general/executive department tags)
# with the real Olist e-commerce dataset. AtliQ's documents were removed
# entirely at the user's explicit request (see conversation) so the whole
# assistant -- RAG, SQL, and RBAC -- operates on one real dataset. That
# also removed every document department AtliQ owned outright ("general",
# "finance", "hr", "executive" as DOCUMENT tags -- "finance" and
# "executive" also exist as ROLE names below, which is a different thing).
# What's left maps 1:1 onto scripts/generate_olist_rag_docs.py's real
# output departments: sales, products, payments, revenue, sellers,
# delivery, orders. "hr" is removed as a ROLE entirely -- there is nothing
# left for it to guard once the employee handbook and payroll policy docs
# are gone, and inventing a placeholder would be exactly the kind of
# unstated-assumption this project's own conventions argue against.
_SALES_DEPTS = {"sales", "products"}
_FINANCE_DEPTS = {"payments", "revenue"}
_OPERATIONS_DEPTS = {"sellers", "delivery", "orders"}

# Phase 16: synthetic expense-policy/vendor-profile documents backing the
# multimodal document Q&A feature (see scripts/generate_expense_docs.py).
# Deliberately its own department, not folded into _FINANCE_DEPTS: the
# expense/invoice explainer is gated to a slightly different, slightly
# broader set of roles (finance/manager/executive/admin) than the existing
# payments/revenue RAG documents happen to be, and keeping it a distinct
# department makes that an explicit, visible choice rather than something
# that falls out of reusing _FINANCE_DEPTS by coincidence.
_EXPENSE_DEPTS = {"expense_docs"}

ALL_DEPARTMENTS = _SALES_DEPTS | _FINANCE_DEPTS | _OPERATIONS_DEPTS | _EXPENSE_DEPTS

ROLE_ACCESS = {
    # Empty set is a real, deliberate outcome, not an oversight: there is
    # no "general" document department anymore (that only ever held
    # AtliQ's company-overview/product-FAQ docs), so a baseline employee
    # has no aggregate-report documents of their own -- they can still
    # query the general SQL tables (see SQL_TABLE_ACCESS below).
    "employee": set(),
    "sales": _SALES_DEPTS,
    "finance": _FINANCE_DEPTS | _EXPENSE_DEPTS,
    "operations": _OPERATIONS_DEPTS,
    "manager": _SALES_DEPTS | _FINANCE_DEPTS | _OPERATIONS_DEPTS | _EXPENSE_DEPTS,
    "executive": ALL_DEPARTMENTS,
    "admin": ALL_DEPARTMENTS,
}

# --- RBAC for Text-to-SQL: role -> set of tables it may query ---
# Spec gives exactly one concrete example: "Finance users may access
# payment data. A normal employee must not be able to query restricted
# finance information." Implemented as exactly that restriction (payments
# gated to finance/manager/executive/admin) rather than inventing
# additional per-table restrictions the spec never asked for -- every role
# can otherwise query the general business tables via Text-to-SQL.
_GENERAL_SQL_TABLES = {"customers", "orders", "order_items", "products",
                        "sellers", "reviews", "categories", "order_facts"}
ALL_SQL_TABLES = _GENERAL_SQL_TABLES | {"payments"}

SQL_TABLE_ACCESS = {
    "employee": _GENERAL_SQL_TABLES,
    "sales": _GENERAL_SQL_TABLES,
    "operations": _GENERAL_SQL_TABLES,
    "finance": ALL_SQL_TABLES,
    "manager": ALL_SQL_TABLES,
    "executive": ALL_SQL_TABLES,
    "admin": ALL_SQL_TABLES,
}

# Column-level SQL RBAC hook: table -> set of columns that role may NOT
# select, even within an allowed table. Empty for every table right now --
# the spec asked for column-level control as a capability but gave no
# concrete example of a column to restrict, so this is left as a real,
# wired mechanism (see validator.py) rather than a restriction invented
# to look complete. Add entries here when a real column-level rule exists.
SQL_COLUMN_BLOCKLIST: dict = {}

# --- Cost tracking (USD per 1M tokens). Check https://groq.com/pricing and update as needed. ---
MODEL_PRICING_PER_1M = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}
DAILY_COST_ALERT_USD = float(os.getenv("DAILY_COST_ALERT_USD", 1.00))

USAGE_LOG_PATH = LOG_DIR / "usage_log.csv"

# --- Olist structured dataset (Phase 2) ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OLIST_DB_PATH = DATA_DIR / "olist.db"

# Filenames exactly as distributed by the Kaggle "Brazilian E-Commerce Public
# Dataset by Olist" (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
# Not committed to git -- see .gitignore and README for download instructions.
OLIST_REQUIRED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

# --- Text-to-SQL (Phase 3) ---
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", 200))
SQL_TIMEOUT_SECONDS = float(os.getenv("SQL_TIMEOUT_SECONDS", 5.0))
SQL_QUERY_LOG_PATH = LOG_DIR / "sql_query_log.csv"

# --- Olist RAG document generation (Phase 5) ---
# Aggregated business-report documents generated from data/olist.db by
# scripts/generate_olist_rag_docs.py. Deliberately NOT one document per raw
# order/product row -- see that script's module docstring for why.
RAG_DOCS_DIR = PROCESSED_DATA_DIR / "rag_documents"

# --- Hybrid retrieval + reranking (Phase 6) ---
# How many candidates each of vector search / BM25 fetch before fusion.
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", 20))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", 0.6))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))
# Minimum fused (normalized) score to keep a candidate at all, before reranking.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.0))
# Final number of chunks returned after cross-encoder reranking.
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", TOP_K))
# ONNX cross-encoder (Xenova's ONNX export -- no torch/sentence-transformers
# dependency, consistent with the rest of this project). Downloaded once
# from the Hugging Face Hub and cached locally by huggingface_hub.
RERANKER_MODEL_REPO = os.getenv("RERANKER_MODEL_REPO", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", 256))
BM25_INDEX_PATH = CHROMA_DIR.parent / "bm25_index.pkl"
