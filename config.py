"""Central configuration for the RBAC RAG Chatbot."""
import os
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
# --- Retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))
TOP_K = int(os.getenv("TOP_K", 4))
COLLECTION_NAME = "company_docs"

# Documents live as flat files named docs_<department>_<slug>.md in this directory.
DOCS_GLOB = "docs_*.md"

# --- RBAC: role -> set of document departments it may retrieve from ---
ROLE_ACCESS = {
    "finance": {"finance", "general"},
    "hr": {"hr", "general"},
    "executive": {"finance", "hr", "general", "executive"},
    "employee": {"general"},
}

# --- Cost tracking (USD per 1M tokens). Check https://groq.com/pricing and update as needed. ---
MODEL_PRICING_PER_1M = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}
DAILY_COST_ALERT_USD = float(os.getenv("DAILY_COST_ALERT_USD", 1.00))

USAGE_LOG_PATH = LOG_DIR / "usage_log.csv"
