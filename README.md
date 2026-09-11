# AtliQ Internal Assistant — Role-Based Access Controlled RAG Chatbot

A runnable implementation of the major project brief: an internal company chatbot that
answers questions from private documents, enforces role-based access control (RBAC),
applies guardrails, tracks cost, and is set up for cloud deployment and automated
evaluation.

## What's real vs. what needs your own credentials

Everything in this repo runs locally today with no paid dependency except the LLM call
itself. Two things require your own accounts and cannot be done on your behalf:

1. **Groq API key** (free tier) — needed for the LLM to actually generate answers.
   Retrieval, RBAC filtering, guardrails, and monitoring all work and are testable
   without it; only the final generation step needs it.
2. **Azure subscription** — needed to actually deploy. `deploy_azure.sh` is a real,
   runnable `az cli` script, but you have to run it yourself after `az login`.

## Project layout

All files are flat in this directory by design (no nested folders) — reorganize into
`src/`, `data/`, `tests/` etc. if you'd prefer a conventional package layout once you
have this in your own git repo; the code has no hard dependency on being flat.

| File | Purpose |
|---|---|
| `config.py` | Central settings: RBAC role→department map, chunking, pricing, paths |
| `auth.py` | Mock login (demo users below) — swap for Azure AD before production |
| `rbac.py` | Role → allowed departments, builds the Chroma metadata filter |
| `ingest.py` | Loads `docs_*.md`, chunks, embeds (local ONNX MiniLM), stores in ChromaDB |
| `guardrails.py` | PII detection/redaction, prompt-injection detection, out-of-scope detection |
| `rag_chain.py` | Retrieval (RBAC-filtered) → guardrails → Groq LLM → guardrails → log |
| `monitoring.py` | Per-query usage log (CSV), token cost estimate, daily cost-threshold alert |
| `app.py` | Streamlit frontend: login, role-aware chat, sidebar usage dashboard |
| `docs_<dept>_*.md` | Synthetic company documents (finance / hr / general / executive) |
| `eval_dataset.json`, `eval_run.py` | RBAC regression check + generation smoke test |
| `test_*.py` | Unit tests (pytest) for RBAC, guardrails, ingestion |
| `Dockerfile`, `docker-compose.yml` | Containerize the app |
| `deploy_azure.sh` | Deploy to Azure Container Apps (template — fill in and run yourself) |
| `ci_workflow.yml` | GitHub Actions CI gate — move to `.github/workflows/ci.yml` |

## How RBAC works

Each document is tagged with a `department` (parsed from its filename, e.g.
`docs_finance_...md` → `finance`). `config.ROLE_ACCESS` maps each role to the set of
departments it may see:

```
finance    -> finance, general
hr         -> hr, general
executive  -> finance, hr, general, executive
employee   -> general
```

`rbac.build_chroma_filter(role)` turns that into a Chroma `where` clause, so the vector
search **never retrieves** unauthorized chunks — access is enforced at retrieval time,
not by asking the LLM nicely.

## How guardrails work

- **Input**: blocks obvious prompt-injection attempts ("ignore previous instructions...");
  flags PII typed into the query.
- **Retrieval**: if nothing relevant is retrieved (or the question matches out-of-scope
  patterns like "tell me a joke"), the bot declines instead of letting the LLM guess.
- **Output**: any PII pattern (email, phone, SSN, card-like number) the LLM echoes back
  from retrieved context is redacted before the user sees it.

## How monitoring & cost tracking work

Every query — blocked, denied, or answered — is appended to `logs/usage_log.csv` with
latency, tokens in/out, and estimated cost (`config.MODEL_PRICING_PER_1M`, update to
match current Groq pricing). `monitoring.check_daily_cost_alert()` prints a warning once
today's total crosses `DAILY_COST_ALERT_USD`. The Streamlit sidebar shows a live summary.

## Quickstart (local)

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and set GROQ_API_KEY (free key: https://console.groq.com)

python ingest.py        # builds the vector index from docs_*.md
pytest -v                # unit tests — no API key needed
python eval_run.py       # RBAC regression (no key needed) + generation smoke test (needs key)

streamlit run app.py     # opens the chatbot at http://localhost:8501
```

### Demo logins

| Username | Password | Role |
|---|---|---|
| alice.finance | finance123 | finance |
| bob.hr | hr123 | hr |
| carol.ceo | exec123 | executive |
| dave.eng | employee123 | employee |

Try asking `bob.hr` about marketing spend, or `dave.eng` about payroll — both should be
denied even though the documents exist in the index, because retrieval is filtered
before the LLM ever sees them.

## Running with Docker

```bash
cp .env.example .env   # set GROQ_API_KEY
docker compose up --build
```

## Deploying to Azure

```bash
export GROQ_API_KEY="gsk_..."
bash deploy_azure.sh
```

Reads `Dockerfile`, builds via ACR Tasks, and deploys to Azure Container Apps with the
Groq key stored as a Container Apps secret. See the note at the bottom of the script
about mounting persistent storage for `chroma_store/` and `logs/` in production.

## Continuous evaluation (the "whenever a new change is deployed" requirement)

Move `ci_workflow.yml` to `.github/workflows/ci.yml` in your repo and add `GROQ_API_KEY`
as a GitHub Actions secret. Every push then: rebuilds the index, runs the unit tests, and
runs `eval_run.py`'s RBAC regression check (fails the build if any role ever retrieves a
document outside its permitted departments) plus a generation smoke test. For deeper
LLM-judged quality metrics (faithfulness, answer relevancy), install
`requirements-eval.txt` and call `run_ragas_eval()` in `eval_run.py`.

## Known limitations / next hardening steps

- Auth is a hardcoded demo user table — replace with Azure AD/Entra ID before any real
  use.
- PII/guardrail detection is regex-based (fast, dependency-free, good for a demo) —
  for production, consider Microsoft Presidio or Azure AI Content Safety for more
  robust PII and harmful-content detection.
- Cost figures in `config.MODEL_PRICING_PER_1M` are estimates — verify against
  https://groq.com/pricing before relying on the cost alert for real budgets.
- `chroma_store/` and `logs/` are local disk by default — mount durable storage
  (Azure Files) if deploying with more than one replica or across restarts.
