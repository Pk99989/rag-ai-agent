# Role-Based Access Controlled RAG AI Agent — RBAC RAG + Text-to-SQL over Real E-Commerce Data

A runnable implementation of the major project brief: an enterprise AI knowledge and
analytics assistant that answers questions from real Olist e-commerce data (Brazilian
Olist marketplace, via Kaggle) combining RAG over generated business-report documents
with Text-to-SQL over the underlying database, enforces role-based access control (RBAC),
applies guardrails, tracks cost, and is set up for cloud deployment and automated
evaluation.

Earlier iterations of this project used a fictional company ("AtliQ") with synthetic
`docs_<dept>_*.md` files as a placeholder document corpus. Those files and everything
that depended on them (the `hr` role, the `general`/`hr`/`executive` document
departments) have been removed -- the project is Olist-only from here on. See
`config.py`'s comments for the current RBAC design.

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
| `ingest.py` | Loads generated Olist business-report docs, chunks, embeds (local ONNX MiniLM), stores in ChromaDB |
| `guardrails.py` | PII detection/redaction, prompt-injection detection, out-of-scope detection |
| `rag_chain.py` | Retrieval (RBAC-filtered) → guardrails → Groq LLM → guardrails → log |
| `monitoring.py` | Per-query usage log (CSV), token cost estimate, daily cost-threshold alert |
| `api/main.py` | FastAPI backend (Phase 12): `/auth/login`, `/query`, `/me`, `/monitoring/summary`, `/health` |
| `api/security.py`, `api/dependencies.py` | JWT issuance/verification; role comes from the verified token, never a client-supplied value |
| `app.py` | Streamlit frontend (Phase 13): a pure API client over HTTP -- no direct import of RBAC/RAG/SQL internals |
| `scripts/generate_olist_rag_docs.py` | Generates the `data/processed/rag_documents/*.md` business-report corpus from `data/olist.db` |
| `scripts/generate_expense_docs.py` | Generates the synthetic expense-policy/vendor-profile corpus backing Phase 16 (clearly labeled as synthetic, not real) |
| `src/rag_agent/vision/document_qa.py`, `pdf_utils.py` | Phase 16: vision-LLM document extraction, injection/PII checks, RBAC-filtered retrieval, grounded answer |
| `eval_dataset.json`, `eval_run.py` | RBAC regression check + generation smoke test |
| `test_*.py` | Unit tests (pytest) for RBAC, guardrails, ingestion, SQL validation, the Phase 12 API's auth/RBAC boundary, and Phase 16's document Q&A logic |
| `Dockerfile.api`, `Dockerfile.ui`, `docker-entrypoint.api.sh`, `docker-compose.yml` | Two containers: API (backend, builds/reuses the index at startup) + UI (thin Streamlit client) |
| `Dockerfile` | Superseded, deliberately broken as a build target -- see the comment at its top |
| `deploy_azure.sh` | Deploy both services to Azure Container Apps (template — fill in and run yourself) |
| `ci_workflow.yml` | GitHub Actions CI gate — move to `.github/workflows/ci.yml` |

## How RBAC works

Each document is tagged with a `department` (from its frontmatter, e.g.
`department: payments`). `config.ROLE_ACCESS` maps each role to the set of real Olist
document departments it may see:

```
employee   -> (no document departments -- can still query general SQL tables)
sales      -> sales, products
finance    -> payments, revenue
operations -> sellers, delivery, orders
manager    -> sales, products, payments, revenue, sellers, delivery, orders
executive  -> sales, products, payments, revenue, sellers, delivery, orders
admin      -> sales, products, payments, revenue, sellers, delivery, orders
```

`rbac.build_chroma_filter(role)` turns that into a Chroma `where` clause, so the vector
search **never retrieves** unauthorized chunks — access is enforced at retrieval time,
not by asking the LLM nicely. A separate, table-level RBAC layer
(`config.SQL_TABLE_ACCESS`) gates which tables each role's Text-to-SQL queries may touch
(e.g. only finance/manager/executive/admin can query `payments`).

## How guardrails work

- **Input**: blocks obvious prompt-injection attempts ("ignore previous instructions...");
  flags PII typed into the query.
- **Retrieval**: if nothing relevant is retrieved (or the question matches out-of-scope
  patterns like "tell me a joke"), the bot declines instead of letting the LLM guess.
- **Output**: any PII pattern (email, phone, SSN, card-like number) the LLM echoes back
  from retrieved context is redacted before the user sees it.

## Multimodal document Q&A ("why was this charge deducted?") -- Phase 16

Upload a credit card statement, invoice, receipt, or expense report (image or PDF) and ask
why a charge was made. Available in the UI as a second tab, "📄 Explain a Charge", shown to
**every** role -- the endpoint itself has no RBAC gate, since reading and extracting fields
from a document a user just uploaded needs none. What's still role-restricted is which
supporting expense-policy/vendor documents ground the answer: `config.ROLE_ACCESS` maps the
`expense_docs` department to finance/manager/executive/admin only, enforced the same way as
every other route, via `hybrid_retrieve(query, role, ...)`. A sales/ops/employee user still
gets a real answer grounded in the document's own extracted fields -- it just won't cite
policy/vendor context that role isn't authorized to see, since retrieval returns zero
`expense_docs` chunks for it, not because anything special-cases the response.

Pipeline (`src/rag_agent/vision/document_qa.py`): input guardrails on the question → PDF
pages rendered to PNG (`pdf_utils.py`, PyMuPDF) or the image used directly → a vision LLM
(`qwen/qwen3.6-27b` on Groq, JSON mode) extracts structured fields (vendor, line items,
total, date) → the extraction output is scanned for embedded prompt injection, the same
threat model as Phase 8's document-borne injection check applied to an adversarial image →
extracted fields are PII-redacted → RBAC-filtered retrieval (`hybrid_retrieve`, same
function every other route uses) pulls supporting context from the `expense_docs`
department → a text LLM answers, grounded in the extracted fields plus retrieved policy/
vendor context → output guardrails → structured log.

The supporting corpus (`scripts/generate_expense_docs.py`) is explicitly **synthetic demo
data** -- fake expense policies and vendor profiles, clearly labeled as such in both the
file content and this README, not real company documents. There's no real company behind
this RBAC demo, so unlike the Olist corpus (generated from real Kaggle data), there was
nothing real to source this from; fabricating something and presenting it as real would
have violated this project's own no-fabrication rule, so it's disclosed instead.

`GROQ_VISION_MODEL`, `MAX_DOCUMENT_PAGES` (5), and `MAX_DOCUMENT_IMAGE_MB` (20) in
`config.py` mirror Groq's real current API limits for `qwen/qwen3.6-27b`, verified against
https://console.groq.com/docs/vision -- uploads are rejected with a clean 400 before ever
calling Groq if they'd exceed these, rather than paying for a request guaranteed to fail.

## How monitoring & cost tracking work

Every query — blocked, denied, or answered — is appended to `logs/usage_log.csv` with
latency, tokens in/out, and estimated cost (`config.MODEL_PRICING_PER_1M`, update to
match current Groq pricing). `monitoring.check_daily_cost_alert()` prints a warning once
today's total crosses `DAILY_COST_ALERT_USD`. The Streamlit sidebar shows a live summary.

## Quickstart (local)

Two processes now: the FastAPI backend (does all the real work) and the Streamlit UI
(a thin client that talks to it over HTTP). Run the backend first.

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: set GROQ_API_KEY (free key: https://console.groq.com), and set
# JWT_SECRET_KEY to a real value -- `python -c "import secrets; print(secrets.token_hex(32))"`
# (if you leave it blank, config.py generates a random one per process start, which is
# fine for a single local run but means restarting the backend logs everyone out)

python scripts/generate_olist_rag_docs.py  # builds the business-report .md corpus from data/olist.db
python scripts/generate_expense_docs.py    # builds the synthetic expense-policy/vendor corpus (Phase 16)
python ingest.py        # builds the vector index from all of the above
pytest -v                # unit tests — no API key needed, includes api/'s auth/RBAC tests
python eval_run.py       # RBAC regression (no key needed) + generation smoke test (needs key)

uvicorn api.main:app --host 0.0.0.0 --port 8000    # terminal 1: the backend
streamlit run app.py                                # terminal 2: the UI, at http://localhost:8501
```

The UI calls the backend at `API_BASE_URL` (default `http://localhost:8000`, set in `.env`).
If you see "Can't reach the API backend" on the login screen, the `uvicorn` process above
isn't running yet.

### Demo logins

| Username | Password | Role |
|---|---|---|
| alice.finance | finance123 | finance |
| carol.ceo | exec123 | executive |
| dave.eng | employee123 | employee |
| erin.sales | sales123 | sales |
| frank.ops | ops123 | operations |
| grace.mgr | manager123 | manager |
| henry.admin | admin123 | admin |

Try asking `erin.sales` about payment method summaries, or `dave.eng` (employee, no
document departments) about anything document-based — both should be denied even though
the documents exist in the index, because retrieval is filtered before the LLM ever sees
them.

## Running with Docker

```bash
cp .env.example .env   # set GROQ_API_KEY and JWT_SECRET_KEY
docker compose up --build
```

Builds and runs two containers: `api` (Dockerfile.api, port 8000) and `ui` (Dockerfile.ui,
port 8501, waits for `api`'s healthcheck before starting). The `api` container mounts your
local `data/` directory read-only and builds the vector index into the `chroma_store`
volume on first startup (see `docker-entrypoint.api.sh`) -- the image itself never bundles
the Kaggle dataset, matching the project's "never commit the dataset" rule.

## Deploying to Azure

```bash
export GROQ_API_KEY="gsk_..."
export JWT_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
bash deploy_azure.sh
```

Builds both images via ACR Tasks and deploys them as two Azure Container Apps (`api`
internal-only, `ui` public-facing, wired together via `API_BASE_URL`) sharing one
Container Apps environment. Creates Azure Files shares for the vector index, usage logs,
and your uploaded dataset. Read the note printed at the end of the script carefully --
attaching those Azure Files shares as actual volume mounts on the API container currently
needs one more `az containerapp update --yaml` step the script can't fully automate with
flat CLI flags; skipping it means the API starts with an empty index.

## Continuous evaluation (the "whenever a new change is deployed" requirement)

Move `ci_workflow.yml` to `.github/workflows/ci.yml` in your repo and add `GROQ_API_KEY`
as a GitHub Actions secret. Every push then: rebuilds the index, runs the unit tests, and
runs `eval_run.py`'s RBAC regression check (fails the build if any role ever retrieves a
document outside its permitted departments) plus a generation smoke test. For deeper
LLM-judged quality metrics (faithfulness, answer relevancy), install
`requirements-eval.txt` and call `run_ragas_eval()` in `eval_run.py`.

## Known limitations / next hardening steps

- Auth issues real, verified JWTs now (Phase 12) -- role escalation via a forged client
  request isn't possible -- but the underlying user store (`auth.py`) is still a
  hardcoded demo table. Replace `auth.authenticate()`'s backing store with Azure AD/Entra
  ID before any real use; nothing else in `api/` needs to change to support that, since
  it only depends on `authenticate()` returning a `User(username, name, role)`.
- Rate limiting (`slowapi`, `config.API_RATE_LIMIT_PER_MINUTE`) is in-memory and
  per-process -- fine for one instance, not correct once you scale to multiple API
  replicas (each would track its own limit independently). Use a shared backend (Redis)
  for real multi-instance rate limiting.
- PII/guardrail detection is regex-based (fast, dependency-free, good for a demo) —
  for production, consider Microsoft Presidio or Azure AI Content Safety for more
  robust PII and harmful-content detection.
- Cost figures in `config.MODEL_PRICING_PER_1M` are estimates — verify against
  https://groq.com/pricing before relying on the cost alert for real budgets.
- `chroma_store/` and `logs/` are local disk by default — mount durable storage
  (Azure Files) if deploying with more than one replica or across restarts.
- Phase 16's `/query/document` endpoint has real unit-test coverage (file-type/size
  validation, RBAC gating, PDF rendering, PII redaction of extracted fields -- see
  `test_api.py`/`test_document_qa.py`) but is NOT yet included in `evaluation/eval_dataset.json`
  or `evaluation/run_evaluation.py`'s 100-question harness, which predates this phase and only
  covers RAG/SQL/RBAC/SECURITY/GENERAL. Extending that harness with a DOCUMENT category
  (needs real or synthetic sample statement/invoice images, not just text) is the natural
  next step before trusting this feature's end-to-end accuracy the way the rest of the
  project's numbers are trusted.
