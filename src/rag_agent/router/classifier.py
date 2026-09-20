"""Deterministic query router.

Honest scope statement: this is keyword/phrase heuristics, not a trained
classifier or an LLM call -- deliberately, per the spec's own instruction
not to rely exclusively on an LLM for routing. It is fast (no network
call), free, and fully unit-testable without an API key, at the cost of
being fooled by phrasing it has no keyword for. The confidence score
reflects how many domain signals were matched, not a calibrated
probability -- treat "confidence: 0.9" as "matched several strong SQL
signals," not as "90% chance this is correct."

Phase 11 fix (found by live execution, not hand-review): SQL_SIGNALS used
to be enough on its own to route to SQL. But phrases like "revenue" and
"how many" are ordinary business English, not evidence a question is about
THIS Olist database specifically -- "What was AtliQ's total revenue in
FY2025?" and "How many days of paid annual leave do employees get?" (both
real company-document RAG questions) matched bare SQL_SIGNALS and were
routed to Text-to-SQL, which then tried to write SQL against an e-commerce
schema that has nothing to do with either question. Fix: an ENTITY_NOUNS
anchor (a real schema concept -- order/customer/seller/product/payment/
review/category/delivery/freight/installment) is now required before any
SQL_SIGNALS phrase counts for anything.

This introduces a known, structural (not accidental) ambiguity of its own:
Phase 5's RAG documents are themselves aggregate SUMMARIES of the same
Olist data Text-to-SQL can query live (e.g. "payment method summary for
credit card" vs "total payment value where payment_type = credit_card" ask
for the same fact two different ways) -- a question with an entity anchor
but no specific SQL_SIGNALS phrase and no RAG_SIGNALS phrase is genuinely
ambiguous between "read the existing summary doc" and "compute it live,"
and is classified HYBRID rather than forced either way. dispatch.py's
HYBRID handling tries RAG first, which naturally prefers an existing
aggregate document when one answers the question, falling back to SQL only
if RAG finds nothing -- this is a deliberate design choice for that
overlap, not an unresolved bug.

If routing accuracy on real traffic turns out to need more than keyword
matching, the natural next step is an LLM tie-breaker ONLY for the
low-confidence middle band this module already identifies -- not replacing
this module, since most real questions should never need an LLM call just
to decide where to send them.
"""
import re
from dataclasses import dataclass

from guardrails import OUT_OF_SCOPE_HINTS

RAG = "RAG"
SQL = "SQL"
HYBRID = "HYBRID"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# Phrases/words that indicate a structured, numerical, Olist-database
# question. Kept as substrings (not exact word-boundary regex) deliberately
# -- "which sellers" should still match a "which seller" signal. On their
# own these are NOT sufficient for SQL routing anymore (see module
# docstring) -- they only count once ENTITY_NOUNS below confirms the
# question is actually about this database.
SQL_SIGNALS = [
    "revenue", "average", "avg ", "total sales", "total revenue", "sum of",
    "how many", "highest", "lowest", "top ", "monthly sales", "month over month",
    "review score", "delivery time", "delivery days", "on time", "late delivery",
    "most common", "percentage", "growth", "trend", "compare", "ranking",
    "which category", "which categories", "which seller", "which state",
    "which states", "which payment", "order value", "number of orders",
    "num orders", "per month", "per state", "per category", "best selling",
    "worst performing", "count of", "how much revenue",
]

# Real entities from the Olist schema (src/rag_agent/sql/schema.py's
# ALLOWED_TABLES, in singular-stem form so "order"/"orders"/"ordered" all
# match via substring). Deliberately does NOT include generic words like
# "state" or "category" alone without a table-name-shaped stem, or purely
# quantitative words -- those belong in SQL_SIGNALS, which requires one of
# these anchors to actually mean anything.
ENTITY_NOUNS = [
    "order", "customer", "seller", "product", "payment", "review",
    "categor", "deliver", "freight", "installment", "database", "olist",
]

# Phrases/words that indicate an unstructured, document/policy question.
# The "policy"/"handbook"/"onboarding"/"company values" style entries below
# predate the Olist-only pivot (they matched the deleted AtliQ HR/policy
# docs) and are now inert -- kept rather than deleted since they're harmless
# and would apply again if a future document set reintroduces that kind of
# content, per this project's convention of not deleting working mechanisms
# without a functional reason.
#
# Olist-pivot bug found by REAL execution of scripts/smoke_test_router_accuracy.py
# against the rewritten eval_dataset.json's RAG items (not hand-review): 9 of
# the 20 RAG questions phrased as "According to the seller performance
# report..." / "...performance reports, which categories..." were hard-
# misrouted to SQL. Why: those questions legitimately contain both a real
# entity anchor (seller/order/categor/review/deliver) AND a SQL_SIGNALS
# phrase (revenue/how many/average/most common/on time), which is exactly
# the aggregate-RAG-doc-vs-live-SQL overlap this module's docstring already
# describes -- but with no RAG_SIGNALS phrase present, classify() committed
# straight to SQL instead of the honest HYBRID/ambiguous result. Adding
# "according to the" and "performance report" (substrings covering "...
# performance reports" too) restores rag_hits for these questions, which
# (given they also have sql_phrase_hits AND an anchor) routes them to
# HYBRID rather than a hard SQL commit -- dispatch.py's HYBRID handling then
# tries RAG first and correctly finds the real generated report document.
# Verified this fixes all 9 hard misroutes with zero regressions against
# both scripts/smoke_test_router.py's 9 cases and the other 31 previously-
# correct eval_dataset.json items (re-run after this change, not assumed).
RAG_SIGNALS = [
    "policy", "handbook", "procedure", "guideline", "guidelines", "process for",
    "what is our", "how do we", "onboarding", "compliance", "terms of service",
    "agreement", "documentation", "benefits", "leave policy", "code of conduct",
    "mission", "company values", "faq", "return policy", "refund policy",
    "warranty", "how does support", "escalation process",
    "according to the", "performance report", "sales report", "status summary",
]


@dataclass
class RouteResult:
    route: str
    confidence: float
    matched_sql_signals: list
    matched_rag_signals: list

    def as_dict(self) -> dict:
        return {"route": self.route, "confidence": self.confidence}


def _matches(question_lower: str, signals: list) -> list:
    return [s.strip() for s in signals if s in question_lower]


def _has_entity_anchor(question_lower: str) -> bool:
    return any(noun in question_lower for noun in ENTITY_NOUNS)


def classify(question: str) -> RouteResult:
    q = question.lower()

    sql_phrase_hits = _matches(q, SQL_SIGNALS)
    rag_hits = _matches(q, RAG_SIGNALS)
    has_anchor = _has_entity_anchor(q)

    # A curated RAG phrase ("policy", "FAQ", "handbook"...) is deliberate,
    # specific evidence of document intent -- it outranks a bare entity
    # mention. "What is the seller policy?" mentions "seller" (an entity
    # noun) but is unambiguously a document question, not a live query
    # about seller data; without this ordering, that case regressed to
    # HYBRID (caught by re-running scripts/smoke_test_router.py's existing
    # 9 cases against this fix before shipping it, not left for later).
    if rag_hits and sql_phrase_hits and has_anchor:
        # Real evidence on both sides at once -- genuinely mixed.
        confidence = round(min(0.9, 0.5 + 0.05 * (len(sql_phrase_hits) + len(rag_hits))), 2)
        return RouteResult(HYBRID, confidence, sql_phrase_hits, rag_hits)

    if rag_hits:
        return RouteResult(RAG, round(min(0.95, 0.6 + 0.1 * len(rag_hits)), 2), [], rag_hits)

    if has_anchor and sql_phrase_hits:
        confidence = round(min(0.95, 0.65 + 0.1 * len(sql_phrase_hits)), 2)
        return RouteResult(SQL, confidence, sql_phrase_hits, [])

    # Anchored, but no specific phrase either way -- see module docstring's
    # note on RAG-aggregate-doc vs live-SQL overlap. Not a hard commit.
    if has_anchor:
        return RouteResult(HYBRID, 0.5, sql_phrase_hits, [])

    if any(hint in q for hint in OUT_OF_SCOPE_HINTS):
        return RouteResult(OUT_OF_SCOPE, 0.9, [], [])

    # No domain signal at all -- default to RAG rather than SQL, since a
    # wrong RAG route degrades to "I don't have that information," while
    # a wrong SQL route risks the LLM inventing a plausible-looking but
    # meaningless query against tables that have nothing to do with the
    # question. Low confidence is the honest signal that this is a guess.
    return RouteResult(RAG, 0.4, [], [])
