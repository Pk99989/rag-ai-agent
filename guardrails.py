"""Input/output guardrails: PII detection & redaction, prompt-injection and
out-of-scope query detection."""
import re

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I),
    re.compile(r"you are now (a|an) ", re.I),
    re.compile(r"reveal (your|the) system prompt", re.I),
    re.compile(r"disregard (your|the) (rules|guardrails)", re.I),
]

OUT_OF_SCOPE_HINTS = [
    "write me a poem", "tell me a joke", "who won the world cup",
    "current weather", "stock price of", "capital of france",
]


def detect_pii(text: str) -> dict:
    found = {}
    for label, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[label] = len(matches)
    return found


def redact_pii(text: str) -> str:
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted


def detect_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def looks_out_of_scope(text: str, retrieved_chunks: list) -> bool:
    """Heuristic: nothing was retrieved AND the query doesn't look like an internal-docs question."""
    if retrieved_chunks:
        return False
    lowered = text.lower()
    return any(hint in lowered for hint in OUT_OF_SCOPE_HINTS) or len(retrieved_chunks) == 0


def apply_input_guardrails(query: str) -> dict:
    """Run pre-retrieval checks. Returns {'blocked': bool, 'reason': str|None, ...}."""
    if detect_prompt_injection(query):
        return {"blocked": True, "reason": "prompt_injection_detected"}
    pii = detect_pii(query)
    if pii:
        return {"blocked": False, "reason": "pii_in_query", "pii": pii, "sanitized": redact_pii(query)}
    return {"blocked": False, "reason": None}


def apply_output_guardrails(answer: str) -> dict:
    """Run post-generation checks. Redacts any PII the LLM echoed back from retrieved context."""
    pii = detect_pii(answer)
    if pii:
        return {"redacted": True, "answer": redact_pii(answer), "pii": pii}
    return {"redacted": False, "answer": answer, "pii": {}}
