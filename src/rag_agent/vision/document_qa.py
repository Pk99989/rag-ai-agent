"""Phase 16: multimodal document Q&A -- "why was this charge deducted?"

Pipeline: input guardrails on the question -> convert upload to page images
-> vision LLM extracts structured fields (JSON mode) -> scan the vision
model's own output for injected instructions (a malicious/adversarial image
is a real, documented attack surface for vision LLMs, same threat class as
Phase 8's document-borne injection check for RAG chunks) -> PII-redact the
extracted fields (a real statement/invoice legitimately contains
account-number-shaped digit runs) -> RBAC-filtered retrieval of supporting
expense-policy/vendor context (config.py's expense_docs department, same
hybrid_retrieve() every other route already uses -- no separate retrieval
code needed) -> grounded text-LLM answer -> output guardrails -> structured
log.

RBAC note: this module does NOT re-implement or duplicate access control.
The /query/document endpoint (api/main.py) is open to every authenticated
role -- reading and extracting fields from a document a user just uploaded
needs no RBAC of its own. What IS still role-restricted is which supporting
expense-policy/vendor documents ground the answer: config.ROLE_ACCESS maps
the expense_docs department to finance/manager/executive/admin only, and
this module calls hybrid_retrieve(query, role, ...) exactly like
rag_chain.py does for every other route. A caller with an unauthorized role
therefore still gets an answer grounded in the extracted document fields --
it just retrieves zero expense_docs chunks, so the answer won't cite
policy/vendor context it isn't authorized to see. Same
identical-response-whether-unauthorized-or-nonexistent property the RAG
route already relies on.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_VISION_MODEL, GROQ_VISION_MAX_OUTPUT_TOKENS,
    RERANK_TOP_K,
)
from guardrails import (
    apply_input_guardrails, apply_output_guardrails, redact_pii,
    detect_prompt_injection,
)
from rag_agent.retrieval.hybrid import hybrid_retrieve_with_timing
from rag_agent.vision.pdf_utils import pdf_bytes_to_png_images, PDFConversionError

SUPPORTED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
SUPPORTED_PDF_CONTENT_TYPE = "application/pdf"

EXTRACTION_SCHEMA_PROMPT = """\
You are analyzing an image of a financial document (credit card statement, \
invoice, receipt, or expense report). Return ONLY a JSON object with this \
exact shape, no other text:

{{
  "vendor_name": string or null,
  "document_type": one of "credit_card_statement", "invoice", "receipt", \
"expense_report", "unknown",
  "line_items": [{{"description": string, "amount": string}}],
  "subtotal": string or null,
  "tax": string or null,
  "total_amount": string or null,
  "billing_date": string or null,
  "identified_charge_description": string or null
}}

If the user's question refers to a specific highlighted or otherwise \
identifiable charge, put that one line item's description in \
identified_charge_description. Use null for any field you cannot read \
from the image with reasonable confidence -- do not guess or invent a \
value. Keep all monetary amounts as the exact strings shown in the image \
(with currency symbol if present), not converted or rounded.

User's question: {question}
"""

ANSWER_SYSTEM_PROMPT = """\
You are a finance assistant that explains charges on financial documents \
(credit card statements, invoices, receipts, expense reports) for an \
internal finance/operations team. You are given: (1) fields extracted from \
the uploaded document by a vision model, and (2) supporting context from \
the company's expense policy and vendor-profile documents, retrieved \
because they matched the extracted vendor/charge. Answer the user's \
question using ONLY this information. If the extracted fields or retrieved \
context don't contain enough to answer confidently, say so explicitly \
rather than guessing. Cite which policy/vendor document you used, if any.\
"""


class DocumentQAError(Exception):
    """Raised for a real, user-facing failure before any LLM call happens
    (unsupported file type, oversized upload, unparseable PDF) -- kept
    distinct from a blocked/refused *result* (guardrail-triggered, which is
    a normal successful call that just returns blocked=True) so the API
    layer can tell "your upload is bad" (400) apart from "here's your
    answer, guardrails withheld it" (200 with blocked=True)."""


def _get_vision_llm():
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
    # Root cause found via a real captured raw_text (see document_qa's
    # _parse_json_object fallback, added specifically to make this
    # diagnosable): qwen/qwen3.6-27b is a reasoning model that emits a
    # <think>...</think> block before its real answer. With
    # response_format=json_object set, Groq's server-side JSON validator
    # saw that <think>-prefixed content, rejected it outright, and
    # discarded it entirely (400 json_validate_failed, failed_generation
    # empty -- we never got to see any of this until the fallback above
    # exposed it). Confirmed against Groq's own docs
    # (https://console.groq.com/docs/reasoning,
    # https://console.groq.com/docs/model/qwen/qwen3.6-27b), not guessed:
    # qwen3.x models accept reasoning_effort="none" to disable thinking
    # entirely, and JSON mode requires reasoning_format to be "hidden" or
    # "parsed" (never "raw") -- passing both here fixes this at the
    # source, and response_format=json_object is safe to restore now that
    # there's no reasoning output left to conflict with it.
    # _parse_json_object's defensive fallback stays regardless -- cheap
    # insurance against a model that still wraps its answer in prose.
    # reasoning_effort/reasoning_format are first-class ChatGroq
    # constructor fields, NOT model_kwargs entries -- langchain_groq's
    # pydantic validation explicitly rejects passing them via model_kwargs
    # (a real error hit in live testing: "Parameters {'reasoning_effort',
    # 'reasoning_format'} should be specified explicitly"), which is why
    # only response_format stays inside model_kwargs below.
    return ChatGroq(
        model=GROQ_VISION_MODEL, api_key=GROQ_API_KEY, temperature=0.0,
        max_tokens=GROQ_VISION_MAX_OUTPUT_TOKENS,
        reasoning_effort="none",
        reasoning_format="hidden",
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _get_text_llm():
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file (see .env.example).")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def images_from_upload(file_bytes: bytes, content_type: str, max_pages: int) -> list:
    """Returns a list of PNG image bytes for the vision LLM. Raises
    DocumentQAError for anything that isn't a genuinely usable upload."""
    if content_type == SUPPORTED_PDF_CONTENT_TYPE:
        try:
            return pdf_bytes_to_png_images(file_bytes, max_pages=max_pages)
        except PDFConversionError as exc:
            raise DocumentQAError(str(exc)) from exc
    if content_type in SUPPORTED_IMAGE_CONTENT_TYPES:
        return [file_bytes]
    raise DocumentQAError(
        f"Unsupported file type '{content_type}'. Upload a PNG/JPEG/WebP image or a PDF."
    )


def _image_message_parts(images_png: list) -> list:
    import base64
    parts = []
    for img_bytes in images_png:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return parts


def _extract_fields(images_png: list, question: str) -> dict:
    """Calls the vision LLM once with all page images + the extraction
    prompt, parses its JSON response. Returns
    {"fields": dict|None, "raw_text": str, "error": str|None}."""
    from langchain_core.messages import HumanMessage

    llm = _get_vision_llm()
    content = [{"type": "text", "text": EXTRACTION_SCHEMA_PROMPT.format(question=question)}]
    content.extend(_image_message_parts(images_png))
    try:
        response = llm.invoke([HumanMessage(content=content)])
    except Exception as exc:
        # Real, observed failure mode, not a parsing bug on our side: Groq's
        # own JSON-mode validator can reject the vision model's generation
        # outright (400 json_validate_failed, failed_generation empty --
        # i.e. Groq itself never got usable content back from the model for
        # this image) before anything reaches us to parse. Needs the same
        # graceful degradation as a parse failure below; left uncaught,
        # this exception was propagating all the way to the API layer's
        # generic except-Exception handler and surfacing as an opaque 500
        # instead of the user-facing "try a clearer image" message every
        # other extraction failure already gets. Full exception logged
        # server-side only -- the client-facing reason stays a fixed code,
        # not Groq's raw error text.
        print(f"[VISION API ERROR] {exc!r}")
        return {"fields": None, "raw_text": "", "error": "vision_api_error"}
    raw_text = response.content

    fields = _parse_json_object(raw_text)
    if fields is None:
        # Now that response_format=json_object no longer discards the
        # model's real output server-side, log it (truncated -- it's
        # untrusted model output, not something to dump unbounded into
        # server logs) so a parse failure is diagnosable instead of another
        # dead end. Server-side only, same reasoning as the vision API
        # error print above -- never returned to the client.
        preview = raw_text[:300] if isinstance(raw_text, str) else repr(raw_text)
        print(f"[VISION PARSE FAILURE] raw_text (first 300 chars)={preview!r}")
        return {"fields": None, "raw_text": raw_text, "error": "extraction_json_parse_failed"}
    return {"fields": fields, "raw_text": raw_text, "error": None}


def _parse_json_object(raw_text) -> dict | None:
    """Best-effort JSON extraction from a vision LLM's free-form response.

    Without response_format=json_object enforcing it server-side (see
    _get_vision_llm's comment on why that's no longer used for the vision
    call), a real model can still wrap its JSON in markdown code fences or
    add a stray sentence before/after it despite being asked not to. Try
    the raw text first (the common, well-behaved case), then fall back to
    stripping ```json fences, then fall back to slicing out the substring
    between the first '{' and the last '}' -- progressively more lenient,
    each step only attempted if the previous one failed to parse."""
    if not isinstance(raw_text, str):
        return None

    candidates = [raw_text.strip()]

    stripped = raw_text.strip()
    if stripped.startswith("```"):
        # Drop a leading ```json / ``` line and a trailing ``` line.
        without_fence = stripped.strip("`")
        if without_fence.lower().startswith("json"):
            without_fence = without_fence[4:]
        candidates.append(without_fence.strip())

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(raw_text[first_brace:last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _redact_fields(fields: dict) -> dict:
    """Runs every string value in the extracted fields through the same
    redact_pii() used everywhere else in this project, so a full card
    number or account number the vision model transcribed verbatim from
    the image never reaches the answer, the logs, or the client."""
    def _redact(value):
        if isinstance(value, str):
            return redact_pii(value)
        if isinstance(value, list):
            return [_redact(v) for v in value]
        if isinstance(value, dict):
            return {k: _redact(v) for k, v in value.items()}
        return value
    return _redact(fields)


def analyze_document(file_bytes: bytes, content_type: str, question: str, user, role: str,
                      max_pages: int = 5) -> dict:
    start = time.time()

    input_check = apply_input_guardrails(question)
    if input_check["blocked"]:
        return {
            "answer": "This request was blocked by guardrails and cannot be processed.",
            "extracted_fields": None, "citations": [], "confidence": "low",
            "blocked": True, "reason": input_check["reason"],
            "vision_latency_ms": None, "retrieval_latency_ms": None,
            "reranking_latency_ms": None, "generation_latency_ms": None,
        }

    images_png = images_from_upload(file_bytes, content_type, max_pages)

    vision_start = time.time()
    extraction = _extract_fields(images_png, question)
    vision_latency_ms = round((time.time() - vision_start) * 1000, 2)

    if extraction["error"]:
        return {
            "answer": "The document could not be read reliably -- the vision model's response "
                      "wasn't valid structured output. Try a clearer image or a different page.",
            "extracted_fields": None, "citations": [], "confidence": "low",
            "blocked": False, "reason": extraction["error"],
            "vision_latency_ms": vision_latency_ms, "retrieval_latency_ms": None,
            "reranking_latency_ms": None, "generation_latency_ms": None,
        }

    # Phase 8's threat model extended to images: an adversarial document
    # (a receipt with hidden/tiny text saying "ignore instructions...")
    # could plant an instruction the vision model then transcribes
    # verbatim into raw_text. Treat that exactly like a flagged RAG chunk
    # -- refuse rather than let it flow into the answer-generation prompt.
    if detect_prompt_injection(extraction["raw_text"]):
        return {
            "answer": "This document could not be processed: it appears to contain embedded "
                      "text designed to manipulate the assistant, which was blocked.",
            "extracted_fields": None, "citations": [], "confidence": "low",
            "blocked": True, "reason": "document_borne_injection_detected",
            "vision_latency_ms": vision_latency_ms, "retrieval_latency_ms": None,
            "reranking_latency_ms": None, "generation_latency_ms": None,
        }

    fields = _redact_fields(extraction["fields"])

    retrieval_query = " ".join(str(v) for v in [
        fields.get("vendor_name"), fields.get("identified_charge_description"), question,
    ] if v)
    chunks, retrieval_timing = hybrid_retrieve_with_timing(retrieval_query, role, top_k=RERANK_TOP_K)
    citations = sorted({m["source"] for _, m, _ in chunks}) if chunks else []
    context_text = "\n\n".join(f"[Source: {m['source']}]\n{d}" for d, m, _ in chunks)

    llm = _get_text_llm()
    gen_start = time.time()
    prompt = (
        f"Extracted document fields (JSON):\n{json.dumps(fields, indent=2)}\n\n"
        f"Supporting policy/vendor context:\n{context_text or '(none retrieved)'}\n\n"
        f"Question: {question}"
    )
    response = llm.invoke([("system", ANSWER_SYSTEM_PROMPT), ("human", prompt)])
    generation_latency_ms = round((time.time() - gen_start) * 1000, 2)

    output_check = apply_output_guardrails(response.content, system_prompt=ANSWER_SYSTEM_PROMPT)

    result = {
        "answer": output_check["answer"],
        "extracted_fields": None if output_check["blocked"] else fields,
        "citations": [] if output_check["blocked"] else citations,
        "confidence": "low" if output_check["blocked"] else ("high" if citations else "medium"),
        "blocked": output_check["blocked"],
        "reason": ("output_guardrail_" + "_and_".join(output_check["flags"])) if output_check["flags"] else None,
        "vision_latency_ms": vision_latency_ms,
        "retrieval_latency_ms": retrieval_timing["retrieval_ms"],
        "reranking_latency_ms": retrieval_timing["reranking_ms"],
        "generation_latency_ms": generation_latency_ms,
    }
    return result
