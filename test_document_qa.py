"""Tests for the non-LLM logic in the Phase 16 multimodal document Q&A
pipeline: file-type routing, PDF rendering, PII redaction of extracted
fields. Nothing here calls Groq -- that's covered by manual/real-execution
testing (needs a live key + a real image), same split as eval_run.py's
generation smoke test vs. its deterministic RBAC check.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_agent.vision import document_qa  # noqa: E402
from rag_agent.vision.document_qa import (  # noqa: E402
    images_from_upload, _redact_fields, _extract_fields, _parse_json_object,
    DocumentQAError, SUPPORTED_IMAGE_CONTENT_TYPES, SUPPORTED_PDF_CONTENT_TYPE,
    EXTRACTION_SCHEMA_PROMPT,
)
from rag_agent.vision.pdf_utils import pdf_bytes_to_png_images, PDFConversionError  # noqa: E402


def test_extract_fields_handles_groq_api_error_gracefully(monkeypatch):
    # Regression test for a real failure hit in live testing: Groq's own
    # JSON-mode validator can reject the vision model's generation outright
    # (400 json_validate_failed, failed_generation empty) and langchain_groq
    # raises that as an exception from llm.invoke() -- not a JSON parse
    # failure on our side, a hard exception. Before this test existed,
    # nothing caught it: it propagated out of _extract_fields(), out of
    # analyze_document(), and the API layer's generic except-Exception
    # handler turned it into an opaque 500 instead of the graceful
    # "try a clearer image" message every other extraction failure gets.
    class _FakeLLM:
        def invoke(self, messages):
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"Failed to validate "
                "JSON...\", 'code': 'json_validate_failed', 'failed_generation': ''}}"
            )

    monkeypatch.setattr(document_qa, "_get_vision_llm", lambda: _FakeLLM())
    result = _extract_fields([b"fake-image-bytes"], "Why was this charged?")
    assert result["fields"] is None
    assert result["error"] == "vision_api_error"
    # The raw Groq exception text must NOT leak into the client-facing
    # error code -- it's logged server-side only (see the print() in
    # _extract_fields), not returned to the caller.
    assert "json_validate_failed" not in result["error"]


def test_extraction_schema_prompt_formats_without_error():
    # Regression test: EXTRACTION_SCHEMA_PROMPT is filled in with
    # .format(question=...) in _extract_fields(), but the prompt also
    # contains a literal JSON example with its own { } characters. Left
    # unescaped, .format() misreads those as field references and raises
    # KeyError('\n  "vendor_name"') -- a real bug that shipped because
    # nothing in the sandbox could actually call _extract_fields() (no
    # network access to Groq), so this line never executed until a live
    # test did. Calling .format() directly here needs no LLM and no
    # network, so it catches this class of bug for good.
    prompt = EXTRACTION_SCHEMA_PROMPT.format(question="Why was this charged?")
    assert "Why was this charged?" in prompt
    assert '"vendor_name"' in prompt  # the JSON example survived escaping intact


def test_images_from_upload_rejects_unsupported_type():
    with pytest.raises(DocumentQAError):
        images_from_upload(b"hello", "text/plain", max_pages=5)


def test_images_from_upload_passes_through_supported_image_types():
    for content_type in SUPPORTED_IMAGE_CONTENT_TYPES:
        result = images_from_upload(b"fake-image-bytes", content_type, max_pages=5)
        assert result == [b"fake-image-bytes"]


def test_images_from_upload_rejects_broken_pdf():
    with pytest.raises(DocumentQAError):
        images_from_upload(b"not a real pdf", SUPPORTED_PDF_CONTENT_TYPE, max_pages=5)


def test_pdf_bytes_to_png_images_rejects_garbage_bytes():
    with pytest.raises(PDFConversionError):
        pdf_bytes_to_png_images(b"this is not a pdf file at all", max_pages=5)


def test_pdf_bytes_to_png_images_renders_a_real_minimal_pdf():
    # A real, valid, single-page, blank PDF -- built with PyMuPDF itself
    # rather than hand-crafting PDF bytes, so this test exercises the same
    # library pdf_utils.py uses, against real (if trivial) PDF structure.
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test invoice $42.00")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    pdf_bytes = buf.getvalue()

    images = pdf_bytes_to_png_images(pdf_bytes, max_pages=5)
    assert len(images) == 1
    assert images[0][:8] == b"\x89PNG\r\n\x1a\n"  # real PNG file signature


def test_pdf_bytes_to_png_images_respects_max_pages():
    import pymupdf as fitz
    doc = fitz.open()
    for i in range(4):
        doc.new_page()
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    images = pdf_bytes_to_png_images(buf.getvalue(), max_pages=2)
    assert len(images) == 2


def test_redact_fields_masks_pii_in_string_values():
    fields = {
        "vendor_name": "Contact billing at jane.doe@example.com",
        "total_amount": "$42.00",
    }
    redacted = _redact_fields(fields)
    assert "jane.doe@example.com" not in redacted["vendor_name"]
    assert redacted["total_amount"] == "$42.00"  # no PII pattern here, unchanged


def test_redact_fields_recurses_into_line_items_list():
    fields = {
        "line_items": [
            {"description": "Call 555-123-4567 for support", "amount": "$10.00"},
        ],
    }
    redacted = _redact_fields(fields)
    assert "555-123-4567" not in redacted["line_items"][0]["description"]


def test_redact_fields_leaves_none_values_alone():
    fields = {"vendor_name": None, "billing_date": None}
    assert _redact_fields(fields) == {"vendor_name": None, "billing_date": None}


def test_parse_json_object_handles_clean_json():
    raw = '{"vendor_name": "AWS", "total_amount": "$339.75"}'
    assert _parse_json_object(raw) == {"vendor_name": "AWS", "total_amount": "$339.75"}


def test_parse_json_object_handles_markdown_fenced_json():
    # A real, documented model behavior even when explicitly told not to:
    # wrapping JSON in a ```json ... ``` fence. response_format=json_object
    # would have prevented this server-side, but that's no longer forced
    # for the vision call (see _get_vision_llm's comment on why) -- this is
    # exactly the case the fallback exists for.
    raw = '```json\n{"vendor_name": "Netflix", "total_amount": "$15.99"}\n```'
    assert _parse_json_object(raw) == {"vendor_name": "Netflix", "total_amount": "$15.99"}


def test_parse_json_object_handles_json_with_surrounding_prose():
    raw = 'Sure, here is the extracted data:\n{"vendor_name": "GreenLeaf"}\nLet me know if you need more.'
    assert _parse_json_object(raw) == {"vendor_name": "GreenLeaf"}


def test_parse_json_object_returns_none_for_non_json_text():
    assert _parse_json_object("I cannot process this image.") is None


def test_parse_json_object_returns_none_for_empty_string():
    assert _parse_json_object("") is None


def test_parse_json_object_rejects_non_dict_json():
    # A JSON array or bare string is technically valid JSON but not a
    # usable extraction result -- must not be returned as "fields".
    assert _parse_json_object('["vendor", "AWS"]') is None
