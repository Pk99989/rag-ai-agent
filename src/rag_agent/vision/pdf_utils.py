"""PDF -> page images, for feeding a multi-page financial document PDF
(credit card statement, invoice) to a vision LLM that only accepts images.

Uses PyMuPDF (fitz) -- a real, widely-used PDF rendering library, not a
hand-rolled parser. Deliberately does NOT use Docling here despite the
project brief mentioning it: Docling is a much heavier dependency (its own
layout-model downloads) aimed at structured document->markdown extraction,
which is more than this feature's "core loop" scope needs when the vision
LLM itself is already doing layout understanding directly on page images.
Revisit if a later phase needs Docling's structured chunking specifically.
"""
import pymupdf as fitz  # `import fitz` directly is deprecated as of PyMuPDF 1.28


class PDFConversionError(Exception):
    """Raised when a PDF can't be parsed/rendered -- e.g. corrupt file,
    password-protected, or zero pages. Callers should surface this as a
    clear user-facing error, not a generic 500."""


def pdf_bytes_to_png_images(pdf_bytes: bytes, max_pages: int, dpi: int = 150) -> list:
    """Renders up to max_pages pages of a PDF to PNG image bytes, in order.
    Raises PDFConversionError on anything that isn't a normal, open-able,
    non-empty PDF -- silently returning an empty list would let a caller
    mistake "this PDF is broken" for "this PDF legitimately has no pages,"
    which is not a real case for a financial statement."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFConversionError(f"Could not open PDF: {exc}") from exc

    if doc.is_encrypted:
        doc.close()
        raise PDFConversionError("PDF is password-protected; cannot render pages.")
    if doc.page_count == 0:
        doc.close()
        raise PDFConversionError("PDF has zero pages.")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    try:
        for page_index in range(min(doc.page_count, max_pages)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images
