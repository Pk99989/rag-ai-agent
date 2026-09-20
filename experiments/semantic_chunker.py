"""A structure-aware alternative to ingest.py's fixed-size sliding-window
chunker, for the Phase 10 experiments/ comparison.

Honest scope statement: this is ONE reasonable definition of "semantic
chunking" -- split on paragraph/header boundaries and greedily merge
consecutive small paragraphs up to a max size, instead of blindly cutting
every N characters regardless of what's in the middle of a sentence. It is
not claimed to be the only or the objectively best semantic-chunking
strategy; it exists so the comparison in run_experiments.py measures a real
alternative against real retrieval metrics, rather than a single scheme
being presented as though it were the only possible approach.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import chunk_text as fixed_window_chunk_text  # noqa: E402

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_HEADER_LINE_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def _split_into_paragraphs(text: str) -> list:
    """Split on blank-line paragraph boundaries; additionally force a
    boundary before any markdown header line so a header always starts a
    fresh paragraph rather than getting merged into the block above it."""
    text = _HEADER_LINE_RE.sub(lambda m: "\n\n" + m.group(0), text)
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text)]
    return [p for p in paragraphs if p]


def semantic_chunk_text(text: str, max_chunk_size: int = 800, min_chunk_size: int = 100) -> list:
    """Greedily merge consecutive paragraphs up to max_chunk_size. A single
    paragraph that alone exceeds max_chunk_size is NOT force-merged with
    anything -- it falls back to the existing fixed-window splitter
    (ingest.chunk_text) applied just to that paragraph, so no information
    is silently dropped for being "too semantic to fit"."""
    text = text.strip()
    if not text:
        return []
    paragraphs = _split_into_paragraphs(text)
    chunks, buffer = [], ""

    for para in paragraphs:
        if len(para) > max_chunk_size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(fixed_window_chunk_text(para, chunk_size=max_chunk_size, overlap=max_chunk_size // 6))
            continue

        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= max_chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = para

    if buffer.strip():
        chunks.append(buffer.strip())

    # Merge a too-small trailing chunk into the previous one rather than
    # leaving a near-empty final chunk (a real failure mode of naive
    # paragraph splitting, not a hypothetical one).
    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk_size:
        chunks[-2] = (chunks[-2] + "\n\n" + chunks[-1]).strip()
        chunks.pop()

    return chunks
