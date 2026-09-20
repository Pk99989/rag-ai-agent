"""Cross-encoder reranker via onnxruntime -- deliberately not
sentence-transformers/torch, per project convention (see requirements.txt).

Model + tokenizer come from Xenova/ms-marco-MiniLM-L-6-v2 on the Hugging
Face Hub: an ONNX export of the standard ms-marco cross-encoder, downloaded
once via huggingface_hub and cached locally (~/.cache/huggingface) -- no
model file bundled in this repo.

This has NOT been executed by the assistant that wrote it (sandbox
outage) -- the input/output tensor names are read defensively from the
ONNX session at load time rather than hardcoded, specifically because I
cannot inspect the real model file myself to confirm exact names in
advance. Run scripts/smoke_test_hybrid.py and paste back the output,
including any traceback -- a wrong tensor/input name would fail loudly
here, not silently produce bad scores.
"""
import numpy as np

from config import RERANKER_MODEL_REPO, RERANKER_MAX_LENGTH

_session = None
_tokenizer = None


def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    model_path = hf_hub_download(RERANKER_MODEL_REPO, "onnx/model.onnx")
    tokenizer_path = hf_hub_download(RERANKER_MODEL_REPO, "tokenizer.json")

    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    _tokenizer.enable_truncation(max_length=RERANKER_MAX_LENGTH)
    _tokenizer.enable_padding(length=RERANKER_MAX_LENGTH)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_pairs(query: str, passages: list) -> list:
    """Returns one relevance score per passage (same order as input),
    roughly in [0,1] after sigmoid -- higher = more relevant to the query."""
    _load()
    if not passages:
        return []

    encodings = _tokenizer.encode_batch([(query, p) for p in passages])
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    available_inputs = {i.name for i in _session.get_inputs()}
    feed = {}
    for name, array in (("input_ids", input_ids), ("attention_mask", attention_mask),
                         ("token_type_ids", type_ids)):
        if name in available_inputs:
            feed[name] = array
    missing = available_inputs - set(feed)
    if missing:
        raise RuntimeError(
            f"ONNX model expects input(s) {sorted(missing)} that this code doesn't provide -- "
            f"the model's actual inputs are {sorted(available_inputs)}. Update reranker.py's "
            f"feed dict to match."
        )

    outputs = _session.run(None, feed)
    logits = np.asarray(outputs[0])

    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim == 2 and logits.shape[1] == 2:
        logits = logits[:, 1] - logits[:, 0]  # 2-class: relevant - not_relevant
    elif logits.ndim != 1:
        raise RuntimeError(
            f"unexpected ONNX output shape {logits.shape} -- expected a 1D score per "
            f"passage or a 2D (N,1)/(N,2) logits array."
        )

    return _sigmoid(logits).tolist()


def rerank(query: str, candidates: list, top_k: int) -> list:
    """candidates: list of (text, metadata, fused_score) from hybrid fusion.
    Returns up to top_k re-ordered by cross-encoder relevance, as
    (text, metadata, rerank_score)."""
    if not candidates:
        return []
    texts = [c[0] for c in candidates]
    scores = score_pairs(query, texts)
    reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [(cand[0], cand[1], float(score)) for score, cand in reranked[:top_k]]
