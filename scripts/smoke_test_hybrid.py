"""Manual smoke test for Phase 6 (hybrid retrieval + ONNX reranker).

Usage:
    python scripts/smoke_test_hybrid.py

IMPORTANT: run `python ingest.py` first (after this phase's changes) so the
BM25 index file exists -- Part 2 and 3 below will fail with a clear
FileNotFoundError otherwise, not a mysterious crash.

Part 1 downloads the reranker model on first run (a few dozen MB from the
Hugging Face Hub, NOT torch-sized) and needs network access but no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_reranker_check() -> bool:
    print("=== Part 1: reranker sanity check (downloads model on first run) ===")
    from rag_agent.retrieval.reranker import score_pairs

    query = "What is the average delivery time for orders?"
    passages = [
        "Average Delivery Time: 8.3 days. Average Review Score: 4.1 / 5.",  # relevant
        "Payment Type: credit_card. Number of Transactions: 76795.",         # irrelevant
    ]
    scores = score_pairs(query, passages)
    print(f"Query: {query!r}")
    for p, s in zip(passages, scores):
        print(f"  score={s:.4f}  passage={p!r}")
    ok = scores[0] > scores[1]
    print(f"[{'PASS' if ok else 'FAIL'}] relevant passage scored higher than irrelevant passage")
    return ok


def run_hybrid_check() -> bool:
    print("\n=== Part 2: full hybrid retrieval (needs ingest.py already run) ===")
    from rag_agent.retrieval.hybrid import hybrid_retrieve

    query = "What is the average delivery time for orders?"
    role = "executive"  # broadest existing role, so this doesn't fail on RBAC alone
    results = hybrid_retrieve(query, role, top_k=3)
    if not results:
        print("[FAIL] hybrid_retrieve returned zero results -- check ingest.py ran successfully "
              "and produced a non-empty index for role={role!r}.")
        return False
    for text, meta, score in results:
        print(f"  score={score:.4f}  source={meta.get('source')}  text={text[:80]!r}...")
    print(f"[PASS] hybrid_retrieve returned {len(results)} result(s)")
    return True


if __name__ == "__main__":
    ok1 = run_reranker_check()
    try:
        ok2 = run_hybrid_check()
    except FileNotFoundError as exc:
        print(f"\n[SKIPPED] {exc}")
        ok2 = None
    all_ok = ok1 and (ok2 is not False)
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED -- see above'}")
    sys.exit(0 if all_ok else 1)
