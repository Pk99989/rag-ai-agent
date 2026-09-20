"""Manual smoke test for Phase 4 (query router). No API key needed -- this
is pure Python keyword matching.

Usage:
    python scripts/smoke_test_router.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.router.classifier import classify, RAG, SQL, HYBRID, OUT_OF_SCOPE

CASES = [
    ("What is the seller policy?", RAG),
    ("Which category has the highest revenue?", SQL),
    ("Which sellers have high revenue and what is the seller policy?", HYBRID),
    ("Tell me a joke.", OUT_OF_SCOPE),
    ("What is the average order value?", SQL),
    ("What is our return policy for damaged items?", RAG),
    ("How many customers are in each state?", SQL),
    ("What's the capital of France?", OUT_OF_SCOPE),
    ("asdkfj random gibberish question", RAG),  # no signal -> low-confidence RAG default
]

if __name__ == "__main__":
    all_ok = True
    for question, expected in CASES:
        result = classify(question)
        ok = result.route == expected
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {question!r}")
        print(f"       -> route={result.route} confidence={result.confidence} "
              f"(expected {expected})")
        if result.matched_sql_signals:
            print(f"       sql signals: {result.matched_sql_signals}")
        if result.matched_rag_signals:
            print(f"       rag signals: {result.matched_rag_signals}")
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED -- see above'}")
    sys.exit(0 if all_ok else 1)
