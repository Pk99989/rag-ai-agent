"""Router accuracy check (Phase 11): classify() is deterministic and free
(no LLM), so this reuses evaluation/eval_dataset.json's 20 RAG + 20 SQL
questions -- each already labeled with its correct category -- to measure
real routing accuracy. This has never been checked before: Phase 4 built
and smoke-tested classify() in isolation, and neither eval_run.py nor
evaluation/run_evaluation.py ever actually calls it (both call
answer_query()/run_text_to_sql() directly). Now that Phase 11 wires
classify() into the live app via dispatch.py, a misrouted question isn't
just a wrong label -- it's a RAG question that gets sent to Text-to-SQL
against an unrelated schema, or vice versa.

Run:
    python scripts/smoke_test_router_accuracy.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_agent.router.classifier import classify, RAG, SQL, HYBRID  # noqa: E402

DATASET_PATH = REPO_ROOT / "evaluation" / "eval_dataset.json"
_EXPECTED_ROUTE = {"RAG": RAG, "SQL": SQL}


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    items = [i for i in dataset["items"] if i["category"] in ("RAG", "SQL")]

    strict_correct = 0
    lenient_correct = 0
    hybrid_count = 0
    hard_misrouted = []  # RAG labeled as SQL, or SQL labeled as RAG -- the real failure mode
    for item in items:
        result = classify(item["question"])
        expected = _EXPECTED_ROUTE[item["category"]]
        if result.route == expected:
            strict_correct += 1
            lenient_correct += 1
        elif result.route == HYBRID:
            # Not a hard miss: dispatch.py tries RAG first for HYBRID and
            # falls back to SQL only if RAG finds nothing, so a HYBRID
            # result on either a RAG- or SQL-labeled question usually still
            # resolves to a correct answer -- see classifier.py's module
            # docstring on the RAG-aggregate-doc vs live-SQL overlap.
            lenient_correct += 1
            hybrid_count += 1
        else:
            hard_misrouted.append({
                "id": item["id"], "question": item["question"],
                "expected": expected, "got": result.route,
                "confidence": result.confidence,
                "matched_sql_signals": result.matched_sql_signals,
                "matched_rag_signals": result.matched_rag_signals,
            })

    n = len(items)
    print(f"Strict accuracy (exact route match): {strict_correct}/{n} "
          f"({round(100 * strict_correct / n, 1)}%)")
    print(f"Lenient accuracy (HYBRID counts as OK, given dispatch.py's RAG-first "
          f"fallback): {lenient_correct}/{n} ({round(100 * lenient_correct / n, 1)}%)")
    print(f"HYBRID results: {hybrid_count}")
    if hard_misrouted:
        print(f"\n{len(hard_misrouted)} HARD misroutes (RAG<->SQL, the real failure mode):")
        for m in hard_misrouted:
            print(f"  [{m['id']}] expected={m['expected']} got={m['got']} "
                  f"(conf={m['confidence']}, sql_signals={m['matched_sql_signals']}, "
                  f"rag_signals={m['matched_rag_signals']})")
            print(f"      Q: {m['question']}")
    else:
        print("\nNo hard RAG<->SQL misroutes.")


if __name__ == "__main__":
    main()
