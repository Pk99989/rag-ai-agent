"""Phase 10 experiments: compares baseline vector-only vs hybrid (no
rerank) vs hybrid+rerank vs semantic-chunking retrieval, using the SAME 20
RAG questions from evaluation/eval_dataset.json and the SAME Recall@K/
Precision@K/MRR metrics as evaluation/run_evaluation.py -- so these numbers
are directly comparable to that report, not a separate ad hoc benchmark.

All four configs are REAL retrieval calls against real indexes (see
retrieval_configs.py for exactly what data each one reads). No metric here
is estimated, interpolated, or carried over from a previous run.

Run (semantic_chunking requires building its isolated index first):
    python experiments/reindex_semantic.py
    python experiments/run_experiments.py
"""
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_configs import CONFIGS  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "eval_dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
TOP_K = 4


def load_rag_items() -> list:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [i for i in dataset["items"] if i["category"] == "RAG"]


def evaluate_config(retrieve_fn, items: list, top_k: int = TOP_K) -> dict:
    recalls, precisions, rrs = [], [], []
    per_item = []
    for item in items:
        chunks = retrieve_fn(item["question"], item["role"], top_k)
        sources = [meta.get("source", "") for _, meta, _ in chunks]
        patterns = item["expected_sources"]
        match_flags = [any(fnmatch.fnmatch(src, pat) for pat in patterns) for src in sources]
        hit = any(match_flags)
        recalls.append(1.0 if hit else 0.0)
        precisions.append(sum(match_flags) / len(match_flags) if match_flags else 0.0)
        rr = 0.0
        for rank, matched in enumerate(match_flags, start=1):
            if matched:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        per_item.append({"id": item["id"], "hit_at_k": hit})
    n = len(items) or 1
    return {
        "recall_at_k": round(sum(recalls) / n, 4),
        "precision_at_k": round(sum(precisions) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "n_items": len(items),
        "per_item": per_item,
    }


def main():
    items = load_rag_items()
    print("=" * 70)
    print(f"PHASE 10 EXPERIMENTS -- {len(items)} RAG questions, top_k={TOP_K}")
    print("Real retrieval calls per config -- see retrieval_configs.py for exactly")
    print("what index/algorithm each config uses.")
    print("=" * 70)

    all_results = {}
    for name, fn in CONFIGS.items():
        print(f"\n--- {name} ---")
        try:
            result = evaluate_config(fn, items)
            all_results[name] = result
            print(f"Recall@{TOP_K}: {result['recall_at_k']}   "
                  f"Precision@{TOP_K}: {result['precision_at_k']}   "
                  f"MRR: {result['mrr']}")
        except FileNotFoundError as exc:
            all_results[name] = {"skipped": str(exc)}
            print(f"SKIPPED: {exc}")

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'config':<22} {'Recall@' + str(TOP_K):<12} {'Precision@' + str(TOP_K):<14} {'MRR':<8}")
    for name, result in all_results.items():
        if "skipped" in result:
            print(f"{name:<22} SKIPPED")
        else:
            print(f"{name:<22} {result['recall_at_k']:<12} {result['precision_at_k']:<14} {result['mrr']:<8}")

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
