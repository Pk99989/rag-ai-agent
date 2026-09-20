"""Automated regression evaluation: RBAC correctness + generation smoke test.

Run this after every change (see ci_workflow.yml) to catch regressions before deploy.
For deeper LLM-judged quality metrics (faithfulness, answer relevancy), install
requirements-eval.txt and use `run_ragas_eval()` below.
"""
import json
import sys
from pathlib import Path

from rag_chain import answer_query, retrieve
from rbac import is_authorized
from config import GROQ_API_KEY

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"


class DummyUser:
    username = "eval-runner"


def run_rbac_regression() -> bool:
    """Fast, deterministic check: does retrieval ever leak cross-department content?"""
    dataset = json.loads(DATASET_PATH.read_text())
    failures = []
    for item in dataset:
        role = item["role"]
        chunks = retrieve(item["question"], role)
        for _, meta in chunks:
            if not is_authorized(role, meta["department"]):
                failures.append((item["question"], role, meta))
    if failures:
        print(f"RBAC REGRESSION: {len(failures)} leak(s) detected:")
        for q, role, meta in failures:
            print(f"  role={role} question={q!r} leaked department={meta['department']}")
        return False
    print(f"RBAC regression check passed ({len(dataset)} queries, no cross-department leaks).")
    return True


def run_generation_smoke_test() -> bool:
    """Runs the full pipeline end-to-end for each dataset item (requires GROQ_API_KEY)."""
    if not GROQ_API_KEY:
        print("Skipping generation smoke test: GROQ_API_KEY not set.")
        return True
    dataset = json.loads(DATASET_PATH.read_text())
    ok = True
    for item in dataset:
        result = answer_query(item["question"], user=DummyUser(), role=item["role"])
        print(f"[{item['role']}] Q: {item['question']}\n -> {result['answer'][:200]}\n")
        if not result["answer"]:
            ok = False
    return ok


def run_ragas_eval():
    """Optional deeper quality eval using RAGAS, judged by the same Groq LLM (no OpenAI key needed).
    Requires: pip install -r requirements-eval.txt
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        from datasets import Dataset
    except ImportError:
        print("ragas/datasets not installed. Run: pip install -r requirements-eval.txt")
        return None
    if not GROQ_API_KEY:
        print("Skipping ragas eval: GROQ_API_KEY not set.")
        return None

    dataset = json.loads(DATASET_PATH.read_text())
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in dataset:
        chunks = retrieve(item["question"], item["role"])
        if not chunks:
            continue
        result = answer_query(item["question"], user=DummyUser(), role=item["role"])
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([c for c, _ in chunks])
        rows["ground_truth"].append(item.get("ground_truth", ""))

    ds = Dataset.from_dict(rows)
    judge = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY))
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy], llm=judge)
    print(scores)
    return scores


if __name__ == "__main__":
    rbac_ok = run_rbac_regression()
    gen_ok = run_generation_smoke_test()
    if not (rbac_ok and gen_ok):
        sys.exit(1)
    print("All evaluation checks passed.")
