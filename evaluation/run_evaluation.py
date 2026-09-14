"""Phase 10 evaluation harness: computes REAL metrics from real execution
against evaluation/eval_dataset.json (100 questions: 20 each of
RAG/SQL/RBAC/SECURITY/GENERAL).

Explicit non-fabrication design notes (read before trusting any number this
prints):
  - RAG retrieval metrics (Recall@K/Precision@K/MRR) come from actually
    calling rag_chain.retrieve() against the live index and checking the
    real returned sources against eval_dataset.json's expected_sources.
  - RAG generation metrics (faithfulness, answer_relevancy, and --
    where eval_dataset.json has a verified 'reference' -- context_precision/
    context_recall) come from RAGAS, which is itself an LLM-as-judge method:
    real Groq API calls score the real generated answer. This is a real
    score, but it's a judge's score, not a mathematically verified ground
    truth -- report it as "RAGAS-judged", not as certainty.
  - SQL accuracy: ground truth is NOT a number this script or the assistant
    typed in advance. For every SQL item we EXECUTE eval_dataset.json's own
    reference_sql against the real data/olist.db right here, at run time,
    and compare that live result to the live result of the pipeline's own
    LLM-generated SQL. If both queries are wrong in the same way, this
    would still report a match -- reference_sql was hand-written against
    the real schema (src/rag_agent/sql/schema.py) but not independently
    re-derived from a second source.
  - RBAC checks call the real is_authorized()/validate_sql() functions --
    fully deterministic, no LLM involved, no judgment call.
  - Security/injection checks call the real guardrails.py functions
    directly -- deterministic.
  - PII handling is reported as TWO separate, honestly-labeled numbers
    (see run_security_eval below) rather than one "PII leakage %" that
    would overstate what's actually being tested. A true end-to-end
    leakage test (PII inside a retrieved document reaching a live LLM's
    answer) is NOT attempted here -- it would require seeding a real
    PII-bearing document into the corpus and depends on a non-deterministic
    LLM choosing to echo it, which would produce a number with a false
    sense of precision. That gap is stated here, not hidden.

Run:
    python evaluation/run_evaluation.py
Requires GROQ_API_KEY for the RAG generation metrics, SQL category, and
GENERAL category (all three call the live LLM). RBAC and SECURITY need no
API key. Install ragas + datasets (see requirements-eval.txt) for the RAG
generation metrics; without them, retrieval-only RAG metrics still run.
"""
import base64
import codecs
import fnmatch
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import GROQ_API_KEY  # noqa: E402
from rag_chain import retrieve, answer_query  # noqa: E402
from rbac import is_authorized  # noqa: E402
from guardrails import detect_pii, redact_pii, detect_prompt_injection, scan_context_for_injection  # noqa: E402
from rag_agent.sql.validator import validate_sql, SQLValidationError  # noqa: E402
from rag_agent.sql.text_to_sql import run_text_to_sql  # noqa: E402
from rag_agent.sql.executor import execute_sql, SQLExecutionError  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

_ZERO_WIDTH_CODEPOINTS = [0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060]


class DummyUser:
    username = "phase10-eval"


def load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _by_category(items: list, category: str) -> list:
    return [i for i in items if i["category"] == category]


# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------

def evaluate_rag_retrieval(items: list, top_k: int = 4) -> dict:
    """Recall@K / Precision@K / MRR -- all from real retrieve() calls, no LLM needed."""
    recalls, precisions, rrs = [], [], []
    per_item = []
    for item in items:
        chunks = retrieve(item["question"], item["role"], top_k=top_k)
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
        per_item.append({"id": item["id"], "hit_at_k": hit, "retrieved_sources": sources})
    n = len(items) or 1
    return {
        "recall_at_k": round(sum(recalls) / n, 4),
        "precision_at_k": round(sum(precisions) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "k": top_k,
        "n_items": len(items),
        "per_item": per_item,
    }


def evaluate_rag_generation(items: list) -> dict:
    """RAGAS-judged faithfulness/answer_relevancy (all items) and
    context_precision/context_recall (only items with a verified 'reference').
    Optional: returns a dict noting ragas unavailable if not installed."""
    if not GROQ_API_KEY:
        return {"skipped": "GROQ_API_KEY not set"}
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        from datasets import Dataset
    except ImportError:
        return {
            "skipped": "ragas/datasets not installed -- run: pip install -r requirements-eval.txt. "
                       "Known Windows issue: current ragas pulls in scikit-network, which needs a C "
                       "compiler to build (see requirements-eval.txt's comment). This is a real gap "
                       "in this report, not a fabricated 'N/A' -- faithfulness/answer_relevancy/"
                       "context_precision/context_recall were not computed for this run."
        }

    rows_all = {"question": [], "answer": [], "contexts": []}
    rows_ref = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in items:
        chunks = retrieve(item["question"], item["role"])
        result = answer_query(item["question"], user=DummyUser(), role=item["role"])
        contexts = [text for text, _, _ in chunks] or [""]
        rows_all["question"].append(item["question"])
        rows_all["answer"].append(result["answer"])
        rows_all["contexts"].append(contexts)
        if item.get("reference"):
            rows_ref["question"].append(item["question"])
            rows_ref["answer"].append(result["answer"])
            rows_ref["contexts"].append(contexts)
            rows_ref["ground_truth"].append(item["reference"])

    judge = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY))
    out = {}
    try:
        ds_all = Dataset.from_dict(rows_all)
        scores_all = ragas_evaluate(ds_all, metrics=[faithfulness, answer_relevancy], llm=judge)
        out["faithfulness_and_relevancy"] = dict(scores_all)
    except Exception as exc:
        out["faithfulness_and_relevancy_error"] = str(exc)

    if rows_ref["question"]:
        try:
            ds_ref = Dataset.from_dict(rows_ref)
            scores_ref = ragas_evaluate(ds_ref, metrics=[context_precision, context_recall], llm=judge)
            out["context_precision_and_recall"] = dict(scores_ref)
            out["context_metrics_n_items"] = len(rows_ref["question"])
        except Exception as exc:
            out["context_precision_and_recall_error"] = str(exc)
    else:
        out["context_precision_and_recall"] = "no items had a verified reference -- skipped, not fabricated"

    return out


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------

def _extract_scalar(sql_result):
    if sql_result.row_count != 1 or len(sql_result.rows[0]) < 1:
        return None
    return sql_result.rows[0][0]


def _values_match(a, b, tol=0.01) -> bool:
    if a is None or b is None:
        return a == b
    try:
        af, bf = float(a), float(b)
        return abs(af - bf) <= tol * max(1.0, abs(bf))
    except (TypeError, ValueError):
        return a == b


def evaluate_sql(items: list) -> dict:
    if not GROQ_API_KEY:
        return {"skipped": "GROQ_API_KEY not set"}
    correct = 0
    details = []
    for item in items:
        try:
            gt_result = execute_sql(item["reference_sql"])
            gt_value = _extract_scalar(gt_result)
        except SQLExecutionError as exc:
            details.append({"id": item["id"], "error": f"reference_sql failed to execute: {exc}"})
            continue

        pipeline = run_text_to_sql(item["question"], user=DummyUser(), role=item["role"])
        if pipeline["blocked"] or not pipeline["sql"]:
            details.append({"id": item["id"], "match": False, "reason": pipeline.get("reason"),
                             "ground_truth": gt_value, "pipeline_sql": None})
            continue
        try:
            pipeline_result = execute_sql(pipeline["sql"])
            pipeline_value = _extract_scalar(pipeline_result)
        except SQLExecutionError as exc:
            details.append({"id": item["id"], "match": False, "error": str(exc)})
            continue

        match = _values_match(gt_value, pipeline_value)
        correct += 1 if match else 0
        details.append({
            "id": item["id"], "match": match,
            "ground_truth": gt_value, "pipeline_value": pipeline_value,
            "pipeline_sql": pipeline["sql"],
        })
    n = len(items) or 1
    return {"sql_accuracy_pct": round(100 * correct / n, 1), "n_items": len(items), "details": details}


# --------------------------------------------------------------------------
# RBAC
# --------------------------------------------------------------------------

def evaluate_rbac(items: list) -> dict:
    correct = 0
    false_allow = []  # expected denied, actually allowed -- the dangerous direction
    false_deny = []   # expected allowed, actually denied -- usability regression
    for item in items:
        if item["check_type"] == "doc":
            actual = is_authorized(item["role"], item["department"])
        else:
            try:
                validate_sql(item["sql"], role=item["role"])
                actual = True
            except SQLValidationError:
                actual = False
        expected = item["expected_allowed"]
        if actual == expected:
            correct += 1
        elif actual and not expected:
            false_allow.append(item["id"])
        elif expected and not actual:
            false_deny.append(item["id"])
    n = len(items) or 1
    return {
        "rbac_security_pct": round(100 * correct / n, 1),
        "n_items": len(items),
        "false_allow_ids": false_allow,  # should be empty -- any entry here is a real security bug
        "false_deny_ids": false_deny,
    }


# --------------------------------------------------------------------------
# SECURITY (injection + PII)
# --------------------------------------------------------------------------

def evaluate_security(items: list) -> dict:
    injection_correct, injection_total = 0, 0
    false_negatives, false_positives = [], []  # missed attack vs blocked-a-benign-input
    pii_detected_correct = 0
    pii_redaction_correct = 0
    pii_total = 0

    for item in items:
        ctype = item["check_type"]
        if ctype == "input_injection":
            actual_blocked = detect_prompt_injection(item["input_text"])
        elif ctype == "input_injection_base64":
            payload = base64.b64encode(item["input_text_raw"].encode()).decode()
            actual_blocked = detect_prompt_injection(item["wrapper"].format(b64=payload))
        elif ctype == "input_injection_rot13":
            payload = codecs.encode(item["input_text_raw"], "rot_13")
            actual_blocked = detect_prompt_injection(item["wrapper"].format(rot13=payload))
        elif ctype == "input_injection_zero_width":
            zwsp = chr(_ZERO_WIDTH_CODEPOINTS[0])
            words = item["input_text_raw"].split(" ")
            obfuscated = (zwsp + " ").join(words)
            actual_blocked = detect_prompt_injection(obfuscated)
        elif ctype == "context_injection":
            _, flagged = scan_context_for_injection([(item["input_text"], {"source": "eval_test_chunk.md"})])
            actual_blocked = len(flagged) > 0
        elif ctype == "pii":
            pii_total += 1
            detected = detect_pii(item["input_text"])
            expected_types = set(item["expected_pii_types"])
            if expected_types.issubset(detected.keys()):
                pii_detected_correct += 1
            # Precise redaction check: after redact_pii, none of the
            # EXPECTED PII types should still be detectable in the output
            # (not just "the text changed somehow", which could pass even
            # if an unrelated character got swapped and the real PII stayed).
            redacted = redact_pii(item["input_text"])
            still_present = expected_types & detect_pii(redacted).keys()
            if not still_present:
                pii_redaction_correct += 1
            continue
        else:
            continue

        injection_total += 1
        expected_blocked = item["expected_blocked"]
        if actual_blocked == expected_blocked:
            injection_correct += 1
        elif actual_blocked and not expected_blocked:
            false_positives.append(item["id"])
        elif expected_blocked and not actual_blocked:
            false_negatives.append(item["id"])

    return {
        "injection_block_accuracy_pct": round(100 * injection_correct / (injection_total or 1), 1),
        "injection_n_items": injection_total,
        "injection_false_negatives_missed_attacks": false_negatives,
        "injection_false_positives_blocked_benign": false_positives,
        "pii_input_detection_rate_pct": round(100 * pii_detected_correct / (pii_total or 1), 1),
        "pii_redaction_function_correct_pct": round(100 * pii_redaction_correct / (pii_total or 1), 1),
        "pii_n_items": pii_total,
        "note": (
            "PII numbers are input-side detection + redaction-function correctness, "
            "not a live end-to-end 'PII leaked into a real LLM answer' measurement -- "
            "see this file's module docstring for why that number isn't attempted here."
        ),
    }


# --------------------------------------------------------------------------
# GENERAL
# --------------------------------------------------------------------------

def evaluate_general(items: list) -> dict:
    if not GROQ_API_KEY:
        return {"skipped": "GROQ_API_KEY not set"}
    correct = 0
    over_refusals = []   # in-scope question incorrectly treated as out-of-scope
    under_refusals = []  # out-of-scope question NOT caught (hallucination risk)
    for item in items:
        result = answer_query(item["question"], user=DummyUser(), role=item["role"])
        actual_out_of_scope = result.get("reason") in ("out_of_scope_or_no_context", "context_flagged_as_injection")
        expected = item["expected_out_of_scope"]
        if actual_out_of_scope == expected:
            correct += 1
        elif actual_out_of_scope and not expected:
            over_refusals.append(item["id"])
        elif expected and not actual_out_of_scope:
            under_refusals.append(item["id"])
    n = len(items) or 1
    return {
        "general_accuracy_pct": round(100 * correct / n, 1),
        "n_items": len(items),
        "over_refusal_ids": over_refusals,
        "under_refusal_ids": under_refusals,
    }


# --------------------------------------------------------------------------

def main():
    dataset = load_dataset()
    items = dataset["items"]

    print("=" * 70)
    print("PHASE 10 EVALUATION -- real execution, see module docstring for")
    print("exactly what each number does and does not certify.")
    print("=" * 70)

    rag_items = _by_category(items, "RAG")
    sql_items = _by_category(items, "SQL")
    rbac_items = _by_category(items, "RBAC")
    sec_items = _by_category(items, "SECURITY")
    gen_items = _by_category(items, "GENERAL")

    print(f"\nDataset: {len(items)} items "
          f"(RAG={len(rag_items)}, SQL={len(sql_items)}, RBAC={len(rbac_items)}, "
          f"SECURITY={len(sec_items)}, GENERAL={len(gen_items)})")

    print("\n--- RAG: retrieval metrics (no LLM required) ---")
    rag_retrieval = evaluate_rag_retrieval(rag_items)
    print(f"Recall@{rag_retrieval['k']}: {rag_retrieval['recall_at_k']}   "
          f"Precision@{rag_retrieval['k']}: {rag_retrieval['precision_at_k']}   "
          f"MRR: {rag_retrieval['mrr']}")

    print("\n--- RAG: generation metrics (RAGAS, requires GROQ_API_KEY + ragas) ---")
    rag_generation = evaluate_rag_generation(rag_items)
    print(json.dumps(rag_generation, indent=2, default=str))

    print("\n--- SQL accuracy (requires GROQ_API_KEY) ---")
    sql_results = evaluate_sql(sql_items)
    if "skipped" in sql_results:
        print(sql_results["skipped"])
    else:
        print(f"SQL accuracy: {sql_results['sql_accuracy_pct']}% "
              f"({sql_results['n_items']} items)")

    print("\n--- RBAC security (deterministic, no LLM) ---")
    rbac_results = evaluate_rbac(rbac_items)
    print(f"RBAC correctness: {rbac_results['rbac_security_pct']}% "
          f"({rbac_results['n_items']} items)")
    if rbac_results["false_allow_ids"]:
        print(f"  !! SECURITY FAILURE -- unauthorized access allowed: {rbac_results['false_allow_ids']}")
    if rbac_results["false_deny_ids"]:
        print(f"  usability regressions (over-denied): {rbac_results['false_deny_ids']}")

    print("\n--- SECURITY: injection + PII (deterministic, no LLM) ---")
    sec_results = evaluate_security(sec_items)
    print(f"Injection block accuracy: {sec_results['injection_block_accuracy_pct']}% "
          f"({sec_results['injection_n_items']} items)")
    if sec_results["injection_false_negatives_missed_attacks"]:
        print(f"  missed attacks: {sec_results['injection_false_negatives_missed_attacks']}")
    if sec_results["injection_false_positives_blocked_benign"]:
        print(f"  false positives (blocked benign input): {sec_results['injection_false_positives_blocked_benign']}")
    print(f"PII input-detection rate: {sec_results['pii_input_detection_rate_pct']}% "
          f"| PII redaction-function correctness: {sec_results['pii_redaction_function_correct_pct']}% "
          f"({sec_results['pii_n_items']} items)")

    print("\n--- GENERAL: scope handling (requires GROQ_API_KEY) ---")
    gen_results = evaluate_general(gen_items)
    if "skipped" in gen_results:
        print(gen_results["skipped"])
    else:
        print(f"Scope-handling accuracy: {gen_results['general_accuracy_pct']}% "
              f"({gen_results['n_items']} items)")
        if gen_results["over_refusal_ids"]:
            print(f"  over-refusals (in-scope treated as out-of-scope): {gen_results['over_refusal_ids']}")
        if gen_results["under_refusal_ids"]:
            print(f"  under-refusals (out-of-scope NOT caught): {gen_results['under_refusal_ids']}")

    all_results = {
        "rag_retrieval": rag_retrieval,
        "rag_generation": rag_generation,
        "sql": sql_results,
        "rbac": rbac_results,
        "security": sec_results,
        "general": gen_results,
    }
    RESULTS_PATH.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
