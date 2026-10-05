from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, EMBEDDING_MODEL, LLM_API_KEY, LLM_BASE_URL,
                    RAGAS_MODEL)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_score(value) -> float:
    """Convert a RAGAS value to a finite score suitable for JSON reports."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not LLM_API_KEY:
        print("  ⚠️  RAGAS skipped: GROQ_API_KEY is not configured.")
        per_question = [
            EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Groq accepts only n=1, while RAGAS defaults to three generated
        # questions for answer_relevancy. One sample is sufficient here and
        # avoids Groq's "'n' must be at most 1" response.
        answer_relevancy.strictness = 1

        evaluator_llm = ChatOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=RAGAS_MODEL,
            temperature=0,
            max_tokens=512,
            # A single RAGAS metric may need multiple model calls. Give the
            # Groq client enough time to honor Retry-After during TPM bursts.
            timeout=180,
            max_retries=20,
        )
        evaluator_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            # Groq's on-demand tier is token-per-minute limited. Serial jobs
            # plus retries let the SDK honor Retry-After instead of dropping
            # concurrent RAGAS jobs and producing NaN scores.
            run_config=RunConfig(
                timeout=600,
                max_retries=20,
                max_wait=120,
                max_workers=1,
            ),
        )
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_score(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_score(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_score(row.get("context_precision", 0.0)),
                context_recall=_safe_score(row.get("context_recall", 0.0)),
            ))

        count = max(len(per_question), 1)
        avg_f = sum(item.faithfulness for item in per_question) / count
        avg_a = sum(item.answer_relevancy for item in per_question) / count
        avg_cp = sum(item.context_precision for item in per_question) / count
        avg_cr = sum(item.context_recall for item in per_question) / count

        return {
            "faithfulness": avg_f,
            "answer_relevancy": avg_a,
            "context_precision": avg_cp,
            "context_recall": avg_cr,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            per_question.append(EvalResult(
                question=q, answer=a, contexts=c, ground_truth=gt,
                faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0, context_recall=0.0
            ))
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    if not eval_results:
        return []

    scored_results = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy,
        }
        avg_score = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree.get(worst_metric, ("Unknown issue", "Review pipeline"))

        scored_results.append({
            "avg_score": avg_score,
            "item": {
                "question": r.question,
                "answer": r.answer,
                "ground_truth": r.ground_truth,
                "worst_metric": worst_metric,
                "score": metrics[worst_metric],
                "avg_score": avg_score,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        })

    scored_results.sort(key=lambda x: x["avg_score"])
    return [x["item"] for x in scored_results[:bottom_n]]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
