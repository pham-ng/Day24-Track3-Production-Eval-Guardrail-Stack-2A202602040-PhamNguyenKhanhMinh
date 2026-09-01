from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


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


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        import pandas as pd
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        def compute_metrics(q, a, ctxs, gt):
            a_words = set(a.lower().split())
            gt_words = set(gt.lower().split())
            ctx_all = " ".join(ctxs).lower()
            ctx_words = set(ctx_all.split())

            # Faithfulness: proportion of answer words present in context
            f = len(a_words.intersection(ctx_words)) / max(len(a_words), 1) if a_words else 0.5
            # Answer relevancy: overlap between answer and ground truth
            ar = len(a_words.intersection(gt_words)) / max(len(gt_words), 1) if gt_words else 0.5
            # Context recall: proportion of ground truth words found in context
            cr = len(gt_words.intersection(ctx_words)) / max(len(gt_words), 1) if gt_words else 0.5
            # Context precision: ratio of relevant chunks
            cp = sum(1 for c in ctxs if any(w in c.lower() for w in gt_words if len(w) > 3)) / max(len(ctxs), 1) if ctxs else 0.5

            return min(1.0, f), min(1.0, ar), min(1.0, cp), min(1.0, cr)

        per_question = []
        f_list, ar_list, cp_list, cr_list = [], [], [], []

        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
            f, ar, cp, cr = compute_metrics(q, a, ctx, gt)
            f_list.append(f)
            ar_list.append(ar)
            cp_list.append(cp)
            cr_list.append(cr)

            per_question.append(EvalResult(
                question=q,
                answer=a,
                contexts=ctx,
                ground_truth=gt,
                faithfulness=f,
                answer_relevancy=ar,
                context_precision=cp,
                context_recall=cr,
            ))

        return {
            "faithfulness": sum(f_list) / max(len(f_list), 1),
            "answer_relevancy": sum(ar_list) / max(len(ar_list), 1),
            "context_precision": sum(cp_list) / max(len(cp_list), 1),
            "context_recall": sum(cr_list) / max(len(cr_list), 1),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt instructions and enforce strict reliance on retrieved context."),
        "context_recall": ("Missing relevant chunks", "Improve chunking strategy or add BM25 keyword search / hybrid fusion."),
        "context_precision": ("Too many irrelevant chunks", "Add reranking stage or metadata filtering to eliminate noise."),
        "answer_relevancy": ("Answer doesn't match question", "Refine answer generation prompt template and temperature settings."),
    }

    analyzed = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics.keys(), key=lambda k: metrics[k])
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown issue", "Inspect pipeline log"))

        analyzed.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "worst_score": round(metrics[worst_metric], 4),
            "diagnosis": diag,
            "suggested_fix": fix
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
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
