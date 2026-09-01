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

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        llm = ChatOpenAI(model="gemini-2.5-flash")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            f_val = float(row["faithfulness"]) if "faithfulness" in row and not pd.isna(row["faithfulness"]) else 0.0
            ar_val = float(row["answer_relevancy"]) if "answer_relevancy" in row and not pd.isna(row["answer_relevancy"]) else 0.0
            cp_val = float(row["context_precision"]) if "context_precision" in row and not pd.isna(row["context_precision"]) else 0.0
            cr_val = float(row["context_recall"]) if "context_recall" in row and not pd.isna(row["context_recall"]) else 0.0

            q_val = str(row.get("question", row.get("user_input", "")))
            a_val = str(row.get("answer", row.get("response", "")))
            c_val = row.get("contexts", row.get("retrieved_contexts", []))
            c_val = list(c_val) if isinstance(c_val, (list, tuple)) else [str(c_val)]
            g_val = str(row.get("ground_truth", row.get("reference", "")))

            per_question.append(EvalResult(
                question=q_val,
                answer=a_val,
                contexts=c_val,
                ground_truth=g_val,
                faithfulness=f_val,
                answer_relevancy=ar_val,
                context_precision=cp_val,
                context_recall=cr_val,
            ))

        f_mean = float(df["faithfulness"].mean()) if "faithfulness" in df and not df["faithfulness"].isna().all() else 0.0
        ar_mean = float(df["answer_relevancy"].mean()) if "answer_relevancy" in df and not df["answer_relevancy"].isna().all() else 0.0
        cp_mean = float(df["context_precision"].mean()) if "context_precision" in df and not df["context_precision"].isna().all() else 0.0
        cr_mean = float(df["context_recall"].mean()) if "context_recall" in df and not df["context_recall"].isna().all() else 0.0

        return {
            "faithfulness": f_mean,
            "answer_relevancy": ar_mean,
            "context_precision": cp_mean,
            "context_recall": cr_mean,
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
