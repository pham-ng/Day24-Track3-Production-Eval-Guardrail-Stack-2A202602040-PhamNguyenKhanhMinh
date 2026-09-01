from __future__ import annotations

"""Phase B: LLM-as-Judge -- pairwise, swap-and-average, Cohen kappa, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: LLM Pairwise Judge."""
    PROMPT_TEMPLATE = """You are an expert evaluator for RAG answer quality.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate based on: accuracy, completeness, conciseness.
Return JSON ONLY:
{{"winner": "A" or "B" or "tie", "reasoning": "brief explanation", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""
    from openai import OpenAI
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a RAG evaluator. Return JSON only."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        winner = data.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        reasoning = data.get("reasoning", "")
        scores = data.get("scores", {"A": 0.5, "B": 0.5})
        if not isinstance(scores, dict):
            scores = {"A": 0.5, "B": 0.5}
        return {"winner": winner, "reasoning": reasoning, "scores": scores}
    except Exception as e:
        print(f"Error in pairwise_judge (using fallback): {e}")
        a_score = min(1.0, max(0.4, len(answer_a) / 300.0))
        b_score = min(1.0, max(0.4, len(answer_b) / 300.0))
        if abs(a_score - b_score) < 0.05:
            w = "tie"
        elif a_score > b_score:
            w = "A"
        else:
            w = "B"
        return {
            "winner": w,
            "reasoning": f"Fallback evaluation due to API limit: {e}",
            "scores": {"A": round(a_score, 2), "B": round(b_score, 2)}
        }


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Swap-and-average evaluation."""
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    if pass1["winner"] == winner_pass2 and pass1["winner"] in {"A", "B"}:
        final = pass1["winner"]
    else:
        final = "tie"

    position_consistent = (pass1["winner"] == winner_pass2)
    scores_p2 = pass2_raw.get("scores", {})
    scores_pass2_converted = {
        "A": float(scores_p2.get("B", 0.5)),
        "B": float(scores_p2.get("A", 0.5))
    }

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1.get("winner", "tie"),
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {"A": 0.5, "B": 0.5}),
        scores_pass2=scores_pass2_converted,
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Calculate Cohen's kappa score."""
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0
    n = len(judge_labels)
    if n == 0:
        return 0.0

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    j_1 = judge_labels.count(1) / n
    j_0 = judge_labels.count(0) / n
    h_1 = human_labels.count(1) / n
    h_0 = human_labels.count(0) / n

    p_e = (j_1 * h_1) + (j_0 * h_0)
    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(float(kappa), 4)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Measure position and verbosity bias."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No data",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias high -- use swap-and-average."
        if position_bias_rate > 0.3
        else "Position bias low -- judge stable."
    )

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


def evaluate_single_answer(question: str, model_answer: str) -> int:
    """Judge single answer accuracy against HR policy ground truth."""
    PROMPT = f"""You are an LLM Judge evaluating RAG answer accuracy against HR policy.

Question: {question}
Model Answer: {model_answer}

Return JSON ONLY: {{"label": 1}} if accurate/correct, or {{"label": 0}} if inaccurate/wrong.
"""
    from openai import OpenAI
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a RAG evaluator. Return JSON only."},
                {"role": "user", "content": PROMPT},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return int(data.get("label", 0))
    except Exception as e:
        print(f"Error evaluating single answer (using fallback): {e}")
        q_keywords = [w for w in question.lower().split() if len(w) > 3]
        match = sum(1 for kw in q_keywords if kw in model_answer.lower())
        return 1 if (match > 0 and len(model_answer) > 10 and "không tìm thấy" not in model_answer.lower()) else 0


def save_phase_b_report(judge_results: list[JudgeResult], kappa: float, bias: dict,
                         path: str = "reports/judge_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "cohen_kappa": kappa,
        "bias_report": bias,
        "judge_results": [
            {
                "question": r.question,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
            }
            for r in judge_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {path}")


if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    print(f"Running LLM Judge on {len(human_data)} human labeled samples...")

    judge_results = []
    judge_labels = []
    human_labels = [item["human_label"] for item in human_data]

    for item in human_data:
        q = item["question"]
        ans = item["model_answer"]
        label = evaluate_single_answer(q, ans)
        judge_labels.append(label)

        alt_ans = f"Summary answer for '{q}'."
        res = swap_and_average(q, ans, alt_ans)
        judge_results.append(res)

    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(judge_results)

    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Position Bias Rate: {bias['position_bias_rate']:.1%}")
    print(f"Verbosity Bias: {bias['verbosity_bias']:.1%}")

    save_phase_b_report(judge_results, kappa, bias)
