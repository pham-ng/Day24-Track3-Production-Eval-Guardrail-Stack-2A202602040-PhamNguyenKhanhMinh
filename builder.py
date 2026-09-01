import os

phase_a_code = '''from __future__ import annotations

"""Phase A: RAGAS Production Evaluation -- 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set with 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers from setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json not found at {path}\\n"
            "Run setup_answers.py first"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": [
            {"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Group 50 questions into 3 distributions."""
    groups: dict[str, list[dict]] = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item.get("distribution", "factual")
        if dist in groups:
            groups[dist].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Run RAGAS 4 metrics on all 50 questions."""
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("src/m4_eval.py not found")
        return []

    questions     = [a["question"]    for a in answers]
    ans_texts     = [a["answer"]      for a in answers]
    contexts      = [a["contexts"]    for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = raw.get("per_question", [])

    results = []
    for a, pq in zip(answers, per_q):
        results.append(RagasResult(
            question_id=a["id"],
            distribution=a.get("distribution", "factual"),
            question=a["question"],
            answer=a["answer"],
            contexts=a["contexts"],
            ground_truth=a["ground_truth"],
            faithfulness=pq.faithfulness,
            answer_relevancy=pq.answer_relevancy,
            context_precision=pq.context_precision,
            context_recall=pq.context_recall,
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Get 10 questions with lowest avg_score."""
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    bottom = sorted_asc[:10]
    output = []
    for i, r in enumerate(bottom):
        diag, fix = DIAGNOSTIC_TREE.get(r.worst_metric, ("Unknown issue", "Inspect pipeline log"))
        output.append({
            "rank": i + 1,
            "question_id": r.question_id,
            "distribution": r.distribution,
            "question": r.question,
            "avg_score": round(r.avg_score, 4),
            "worst_metric": r.worst_metric,
            "diagnosis": diag,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Failure cluster analysis by (worst_metric x distribution)."""
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        if r.worst_metric in matrix and r.distribution in matrix[r.worst_metric]:
            matrix[r.worst_metric][r.distribution] += 1

    distributions = ["factual", "multi_hop", "adversarial"]
    dominant_dist   = max(distributions, key=lambda d: sum(matrix[m][d] for m in matrix))
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    insight = (f"Distribution '{dominant_dist}' has most failures. "
               f"Metric '{dominant_metric}' is primary weakness. "
               f"Fix: {DIAGNOSTIC_TREE[dominant_metric][1]}")

    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("No results - implement run_ragas_50q() first.")
'''

phase_b_code = '''from __future__ import annotations

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
        print(f"Error in pairwise_judge: {e}")
        return {"winner": "tie", "reasoning": str(e), "scores": {"A": 0.5, "B": 0.5}}


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
        print(f"Error evaluating single answer: {e}")
        return 0


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
'''

phase_c_code = '''from __future__ import annotations

"""Phase C: Production Guardrails -- Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

_ANALYZER = None
_ANONYMIZER = None
_RAILS = None


def setup_presidio():
    """Initialize Presidio engine with custom Vietnamese PII recognizers (singleton)."""
    global _ANALYZER, _ANONYMIZER
    if _ANALYZER is not None and _ANONYMIZER is not None:
        return _ANALYZER, _ANONYMIZER

    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\\b\\d{12}\\b", 0.9),
            Pattern("CMND 9 digits",  r"\\b\\d{9}\\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\\b0[3-9]\\d{8}\\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    _ANALYZER   = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    _ANONYMIZER = AnonymizerEngine()
    return _ANALYZER, _ANONYMIZER


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: PII scan via Presidio."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {
            "type": r.entity_type,
            "text": text[r.start:r.end],
            "score": round(r.score, 3),
            "start": r.start,
            "end": r.end,
        }
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


def setup_nemo_rails():
    """Initialize NeMo Guardrails (singleton)."""
    global _RAILS
    if _RAILS is not None:
        return _RAILS

    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    _RAILS = LLMRails(config)
    return _RAILS


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: NeMo Input Rail."""
    if rails is None:
        rails = setup_nemo_rails()

    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
        if isinstance(response, dict):
            resp_text = response.get("content", str(response))
        else:
            resp_text = str(response)

        refuse_keywords = [
            "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
            "tôi không thể", "từ chối", "không hỗ trợ", "vi phạm"
        ]
        blocked = any(kw in resp_text.lower() for kw in refuse_keywords)
        return {
            "allowed": not blocked,
            "blocked_reason": "nemo_input_rail" if blocked else None,
            "response": resp_text,
        }
    except Exception as e:
        print(f"NeMo input rail fallback: {e}")
        return {"allowed": True, "blocked_reason": None, "response": text}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: NeMo Output Rail."""
    if rails is None:
        rails = setup_nemo_rails()

    try:
        response = await rails.generate_async(messages=[
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ])
        if isinstance(response, dict):
            resp_text = response.get("content", str(response))
        else:
            resp_text = str(response)

        refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot"]
        flagged = any(kw in resp_text.lower() for kw in refuse_keywords)
        return {
            "safe": not flagged,
            "flagged_reason": "nemo_output_rail" if flagged else None,
            "final_answer": resp_text if flagged else answer,
        }
    except Exception as e:
        print(f"NeMo output rail fallback: {e}")
        return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Run adversarial suite across full guard stack."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })
        return results

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(_run_all())

    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Measure P50/P95/P99 latency."""
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        inputs = (test_inputs * (n_runs // len(test_inputs) + 1))[:n_runs]
        for text in inputs:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            # NeMo input rail
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    try:
        asyncio.run(_measure())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_measure())

    def percentiles(times):
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        return {
            "p50": round(s[int(n * 0.50)], 2),
            "p95": round(s[int(n * 0.95)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms":     percentiles(nemo_times),
        "total_ms":    total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def save_phase_c_report(adv_results: list[dict], latency: dict,
                         path: str = "reports/guard_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "adversarial_suite": {
            "total": len(adv_results),
            "passed": sum(1 for r in adv_results if r["passed"]),
            "pass_rate": round(sum(1 for r in adv_results if r["passed"]) / len(adv_results), 3) if adv_results else 0.0,
            "details": adv_results,
        },
        "latency_measurement": latency,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")


if __name__ == "__main__":
    analyzer, anonymizer = setup_presidio()
    rails = setup_nemo_rails()

    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    res = pii_scan(test_pii, analyzer, anonymizer)
    print(f"PII detected: {res['has_pii']}, anonymized: {res['anonymized']}")

    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adv_set = json.load(f)
    adv_results = run_adversarial_suite(adv_set, rails, analyzer, anonymizer)

    sample_inputs = [item["input"] for item in adv_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10, rails=rails, analyzer=analyzer, anonymizer=anonymizer)

    save_phase_c_report(adv_results, latency)
'''

with open('src/phase_a_ragas.py', 'w', encoding='utf-8') as f:
    f.write(phase_a_code)
with open('src/phase_b_judge.py', 'w', encoding='utf-8') as f:
    f.write(phase_b_code)
with open('src/phase_c_guard.py', 'w', encoding='utf-8') as f:
    f.write(phase_c_code)

print('Updated builder.py and written modules successfully!')
