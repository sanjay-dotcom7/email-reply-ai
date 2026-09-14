"""
Runs the generator + evaluator over the ENTIRE held-out test split and
produces:
  1. A per-response report (score + breakdown for every test email).
  2. An overall system score (mean of per-response overall scores).
  3. A calibration / validation harness: for each test email, scores four
     variants of "a reply" (generated / ground-truth / mismatched /
     empty) and checks whether the metric ranks them the way a sensible
     metric should. This is how we validate the metric reflects real
     quality without needing a separate human-labeled dataset - see
     the big docstring in evaluator.py and the README for the reasoning.

Usage:
    python -m src.run_eval_suite
    python -m src.run_eval_suite --k 3 --out results/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .evaluator import Evaluator
from .generator import ReplyGenerator
from .llm_client import LLMClient
from .retrieval import load_corpus

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.json"


def run(k: int, out_path: Path):
    llm = LLMClient()
    generator = ReplyGenerator(DATASET_PATH, k=k, llm=llm)
    evaluator = Evaluator(generator.retriever, llm=llm)

    test_items = load_corpus(DATASET_PATH, split="test")
    all_items = json.loads(DATASET_PATH.read_text())
    key_points_by_id = {e["id"]: e["key_points"] for e in all_items}

    random.seed(7)
    per_response = []
    calibration_rows = []

    for i, item in enumerate(test_items):
        key_points = key_points_by_id[item.id]

        gen = generator.generate(item.subject, item.incoming_email)
        generated_reply = gen["reply"]

        ev_generated = evaluator.evaluate(item.incoming_email, key_points, generated_reply, item.reference_reply)
        ev_reference = evaluator.evaluate(item.incoming_email, key_points, item.reference_reply, item.reference_reply)

        # mismatched: reply that was actually sent for a DIFFERENT test email
        other = test_items[(i + 1) % len(test_items)]
        ev_mismatched = evaluator.evaluate(item.incoming_email, key_points, other.reference_reply, item.reference_reply)

        ev_empty = evaluator.evaluate(item.incoming_email, key_points, "", item.reference_reply)

        ranking_ok = (
            ev_reference.overall >= ev_generated.overall
            and ev_generated.overall > ev_mismatched.overall
            and ev_mismatched.overall > ev_empty.overall
        )

        per_response.append({
            "id": item.id,
            "category": item.category,
            "subject": item.subject,
            "incoming_email": item.incoming_email,
            "retrieved_example_ids": gen["retrieved_example_ids"],
            "generated_reply": generated_reply,
            "reference_reply": item.reference_reply,
            "overall_score": round(ev_generated.overall, 3),
            "key_point_coverage": round(ev_generated.coverage, 3),
            "coverage_detail": ev_generated.coverage_detail,
            "rubric": ev_generated.rubric,
            "rubric_avg": round(ev_generated.rubric_avg, 3),
            "reference_similarity": round(ev_generated.similarity, 3),
            "hallucination_flag": ev_generated.hallucination_flag,
            "hallucination_why": ev_generated.hallucination_why,
            "judge_fallback_used": not ev_generated.judge_raw_ok,
        })

        calibration_rows.append({
            "id": item.id,
            "generated_score": round(ev_generated.overall, 3),
            "reference_score": round(ev_reference.overall, 3),
            "mismatched_score": round(ev_mismatched.overall, 3),
            "empty_score": round(ev_empty.overall, 3),
            "ranking_as_expected (ref >= gen > mismatched > empty)": ranking_ok,
        })

    overall_system_score = sum(r["overall_score"] for r in per_response) / len(per_response)
    hallucination_rate = sum(1 for r in per_response if r["hallucination_flag"]) / len(per_response)
    calibration_pass_rate = sum(1 for c in calibration_rows if c["ranking_as_expected (ref >= gen > mismatched > empty)"]) / len(calibration_rows)

    report = {
        "mode": "MOCK (no ANTHROPIC_API_KEY set - see README)" if llm.mock else f"LIVE ({llm.model})",
        "num_test_emails": len(test_items),
        "overall_system_score": round(overall_system_score, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "calibration_pass_rate": round(calibration_pass_rate, 3),
        "per_response": per_response,
        "calibration_detail": calibration_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    print(f"Mode: {report['mode']}")
    print(f"Test emails evaluated: {report['num_test_emails']}")
    print(f"Overall system score: {report['overall_system_score']}")
    print(f"Hallucination rate:   {report['hallucination_rate']}")
    print(f"Calibration pass rate (metric ranks ref>=gen>mismatched>empty): {report['calibration_pass_rate']}")
    print()
    for r in per_response:
        print(f"  [{r['id']:>8}] overall={r['overall_score']:.2f}  coverage={r['key_point_coverage']:.2f}  "
              f"rubric_avg={r['rubric_avg']:.2f}  sim={r['reference_similarity']:.2f}  "
              f"hallucination={r['hallucination_flag']}")
    print(f"\nFull report written to {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "eval_report.json")
    args = parser.parse_args()
    run(args.k, args.out)
