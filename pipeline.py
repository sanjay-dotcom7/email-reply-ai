"""
Run the full pipeline on ONE email: retrieve -> generate -> evaluate.

Usage:
    python -m src.pipeline --id cs_05
    python -m src.pipeline --subject "Late refund" --body "I never got my refund from last week, can you check?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import Evaluator
from .generator import ReplyGenerator
from .llm_client import LLMClient
from .retrieval import load_corpus

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="ID of a test-split email in the dataset to run", default=None)
    parser.add_argument("--subject", help="Subject line of a free-form new email", default=None)
    parser.add_argument("--body", help="Body of a free-form new email", default=None)
    parser.add_argument("--k", type=int, default=3, help="Number of few-shot examples to retrieve")
    args = parser.parse_args()

    llm = LLMClient()
    generator = ReplyGenerator(DATASET_PATH, k=args.k, llm=llm)
    evaluator = Evaluator(generator.retriever, llm=llm)

    if args.id:
        all_items = {e["id"]: e for e in json.loads(DATASET_PATH.read_text())}
        if args.id not in all_items:
            raise SystemExit(f"No email with id={args.id} in dataset")
        item = all_items[args.id]
        subject, body = item["subject"], item["incoming_email"]
        key_points = item["key_points"]
        reference_reply = item["reference_reply"]
    elif args.subject and args.body:
        subject, body = args.subject, args.body
        key_points, reference_reply = [], ""
    else:
        raise SystemExit("Provide either --id <dataset id> or both --subject and --body")

    print(f"MODE: {'MOCK (no ANTHROPIC_API_KEY set)' if llm.mock else f'LIVE ({llm.model})'}\n")
    print(f"SUBJECT: {subject}")
    print(f"INCOMING EMAIL:\n{body}\n")

    result = generator.generate(subject, body)
    print(f"RETRIEVED EXAMPLES: {result['retrieved_example_ids']}\n")
    print(f"GENERATED REPLY:\n{result['reply']}\n")

    if key_points:
        ev = evaluator.evaluate(body, key_points, result["reply"], reference_reply)
        print("--- EVALUATION ---")
        print(f"Overall score:        {ev.overall:.2f}")
        print(f"Key-point coverage:   {ev.coverage:.2f}  ({sum(1 for c in ev.coverage_detail if c.get('addressed'))}/{len(ev.coverage_detail)})")
        for c in ev.coverage_detail:
            mark = "✓" if c.get("addressed") else "✗"
            print(f"    {mark} {c.get('point')} - {c.get('why')}")
        print(f"Rubric avg (0-1):     {ev.rubric_avg:.2f}  raw={ev.rubric}")
        print(f"Reference similarity: {ev.similarity:.2f}")
        print(f"Hallucination flag:   {ev.hallucination_flag} ({ev.hallucination_why})")
        if ev.notes:
            print(f"NOTE: {ev.notes}")
    else:
        print("(No key-points/reference available for free-form input - skipping scored evaluation.)")


if __name__ == "__main__":
    main()
