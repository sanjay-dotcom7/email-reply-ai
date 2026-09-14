"""
Evaluates a generated reply. This is the core of the project.

WHAT "ACCURATE" MEANS HERE
Exact match against the historically-sent reply is the wrong bar: two
people can write completely different sentences and both resolve the
email perfectly well, and matching the literal wording someone happened
to type is not actually the property we care about. So instead of one
"accuracy" number, this evaluator scores four independent properties of
"is this a good reply", each measurable on its own terms, then combines
them:

  1. KEY-POINT COVERAGE (0-1, weight 0.40) - grounded, checklist-based.
     Each test email was hand-labeled with the concrete facts/actions a
     good reply MUST address (e.g. "apologize", "give the refund amount",
     "commit to a timeframe"). An LLM judge checks each one off against
     the generated reply. This is the most "objective" signal in the
     system because the checklist is fixed ahead of time by a human,
     not invented by the same judge scoring the answer.

  2. RUBRIC SCORE (0-1, weight 0.35) - LLM-as-judge on four qualitative
     dimensions (relevance, completeness, tone/professionalism,
     actionability), 1-5 each, averaged and normalized. Captures things a
     fixed checklist can't (is this well-written? too curt? off-tone?).

  3. REFERENCE SIMILARITY (0-1, weight 0.15) - TF-IDF cosine similarity
     to the reply a human actually sent. Deliberately weighted LOW: it's
     a weak, noisy signal (good replies can look very different from
     each other) but it's cheap, deterministic, and catches the
     pathological case of a reply that's topically unrelated altogether.

  4. HALLUCINATION PENALTY (subtractive, up to -0.20) - LLM judge flags
     whether the reply invents a fact (a price, date, policy, promise)
     that isn't supported by the incoming email or the retrieved
     examples. This exists because coverage + rubric alone can both be
     high on a reply that sounds great but is making things up - a
     failure mode that's actively dangerous for a tool that ships email
     on someone's behalf.

overall = 0.40*coverage + 0.35*rubric_avg + 0.15*similarity - hallucination_penalty

WHY THIS COMBINATION, AND HOW IT'S VALIDATED (not just asserted)
A metric is only trustworthy if we've checked it behaves sensibly on
cases where we already know the right answer. `run_eval_suite.py` runs a
calibration harness for exactly this: for every test email, it scores
FOUR variants of "a reply" -
  (a) the reply our generator actually produced,
  (b) the ground-truth reply a human actually sent (should score highest),
  (c) a reply copied from a DIFFERENT, unrelated email (should score low -
      topically wrong, addresses the wrong problem),
  (d) an empty reply (should score at the floor).
If the metric is doing its job, scores should rank (b) >= (a) > (c) > (d)
for the large majority of test items. That ranking-consistency rate is
reported as a "sanity check pass rate" alongside the raw scores - it's a
proxy for "does this metric reflect real quality" that doesn't require
us to already have human quality ratings (which we don't have in this
project). See README for what a full human-validation study would add.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List

from .llm_client import LLMClient
from .retrieval import Retriever

JUDGE_SYSTEM_PROMPT = """JUDGE
You are a strict, careful evaluator of customer-facing email replies. You
will be given: the incoming email, a checklist of required key points, and
a candidate reply. You must return ONLY valid JSON (no markdown fences, no
commentary) with this exact shape:

{
  "key_point_coverage": [
    {"point": "<the key point text, verbatim>", "addressed": true or false, "why": "<one short sentence>"}
  ],
  "rubric": {
    "relevance": <integer 1-5>,
    "completeness": <integer 1-5>,
    "tone": <integer 1-5>,
    "actionability": <integer 1-5>
  },
  "hallucination": {
    "flag": true or false,
    "why": "<one short sentence, empty string if flag is false>"
  }
}

Scoring guidance:
- relevance: does the reply address what the sender actually asked/needs?
- completeness: does it cover everything needed for the sender to be unblocked, not just part of it?
- tone: professional, warm, appropriately apologetic/collaborative for the situation?
- actionability: is there a clear concrete next step, timeframe, or resolution (not vague "we'll look into it")?
- hallucination.flag = true ONLY if the reply states a specific fact (price, date, policy, promise, order status) that is NOT supported by the incoming email or general reasonable business practice. Do not flag reasonable, unremarkable phrasing.
Be strict: do not mark a key point "addressed" unless the reply actually, substantively covers it.
"""


@dataclass
class EvalResult:
    coverage: float
    coverage_detail: List[dict]
    rubric: Dict[str, int]
    rubric_avg: float
    similarity: float
    hallucination_flag: bool
    hallucination_why: str
    overall: float
    judge_raw_ok: bool = True
    notes: str = ""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip mock tag / markdown fences defensively
    text = re.sub(r"^\[MOCK MODE\]\s*", "", text)
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object found in judge output")
    return json.loads(match.group(0))


def _heuristic_judge(incoming_email: str, key_points: List[str], reply: str) -> dict:
    """Deterministic fallback used only when the real judge output can't be
    parsed (e.g. mock mode with no API key). Clearly a weaker proxy than a
    real LLM judge - keyword/substring overlap - but keeps the pipeline
    runnable end-to-end and is labeled as such in the report."""
    reply_lower = reply.lower()
    coverage_detail = []
    for kp in key_points:
        # crude signal: does the reply share a notable content word with the key point?
        kp_words = [w for w in re.findall(r"[a-z]+", kp.lower()) if len(w) > 4]
        hit = any(w in reply_lower for w in kp_words)
        coverage_detail.append({"point": kp, "addressed": hit, "why": "heuristic keyword match (fallback mode)"})
    length_score = min(len(reply.split()) / 40, 1.0)  # very rough "did it say enough"
    rubric = {
        "relevance": 3,
        "completeness": 2 + round(length_score * 2),
        "tone": 3,
        "actionability": 3,
    }
    return {
        "key_point_coverage": coverage_detail,
        "rubric": rubric,
        "hallucination": {"flag": False, "why": ""},
    }


class Evaluator:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()

    def _judge(self, incoming_email: str, key_points: List[str], reply: str) -> tuple[dict, bool]:
        prompt = (
            f"INCOMING EMAIL:\n{incoming_email}\n\n"
            f"REQUIRED KEY POINTS (checklist):\n"
            + "\n".join(f"- {kp}" for kp in key_points)
            + f"\n\nCANDIDATE REPLY:\n{reply}\n\nReturn the JSON now."
        )
        raw = self.llm.complete(JUDGE_SYSTEM_PROMPT, prompt, max_tokens=700)
        try:
            return _extract_json(raw), True
        except Exception:
            return _heuristic_judge(incoming_email, key_points, reply), False

    def evaluate(
        self,
        incoming_email: str,
        key_points: List[str],
        reply: str,
        reference_reply: str,
    ) -> EvalResult:
        judged, ok = self._judge(incoming_email, key_points, reply)

        coverage_detail = judged.get("key_point_coverage", [])
        if coverage_detail:
            coverage = sum(1 for c in coverage_detail if c.get("addressed")) / len(coverage_detail)
        else:
            coverage = 0.0

        rubric = judged.get("rubric", {"relevance": 3, "completeness": 3, "tone": 3, "actionability": 3})
        rubric_vals = [rubric.get(k, 3) for k in ("relevance", "completeness", "tone", "actionability")]
        rubric_avg_raw = sum(rubric_vals) / len(rubric_vals)
        rubric_avg = (rubric_avg_raw - 1) / 4  # normalize 1-5 -> 0-1

        similarity = self.retriever.similarity_to(reply, reference_reply)

        halluc = judged.get("hallucination", {"flag": False, "why": ""})
        hallucination_flag = bool(halluc.get("flag", False))
        hallucination_why = halluc.get("why", "")
        penalty = 0.20 if hallucination_flag else 0.0

        overall = 0.40 * coverage + 0.35 * rubric_avg + 0.15 * similarity - penalty
        overall = max(0.0, min(1.0, overall))

        return EvalResult(
            coverage=coverage,
            coverage_detail=coverage_detail,
            rubric=rubric,
            rubric_avg=rubric_avg,
            similarity=similarity,
            hallucination_flag=hallucination_flag,
            hallucination_why=hallucination_why,
            overall=overall,
            judge_raw_ok=ok,
            notes="" if ok else "judge output fell back to heuristic scoring (mock/parse failure)",
        )
