# AI Email Suggested-Response System

An end-to-end system that (1) learns from a dataset of past emails and the
replies that were sent, (2) generates a suggested reply to a new incoming
email using an LLM, grounded in that history, and (3) scores how good the
generated reply actually is, with a report per response and an overall
system score.

Runs with zero setup in **mock mode** (no API key needed — see below), and
with real generation/judging once you set `ANTHROPIC_API_KEY`.

## Quick start

```bash
pip install -r requirements.txt

# Optional but recommended for real (non-mock) results:
export ANTHROPIC_API_KEY=sk-ant-...
# export CLAUDE_MODEL=claude-sonnet-4-5   # optional override

# Run one email through the full pipeline (generate + evaluate):
python -m src.pipeline --id cs_05

# Or a free-form new email (no scoring, since there's no reference/checklist):
python -m src.pipeline --subject "Refund status" --body "It's been 10 days since I cancelled and I still haven't seen my refund."

# Run the full evaluation suite over the held-out test set:
python -m src.run_eval_suite
# -> writes results/eval_report.json and prints a summary to the console
```

Without `ANTHROPIC_API_KEY` set, everything above still runs, but
generation produces a clearly-labeled `[MOCK MODE]` placeholder reply and
the judge falls back to a deterministic heuristic scorer (also labeled).
This is intentional: the whole pipeline — dataset, retrieval, generation,
scoring, reporting — is provable end-to-end without requiring credentials.
**The numbers in mock mode are not meaningful quality scores** — they exist
to prove the plumbing works. Set the API key for real results.

## 1. The dataset (`data/dataset.json`)

**What it is:** 30 hand-authored (email, reply) pairs across 6 common
business categories — customer support, sales inquiry, scheduling,
billing, complaint escalation, and vendor negotiation — 5 pairs per
category. I wrote these myself (with Claude drafting candidate pairs
which I then edited/curated — see "How I used AI tools" below), rather
than sourcing a public corpus.

**Why synthetic/hand-authored instead of a public corpus (e.g. Enron):**
Public email corpora are either (a) not paired with the actual reply that
was sent in a clean, labeled way, (b) full of PII/legal baggage that's a
distraction for a take-home project, or (c) heavily skewed toward one
narrow domain (Enron is almost entirely internal energy-trading
correspondence — not representative of general business email). A small,
hand-curated dataset lets me control for exactly the property the
evaluator needs: **each email has an explicit, human-defined checklist of
what a good reply must cover** (`key_points`). That checklist is the
backbone of the whole evaluation system, and it doesn't exist in any
public corpus I'm aware of — I'd have had to hand-label it anyway.

**Is it representative?** Reasonably, for what it claims to be: common,
short-form transactional business email (support tickets, sales
questions, meeting logistics, invoices, complaints, B2B negotiation). It
is *not* representative of, e.g., highly technical engineering
threads, legal correspondence, or multi-paragraph relationship-building
emails — a real production system would need a much larger and more
varied set, ideally sourced from the company's own historical send
records (with PII scrubbed). I say this plainly rather than overclaiming
generality from 30 examples.

**Split:** 24 "train" (used only as the retrieval corpus / few-shot
source) and 6 "test" (held out — one per category — used to generate and
score new replies). The generator never sees a test email's reference
reply or checklist; retrieval only searches the train split.

## 2. The generator (`src/generator.py`, `src/retrieval.py`)

**Approach: retrieval-augmented few-shot prompting**, not fine-tuning and
not a classical classifier.

1. `Retriever` builds a TF-IDF index over the 24 training emails.
2. For a new incoming email, it retrieves the top-k (default 3) most
   similar past emails *and their real sent replies*.
3. `generator.py` builds a prompt that shows the model those k examples
   as worked precedent, then asks it to write a reply to the new email —
   plain text only, matching the tone/structure/concreteness of the
   examples, and explicitly instructed not to invent unsupported facts.

**Why prompting + retrieval over fine-tuning:** 24 training examples is
far too few to fine-tune without overfitting to their specific phrasing.
Retrieval-augmented prompting adapts immediately when new examples are
added (no retraining), and is fully auditable — the report shows exactly
which past emails grounded each generated reply. The trade-off is weaker
"baked-in" house style than a fine-tune would eventually give at scale,
and prompt length grows with k; if this dataset grew to thousands of real
historical emails, fine-tuning or a proper vector-DB RAG setup would
become worth revisiting.

**Why TF-IDF retrieval over embeddings:** no extra API cost/dependency,
instant, deterministic, and for this domain (formulaic business email
with concrete tokens like order numbers and plan names) exact/near-exact
term overlap is a strong, transparent signal — you can inspect the
vocabulary and see exactly *why* an example was retrieved. The honest
trade-off: it won't catch pure paraphrase ("package never showed up" vs.
"delivery never arrived") as well as a dense embedding model would. An
embedding-based `Retriever` swap is the natural upgrade path and would
sit behind the same interface.

## 3. The evaluation system (`src/evaluator.py`) — the core of this project

**What "accurate" means here.** Exact string match against the reply a
human happened to send is the wrong bar — two completely different
sentences can both resolve an email perfectly well. So instead of one
brittle "accuracy" number, the evaluator scores four independent,
individually-meaningful properties and combines them:

| Signal | Weight | What it measures | How it's computed |
|---|---|---|---|
| **Key-point coverage** | 0.40 | Does the reply hit the specific facts/actions this email requires? | LLM judge checks off a human-authored checklist (`key_points`) item by item |
| **Rubric score** | 0.35 | Relevance, completeness, tone, actionability (1–5 each) | LLM-as-judge, normalized to 0–1 |
| **Reference similarity** | 0.15 | Is this even topically in the right neighborhood? | TF-IDF cosine vs. the actually-sent reply |
| **Hallucination penalty** | −0.20 if flagged | Did the reply invent a fact (price, date, promise) not supported by the email? | LLM judge, binary flag + reason |

```
overall = 0.40·coverage + 0.35·rubric_avg + 0.15·similarity − hallucination_penalty
```

**Why weighted this way:** coverage is weighted highest because it's the
most *grounded* signal — the checklist was written by a human ahead of
time, independent of whatever the judge later says, so it's the hardest
signal to game. The rubric captures writing quality a checklist can't.
Reference similarity is weighted lowest on purpose: it's the weakest,
noisiest signal (good replies can look very different from each other),
but it's cheap and catches the pathological "topically unrelated"
failure. The hallucination penalty is separate and subtractive rather
than folded into the rubric, because a reply can score well on coverage
*and* tone while still fabricating a commitment — that's a distinct and
more dangerous failure mode for something that ships email on someone's
behalf, and it deserves its own explicit flag rather than getting
averaged away.

**How I validated the metric actually reflects quality (not just a
number I made up).** With no separate human-labeled quality dataset
available, `run_eval_suite.py` runs a calibration harness: for every test
email, it scores **four variants** of "a reply":

- (a) the reply the generator actually produced,
- (b) the ground-truth reply a human actually sent — should score
  highest or tied,
- (c) a reply that was actually sent for a *different, unrelated* test
  email — should score noticeably lower (wrong topic, wrong facts),
- (d) an empty string — should score at the floor.

If the metric is sound, scores should rank **(b) ≥ (a) > (c) > (d)** for
the large majority of test emails. That ranking-consistency rate is
reported as `calibration_pass_rate` in `results/eval_report.json`
alongside the raw scores. This doesn't prove the metric matches human
judgment perfectly, but it's a concrete, falsifiable sanity check — if
the metric couldn't even separate a real reply from an empty string or a
mismatched one, nothing else about it would be trustworthy. **The
natural next step for a production version** would be collecting ~50–100
real human quality ratings (e.g. 1–5 "would I send this") and computing
correlation (Spearman) between human ratings and `overall_score` — I did
not do that here since it requires actual human raters, but the harness
is structured so that swap-in would be straightforward.

**Reporting.** `results/eval_report.json` contains, per test email: the
generated reply, which past examples grounded it, the full checklist
breakdown (which points hit/missed and why), raw rubric scores, the
similarity score, the hallucination flag, and the combined overall score
— plus the calibration detail described above. The console summary and
the report both also surface an **overall system score** (mean of
per-response scores) and an aggregate **hallucination rate**.

## Project structure

```
data/dataset.json        30 hand-authored (email, reply, key_points) triples
src/retrieval.py         TF-IDF retriever over the training split
src/llm_client.py        Anthropic API wrapper + honest mock-mode fallback
src/generator.py         retrieval + few-shot prompt -> generated reply
src/evaluator.py         the 4-signal scorer described above
src/pipeline.py          CLI: run one email end-to-end
src/run_eval_suite.py    CLI: run + score the full test set, incl. calibration harness
results/eval_report.json output of the last run_eval_suite run
```

## Known limitations / honest caveats

- 30 emails is a small dataset. It's enough to demonstrate the approach
  and let the evaluation methodology be checked, but too small to
  generalize confidently to open-ended real-world email.
- The rubric and coverage judge is itself an LLM — it can be wrong or
  inconsistent. The calibration harness checks *relative* sanity
  (ranking), not absolute correctness against ground-truth human labels,
  because none were collected here.
- TF-IDF similarity is a weak proxy for semantic closeness; it's
  deliberately low-weighted for this reason.
- Mock-mode scores are not meaningful — they only demonstrate the
  pipeline runs without credentials.

## How I used AI tools

I (Claude) built this repository directly, end-to-end, in collaboration
with the person I was working with — drafting the dataset entries,
writing all the Python (retrieval, generation, evaluation, CLI/report
code), designing the scoring rubric and the calibration/validation
approach, and writing this README. The person guided scope and reviewed
the design as we went. I actually ran the code in a sandbox while
building it (installing dependencies, running the pipeline in mock mode,
catching and fixing a real bug in the mock-judge fallback) rather than
just writing code I expected to work — the commands and output shown in
this README reflect an actual run, not a prediction.
