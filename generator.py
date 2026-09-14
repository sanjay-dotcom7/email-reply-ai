"""
Generates a suggested reply for an incoming email.

Approach: retrieval-augmented few-shot prompting.
  1. Retrieve the top-k most similar past (email, reply) pairs from the
     TRAINING split only (never the test split - that would leak the
     answer we're trying to evaluate).
  2. Build a prompt that shows the model those pairs as worked examples,
     grounding tone, structure, and house-style in real precedent rather
     than the model's generic instincts.
  3. Ask for a single plain-text reply, nothing else.

Why prompting + retrieval instead of fine-tuning:
  - Dataset is small (24 training examples) - nowhere near enough to
    fine-tune a model without overfitting to idiosyncrasies of 24 emails.
  - Few-shot + retrieval adapts instantly when new examples are added to
    the dataset (no retraining), and is fully auditable - you can see
    exactly which past emails informed a given reply.
  - Trade-off: less "baked in" style consistency than fine-tuning would
    give at scale, and prompt length grows with k. For a much larger
    dataset (thousands of real historical emails), fine-tuning or a real
    vector-DB RAG setup would start to pay off - noted in the README.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .llm_client import LLMClient
from .retrieval import Example, Retriever, load_corpus

SYSTEM_PROMPT = """GENERATE_REPLY
You are an assistant that drafts email replies on behalf of a company's
support/sales/ops team. You are shown a few real past examples of
(incoming email -> reply that was actually sent) to calibrate tone,
structure, and the kind of concrete commitments this company makes.

Rules:
- Write ONLY the reply body. No subject line, no "Here is a reply:", no
  markdown formatting, no explanation.
- Match the tone and length of the example replies: warm, direct,
  concrete (name specific numbers/dates/order-IDs when the incoming email
  gives you any), and always propose a clear next step or resolution.
- Do not invent facts (prices, dates, policy details) that aren't implied
  by the incoming email or the examples. If something is genuinely
  unknown, say you'll confirm and follow up, rather than fabricating it.
"""


def _format_examples(examples: List[Example]) -> str:
    blocks = []
    for e in examples:
        blocks.append(
            f"--- Past example (category: {e.category}) ---\n"
            f"Incoming email: {e.incoming_email}\n"
            f"Reply that was sent: {e.reference_reply}\n"
        )
    return "\n".join(blocks)


def build_user_prompt(subject: str, incoming_email: str, examples: List[Example]) -> str:
    return (
        f"MOST SIMILAR PAST EXAMPLES (for grounding, not copying verbatim):\n\n"
        f"{_format_examples(examples)}\n"
        f"INCOMING EMAIL:\n"
        f"Subject: {subject}\n"
        f"{incoming_email}\n\n"
        f"MOST SIMILAR PAST EXAMPLES ARE ABOVE. Now write the reply to the "
        f"incoming email above, following the Rules."
    )


class ReplyGenerator:
    def __init__(self, dataset_path: Path, k: int = 3, llm: LLMClient | None = None):
        self.train_corpus = load_corpus(dataset_path, split="train")
        self.retriever = Retriever(self.train_corpus)
        self.k = k
        self.llm = llm or LLMClient()

    def retrieve(self, subject: str, incoming_email: str) -> List[Example]:
        return self.retriever.top_k(subject, incoming_email, k=self.k)

    def generate(self, subject: str, incoming_email: str) -> dict:
        examples = self.retrieve(subject, incoming_email)
        user_prompt = build_user_prompt(subject, incoming_email, examples)
        reply = self.llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=400).strip()
        return {
            "reply": reply,
            "retrieved_example_ids": [e.id for e in examples],
        }
