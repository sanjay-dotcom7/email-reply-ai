"""
Retrieval over the training corpus.

Why TF-IDF instead of a neural embedding model:
- Zero external dependency / API cost / no model download - runs instantly,
  anywhere, deterministically.
- The dataset is small (24 training emails) and vocabulary-driven (order
  numbers, plan names, "refund", "reschedule", etc.) - exact/near-exact term
  overlap is actually a *better* signal here than dense semantic embeddings,
  which tend to blur together short, formulaic business emails.
- It's transparent and debuggable: you can inspect the vocabulary and see
  exactly why an example was retrieved.

Trade-off we're accepting: TF-IDF won't catch pure paraphrase similarity
("my package never showed up" vs "delivery never arrived") as well as an
embedding model would. For a production system with a larger, more
linguistically diverse corpus, swapping this for embeddings (e.g. Voyage or
OpenAI embeddings) behind the same `Retriever` interface would be the
natural upgrade - see README "Trade-offs" section.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Example:
    id: str
    category: str
    subject: str
    incoming_email: str
    reference_reply: str
    key_points: List[str]


class Retriever:
    def __init__(self, corpus: List[Example]):
        self.corpus = corpus
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        texts = [f"{e.subject} {e.incoming_email}" for e in corpus]
        self._matrix = self._vectorizer.fit_transform(texts)

    def top_k(self, query_subject: str, query_email: str, k: int = 3) -> List[Example]:
        query_vec = self._vectorizer.transform([f"{query_subject} {query_email}"])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = sims.argsort()[::-1][:k]
        return [self.corpus[i] for i in ranked_idx]

    def similarity_to(self, text_a: str, text_b: str) -> float:
        """Generic TF-IDF cosine similarity between two arbitrary strings,
        refit on the fly against just these two docs plus the training
        corpus vocabulary, so it stays comparable across calls."""
        vecs = self._vectorizer.transform([text_a, text_b])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(sim)


def load_corpus(dataset_path: Path, split: str | None = None) -> List[Example]:
    raw = json.loads(dataset_path.read_text())
    if split:
        raw = [r for r in raw if r["split"] == split]
    return [
        Example(
            id=r["id"],
            category=r["category"],
            subject=r["subject"],
            incoming_email=r["incoming_email"],
            reference_reply=r["reference_reply"],
            key_points=r["key_points"],
        )
        for r in raw
    ]
