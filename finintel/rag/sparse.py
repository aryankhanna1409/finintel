"""Okapi BM25 over finance-aware tokens (exact GAAP terms, tickers, ratios, numbers)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

from .text import query_tokens, tokenize


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.72) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    def build(self, docs: Sequence[str]) -> "BM25Index":
        self.postings.clear()
        self.doc_len = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = max(self.n_docs, 1)
        self.avgdl = (sum(self.doc_len) / n) if self.doc_len else 0.0
        # Lucene-style non-negative IDF.
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        return self

    def query_terms(self, query: str) -> list[str]:
        return query_tokens(query)

    def scores(self, query: str, candidate_mask: np.ndarray | None = None) -> np.ndarray:
        s = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return s
        for term, qtf in Counter(self.query_terms(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                if candidate_mask is not None and not candidate_mask[i]:
                    continue
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s[i] += idf * tf * (self.k1 + 1) / denom * (1 + 0.25 * (qtf - 1))
        return s

    def term_idf(self, terms: Iterable[str]) -> dict[str, float]:
        return {t: self.idf.get(t, 0.0) for t in terms}
