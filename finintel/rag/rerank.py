"""Second-stage rerankers.

* :class:`CrossEncoderReranker`: true cross-encoder (query and passage jointly encoded),
  e.g. ``BAAI/bge-reranker-base`` or ``cross-encoder/ms-marco-MiniLM-L-6-v2``. Optional.
* :class:`FinancialFeatureReranker`: dependency-free pairwise scorer that approximates
  cross-attention with explicit query-passage interaction features: IDF-weighted term
  coverage, ordered bigram/phrase hits, minimal covering-window proximity, exact
  numeric/ticker agreement, and intent-to-section priors (risk → Item 1A, results →
  MD&A/tables, analyst concerns → Q&A).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Protocol, Sequence

from ..schemas.retrieval import Chunk
from .text import FINANCE_CONCEPTS, NUMBER_RE, query_tokens, tokenize

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model_name, max_length=max_length)
        self.name = model_name

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        if not chunks:
            return []
        raw = self.model.predict([(query, c.index_text or c.text) for c in chunks])
        return [1 / (1 + math.exp(-float(x))) for x in raw]


_QUANT_RE = re.compile(r"\b(how much|how many|what (?:was|were|is|are) the|amount|total|spent|spend|paid|percent|ratio|level)\b", re.I)

_INTENTS: list[tuple[re.Pattern[str], dict[str, float], dict[str, float]]] = [
    # (query pattern, item priors, chunk-type priors)
    (re.compile(r"\brisk|exposure|threat|vulnerab|uncertain", re.I), {"1A": 0.35, "II-1A": 0.35}, {}),
    (re.compile(r"\b(result|sales|revenue|margin|earnings|eps|income|cash flow|inventory|how much|what was)\b", re.I), {"7": 0.25, "I-2": 0.25, "2.02": 0.15}, {"table": 0.2}),
    (re.compile(r"\b(analyst|investor|concern|question|call|asked|q&a|debate)\b", re.I), {"Q&A": 0.4}, {"qa_pair": 0.2}),
    (re.compile(r"\b(ceo|succession|leadership|executive|officer)\b", re.I), {"5.02": 0.35}, {}),
    (re.compile(r"\b(buyback|repurchase|dividend|capital allocation|return cash)\b", re.I), {"7": 0.15, "I-2": 0.15, "Q&A": 0.1}, {}),
]


class FinancialFeatureReranker:
    name = "financial-feature-reranker"

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        self.idf = idf or {}

    def _idf(self, t: str) -> float:
        w = self.idf.get(t, 2.0)
        return w * 2.5 if t in FINANCE_CONCEPTS else w

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        q_toks = query_tokens(query)
        quant_intent = bool(_QUANT_RE.search(query))
        if not q_toks:
            return [0.0] * len(chunks)
        q_set = list(dict.fromkeys(q_toks))
        q_bigrams = set(zip(q_toks, q_toks[1:]))
        q_nums = {t for t in q_set if t[0].isdigit()}
        q_upper = set(re.findall(r"\b[A-Z]{2,5}\b", query))
        total_idf = sum(self._idf(t) for t in q_set) or 1.0
        priors_item: dict[str, float] = {}
        priors_type: dict[str, float] = {}
        for rx, items, types in _INTENTS:
            if rx.search(query):
                for k, v in items.items():
                    priors_item[k] = max(priors_item.get(k, 0.0), v)
                for k, v in types.items():
                    priors_type[k] = max(priors_type.get(k, 0.0), v)
        out = []
        for c in chunks:
            d_toks = tokenize(c.text)
            d_set = set(d_toks)
            coverage = sum(self._idf(t) for t in q_set if t in d_set) / total_idf
            bigram_hits = len(q_bigrams & set(zip(d_toks, d_toks[1:]))) / max(len(q_bigrams), 1)
            prox = self._proximity(q_set, d_toks)
            num_match = (len(q_nums & d_set) / len(q_nums)) if q_nums else 0.0
            ticker_match = 1.0 if (q_upper and c.metadata.ticker in q_upper) else 0.0
            item_prior = priors_item.get(c.metadata.item or "", 0.0)
            type_prior = priors_type.get(c.metadata.chunk_type, 0.0)
            boiler = 1.0 if (c.metadata.item == "PREAMBLE" and c.metadata.form_type.value in ("10-K", "10-Q", "8-K")) else 0.0
            has_amounts = 1.0 if (quant_intent and re.search(r"\$\s?\d|\d+(?:\.\d+)?\s?(?:%|percent|billion|million)", c.text)) else 0.0
            z = (
                -2.2 + 3.2 * coverage + 1.0 * bigram_hits + 0.8 * prox + 0.6 * num_match
                + 0.5 * ticker_match + 1.2 * item_prior + 1.0 * type_prior + 0.6 * has_amounts - 1.0 * boiler
            )
            out.append(1 / (1 + math.exp(-z)))
        return out

    @staticmethod
    def _proximity(q_terms: list[str], d_toks: list[str]) -> float:
        """1 / (1 + minimal window length covering the matched query terms, normalized)."""
        positions = {t: [i for i, d in enumerate(d_toks) if d == t] for t in q_terms}
        present = {t: p for t, p in positions.items() if p}
        if len(present) < 2:
            return 0.0
        events = sorted((i, t) for t, ps in present.items() for i in ps)
        need, have, best = len(present), {}, math.inf
        left = 0
        for right, (pos_r, t_r) in enumerate(events):
            have[t_r] = have.get(t_r, 0) + 1
            while len(have) == need:
                pos_l, t_l = events[left]
                best = min(best, pos_r - pos_l + 1)
                have[t_l] -= 1
                if have[t_l] == 0:
                    del have[t_l]
                left += 1
        return need / best if best != math.inf else 0.0


def get_reranker(spec: str | None, idf: dict[str, float] | None = None) -> Reranker:
    if not spec or spec == "features":
        return FinancialFeatureReranker(idf)
    try:
        return CrossEncoderReranker(spec)
    except Exception as exc:  # noqa: BLE001 - optional backend
        log.warning("cross-encoder %r unavailable (%s); using FinancialFeatureReranker", spec, exc)
        return FinancialFeatureReranker(idf)
