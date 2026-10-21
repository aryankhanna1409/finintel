"""Risk-theme taxonomy and intensity profiling over Item 1A / Q&A chunks.

Each theme is a set of regexes. A company's *intensity* for a theme is the share of its
risk-factor sentences that hit the theme, which normalizes for 10-K length differences
(a 60-page 1A shouldn't look riskier than a 20-page one just because it is longer).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..rag.text import split_sentences
from ..schemas.common import FormType
from ..schemas.retrieval import Chunk

THEMES: dict[str, list[str]] = {
    "discretionary_demand": [r"discretionary", r"consumer (?:budget|spending|demand)", r"\btraffic\b", r"cautious", r"financially constrained"],
    "tariffs_trade": [r"tariff", r"trade polic", r"import(?:ed)? (?:merchandise|goods)", r"sourcing countr"],
    "brand_reputation": [r"brand and reputation", r"reputation", r"boycott", r"advocacy", r"social media", r"brand perception", r"rebuilding trust"],
    "shrink_theft": [r"shrink", r"theft", r"organized retail crime"],
    "competition_pricing": [r"competition", r"compet(?:e|itor|itive)", r"pricing pressure", r"gaining share", r"price"],
    "inventory_markdowns": [r"inventory", r"markdown", r"under-assorted", r"excess"],
    "leadership_transition": [r"chief executive officer succession", r"ceo transition", r"transition of our chief executive", r"senior leadership", r"key leaders", r"succession"],
    "cybersecurity": [r"cyber", r"information systems", r"data breach"],
    "leverage_liquidity": [r"indebtedness", r"lease obligations", r"credit rating", r"liquidity", r"debt reduction"],
    "labor_costs": [r"labor", r"union", r"wage"],
    "membership_model": [r"membership", r"renewal rate"],
    "regulatory_legal": [r"litigation", r"regulat", r"merger agreement", r"reimbursement"],
    "capital_return": [r"repurchase", r"buyback", r"dividend"],
}
_COMPILED = {t: [re.compile(p, re.I) for p in ps] for t, ps in THEMES.items()}


def themes_in(text: str) -> set[str]:
    return {t for t, rxs in _COMPILED.items() if any(r.search(text) for r in rxs)}


@dataclass
class ThemeProfile:
    issuer_cik: str
    intensity: dict[str, float]  # share of sentences hitting the theme
    sentence_count: int
    evidence: dict[str, list[str]] = field(default_factory=dict)  # theme -> chunk ids


def profile_chunks(issuer_cik: str, chunks: list[Chunk]) -> ThemeProfile:
    hits: dict[str, int] = defaultdict(int)
    ev: dict[str, list[str]] = defaultdict(list)
    n = 0
    for c in chunks:
        for s in split_sentences(c.text):
            n += 1
            for t in themes_in(s):
                hits[t] += 1
                if c.metadata.chunk_id not in ev[t]:
                    ev[t].append(c.metadata.chunk_id)
    n = max(n, 1)
    return ThemeProfile(issuer_cik=issuer_cik, intensity={t: hits.get(t, 0) / n for t in THEMES}, sentence_count=n, evidence=dict(ev))


def risk_factor_chunks(chunks: list[Chunk], issuer_cik: str, latest: bool = True) -> list[Chunk]:
    """Item 1A chunks from the latest (or the prior) 10-K, plus 10-Q Part II 1A updates
    filed after that 10-K when ``latest`` is set."""
    tenk = sorted({(c.metadata.filed_date, c.metadata.accession) for c in chunks
                   if c.metadata.issuer_cik == issuer_cik and c.metadata.form_type == FormType.TEN_K and c.metadata.item == "1A"})
    if not tenk:
        return []
    pick = tenk[-1] if latest else (tenk[-2] if len(tenk) > 1 else None)
    if pick is None:
        return []
    out = [c for c in chunks if c.metadata.accession == pick[1] and c.metadata.item == "1A"]
    if latest:
        out += [c for c in chunks if c.metadata.issuer_cik == issuer_cik and c.metadata.form_type == FormType.TEN_Q
                and c.metadata.item == "II-1A" and c.metadata.filed_date >= pick[0]]
    return out


def analyst_question_themes(chunks: list[Chunk], issuer_cik: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Count analyst questions per theme on the latest earnings call."""
    calls = sorted({(c.metadata.filed_date, c.metadata.accession) for c in chunks
                    if c.metadata.issuer_cik == issuer_cik and c.metadata.form_type == FormType.TRANSCRIPT})
    if not calls:
        return {}, {}
    acc = calls[-1][1]
    counts: dict[str, int] = defaultdict(int)
    ev: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        if c.metadata.accession != acc or c.metadata.chunk_type != "qa_pair":
            continue
        question = c.text.split("\n", 1)[0]
        for t in themes_in(question):
            counts[t] += 1
            ev[t].append(c.metadata.chunk_id)
    return dict(counts), dict(ev)
