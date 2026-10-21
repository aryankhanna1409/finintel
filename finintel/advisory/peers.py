"""Peer-group selection: graph competitors ranked by relationship strength, industry match,
size proximity and ownership overlap; returns 3-5 peers with data coverage."""

from __future__ import annotations

import math

from ..graph.builder import FinancialKnowledgeGraph
from ..ingestion.sources import CorpusBundle


def select_peers(kg: FinancialKnowledgeGraph, bundle: CorpusBundle, target_cik: str, k_min: int = 3, k_max: int = 5) -> list[str]:
    t_iss = bundle.registry.issuers[target_cik]
    t_mcap = bundle.market[target_cik].market_cap if target_cik in bundle.market else None
    candidates: dict[str, float] = {}
    for cik, strength in kg.peers_of(target_cik):
        candidates[cik] = strength
    # Same-SIC issuers with data are admissible even without a disclosed competitor edge.
    for cik, iss in bundle.registry.issuers.items():
        if cik != target_cik and iss.sic and iss.sic == t_iss.sic:
            candidates.setdefault(cik, 0.4)
    scored = []
    for cik, strength in candidates.items():
        if cik not in bundle.fundamentals or cik not in bundle.market:
            continue  # peers without fundamentals can't be benchmarked
        iss = bundle.registry.issuers[cik]
        industry = 1.0 if iss.industry == t_iss.industry else (0.6 if iss.sector == t_iss.sector else 0.2)
        size = 1.0
        if t_mcap:
            size = math.exp(-abs(math.log(bundle.market[cik].market_cap / t_mcap)) / 3)
        scored.append((0.55 * strength + 0.25 * industry + 0.20 * size, cik))
    scored.sort(reverse=True)
    picks = [c for _, c in scored[:k_max]]
    if len(picks) < k_min:
        raise ValueError(f"only {len(picks)} peers with data for {t_iss.name}; need at least {k_min}")
    return picks
