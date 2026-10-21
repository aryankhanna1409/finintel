"""Peer benchmarking: valuation joins and robust peer statistics."""

from __future__ import annotations

import statistics
from datetime import date
from typing import Callable

from ..schemas.metrics import CompanyValuation, PeerStatistic


def _q(vals: list[float], p: float) -> float:
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


METRICS: dict[str, tuple[Callable[[CompanyValuation], float | None], bool]] = {
    # name: (extractor, higher_is_better from a shareholder's perspective)
    "ev_to_ebitda": (lambda v: v.ev_to_ebitda, True),  # higher multiple = market rewards more
    "pe": (lambda v: v.pe, True),
    "fcf_yield": (lambda v: v.fcf_yield, False),  # high yield = cheap
    "dividend_yield": (lambda v: v.market.dividend_yield, False),
    "shareholder_yield": (lambda v: v.shareholder_yield, False),
    "operating_margin": (lambda v: v.fundamentals.operating_margin, True),
    "roic": (lambda v: v.fundamentals.roic, True),
    "net_debt_to_ebitda": (lambda v: v.fundamentals.net_debt_to_ebitda, False),
    "interest_coverage": (lambda v: v.fundamentals.interest_coverage, True),
    "cash_conversion_cycle": (lambda v: v.fundamentals.cash_conversion_cycle, False),
    "dio": (lambda v: v.fundamentals.dio, False),
    "tsr_1y": (lambda v: v.market.return_1y, True),
    "tsr_3y": (lambda v: v.market.return_3y, True),
    "debt_to_capital_market": (lambda v: v.debt_to_capital_market, False),
}


def peer_statistic(name: str, target: CompanyValuation, peers: list[CompanyValuation], as_of: date | None = None) -> PeerStatistic:
    fn, hib = METRICS[name]
    tv = fn(target)
    pv = {p.ticker: fn(p) for p in peers}
    vals = [v for v in pv.values() if v is not None]
    if not vals:
        return PeerStatistic(metric=name, target_value=tv, peer_values=pv, peer_median=None, peer_p25=None, peer_p75=None,
                             z_score=None, percentile_rank=None, higher_is_better=hib, as_of=as_of)
    med = statistics.median(vals)
    # Robust z: (x - median) / (1.4826 * MAD); falls back to stdev for tiny/degenerate samples.
    mad = statistics.median([abs(v - med) for v in vals]) * 1.4826
    scale = mad if mad > 1e-12 else (statistics.pstdev(vals) or None)
    z = (tv - med) / scale if (tv is not None and scale) else None
    pct = (sum(1 for v in vals if v < tv) + 0.5 * sum(1 for v in vals if v == tv)) / len(vals) if tv is not None else None
    return PeerStatistic(metric=name, target_value=tv, peer_values=pv, peer_median=med, peer_p25=_q(vals, 0.25), peer_p75=_q(vals, 0.75),
                         z_score=round(z, 3) if z is not None else None, percentile_rank=pct, higher_is_better=hib, as_of=as_of)


def benchmark(target: CompanyValuation, peers: list[CompanyValuation], as_of: date | None = None) -> list[PeerStatistic]:
    return [peer_statistic(m, target, peers, as_of) for m in METRICS]


def stat(stats: list[PeerStatistic], name: str) -> PeerStatistic:
    return next(s for s in stats if s.metric == name)
