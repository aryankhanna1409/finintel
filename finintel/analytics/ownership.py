"""Ownership analytics: concentration (HHI), position flows, common ownership, insider signals."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta

from ..schemas.advisory import InsiderSignal
from ..schemas.common import InstitutionType
from ..schemas.entities import Institution
from ..schemas.graph import CommonOwnershipLink
from ..schemas.holdings import HoldingRecord, InsiderTransaction, InsiderTxnCode, OwnershipConcentration, OwnershipFlow


def equity_positions(holdings: list[HoldingRecord], period: date) -> dict[tuple[str, str], tuple[float, float, str]]:
    """(filer, issuer) -> (shares, value, filer_name) for equity rows only (no debt, no options)."""
    pos: dict[tuple[str, str], list] = {}
    for h in holdings:
        if h.report_period != period or not h.issuer_cik or not h.counts_toward_equity_ownership:
            continue
        k = (h.filer_cik, h.issuer_cik)
        cur = pos.setdefault(k, [0.0, 0.0, h.filer_name])
        cur[0] += h.shares
        cur[1] += h.value_usd
    return {k: (v[0], v[1], v[2]) for k, v in pos.items()}


def concentration(
    holdings: list[HoldingRecord], issuer_cik: str, period: date, shares_outstanding: float, institutions: dict[str, Institution]
) -> OwnershipConcentration:
    pos = {f: v for (f, i), v in equity_positions(holdings, period).items() if i == issuer_cik and v[0] > 0}
    total = sum(v[0] for v in pos.values())
    shares = sorted((v[0] for v in pos.values()), reverse=True)
    hhi = sum((s / total * 100) ** 2 for s in shares) if total else 0.0
    activist = sum(v[0] for f, v in pos.items() if institutions.get(f) and (institutions[f].is_known_activist or institutions[f].institution_type == InstitutionType.ACTIVIST))
    index = sum(v[0] for f, v in pos.items() if institutions.get(f) and institutions[f].institution_type == InstitutionType.INDEX_MANAGER)
    return OwnershipConcentration(
        issuer_cik=issuer_cik, report_period=period, holders=len(pos), institutional_shares=total,
        institutional_pct_outstanding=total / shares_outstanding, hhi=hhi,
        top10_pct_institutional=(sum(shares[:10]) / total) if total else 0.0,
        activist_pct_outstanding=activist / shares_outstanding, index_pct_outstanding=index / shares_outstanding,
    )


def flows(holdings: list[HoldingRecord], period_from: date, period_to: date) -> list[OwnershipFlow]:
    a = equity_positions(holdings, period_from)
    b = equity_positions(holdings, period_to)
    out = []
    for k in set(a) | set(b):
        sa, va, na = a.get(k, (0.0, 0.0, ""))
        sb, vb, nb = b.get(k, (0.0, 0.0, ""))
        out.append(OwnershipFlow(filer_cik=k[0], filer_name=nb or na, issuer_cik=k[1], period_from=period_from, period_to=period_to,
                                 shares_from=sa, shares_to=sb, value_from=va, value_to=vb))
    return out


def price_neutral_value_change(f: OwnershipFlow) -> float:
    """Value change attributable to trading, not price moves: share change × end price."""
    if f.shares_to > 0:
        px = f.value_to / f.shares_to
    elif f.shares_from > 0:
        px = f.value_from / f.shares_from
    else:
        return 0.0
    return f.share_change * px


def common_ownership(
    holdings: list[HoldingRecord],
    period: date,
    ciks: list[str],
    names: dict[str, str],
    shares_out: dict[str, float],
    institutions: dict[str, Institution] | None = None,
    exclude_types: frozenset[InstitutionType] = frozenset({InstitutionType.INDEX_MANAGER}),
) -> list[CommonOwnershipLink]:
    """Cosine similarity of companies' *active* holder vectors (% of shares outstanding held
    by each filer). Index funds are excluded by default: they own every name pro rata and
    would push every pair toward 1.0. High active overlap = same marginal owners =
    correlated flows and shared governance views."""
    pos = equity_positions(holdings, period)
    vec: dict[str, dict[str, float]] = defaultdict(dict)
    for (f, i), (s, _v, n) in pos.items():
        inst = (institutions or {}).get(f)
        if inst is not None and inst.institution_type in exclude_types:
            continue
        if i in ciks and shares_out.get(i):
            vec[i][f] = s / shares_out[i]
    filer_names = {f: n for (f, _i), (_s, _v, n) in pos.items()}
    out = []
    for x in range(len(ciks)):
        for y in range(x + 1, len(ciks)):
            a, b = vec.get(ciks[x], {}), vec.get(ciks[y], {})
            shared = set(a) & set(b)
            dot = sum(a[f] * b[f] for f in shared)
            na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
            cos = dot / (na * nb) if na and nb else 0.0
            top = sorted(shared, key=lambda f: -(a[f] * b[f]))[:3]
            out.append(CommonOwnershipLink(company_a=names.get(ciks[x], ciks[x]), company_b=names.get(ciks[y], ciks[y]), overlap_cosine=round(cos, 4),
                                           shared_holders=len(shared), top_shared=[filer_names[f] for f in top]))
    return sorted(out, key=lambda l: -l.overlap_cosine)


def insider_signal(txns: list[InsiderTransaction], issuer_cik: str, as_of: date, window_days: int = 120, cluster_days: int = 30) -> InsiderSignal:
    """Open-market (P/S) insider activity, discounting pre-planned 10b5-1 sales.

    Cluster buy = 3+ distinct insiders buying in the open market within ``cluster_days``,
    which is historically the most informative insider pattern.
    """
    start = as_of - timedelta(days=window_days)
    rel = [t for t in txns if t.issuer_cik == issuer_cik and start <= t.transaction_date <= as_of and t.code.is_discretionary]
    buys = [t for t in rel if t.code == InsiderTxnCode.P]
    sells = [t for t in rel if t.code == InsiderTxnCode.S]
    bv = sum(t.value_usd for t in buys)
    sv = sum(t.value_usd for t in sells)
    planned = sum(t.value_usd for t in sells if t.is_10b5_1)
    discretionary_sells = sv - planned
    buyers = {t.insider_cik or t.insider_name for t in buys}
    cluster = False
    bdates = sorted(buys, key=lambda t: t.transaction_date)
    for i, t in enumerate(bdates):
        window = {u.insider_cik or u.insider_name for u in bdates[i:] if (u.transaction_date - t.transaction_date).days <= cluster_days}
        if len(window) >= 3:
            cluster = True
            break
    # Score in [-1, 1]: log-scaled net discretionary flow plus breadth and cluster premia.
    net = bv - discretionary_sells
    mag = math.copysign(min(math.log10(1 + abs(net)) / 7.0, 1.0), net) if net else 0.0
    breadth = min(len(buyers) / 5.0, 1.0) * 0.3
    score = max(-1.0, min(1.0, 0.6 * mag + breadth + (0.25 if cluster else 0.0) - (0.15 if discretionary_sells > 2 * bv and discretionary_sells > 0 else 0.0)))
    signal = "bullish" if score >= 0.35 else ("bearish" if score <= -0.25 else "neutral")
    return InsiderSignal(
        issuer_cik=issuer_cik, window_days=window_days, open_market_buys=len(buys), open_market_sells=len(sells), buy_value_usd=bv,
        sell_value_usd=sv, net_value_usd=bv - sv, distinct_buyers=len(buyers), cluster_buy=cluster,
        planned_sale_share=(planned / sv) if sv else 0.0, signal=signal, score=round(score, 3),
    )
