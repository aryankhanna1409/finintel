"""Corporate-finance engines: synthetic ratings, WACC curve / optimal capital structure,
and capital-return (buyback vs. dividend) economics."""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

from ..ingestion.market_client import RATING_SCALE
from ..schemas.advisory import CapitalReturnAnalysis, CapitalStructureAnalysis, InsiderSignal, WaccPoint
from ..schemas.metrics import CompanyValuation, ConsensusEstimate, MacroSnapshot

# Interest-coverage → rating map for large non-financial firms (after Damodaran's
# synthetic-rating table). Lower bound of coverage for each rating.
COVERAGE_TO_RATING: list[tuple[float, str]] = [
    (8.5, "AAA"), (6.5, "AA"), (5.5, "A+"), (4.25, "A"), (3.0, "A-"), (2.5, "BBB"), (2.25, "BB+"), (2.0, "BB"),
    (1.75, "B+"), (1.5, "B"), (1.25, "B-"), (0.8, "CCC"), (-math.inf, "CC"),
]


# Cumulative ~10y default probabilities by rating (after Damodaran / S&P long-run studies);
# used for expected distress costs, which create the right-hand side of the U-shaped
# value curve.
CUMULATIVE_PD = {
    "AAA": 0.0007, "AA+": 0.0035, "AA": 0.0051, "AA-": 0.0055, "A+": 0.0060, "A": 0.0066, "A-": 0.0250, "BBB+": 0.0500,
    "BBB": 0.0754, "BBB-": 0.1000, "BB+": 0.1400, "BB": 0.1663, "BB-": 0.2000, "B+": 0.2500, "B": 0.3680, "B-": 0.4500, "CCC": 0.5900,
}
DISTRESS_COST_OF_VALUE = 0.30  # deadweight costs of financial distress as share of firm value


def shift_rating(rating: str, offset: int) -> str:
    i = notch(rating) + offset
    i = max(0, min(i, len(RATING_SCALE)))
    return RATING_SCALE[i] if i < len(RATING_SCALE) else "CC"


def rating_calibration_offset(v: CompanyValuation, macro: MacroSnapshot) -> int:
    """Agencies rate on lease- and pension-adjusted metrics plus business risk; the
    coverage table alone over-rates asset-light balance sheets and under-rates others.
    We compute the synthetic rating at today's debt *marked to market* (interest at the
    current rating's yield, as the curve does) and carry the notch gap to the actual
    rating across the whole curve, so the curve is agency-consistent at the origin."""
    actual = v.market.credit_rating
    if not actual or actual.upper() not in RATING_SCALE or v.fundamentals.total_debt <= 0:
        return 0
    kd = macro.risk_free_rate + spread_for(actual.upper(), macro)
    cov = v.fundamentals.operating_income / (v.fundamentals.total_debt * kd)
    return max(-3, min(6, notch(actual) - notch(synthetic_rating(cov))))


def synthetic_rating(coverage: float | None) -> str:
    if coverage is None or math.isinf(coverage):
        return "AAA"
    for lo, r in COVERAGE_TO_RATING:
        if coverage >= lo:
            return r
    return "CC"


def notch(r: str) -> int:
    """Lower is better. Unknown/below-CCC ratings sit past the end of the scale."""
    r = r.upper()
    return RATING_SCALE.index(r) if r in RATING_SCALE else len(RATING_SCALE)


def spread_for(rating: str, macro: MacroSnapshot) -> float:
    if rating in macro.credit_spreads:
        return macro.credit_spreads[rating]
    if notch(rating) >= len(RATING_SCALE):
        return macro.credit_spreads.get("CCC", 0.09) * 1.6
    return macro.credit_spreads.get(rating.rstrip("+-"), 0.012)


_COMMIT_PATTERNS = [
    (re.compile(r"\b(?:middle|mid)[- ]?A\b", re.I), "A"),
    (re.compile(r"\bstrong AA\b|\bAA credit rating", re.I), "AA-"),
    (re.compile(r"\bsingle[- ]A\b|\bA ratings?\b", re.I), "A-"),
    (re.compile(r"\bstrong investment[- _]grade\b", re.I), "BBB+"),
    (re.compile(r"\binvestment[- _]grade\b", re.I), "BBB-"),
]


def infer_rating_floor(texts: list[str], actual: str | None) -> tuple[str, str | None]:
    """Most restrictive rating commitment stated in management text, else one notch below
    the current rating, never below BBB- (losing investment grade is out of policy)."""
    best: str | None = None
    src = None
    for t in texts:
        for rx, r in _COMMIT_PATTERNS:
            if rx.search(t) and (best is None or notch(r) < notch(best)):
                best, src = r, t[:200]
    if best:
        return best, src
    if actual and actual.upper() in RATING_SCALE:
        return RATING_SCALE[min(notch(actual) + 1, RATING_SCALE.index("BBB-"))], None
    return "BBB-", None


def _cost_of_debt_fixed_point(ebit: float, debt: float, macro: MacroSnapshot, start: str, offset: int = 0) -> tuple[str, float, float | None]:
    rating = start
    for _ in range(12):
        kd = macro.risk_free_rate + spread_for(rating, macro)
        interest = debt * kd
        cov = ebit / interest if interest > 0 else None
        new = shift_rating(synthetic_rating(cov), offset)
        if new == rating:
            break
        rating = new
    kd = macro.risk_free_rate + spread_for(rating, macro)
    interest = debt * kd
    return rating, kd, (ebit / interest if interest > 0 else None)


def wacc_curve(v: CompanyValuation, macro: MacroSnapshot, step: float = 0.05, max_w: float = 0.7) -> tuple[list[WaccPoint], float]:
    """WACC across debt/capital ratios holding enterprise value fixed (Damodaran method).

    Beta is unlevered with Hamada at the current market D/E, then relevered at each point;
    the tax shield is capped when interest exceeds EBIT. Ratings along the curve are
    synthetic coverage ratings shifted by the calibration offset.
    """
    f, m = v.fundamentals, v.market
    offset = rating_calibration_offset(v, macro)
    t = macro.marginal_tax_rate
    E, D = m.market_cap, f.total_debt
    V = E + D
    beta_u = m.beta / (1 + (1 - t) * (D / E))
    ebit = f.operating_income
    pts: list[WaccPoint] = []
    w = 0.0
    while w <= max_w + 1e-9:
        debt = w * V
        if debt > 0:
            rating, kd, cov = _cost_of_debt_fixed_point(ebit, debt, macro, "AAA", offset)
        else:
            rating = shift_rating("AAA", offset)
            kd, cov = macro.risk_free_rate + spread_for(rating, macro), None
        interest = debt * kd
        t_eff = t * min(1.0, ebit / interest) if interest > 0 and ebit > 0 else (t if interest == 0 else 0.0)
        beta_l = beta_u * (1 + (1 - t_eff) * (w / (1 - w))) if w < 1 else float("inf")
        ke = macro.risk_free_rate + beta_l * macro.equity_risk_premium
        wacc = (1 - w) * ke + w * kd * (1 - t_eff)
        pts.append(WaccPoint(debt_to_capital=round(w, 4), implied_rating=rating, pre_tax_cost_of_debt=kd, levered_beta=beta_l,
                             cost_of_equity=ke, wacc=wacc, interest_coverage=cov))
        w += step
    return pts, beta_u


def capital_structure(v: CompanyValuation, macro: MacroSnapshot, peer_nd_ebitda_median: float | None, rating_floor: str) -> CapitalStructureAnalysis:
    f, m = v.fundamentals, v.market
    pts, beta_u = wacc_curve(v, macro)
    E, D = m.market_cap, f.total_debt
    w_cur = D / (D + E)
    t = macro.marginal_tax_rate
    cur_rating = m.credit_rating or synthetic_rating(f.interest_coverage)
    kd_cur = macro.risk_free_rate + spread_for(m.credit_rating or cur_rating, macro)
    ke_cur = macro.risk_free_rate + m.beta * macro.equity_risk_premium
    wacc_cur = (1 - w_cur) * ke_cur + w_cur * kd_cur * (1 - t)
    # Value index: perpetuity value relative to the all-equity firm, net of expected
    # distress costs. Maximising it (subject to the rating floor) gives the optimum.
    w0 = pts[0].wacc

    def value_index(p: WaccPoint) -> float:
        return w0 / p.wacc - CUMULATIVE_PD.get(p.implied_rating, 0.75) * DISTRESS_COST_OF_VALUE

    feasible = [p for p in pts if notch(p.implied_rating) <= notch(rating_floor)]
    best = max(feasible or pts[:1], key=value_index)
    V = D + E
    capacity = max(0.0, (best.debt_to_capital - w_cur) * V)
    target_nd = None
    if f.ebitda > 0:
        target_nd = (best.debt_to_capital * V - f.cash_and_equivalents) / f.ebitda
    gap = best.debt_to_capital - w_cur
    rec = "add_debt" if gap > 0.05 else ("delever" if gap < -0.05 else "hold")
    return CapitalStructureAnalysis(
        current_debt_to_capital=w_cur, current_wacc=wacc_cur, current_implied_rating=cur_rating, actual_rating=m.credit_rating,
        optimal_debt_to_capital=best.debt_to_capital, optimal_wacc=best.wacc, optimal_rating=best.implied_rating, wacc_curve=pts,
        peer_net_debt_to_ebitda_median=peer_nd_ebitda_median, target_net_debt_to_ebitda=target_nd, incremental_debt_capacity_usd=capacity,
        rating_floor=rating_floor, recommendation=rec,
    )


def capital_return(
    v: CompanyValuation,
    macro: MacroSnapshot,
    ev_ebitda_peer_median: float | None,
    ev_ebitda_z: float | None,
    insider: InsiderSignal,
    consensus: ConsensusEstimate | None,
    cs: CapitalStructureAnalysis,
    last_earnings_date: date | None,
    as_of: date,
    long_term_growth: float = 0.02,
) -> CapitalReturnAnalysis:
    f, m = v.fundamentals, v.market
    ke = macro.risk_free_rate + m.beta * macro.equity_risk_premium
    discount = None
    if ev_ebitda_peer_median and v.ev_to_ebitda:
        discount = (ev_ebitda_peer_median - v.ev_to_ebitda) / ev_ebitda_peer_median
    # Implied return on repurchased stock (Gordon): FCF yield, haircut by the consensus
    # next-year EPS revision so a falling-earnings year doesn't flatter the yield, plus a
    # long-run nominal growth assumption. Next-year EPS change is NOT used as perpetual growth.
    revision = 1.0
    if consensus and f.diluted_eps and f.diluted_eps > 0:
        revision = max(0.5, min(1.2, consensus.eps_mean / f.diluted_eps))
    norm_fcf = v.fcf_yield * revision if v.fcf_yield is not None else None
    irr = (norm_fcf + long_term_growth) if norm_fcf is not None else None
    min_cash = 0.02 * f.revenue
    excess_cash = max(0.0, f.cash_and_equivalents - min_cash)
    fcf_after_div = max(0.0, f.free_cash_flow - f.dividends_paid)
    annual_capacity = fcf_after_div + cs.incremental_debt_capacity_usd / 2.0 + excess_cash / 2.0

    # Attractiveness of repurchases: valuation gap, IRR spread over cost of equity, insider conviction.
    irr_spread = (irr - ke) if irr is not None else 0.0
    a = 2.5 * (discount or 0.0) + 8.0 * irr_spread + 0.8 * insider.score
    buyback_w = 1 / (1 + math.exp(-a))
    special_w = 0.0
    if (discount is not None and discount < 0.0) and excess_cash > 0.05 * m.market_cap:
        special_w = min(0.5, excess_cash / m.market_cap)  # fairly/over-valued and cash-rich: return via special dividend
    payout = f.dividends_paid / f.net_income if f.net_income > 0 else 1.0
    regular_growth_w = 0.15 if payout < 0.45 else 0.05
    tot = buyback_w + special_w + regular_growth_w
    mix = {
        "buyback": round(annual_capacity * buyback_w / tot, -6),
        "special_dividend": round(annual_capacity * special_w / tot, -6),
        "regular_dividend_growth": round(annual_capacity * regular_growth_w / tot, -6),
    }
    if insider.cluster_buy and (discount or 0) > 0.15:
        execution = ("Front-load: execute ~40% of the year-one buyback via an accelerated share repurchase (ASR) to capture the "
                     "current discount, and run the balance through a 10b5-1 grid that buys more below the 52-week VWAP.")
    elif buyback_w > 0.6:
        execution = "Programmatic 10b5-1 repurchases with a price-sensitive grid; reassess quarterly against intrinsic value."
    else:
        execution = "Hold repurchases to offsetting dilution; prioritise balance-sheet capacity until valuation or cash generation improves."
    if last_earnings_date:
        nxt = last_earnings_date + timedelta(days=91)
        blackout = (f"Next earnings expected around {nxt.isoformat()}; open-market purchases outside a 10b5-1 plan should stop "
                    f"~{(nxt - timedelta(days=28)).isoformat()} (quarter-end blackout) and resume 2 trading days after the release.")
    else:
        blackout = "Adopt or amend a 10b5-1 plan while in an open window so repurchases can continue through blackout periods."
    return CapitalReturnAnalysis(
        valuation_discount_ev_ebitda=discount, valuation_z_ev_ebitda=ev_ebitda_z, fcf_yield=v.fcf_yield, normalized_fcf_yield=norm_fcf,
        long_term_growth=long_term_growth, cost_of_equity=ke,
        buyback_irr_proxy=irr, insider_signal=insider, excess_cash_usd=excess_cash, annual_capacity_usd=annual_capacity,
        recommended_mix=mix, execution=execution, blackout_note=blackout,
    )
