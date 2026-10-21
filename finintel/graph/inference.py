"""Cross-company inference over the knowledge graph.

* :func:`capital_rotation`: "Which funds are rotating capital from A into peer B, and
  which risk-factor discrepancies line up with the move?" Flows are measured
  *price-neutrally* (share change × end price) so a falling stock isn't mistaken for
  selling.
* :func:`activist_vulnerability`: composite 0–100 score from peer-relative valuation,
  TSR, cash-conversion-cycle and inventory gaps, dividend-yield/payout gaps, balance-sheet
  slack, margin gap, register composition and leadership-transition signals.
* :func:`supply_chain_exposure`: shared-supplier overlap and concentration.
"""

from __future__ import annotations

import statistics
from datetime import date

from ..analytics.ownership import flows, price_neutral_value_change
from ..ingestion.sources import CorpusBundle
from ..schemas.common import InstitutionType
from ..schemas.graph import ActivistVulnerability, CapitalRotationFinding, RiskDiscrepancy, SupplyChainExposure
from ..schemas.metrics import CompanyValuation
from .builder import FinancialKnowledgeGraph


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _median(xs: list[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return statistics.median(v) if v else None


def risk_discrepancies(kg: FinancialKnowledgeGraph, source_cik: str, dest_cik: str, min_gap: float = 0.05) -> list[RiskDiscrepancy]:
    src_now = kg.risk_intensity(source_cik, "current")
    src_prior = kg.risk_intensity(source_cik, "prior")
    dst_now = kg.risk_intensity(dest_cik, "current")
    out = []
    for theme in set(src_now) | set(dst_now):
        s, s_ev = src_now.get(theme, (0.0, []))
        d, d_ev = dst_now.get(theme, (0.0, []))
        gap = s - d
        if gap >= min_gap:
            new = theme not in src_prior and s > 0 and bool(src_prior)
            out.append(RiskDiscrepancy(theme=theme, source_intensity=round(s, 4), destination_intensity=round(d, 4), gap=round(gap, 4),
                                       new_in_source=new, evidence_chunk_ids=s_ev[:3] + d_ev[:1]))
    return sorted(out, key=lambda r: (-int(r.new_in_source), -r.gap))


def capital_rotation(
    kg: FinancialKnowledgeGraph,
    bundle: CorpusBundle,
    source_cik: str,
    dest_ciks: list[str],
    period_from: date,
    period_to: date,
    min_rotation_usd: float = 25e6,
) -> list[CapitalRotationFinding]:
    fl = flows(bundle.holdings, period_from, period_to)
    by_filer: dict[str, dict[str, float]] = {}
    prior_val: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    for f in fl:
        if f.issuer_cik not in (source_cik, *dest_ciks):
            continue
        by_filer.setdefault(f.filer_cik, {})[f.issuer_cik] = price_neutral_value_change(f)
        prior_val.setdefault(f.filer_cik, {})[f.issuer_cik] = f.value_from
        names[f.filer_cik] = f.filer_name
    disc_cache = {d: risk_discrepancies(kg, source_cik, d) for d in dest_ciks}
    out: list[CapitalRotationFinding] = []
    for filer, changes in by_filer.items():
        outflow = changes.get(source_cik, 0.0)
        if outflow >= 0:
            continue
        total_in = sum(max(changes.get(d, 0.0), 0.0) for d in dest_ciks)
        for d in dest_ciks:
            inflow = changes.get(d, 0.0)
            if inflow <= 0:
                continue
            # Allocate the outflow pro rata across destinations that received inflows, so
            # one sale is never double-counted as rotation into several peers.
            rot = min(-outflow * inflow / total_in, inflow)
            if rot < min_rotation_usd:
                continue
            base = prior_val[filer].get(source_cik, 0.0) + prior_val[filer].get(d, 0.0)
            inst = bundle.institutions.get(filer)
            out.append(CapitalRotationFinding(
                filer_cik=filer, filer_name=names[filer], institution_type=inst.institution_type.value if inst else "other",
                source_cik=source_cik, source_ticker=bundle.ticker(source_cik), destination_cik=d, destination_ticker=bundle.ticker(d),
                period_from=period_from, period_to=period_to, source_value_change_usd=outflow, destination_value_change_usd=inflow,
                rotation_usd=rot, rotation_intensity=rot / base if base else 1.0, risk_discrepancies=disc_cache[d],
            ))
    return sorted(out, key=lambda r: -r.rotation_usd)


def net_flows_by_type(bundle: CorpusBundle, cik: str, period_from: date, period_to: date) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in flows(bundle.holdings, period_from, period_to):
        if f.issuer_cik != cik:
            continue
        inst = bundle.institutions.get(f.filer_cik)
        t = inst.institution_type.value if inst else "other"
        out[t] = out.get(t, 0.0) + price_neutral_value_change(f)
    return out


ACTIVIST_WEIGHTS = {
    "valuation_discount": 0.22,
    "tsr_underperformance": 0.20,
    "working_capital_gap": 0.10,
    "yield_and_payout_gap": 0.08,
    "balance_sheet_slack": 0.10,
    "margin_gap": 0.08,
    "register_vulnerability": 0.14,
    "governance_transition": 0.08,
}


def activist_vulnerability(
    kg: FinancialKnowledgeGraph,
    bundle: CorpusBundle,
    target: CompanyValuation,
    peers: list[CompanyValuation],
    period: date,
) -> ActivistVulnerability:
    t, f, m = target, target.fundamentals, target.market
    peer_ev = _median([p.ev_to_ebitda for p in peers])
    peer_tsr = _median([p.market.return_1y for p in peers])
    peer_ccc = _median([p.fundamentals.cash_conversion_cycle for p in peers])
    peer_dio = _median([p.fundamentals.dio for p in peers])
    peer_yield = _median([p.market.dividend_yield for p in peers])
    peer_sh_yield = _median([p.shareholder_yield for p in peers])
    peer_lev = _median([p.fundamentals.net_debt_to_ebitda for p in peers])
    peer_margin = _median([p.fundamentals.operating_margin for p in peers])

    raw: dict[str, float | None] = {
        "ev_to_ebitda": t.ev_to_ebitda, "peer_ev_to_ebitda_median": peer_ev, "tsr_1y": m.return_1y, "peer_tsr_1y_median": peer_tsr,
        "ccc_days": f.cash_conversion_cycle, "peer_ccc_median": peer_ccc, "dio_days": f.dio, "peer_dio_median": peer_dio,
        "dividend_yield": m.dividend_yield, "peer_dividend_yield_median": peer_yield, "shareholder_yield": t.shareholder_yield,
        "peer_shareholder_yield_median": peer_sh_yield, "net_debt_to_ebitda": f.net_debt_to_ebitda, "peer_net_debt_to_ebitda_median": peer_lev,
        "operating_margin": f.operating_margin, "peer_operating_margin_median": peer_margin,
    }
    comp: dict[str, float] = {}
    comp["valuation_discount"] = _clip01(((peer_ev - t.ev_to_ebitda) / peer_ev) / 0.5) if (peer_ev and t.ev_to_ebitda) else 0.0
    comp["tsr_underperformance"] = _clip01((peer_tsr - m.return_1y) / 0.5) if (peer_tsr is not None and m.return_1y is not None) else 0.0
    ccc_gap = ((f.cash_conversion_cycle or 0) - (peer_ccc or 0)) / 30.0
    dio_gap = ((f.dio or 0) - (peer_dio or 0)) / 30.0
    comp["working_capital_gap"] = _clip01(0.5 * ccc_gap + 0.5 * dio_gap)
    # Yield gap cuts both ways: a stingy payer invites "return the cash" campaigns; a high
    # dividend with *no* buyback at a deep discount invites "buy back stock" campaigns.
    stingy = _clip01(((peer_sh_yield or 0) - (t.shareholder_yield or 0)) / 0.04)
    no_buyback_at_discount = 1.0 if (f.buybacks < 0.01 * m.market_cap and comp["valuation_discount"] > 0.3) else 0.0
    comp["yield_and_payout_gap"] = max(stingy, 0.6 * no_buyback_at_discount)
    comp["balance_sheet_slack"] = _clip01(((peer_lev or 0) - (f.net_debt_to_ebitda or 0)) / 1.5)
    comp["margin_gap"] = _clip01(((peer_margin or 0) - (f.operating_margin or 0)) / 0.03)

    # Register: dispersed ownership, hedge-fund/event-driven presence, activist toe-holds.
    holders = kg.holders_of(t.issuer_cik, period.isoformat())
    so = m.shares_outstanding
    activist_names, hf_pct, act_pct, shares = [], 0.0, 0.0, []
    for node, d in holders.items():
        nd = kg.g.nodes[node]
        pct = d["shares"] / so
        shares.append(d["shares"])
        if nd.get("is_activist") or nd.get("institution_type") == InstitutionType.ACTIVIST.value:
            activist_names.append(nd["name"])
            act_pct += pct
        elif nd.get("institution_type") == InstitutionType.HEDGE_FUND.value:
            hf_pct += pct
    tot = sum(shares)
    hhi = sum((s / tot * 100) ** 2 for s in shares) if tot else 10000
    dispersion = _clip01((2000 - hhi) / 2000)
    comp["register_vulnerability"] = _clip01(0.35 * dispersion + 0.35 * _clip01(act_pct / 0.02) + 0.3 * _clip01(hf_pct / 0.03) + (0.3 if activist_names else 0.0))
    raw.update({"register_hhi": hhi, "activist_pct_outstanding": act_pct, "hedge_fund_pct_outstanding": hf_pct})

    intens = kg.risk_intensity(t.issuer_cik, "current")
    lt = intens.get("leadership_transition", (0.0, []))[0]
    comp["governance_transition"] = _clip01(lt / 0.1)

    score = 100 * sum(ACTIVIST_WEIGHTS[k] * comp[k] for k in ACTIVIST_WEIGHTS)
    band = "high" if score >= 60 else "elevated" if score >= 45 else "moderate" if score >= 30 else "low"
    labels = {
        "valuation_discount": "EV/EBITDA discount to peers", "tsr_underperformance": "1-year TSR lagging peers",
        "working_capital_gap": "inventory days / cash conversion cycle above peers", "yield_and_payout_gap": "capital-return mix gap vs peers",
        "balance_sheet_slack": "unused leverage capacity vs peers", "margin_gap": "operating margin below peers",
        "register_vulnerability": "activist/hedge-fund presence in a dispersed register", "governance_transition": "leadership transition disclosed",
    }
    drivers = [labels[k] for k, _ in sorted(comp.items(), key=lambda kv: -ACTIVIST_WEIGHTS[kv[0]] * kv[1]) if comp[k] > 0.25][:4]
    return ActivistVulnerability(issuer_cik=t.issuer_cik, ticker=t.ticker, score=round(score, 1), band=band,
                                 components={k: round(v, 3) for k, v in comp.items()}, raw_inputs=raw,
                                 activists_present=sorted(activist_names), narrative_drivers=drivers)


def supply_chain_exposure(kg: FinancialKnowledgeGraph, bundle: CorpusBundle, cik: str, peer_ciks: list[str]) -> SupplyChainExposure:
    mine = set(kg.suppliers_of(cik))
    shared = {bundle.ticker(p): sorted(bundle.ticker(s) for s in mine & set(kg.suppliers_of(p))) for p in peer_ciks}
    shares = []
    for s in mine:
        for _, v, d in kg.g.out_edges(f"company:{s}", data=True):
            if d.get("type") == "SUPPLIES":
                shares.append((bundle.ticker(s), kg.g.nodes[v].get("ticker"), d.get("revenue_share_of_supplier") or 0))
    note_parts = []
    for s in sorted(mine):
        cust = sorted([(c, w) for sup, c, w in shares if sup == bundle.ticker(s)], key=lambda x: -x[1])
        if cust:
            top_c, top_w = cust[0]
            note_parts.append(f"{bundle.ticker(s)}'s largest disclosed customer in the set is {top_c} (~{top_w:.0%} of its revenue)")
    return SupplyChainExposure(issuer_cik=cik, ticker=bundle.ticker(cik), upstream_suppliers=sorted(bundle.ticker(s) for s in mine),
                               shared_suppliers_with_peers=shared,
                               concentration_note="; ".join(note_parts) or "no supplier relationships on file")
