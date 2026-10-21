"""Render a StrategicAdvisoryReport to Markdown (board-pack style) or JSON."""

from __future__ import annotations

from ..schemas.advisory import StrategicAdvisoryReport


def _p(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def _n(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{d}f}"


def _bn(x: float | None) -> str:
    return "n/a" if x is None else f"${x / 1e9:,.2f}bn"


AREA_LABELS = {"capital_structure": "Capital structure", "capital_return": "Capital return", "investor_relations": "Investor relations",
               "activist_defense": "Activist defense", "m_and_a": "M&A", "valuation": "Valuation"}

PCT_METRICS = {"fcf_yield", "dividend_yield", "shareholder_yield", "operating_margin", "roic", "tsr_1y", "tsr_3y", "debt_to_capital_market"}


def to_markdown(r: StrategicAdvisoryReport) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Strategic Advisory Report: {r.target_name} ({r.target_ticker})")
    a(f"*As of {r.as_of} · peers: {', '.join(r.peers)} · narrative engine: `{r.narrative_engine}` · report {r.report_id}*\n")
    a(f"> **Data notice.** {r.data_disclaimer}\n")
    a("## Executive summary\n")
    a(r.executive_summary + "\n")
    a("## Key findings\n")
    for f in r.key_findings:
        a(f"- {f}")
    a("\n## Recommendations\n")
    a("| # | Area | Recommendation | Priority | Horizon | Confidence |")
    a("|---|---|---|---|---|---|")
    for x in r.recommendations:
        a(f"| {x.rec_id} | {AREA_LABELS.get(x.area.value, x.area.value)} | {x.title} | {x.priority.value} | {x.horizon.value} | {x.confidence:.0%} |")
    for x in r.recommendations:
        a(f"\n### {x.rec_id}. {x.title}\n")
        a(f"**Action.** {x.action}\n")
        a(f"**Why.** {x.rationale}\n")
        if x.quantitative_basis:
            a("**Quantitative basis:** " + "; ".join(f"{k} = {_n(v, 4) if isinstance(v, float) else v}" for k, v in x.quantitative_basis.items()) + "\n")
        if x.kpis_to_track:
            a("**KPIs:** " + ", ".join(x.kpis_to_track) + "  ")
        if x.risks:
            a("**Risks:** " + "; ".join(x.risks))

    a("\n## Peer benchmark\n")
    peers = r.benchmark.peers
    a("| Metric | " + r.target_ticker + " | " + " | ".join(peers) + " | Peer median | Robust z |")
    a("|---|" + "---|" * (len(peers) + 3))
    for s in r.benchmark.statistics:
        fmt = (lambda v: _p(v)) if s.metric in PCT_METRICS else (lambda v: _n(v))
        a(f"| {s.metric} | **{fmt(s.target_value)}** | " + " | ".join(fmt(s.peer_values.get(p)) for p in peers) + f" | {fmt(s.peer_median)} | {_n(s.z_score)} |")

    a("\n## Ownership and capital flows\n")
    a("| Company | Holders | Inst. % out | HHI | Top-10 share | Activist % | Index % |")
    a("|---|---|---|---|---|---|---|")
    for t, o in r.ownership.items():
        a(f"| {t} | {o.holders} | {_p(o.institutional_pct_outstanding)} | {o.hhi:,.0f} | {_p(o.top10_pct_institutional)} | {_p(o.activist_pct_outstanding, 2)} | {_p(o.index_pct_outstanding)} |")
    if r.capital_rotation:
        a(f"\n**Capital rotation out of {r.target_ticker}** (price-neutral value moved):\n")
        a("| Fund | Type | To | Window | Rotation | Intensity | Risk-disclosure gaps (target > peer) |")
        a("|---|---|---|---|---|---|---|")
        for c in r.capital_rotation:
            gaps = ", ".join(f"{d.theme}{' (new)' if d.new_in_source else ''} +{d.gap:.2f}" for d in c.risk_discrepancies[:3])
            a(f"| {c.filer_name} | {c.institution_type} | {c.destination_ticker} | {c.period_from}→{c.period_to} | {_bn(c.rotation_usd)} | {_p(c.rotation_intensity, 0)} | {gaps} |")
    flows = r.grounding.get("net_flows_by_institution_type_usd")
    if flows:
        a("\nNet price-neutral flows into the target by institution type: " + "; ".join(flows) + ".")
    if r.common_ownership:
        a("\n**Common active ownership (holder-vector cosine, index funds excluded):** " + "; ".join(f"{c.company_a}–{c.company_b} {c.overlap_cosine:.2f}" for c in r.common_ownership[:6]))

    a("\n## Activist vulnerability\n")
    a("| Company | Score | Band | Drivers | Activists on register |")
    a("|---|---|---|---|---|")
    for t, v in sorted(r.activist_vulnerability.items(), key=lambda kv: -kv[1].score):
        a(f"| {t} | {v.score:.1f} | {v.band} | {', '.join(v.narrative_drivers) or '-'} | {', '.join(v.activists_present) or '-'} |")

    cs = r.capital_structure
    a("\n## Capital structure\n")
    a(f"Current debt/capital {_p(cs.current_debt_to_capital)} (rating {cs.actual_rating or cs.current_implied_rating}), WACC {_p(cs.current_wacc, 2)}. "
      f"Rating floor from management commitments: **{cs.rating_floor}**. Constrained optimum {_p(cs.optimal_debt_to_capital)} ({cs.optimal_rating}) at WACC "
      f"{_p(cs.optimal_wacc, 2)}; incremental capacity {_bn(cs.incremental_debt_capacity_usd)}. Recommendation: **{cs.recommendation.replace('_', ' ')}**.\n")
    a("| Debt/capital | Implied rating | Kd (pre-tax) | Levered beta | Ke | WACC |")
    a("|---|---|---|---|---|---|")
    for p in cs.wacc_curve:
        a(f"| {_p(p.debt_to_capital, 0)} | {p.implied_rating} | {_p(p.pre_tax_cost_of_debt, 2)} | {p.levered_beta:.2f} | {_p(p.cost_of_equity, 2)} | {_p(p.wacc, 2)} |")

    cr = r.capital_return
    a("\n## Capital return\n")
    a(f"EV/EBITDA discount to peers {_p(cr.valuation_discount_ev_ebitda)}; FCF yield {_p(cr.fcf_yield)} ({_p(cr.normalized_fcf_yield)} after the consensus EPS revision); "
      f"implied buyback return {_p(cr.buyback_irr_proxy)} (incl. {_p(cr.long_term_growth)} long-run growth) vs cost of equity {_p(cr.cost_of_equity)}. Insider signal: **{cr.insider_signal.signal}** (score {cr.insider_signal.score}, "
      f"{cr.insider_signal.open_market_buys} buys by {cr.insider_signal.distinct_buyers} insiders, cluster={cr.insider_signal.cluster_buy}).\n")
    a(f"Annual capacity {_bn(cr.annual_capacity_usd)} → " + ", ".join(f"{k.replace('_', ' ')} {_bn(v)}" for k, v in cr.recommended_mix.items()) + ".\n")
    a(f"*Execution:* {cr.execution}  \n*Timing:* {cr.blackout_note}")

    ir = r.investor_relations
    a("\n## Investor relations plan\n")
    a("| Concern | Analyst Qs | Risk intensity | Δ vs prior 10-K | Rotation pressure | Severity |")
    a("|---|---|---|---|---|---|")
    for c in ir.concerns:
        a(f"| {c.theme} | {c.analyst_question_count} | {c.risk_factor_intensity:.2f} | {c.risk_factor_delta_vs_t4:+.2f} | {_bn(c.holder_pressure_usd)} | {c.severity:.2f} |")
    for tp in ir.talking_points:
        a(f"\n**{tp.theme}.** {tp.message}")
        for pp in tp.proof_points:
            a(f"- proof: {pp}")
        if tp.kpi_commitments:
            a(f"- commit to: {', '.join(tp.kpi_commitments)}")
        if tp.avoid:
            a(f"- avoid: {', '.join(tp.avoid)}")
    a(f"\nHolders to engage: {', '.join(ir.holders_to_engage) or '-'}  \nHolders at risk (rotating out): {', '.join(ir.holders_at_risk) or '-'}")

    if r.supply_chain:
        sc = r.supply_chain
        a("\n## Supply chain\n")
        a(f"Upstream suppliers on file: {', '.join(sc.upstream_suppliers) or '-'}. Shared with peers: "
          + "; ".join(f"{k}: {', '.join(v) or '-'}" for k, v in sc.shared_suppliers_with_peers.items()) + f". {sc.concentration_note}.")

    g = r.grounding
    a("\n## Grounding report\n")
    a(f"Sentences checked: {g.get('sentences')}; factual: {g.get('factual_sentences')}; cited: {g.get('cited_sentences')}; grounding rate: **{g.get('grounding_rate')}**. "
      f"Unknown evidence ids: {g.get('unknown_evidence_ids') or 'none'}; unsupported numbers: {g.get('unsupported_numbers') or 'none'}.")
    if r.llm_trace:
        a("\n<details><summary>Agent trace</summary>\n")
        for t in r.llm_trace:
            a(f"    {t}")
        a("\n</details>")

    a("\n## Citations\n")
    for c in r.citations:
        a(f"- {c.render()}: \"{c.quote[:160].strip()}…\"")
    a("\n## Evidence ledger\n")
    for e in r.evidence:
        if e.kind != "chunk":
            a(f"- **{e.evidence_id}** ({e.kind}): {e.summary} _(derived from: {', '.join(e.derived_from[:4])}{'…' if len(e.derived_from) > 4 else ''})_")
    return "\n".join(L) + "\n"


def to_json(r: StrategicAdvisoryReport) -> str:
    return r.model_dump_json(indent=2)
