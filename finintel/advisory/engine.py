"""Strategic Prescriptive Engine: turns benchmarks, ownership inference and retrieved
evidence into a CFO-level playbook.

Every recommendation is computed deterministically from data first. The LLM layer
(:mod:`finintel.agents`) may rewrite *language*, but quantities come from here and every
sentence must cite evidence ids registered in the :class:`EvidenceLedger`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

from ..analytics import capital as cap
from ..analytics.fundamentals import benchmark, stat
from ..analytics.ownership import common_ownership, concentration, flows, insider_signal, price_neutral_value_change
from ..analytics.risk_themes import analyst_question_themes
from ..graph.builder import FinancialKnowledgeGraph
from ..graph.inference import activist_vulnerability, capital_rotation, net_flows_by_type, supply_chain_exposure
from ..ingestion.sources import CorpusBundle
from ..rag.grounding import EvidenceLedger, GroundingValidator
from ..rag.index import FinancialRAGIndex
from ..schemas.advisory import (
    CapitalReturnAnalysis,
    CapitalStructureAnalysis,
    Horizon,
    InvestorConcern,
    InvestorRelationsPlan,
    PeerBenchmark,
    PlaybookArea,
    Priority,
    Recommendation,
    StrategicAdvisoryReport,
    TalkingPoint,
)
from ..schemas.common import FormType
from ..schemas.graph import ActivistVulnerability, CapitalRotationFinding
from ..schemas.metrics import CompanyValuation, PeerStatistic
from ..schemas.retrieval import RetrievalFilter, RetrievalQuery
from .peers import select_peers

THEME_LABELS = {
    "discretionary_demand": "discretionary demand and traffic",
    "tariffs_trade": "tariff exposure",
    "brand_reputation": "brand and reputation",
    "shrink_theft": "inventory shrink",
    "competition_pricing": "competitive pricing and share loss",
    "inventory_markdowns": "inventory and markdowns",
    "leadership_transition": "CEO transition and strategy",
    "cybersecurity": "cybersecurity",
    "leverage_liquidity": "balance sheet and ratings",
    "labor_costs": "labor costs",
    "membership_model": "membership economics",
    "regulatory_legal": "legal and regulatory",
    "capital_return": "capital return",
}

THEME_QUERIES = {
    "discretionary_demand": "discretionary demand traffic comparable sales decline",
    "tariffs_trade": "tariffs trade policy imported merchandise cost offset",
    "brand_reputation": "brand reputation boycott guest trust",
    "shrink_theft": "inventory shrink theft gross margin",
    "competition_pricing": "competition pricing share gains Walmart Costco",
    "inventory_markdowns": "inventory markdowns clearance",
    "leadership_transition": "chief executive officer succession transition strategy",
    "leverage_liquidity": "credit ratings leverage balance sheet",
    "capital_return": "share repurchase dividend capital allocation priorities",
}

# Concern → what management should say, which KPIs to commit to, and what to avoid.
TALKING_POINT_PLAYBOOK = {
    "discretionary_demand": (
        "Separate cyclical from company-specific traffic: show traffic by category and income cohort, and commit to a dated traffic-recovery plan.",
        ["store traffic by category (quarterly)", "discretionary vs. essentials comp split", "new-item penetration"],
        ["blaming the macro without showing share data"],
    ),
    "tariffs_trade": (
        "Quantify the tariff bridge (gross exposure → vendor share → sourcing shifts → price) instead of folding it into the guidance range.",
        ["tariff cost bridge in bps of gross margin", "% of owned-brand units sourced outside high-tariff countries"],
        ["declining to size the exposure", "implying full pass-through without elasticity evidence"],
    ),
    "brand_reputation": (
        "Acknowledge the guest reaction directly, show the trust-recovery metrics you track, and keep the topic short and data-led.",
        ["guest NPS / trust index trend", "traffic in affected regions vs. company average"],
        ["re-litigating the original decision on the call"],
    ),
    "leadership_transition": (
        "De-risk the CEO handover: name the 2-3 operating priorities, date the investor day, and confirm financial targets carry over.",
        ["investor-day date", "multi-year operating-margin and EPS framework"],
        ["'strategy is intact' without new milestones"],
    ),
    "inventory_markdowns": (
        "Frame inventory as a cash lever: show the DIO gap to peers and the cash released by closing it.",
        ["days inventory outstanding", "markdown rate", "in-stock rate"],
        ["claiming inventory is 'clean' without turns data"],
    ),
    "competition_pricing": (
        "Defend the differentiated assortment with price-gap data versus the largest competitors and owned-brand growth.",
        ["price-gap index vs. top 2 competitors", "owned-brand sales growth"],
        ["generic 'sharpening value' language"],
    ),
    "capital_return": (
        "Give an explicit capital-return framework: dividend commitment, repurchase resumption trigger, leverage target.",
        ["net debt/EBITDA target range", "repurchase authorization usage"],
        ["vague 'over time' repurchase language at a valuation trough"],
    ),
    "shrink_theft": (
        "Show the shrink trajectory and the margin it has returned.",
        ["shrink as % of sales", "gross-margin bridge"],
        ["anecdotes without numbers"],
    ),
}


@dataclass
class AdvisoryContext:
    bundle: CorpusBundle
    index: FinancialRAGIndex
    kg: FinancialKnowledgeGraph
    target_cik: str
    peer_ciks: list[str]
    valuations: dict[str, CompanyValuation]
    stats: list[PeerStatistic]
    ledger: EvidenceLedger
    period_t: date
    period_t1: date
    period_t4: date | None
    rotations: list[CapitalRotationFinding] = field(default_factory=list)
    activist: dict[str, ActivistVulnerability] = field(default_factory=dict)
    capital_structure: CapitalStructureAnalysis | None = None
    capital_return: CapitalReturnAnalysis | None = None
    ir_plan: InvestorRelationsPlan | None = None
    theme_evidence: dict[str, list[str]] = field(default_factory=dict)
    metric_evidence: dict[str, str] = field(default_factory=dict)

    @property
    def target(self) -> CompanyValuation:
        return self.valuations[self.target_cik]

    @property
    def peers(self) -> list[CompanyValuation]:
        return [self.valuations[c] for c in self.peer_ciks]

    def t(self, cik: str) -> str:
        return self.bundle.ticker(cik)


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _x(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}x"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _bn(x: float) -> str:
    return f"${x / 1e9:.2f}bn"


class StrategicPrescriptiveEngine:
    def __init__(self, bundle: CorpusBundle, index: FinancialRAGIndex, kg: FinancialKnowledgeGraph) -> None:
        self.bundle = bundle
        self.index = index
        self.kg = kg

    # ================================================================== context
    def build_context(self, target_ticker: str, peer_tickers: list[str] | None = None) -> AdvisoryContext:
        b = self.bundle
        tcik = b.cik(target_ticker)
        peers = [b.cik(p) for p in peer_tickers] if peer_tickers else select_peers(self.kg, b, tcik)
        if not 3 <= len(peers) <= 5:
            raise ValueError("benchmark against 3-5 peers")
        missing = [b.ticker(c) for c in [tcik, *peers] if c not in b.fundamentals or c not in b.market]
        if missing:
            raise ValueError(f"missing fundamentals/market data for {missing}")
        vals = {c: CompanyValuation(issuer_cik=c, ticker=b.ticker(c), name=b.name(c), fundamentals=b.fundamentals[c], market=b.market[c]) for c in [tcik, *peers]}
        stats = benchmark(vals[tcik], [vals[c] for c in peers], b.as_of)
        periods = [p for p in b.holding_periods() if p <= b.as_of]
        if len(periods) < 2:
            raise ValueError("need at least two 13F periods for flow analysis")
        t, t1 = periods[-1], periods[-2]
        t4 = next((p for p in periods if (t - p).days >= 330), None)
        return AdvisoryContext(bundle=b, index=self.index, kg=self.kg, target_cik=tcik, peer_ciks=peers, valuations=vals, stats=stats,
                               ledger=EvidenceLedger(), period_t=t, period_t1=t1, period_t4=t4)

    # ================================================================== evidence helpers
    def _retrieve(self, ctx: AdvisoryContext, query: str, ciks: list[str] | None = None, top_k: int = 3, items: list[str] | None = None,
                  forms: list[FormType] | None = None, min_authority: float = 0.7) -> list[str]:
        q = RetrievalQuery(text=query, top_k=top_k, filters=RetrievalFilter(
            issuer_ciks=ciks or [ctx.target_cik], items=items, form_types=forms, filed_on_or_before=self.bundle.as_of, min_authority=min_authority))
        return [ctx.ledger.add_chunk(h) for h in self.index.search(q, as_of=self.bundle.as_of)]

    def _fund_sources(self, ctx: AdvisoryContext, ciks: list[str]) -> list[str]:
        return [s for c in ciks for s in (self.bundle.fundamentals[c].source_accession, self.bundle.market[c].provenance.source) if s]

    def _register_metrics(self, ctx: AdvisoryContext) -> None:
        all_ciks = [ctx.target_cik, *ctx.peer_ciks]
        for s in ctx.stats:
            vals: dict[str, float | str | None] = {f"{ctx.target.ticker}": s.target_value, "peer_median": s.peer_median, "z_score": s.z_score}
            vals.update({k: v for k, v in s.peer_values.items()})
            summary = (f"{s.metric}: {ctx.target.ticker} {s.target_value:.4g} vs peer median {s.peer_median:.4g} (robust z {s.z_score})"
                       if s.target_value is not None and s.peer_median is not None else f"{s.metric}: insufficient data")
            ctx.metric_evidence[s.metric] = ctx.ledger.add_fact("metric", summary, vals, self._fund_sources(ctx, all_ciks))

    # ================================================================== analyses
    def analyze(self, ctx: AdvisoryContext) -> AdvisoryContext:
        b = self.bundle
        self._register_metrics(ctx)
        # Ownership flows and rotation.
        ctx.rotations = capital_rotation(self.kg, b, ctx.target_cik, ctx.peer_ciks, ctx.period_t1, ctx.period_t)
        if ctx.period_t4:
            ctx.rotations += [r for r in capital_rotation(self.kg, b, ctx.target_cik, ctx.peer_ciks, ctx.period_t4, ctx.period_t)
                              if r.filer_cik not in {x.filer_cik for x in ctx.rotations}]
        for c in [ctx.target_cik, *ctx.peer_ciks]:
            peers = [ctx.valuations[p] for p in [ctx.target_cik, *ctx.peer_ciks] if p != c]
            ctx.activist[ctx.t(c)] = activist_vulnerability(self.kg, b, ctx.valuations[c], peers, ctx.period_t)
        # Risk-theme evidence (retrieved chunks).
        for theme, q in THEME_QUERIES.items():
            ctx.theme_evidence[theme] = self._retrieve(ctx, q, top_k=2)
        # Capital structure.
        commitments = self._retrieve(ctx, "credit ratings commitment investment grade middle-A balance sheet", top_k=4)
        texts = [ctx.ledger.items[e].summary for e in commitments]
        floor, _src = cap.infer_rating_floor(texts, ctx.target.market.credit_rating)
        ctx.theme_evidence["rating_commitment"] = commitments
        nd_med = stat(ctx.stats, "net_debt_to_ebitda").peer_median
        ctx.capital_structure = cap.capital_structure(ctx.target, b.macro, nd_med, floor)
        # Capital return.
        ins = insider_signal(b.insider_transactions, ctx.target_cik, b.as_of)
        ev = stat(ctx.stats, "ev_to_ebitda")
        last_call = max((f.filed_date for f in b.filings_for(ctx.target_cik, [FormType.TRANSCRIPT, FormType.EIGHT_K]) if f.filed_date <= b.as_of), default=None)
        ctx.capital_return = cap.capital_return(ctx.target, b.macro, ev.peer_median, ev.z_score, ins, b.consensus.get(ctx.target_cik),
                                                ctx.capital_structure, last_call, b.as_of)
        ctx.theme_evidence["buyback_policy"] = self._retrieve(ctx, "share repurchase program remaining authorization did not repurchase", top_k=3)
        ctx.ir_plan = self._investor_relations(ctx)
        return ctx

    def _investor_relations(self, ctx: AdvisoryContext) -> InvestorRelationsPlan:
        b = self.bundle
        q_counts, q_ev = analyst_question_themes(self.index.chunks, ctx.target_cik)
        now = self.kg.risk_intensity(ctx.target_cik, "current")
        prior = self.kg.risk_intensity(ctx.target_cik, "prior")
        # Holder pressure: value sold by rotating funds, attributed to themes where the
        # target's disclosure intensity exceeds the destination's.
        pressure: dict[str, float] = {}
        for r in ctx.rotations:
            for d in r.risk_discrepancies:
                pressure[d.theme] = pressure.get(d.theme, 0.0) + r.rotation_usd
        themes = set(q_counts) | {t for t, (i, _) in now.items() if i > 0}
        themes -= {"cybersecurity", "capital_return"} if not q_counts.get("capital_return") else {"cybersecurity"}
        max_p = max(pressure.values(), default=1.0) or 1.0
        concerns = []
        for t in themes:
            inten = now.get(t, (0.0, []))[0]
            delta = inten - prior.get(t, (0.0, []))[0]
            qc = q_counts.get(t, 0)
            sev = min(1.0, 0.35 * min(qc / 2, 1) + 0.25 * min(inten / 0.25, 1) + 0.15 * max(min(delta / 0.15, 1), 0) + 0.25 * pressure.get(t, 0) / max_p)
            ev_ids = [ctx.ledger.add_chunk(self.index.get(cid)) for cid in q_ev.get(t, [])[:2] if self.index.get(cid)] + ctx.theme_evidence.get(t, [])[:1]
            concerns.append(InvestorConcern(theme=t, analyst_question_count=qc, risk_factor_intensity=round(inten, 4),
                                            risk_factor_delta_vs_t4=round(delta, 4), holder_pressure_usd=pressure.get(t, 0.0), severity=round(sev, 3),
                                            evidence_ids=sorted(set(ev_ids), key=lambda e: int(e[1:]))))
        concerns.sort(key=lambda c: -c.severity)
        if concerns:
            ctx.metric_evidence["concerns"] = ctx.ledger.add_fact(
                "metric",
                "Investor-concern severity model (analyst questions, risk-factor intensity and change vs prior 10-K, rotation pressure): "
                + "; ".join(f"{c.theme} severity {c.severity:.2f}, {c.analyst_question_count} questions" for c in concerns),
                {**{f"{c.theme}_severity": c.severity for c in concerns}, **{f"{c.theme}_questions": float(c.analyst_question_count) for c in concerns}},
                sorted({ctx.ledger.items[e].derived_from[0] for c in concerns for e in c.evidence_ids if ctx.ledger.items[e].derived_from}),
            )
        tps = []
        for c in concerns[:5]:
            msg, kpis, avoid = TALKING_POINT_PLAYBOOK.get(c.theme, ("Address the concern with specific, dated metrics.", ["theme-specific KPI"], ["generic reassurance"]))
            proof = []
            if c.theme == "inventory_markdowns":
                dio = stat(ctx.stats, "dio")
                release = ((dio.target_value or 0) - (dio.peer_median or 0)) / 365 * ctx.target.fundamentals.cogs
                proof.append(f"DIO {dio.target_value:.0f} days vs peer median {dio.peer_median:.0f}; closing half the gap releases ~{_bn(max(release, 0) / 2)} of cash [{ctx.metric_evidence['dio']}]")
            if c.theme in ("competition_pricing", "discretionary_demand"):
                tsr = stat(ctx.stats, "tsr_1y")
                proof.append(f"1-year TSR {_pct(tsr.target_value, 0)} vs peer median {_pct(tsr.peer_median, 0)} [{ctx.metric_evidence['tsr_1y']}]")
            if c.theme == "capital_return":
                proof.append(f"FCF yield {_pct(ctx.capital_return.fcf_yield)} vs peer median {_pct(stat(ctx.stats, 'fcf_yield').peer_median)} [{ctx.metric_evidence['fcf_yield']}]")
            if c.holder_pressure_usd > 0:
                proof.append(f"~{_bn(c.holder_pressure_usd)} of price-neutral selling by funds rotating into peers with lower {THEME_LABELS.get(c.theme, c.theme)} disclosure")
            tps.append(TalkingPoint(theme=c.theme, message=msg, proof_points=proof, kpi_commitments=kpis, avoid=avoid, evidence_ids=c.evidence_ids))
        sellers = sorted({r.filer_name for r in ctx.rotations})
        # Engage: largest holders that are adding or stable, plus any activist.
        fl = [f for f in flows(b.holdings, ctx.period_t1, ctx.period_t) if f.issuer_cik == ctx.target_cik]
        engage = sorted([f for f in fl if f.shares_to > 0 and price_neutral_value_change(f) >= 0], key=lambda f: -f.value_to)
        return InvestorRelationsPlan(concerns=concerns, talking_points=tps, holders_to_engage=[f.filer_name for f in engage[:6]], holders_at_risk=sellers)

    # ================================================================== recommendations
    def recommendations(self, ctx: AdvisoryContext) -> list[Recommendation]:
        b, tv = self.bundle, ctx.target
        cs, cr, ir = ctx.capital_structure, ctx.capital_return, ctx.ir_plan
        assert cs and cr and ir
        me = ctx.metric_evidence
        recs: list[Recommendation] = []
        cs_eid = ctx.ledger.add_fact(
            "metric",
            f"WACC curve: current debt/capital {_pct(cs.current_debt_to_capital)} at WACC {_pct(cs.current_wacc, 2)}; optimum within the {cs.rating_floor} rating floor "
            f"is {_pct(cs.optimal_debt_to_capital)} ({cs.optimal_rating}) at WACC {_pct(cs.optimal_wacc, 2)}; incremental capacity {_bn(cs.incremental_debt_capacity_usd)}.",
            {"current_debt_to_capital": cs.current_debt_to_capital, "current_wacc": cs.current_wacc, "optimal_debt_to_capital": cs.optimal_debt_to_capital,
             "optimal_wacc": cs.optimal_wacc, "incremental_debt_capacity_usd": cs.incremental_debt_capacity_usd, "rating_floor": cs.rating_floor,
             "risk_free_rate": b.macro.risk_free_rate, "peer_net_debt_to_ebitda_median": cs.peer_net_debt_to_ebitda_median,
             "net_debt_to_ebitda": tv.fundamentals.net_debt_to_ebitda},
            self._fund_sources(ctx, [ctx.target_cik]) + [b.macro.provenance.source],
        )
        ctx.metric_evidence["capital_structure"] = cs_eid
        commit_ev = ctx.theme_evidence.get("rating_commitment", [])[:2]
        if cs.recommendation == "add_debt":
            action = (f"Raise up to {_bn(cs.incremental_debt_capacity_usd)} of senior unsecured debt in laddered 5/10/30-year tranches, moving debt/capital from "
                      f"{_pct(cs.current_debt_to_capital)} toward {_pct(cs.optimal_debt_to_capital)} while holding the {cs.rating_floor} floor.")
        elif cs.recommendation == "delever":
            action = f"Reduce debt toward {_pct(cs.optimal_debt_to_capital)} debt/capital to restore headroom above the {cs.rating_floor} floor before adding shareholder returns."
        else:
            action = (f"Hold leverage near {_pct(cs.current_debt_to_capital)} debt/capital: the {cs.rating_floor} rating commitment, not the cost of capital, is the binding "
                      f"constraint (optimum within the floor is {_pct(cs.optimal_debt_to_capital)}). Fund capital returns from free cash flow and working capital, and refinance maturities at the long end.")
        recs.append(Recommendation(
            rec_id="R1", area=PlaybookArea.CAPITAL_STRUCTURE, title="Capital structure: stay within the rating commitment", action=action,
            rationale=(f"Net debt/EBITDA is {_x(tv.fundamentals.net_debt_to_ebitda, 2)} vs a peer median of {_x(cs.peer_net_debt_to_ebitda_median, 2)} [{me['net_debt_to_ebitda']}]; "
                       f"the WACC curve shows {_pct(cs.current_wacc, 2)} today vs {_pct(cs.optimal_wacc, 2)} at the constrained optimum [{cs_eid}]. "
                       f"Management has tied leverage to its ratings commitment [{', '.join(commit_ev) or cs_eid}]."),
            quantitative_basis={"current_debt_to_capital": cs.current_debt_to_capital, "optimal_debt_to_capital": cs.optimal_debt_to_capital,
                                "current_wacc": cs.current_wacc, "optimal_wacc": cs.optimal_wacc, "incremental_debt_capacity_usd": cs.incremental_debt_capacity_usd,
                                "rating_floor": cs.rating_floor},
            evidence_ids=[me["net_debt_to_ebitda"], cs_eid, *commit_ev], priority=Priority.MEDIUM, horizon=Horizon.NEAR, confidence=0.7,
            kpis_to_track=["net debt / EBITDA", "EBIT / interest", "agency outlook"],
            risks=["EBIT decline compresses coverage faster than modeled", "agency lease adjustments differ from the calibration offset"],
        ))

        ins = cr.insider_signal
        cr_eid = ctx.ledger.add_fact(
            "insider",
            f"Insider activity ({ins.window_days}d): {ins.open_market_buys} open-market buys by {ins.distinct_buyers} insiders "
            f"(${ins.buy_value_usd / 1e6:.2f}m), {ins.open_market_sells} sells; cluster buy={ins.cluster_buy}; signal {ins.signal} ({ins.score}).",
            {"buys": ins.open_market_buys, "distinct_buyers": ins.distinct_buyers, "buy_value_usd": ins.buy_value_usd, "sell_value_usd": ins.sell_value_usd,
             "cluster_buy": str(ins.cluster_buy), "score": ins.score},
            [t.accession.value for t in b.insider_transactions if t.issuer_cik == ctx.target_cik],
        )
        ret_eid = ctx.ledger.add_fact(
            "metric",
            f"Capital return model: EV/EBITDA discount {_pct(cr.valuation_discount_ev_ebitda)} vs peers, FCF yield {_pct(cr.fcf_yield)} "
            f"({_pct(cr.normalized_fcf_yield)} after the consensus EPS revision), implied buyback return {_pct(cr.buyback_irr_proxy)} "
            f"(with {_pct(cr.long_term_growth)} long-run growth) vs cost of equity {_pct(cr.cost_of_equity)}; annual capacity {_bn(cr.annual_capacity_usd)}; "
            f"mix buyback {_bn(cr.recommended_mix['buyback'])}, special dividend {_bn(cr.recommended_mix['special_dividend'])}, dividend growth {_bn(cr.recommended_mix['regular_dividend_growth'])}.",
            {"valuation_discount": cr.valuation_discount_ev_ebitda, "fcf_yield": cr.fcf_yield, "normalized_fcf_yield": cr.normalized_fcf_yield,
             "long_term_growth": cr.long_term_growth, "buyback_irr_proxy": cr.buyback_irr_proxy,
             "cost_of_equity": cr.cost_of_equity, "annual_capacity_usd": cr.annual_capacity_usd, **{f"mix_{k}": v for k, v in cr.recommended_mix.items()}},
            self._fund_sources(ctx, [ctx.target_cik, *ctx.peer_ciks]),
        )
        ctx.metric_evidence["capital_return"] = ret_eid
        ctx.metric_evidence["insider"] = cr_eid
        bb_ev = ctx.theme_evidence.get("buyback_policy", [])[:2]
        recs.append(Recommendation(
            rec_id="R2", area=PlaybookArea.CAPITAL_RETURN,
            title="Resume repurchases now; no special dividend" if cr.recommended_mix["special_dividend"] == 0 else "Pair repurchases with a special dividend",
            action=(f"Restart the buyback at ~{_bn(cr.recommended_mix['buyback'])} per year from existing authorization. {cr.execution} "
                    f"Keep the regular dividend; earmark ~{_bn(cr.recommended_mix['regular_dividend_growth'])} for dividend growth. {cr.blackout_note}"),
            rationale=(f"Shares trade at a {_pct(cr.valuation_discount_ev_ebitda, 0)} EV/EBITDA discount to peers [{me['ev_to_ebitda']}, {ret_eid}]. Even after haircutting the "
                       f"{_pct(cr.fcf_yield)} FCF yield to {_pct(cr.normalized_fcf_yield)} for the consensus EPS decline, the implied return on repurchased stock is "
                       f"{_pct(cr.buyback_irr_proxy)} vs a {_pct(cr.cost_of_equity)} cost of equity [{ret_eid}]. Insiders are buying in the open market ({ins.distinct_buyers} buyers, cluster={ins.cluster_buy}) [{cr_eid}]. "
                       f"Repurchases were paused while authorization remains [{', '.join(bb_ev) or ret_eid}]."),
            quantitative_basis={"valuation_discount": cr.valuation_discount_ev_ebitda, "fcf_yield": cr.fcf_yield, "buyback_irr_proxy": cr.buyback_irr_proxy, "cost_of_equity": cr.cost_of_equity,
                                "annual_capacity_usd": cr.annual_capacity_usd, "buyback_usd": cr.recommended_mix["buyback"], "insider_score": ins.score},
            evidence_ids=[me["ev_to_ebitda"], ret_eid, cr_eid, *bb_ev],
            priority=Priority.HIGH if (cr.valuation_discount_ev_ebitda or 0) > 0.2 else Priority.MEDIUM, horizon=Horizon.IMMEDIATE, confidence=0.75,
            kpis_to_track=["shares repurchased vs. plan", "average repurchase price vs. intrinsic value", "net leverage after returns"],
            risks=["buying ahead of further estimate cuts", "repurchases crowding out turnaround capex"],
        ))

        top = ir.concerns[:3]
        if top:
            ir_ev = sorted({e for c in top for e in c.evidence_ids}, key=lambda e: int(e[1:]))[:6]
            recs.append(Recommendation(
                rec_id="R3", area=PlaybookArea.INVESTOR_RELATIONS, title="Narrative defense on the next earnings call",
                action=" ".join(f"({i + 1}) {_cap(THEME_LABELS.get(tp.theme, tp.theme))}: {tp.message}" for i, tp in enumerate(ir.talking_points[:3])),
                rationale=("The most persistent investor concerns are " + ", ".join(f"{THEME_LABELS.get(c.theme, c.theme)} ({c.analyst_question_count} analyst question{'s' if c.analyst_question_count != 1 else ''})" for c in top)
                           + f" [{me['concerns']}], and they line up with themes where the company's risk disclosure exceeds that of the peers funds are rotating into [{', '.join(ir_ev)}]."),
                quantitative_basis={c.theme: c.severity for c in top},
                evidence_ids=[me["concerns"], *ir_ev], priority=Priority.HIGH, horizon=Horizon.IMMEDIATE, confidence=0.7,
                kpis_to_track=sorted({k for tp in ir.talking_points[:3] for k in tp.kpi_commitments}),
                risks=["over-committing to KPIs that are volatile quarter to quarter"],
            ))

        av = ctx.activist[tv.ticker]
        av_eid = ctx.ledger.add_fact(
            "graph",
            f"Activist vulnerability {tv.ticker}: {av.score}/100 ({av.band}); drivers: {', '.join(av.narrative_drivers)}; activists on register: {', '.join(av.activists_present) or 'none'}. "
            + "Peer scores: " + ", ".join(f"{k} {v.score}" for k, v in ctx.activist.items() if k != tv.ticker) + ".",
            {"score": av.score, **{f"c_{k}": v for k, v in av.components.items()}, **{f"peer_{k}": v.score for k, v in ctx.activist.items() if k != tv.ticker}},
            sorted({h.accession.value for h in b.holdings if h.issuer_cik == ctx.target_cik and h.report_period == ctx.period_t}),
        )
        ctx.metric_evidence["activist"] = av_eid
        rot_eid = None
        if ctx.rotations:
            total = sum(r.rotation_usd for r in ctx.rotations)
            rot_eid = ctx.ledger.add_fact(
                "ownership",
                f"Capital rotation out of {tv.ticker}: {len({r.filer_cik for r in ctx.rotations})} funds rotated ~{_bn(total)} into "
                + ", ".join(sorted({r.destination_ticker for r in ctx.rotations})) + "; largest: "
                + "; ".join(f"{r.filer_name} → {r.destination_ticker} {_bn(r.rotation_usd)}" for r in ctx.rotations[:3]) + ".",
                {"total_rotation_usd": total, **{f"{r.filer_name[:24]}->{r.destination_ticker}": r.rotation_usd for r in ctx.rotations[:6]}},
                sorted({h.accession.value for h in b.holdings if h.issuer_cik in (ctx.target_cik, *ctx.peer_ciks) and h.report_period in (ctx.period_t, ctx.period_t1)})[:12],
            )
            ctx.metric_evidence["rotation"] = rot_eid
        if av.band in ("elevated", "high"):
            recs.append(Recommendation(
                rec_id="R4", area=PlaybookArea.ACTIVIST_DEFENSE, title="Get ahead of an activist campaign",
                action=("Run a pre-emptive 'activist audit' this quarter: (a) meet the top holders adding to positions and any activist on the register before the proxy "
                        "nomination window; (b) announce a dated investor day with a multi-year margin and capital-return framework; (c) review board skills for retail "
                        "operations and capital allocation; (d) prepare a fact base on the valuation gap and the turnaround milestones."),
                rationale=(f"The activist vulnerability score is {av.score}/100 ({av.band}), driven by {', '.join(av.narrative_drivers)} [{av_eid}]"
                           + (f"; funds have been rotating out of the stock into peers [{rot_eid}]" if rot_eid else "") + "."),
                quantitative_basis={"score": av.score, **av.components},
                evidence_ids=[av_eid] + ([rot_eid] if rot_eid else []), priority=Priority.CRITICAL if av.band == "high" else Priority.HIGH,
                horizon=Horizon.IMMEDIATE, confidence=0.65, kpis_to_track=["activist % of shares outstanding", "say-on-pay support", "13D/13G filings"],
                risks=["defensive posture read as entrenchment if not paired with targets"],
            ))

        dio = stat(ctx.stats, "dio")
        if dio.target_value and dio.peer_median and dio.target_value > dio.peer_median:
            release = (dio.target_value - dio.peer_median) / 365 * tv.fundamentals.cogs
            recs.append(Recommendation(
                rec_id="R5", area=PlaybookArea.VALUATION, title="Release working capital to fund returns",
                action=(f"Set a 24-month target to cut days inventory outstanding by {0.5 * (dio.target_value - dio.peer_median):.0f} days (half the gap to peers), releasing "
                        f"~{_bn(release / 2)} of cash, and report DIO quarterly."),
                rationale=f"DIO is {dio.target_value:.0f} days vs a peer median of {dio.peer_median:.0f} [{me['dio']}].",
                quantitative_basis={"dio": dio.target_value, "peer_median_dio": dio.peer_median, "cash_release_half_gap_usd": release / 2},
                evidence_ids=[me["dio"]], priority=Priority.MEDIUM, horizon=Horizon.NEAR, confidence=0.6,
                kpis_to_track=["DIO", "in-stock rate", "markdown rate"], risks=["under-assortment in discretionary categories during holiday"],
            ))

        news_ev = self._retrieve(ctx, "debt-funded acquisition rumor dividend cut", top_k=1, forms=[FormType.NEWS], min_authority=0.0)
        recs.append(Recommendation(
            rec_id="R6", area=PlaybookArea.M_AND_A, title="M&A posture: no large deals at a trough multiple",
            action=("Avoid stock-funded or large debt-funded acquisitions while the shares trade at a discount; limit M&A to bolt-ons that accelerate digital, "
                    "advertising or supply-chain capability and fit inside the rating floor. Publicly rebut unsupported market chatter with the stated capital-allocation order."),
            rationale=(f"Using undervalued stock as currency transfers value to sellers [{me['ev_to_ebitda']}], and leverage headroom within the rating floor is limited [{cs_eid}]. "
                       f"Low-authority market chatter about a debt-funded deal or a dividend cut conflicts with management's stated priorities [{', '.join(news_ev + commit_ev[:1])}]."),
            quantitative_basis={"ev_to_ebitda": tv.ev_to_ebitda, "incremental_debt_capacity_usd": cs.incremental_debt_capacity_usd},
            evidence_ids=[me["ev_to_ebitda"], cs_eid, *news_ev, *commit_ev[:1]], priority=Priority.MEDIUM, horizon=Horizon.STRATEGIC, confidence=0.6,
            kpis_to_track=["ROIC on acquired capital", "EV/EBITDA vs peers"], risks=["missing a capability that peers acquire"],
        ))
        return recs

    # ================================================================== narrative
    def deterministic_narrative(self, ctx: AdvisoryContext, recs: list[Recommendation]) -> tuple[str, list[str]]:
        tv, me = ctx.target, ctx.metric_evidence
        ev = stat(ctx.stats, "ev_to_ebitda")
        tsr = stat(ctx.stats, "tsr_1y")
        fy = stat(ctx.stats, "fcf_yield")
        av = ctx.activist[tv.ticker]
        cr = ctx.capital_return
        cs = ctx.capital_structure
        assert cr and cs and ctx.ir_plan
        peers = ", ".join(ctx.t(c) for c in ctx.peer_ciks)
        summary = [
            f"{tv.name} trades at {_x(ev.target_value)} EV/EBITDA versus a peer median of {_x(ev.peer_median)} ({peers}) [{me['ev_to_ebitda']}].",
            f"One-year TSR of {_pct(tsr.target_value, 0)} trails the peer median of {_pct(tsr.peer_median, 0)} [{me['tsr_1y']}], while the FCF yield of {_pct(fy.target_value)} is well above peers [{me['fcf_yield']}].",
        ]
        if "rotation" in me:
            summary.append(f"Institutional money is rotating out of the stock and into peers with lighter risk disclosure [{me['rotation']}].")
        summary += [
            f"The activist vulnerability score is {av.score}/100 ({av.band}) [{me['activist']}].",
            f"Insiders are buying in the open market [{me['insider']}], and the capital-return model favors repurchases over a special dividend [{me['capital_return']}].",
            f"Leverage should stay near current levels because the rating commitment is the binding constraint [{me['capital_structure']}].",
            "The priority is to resume buybacks, address the top investor concerns with hard KPIs on the next call, and pre-empt activist pressure with a dated investor day.",
        ]
        findings = [
            f"Valuation: EV/EBITDA {_x(ev.target_value)} vs peer median {_x(ev.peer_median)}; P/E {_x(stat(ctx.stats, 'pe').target_value)} vs {_x(stat(ctx.stats, 'pe').peer_median)} [{me['ev_to_ebitda']}, {me['pe']}].",
            f"Shareholder yield {_pct(stat(ctx.stats, 'shareholder_yield').target_value)} vs peer median {_pct(stat(ctx.stats, 'shareholder_yield').peer_median)}; dividend yield {_pct(tv.market.dividend_yield)} [{me['shareholder_yield']}, {me['dividend_yield']}].",
            f"Working capital: DIO {stat(ctx.stats, 'dio').target_value:.0f} days vs peer median {stat(ctx.stats, 'dio').peer_median:.0f}; cash conversion cycle {stat(ctx.stats, 'cash_conversion_cycle').target_value:.1f} days [{me['dio']}, {me['cash_conversion_cycle']}].",
        ]
        if ctx.rotations:
            r = ctx.rotations[0]
            disc = ", ".join(THEME_LABELS.get(d.theme, d.theme) + (" (new vs prior 10-K)" if d.new_in_source else "") for d in r.risk_discrepancies[:3])
            findings.append(f"Largest rotation: {r.filer_name} moved {_bn(r.rotation_usd)} from {r.source_ticker} into {r.destination_ticker}; the target's risk disclosure is heavier on {disc} [{me['rotation']}].")
        findings.append(f"Activist drivers: {', '.join(av.narrative_drivers)}; activists on the register: {', '.join(av.activists_present) or 'none'} [{me['activist']}].")
        top = ctx.ir_plan.concerns[:3]
        if top:
            findings.append("Top investor concerns: " + "; ".join(f"{THEME_LABELS.get(c.theme, c.theme)} (severity {c.severity:.2f})" for c in top)
                            + f" [{', '.join(sorted({me['concerns'], *(e for c in top for e in c.evidence_ids[:1])}, key=lambda e: int(e[1:])))}].")
        return " ".join(summary), findings

    # ================================================================== assemble
    def assemble(self, ctx: AdvisoryContext, recs: list[Recommendation], summary: str, findings: list[str], engine_name: str, trace: list[str] | None = None) -> StrategicAdvisoryReport:
        b = self.bundle
        validator = GroundingValidator(ctx.ledger)
        text = summary + "\n" + "\n".join(findings) + "\n" + "\n".join(r.rationale for r in recs)
        _, rep = validator.validate(text)
        own = {}
        for c in [ctx.target_cik, *ctx.peer_ciks]:
            own[ctx.t(c)] = concentration(b.holdings, c, ctx.period_t, b.market[c].shares_outstanding, b.institutions)
        co = common_ownership(b.holdings, ctx.period_t, [ctx.target_cik, *ctx.peer_ciks], {c: ctx.t(c) for c in [ctx.target_cik, *ctx.peer_ciks]},
                              {c: b.market[c].shares_outstanding for c in [ctx.target_cik, *ctx.peer_ciks]}, b.institutions)
        used = {e for r in recs for e in r.evidence_ids} | {e for txt in [summary, *findings] for e in _ids(txt)}
        for tp in (ctx.ir_plan.talking_points if ctx.ir_plan else []):
            used |= set(tp.evidence_ids)
        cits = [c for c in ctx.ledger.citations() if c.citation_id in used]
        flows_note = net_flows_by_type(b, ctx.target_cik, ctx.period_t4 or ctx.period_t1, ctx.period_t)
        disclaimer = (
            "Offline fixture data: issuer identifiers are real, fundamentals and prices are approximations of public figures, and all institutions, 13F positions, "
            "insider trades, filing excerpts and transcripts are synthetic (accessions prefixed FIXTURE-). Not investment advice; refresh with --live before use."
            if b.is_synthetic else
            "Built from SEC EDGAR, Stooq and FRED data as of the stated date. Verify figures against source filings before acting. Not investment advice."
        )
        return StrategicAdvisoryReport(
            report_id=uuid.uuid4().hex[:12], generated_at=date.today(), as_of=b.as_of, target_ticker=ctx.target.ticker, target_name=ctx.target.name,
            peers=[ctx.t(c) for c in ctx.peer_ciks], executive_summary=summary, key_findings=findings,
            benchmark=PeerBenchmark(target_ticker=ctx.target.ticker, peers=[ctx.t(c) for c in ctx.peer_ciks], statistics=ctx.stats),
            ownership=own, capital_rotation=ctx.rotations, activist_vulnerability=ctx.activist, common_ownership=co,
            supply_chain=supply_chain_exposure(self.kg, b, ctx.target_cik, ctx.peer_ciks),
            capital_structure=ctx.capital_structure, capital_return=ctx.capital_return, investor_relations=ctx.ir_plan,  # type: ignore[arg-type]
            recommendations=recs, evidence=list(ctx.ledger.items.values()), citations=cits,
            grounding={**rep.as_dict(), "net_flows_by_institution_type_usd": [f"{k}: {v / 1e6:,.0f}m" for k, v in sorted(flows_note.items())]},
            narrative_engine=engine_name, data_disclaimer=disclaimer, llm_trace=trace or [],
        )

    def run(self, target_ticker: str, peer_tickers: list[str] | None = None) -> StrategicAdvisoryReport:
        ctx = self.analyze(self.build_context(target_ticker, peer_tickers))
        recs = self.recommendations(ctx)
        summary, findings = self.deterministic_narrative(ctx, recs)
        return self.assemble(ctx, recs, summary, findings, "deterministic")


def _ids(text: str) -> set[str]:
    import re

    return {i.strip() for grp in re.findall(r"\[(E\d+(?:\s*,\s*E\d+)*)\]", text) for i in grp.split(",")}
