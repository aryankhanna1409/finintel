"""Tools the Claude analyst agent can call. Every tool returns JSON whose facts carry
evidence ids from the shared :class:`EvidenceLedger`, so anything the model writes can be
traced back and validated."""

from __future__ import annotations

import json
from typing import Any, Callable

from ..advisory.engine import THEME_LABELS, AdvisoryContext
from ..analytics.fundamentals import stat
from ..schemas.common import FormType
from ..schemas.retrieval import RetrievalFilter, RetrievalQuery


def _schema(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "search_filings",
        "description": (
            "Hybrid (BM25 + dense + rerank) search over SEC filings, earnings-call transcripts, consensus notes and news for the target and its peers. "
            "Returns passages with evidence ids (E#) and citations. Filings outrank transcripts, which outrank news. Use ticker='' to search all companies "
            "and item='' for all sections (common items: '1A' risk factors, '7' MD&A, 'Q&A' analyst questions, 'II-1A' 10-Q risk updates)."
        ),
        "input_schema": _schema({
            "query": {"type": "string", "description": "Natural-language or keyword query"},
            "ticker": {"type": "string", "description": "Restrict to one company ticker, or '' for all"},
            "item": {"type": "string", "description": "Restrict to one filing item, or ''"},
            "top_k": {"type": "integer", "description": "Number of passages, 1-8"},
        }),
        "strict": True,
    },
    {
        "name": "get_peer_benchmark",
        "description": "Target vs. peer statistics (valuation, yields, margins, ROIC, leverage, cash conversion cycle, TSR) with peer median, robust z-score and evidence ids.",
        "input_schema": _schema({}),
        "strict": True,
    },
    {
        "name": "get_ownership_rotation",
        "description": "13F-based capital rotation out of the target into peers (price-neutral), with the risk-factor discrepancies that line up with each move, plus net flows by institution type.",
        "input_schema": _schema({}),
        "strict": True,
    },
    {
        "name": "get_activist_vulnerability",
        "description": "Activist vulnerability score (0-100) and its components for a ticker in the peer set.",
        "input_schema": _schema({"ticker": {"type": "string"}}),
        "strict": True,
    },
    {
        "name": "get_capital_structure",
        "description": "WACC curve, rating floor inferred from management commitments, optimal debt/capital and incremental debt capacity.",
        "input_schema": _schema({}),
        "strict": True,
    },
    {
        "name": "get_capital_return",
        "description": "Buyback vs. special dividend economics: valuation discount, FCF yield vs cost of equity, insider signal, capacity, recommended mix and execution.",
        "input_schema": _schema({}),
        "strict": True,
    },
    {
        "name": "get_investor_concerns",
        "description": "Ranked investor concerns (analyst questions, risk-factor intensity and change, rotation pressure) with draft talking points.",
        "input_schema": _schema({}),
        "strict": True,
    },
    {
        "name": "get_engine_recommendations",
        "description": "The quantitative engine's recommendations (R1..Rn) with quantitative basis and evidence ids. You may rewrite wording and re-prioritize, but not change numbers.",
        "input_schema": _schema({}),
        "strict": True,
    },
]


class AgentToolbox:
    def __init__(self, ctx: AdvisoryContext, recs: list) -> None:  # noqa: ANN001
        self.ctx = ctx
        self.recs = recs
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "search_filings": self.search_filings,
            "get_peer_benchmark": lambda _a: self.peer_benchmark(),
            "get_ownership_rotation": lambda _a: self.ownership_rotation(),
            "get_activist_vulnerability": self.activist,
            "get_capital_structure": lambda _a: self.capital_structure(),
            "get_capital_return": lambda _a: self.capital_return(),
            "get_investor_concerns": lambda _a: self.investor_concerns(),
            "get_engine_recommendations": lambda _a: self.engine_recommendations(),
        }

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if name not in self.handlers:
            return json.dumps({"error": f"unknown tool {name}"}), True
        try:
            return json.dumps(self.handlers[name](args), default=str), False
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), True

    # ------------------------------------------------------------------ handlers
    def search_filings(self, a: dict[str, Any]) -> dict[str, Any]:
        ctx = self.ctx
        ciks = [ctx.bundle.cik(a["ticker"])] if a.get("ticker") else [ctx.target_cik, *ctx.peer_ciks]
        k = max(1, min(int(a.get("top_k") or 5), 8))
        q = RetrievalQuery(text=a["query"], top_k=k, filters=RetrievalFilter(
            issuer_ciks=ciks, items=[a["item"]] if a.get("item") else None, filed_on_or_before=ctx.bundle.as_of))
        out = []
        for h in ctx.index.search(q, as_of=ctx.bundle.as_of):
            eid = ctx.ledger.add_chunk(h)
            cit = ctx.ledger.items[eid].citation
            out.append({"evidence_id": eid, "citation": cit.render() if cit else "", "authority": h.chunk.metadata.authority_weight,
                        "low_authority": h.chunk.metadata.form_type in (FormType.NEWS, FormType.CONSENSUS),
                        "relative_period": h.chunk.metadata.relative_period, "text": h.chunk.text[:1200]})
        return {"results": out}

    def peer_benchmark(self) -> dict[str, Any]:
        ctx = self.ctx
        return {"target": ctx.target.ticker, "peers": [ctx.t(c) for c in ctx.peer_ciks], "metrics": [
            {"metric": s.metric, "evidence_id": ctx.metric_evidence.get(s.metric), "target": s.target_value, "peer_median": s.peer_median,
             "p25": s.peer_p25, "p75": s.peer_p75, "robust_z": s.z_score, "peer_values": s.peer_values, "higher_is_better": s.higher_is_better}
            for s in ctx.stats]}

    def ownership_rotation(self) -> dict[str, Any]:
        ctx = self.ctx
        from ..graph.inference import net_flows_by_type

        return {
            "evidence_id": ctx.metric_evidence.get("rotation"),
            "periods": {"T": ctx.period_t, "T-1": ctx.period_t1, "T-4": ctx.period_t4},
            "rotations": [{"fund": r.filer_name, "type": r.institution_type, "from": r.source_ticker, "to": r.destination_ticker,
                           "rotation_usd": r.rotation_usd, "intensity": r.rotation_intensity, "window": f"{r.period_from}..{r.period_to}",
                           "risk_discrepancies": [{"theme": THEME_LABELS.get(d.theme, d.theme), "target_intensity": d.source_intensity,
                                                   "peer_intensity": d.destination_intensity, "new_vs_prior_10k": d.new_in_source} for d in r.risk_discrepancies[:4]]}
                          for r in ctx.rotations[:10]],
            "net_flows_by_institution_type_usd": net_flows_by_type(ctx.bundle, ctx.target_cik, ctx.period_t4 or ctx.period_t1, ctx.period_t),
        }

    def activist(self, a: dict[str, Any]) -> dict[str, Any]:
        t = (a.get("ticker") or self.ctx.target.ticker).upper()
        av = self.ctx.activist.get(t)
        if av is None:
            raise KeyError(f"{t} not in peer set {sorted(self.ctx.activist)}")
        return {"evidence_id": self.ctx.metric_evidence.get("activist"), **av.model_dump()}

    def capital_structure(self) -> dict[str, Any]:
        cs = self.ctx.capital_structure
        assert cs
        d = cs.model_dump()
        d["wacc_curve"] = [{"debt_to_capital": p.debt_to_capital, "rating": p.implied_rating, "wacc": round(p.wacc, 5)} for p in cs.wacc_curve]
        return {"evidence_id": self.ctx.metric_evidence.get("capital_structure"), "rating_commitment_evidence": self.ctx.theme_evidence.get("rating_commitment", []), **d}

    def capital_return(self) -> dict[str, Any]:
        cr = self.ctx.capital_return
        assert cr
        return {"evidence_id": self.ctx.metric_evidence.get("capital_return"), "insider_evidence_id": self.ctx.metric_evidence.get("insider"), **cr.model_dump()}

    def investor_concerns(self) -> dict[str, Any]:
        ir = self.ctx.ir_plan
        assert ir
        return {"evidence_id": self.ctx.metric_evidence.get("concerns"), **ir.model_dump()}

    def engine_recommendations(self) -> dict[str, Any]:
        return {"recommendations": [r.model_dump(mode="json") for r in self.recs]}
