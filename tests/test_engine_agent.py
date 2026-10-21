import json
import re
from types import SimpleNamespace

import pytest

from finintel.agents import TOOL_DEFS, ClaudeAdvisoryAgent, FinancialIntelligencePipeline, PipelineConfig
from finintel.agents.tools import AgentToolbox
from finintel.cli import main
from finintel.reporting import to_json, to_markdown
from finintel.schemas import LLMNarrative, LLMRecommendationEdit, StrategicAdvisoryReport

PEERS = ["WMT", "COST", "DG", "KR"]


# ------------------------------------------------------------------ deterministic engine
def test_report_structure(report):
    assert report.target_ticker == "TGT" and report.peers == PEERS
    assert {r.rec_id for r in report.recommendations} >= {"R1", "R2", "R3", "R4", "R6"}
    assert report.narrative_engine == "deterministic"
    assert report.activist_vulnerability["TGT"].band in ("elevated", "high")
    assert report.capital_structure.rating_floor == "A"  # inferred from "middle-A" commitment text
    assert report.capital_return.insider_signal.cluster_buy
    assert report.capital_return.recommended_mix["buyback"] > report.capital_return.recommended_mix["special_dividend"]


def test_every_cited_evidence_id_exists_and_has_lineage(report):
    ids = {e.evidence_id for e in report.evidence}
    for r in report.recommendations:
        assert r.evidence_ids and set(r.evidence_ids) <= ids
        for found in re.findall(r"E\d+", r.rationale):
            assert found in ids
    for e in report.evidence:
        assert e.derived_from, e.evidence_id
        if e.kind == "chunk":
            assert e.citation and e.citation.accession.startswith("FIXTURE-")


def test_grounding_is_complete(report):
    g = report.grounding
    assert g["grounding_rate"] == 1.0 and not g["unknown_evidence_ids"] and not g["unsupported_numbers"] and not g["uncited_factual"]


def test_numbers_in_report_trace_to_data(report):
    ev = next(s for s in report.benchmark.statistics if s.metric == "ev_to_ebitda")
    assert f"{ev.target_value:.1f}x" in report.executive_summary
    assert report.capital_structure.optimal_debt_to_capital in [p.debt_to_capital for p in report.capital_structure.wacc_curve]


def test_peer_count_enforced(engine):
    with pytest.raises(ValueError):
        engine.build_context("TGT", ["WMT", "COST"])


def test_renderers(report):
    md = to_markdown(report)
    for h in ("## Executive summary", "## Recommendations", "## Peer benchmark", "## Capital structure", "## Investor relations plan", "## Citations"):
        assert h in md
    assert "synthetic" in md.lower()
    StrategicAdvisoryReport.model_validate_json(to_json(report))


# ------------------------------------------------------------------ agent with a fake Anthropic client
def _block(**kw):
    return SimpleNamespace(**kw)


class FakeMessages:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.script(len(self.calls), kwargs)


class FakeClient:
    def __init__(self, script):
        self.beta = SimpleNamespace(messages=FakeMessages(script))


def _scripted(final_builder):
    def script(n, kwargs):
        if n == 1:
            return SimpleNamespace(stop_reason="tool_use", parsed_output=None, content=[
                _block(type="text", text="Gathering evidence."),
                _block(type="tool_use", id="tu1", name="get_engine_recommendations", input={}),
                _block(type="tool_use", id="tu2", name="search_filings", input={"query": "tariff exposure", "ticker": "TGT", "item": "", "top_k": 3}),
            ])
        if n == 2:
            return SimpleNamespace(stop_reason="tool_use", parsed_output=None, content=[
                _block(type="tool_use", id="tu3", name="get_activist_vulnerability", input={"ticker": "ZZZZ"}),
            ])
        # Final turn: build a narrative from the evidence ids the tools returned.
        tool_results = [json.loads(c["content"]) for m in kwargs["messages"] if m["role"] == "user" and isinstance(m["content"], list) for c in m["content"]]
        return SimpleNamespace(stop_reason="end_turn", content=[], parsed_output=final_builder(tool_results))
    return script


def _good_and_bad_narrative(tool_results):
    recs = tool_results[0]["recommendations"]
    search_eid = tool_results[1]["results"][0]["evidence_id"]
    r2 = next(r for r in recs if r["rec_id"] == "R2")
    ev = r2["evidence_ids"][0]
    return LLMNarrative(
        executive_summary=(f"Target screens as deeply undervalued versus peers [{ev}]. Tariff exposure is a disclosed risk that investors are pressing on [{search_eid}]. "
                           f"Management should resume repurchases while protecting its rating commitment [{ev}]. Insider buying supports that call [{ev}]. "
                           f"The register now includes an activist [{ev}]. Leadership transition adds execution risk [{search_eid}]. "
                           f"EBITDA margins will reach 42.7% next year [{ev}]."),
        key_findings=[f"Valuation discount is the central fact [{ev}].", f"Tariffs feature in the risk factors [{search_eid}].", "Fabricated claim of 99.9% share [E9999]."],
        recommendations=[
            LLMRecommendationEdit(rec_id="R2", title="Restart buybacks now", action="Resume repurchases immediately with an ASR of $77bn.",
                                  rationale=f"The discount to peers is large [{ev}].", evidence_ids=[ev, "E9999"], priority="critical"),
            LLMRecommendationEdit(rec_id="R99", title="x", action="x", rationale="x", evidence_ids=[], priority="low"),
        ],
        talking_points=[f"Quantify the tariff bridge [{search_eid}].", "Unsupported 12.3% claim [E9999]."],
        open_questions=["Confirm agency lease adjustments."],
    )


def test_agent_tool_loop_and_grounding_merge(bundle):
    client = FakeClient(_scripted(_good_and_bad_narrative))
    pipe = FinancialIntelligencePipeline(bundle, PipelineConfig(use_llm=True), llm_client=client)
    rep = pipe.advise("TGT", PEERS)
    calls = client.beta.messages.calls
    assert len(calls) == 3
    k = calls[0]
    assert k["model"] == "claude-opus-5-5" and k["thinking"] == {"type": "adaptive"} and k["output_config"] == {"effort": "high"}
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"] and k["output_format"] is LLMNarrative
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in k["tools"])
    # Tool results were returned in one user message, errors flagged.
    tr = calls[2]["messages"][-1]["content"]
    assert tr[0]["is_error"] and "ZZZZ" in tr[0]["content"]
    assert rep.narrative_engine == "anthropic:claude-opus-5-5"
    assert "42.7" not in rep.executive_summary and "undervalued" in rep.executive_summary  # one unsupported sentence removed, rest kept
    assert any("removed 1 unsupported summary sentence" in t for t in rep.llm_trace)
    assert not any("E9999" in f for f in rep.key_findings)
    r2 = next(r for r in rep.recommendations if r.rec_id == "R2")
    assert r2.title == "Restart buybacks now" and r2.priority.value == "critical"
    assert "$77bn" not in r2.action  # unsupported number in action → engine action kept
    assert "E9999" not in r2.evidence_ids
    assert any("R99" in t for t in rep.llm_trace) and any("open question" in t for t in rep.llm_trace)
    assert rep.grounding["unknown_evidence_ids"] == []


def test_heavily_fabricated_summary_reverts_to_engine_text(bundle, engine):
    ctx = engine.analyze(engine.build_context("TGT", PEERS))
    recs = engine.recommendations(ctx)
    summary, findings = engine.deterministic_narrative(ctx, recs)
    from finintel.agents import merge_narrative

    bad = LLMNarrative(executive_summary="Revenue will grow 55% [E1]. Margins hit 31% [E2]. The CEO resigned [E9999]. Shares are cheap [E1].",
                       key_findings=[], recommendations=[], talking_points=[], open_questions=[])
    s2, f2, _, _, notes = merge_narrative(ctx, recs, bad, summary, findings)
    assert s2 == summary and f2 == findings and any("reverted" in n for n in notes)


def test_agent_refusal_falls_back(bundle):
    client = FakeClient(lambda n, kw: SimpleNamespace(stop_reason="refusal", content=[], parsed_output=None))
    rep = FinancialIntelligencePipeline(bundle, PipelineConfig(use_llm=True), llm_client=client).advise("TGT", PEERS)
    assert rep.narrative_engine == "deterministic" and any("refusal" in t for t in rep.llm_trace)


def test_agent_exception_never_loses_report(bundle):
    def boom(n, kw):
        raise RuntimeError("network down")
    rep = FinancialIntelligencePipeline(bundle, PipelineConfig(use_llm=True), llm_client=FakeClient(boom)).advise("TGT", PEERS)
    assert rep.narrative_engine == "deterministic" and "network down" in rep.llm_trace[0]


def test_agent_step_budget(engine):
    ctx = engine.analyze(engine.build_context("TGT", PEERS))
    loop = FakeClient(lambda n, kw: SimpleNamespace(stop_reason="tool_use", parsed_output=None,
                                                    content=[_block(type="tool_use", id=f"t{n}", name="get_peer_benchmark", input={})]))
    res = ClaudeAdvisoryAgent(client=loop, max_steps=3).run(ctx, engine.recommendations(ctx))
    assert res.narrative is None and "without a final narrative" in res.trace[-1]


def test_toolbox_outputs_reference_ledger(engine):
    ctx = engine.analyze(engine.build_context("TGT", PEERS))
    tb = AgentToolbox(ctx, engine.recommendations(ctx))
    for t in TOOL_DEFS:
        args = {"query": "buyback", "ticker": "", "item": "", "top_k": 2} if t["name"] == "search_filings" else ({"ticker": "TGT"} if t["name"] == "get_activist_vulnerability" else {})
        out, err = tb.call(t["name"], args)
        assert not err, (t["name"], out)
        for eid in re.findall(r'"(E\d+)"', out):
            assert eid in ctx.ledger.items


# ------------------------------------------------------------------ CLI
def test_cli_report_and_ask(tmp_path, capsys):
    assert main(["report", "TGT", "--peers", *PEERS, "--out", str(tmp_path), "--no-llm"]) == 0
    assert list(tmp_path.glob("TGT_advisory_*.md")) and list(tmp_path.glob("TGT_advisory_*.json"))
    assert main(["ask", "What did Walmart say about tariffs?", "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "WMT" in out and "ABSTAINED" not in out
