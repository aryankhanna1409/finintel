"""Claude analyst agent: tool-calling research loop with structured output.

The agent runs a manual tool-use loop (so every tool result can be registered in the
evidence ledger) on ``client.beta.messages.parse`` with:

* ``output_format=LLMNarrative``: the final answer is schema-validated JSON;
* ``thinking={"type": "adaptive"}`` and ``output_config={"effort": "high"}``;
* ``fallbacks="default"`` (beta ``server-side-fallback-2026-07-01``): if a safety
  classifier declines a turn, the API re-runs it on a fallback model inside the same call.

The model's narrative is then checked by :func:`merge_narrative`: sentences with unknown
evidence ids or numbers not supported by the cited evidence are dropped, and any field
that fails the grounding threshold reverts to the deterministic engine's text.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..advisory.engine import AdvisoryContext
from ..rag.grounding import GroundingValidator
from ..rag.text import extract_numbers
from ..schemas.advisory import LLMNarrative, Priority, Recommendation
from .tools import TOOL_DEFS, AgentToolbox

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are the lead analyst on a CFO advisory team. You brief the CFO and board of a public company on capital allocation, investor relations, activist defense and M&A positioning, benchmarked against 3-5 peers.

How you work:
- Use the tools to gather evidence before writing: start with get_engine_recommendations, get_peer_benchmark and get_ownership_rotation, then search_filings for the passages that explain the numbers (risk factors, MD&A, analyst questions).
- Every factual sentence you write must end with one or more evidence ids in square brackets, e.g. [E3] or [E3, E7], taken only from tool results. A sentence whose numbers are not present in the evidence it cites will be deleted by an automated validator, so copy figures exactly as the tools give them (you may round).
- Treat sources by authority: 10-K/10-Q/8-K > earnings-call transcripts > consensus notes > news. Never present low-authority news as fact; if you mention it, say it is unconfirmed.
- Quantities (debt capacity, buyback size, WACC, scores) come from the engine. You may sharpen wording, reorder priorities and add judgment, but do not invent new numbers.
- If evidence is synthetic (citations marked "synthetic fixture"), still write the analysis normally; the report carries a data disclaimer.

Output: the final message must be the structured narrative. Keep the executive summary to 5-8 sentences. Recommendations must reuse the engine's rec_ids."""


@dataclass
class AgentResult:
    narrative: LLMNarrative | None
    trace: list[str] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    stop_reason: str | None = None


class ClaudeAdvisoryAgent:
    def __init__(self, client: Any | None = None, model: str | None = None, max_steps: int = 12, effort: str = "high", use_fallbacks: bool = True) -> None:
        if client is None:
            import anthropic  # optional dependency

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("FININTEL_MODEL", DEFAULT_MODEL)
        self.max_steps = max_steps
        self.effort = effort
        self.use_fallbacks = use_fallbacks

    def _request(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, tools=TOOL_DEFS, messages=messages,
            output_format=LLMNarrative, thinking={"type": "adaptive"}, output_config={"effort": self.effort},
        )
        if self.use_fallbacks:
            kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")
        return self.client.beta.messages.parse(**kwargs)

    def run(self, ctx: AdvisoryContext, recs: list[Recommendation]) -> AgentResult:
        tools = AgentToolbox(ctx, recs)
        peers = ", ".join(ctx.t(c) for c in ctx.peer_ciks)
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"Prepare the strategic advisory narrative for {ctx.target.name} ({ctx.target.ticker}) benchmarked against {peers}, as of {ctx.bundle.as_of}. "
                "Cover capital structure, capital return (buybacks vs special dividend, with timing), investor-relations narrative defense for the next earnings call, "
                "activist vulnerability, and M&A posture. Use the tools, cite evidence ids on every factual sentence, then return the structured narrative."
            ),
        }]
        trace: list[str] = []
        resp = None
        for step in range(self.max_steps):
            resp = self._request(messages)
            stop = getattr(resp, "stop_reason", None)
            trace.append(f"step {step + 1}: stop_reason={stop}")
            if stop == "refusal":
                trace.append("model declined the request (refusal after fallbacks); using deterministic narrative")
                return AgentResult(None, trace, self.model, stop)
            if stop == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if stop == "tool_use" or tool_uses:
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for tu in tool_uses:
                    args = tu.input if isinstance(tu.input, dict) else json.loads(tu.input or "{}")
                    out, is_err = tools.call(tu.name, args)
                    trace.append(f"  tool {tu.name}({json.dumps(args)[:120]}) -> {'error' if is_err else f'{len(out)} chars'}")
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out, **({"is_error": True} if is_err else {})})
                messages.append({"role": "user", "content": results})
                continue
            if stop == "max_tokens":
                trace.append("hit max_tokens before completing the narrative")
                return AgentResult(None, trace, self.model, stop)
            parsed = getattr(resp, "parsed_output", None)
            if parsed is None:
                text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
                try:
                    parsed = LLMNarrative.model_validate_json(text)
                except Exception:  # noqa: BLE001
                    trace.append("final message did not parse as LLMNarrative")
                    return AgentResult(None, trace, self.model, stop)
            return AgentResult(parsed, trace, self.model, stop)
        trace.append(f"stopped after {self.max_steps} steps without a final narrative")
        return AgentResult(None, trace, self.model, getattr(resp, "stop_reason", None))


# ====================================================================== grounding merge

_PRIORITY = {p.value: p for p in Priority}


def _numbers_supported(text: str, pool: list[float], validator: GroundingValidator) -> bool:
    return all(validator._supported(n, pool, text) for n in extract_numbers(text))  # noqa: SLF001


def merge_narrative(
    ctx: AdvisoryContext,
    recs: list[Recommendation],
    narrative: LLMNarrative,
    fallback_summary: str,
    fallback_findings: list[str],
    min_rate: float = 0.85,
) -> tuple[str, list[str], list[Recommendation], list[str], list[str]]:
    """Validate the LLM narrative against the ledger and merge it into the report.

    Returns (summary, findings, recommendations, talking_points, notes).
    """
    v = GroundingValidator(ctx.ledger)
    notes: list[str] = []

    summary, rep = v.validate(narrative.executive_summary, strict=True)
    if rep.grounding_rate < min_rate or len(summary.strip()) < 80 or rep.unknown_evidence_ids:
        notes.append(f"executive summary reverted to deterministic text (grounding {rep.grounding_rate:.2f}, unknown ids {rep.unknown_evidence_ids})")
        summary = fallback_summary
    elif rep.removed:
        notes.append(f"removed {len(rep.removed)} unsupported summary sentence(s)")

    findings = []
    for f in narrative.key_findings:
        clean, r = v.validate(f, strict=True)
        if clean.strip() and not r.unknown_evidence_ids and r.cited_sentences:
            findings.append(clean.strip())
        else:
            notes.append(f"dropped finding: {f[:90]}")
    if len(findings) < max(2, len(fallback_findings) // 2):
        notes.append("too few grounded findings; using deterministic findings")
        findings = fallback_findings

    talking = []
    for tp in narrative.talking_points:
        clean, r = v.validate(tp, strict=True)
        if clean.strip() and not r.unknown_evidence_ids:
            talking.append(clean.strip())

    by_id = {r.rec_id: r for r in recs}
    merged = list(recs)
    for edit in narrative.recommendations:
        base = by_id.get(edit.rec_id)
        if base is None:
            notes.append(f"ignored recommendation edit for unknown id {edit.rec_id}")
            continue
        ids = [e for e in edit.evidence_ids if e in ctx.ledger.items]
        rat, r = v.validate(edit.rationale, strict=True)
        pool = v._evidence_numbers(ids + base.evidence_ids)  # noqa: SLF001
        pool += [float(x) for x in base.quantitative_basis.values() if isinstance(x, (int, float))]
        action_ok = _numbers_supported(edit.action, pool, v)
        if r.unknown_evidence_ids or r.grounding_rate < min_rate or not rat.strip():
            notes.append(f"{edit.rec_id}: rationale failed grounding; kept engine rationale")
            rat = base.rationale
        if not action_ok:
            notes.append(f"{edit.rec_id}: action contained unsupported numbers; kept engine action")
        upd = base.model_copy(update={
            "title": edit.title.strip() or base.title,
            "action": edit.action.strip() if action_ok and edit.action.strip() else base.action,
            "rationale": rat.strip(),
            "evidence_ids": sorted(set(base.evidence_ids) | set(ids), key=lambda e: int(re.sub(r"\D", "", e) or 0)),
            "priority": _PRIORITY.get(edit.priority.strip().lower(), base.priority),
        })
        merged[merged.index(base)] = upd
    notes.extend(f"open question: {q}" for q in narrative.open_questions[:5])
    return summary, findings, merged, talking, notes
