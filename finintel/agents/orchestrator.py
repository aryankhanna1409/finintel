"""End-to-end orchestration: ingest → index → graph → engine → (Claude agent) → report.

::

    pipeline = FinancialIntelligencePipeline.from_fixtures()
    report = pipeline.advise("TGT", ["WMT", "COST", "DG", "KR"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ..advisory.engine import StrategicPrescriptiveEngine
from ..graph.builder import FinancialKnowledgeGraph
from ..ingestion.sources import CorpusBundle, FixtureSource, LiveEdgarSource
from ..rag.embeddings import get_embedder
from ..rag.grounding import ExtractiveAnswerer
from ..rag.index import FinancialRAGIndex
from ..rag.rerank import get_reranker
from ..schemas.advisory import StrategicAdvisoryReport
from ..schemas.retrieval import GroundedAnswer
from .claude_agent import ClaudeAdvisoryAgent, merge_narrative

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    embedder: str | None = None  # None/"hashing" or a sentence-transformers id, e.g. "BAAI/bge-base-en-v1.5"
    reranker: str | None = None  # None/"features" or a cross-encoder id, e.g. "BAAI/bge-reranker-base"
    use_llm: bool | None = None  # None = auto (use Claude when ANTHROPIC_API_KEY/credentials exist)
    model: str | None = None


class FinancialIntelligencePipeline:
    def __init__(self, bundle: CorpusBundle, config: PipelineConfig | None = None, llm_client: object | None = None) -> None:
        self.config = config or PipelineConfig()
        self.bundle = bundle
        embedder = get_embedder(self.config.embedder)
        self.index = FinancialRAGIndex(embedder=embedder)
        self.index.build_from_bundle(bundle)
        if self.config.reranker:
            self.index._reranker = get_reranker(self.config.reranker, self.index.bm25.idf)  # noqa: SLF001
        self.kg = FinancialKnowledgeGraph().build(bundle, self.index.chunks)
        self.engine = StrategicPrescriptiveEngine(bundle, self.index, self.kg)
        self.llm_client = llm_client

    @classmethod
    def from_fixtures(cls, config: PipelineConfig | None = None, llm_client: object | None = None) -> "FinancialIntelligencePipeline":
        return cls(FixtureSource().load(), config, llm_client)

    @classmethod
    def from_live(cls, tickers: list[str], user_agent: str, config: PipelineConfig | None = None, **kw: object) -> "FinancialIntelligencePipeline":
        return cls(LiveEdgarSource(tickers, user_agent, **kw).load(), config)  # type: ignore[arg-type]

    def _llm_enabled(self) -> bool:
        if self.config.use_llm is not None:
            return self.config.use_llm
        if self.llm_client is not None:
            return True
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def advise(self, target: str, peers: list[str] | None = None) -> StrategicAdvisoryReport:
        eng = self.engine
        ctx = eng.analyze(eng.build_context(target, peers))
        recs = eng.recommendations(ctx)
        summary, findings = eng.deterministic_narrative(ctx, recs)
        if not self._llm_enabled():
            return eng.assemble(ctx, recs, summary, findings, "deterministic", ["LLM disabled or no Anthropic credentials; deterministic narrative"])
        try:
            agent = ClaudeAdvisoryAgent(client=self.llm_client, model=self.config.model)
            result = agent.run(ctx, recs)
        except Exception as exc:  # noqa: BLE001 - never lose the report to an API failure
            log.exception("Claude agent failed")
            return eng.assemble(ctx, recs, summary, findings, "deterministic", [f"LLM error: {type(exc).__name__}: {exc}"])
        if result.narrative is None:
            return eng.assemble(ctx, recs, summary, findings, "deterministic", result.trace)
        s2, f2, recs2, talking, notes = merge_narrative(ctx, recs, result.narrative, summary, findings)
        if talking and ctx.ir_plan:
            for tp, text in zip(ctx.ir_plan.talking_points, talking):
                tp.message = text
        return eng.assemble(ctx, recs2, s2, f2, f"anthropic:{result.model}", result.trace + notes)

    def ask(self, question: str, ticker: str | None = None) -> GroundedAnswer:
        from ..schemas.retrieval import RetrievalFilter, RetrievalQuery

        ciks = [self.bundle.cik(ticker)] if ticker else (self.bundle.registry.mentions(question) or None)
        filt = RetrievalFilter(issuer_ciks=ciks, filed_on_or_before=self.bundle.as_of)
        return ExtractiveAnswerer(self.index).answer(question, RetrievalQuery(text=question, top_k=6, filters=filt), as_of=self.bundle.as_of)
