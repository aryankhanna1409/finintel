"""Chunk, retrieval and citation models (the lineage backbone of the RAG pipeline)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from .common import FormType, SourceTier, StrictModel


class ChunkMetadata(StrictModel):
    chunk_id: str
    doc_id: str  # accession value
    parent_id: str | None = None  # section-level parent chunk (hierarchical "small-to-big")
    issuer_cik: str
    ticker: str | None
    company_name: str
    form_type: FormType
    source_tier: SourceTier
    authority_weight: float = Field(ge=0, le=1)
    accession: str
    is_synthetic: bool = False
    filed_date: date
    period_end: date | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None  # 0 = annual
    relative_period: str | None = None  # "T", "T-1", "T-4" relative to index anchor
    item: str | None = None  # "1A", "7", "Q&A" ...
    section_title: str | None = None
    page: int | None = None
    chunk_type: Literal["paragraph", "table", "qa_pair", "remarks", "section"] = "paragraph"
    speaker: str | None = None
    speaker_role: str | None = None
    source_url: str | None = None
    position: int = 0  # ordinal within the section


class Chunk(StrictModel):
    text: str
    metadata: ChunkMetadata
    # Text actually embedded/indexed: contextual header + body, so a table row
    # retrieved in isolation still knows company, period and section.
    index_text: str = ""

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)


class RetrievalFilter(StrictModel):
    issuer_ciks: list[str] | None = None
    form_types: list[FormType] | None = None
    items: list[str] | None = None
    chunk_types: list[str] | None = None
    filed_on_or_before: date | None = None  # point-in-time guard (no look-ahead)
    filed_on_or_after: date | None = None
    relative_periods: list[str] | None = None
    min_authority: float = 0.0
    exclude_synthetic: bool = False


class RetrievalQuery(StrictModel):
    text: str
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    top_k: int = Field(default=8, ge=1, le=100)
    candidate_pool: int = Field(default=50, ge=1, le=1000)
    dense_weight: float = Field(default=0.5, ge=0, le=1)
    use_rrf: bool = True
    authority_boost: float = Field(default=0.25, ge=0, le=2)
    recency_half_life_days: float = Field(default=365.0, gt=0)
    mmr_lambda: float = Field(default=0.75, ge=0, le=1)
    expand_parents: bool = False


class ScoredChunk(StrictModel):
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None


class Citation(StrictModel):
    citation_id: str  # stable short id, e.g. "C3"
    chunk_id: str
    accession: str
    form_type: FormType
    company: str
    ticker: str | None
    fiscal_period: str | None
    item: str | None
    page: int | None
    filed_date: date
    source_url: str | None = None
    is_synthetic: bool = False
    quote: str = ""

    def render(self) -> str:
        parts = [self.ticker or self.company, self.form_type.value]
        if self.fiscal_period:
            parts.append(self.fiscal_period)
        if self.item:
            parts.append(f"Item {self.item}" if self.item[0].isdigit() else self.item)
        if self.page:
            parts.append(f"p.{self.page}")
        parts.append(f"acc. {self.accession}")
        tag = ", ".join(parts)
        return f"[{self.citation_id}: {tag}{' (synthetic fixture)' if self.is_synthetic else ''}]"


class EvidenceItem(StrictModel):
    """A citable unit of evidence: either a retrieved chunk or a computed metric/graph fact."""

    evidence_id: str
    kind: Literal["chunk", "metric", "graph", "ownership", "insider"]
    summary: str
    citation: Citation | None = None
    values: dict[str, float | str | None] = Field(default_factory=dict)
    derived_from: list[str] = Field(default_factory=list)  # chunk ids / accessions


class GroundedClaim(StrictModel):
    claim: str
    evidence_ids: list[str] = Field(min_length=1)
    verified: bool = False
    verification_notes: list[str] = Field(default_factory=list)


class GroundedAnswer(StrictModel):
    question: str
    claims: list[GroundedClaim]
    citations: list[Citation]
    abstained: bool = False
    abstain_reason: str | None = None
