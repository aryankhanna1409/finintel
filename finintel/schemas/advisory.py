"""Prescriptive advisory output models (the Strategic Advisory Report)."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

from .common import StrictModel
from .graph import ActivistVulnerability, CapitalRotationFinding, CommonOwnershipLink, SupplyChainExposure
from .holdings import OwnershipConcentration
from .metrics import PeerStatistic
from .retrieval import Citation, EvidenceItem


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Horizon(str, Enum):
    IMMEDIATE = "0-3 months"
    NEAR = "3-12 months"
    STRATEGIC = "12-36 months"


class PlaybookArea(str, Enum):
    CAPITAL_STRUCTURE = "capital_structure"
    CAPITAL_RETURN = "capital_return"
    INVESTOR_RELATIONS = "investor_relations"
    ACTIVIST_DEFENSE = "activist_defense"
    M_AND_A = "m_and_a"
    VALUATION = "valuation"


class Recommendation(StrictModel):
    rec_id: str
    area: PlaybookArea
    title: str
    action: str
    rationale: str
    quantitative_basis: dict[str, float | str | None] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    priority: Priority
    horizon: Horizon
    confidence: float = Field(ge=0, le=1)
    kpis_to_track: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class WaccPoint(StrictModel):
    debt_to_capital: float
    implied_rating: str
    pre_tax_cost_of_debt: float
    levered_beta: float
    cost_of_equity: float
    wacc: float
    interest_coverage: float | None


class CapitalStructureAnalysis(StrictModel):
    current_debt_to_capital: float
    current_wacc: float
    current_implied_rating: str
    actual_rating: str | None
    optimal_debt_to_capital: float
    optimal_wacc: float
    optimal_rating: str
    wacc_curve: list[WaccPoint]
    peer_net_debt_to_ebitda_median: float | None
    target_net_debt_to_ebitda: float | None
    incremental_debt_capacity_usd: float  # to move to the optimal (rating-constrained) point
    rating_floor: str
    recommendation: str  # "add_debt" | "hold" | "delever"


class InsiderSignal(StrictModel):
    issuer_cik: str
    window_days: int
    open_market_buys: int
    open_market_sells: int
    buy_value_usd: float
    sell_value_usd: float
    net_value_usd: float
    distinct_buyers: int
    cluster_buy: bool
    planned_sale_share: float  # fraction of sale value under 10b5-1
    signal: str  # "bullish" | "neutral" | "bearish"
    score: float  # -1..1


class CapitalReturnAnalysis(StrictModel):
    valuation_discount_ev_ebitda: float | None  # vs peer median, decimal (positive = cheaper)
    valuation_z_ev_ebitda: float | None
    fcf_yield: float | None
    normalized_fcf_yield: float | None  # FCF yield haircut by the consensus EPS revision
    long_term_growth: float
    cost_of_equity: float
    buyback_irr_proxy: float | None  # normalized FCF yield + long-term growth (Gordon), vs cost of equity
    insider_signal: InsiderSignal
    excess_cash_usd: float
    annual_capacity_usd: float
    recommended_mix: dict[str, float]  # {"buyback": x, "special_dividend": y, "regular_dividend_growth": z} in USD
    execution: str
    blackout_note: str


class InvestorConcern(StrictModel):
    theme: str
    analyst_question_count: int
    risk_factor_intensity: float
    risk_factor_delta_vs_t4: float
    holder_pressure_usd: float  # value sold by holders whose exits correlate with theme
    severity: float  # 0..1
    evidence_ids: list[str]


class TalkingPoint(StrictModel):
    theme: str
    message: str
    proof_points: list[str]
    kpi_commitments: list[str]
    avoid: list[str]
    evidence_ids: list[str]


class InvestorRelationsPlan(StrictModel):
    concerns: list[InvestorConcern]
    talking_points: list[TalkingPoint]
    holders_to_engage: list[str]
    holders_at_risk: list[str]


class PeerBenchmark(StrictModel):
    target_ticker: str
    peers: list[str]
    statistics: list[PeerStatistic]


class StrategicAdvisoryReport(StrictModel):
    report_id: str
    generated_at: date
    as_of: date
    target_ticker: str
    target_name: str
    peers: list[str]
    executive_summary: str
    key_findings: list[str]
    benchmark: PeerBenchmark
    ownership: dict[str, OwnershipConcentration]
    capital_rotation: list[CapitalRotationFinding]
    activist_vulnerability: dict[str, ActivistVulnerability]
    common_ownership: list[CommonOwnershipLink]
    supply_chain: SupplyChainExposure | None
    capital_structure: CapitalStructureAnalysis
    capital_return: CapitalReturnAnalysis
    investor_relations: InvestorRelationsPlan
    recommendations: list[Recommendation]
    evidence: list[EvidenceItem]
    citations: list[Citation]
    grounding: dict[str, int | float | list[str]]
    narrative_engine: str  # "anthropic:<model>" or "deterministic"
    data_disclaimer: str
    llm_trace: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# LLM-facing structured output. Deliberately flat and constraint-light so it maps cleanly
# onto JSON-schema structured outputs; numbers are never invented here: the LLM may only
# reference evidence ids, and the grounding validator rejects anything else.
# --------------------------------------------------------------------------------------


class LLMRecommendationEdit(BaseModel):
    rec_id: str = Field(description="id of an existing engine recommendation, e.g. R1")
    title: str
    action: str = Field(description="Concrete action for the CFO / board, imperative voice")
    rationale: str = Field(description="Why, referencing evidence ids in square brackets like [E3]")
    evidence_ids: list[str]
    priority: str = Field(description="critical | high | medium | low")


class LLMNarrative(BaseModel):
    executive_summary: str = Field(description="5-8 sentences for a CFO; every factual sentence cites [E#]")
    key_findings: list[str] = Field(description="Each finding ends with one or more [E#] citations")
    recommendations: list[LLMRecommendationEdit]
    talking_points: list[str] = Field(description="Earnings-call talking points, each citing [E#]")
    open_questions: list[str] = Field(description="Data gaps or items management should verify")
