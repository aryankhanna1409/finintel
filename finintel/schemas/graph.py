"""Knowledge-graph schema and inference-result models."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import Field

from .common import StrictModel


class NodeLabel(str, Enum):
    COMPANY = "Company"
    INSTITUTION = "Institution"
    SECURITY = "Security"
    INSIDER = "Insider"
    RISK_THEME = "RiskTheme"
    FILING = "Filing"


class EdgeType(str, Enum):
    HOLDS_EQUITY = "HOLDS_EQUITY"  # (Institution)-[:HOLDS_EQUITY {period}]->(Company)
    HOLDS_DEBT = "HOLDS_DEBT"
    HOLDS_DERIVATIVE = "HOLDS_DERIVATIVE"
    COMPETES_WITH = "COMPETES_WITH"  # symmetric, stored both directions
    SUPPLIES = "SUPPLIES"  # (Supplier)-[:SUPPLIES]->(Customer)
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    ISSUED = "ISSUED"  # (Company)-[:ISSUED]->(Security)
    INSIDER_OF = "INSIDER_OF"
    TRADED = "TRADED"  # (Insider)-[:TRADED {date, code}]->(Company)
    DISCLOSES_RISK = "DISCLOSES_RISK"  # (Company)-[:DISCLOSES_RISK {period, intensity}]->(RiskTheme)
    FILED = "FILED"


class GraphStats(StrictModel):
    nodes_by_label: dict[str, int]
    edges_by_type: dict[str, int]


class RiskDiscrepancy(StrictModel):
    theme: str
    source_intensity: float  # company capital is leaving
    destination_intensity: float  # company capital is moving to
    gap: float
    new_in_source: bool = False  # theme newly disclosed vs T-4
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class CapitalRotationFinding(StrictModel):
    """An institution reducing exposure to ``source`` while adding to ``destination``."""

    filer_cik: str
    filer_name: str
    institution_type: str
    source_cik: str
    source_ticker: str
    destination_cik: str
    destination_ticker: str
    period_from: date
    period_to: date
    source_value_change_usd: float
    destination_value_change_usd: float
    rotation_usd: float  # min(|outflow|, inflow)
    rotation_intensity: float  # rotation / filer's prior combined exposure
    risk_discrepancies: list[RiskDiscrepancy] = Field(default_factory=list)


class ActivistVulnerability(StrictModel):
    issuer_cik: str
    ticker: str
    score: float = Field(ge=0, le=100)
    band: str  # "low" | "moderate" | "elevated" | "high"
    components: dict[str, float]  # normalized 0..1 contributions
    raw_inputs: dict[str, float | None]
    activists_present: list[str]
    narrative_drivers: list[str]


class CommonOwnershipLink(StrictModel):
    company_a: str
    company_b: str
    overlap_cosine: float
    shared_holders: int
    top_shared: list[str]


class SupplyChainExposure(StrictModel):
    issuer_cik: str
    ticker: str
    upstream_suppliers: list[str]
    shared_suppliers_with_peers: dict[str, list[str]]
    concentration_note: str
