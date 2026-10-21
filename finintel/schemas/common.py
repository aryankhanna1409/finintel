"""Shared enums, identifiers and temporal primitives used by every layer."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ACCESSION_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
FIXTURE_ACCESSION_RE = re.compile(r"^FIXTURE-[A-Z0-9.\-]+$")


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FormType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    THIRTEEN_F_HR = "13F-HR"
    FORM_4 = "4"
    SC_13D = "SC 13D"
    SC_13G = "SC 13G"
    DEF_14A = "DEF 14A"
    TRANSCRIPT = "TRANSCRIPT"
    CONSENSUS = "CONSENSUS"
    NEWS = "NEWS"
    MARKET_DATA = "MARKET_DATA"
    COMPUTED = "COMPUTED"


# Source-authority weighting: audited/attested SEC filings dominate, then
# regulatory ownership reports, then management speech, then third parties.
AUTHORITY_WEIGHTS: dict[FormType, float] = {
    FormType.TEN_K: 1.00,
    FormType.TEN_Q: 0.95,
    FormType.DEF_14A: 0.92,
    FormType.EIGHT_K: 0.90,
    FormType.THIRTEEN_F_HR: 0.90,
    FormType.FORM_4: 0.90,
    FormType.SC_13D: 0.90,
    FormType.SC_13G: 0.88,
    FormType.COMPUTED: 0.85,
    FormType.MARKET_DATA: 0.80,
    FormType.TRANSCRIPT: 0.75,
    FormType.CONSENSUS: 0.60,
    FormType.NEWS: 0.40,
}


class SourceTier(str, Enum):
    SEC_FILING = "sec_filing"
    TRANSCRIPT = "transcript"
    RESEARCH = "research"
    NEWS = "news"
    DERIVED = "derived"


def tier_for(form: FormType) -> SourceTier:
    if form in (FormType.TRANSCRIPT,):
        return SourceTier.TRANSCRIPT
    if form in (FormType.CONSENSUS,):
        return SourceTier.RESEARCH
    if form in (FormType.NEWS,):
        return SourceTier.NEWS
    if form in (FormType.COMPUTED, FormType.MARKET_DATA):
        return SourceTier.DERIVED
    return SourceTier.SEC_FILING


class SecurityType(str, Enum):
    COMMON_EQUITY = "common_equity"
    PREFERRED_EQUITY = "preferred_equity"
    SENIOR_DEBT = "senior_debt"
    SUBORDINATED_DEBT = "subordinated_debt"
    CONVERTIBLE_DEBT = "convertible_debt"
    OPTION_CALL = "option_call"
    OPTION_PUT = "option_put"
    WARRANT = "warrant"
    ADR = "adr"
    OTHER = "other"

    @property
    def is_equity_ownership(self) -> bool:
        return self in (SecurityType.COMMON_EQUITY, SecurityType.ADR)


class InstitutionType(str, Enum):
    INDEX_MANAGER = "index_manager"
    ACTIVE_MANAGER = "active_manager"
    HEDGE_FUND = "hedge_fund"
    ACTIVIST = "activist"
    PENSION = "pension"
    SOVEREIGN = "sovereign"
    BANK = "bank"
    INSURANCE = "insurance"
    OTHER = "other"


class FiscalPeriod(FrozenModel):
    """A fiscal reporting period. ``fiscal_quarter == 0`` denotes a full fiscal year."""

    fiscal_year: int = Field(ge=1990, le=2100)
    fiscal_quarter: int = Field(ge=0, le=4)
    period_end: date

    @property
    def is_annual(self) -> bool:
        return self.fiscal_quarter == 0

    @property
    def label(self) -> str:
        return f"FY{self.fiscal_year}" if self.is_annual else f"Q{self.fiscal_quarter} FY{self.fiscal_year}"

    def quarter_index(self) -> int:
        """Monotone index for quarter arithmetic; annual periods map to their Q4."""
        q = 4 if self.is_annual else self.fiscal_quarter
        return self.fiscal_year * 4 + (q - 1)


def calendar_quarter_end(d: date) -> date:
    q_end_month = ((d.month - 1) // 3 + 1) * 3
    return {3: date(d.year, 3, 31), 6: date(d.year, 6, 30), 9: date(d.year, 9, 30), 12: date(d.year, 12, 31)}[q_end_month]


def quarters_between(earlier: date, later: date) -> int:
    """Number of calendar quarters from ``earlier`` to ``later`` (by quarter-end bucket)."""
    e, l = calendar_quarter_end(earlier), calendar_quarter_end(later)
    return (l.year - e.year) * 4 + (l.month - e.month) // 3


def relative_period_label(period_end: date, anchor_period_end: date) -> str:
    """Encode a period relative to the analysis anchor: ``T``, ``T-1`` … ``T-4`` or ``T+n`` (future, suspicious)."""
    lag = quarters_between(period_end, anchor_period_end)
    if lag == 0:
        return "T"
    if lag > 0:
        return f"T-{lag}"
    return f"T+{-lag}"


class Accession(FrozenModel):
    """SEC accession number. Offline fixtures use an explicit ``FIXTURE-`` namespace so
    they can never be mistaken for (or cited as) a real EDGAR accession."""

    value: str

    @field_validator("value")
    @classmethod
    def _check(cls, v: str) -> str:
        v = v.strip()
        if ACCESSION_RE.match(v) or FIXTURE_ACCESSION_RE.match(v):
            return v
        if re.fullmatch(r"\d{18}", v):
            return f"{v[:10]}-{v[10:12]}-{v[12:]}"
        raise ValueError(f"invalid accession number: {v!r}")

    @property
    def is_fixture(self) -> bool:
        return self.value.startswith("FIXTURE-")

    @property
    def no_dashes(self) -> str:
        return self.value.replace("-", "")


class DataProvenance(FrozenModel):
    """Where a record came from; every derived number must be traceable to one of these."""

    source: str  # e.g. "sec_edgar", "stooq", "fred", "fixture"
    url: str | None = None
    retrieved_at: date | None = None
    is_synthetic: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def _fixture_flag(self) -> "DataProvenance":
        if self.source == "fixture" and not self.is_synthetic:
            object.__setattr__(self, "is_synthetic", True)
        return self
