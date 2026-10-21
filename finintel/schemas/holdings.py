"""Ownership models: 13F positions, insider (Form 4) transactions, float and cap table."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import Field, computed_field, model_validator

from .common import Accession, DataProvenance, SecurityType, StrictModel


class HoldingRecord(StrictModel):
    """One row of a 13F-HR information table, resolved to an issuer and security type."""

    filer_cik: str
    filer_name: str
    accession: Accession
    report_period: date  # 13F "period of report" (calendar quarter end)
    filed_date: date
    issuer_name_raw: str
    cusip: str = Field(min_length=9, max_length=9)
    title_of_class: str
    issuer_cik: str | None = None  # resolved
    security_type: SecurityType = SecurityType.COMMON_EQUITY
    value_usd: float = Field(ge=0)
    shares: float = Field(ge=0)
    share_type: str = "SH"  # SH | PRN
    put_call: str | None = None
    investment_discretion: str = "SOLE"
    voting_sole: float = 0
    voting_shared: float = 0
    voting_none: float = 0
    provenance: DataProvenance

    @model_validator(mode="after")
    def _derivatives(self) -> "HoldingRecord":
        if self.put_call:
            pc = self.put_call.strip().upper()
            self.security_type = SecurityType.OPTION_PUT if pc == "PUT" else SecurityType.OPTION_CALL
        elif self.share_type == "PRN" and self.security_type == SecurityType.COMMON_EQUITY:
            self.security_type = SecurityType.SENIOR_DEBT
        return self

    @property
    def counts_toward_equity_ownership(self) -> bool:
        return self.security_type.is_equity_ownership and self.share_type == "SH"


class InsiderTxnCode(str, Enum):
    """Form 4 Table I transaction codes (subset relevant to signal extraction)."""

    P = "P"  # open-market purchase
    S = "S"  # open-market sale
    A = "A"  # grant / award
    M = "M"  # option exercise
    F = "F"  # tax withholding
    G = "G"  # gift
    D = "D"  # disposition to issuer
    C = "C"  # conversion
    X = "X"  # exercise of in-the-money derivative

    @property
    def is_discretionary(self) -> bool:
        """Open-market trades carry information; awards/withholding/gifts mostly don't."""
        return self in (InsiderTxnCode.P, InsiderTxnCode.S)


class InsiderTransaction(StrictModel):
    accession: Accession
    issuer_cik: str
    ticker: str | None = None
    insider_cik: str | None = None
    insider_name: str
    insider_role: str  # "Director", "CEO", "CFO", "10% Owner", "Officer"
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    transaction_date: date
    filed_date: date
    code: InsiderTxnCode
    shares: float = Field(gt=0)
    price: float | None = Field(default=None, ge=0)
    acquired: bool  # A (True) / D (False)
    shares_owned_after: float | None = Field(default=None, ge=0)
    is_10b5_1: bool = False
    provenance: DataProvenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value_usd(self) -> float:
        return self.shares * (self.price or 0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def filing_lag_days(self) -> int:
        return (self.filed_date - self.transaction_date).days


class FloatSnapshot(StrictModel):
    issuer_cik: str
    as_of: date
    shares_outstanding: float = Field(gt=0)
    insider_shares: float = Field(default=0, ge=0)
    restricted_shares: float = Field(default=0, ge=0)
    short_interest_shares: float | None = Field(default=None, ge=0)
    avg_daily_volume: float | None = Field(default=None, ge=0)
    provenance: DataProvenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def free_float(self) -> float:
        return max(self.shares_outstanding - self.insider_shares - self.restricted_shares, 0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def short_interest_pct_float(self) -> float | None:
        if self.short_interest_shares is None or self.free_float == 0:
            return None
        return self.short_interest_shares / self.free_float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days_to_cover(self) -> float | None:
        if not self.short_interest_shares or not self.avg_daily_volume:
            return None
        return self.short_interest_shares / self.avg_daily_volume


class OwnershipConcentration(StrictModel):
    issuer_cik: str
    report_period: date
    holders: int
    institutional_shares: float
    institutional_pct_outstanding: float
    hhi: float  # 0..10_000 on institutional-share basis
    top10_pct_institutional: float
    activist_pct_outstanding: float
    index_pct_outstanding: float


class OwnershipFlow(StrictModel):
    """Change in one institution's position in one issuer between two report periods."""

    filer_cik: str
    filer_name: str
    issuer_cik: str
    period_from: date
    period_to: date
    shares_from: float
    shares_to: float
    value_from: float
    value_to: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def share_change(self) -> float:
        return self.shares_to - self.shares_from

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pct_change(self) -> float | None:
        if self.shares_from == 0:
            return None
        return self.share_change / self.shares_from

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> str:
        if self.shares_from == 0 and self.shares_to > 0:
            return "new"
        if self.shares_from > 0 and self.shares_to == 0:
            return "exited"
        if self.share_change > 0:
            return "increased"
        if self.share_change < 0:
            return "decreased"
        return "unchanged"
