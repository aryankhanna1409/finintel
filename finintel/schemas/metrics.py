"""Financial statement and market metric models plus derived ratios."""

from __future__ import annotations

from datetime import date

from pydantic import Field, computed_field

from .common import DataProvenance, FiscalPeriod, StrictModel


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


class FinancialStatementSnapshot(StrictModel):
    """Annual (or TTM) fundamentals in USD. Field names follow XBRL us-gaap concepts loosely."""

    issuer_cik: str
    ticker: str
    period: FiscalPeriod
    revenue: float
    cogs: float
    operating_income: float
    depreciation_amortization: float
    interest_expense: float = Field(ge=0)
    income_tax_expense: float
    pretax_income: float
    net_income: float
    diluted_eps: float | None = None
    shares_diluted: float = Field(gt=0)
    cash_and_equivalents: float = Field(ge=0)
    short_term_debt: float = Field(default=0, ge=0)
    long_term_debt: float = Field(ge=0)
    operating_lease_liabilities: float = Field(default=0, ge=0)
    total_assets: float = Field(gt=0)
    total_equity: float
    accounts_receivable: float = Field(ge=0)
    inventory: float = Field(ge=0)
    accounts_payable: float = Field(ge=0)
    operating_cash_flow: float
    capex: float = Field(ge=0)
    dividends_paid: float = Field(default=0, ge=0)
    buybacks: float = Field(default=0, ge=0)
    provenance: DataProvenance
    source_accession: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ebitda(self) -> float:
        return self.operating_income + self.depreciation_amortization

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_debt(self) -> float:
        return self.short_term_debt + self.long_term_debt

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash_and_equivalents

    @computed_field  # type: ignore[prop-decorator]
    @property
    def free_cash_flow(self) -> float:
        return self.operating_cash_flow - self.capex

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_tax_rate(self) -> float | None:
        r = _div(self.income_tax_expense, self.pretax_income)
        return None if r is None else min(max(r, 0.0), 0.5)

    # Working-capital cycle (days), on annual flows.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def dso(self) -> float | None:
        r = _div(self.accounts_receivable, self.revenue)
        return None if r is None else r * 365

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dio(self) -> float | None:
        r = _div(self.inventory, self.cogs)
        return None if r is None else r * 365

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dpo(self) -> float | None:
        r = _div(self.accounts_payable, self.cogs)
        return None if r is None else r * 365

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cash_conversion_cycle(self) -> float | None:
        if None in (self.dso, self.dio, self.dpo):
            return None
        return self.dso + self.dio - self.dpo  # type: ignore[operator]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def operating_margin(self) -> float | None:
        return _div(self.operating_income, self.revenue)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def interest_coverage(self) -> float | None:
        if self.interest_expense == 0:
            return None
        return self.operating_income / self.interest_expense

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_debt_to_ebitda(self) -> float | None:
        return _div(self.net_debt, self.ebitda)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def roic(self) -> float | None:
        invested = self.total_debt + self.total_equity - self.cash_and_equivalents
        tax = self.effective_tax_rate if self.effective_tax_rate is not None else 0.21
        return _div(self.operating_income * (1 - tax), invested)


class MarketSnapshot(StrictModel):
    issuer_cik: str
    ticker: str
    as_of: date
    price: float = Field(gt=0)
    shares_outstanding: float = Field(gt=0)
    dividend_per_share_annual: float = Field(default=0, ge=0)
    beta: float = 1.0
    return_1y: float | None = None  # total shareholder return, decimal
    return_3y: float | None = None
    credit_rating: str | None = None  # S&P-style long-term issuer rating
    provenance: DataProvenance

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_cap(self) -> float:
        return self.price * self.shares_outstanding

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dividend_yield(self) -> float:
        return self.dividend_per_share_annual / self.price


class ConsensusEstimate(StrictModel):
    issuer_cik: str
    ticker: str
    as_of: date
    fiscal_year: int
    eps_mean: float
    eps_high: float | None = None
    eps_low: float | None = None
    revenue_mean: float | None = None
    num_analysts: int = 0
    rating_mean: float | None = None  # 1 = strong buy ... 5 = strong sell
    price_target_mean: float | None = None
    provenance: DataProvenance


class MacroSnapshot(StrictModel):
    as_of: date
    risk_free_rate: float  # 10Y UST, decimal
    equity_risk_premium: float = 0.05
    # option-adjusted spreads (decimal) by rating bucket, e.g. {"AA": 0.006, ...}
    credit_spreads: dict[str, float]
    marginal_tax_rate: float = 0.25
    provenance: DataProvenance


class CompanyValuation(StrictModel):
    """Joined fundamentals + market data with valuation multiples."""

    issuer_cik: str
    ticker: str
    name: str
    fundamentals: FinancialStatementSnapshot
    market: MarketSnapshot

    @computed_field  # type: ignore[prop-decorator]
    @property
    def enterprise_value(self) -> float:
        return self.market.market_cap + self.fundamentals.net_debt

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ev_to_ebitda(self) -> float | None:
        return _div(self.enterprise_value, self.fundamentals.ebitda)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pe(self) -> float | None:
        eps = self.fundamentals.diluted_eps or _div(self.fundamentals.net_income, self.fundamentals.shares_diluted)
        return _div(self.market.price, eps)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fcf_yield(self) -> float | None:
        return _div(self.fundamentals.free_cash_flow, self.market.market_cap)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def shareholder_yield(self) -> float | None:
        return _div(self.fundamentals.dividends_paid + self.fundamentals.buybacks, self.market.market_cap)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def debt_to_capital_market(self) -> float | None:
        return _div(self.fundamentals.total_debt, self.fundamentals.total_debt + self.market.market_cap)


class PeerStatistic(StrictModel):
    metric: str
    target_value: float | None
    peer_values: dict[str, float | None]
    peer_median: float | None
    peer_p25: float | None
    peer_p75: float | None
    z_score: float | None
    percentile_rank: float | None  # 0..1, target vs peers
    higher_is_better: bool
    as_of: date | None = None
