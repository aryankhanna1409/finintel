"""Entity models: issuers, securities, institutions, and the relationships between them."""

from __future__ import annotations

from pydantic import Field, field_validator

from .common import FrozenModel, InstitutionType, SecurityType, StrictModel


def normalize_ticker(raw: str) -> str:
    """Canonical ticker form: upper case, class separator normalized to '.'.

    ``brk-b``, ``BRK/B``, ``BRK.B`` and ``BRK B`` all become ``BRK.B``.
    """
    t = raw.strip().upper().replace("/", ".").replace("-", ".").replace(" ", ".")
    while ".." in t:
        t = t.replace("..", ".")
    return t.strip(".")


def normalize_cik(raw: str | int) -> str:
    s = str(raw).strip().lstrip("0") or "0"
    if not s.isdigit():
        raise ValueError(f"CIK must be numeric: {raw!r}")
    return s.zfill(10)


class Security(FrozenModel):
    """A single tradable/holdable instrument. Equity classes and debt issues are distinct
    securities even when they share an issuer; this is what prevents subsidiary-debt vs
    parent-equity drift during ownership aggregation."""

    cusip: str | None = Field(default=None, min_length=9, max_length=9)
    ticker: str | None = None
    issuer_cik: str
    security_type: SecurityType
    share_class: str | None = None  # "A", "B", "C" ...
    votes_per_share: float = 1.0
    economic_conversion_ratio: float = 1.0  # e.g. BRK.A = 1500 BRK.B
    description: str = ""
    obligor_cik: str | None = None  # for debt issued by a financing subsidiary

    @field_validator("ticker")
    @classmethod
    def _norm_ticker(cls, v: str | None) -> str | None:
        return normalize_ticker(v) if v else v

    @field_validator("issuer_cik", "obligor_cik")
    @classmethod
    def _norm_cik(cls, v: str | None) -> str | None:
        return normalize_cik(v) if v else v


class Issuer(FrozenModel):
    cik: str
    name: str
    tickers: tuple[str, ...] = ()
    sic: str | None = None
    sector: str | None = None
    industry: str | None = None
    fiscal_year_end: str | None = None  # "MMDD"
    parent_cik: str | None = None  # set for subsidiaries / financing vehicles
    aliases: tuple[str, ...] = ()

    @field_validator("cik", "parent_cik")
    @classmethod
    def _norm_cik(cls, v: str | None) -> str | None:
        return normalize_cik(v) if v else v

    @field_validator("tickers")
    @classmethod
    def _norm_tickers(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(normalize_ticker(t) for t in v)

    @property
    def primary_ticker(self) -> str | None:
        return self.tickers[0] if self.tickers else None


class Institution(FrozenModel):
    """A 13F filer / fund manager."""

    cik: str
    name: str
    institution_type: InstitutionType = InstitutionType.OTHER
    is_known_activist: bool = False
    aum_usd: float | None = None
    style: str | None = None  # "value", "growth", "quant", "index", ...

    @field_validator("cik")
    @classmethod
    def _norm_cik(cls, v: str) -> str:
        return normalize_cik(v)


class SupplyRelationship(FrozenModel):
    supplier_cik: str
    customer_cik: str
    revenue_share_of_supplier: float | None = Field(default=None, ge=0, le=1)
    disclosed_in: str | None = None  # accession / source tag
    note: str | None = None


class CompetitorRelationship(FrozenModel):
    company_cik: str
    peer_cik: str
    basis: str = "disclosed"  # disclosed | sic | overlap | analyst
    strength: float = Field(default=1.0, ge=0, le=1)


class ResolvedEntity(StrictModel):
    """Output of the entity resolver: what a raw mention maps to, and how confident we are."""

    raw: str
    issuer_cik: str | None = None
    security: Security | None = None
    method: str = "unresolved"
    confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.issuer_cik is not None
