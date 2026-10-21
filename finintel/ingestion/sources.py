"""Data sources that assemble a :class:`CorpusBundle`, the normalized multimodal input
to the RAG index, the knowledge graph and the advisory engine.

* :class:`FixtureSource`: offline, deterministic, no keys (tests + demo).
* :class:`LiveEdgarSource`: SEC EDGAR (filings, XBRL, 13F, Form 4) + Stooq + FRED.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import resources
from pathlib import Path
from typing import Iterable

from ..entities.resolver import EntityRegistry
from ..schemas.common import Accession, DataProvenance, FiscalPeriod, FormType, InstitutionType, SecurityType
from ..schemas.entities import (
    CompetitorRelationship,
    Institution,
    Issuer,
    Security,
    SupplyRelationship,
    normalize_cik,
    normalize_ticker,
)
from ..schemas.filings import RawFiling
from ..schemas.holdings import FloatSnapshot, HoldingRecord, InsiderTransaction
from ..schemas.metrics import ConsensusEstimate, FinancialStatementSnapshot, MacroSnapshot, MarketSnapshot
from .market_client import interpolate_spread_curve
from .parsers import parse_13f_infotable, parse_form4

log = logging.getLogger(__name__)


@dataclass
class CorpusBundle:
    as_of: date
    registry: EntityRegistry
    institutions: dict[str, Institution]
    holdings: list[HoldingRecord]
    insider_transactions: list[InsiderTransaction]
    fundamentals: dict[str, FinancialStatementSnapshot]  # by CIK
    market: dict[str, MarketSnapshot]  # by CIK
    consensus: dict[str, ConsensusEstimate]
    floats: dict[str, FloatSnapshot]
    macro: MacroSnapshot
    filings: list[RawFiling]
    competitors: list[CompetitorRelationship]
    suppliers: list[SupplyRelationship]
    is_synthetic: bool = False
    warnings: list[str] = field(default_factory=list)

    def cik(self, ticker: str) -> str:
        iss = self.registry.issuer_by_ticker(ticker)
        if iss is None:
            raise KeyError(f"unknown ticker {ticker}")
        return iss.cik

    def ticker(self, cik: str) -> str:
        return self.registry.ticker_for(cik) or cik

    def name(self, cik: str) -> str:
        iss = self.registry.issuers.get(normalize_cik(cik))
        return iss.name if iss else cik

    def holding_periods(self) -> list[date]:
        return sorted({h.report_period for h in self.holdings})

    def filings_for(self, cik: str, forms: Iterable[FormType] | None = None) -> list[RawFiling]:
        cik = normalize_cik(cik)
        fs = [f for f in self.filings if normalize_cik(f.cik) == cik]
        if forms is not None:
            allowed = set(forms)
            fs = [f for f in fs if f.form_type in allowed]
        return sorted(fs, key=lambda f: f.filed_date)


# =========================================================================== fixtures


def _fixture_dir() -> Path:
    return Path(str(resources.files("finintel") / "data" / "fixtures"))


class FixtureSource:
    """Loads the bundled offline dataset (see ``scripts/build_fixtures.py``)."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else _fixture_dir()

    def _j(self, name: str):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def load(self) -> CorpusBundle:
        prov = DataProvenance(source="fixture", is_synthetic=True, note="offline fixture; see scripts/build_fixtures.py")
        ent = self._j("entities.json")
        registry = EntityRegistry().bulk_load(
            (Issuer(**{**i, "tickers": tuple(i["tickers"]), "aliases": tuple(i.get("aliases", ()))}) for i in ent["issuers"]),
            (Security(**{**s, "security_type": SecurityType(s["security_type"])}) for s in ent["securities"]),
        )
        institutions = {
            normalize_cik(i["cik"]): Institution(**{**i, "institution_type": InstitutionType(i["institution_type"])})
            for i in ent["institutions"]
        }
        holdings: list[HoldingRecord] = []
        for f in self._j("thirteenf.json"):
            holdings.extend(
                parse_13f_infotable(
                    f["xml"], f["filer_cik"], f["filer_name"], f["accession"],
                    date.fromisoformat(f["report_period"]), date.fromisoformat(f["filed_date"]),
                    registry=registry, value_in_thousands=False, is_synthetic=True,
                )
            )
        insiders: list[InsiderTransaction] = []
        from .parsers import parse_form4

        for f in self._j("form4.json"):
            insiders.extend(parse_form4(f["xml"], f["accession"], date.fromisoformat(f["filed_date"]), is_synthetic=True))

        fundamentals: dict[str, FinancialStatementSnapshot] = {}
        for r in self._j("fundamentals.json"):
            r = dict(r)
            fy, pe = r.pop("fiscal_year"), date.fromisoformat(r.pop("period_end"))
            cik = normalize_cik(r.pop("issuer_cik"))
            fundamentals[cik] = FinancialStatementSnapshot(
                issuer_cik=cik, period=FiscalPeriod(fiscal_year=fy, fiscal_quarter=0, period_end=pe),
                provenance=prov.model_copy(update={"note": "approximate figures hand-entered from public 10-K; verify with --live"}),
                source_accession=f"FIXTURE-{r['ticker']}-10K-FY{fy}", **r,
            )
        market = {
            normalize_cik(m["issuer_cik"]): MarketSnapshot(**{**m, "issuer_cik": normalize_cik(m["issuer_cik"]), "as_of": date.fromisoformat(m["as_of"])}, provenance=prov)
            for m in self._j("market.json")
        }
        consensus = {
            normalize_cik(c["issuer_cik"]): ConsensusEstimate(**{**c, "issuer_cik": normalize_cik(c["issuer_cik"]), "as_of": date.fromisoformat(c["as_of"])}, provenance=prov)
            for c in self._j("consensus.json")
        }
        floats = {}
        for f in self._j("float.json"):
            f = dict(f)
            f.pop("ticker")
            floats[normalize_cik(f["issuer_cik"])] = FloatSnapshot(**{**f, "issuer_cik": normalize_cik(f["issuer_cik"]), "as_of": date.fromisoformat(f["as_of"])}, provenance=prov)
        mj = self._j("macro.json")
        macro = MacroSnapshot(
            as_of=date.fromisoformat(mj["as_of"]), risk_free_rate=mj["risk_free_rate"], equity_risk_premium=mj["equity_risk_premium"],
            marginal_tax_rate=mj["marginal_tax_rate"], credit_spreads=interpolate_spread_curve(mj["credit_spreads_bucket"]), provenance=prov,
        )
        fm = self._j("filings.json")
        filings = []
        for d in fm["documents"]:
            content = (self.root / "filings" / d["file"]).read_text(encoding="utf-8")
            filings.append(
                RawFiling(
                    accession=Accession(value=d["accession"]), cik=normalize_cik(d["cik"]), company_name=d["company"], ticker=d["ticker"],
                    form_type=FormType(d["form"]), filed_date=date.fromisoformat(d["filed"]),
                    period=FiscalPeriod(fiscal_year=d["fy"], fiscal_quarter=d["fq"], period_end=date.fromisoformat(d["period_end"])),
                    content_type=d["content_type"], content=content, provenance=prov,
                )
            )
        competitors = [CompetitorRelationship(company_cik=normalize_cik(c["company_cik"]), peer_cik=normalize_cik(c["peer_cik"]), strength=c["strength"], basis=c["basis"]) for c in ent["competitors"]]
        suppliers = [SupplyRelationship(supplier_cik=normalize_cik(s["supplier_cik"]), customer_cik=normalize_cik(s["customer_cik"]), revenue_share_of_supplier=s["revenue_share_of_supplier"], note=s["note"]) for s in ent["suppliers"]]
        return CorpusBundle(
            as_of=date.fromisoformat(fm["as_of"]), registry=registry, institutions=institutions, holdings=holdings,
            insider_transactions=insiders, fundamentals=fundamentals, market=market, consensus=consensus, floats=floats,
            macro=macro, filings=filings, competitors=competitors, suppliers=suppliers, is_synthetic=True,
        )


# =========================================================================== live

# Default 13F universe. CIKs should be verified against EDGAR before production use; any
# filer that fails to resolve is skipped with a warning. Override with --institutions.
DEFAULT_13F_UNIVERSE: dict[str, tuple[str, InstitutionType, bool]] = {
    "102909": ("Vanguard Group Inc", InstitutionType.INDEX_MANAGER, False),
    "93751": ("State Street Corp", InstitutionType.INDEX_MANAGER, False),
    "2012383": ("BlackRock, Inc.", InstitutionType.INDEX_MANAGER, False),
    "1364742": ("BlackRock Inc. (legacy filer)", InstitutionType.INDEX_MANAGER, False),
    "1214717": ("Geode Capital Management", InstitutionType.INDEX_MANAGER, False),
    "1067983": ("Berkshire Hathaway Inc", InstitutionType.ACTIVE_MANAGER, False),
    "1037389": ("Renaissance Technologies", InstitutionType.HEDGE_FUND, False),
    "1336528": ("Pershing Square Capital Management", InstitutionType.ACTIVIST, True),
    "1345471": ("Trian Fund Management", InstitutionType.ACTIVIST, True),
    "1517137": ("Starboard Value LP", InstitutionType.ACTIVIST, True),
}

# Generic BBB-centred spread curve used only if FRED is unreachable.
FALLBACK_SPREADS = {"AAA": 0.0045, "AA": 0.0060, "A": 0.0085, "BBB": 0.0120, "BB": 0.0220, "B": 0.0350, "CCC": 0.0900}


class LiveEdgarSource:
    def __init__(
        self,
        tickers: list[str],
        user_agent: str,
        as_of: date | None = None,
        cache_dir: str | Path | None = ".finintel_cache",
        institutions: dict[str, tuple[str, InstitutionType, bool]] | None = None,
        quarters_of_13f: int = 5,
        transcripts_dir: str | Path | None = None,
        cusips: dict[str, str] | None = None,
        credit_ratings: dict[str, str] | None = None,
    ) -> None:
        from .edgar_client import EdgarClient

        self.tickers = [normalize_ticker(t) for t in tickers]
        self.as_of = as_of or date.today()
        self.edgar = EdgarClient(user_agent=user_agent, cache_dir=cache_dir)
        self.universe = institutions or DEFAULT_13F_UNIVERSE
        self.quarters_of_13f = quarters_of_13f
        self.transcripts_dir = Path(transcripts_dir) if transcripts_dir else None
        self.cusips = {normalize_ticker(k): v.upper() for k, v in (cusips or {}).items()}
        self.ratings = {normalize_ticker(k): v for k, v in (credit_ratings or {}).items()}

    def load(self) -> CorpusBundle:
        from .edgar_client import FilingRef
        from .market_client import FredClient, StooqPriceProvider, beta_from_history, total_return
        from .parsers import company_facts_to_snapshot

        warnings: list[str] = []
        registry = EntityRegistry()
        fundamentals: dict[str, FinancialStatementSnapshot] = {}
        market: dict[str, MarketSnapshot] = {}
        filings: list[RawFiling] = []
        insiders: list[InsiderTransaction] = []
        floats: dict[str, FloatSnapshot] = {}
        stooq = StooqPriceProvider()
        try:
            spy = stooq.history("SPY", self.as_of - timedelta(days=400), self.as_of)
        except Exception as exc:  # noqa: BLE001
            spy = []
            warnings.append(f"SPY history unavailable ({exc}); betas default to 1.0")

        for t in self.tickers:
            cik = self.edgar.cik_for_ticker(t)
            sub = self.edgar.submissions(cik)
            registry.add_issuer(Issuer(cik=cik, name=sub.get("name", t), tickers=tuple(sub.get("tickers") or [t]), sic=str(sub.get("sic") or "") or None,
                                       industry=sub.get("sicDescription"), fiscal_year_end=sub.get("fiscalYearEnd")))
            if t in self.cusips:
                registry.add_security(Security(cusip=self.cusips[t], ticker=t, issuer_cik=cik, security_type=SecurityType.COMMON_EQUITY))
            try:
                fundamentals[cik] = company_facts_to_snapshot(self.edgar.company_facts(cik), t, as_of=self.as_of)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{t}: XBRL fundamentals unavailable: {exc}")
            # Text filings.
            for forms, limit in (({"10-K"}, 2), ({"10-Q"}, 1), ({"8-K"}, 3)):
                for ref in self.edgar.list_filings(cik, forms, until=self.as_of, limit=limit):
                    try:
                        fy = (ref.report_date or ref.filed).year
                        fq = 0 if ref.form.startswith("10-K") else (((ref.report_date or ref.filed).month - 1) // 3 + 1)
                        period = FiscalPeriod(fiscal_year=fy, fiscal_quarter=min(fq, 4), period_end=ref.report_date or ref.filed)
                        filings.append(self.edgar.fetch_filing(ref, sub.get("name", t), t, period))
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"{t} {ref.form} {ref.accession}: {exc}")
            # Form 4 (last 180 days).
            for ref in self.edgar.list_filings(cik, {"4"}, since=self.as_of - timedelta(days=180), until=self.as_of, limit=60):
                try:
                    url, xml = self.edgar.form4_xml(ref)
                    insiders.extend(t_ for t_ in parse_form4(xml, ref.accession, ref.filed, url) if t_.issuer_cik == cik)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{t} Form 4 {ref.accession}: {exc}")
            # Market data.
            try:
                hist = stooq.history(t, self.as_of - timedelta(days=400), self.as_of)
                price = hist[-1][1]
                one_year = [h for h in hist if h[0] >= self.as_of - timedelta(days=365)]
                f = fundamentals.get(cik)
                dps = (f.dividends_paid / f.shares_diluted) if f else 0.0
                beta = beta_from_history(hist, spy) if spy else 1.0
                shares_out = f.shares_diluted if f else 1.0
                market[cik] = MarketSnapshot(
                    issuer_cik=cik, ticker=t, as_of=hist[-1][0], price=price, shares_outstanding=shares_out, dividend_per_share_annual=dps,
                    beta=beta, return_1y=total_return(one_year, dps), credit_rating=self.ratings.get(t),
                    provenance=DataProvenance(source="stooq", url=StooqPriceProvider.BASE, retrieved_at=date.today()),
                )
                floats[cik] = FloatSnapshot(issuer_cik=cik, as_of=hist[-1][0], shares_outstanding=shares_out,
                                            provenance=DataProvenance(source="sec_edgar+stooq", retrieved_at=date.today()))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{t}: market data unavailable: {exc}")
            if self.transcripts_dir:
                for p in sorted(self.transcripts_dir.glob(f"{t}_*.txt")):
                    filings.append(RawFiling(
                        accession=Accession(value=f"FIXTURE-{t}-CALL-{p.stem.split('_', 1)[-1].upper()}"), cik=cik, company_name=sub.get("name", t), ticker=t,
                        form_type=FormType.TRANSCRIPT, filed_date=date.fromtimestamp(p.stat().st_mtime), content_type="text", content=p.read_text(encoding="utf-8"),
                        provenance=DataProvenance(source="local_transcript", url=str(p), retrieved_at=date.today()),
                    ))

        # 13F: walk the institution universe, keep rows that resolve into the peer set.
        institutions: dict[str, Institution] = {}
        holdings: list[HoldingRecord] = []
        wanted = {registry.issuer_by_ticker(t).cik for t in self.tickers if registry.issuer_by_ticker(t)}  # type: ignore[union-attr]
        for fcik, (fname, ftype, activist) in self.universe.items():
            try:
                refs = self.edgar.list_filings(fcik, {"13F-HR"}, until=self.as_of, limit=self.quarters_of_13f + 2)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"13F filer {fcik} ({fname}) unavailable: {exc}")
                continue
            seen_periods: set[date] = set()
            institutions[normalize_cik(fcik)] = Institution(cik=fcik, name=fname, institution_type=ftype, is_known_activist=activist)
            for ref in refs:
                period = ref.report_date or ref.filed
                if period in seen_periods or len(seen_periods) >= self.quarters_of_13f:
                    continue
                try:
                    url, xml = self.edgar.thirteen_f_infotable_xml(ref)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"13F {fname} {ref.accession}: {exc}")
                    continue
                seen_periods.add(period)
                rows = parse_13f_infotable(xml, fcik, fname, ref.accession, period, ref.filed, registry=registry, source_url=url)
                for r in rows:
                    if r.issuer_cik in wanted:
                        # Learn CUSIPs from confident name matches so later rows resolve by CUSIP.
                        if r.cusip not in registry.securities_by_cusip and r.title_of_class.upper().startswith(("COM", "CL ")):
                            registry.add_security(Security(cusip=r.cusip, issuer_cik=r.issuer_cik, security_type=r.security_type,
                                                           ticker=registry.ticker_for(r.issuer_cik) if r.security_type == SecurityType.COMMON_EQUITY else None))
                        holdings.append(r)

        try:
            macro = FredClient().macro_snapshot(self.as_of)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"FRED unavailable ({exc}); using fallback spread curve and 4.25% risk-free rate")
            macro = MacroSnapshot(as_of=self.as_of, risk_free_rate=0.0425, credit_spreads=interpolate_spread_curve(FALLBACK_SPREADS),
                                  provenance=DataProvenance(source="fallback", note="FRED unreachable"))
        competitors = [
            CompetitorRelationship(company_cik=registry.issuer_by_ticker(self.tickers[0]).cik, peer_cik=registry.issuer_by_ticker(p).cik, basis="peer_set")  # type: ignore[union-attr]
            for p in self.tickers[1:] if registry.issuer_by_ticker(p)
        ]
        _ = FilingRef  # imported for type clarity in this scope
        return CorpusBundle(
            as_of=self.as_of, registry=registry, institutions=institutions, holdings=holdings, insider_transactions=insiders,
            fundamentals=fundamentals, market=market, consensus={}, floats=floats, macro=macro, filings=filings,
            competitors=competitors, suppliers=[], is_synthetic=False, warnings=warnings,
        )
