"""LiveEdgarSource end to end against a mocked network (EDGAR + Stooq + FRED)."""

from datetime import date, timedelta

import httpx
import pytest

from finintel.ingestion import LiveEdgarSource
from finintel.schemas import FormType, InstitutionType

TICKERS = {"0": {"cik_str": 27419, "ticker": "TGT", "title": "TARGET CORP"}, "1": {"cik_str": 104169, "ticker": "WMT", "title": "WALMART INC"}}


def _subs(name, rows):
    keys = ["accessionNumber", "form", "filingDate", "reportDate", "primaryDocument"]
    return {"name": name, "sic": "5331", "sicDescription": "Retail-Variety Stores", "tickers": [], "fiscalYearEnd": "0201",
            "filings": {"recent": {k: [r[i] for r in rows] for i, k in enumerate(keys)}}}


def _facts(cik, scale):
    def f(v, start=True, end="2025-02-01"):
        d = {"val": v * scale, "end": end, "form": "10-K", "fp": "FY", "filed": "2025-03-12", "accn": f"{cik:010d}-25-000010"}
        if start:
            d["start"] = "2024-02-04"
        return [d]
    inst = lambda v: f(v, start=False)  # noqa: E731
    g = {"Revenues": f(100), "CostOfRevenue": f(70), "OperatingIncomeLoss": f(6), "DepreciationDepletionAndAmortization": f(3), "InterestExpense": f(0.5),
         "IncomeTaxExpenseBenefit": f(1.2), "NetIncomeLoss": f(4.3), "WeightedAverageNumberOfDilutedSharesOutstanding": {"shares": f(0.5)},
         "CashAndCashEquivalentsAtCarryingValue": inst(4), "LongTermDebtNoncurrent": inst(14), "Assets": [dict(inst(57)[0], fp="FY")],
         "StockholdersEquity": inst(15), "InventoryNet": inst(12), "AccountsPayableCurrent": inst(13), "NetCashProvidedByUsedInOperatingActivities": f(7),
         "PaymentsToAcquirePropertyPlantAndEquipment": f(3), "PaymentsOfDividendsCommonStock": f(2)}
    facts = {k: ({"units": v} if isinstance(v, dict) else {"units": {"USD": v}}) for k, v in g.items()}
    return {"cik": cik, "facts": {"us-gaap": facts}}


TENK = "<html><body><p>Item 1A. Risk Factors</p><p>Tariffs could raise our costs.</p><p>Item 7. MD&amp;A</p><p>We remain committed to our middle-A ratings.</p></body></html>"
FORM4 = """<ownershipDocument><issuer><issuerCik>0000027419</issuerCik><issuerTradingSymbol>TGT</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerCik>0000000077</rptOwnerCik><rptOwnerName>Doe J</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner><nonDerivativeTable><nonDerivativeTransaction>
<transactionDate><value>2025-09-01</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>1000</value></transactionShares><transactionPricePerShare><value>90</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""
INFO = """<informationTable><infoTable><nameOfIssuer>TARGET CORP</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>87612E106</cusip><value>9000000</value>
<shrsOrPrnAmt><sshPrnamt>100000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable>
<infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>037833100</cusip><value>5</value>
<shrsOrPrnAmt><sshPrnamt>1</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt></infoTable></informationTable>"""


def _stooq_csv():
    start = date(2024, 9, 1)
    rows = ["Date,Open,High,Low,Close,Volume"]
    for i in range(0, 420):
        d = start + timedelta(days=i)
        px = 100 + 10 * ((i % 37) / 37) + i * 0.01
        rows.append(f"{d},{px},{px},{px},{px},1000")
    return "\n".join(rows)


def handler(req: httpx.Request) -> httpx.Response:
    u = str(req.url)
    if "company_tickers.json" in u:
        return httpx.Response(200, json=TICKERS)
    if "submissions/CIK0000027419" in u:
        return httpx.Response(200, json=_subs("TARGET CORP", [
            ("0000027419-25-000010", "10-K", "2025-03-12", "2025-02-01", "tgt-10k.htm"),
            ("0000027419-25-000099", "4", "2025-09-02", "", "xslF345X05/form4.xml"),
        ]))
    if "submissions/CIK0000104169" in u:
        return httpx.Response(200, json=_subs("WALMART INC", [("0000104169-25-000010", "10-K", "2025-03-14", "2025-01-31", "wmt-10k.htm")]))
    if "submissions/CIK0000102909" in u:
        return httpx.Response(200, json=_subs("VANGUARD GROUP INC", [("0000102909-25-000001", "13F-HR", "2025-08-14", "2025-06-30", "primary_doc.xml")]))
    if "submissions/" in u:
        return httpx.Response(404)
    if "companyfacts/CIK0000027419" in u:
        return httpx.Response(200, json=_facts(27419, 1e9))
    if "companyfacts/CIK0000104169" in u:
        return httpx.Response(200, json=_facts(104169, 6e9))
    if u.endswith(".htm"):
        return httpx.Response(200, text=TENK)
    if "000002741925000099/index.json" in u:
        return httpx.Response(200, json={"directory": {"item": [{"name": "form4.xml"}]}})
    if u.endswith("000002741925000099/form4.xml"):
        return httpx.Response(200, text=FORM4)
    if "000010290925000001/index.json" in u:
        return httpx.Response(200, json={"directory": {"item": [{"name": "primary_doc.xml"}, {"name": "infotable.xml"}]}})
    if u.endswith("infotable.xml"):
        return httpx.Response(200, text=INFO)
    if "stooq.com" in u:
        return httpx.Response(200, text=_stooq_csv())
    if "fred.stlouisfed.org" in u:
        sid = req.url.params["id"]
        val = "4.10" if sid == "DGS10" else {"BAMLC0A1CAAA": "0.40", "BAMLC0A2CAA": "0.50", "BAMLC0A3CA": "0.70", "BAMLC0A4CBBB": "1.00",
                                             "BAMLH0A1HYBB": "1.80", "BAMLH0A2HYB": "3.00", "BAMLH0A3HYC": "8.50"}[sid]
        return httpx.Response(200, text=f"observation_date,{sid}\n2025-10-14,.\n2025-10-15,{val}\n")
    return httpx.Response(404)


@pytest.fixture
def mocked_network(monkeypatch):
    real = httpx.Client.__init__

    def init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        real(self, *a, **kw)

    monkeypatch.setattr(httpx.Client, "__init__", init)


def test_live_source_assembles_bundle(mocked_network, tmp_path):
    src = LiveEdgarSource(["TGT", "WMT"], "finintel-tests test@example.com", as_of=date(2025, 10, 15), cache_dir=tmp_path,
                          institutions={"102909": ("Vanguard Group Inc", InstitutionType.INDEX_MANAGER, False), "999": ("Missing Filer", InstitutionType.OTHER, False)},
                          cusips={"TGT": "87612E106"}, credit_ratings={"TGT": "A"})
    b = src.load()
    tgt, wmt = b.cik("TGT"), b.cik("WMT")
    assert b.fundamentals[tgt].revenue == 100e9 and b.fundamentals[wmt].revenue == 600e9
    assert b.fundamentals[tgt].capex == 3e9
    assert {f.form_type for f in b.filings} == {FormType.TEN_K}
    assert b.filings[0].provenance.source == "sec_edgar" and not b.filings[0].accession.is_fixture
    assert len(b.insider_transactions) == 1 and b.insider_transactions[0].value_usd == 90_000
    assert [h.issuer_cik for h in b.holdings] == [tgt]  # AAPL row filtered out, TGT resolved by CUSIP
    assert b.holdings[0].value_usd == 9_000_000  # post-2023 whole dollars
    assert b.market[tgt].credit_rating == "A" and b.market[tgt].price > 0
    assert b.macro.risk_free_rate == pytest.approx(0.041) and b.macro.credit_spreads["BBB"] == pytest.approx(0.01)
    assert any("Missing Filer" in w for w in b.warnings)
    assert not b.is_synthetic
