import json
from datetime import date

import httpx
import pytest

from finintel.ingestion.edgar_client import EdgarClient, EdgarError, FilingRef
from finintel.ingestion.market_client import interpolate_spread_curve
from finintel.ingestion.parsers import (
    company_facts_to_snapshot,
    parse_13f_infotable,
    parse_document,
    parse_form4,
)
from finintel.schemas import Accession, DataProvenance, FiscalPeriod, FormType, RawFiling, SecurityType

INFO = """<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
<infoTable><nameOfIssuer>TARGET CORP</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>87612E106</cusip><value>1000</value>
<shrsOrPrnAmt><sshPrnamt>10</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion>
<votingAuthority><Sole>10</Sole><Shared>0</Shared><None>0</None></votingAuthority></infoTable>
<infoTable><nameOfIssuer>TARGET CORP</nameOfIssuer><titleOfClass>COM</titleOfClass><cusip>87612E106</cusip><value>500</value>
<shrsOrPrnAmt><sshPrnamt>5</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt><putCall>Put</putCall><investmentDiscretion>SOLE</investmentDiscretion>
<votingAuthority><Sole>0</Sole><Shared>0</Shared><None>0</None></votingAuthority></infoTable>
<infoTable><nameOfIssuer>TARGET CORP</nameOfIssuer><titleOfClass>NOTE 4.5% 2034</titleOfClass><cusip>87612EBM7</cusip><value>900</value>
<shrsOrPrnAmt><sshPrnamt>1000</sshPrnamt><sshPrnamtType>PRN</sshPrnamtType></shrsOrPrnAmt><investmentDiscretion>SOLE</investmentDiscretion>
<votingAuthority><Sole>0</Sole><Shared>0</Shared><None>0</None></votingAuthority></infoTable>
</informationTable>"""


def test_13f_value_units_and_security_typing(bundle):
    new = parse_13f_infotable(INFO, "1", "F", "0000000001-25-000001", date(2025, 6, 30), date(2025, 8, 14), registry=bundle.registry)
    old = parse_13f_infotable(INFO, "1", "F", "0000000001-22-000001", date(2022, 6, 30), date(2022, 8, 14), registry=bundle.registry)
    assert new[0].value_usd == 1000 and old[0].value_usd == 1_000_000  # pre-2023 values were in $000s
    assert new[0].counts_toward_equity_ownership and new[0].issuer_cik == bundle.cik("TGT")
    assert new[1].security_type == SecurityType.OPTION_PUT and not new[1].counts_toward_equity_ownership
    assert new[2].security_type == SecurityType.SENIOR_DEBT and not new[2].counts_toward_equity_ownership


def test_form4_parsing_and_10b5_1_flag(bundle):
    tx = bundle.insider_transactions
    planned = [t for t in tx if t.insider_name.startswith("EVP Merchandising")]
    assert planned and planned[0].is_10b5_1 and planned[0].code.value == "S"
    buys = [t for t in tx if t.ticker == "TGT" and t.code.value == "P"]
    assert len(buys) == 4 and all(t.value_usd > 0 for t in buys)
    assert all(t.filing_lag_days <= 2 for t in buys)


def test_form4_raw_xml():
    xml = """<ownershipDocument><issuer><issuerCik>0000027419</issuerCik><issuerTradingSymbol>TGT</issuerTradingSymbol></issuer>
    <reportingOwner><reportingOwnerId><rptOwnerCik>0000000009</rptOwnerCik><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>
    <nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2025-09-01</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding><transactionAmounts><transactionShares><value>100</value></transactionShares>
    <transactionPricePerShare><value>90</value></transactionPricePerShare><transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
    </nonDerivativeTransaction><nonDerivativeTransaction><transactionDate><value>2025-09-02</value></transactionDate>
    <transactionCoding><transactionCode>Z</transactionCode></transactionCoding><transactionAmounts><transactionShares><value>1</value></transactionShares>
    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""
    out = parse_form4(xml, "0000000009-25-000001", date(2025, 9, 2))
    assert len(out) == 1  # unknown code skipped
    assert out[0].is_director and out[0].insider_role == "Director" and out[0].value_usd == 9000


def _raw(content, form=FormType.TEN_K, ctype="html"):
    return RawFiling(accession=Accession(value="FIXTURE-T-1"), cik="1", company_name="Co", ticker="CO", form_type=form, filed_date=date(2025, 3, 1),
                     period=FiscalPeriod(fiscal_year=2024, fiscal_quarter=0, period_end=date(2025, 2, 1)), content_type=ctype, content=content,
                     provenance=DataProvenance(source="fixture"))


def test_html_sections_toc_dedup_tables_and_hidden_xbrl():
    html = """<html><body><div style="display:none"><span>hidden xbrl junk Item 7. Fake</span></div>
    <table><tr><td>Item 1A.</td><td>Risk Factors</td></tr><tr><td>Item 7.</td><td>MD&amp;A</td></tr></table>
    <p>Item 1A. Risk Factors</p><p>Tariffs could raise costs.</p>
    <hr style="page-break-after:always"/>
    <p>Item 7. Management's Discussion</p><p>Results (in millions)</p>
    <table><tr><th>Metric</th><th>2024</th></tr><tr><td>Revenue</td><td>$</td><td>1,234</td></tr><tr><td>Loss</td><td>(</td><td>12</td><td>)</td></tr></table>
    </body></html>"""
    doc = parse_document(_raw(html))
    items = [s.item for s in doc.sections]
    assert "1A" in items and "7" in items
    assert all("hidden xbrl" not in b.text for s in doc.sections for b in s.blocks)
    mda = next(s for s in doc.sections if s.item == "7")
    table = next(b.table for b in mda.blocks if b.kind == "table")
    assert table.header == ["Metric", "2024"] and table.rows[0] == ["Revenue", "$1,234"] and table.rows[1] == ["Loss", "(12)"]
    assert table.units == "$ in millions"
    assert mda.page_start == 2


def test_transcript_parsing():
    txt = "=== PREPARED REMARKS ===\n[CEO | management]: Hello.\n=== Q&A ===\n[Analyst, X | analyst]: Q1?\n[CFO | management]: A1.\ncontinued answer\n"
    doc = parse_document(_raw(txt, FormType.TRANSCRIPT, "text"))
    qa = next(s for s in doc.sections if s.item == "Q&A")
    assert [b.speaker_role for b in qa.blocks] == ["analyst", "management"]
    assert qa.blocks[1].text.endswith("continued answer")


def test_xbrl_company_facts_mapping_point_in_time():
    def fact(val, end, start=None, filed="2025-03-12", fy=2024):
        d = {"val": val, "end": end, "form": "10-K", "fp": "FY", "filed": filed, "accn": f"0000027419-25-00000{fy % 10}", "fy": fy}
        if start:
            d["start"] = start
        return d

    def dur(v, fy_end="2025-02-01", start="2024-02-04", **kw):
        return [fact(v, fy_end, start, **kw)]

    usgaap = {
        "Revenues": {"units": {"USD": dur(100.0)}}, "CostOfRevenue": {"units": {"USD": dur(70.0)}}, "OperatingIncomeLoss": {"units": {"USD": dur(10.0)}},
        "DepreciationDepletionAndAmortization": {"units": {"USD": dur(3.0)}}, "InterestExpense": {"units": {"USD": dur(1.0)}},
        "IncomeTaxExpenseBenefit": {"units": {"USD": dur(2.0)}}, "NetIncomeLoss": {"units": {"USD": dur(7.0)}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": dur(1.4)}}, "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": dur(5.0)}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [fact(4.0, "2025-02-01")]}}, "LongTermDebtNoncurrent": {"units": {"USD": [fact(12.0, "2025-02-01")]}},
        "Assets": {"units": {"USD": [fact(50.0, "2025-02-01"), fact(60.0, "2026-01-31", filed="2026-03-10", fy=2025)]}},
        "StockholdersEquity": {"units": {"USD": [fact(20.0, "2025-02-01")]}}, "InventoryNet": {"units": {"USD": [fact(9.0, "2025-02-01")]}},
        "AccountsPayableCurrent": {"units": {"USD": [fact(8.0, "2025-02-01")]}}, "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": dur(9.0)}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": dur(-3.0)}},
    }
    snap = company_facts_to_snapshot({"cik": 27419, "facts": {"us-gaap": usgaap}}, "TGT", as_of=date(2025, 10, 1))
    assert snap.period.period_end == date(2025, 2, 1)  # FY2025 10-K (filed 2026) excluded by as_of
    assert snap.capex == 3.0 and snap.pretax_income == 9.0 and snap.ebitda == 13.0 and snap.short_term_debt == 0


def _mock_client(handler, tmp_path=None):
    return EdgarClient("finintel-tests test@example.com", cache_dir=tmp_path, rate_per_sec=1000, transport=httpx.MockTransport(handler))


def test_edgar_client_requires_contact_user_agent():
    with pytest.raises(ValueError):
        EdgarClient("no-email-agent")


def test_edgar_client_retries_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        assert "@" in req.headers["user-agent"]
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    c = _mock_client(handler, tmp_path)
    assert c.get_json("https://data.sec.gov/x.json") == {"ok": True}
    assert c.get_json("https://data.sec.gov/x.json") == {"ok": True}
    assert calls["n"] == 2  # second read served from disk cache


def test_edgar_client_gives_up(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    c = _mock_client(lambda r: httpx.Response(500))
    with pytest.raises(EdgarError):
        c.get_text("https://data.sec.gov/y")


def test_list_filings_and_13f_discovery():
    sub = {"name": "Fund", "filings": {"recent": {
        "accessionNumber": ["0000000001-25-000002", "0000000001-25-000001"], "form": ["13F-HR", "8-K"],
        "filingDate": ["2025-08-14", "2025-07-01"], "reportDate": ["2025-06-30", ""], "primaryDocument": ["primary_doc.xml", "x.htm"]}}}

    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if u.endswith("submissions/CIK0000000001.json"):
            return httpx.Response(200, json=sub)
        if u.endswith("index.json"):
            return httpx.Response(200, json={"directory": {"item": [{"name": "primary_doc.xml"}, {"name": "infotable.xml"}]}})
        if u.endswith("infotable.xml"):
            return httpx.Response(200, text=INFO)
        return httpx.Response(404)

    c = _mock_client(handler)
    refs = c.list_filings("1", {"13F-HR"})
    assert len(refs) == 1 and refs[0].report_date == date(2025, 6, 30)
    url, xml = c.thirteen_f_infotable_xml(refs[0])
    assert url.endswith("infotable.xml") and "informationTable" in xml
    assert FilingRef(accession="0000000001-25-000002", form="4", filed=date(2025, 1, 1), report_date=None, primary_document="a.xml", cik="0000000001").primary_url.endswith("/1/000000000125000002/a.xml")


def test_spread_curve_interpolation_monotone():
    curve = interpolate_spread_curve({"AAA": 0.004, "AA": 0.005, "A": 0.007, "BBB": 0.01, "BB": 0.018, "B": 0.03, "CCC": 0.085})
    vals = list(curve.values())
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert curve["A-"] > curve["A"] and curve["BBB+"] < curve["BBB"]


def test_fixture_bundle_is_flagged_synthetic(bundle):
    assert bundle.is_synthetic
    assert all(h.provenance.is_synthetic for h in bundle.holdings)
    assert all(f.accession.is_fixture for f in bundle.filings)
    assert json.loads(bundle.macro.model_dump_json())["risk_free_rate"] > 0
