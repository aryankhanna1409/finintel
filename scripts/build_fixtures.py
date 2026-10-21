"""Generate the offline fixture dataset used by tests and the offline demo.

WORKED EXAMPLE: Target Corp (TGT) benchmarked against Walmart (WMT), Costco (COST),
Dollar General (DG) and Kroger (KR).

Provenance and honesty rules for this dataset:
* Issuer CIKs, tickers and equity CUSIPs are the real identifiers.
* Fundamentals are APPROXIMATIONS of each company's most recent annual 10-K figures,
  rounded and hand-entered; market data is an approximate snapshot. They are good
  enough to exercise the analytics but must be refreshed via ``--live`` before use.
* Institutions, 13F positions, Form 4 insiders, filing excerpts and transcripts are
  SYNTHETIC. Institution names are fictional, CIKs live in a reserved 99xxxxxxxx range,
  accessions use the ``FIXTURE-`` namespace, and every record carries
  ``is_synthetic=True`` so nothing here can be cited as a real filing.

Run: ``python scripts/build_fixtures.py`` (output: finintel/data/fixtures/).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "finintel" / "data" / "fixtures"
FIL = OUT / "filings"

AS_OF = "2025-10-15"

# ----------------------------------------------------------------------------- issuers
ISSUERS = [
    {"cik": "27419", "name": "Target Corporation", "tickers": ["TGT"], "sic": "5331", "sector": "Consumer Staples", "industry": "Broadline Retail", "fiscal_year_end": "0201", "aliases": ["Target Corp", "TARGET CORP"]},
    {"cik": "104169", "name": "Walmart Inc.", "tickers": ["WMT"], "sic": "5331", "sector": "Consumer Staples", "industry": "Broadline Retail", "fiscal_year_end": "0131", "aliases": ["Wal-Mart Stores, Inc.", "WALMART INC"]},
    {"cik": "909832", "name": "Costco Wholesale Corporation", "tickers": ["COST"], "sic": "5331", "sector": "Consumer Staples", "industry": "Warehouse Clubs", "fiscal_year_end": "0901", "aliases": ["COSTCO WHSL CORP NEW", "Costco"]},
    {"cik": "29534", "name": "Dollar General Corporation", "tickers": ["DG"], "sic": "5331", "sector": "Consumer Staples", "industry": "Discount Stores", "fiscal_year_end": "0131", "aliases": ["DOLLAR GEN CORP NEW", "Dollar General"]},
    {"cik": "56873", "name": "The Kroger Co.", "tickers": ["KR"], "sic": "5411", "sector": "Consumer Staples", "industry": "Grocery", "fiscal_year_end": "0201", "aliases": ["KROGER CO", "Kroger"]},
    {"cik": "80424", "name": "The Procter & Gamble Company", "tickers": ["PG"], "sic": "2840", "sector": "Consumer Staples", "industry": "Household Products", "fiscal_year_end": "0630", "aliases": ["PROCTER AND GAMBLE CO", "P&G"]},
    {"cik": "77476", "name": "PepsiCo, Inc.", "tickers": ["PEP"], "sic": "2080", "sector": "Consumer Staples", "industry": "Beverages", "fiscal_year_end": "1231", "aliases": ["PEPSICO INC"]},
    {"cik": "55785", "name": "Kimberly-Clark Corporation", "tickers": ["KMB"], "sic": "2670", "sector": "Consumer Staples", "industry": "Household Products", "fiscal_year_end": "1231", "aliases": ["KIMBERLY CLARK CORP"]},
]

SECURITIES = [
    {"cusip": "87612E106", "ticker": "TGT", "issuer_cik": "27419", "security_type": "common_equity", "description": "Target Corp common stock"},
    {"cusip": "931142103", "ticker": "WMT", "issuer_cik": "104169", "security_type": "common_equity", "description": "Walmart Inc common stock"},
    {"cusip": "22160K105", "ticker": "COST", "issuer_cik": "909832", "security_type": "common_equity", "description": "Costco Wholesale common stock"},
    {"cusip": "256677105", "ticker": "DG", "issuer_cik": "29534", "security_type": "common_equity", "description": "Dollar General common stock"},
    {"cusip": "501044101", "ticker": "KR", "issuer_cik": "56873", "security_type": "common_equity", "description": "Kroger common stock"},
    {"cusip": "742718109", "ticker": "PG", "issuer_cik": "80424", "security_type": "common_equity", "description": "Procter & Gamble common stock"},
    {"cusip": "713448108", "ticker": "PEP", "issuer_cik": "77476", "security_type": "common_equity", "description": "PepsiCo common stock"},
    {"cusip": "494368103", "ticker": "KMB", "issuer_cik": "55785", "security_type": "common_equity", "description": "Kimberly-Clark common stock"},
    # Illustrative (not real) senior-note CUSIP under Target's issuer prefix: used to prove
    # debt rows never inflate equity ownership.
    {"cusip": "87612EBM7", "ticker": None, "issuer_cik": "27419", "security_type": "senior_debt", "description": "Target Corp 4.5% senior notes (illustrative CUSIP)"},
]

COMPETITORS = [
    ("27419", "104169", 1.0), ("27419", "909832", 0.8), ("27419", "29534", 0.7), ("27419", "56873", 0.6),
    ("104169", "909832", 0.9), ("104169", "29534", 0.7), ("104169", "56873", 0.8), ("909832", "56873", 0.5),
    ("29534", "56873", 0.4),
]

# Supplier -> customer with approximate customer revenue share of supplier (illustrative).
SUPPLIERS = [
    ("80424", "104169", 0.15), ("80424", "27419", 0.04), ("80424", "909832", 0.04), ("80424", "29534", 0.03), ("80424", "56873", 0.03),
    ("77476", "104169", 0.14), ("77476", "56873", 0.04), ("77476", "27419", 0.03),
    ("55785", "104169", 0.14), ("55785", "27419", 0.04), ("55785", "909832", 0.03),
]

# ----------------------------------------------------------------------------- fundamentals (USD millions, approx.)
M = 1_000_000.0
FUNDAMENTALS = {
    "TGT": dict(fy=2024, end="2025-02-01", revenue=106566, cogs=76502, operating_income=5566, depreciation_amortization=2981, interest_expense=411, income_tax_expense=1175, pretax_income=5266, net_income=4091, diluted_eps=8.86, shares_diluted=461.5, cash_and_equivalents=4762, short_term_debt=1636, long_term_debt=14304, operating_lease_liabilities=3600, total_assets=57769, total_equity=14666, accounts_receivable=800, inventory=12740, accounts_payable=13053, operating_cash_flow=7367, capex=2891, dividends_paid=2046, buybacks=1007),
    "WMT": dict(fy=2025, end="2025-01-31", revenue=680985, cogs=511753, operating_income=29348, depreciation_amortization=12973, interest_expense=2728, income_tax_expense=6152, pretax_income=26309, net_income=19436, diluted_eps=2.41, shares_diluted=8071, cash_and_equivalents=9037, short_term_debt=5600, long_term_debt=33401, operating_lease_liabilities=13900, total_assets=260823, total_equity=91013, accounts_receivable=9975, inventory=56435, accounts_payable=58666, operating_cash_flow=36443, capex=23783, dividends_paid=6688, buybacks=4494),
    "COST": dict(fy=2024, end="2024-09-01", revenue=254453, cogs=222358, operating_income=9285, depreciation_amortization=2237, interest_expense=169, income_tax_expense=2373, pretax_income=9740, net_income=7367, diluted_eps=16.56, shares_diluted=444.8, cash_and_equivalents=9906, short_term_debt=103, long_term_debt=5794, operating_lease_liabilities=2400, total_assets=69831, total_equity=23622, accounts_receivable=3203, inventory=18647, accounts_payable=19421, operating_cash_flow=11339, capex=4710, dividends_paid=8595, buybacks=700),
    "DG": dict(fy=2024, end="2025-01-31", revenue=40612, cogs=28803, operating_income=1714, depreciation_amortization=1030, interest_expense=283, income_tax_expense=330, pretax_income=1455, net_income=1125, diluted_eps=5.11, shares_diluted=220.0, cash_and_equivalents=932, short_term_debt=500, long_term_debt=6000, operating_lease_liabilities=11400, total_assets=31130, total_equity=7380, accounts_receivable=0, inventory=6857, accounts_payable=3428, operating_cash_flow=2980, capex=1310, dividends_paid=519, buybacks=0),
    "KR": dict(fy=2024, end="2025-02-01", revenue=147123, cogs=114200, operating_income=3849, depreciation_amortization=3200, interest_expense=620, income_tax_expense=670, pretax_income=3340, net_income=2665, diluted_eps=3.67, shares_diluted=723.0, cash_and_equivalents=3959, short_term_debt=900, long_term_debt=17100, operating_lease_liabilities=6800, total_assets=52600, total_equity=8000, accounts_receivable=2000, inventory=7000, accounts_payable=10500, operating_cash_flow=5800, capex=3600, dividends_paid=890, buybacks=5000),
}
PER_SHARE = {"diluted_eps"}
CIK_BY_TICKER = {"TGT": "27419", "WMT": "104169", "COST": "909832", "DG": "29534", "KR": "56873", "PG": "80424", "PEP": "77476", "KMB": "55785"}

MARKET = {
    # price, shares outstanding (M), annual DPS, beta, 1y TSR, 3y TSR, rating
    "TGT": (90.0, 455.0, 4.56, 1.10, -0.38, -0.36, "A"),
    "WMT": (103.0, 7975.0, 0.94, 0.65, 0.28, 1.25, "AA"),
    "COST": (935.0, 443.5, 5.20, 0.85, 0.05, 0.95, "A+"),
    "DG": (105.0, 220.0, 2.36, 0.45, 0.30, -0.52, "BBB"),
    "KR": (65.0, 662.0, 1.40, 0.45, 0.18, 0.55, "BBB"),
}

CONSENSUS = {
    "TGT": (2025, 7.35, 7.90, 6.95, 104200, 31, 2.9, 104.0),
    "WMT": (2026, 2.61, 2.70, 2.52, 703000, 38, 1.6, 112.0),
    "COST": (2026, 19.90, 20.60, 19.10, 275000, 30, 2.1, 1060.0),
    "DG": (2025, 6.05, 6.40, 5.70, 42100, 26, 2.5, 112.0),
    "KR": (2025, 4.75, 4.90, 4.60, 147500, 22, 2.3, 72.0),
}

FLOAT = {
    "TGT": (455.0, 1.2, 0.0, 11.4, 6.1),
    "WMT": (7975.0, 3600.0, 0.0, 45.0, 17.0),
    "COST": (443.5, 1.0, 0.0, 6.2, 2.1),
    "DG": (220.0, 0.6, 0.0, 7.9, 3.4),
    "KR": (662.0, 4.0, 0.0, 19.0, 6.8),
}

MACRO = {
    "as_of": AS_OF, "risk_free_rate": 0.0405, "equity_risk_premium": 0.050, "marginal_tax_rate": 0.25,
    "credit_spreads_bucket": {"AAA": 0.0040, "AA": 0.0050, "A": 0.0070, "BBB": 0.0100, "BB": 0.0180, "B": 0.0300, "CCC": 0.0850},
}

# ----------------------------------------------------------------------------- synthetic institutions & 13F
INSTITUTIONS = [
    {"cik": "9900000001", "name": "Meridian Index Advisors (synthetic)", "institution_type": "index_manager", "style": "index", "aum_usd": 9.0e12},
    {"cik": "9900000002", "name": "Bluewater Passive Funds (synthetic)", "institution_type": "index_manager", "style": "index", "aum_usd": 7.5e12},
    {"cik": "9900000003", "name": "Keystone Street Global (synthetic)", "institution_type": "index_manager", "style": "index", "aum_usd": 4.2e12},
    {"cik": "9900000004", "name": "Lakeshore Fundamental Equity (synthetic)", "institution_type": "active_manager", "style": "value", "aum_usd": 6.0e11},
    {"cik": "9900000005", "name": "Summit Growth Partners (synthetic)", "institution_type": "active_manager", "style": "growth", "aum_usd": 4.4e11},
    {"cik": "9900000006", "name": "Halcyon Ridge Capital (synthetic)", "institution_type": "hedge_fund", "style": "long_short", "aum_usd": 3.8e10},
    {"cik": "9900000007", "name": "Vector Quant Strategies (synthetic)", "institution_type": "hedge_fund", "style": "quant", "aum_usd": 6.1e10},
    {"cik": "9900000008", "name": "Granite Harbor Activist Partners (synthetic)", "institution_type": "activist", "style": "activist", "aum_usd": 1.9e10, "is_known_activist": True},
    {"cik": "9900000009", "name": "Pinecrest Teachers Retirement (synthetic)", "institution_type": "pension", "style": "blend", "aum_usd": 3.1e11},
    {"cik": "9900000010", "name": "Northgate Dividend Income (synthetic)", "institution_type": "active_manager", "style": "income", "aum_usd": 1.2e11},
    {"cik": "9900000011", "name": "Corvid Event-Driven Fund (synthetic)", "institution_type": "hedge_fund", "style": "event_driven", "aum_usd": 1.4e10},
    {"cik": "9900000012", "name": "Atlas Sovereign Wealth (synthetic)", "institution_type": "sovereign", "style": "blend", "aum_usd": 1.4e12},
]

PERIODS = {"T-4": ("2024-06-30", "2024-08-14"), "T-1": ("2025-03-31", "2025-05-15"), "T": ("2025-06-30", "2025-08-14")}
PRICES = {
    "T-4": {"TGT": 148.0, "WMT": 67.7, "COST": 850.0, "DG": 132.2, "KR": 49.9, "PG": 164.9, "PEP": 164.9, "KMB": 138.2},
    "T-1": {"TGT": 104.4, "WMT": 87.8, "COST": 945.8, "DG": 87.9, "KR": 67.7, "PG": 170.4, "PEP": 149.9, "KMB": 142.2},
    "T": {"TGT": 98.7, "WMT": 97.8, "COST": 989.9, "DG": 114.4, "KR": 71.7, "PG": 159.3, "PEP": 132.0, "KMB": 128.9},
}
SHARES_OUT = {"TGT": 455e6, "WMT": 8010e6, "COST": 443.5e6, "DG": 220e6, "KR": 662e6, "PG": 2345e6, "PEP": 1371e6, "KMB": 332e6}

# Fraction of shares outstanding held, per (institution index -> period -> ticker).
# Designed patterns:
#   * Halcyon Ridge (6) and Summit Growth (5) rotate out of TGT into COST/WMT.
#   * Vector Quant (7) rotates TGT -> DG.
#   * Granite Harbor (8), a known activist, builds a new TGT stake.
#   * Northgate Dividend (10) adds TGT for its yield.
#   * Index managers (1-3) are near-constant.
def _base(tgt, wmt, cost, dg, kr, pg=0.0, pep=0.0, kmb=0.0):
    return {"TGT": tgt, "WMT": wmt, "COST": cost, "DG": dg, "KR": kr, "PG": pg, "PEP": pep, "KMB": kmb}

POSITIONS = {
    "9900000001": {"T-4": _base(.093, .052, .089, .101, .095, .095, .094, .096), "T-1": _base(.094, .052, .090, .102, .096, .095, .094, .096), "T": _base(.095, .053, .090, .102, .096, .096, .095, .097)},
    "9900000002": {"T-4": _base(.078, .043, .074, .085, .080, .079, .078, .080), "T-1": _base(.078, .043, .075, .085, .080, .079, .078, .080), "T": _base(.079, .044, .075, .086, .081, .080, .079, .081)},
    "9900000003": {"T-4": _base(.045, .022, .041, .048, .046, .044, .043, .045), "T-1": _base(.045, .022, .041, .048, .046, .044, .043, .045), "T": _base(.045, .022, .041, .047, .046, .044, .043, .045)},
    "9900000004": {"T-4": _base(.021, .004, .006, .015, .018, .010, .012, .014), "T-1": _base(.019, .004, .006, .017, .019, .010, .012, .014), "T": _base(.018, .004, .006, .018, .019, .010, .012, .014)},
    "9900000005": {"T-4": _base(.024, .006, .019, .004, .002), "T-1": _base(.011, .008, .022, .004, .002), "T": _base(.004, .010, .025, .003, .002)},
    "9900000006": {"T-4": _base(.016, .001, .002, .000, .003), "T-1": _base(.009, .003, .005, .000, .003), "T": _base(.000, .004, .008, .000, .004)},
    "9900000007": {"T-4": _base(.006, .001, .001, .002, .004), "T-1": _base(.004, .001, .001, .006, .004), "T": _base(.001, .001, .001, .011, .005)},
    "9900000008": {"T-4": _base(.000, .000, .000, .000, .000), "T-1": _base(.003, .000, .000, .000, .000), "T": _base(.012, .000, .000, .000, .000)},
    "9900000009": {"T-4": _base(.005, .003, .004, .005, .005, .004, .004, .004), "T-1": _base(.005, .003, .004, .005, .005, .004, .004, .004), "T": _base(.005, .003, .004, .005, .005, .004, .004, .004)},
    "9900000010": {"T-4": _base(.008, .000, .000, .006, .009, .012, .011, .015), "T-1": _base(.011, .000, .000, .006, .009, .012, .011, .015), "T": _base(.014, .000, .000, .005, .009, .012, .011, .015)},
    "9900000011": {"T-4": _base(.000, .000, .000, .000, .006), "T-1": _base(.000, .000, .000, .002, .004), "T": _base(.002, .000, .000, .004, .000)},
    "9900000012": {"T-4": _base(.006, .004, .005, .004, .005, .004, .004, .004), "T-1": _base(.006, .004, .005, .004, .005, .004, .004, .004), "T": _base(.006, .004, .006, .004, .005, .004, .004, .004)},
}

CUSIP_BY_TICKER = {s["ticker"]: s["cusip"] for s in SECURITIES if s["ticker"]}
NAME_13F = {"TGT": "TARGET CORP", "WMT": "WALMART INC", "COST": "COSTCO WHSL CORP NEW", "DG": "DOLLAR GEN CORP NEW", "KR": "KROGER CO", "PG": "PROCTER AND GAMBLE CO", "PEP": "PEPSICO INC", "KMB": "KIMBERLY CLARK CORP"}


def info_table_xml(rows: list[dict]) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">']
    for r in rows:
        put_call = f"<putCall>{r['put_call']}</putCall>" if r.get("put_call") else ""
        parts.append(
            "<infoTable>"
            f"<nameOfIssuer>{escape(r['name'])}</nameOfIssuer><titleOfClass>{escape(r['title'])}</titleOfClass><cusip>{r['cusip']}</cusip>"
            f"<value>{int(round(r['value']))}</value>"
            f"<shrsOrPrnAmt><sshPrnamt>{int(round(r['shares']))}</sshPrnamt><sshPrnamtType>{r.get('type', 'SH')}</sshPrnamtType></shrsOrPrnAmt>"
            f"{put_call}<investmentDiscretion>SOLE</investmentDiscretion>"
            f"<votingAuthority><Sole>{int(round(r['shares'] * 0.9))}</Sole><Shared>0</Shared><None>{int(round(r['shares'] * 0.1))}</None></votingAuthority>"
            "</infoTable>"
        )
    parts.append("</informationTable>")
    return "".join(parts)


def build_13f() -> list[dict]:
    filings = []
    for inst in INSTITUTIONS:
        cik = inst["cik"]
        for label, (period, filed) in PERIODS.items():
            rows = []
            for tkr, frac in POSITIONS[cik][label].items():
                if frac <= 0:
                    continue
                shares = frac * SHARES_OUT[tkr]
                rows.append({"name": NAME_13F[tkr], "title": "COM", "cusip": CUSIP_BY_TICKER[tkr], "value": shares * PRICES[label][tkr], "shares": shares})
            # Entity-drift traps: a Target senior-note position (PRN) and TGT put options.
            if cik == "9900000004":
                rows.append({"name": "TARGET CORP", "title": "NOTE 4.500% 9/15/34", "cusip": "87612EBM7", "value": 48_000_000, "shares": 50_000_000, "type": "PRN"})
            if cik == "9900000006" and label == "T":
                rows.append({"name": "TARGET CORP", "title": "COM", "cusip": "87612E106", "value": 2_500_000 * PRICES[label]["TGT"], "shares": 2_500_000, "put_call": "Put"})
            filings.append({
                "filer_cik": cik, "filer_name": inst["name"], "accession": f"FIXTURE-13F-{cik[-3:]}-{period.replace('-', '')}",
                "report_period": period, "filed_date": filed, "relative_label": label, "xml": info_table_xml(rows),
            })
    return filings


# ----------------------------------------------------------------------------- synthetic Form 4
def form4_xml(issuer_cik, ticker, owner_cik, owner, is_dir, is_off, title, txns, plan=False, footnote=""):
    tx = []
    for d, code, shares, price, ad, after in txns:
        tx.append(
            "<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>"
            f"<transactionDate><value>{d}</value></transactionDate>"
            f"<transactionCoding><transactionFormType>4</transactionFormType><transactionCode>{code}</transactionCode><equitySwapInvolved>0</equitySwapInvolved></transactionCoding>"
            f"<transactionAmounts><transactionShares><value>{shares}</value></transactionShares><transactionPricePerShare><value>{price}</value></transactionPricePerShare>"
            f"<transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode></transactionAmounts>"
            f"<postTransactionAmounts><sharesOwnedFollowingTransaction><value>{after}</value></sharesOwnedFollowingTransaction></postTransactionAmounts>"
            "<ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature></nonDerivativeTransaction>"
        )
    fn = f"<footnotes><footnote id=\"F1\">{escape(footnote)}</footnote></footnotes>" if footnote else ""
    return (
        '<?xml version="1.0"?><ownershipDocument><schemaVersion>X0508</schemaVersion><documentType>4</documentType>'
        f"<aff10b5One>{1 if plan else 0}</aff10b5One>"
        f"<issuer><issuerCik>{issuer_cik.zfill(10)}</issuerCik><issuerName>x</issuerName><issuerTradingSymbol>{ticker}</issuerTradingSymbol></issuer>"
        f"<reportingOwner><reportingOwnerId><rptOwnerCik>{owner_cik}</rptOwnerCik><rptOwnerName>{escape(owner)}</rptOwnerName></reportingOwnerId>"
        f"<reportingOwnerRelationship><isDirector>{1 if is_dir else 0}</isDirector><isOfficer>{1 if is_off else 0}</isOfficer><isTenPercentOwner>0</isTenPercentOwner><officerTitle>{escape(title)}</officerTitle></reportingOwnerRelationship></reportingOwner>"
        f"<nonDerivativeTable>{''.join(tx)}</nonDerivativeTable>{fn}</ownershipDocument>"
    )


FORM4 = [
    # Target: cluster of open-market director purchases after the August print.
    ("27419", "TGT", "9910000001", "Director One (synthetic)", True, False, "", [("2025-08-25", "P", 5000, 94.10, "A", 21000)], False, "", "2025-08-26"),
    ("27419", "TGT", "9910000002", "Director Two (synthetic)", True, False, "", [("2025-08-27", "P", 3200, 92.85, "A", 9800)], False, "", "2025-08-28"),
    ("27419", "TGT", "9910000003", "Director Three (synthetic)", True, False, "", [("2025-09-03", "P", 10600, 91.40, "A", 30500)], False, "", "2025-09-04"),
    ("27419", "TGT", "9910000004", "Chief Financial Officer (synthetic)", False, True, "EVP & CFO", [("2025-09-10", "P", 2100, 89.75, "A", 64000)], False, "", "2025-09-11"),
    ("27419", "TGT", "9910000005", "EVP Merchandising (synthetic)", False, True, "EVP", [("2025-06-12", "S", 4000, 99.20, "D", 22000)], True, "Sale effected pursuant to a Rule 10b5-1 trading plan adopted 2024-12-05.", "2025-06-13"),
    ("27419", "TGT", "9910000006", "EVP Stores (synthetic)", False, True, "EVP", [("2025-04-03", "F", 1800, 104.0, "D", 30100)], False, "", "2025-04-04"),
    # Peers: routine planned sales, no clusters.
    ("909832", "COST", "9910000101", "EVP Operations (synthetic)", False, True, "EVP", [("2025-07-15", "S", 1500, 975.0, "D", 12000)], True, "Rule 10b5-1 plan.", "2025-07-16"),
    ("104169", "WMT", "9910000201", "EVP International (synthetic)", False, True, "EVP", [("2025-09-02", "S", 21000, 101.5, "D", 410000)], True, "Rule 10b5-1 plan.", "2025-09-03"),
    ("29534", "DG", "9910000301", "Director A (synthetic)", True, False, "", [("2025-06-10", "P", 1000, 102.4, "A", 5200)], False, "", "2025-06-11"),
    ("56873", "KR", "9910000401", "SVP Retail (synthetic)", False, True, "SVP", [("2025-07-01", "S", 8000, 71.6, "D", 60000)], False, "", "2025-07-02"),
]

# ----------------------------------------------------------------------------- filings
PB = '<hr style="page-break-after:always"/>'


def tbl(caption, header, rows):
    h = "".join(f"<th>{c}</th>" for c in header)
    body = ""
    for r in rows:
        cells = ""
        for c in r:
            if isinstance(c, str) and c.startswith("$"):
                cells += f"<td>$</td><td>{c[1:]}</td>"
            else:
                cells += f"<td>{c}</td>"
        body += f"<tr>{cells}</tr>"
    return f"<p>{caption}</p><table><tr>{h.replace('<th>', '<th>', 1)}</tr>{body}</table>"


def fin_table(t):
    f = FUNDAMENTALS[t]
    return tbl(
        f"Selected Consolidated Financial Data (in millions, except per share data)",
        ["Metric", f"Fiscal {f['fy']}"],
        [
            ["Net sales / total revenue", f"${f['revenue']:,}"],
            ["Cost of sales", f"${f['cogs']:,}"],
            ["Operating income", f"${f['operating_income']:,}"],
            ["Depreciation and amortization", f"${f['depreciation_amortization']:,}"],
            ["Net interest expense", f"${f['interest_expense']:,}"],
            ["Net earnings", f"${f['net_income']:,}"],
            ["Diluted earnings per share", f"${f['diluted_eps']:.2f}"],
            ["Inventory", f"${f['inventory']:,}"],
            ["Accounts payable", f"${f['accounts_payable']:,}"],
            ["Cash provided by operating activities", f"${f['operating_cash_flow']:,}"],
            ["Capital expenditures", f"${f['capex']:,}"],
            ["Share repurchases", f"${f['buybacks']:,}"],
        ],
    )


def tenk(company, fy, end, items):
    toc = "".join(f"<tr><td>Item {k}.</td><td>{v[0]}</td></tr>" for k, v in items.items())
    body = ""
    page = 1
    for k, (title, html) in items.items():
        body += f'{PB}<h2>Item {k}. {title}</h2>{html}'
        page += 1
    return (
        f"<html><body><div style=\"display:none\"><ix:header>hidden xbrl context</ix:header></div>"
        f"<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 10-K</p>"
        f"<p>{company}: fiscal year ended {end}</p>"
        f"<p>SYNTHETIC FIXTURE: illustrative excerpt written for the finintel demo; not the actual filing text.</p>"
        f"<table><tr><td>Item</td><td>Description</td></tr>{toc}</table>"
        f"<p>PART I</p>{body}</body></html>"
    )


TGT_RISK_FY2024 = """
<p>Our results depend on discretionary consumer demand. A significant share of our sales mix is in discretionary categories such as home, apparel and hardlines, and continued pressure on consumer budgets from elevated prices and interest rates reduced traffic and comparable sales during the year. Prolonged softness in discretionary spending would adversely affect our sales, margin mix and earnings.</p>
<p>Changes in trade policy, including tariffs on imported merchandise, could increase our costs. A meaningful portion of our owned-brand and national-brand assortment is sourced from outside the United States. New or increased tariffs may raise product costs, and we may be unable to fully offset them through vendor negotiations, sourcing changes or pricing without harming demand.</p>
<p>Our brand and reputation may be harmed by public reaction to our positions, merchandise decisions or business practices. Customer and advocacy-group responses, including calls for boycotts and negative social media attention, have in the past and could in the future reduce guest traffic and sales.</p>
<p>Inventory shrink, including theft and organized retail crime, has affected and may continue to affect our gross margin. While shrink moderated from the prior year, elevated levels would pressure profitability.</p>
<p>We face intense competition from multi-category retailers, warehouse clubs, dollar stores and digital marketplaces, many of which compete aggressively on price, assortment and delivery speed. Competitors with greater scale may be able to absorb cost increases more effectively.</p>
<p>A failure to manage inventory levels and assortment, including excess discretionary inventory, could require markdowns that lower our gross margin rate.</p>
<p>We rely on information systems and could be subject to a significant cybersecurity incident that compromises guest or team member data.</p>
<p>Our ability to execute our strategy depends on attracting and retaining key leaders. A change in senior leadership could disrupt execution of strategic initiatives.</p>
"""

TGT_RISK_FY2023 = """
<p>Our results depend on discretionary consumer demand, which may be affected by inflation and higher interest rates.</p>
<p>Inventory shrink, including theft and organized retail crime, increased significantly and reduced our gross margin; we closed several stores in part due to theft and safety concerns.</p>
<p>We face intense competition from multi-category retailers, warehouse clubs and digital marketplaces.</p>
<p>A failure to manage inventory levels could require markdowns.</p>
<p>We rely on information systems and could be subject to a cybersecurity incident.</p>
"""

TGT_MDA_FY2024 = f"""
<p>Comparable sales were approximately flat for the year as growth in digitally originated sales and same-day services offset softer store traffic in discretionary categories. Gross margin rate expanded on lower markdowns and moderating shrink, partially offset by higher digital fulfillment and supply chain costs.</p>
<p>Operating income margin rate was 5.2 percent. We returned cash to shareholders through dividends of approximately $2.0 billion and share repurchases of approximately $1.0 billion, and we remained committed to maintaining our middle-A credit ratings.</p>
{fin_table('TGT')}
<p>Our capital allocation priorities are, in order: investing in our business through capital expenditures, supporting the dividend, and returning excess cash through share repurchases over time while maintaining strong investment-grade ratings. Our long-term target is EBIT-based leverage consistent with those ratings.</p>
"""

TGT_10Q_Q2 = f"""
<p>Comparable sales decreased 1.9 percent in the second quarter, reflecting a decrease in comparable store traffic partially offset by growth in digital comparable sales. Discretionary categories, particularly home and apparel, remained soft, while food and essentials and beauty were relatively stable.</p>
<p>Gross margin rate decreased primarily due to higher markdowns and merchandising costs, partially offset by lower supply chain costs. Tariff-related cost increases are expected to pressure margins in the second half of the year.</p>
{tbl("Second Quarter Results (in millions, except per share data)", ["Metric", "Q2 2025", "Q2 2024"], [["Net sales", "$25,211", "$25,452"], ["Operating income", "$1,317", "$1,921"], ["Diluted EPS", "$2.05", "$2.57"], ["Comparable sales change", "(1.9)%", "2.0%"], ["Inventory", "$12,605", "$12,951"]])}
<p>We did not repurchase shares during the quarter. As of quarter end, approximately $8.8 billion remained available under the share repurchase program authorized by our Board.</p>
"""

TGT_10Q_RISK = """
<p>There have been no material changes to the risk factors disclosed in our most recent Annual Report, except as follows: the announced transition of our Chief Executive Officer may create uncertainty among team members, vendors and investors, and the imposition of additional tariffs on goods imported from several sourcing countries has increased the risk that our product costs will rise faster than we can offset.</p>
"""

TGT_8K = """<html><body><p>FORM 8-K: Target Corporation (SYNTHETIC FIXTURE excerpt)</p>
<h2>Item 5.02. Departure of Directors or Certain Officers; Election of Directors; Appointment of Certain Officers</h2>
<p>The Board of Directors approved a chief executive officer succession plan under which the current Chief Operating Officer will become Chief Executive Officer effective at the start of the next fiscal year and the current Chief Executive Officer will transition to the role of Executive Chair.</p>
<h2>Item 2.02. Results of Operations and Financial Condition</h2>
<p>The Company reported second quarter comparable sales of negative 1.9 percent and reaffirmed its full-year adjusted EPS guidance range of approximately $7.00 to $9.00.</p>
<h2>Item 7.01. Regulation FD Disclosure</h2>
<p>Management outlined an enterprise acceleration office focused on speed, technology and merchandising authority, and reiterated its commitment to the dividend and to strong investment-grade ratings.</p>
</body></html>"""

TGT_TRANSCRIPT = """=== PREPARED REMARKS ===
[Chief Executive Officer | management]: Our second-quarter results reflect a challenging environment for discretionary categories and a guest who is cautious and value focused. We are taking decisive steps to restore growth, including faster merchandising decisions, sharper pricing on everyday essentials, and continued investment in same-day services.
[Chief Financial Officer | management]: Comparable sales declined 1.9 percent. Gross margin was pressured by markdowns and tariff-related costs. Our balance sheet remains strong, we ended the quarter with healthy liquidity, and we remain committed to our middle-A ratings and to paying and growing our dividend. We did not repurchase shares in the quarter.
=== Q&A ===
[Analyst, Firm A | analyst]: Can you quantify the tariff exposure for the back half and how much you expect to offset through pricing versus vendor cost sharing?
[Chief Financial Officer | management]: Tariffs are a headwind we are managing through sourcing diversification, vendor negotiations and selective pricing. We are not going to give a specific number today, but it is embedded in our guidance range.
[Analyst, Firm B | analyst]: Traffic has been negative for several quarters. How much of that is the broader discretionary environment versus brand perception issues and the boycott earlier this year?
[Chief Executive Officer | management]: We are listening to our guests. Some of the pressure is macro and some is specific to us, including reaction to decisions we made earlier in the year. We are focused on rebuilding trust through value, newness and a great in-store experience.
[Analyst, Firm C | analyst]: With the stock at these levels and the balance sheet where it is, why not lean into buybacks? How do you think about repurchases versus protecting the rating?
[Chief Financial Officer | management]: Our priorities remain investing in the business first, then the dividend, then share repurchases over time. We will return to buybacks when we are confident in the trajectory and consistent with our ratings commitment.
[Analyst, Firm D | analyst]: On the CEO transition, what changes should investors expect in strategy and timeline, and is there any change to the long-term operating margin target?
[Chief Executive Officer | management]: The transition is about accelerating execution. The strategy is intact. We will share more about our multi-year plan at a future investor event.
[Analyst, Firm E | analyst]: Inventory is down year over year. Are you at risk of being under-assorted in discretionary for holiday, and how are markdowns trending?
[Chief Financial Officer | management]: Inventory is clean and in a good position. We expect markdown pressure to moderate as we enter the holiday season.
[Analyst, Firm F | analyst]: How are you thinking about competitive pricing pressure from Walmart and Costco, who are gaining share with higher-income households?
[Chief Executive Officer | management]: We compete on a differentiated assortment, owned brands and convenience. We are sharpening value on thousands of everyday items.
"""

WMT_RISK = """
<p>We face strong competition from other retailers, wholesale club operators and omni-channel retailers, which could materially adversely affect our financial performance.</p>
<p>Changes in trade policy, including tariffs, could increase the cost of merchandise; our scale and sourcing flexibility partially mitigate this risk.</p>
<p>Our e-commerce, advertising and membership businesses depend on continued investment in technology and automation; failure to execute could affect our growth.</p>
<p>We could be subject to a significant cybersecurity incident.</p>
<p>Our international operations subject us to foreign currency and geopolitical risks.</p>
"""
WMT_MDA = f"""
<p>Consolidated net sales increased 5.1 percent, driven by comparable sales growth and strong e-commerce performance, with transaction counts and unit volumes increasing. Share gains were led by upper-income households. Operating income grew faster than sales as higher-margin businesses such as advertising and membership scaled.</p>
{fin_table('WMT')}
<p>We returned cash through dividends and share repurchases while maintaining our strong AA credit ratings.</p>
"""

COST_RISK = """
<p>We are highly dependent on the financial performance of our U.S. and Canadian operations and on membership renewal rates.</p>
<p>We face strong competition from other retailers and warehouse club operators.</p>
<p>Tariffs and changes in trade policy could increase merchandise costs; our limited-assortment model allows us to adjust sourcing quickly.</p>
<p>We could be subject to a significant cybersecurity incident.</p>
"""
COST_MDA = f"""
<p>Net sales increased 5.0 percent and comparable sales increased 5.3 percent excluding gasoline and foreign exchange. Membership fee income increased reflecting new member sign-ups, a renewal rate of approximately 90 percent in the U.S. and Canada, and the fee increase. Traffic increased across regions.</p>
{fin_table('COST')}
<p>During the year we paid a special cash dividend of $15.00 per share in addition to our quarterly dividend, funded from cash on hand.</p>
"""

DG_RISK = """
<p>Our business is sensitive to the financial condition of our core customer, who has been financially constrained by inflation.</p>
<p>Inventory shrink and store standards issues have affected our results and our reputation with customers.</p>
<p>We have substantial operating lease obligations and indebtedness, which could limit our flexibility.</p>
<p>Tariffs could increase the cost of consumables and discretionary goods.</p>
<p>We face intense competition from mass merchants, grocers and digital retailers.</p>
"""
DG_MDA = f"""
<p>Net sales increased 5.0 percent; same-store sales increased 1.4 percent driven by higher average ticket. Gross margin declined due to higher shrink and markdowns. Our back-to-basics plan focuses on store standards, labor investment and inventory reduction.</p>
{fin_table('DG')}
<p>We suspended share repurchases and prioritized debt reduction to support our investment-grade ratings.</p>
"""

KR_RISK = """
<p>The termination of the proposed merger with Albertsons and related litigation may divert management attention and result in costs.</p>
<p>We face intense competition in the grocery industry, including from mass merchants and warehouse clubs.</p>
<p>Our pharmacy business is subject to reimbursement pressure and regulation.</p>
<p>Labor relations and the cost of union contracts could increase operating expenses.</p>
<p>Food inflation and changes in consumer behavior may affect our results.</p>
"""
KR_MDA = f"""
<p>Identical sales without fuel increased 2.0 percent, led by pharmacy and digital. Following the termination of the merger agreement, we executed a $7.5 billion accelerated share repurchase program.</p>
{fin_table('KR')}
"""

COST_TRANSCRIPT = """=== PREPARED REMARKS ===
[Chief Financial Officer | management]: Comparable sales grew mid-single digits with strong traffic. Renewal rates remain near 90 percent in the U.S. and Canada. We continue to manage tariff impacts through sourcing flexibility and our limited-assortment model.
=== Q&A ===
[Analyst, Firm A | analyst]: Any change in your approach to special dividends given the cash position?
[Chief Financial Officer | management]: We have historically returned excess cash through special dividends and will continue to evaluate that alongside our regular dividend and buybacks.
[Analyst, Firm B | analyst]: Are you seeing trade-down from higher-income members?
[Chief Financial Officer | management]: We continue to see membership growth across income cohorts, including higher-income households.
"""

WMT_TRANSCRIPT = """=== PREPARED REMARKS ===
[Chief Executive Officer | management]: We gained market share again, with higher-income households continuing to drive growth. E-commerce was profitable and our advertising and membership businesses are scaling.
=== Q&A ===
[Analyst, Firm A | analyst]: How are tariffs affecting pricing?
[Chief Financial Officer | management]: We are absorbing some costs and passing through some price increases selectively. Our scale gives us flexibility.
"""

TGT_NEWS = """<html><body><p>NEWS (SYNTHETIC, low authority): Market chatter suggests Target could pursue a large debt-funded acquisition; the company has not commented.</p>
<p>Unnamed sources said the retailer may also cut its dividend, which the company has publicly committed to maintaining.</p></body></html>"""

TGT_CONSENSUS = """<html><body><p>CONSENSUS SNAPSHOT (SYNTHETIC)</p>
<p>Street consensus adjusted EPS for fiscal 2025 is $7.35 across 31 analysts, with a mean price target of $104. Ratings skew to Hold. Analysts cite traffic declines, tariff exposure and execution risk around the leadership transition as primary debates; bulls point to the valuation discount and the dividend yield.</p></body></html>"""


def filings_manifest():
    return [
        dict(file="TGT_10K_FY2024.html", accession="FIXTURE-TGT-10K-FY2024", cik="27419", ticker="TGT", company="Target Corporation", form="10-K", filed="2025-03-12", fy=2024, fq=0, period_end="2025-02-01", content_type="html",
             content=tenk("Target Corporation", 2024, "February 1, 2025", {"1A": ("Risk Factors", TGT_RISK_FY2024), "7": ("Management's Discussion and Analysis of Financial Condition and Results of Operations", TGT_MDA_FY2024)})),
        dict(file="TGT_10K_FY2023.html", accession="FIXTURE-TGT-10K-FY2023", cik="27419", ticker="TGT", company="Target Corporation", form="10-K", filed="2024-03-13", fy=2023, fq=0, period_end="2024-02-03", content_type="html",
             content=tenk("Target Corporation", 2023, "February 3, 2024", {"1A": ("Risk Factors", TGT_RISK_FY2023)})),
        dict(file="TGT_10Q_Q2FY2025.html", accession="FIXTURE-TGT-10Q-Q2FY2025", cik="27419", ticker="TGT", company="Target Corporation", form="10-Q", filed="2025-08-27", fy=2025, fq=2, period_end="2025-08-02", content_type="html",
             content=f"<html><body><p>FORM 10-Q (SYNTHETIC FIXTURE excerpt)</p><p>PART I</p>{PB}<h2>Item 2. Management's Discussion and Analysis</h2>{TGT_10Q_Q2}<p>PART II</p>{PB}<h2>Item 1A. Risk Factors</h2>{TGT_10Q_RISK}</body></html>"),
        dict(file="TGT_8K_20250820.html", accession="FIXTURE-TGT-8K-20250820", cik="27419", ticker="TGT", company="Target Corporation", form="8-K", filed="2025-08-20", fy=2025, fq=2, period_end="2025-08-02", content_type="html", content=TGT_8K),
        dict(file="TGT_TRANSCRIPT_Q2FY2025.txt", accession="FIXTURE-TGT-CALL-Q2FY2025", cik="27419", ticker="TGT", company="Target Corporation", form="TRANSCRIPT", filed="2025-08-20", fy=2025, fq=2, period_end="2025-08-02", content_type="text", content=TGT_TRANSCRIPT),
        dict(file="TGT_NEWS_20250915.html", accession="FIXTURE-TGT-NEWS-20250915", cik="27419", ticker="TGT", company="Target Corporation", form="NEWS", filed="2025-09-15", fy=2025, fq=3, period_end="2025-09-15", content_type="html", content=TGT_NEWS),
        dict(file="TGT_CONSENSUS_20251001.html", accession="FIXTURE-TGT-CONS-20251001", cik="27419", ticker="TGT", company="Target Corporation", form="CONSENSUS", filed="2025-10-01", fy=2025, fq=3, period_end="2025-10-01", content_type="html", content=TGT_CONSENSUS),
        dict(file="WMT_10K_FY2025.html", accession="FIXTURE-WMT-10K-FY2025", cik="104169", ticker="WMT", company="Walmart Inc.", form="10-K", filed="2025-03-14", fy=2025, fq=0, period_end="2025-01-31", content_type="html",
             content=tenk("Walmart Inc.", 2025, "January 31, 2025", {"1A": ("Risk Factors", WMT_RISK), "7": ("Management's Discussion and Analysis", WMT_MDA)})),
        dict(file="WMT_TRANSCRIPT_Q2FY2026.txt", accession="FIXTURE-WMT-CALL-Q2FY2026", cik="104169", ticker="WMT", company="Walmart Inc.", form="TRANSCRIPT", filed="2025-08-21", fy=2026, fq=2, period_end="2025-07-31", content_type="text", content=WMT_TRANSCRIPT),
        dict(file="COST_10K_FY2024.html", accession="FIXTURE-COST-10K-FY2024", cik="909832", ticker="COST", company="Costco Wholesale Corporation", form="10-K", filed="2024-10-09", fy=2024, fq=0, period_end="2024-09-01", content_type="html",
             content=tenk("Costco Wholesale Corporation", 2024, "September 1, 2024", {"1A": ("Risk Factors", COST_RISK), "7": ("Management's Discussion and Analysis", COST_MDA)})),
        dict(file="COST_TRANSCRIPT_Q4FY2025.txt", accession="FIXTURE-COST-CALL-Q4FY2025", cik="909832", ticker="COST", company="Costco Wholesale Corporation", form="TRANSCRIPT", filed="2025-09-25", fy=2025, fq=4, period_end="2025-08-31", content_type="text", content=COST_TRANSCRIPT),
        dict(file="DG_10K_FY2024.html", accession="FIXTURE-DG-10K-FY2024", cik="29534", ticker="DG", company="Dollar General Corporation", form="10-K", filed="2025-03-21", fy=2024, fq=0, period_end="2025-01-31", content_type="html",
             content=tenk("Dollar General Corporation", 2024, "January 31, 2025", {"1A": ("Risk Factors", DG_RISK), "7": ("Management's Discussion and Analysis", DG_MDA)})),
        dict(file="KR_10K_FY2024.html", accession="FIXTURE-KR-10K-FY2024", cik="56873", ticker="KR", company="The Kroger Co.", form="10-K", filed="2025-04-01", fy=2024, fq=0, period_end="2025-02-01", content_type="html",
             content=tenk("The Kroger Co.", 2024, "February 1, 2025", {"1A": ("Risk Factors", KR_RISK), "7": ("Management's Discussion and Analysis", KR_MDA)})),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIL.mkdir(parents=True, exist_ok=True)
    w = lambda name, obj: (OUT / name).write_text(json.dumps(obj, indent=1), encoding="utf-8")  # noqa: E731

    w("entities.json", {"issuers": ISSUERS, "securities": SECURITIES, "institutions": INSTITUTIONS,
                        "competitors": [{"company_cik": a, "peer_cik": b, "strength": s, "basis": "disclosed"} for a, b, s in COMPETITORS],
                        "suppliers": [{"supplier_cik": a, "customer_cik": b, "revenue_share_of_supplier": s, "note": "illustrative customer concentration"} for a, b, s in SUPPLIERS]})
    funds = []
    for t, f in FUNDAMENTALS.items():
        rec = {k: (v if k in PER_SHARE else v * M) for k, v in f.items() if k not in ("fy", "end")}
        rec["shares_diluted"] = f["shares_diluted"] * 1e6
        funds.append({"ticker": t, "issuer_cik": CIK_BY_TICKER[t], "fiscal_year": f["fy"], "period_end": f["end"], **rec})
    w("fundamentals.json", funds)
    w("market.json", [{"ticker": t, "issuer_cik": CIK_BY_TICKER[t], "as_of": AS_OF, "price": p, "shares_outstanding": so * 1e6, "dividend_per_share_annual": dps,
                       "beta": b, "return_1y": r1, "return_3y": r3, "credit_rating": rt} for t, (p, so, dps, b, r1, r3, rt) in MARKET.items()])
    w("consensus.json", [{"ticker": t, "issuer_cik": CIK_BY_TICKER[t], "as_of": AS_OF, "fiscal_year": fy, "eps_mean": m, "eps_high": hi, "eps_low": lo, "revenue_mean": rev * M,
                          "num_analysts": n, "rating_mean": rm, "price_target_mean": pt} for t, (fy, m, hi, lo, rev, n, rm, pt) in CONSENSUS.items()])
    w("float.json", [{"ticker": t, "issuer_cik": CIK_BY_TICKER[t], "as_of": AS_OF, "shares_outstanding": so * 1e6, "insider_shares": ins * 1e6, "restricted_shares": rs * 1e6,
                      "short_interest_shares": si * 1e6, "avg_daily_volume": adv * 1e6} for t, (so, ins, rs, si, adv) in FLOAT.items()])
    w("macro.json", MACRO)
    w("thirteenf.json", build_13f())
    w("form4.json", [{"issuer_cik": ic, "ticker": tk, "accession": f"FIXTURE-F4-{tk}-{i:03d}", "filed_date": filed,
                      "xml": form4_xml(ic, tk, oc, on, d, o, ti, tx, plan, fn)} for i, (ic, tk, oc, on, d, o, ti, tx, plan, fn, filed) in enumerate(FORM4)])
    manifest = []
    for f in filings_manifest():
        (FIL / f["file"]).write_text(f.pop("content"), encoding="utf-8")
        manifest.append(f)
    w("filings.json", {"as_of": AS_OF, "documents": manifest})
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
