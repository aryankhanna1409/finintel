"""Parsers for SEC payloads: 13F information tables, Form 4 ownership documents,
XBRL company facts, filing HTML (section + table aware) and earnings-call transcripts."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from html import unescape
from html.parser import HTMLParser
from typing import Any, Iterable

from ..entities.resolver import EntityRegistry
from ..schemas.common import Accession, DataProvenance, FiscalPeriod, FormType
from ..schemas.entities import normalize_cik
from ..schemas.filings import ContentBlock, FilingSection, ParsedDocument, RawFiling, TableData
from ..schemas.holdings import HoldingRecord, InsiderTransaction, InsiderTxnCode
from ..schemas.metrics import FinancialStatementSnapshot

# ====================================================================== XML helpers


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _txt(el: ET.Element | None, path: str, default: str = "") -> str:
    if el is None:
        return default
    found = el.find(path)
    if found is None or found.text is None:
        return default
    return found.text.strip()


def _num(s: str, default: float = 0.0) -> float:
    s = s.replace(",", "").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


# ====================================================================== 13F


def parse_13f_infotable(
    xml: str,
    filer_cik: str,
    filer_name: str,
    accession: str,
    report_period: date,
    filed_date: date,
    registry: EntityRegistry | None = None,
    source_url: str | None = None,
    value_in_thousands: bool | None = None,
    is_synthetic: bool = False,
) -> list[HoldingRecord]:
    """Parse a 13F-HR information table.

    Since the 2023 Q1 rule change (filings after 2023-01-03), ``value`` is reported in
    whole dollars; before that it was in thousands. ``value_in_thousands=None`` infers
    this from the filing date.
    """
    root = _strip_ns(ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml))
    if value_in_thousands is None:
        value_in_thousands = filed_date < date(2023, 1, 3)
    mult = 1000.0 if value_in_thousands else 1.0
    prov = DataProvenance(
        source="fixture" if is_synthetic else "sec_edgar", url=source_url, retrieved_at=date.today(), is_synthetic=is_synthetic
    )
    out: list[HoldingRecord] = []
    for it in root.iter("infoTable"):
        cusip = _txt(it, "cusip").upper()
        name = _txt(it, "nameOfIssuer")
        title = _txt(it, "titleOfClass")
        shrs = it.find("shrsOrPrnAmt")
        share_type = _txt(shrs, "sshPrnamtType", "SH").upper()
        put_call = _txt(it, "putCall") or None
        issuer_cik = None
        if registry is not None:
            res = registry.resolve_holding(cusip, name, title, share_type, put_call)
            issuer_cik = res.issuer_cik
            stype = res.security.security_type if res.security else None
        else:
            stype = None
        voting = it.find("votingAuthority")
        rec = HoldingRecord(
            filer_cik=normalize_cik(filer_cik),
            filer_name=filer_name,
            accession=Accession(value=accession),
            report_period=report_period,
            filed_date=filed_date,
            issuer_name_raw=name,
            cusip=cusip,
            title_of_class=title,
            issuer_cik=issuer_cik,
            value_usd=_num(_txt(it, "value")) * mult,
            shares=_num(_txt(shrs, "sshPrnamt")),
            share_type=share_type,
            put_call=put_call,
            investment_discretion=_txt(it, "investmentDiscretion", "SOLE"),
            voting_sole=_num(_txt(voting, "Sole")),
            voting_shared=_num(_txt(voting, "Shared")),
            voting_none=_num(_txt(voting, "None")),
            provenance=prov,
        )
        if stype is not None and not put_call:
            rec.security_type = stype
        out.append(rec)
    return out


# ====================================================================== Form 4


def parse_form4(
    xml: str, accession: str, filed_date: date, source_url: str | None = None, is_synthetic: bool = False
) -> list[InsiderTransaction]:
    root = _strip_ns(ET.fromstring(xml.encode("utf-8")))
    issuer_cik = normalize_cik(_txt(root, "issuer/issuerCik"))
    ticker = _txt(root, "issuer/issuerTradingSymbol") or None
    owner = root.find("reportingOwner")
    name = _txt(owner, "reportingOwnerId/rptOwnerName")
    owner_cik = _txt(owner, "reportingOwnerId/rptOwnerCik") or None
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None

    def flag(tag: str) -> bool:
        return _txt(rel, tag).lower() in ("1", "true")

    is_dir, is_off, is_ten = flag("isDirector"), flag("isOfficer"), flag("isTenPercentOwner")
    title = _txt(rel, "officerTitle")
    role = title or ("Director" if is_dir else ("10% Owner" if is_ten else "Officer"))
    # 10b5-1 checkbox (added to Form 4 in 2023) or footnote language.
    plan_flag = _txt(root, "aff10b5One").lower() in ("1", "true")
    footnotes = " ".join((f.text or "") for f in root.iter("footnote")).lower()
    plan_footnote = "10b5-1" in footnotes
    prov = DataProvenance(
        source="fixture" if is_synthetic else "sec_edgar", url=source_url, retrieved_at=date.today(), is_synthetic=is_synthetic
    )
    out: list[InsiderTransaction] = []
    for t in root.iter("nonDerivativeTransaction"):
        code_raw = _txt(t, "transactionCoding/transactionCode")
        try:
            code = InsiderTxnCode(code_raw)
        except ValueError:
            continue
        shares = _num(_txt(t, "transactionAmounts/transactionShares/value"))
        if shares <= 0:
            continue
        price_s = _txt(t, "transactionAmounts/transactionPricePerShare/value")
        ad = _txt(t, "transactionAmounts/transactionAcquiredDisposedCode/value", "A")
        after = _txt(t, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")
        out.append(
            InsiderTransaction(
                accession=Accession(value=accession),
                issuer_cik=issuer_cik,
                ticker=ticker,
                insider_cik=normalize_cik(owner_cik) if owner_cik else None,
                insider_name=name,
                insider_role=role,
                is_director=is_dir,
                is_officer=is_off,
                is_ten_percent_owner=is_ten,
                transaction_date=date.fromisoformat(_txt(t, "transactionDate/value")[:10]),
                filed_date=filed_date,
                code=code,
                shares=shares,
                price=_num(price_s) if price_s else None,
                acquired=ad.upper() == "A",
                shares_owned_after=_num(after) if after else None,
                is_10b5_1=plan_flag or (plan_footnote and code == InsiderTxnCode.S),
                provenance=prov,
            )
        )
    return out


# ====================================================================== XBRL company facts

XBRL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "cogs": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"),
    "operating_income": ("OperatingIncomeLoss",),
    "depreciation_amortization": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation"),
    "interest_expense": ("InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestPaidNet"),
    "income_tax_expense": ("IncomeTaxExpenseBenefit",),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "diluted_eps": ("EarningsPerShareDiluted",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "cash_and_equivalents": ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
    "short_term_debt": ("LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "LongTermDebt"),
    "operating_lease_liabilities": ("OperatingLeaseLiability", "OperatingLeaseLiabilityNoncurrent"),
    "total_assets": ("Assets",),
    "total_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "accounts_receivable": ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent"),
    "inventory": ("InventoryNet", "RetailRelatedInventoryMerchandise"),
    "accounts_payable": ("AccountsPayableCurrent", "AccountsPayableTradeCurrent"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
    "dividends_paid": ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends"),
    "buybacks": ("PaymentsForRepurchaseOfCommonStock",),
}

_INSTANT_FIELDS = {
    "cash_and_equivalents", "short_term_debt", "long_term_debt", "operating_lease_liabilities", "total_assets",
    "total_equity", "accounts_receivable", "inventory", "accounts_payable",
}
_OPTIONAL_ZERO = {"short_term_debt", "operating_lease_liabilities", "dividends_paid", "buybacks", "accounts_receivable", "interest_expense"}


def _fy_facts(facts: dict[str, Any], concept: str, unit: str) -> list[dict[str, Any]]:
    node = facts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not node:
        return []
    return [f for f in node.get("units", {}).get(unit, []) if f.get("form", "").startswith("10-K")]


def _pick(facts: dict[str, Any], field: str, fy_end: date) -> tuple[float | None, str | None]:
    unit = "USD/shares" if field == "diluted_eps" else ("shares" if field == "shares_diluted" else "USD")
    for concept in XBRL_CONCEPTS[field]:
        cands = []
        for f in _fy_facts(facts, concept, unit):
            if f.get("end") != fy_end.isoformat():
                continue
            if field not in _INSTANT_FIELDS:
                start = f.get("start")
                if not start:
                    continue
                dur = (fy_end - date.fromisoformat(start)).days
                if not 350 <= dur <= 380:
                    continue
            cands.append(f)
        if cands:
            best = max(cands, key=lambda f: f.get("filed", ""))  # latest restatement wins
            return float(best["val"]), best.get("accn")
    return None, None


def latest_fiscal_year_end(facts: dict[str, Any], as_of: date | None = None) -> date:
    ends = set()
    for f in _fy_facts(facts, "Assets", "USD"):
        if f.get("fp") == "FY":
            d = date.fromisoformat(f["end"])
            filed = date.fromisoformat(f["filed"])
            if as_of is None or filed <= as_of:
                ends.add(d)
    if not ends:
        raise ValueError("no annual Assets facts found")
    return max(ends)


def company_facts_to_snapshot(
    facts: dict[str, Any], ticker: str, fy_end: date | None = None, as_of: date | None = None
) -> FinancialStatementSnapshot:
    """Map an XBRL companyfacts payload onto an annual FinancialStatementSnapshot.

    Point-in-time: with ``as_of`` set, only fiscal years whose 10-K was filed on or before
    that date are eligible, preventing look-ahead bias in backtests.
    """
    fy_end = fy_end or latest_fiscal_year_end(facts, as_of)
    values: dict[str, float] = {}
    accns: set[str] = set()
    missing: list[str] = []
    for field in XBRL_CONCEPTS:
        v, accn = _pick(facts, field, fy_end)
        if v is None:
            if field in _OPTIONAL_ZERO:
                v = 0.0
            elif field == "pretax_income":
                continue
            elif field == "diluted_eps":
                continue
            else:
                missing.append(field)
                continue
        if field in ("capex", "dividends_paid", "buybacks", "interest_expense"):
            v = abs(v)
        values[field] = v
        if accn:
            accns.add(accn)
    if "depreciation_amortization" in missing:
        missing.remove("depreciation_amortization")
        values["depreciation_amortization"] = 0.0
    if missing:
        raise ValueError(f"XBRL facts missing required fields for FY ending {fy_end}: {missing}")
    if "pretax_income" not in values:
        values["pretax_income"] = values["net_income"] + values["income_tax_expense"]
    cik = normalize_cik(facts.get("cik", 0))
    period = FiscalPeriod(fiscal_year=fy_end.year if fy_end.month > 6 else fy_end.year - 1, fiscal_quarter=0, period_end=fy_end)
    return FinancialStatementSnapshot(
        issuer_cik=cik,
        ticker=ticker,
        period=period,
        provenance=DataProvenance(
            source="sec_edgar",
            url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            retrieved_at=date.today(),
        ),
        source_accession=sorted(accns)[-1] if accns else None,
        **values,
    )


# ====================================================================== HTML filing parser


class _BlockHTMLParser(HTMLParser):
    """Linearises filing HTML into paragraph / table blocks, tracking page breaks."""

    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[ContentBlock] = []
        self.page = 1
        self._buf: list[str] = []
        self._heading = False
        self._in_table = 0
        self._table_rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._header_rows: set[int] = set()
        self._row_is_header = False
        self._hidden_depth = 0

    VOID_TAGS = {"br", "hr", "img", "meta", "input", "link", "col", "area", "base", "wbr"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        style = a.get("style", "").lower().replace(" ", "")
        if self._hidden_depth:
            if tag not in self.VOID_TAGS:
                self._hidden_depth += 1
            return
        if tag in ("script", "style", "head") or "display:none" in style:
            # Scripts/styles and inline-XBRL hidden header blocks carry no prose.
            if tag not in self.VOID_TAGS:
                self._hidden_depth = 1
            return
        if "page-break-before:always" in style:
            self._flush()
            self.page += 1
        if tag == "table":
            self._flush()
            self._in_table += 1
            if self._in_table == 1:
                self._table_rows, self._header_rows = [], set()
            return
        if self._in_table:
            if tag == "tr":
                self._row, self._row_is_header = [], False
            elif tag in ("td", "th"):
                self._cell = []
                if tag == "th":
                    self._row_is_header = True
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self._heading = True
        elif tag in self.BLOCK_TAGS:
            self._flush()
        elif tag == "hr":
            self._flush()
            if "page-break" in style:
                self.page += 1

    def handle_endtag(self, tag: str) -> None:
        if self._hidden_depth:
            self._hidden_depth -= 1
            return
        if self._in_table:
            if tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
                self._cell = None
            elif tag == "tr" and self._row is not None:
                if any(c for c in self._row):
                    if self._row_is_header:
                        self._header_rows.add(len(self._table_rows))
                    self._table_rows.append(self._row)
                self._row = None
            elif tag == "table":
                self._in_table -= 1
                if self._in_table == 0:
                    self._emit_table()
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush(heading=True)
            self._heading = False
        elif tag in self.BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        if self._in_table:
            if self._cell is not None:
                self._cell.append(data)
            return
        self._buf.append(data)

    def _flush(self, heading: bool = False) -> None:
        text = re.sub(r"\s+", " ", unescape("".join(self._buf))).strip()
        self._buf = []
        if not text:
            return
        if re.fullmatch(r"(page\s*)?\d{1,3}", text, flags=re.I):  # bare page numbers
            return
        kind = "heading" if (heading or self._heading) else "paragraph"
        self.blocks.append(ContentBlock(kind=kind, text=text, page=self.page))

    def _emit_table(self) -> None:
        rows = _clean_table_rows(self._table_rows)
        if not rows:
            return
        header_idx = min(self._header_rows) if self._header_rows else 0
        header = rows[header_idx] if header_idx < len(rows) else rows[0]
        body = [r for i, r in enumerate(rows) if i != header_idx]
        # Single-column "tables" are layout wrappers; treat them as paragraphs.
        if len(header) <= 1 and all(len(r) <= 1 for r in body):
            for r in rows:
                if r and r[0]:
                    self.blocks.append(ContentBlock(kind="paragraph", text=r[0], page=self.page))
            return
        caption, units = None, None
        if self.blocks and self.blocks[-1].kind in ("paragraph", "heading") and len(self.blocks[-1].text) < 200:
            caption = self.blocks[-1].text
        joined = " ".join([caption or ""] + header)
        m = re.search(r"\((?:\$|dollars|amounts)?\s*in\s+(millions|thousands|billions)[^)]*\)", joined, re.I)
        if m:
            units = f"$ in {m.group(1).lower()}"
        table = TableData(caption=caption, header=header, rows=body, units=units)
        self.blocks.append(ContentBlock(kind="table", text=table.to_markdown(), table=table, page=self.page))

    def close(self) -> None:
        super().close()
        self._flush()


def _clean_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """Merge SEC-style split currency cells ("$", "1,234") and ")"/"%" suffix cells, then
    drop columns that are empty in every row."""
    merged: list[list[str]] = []
    for r in rows:
        out: list[str] = []
        i = 0
        while i < len(r):
            c = r[i]
            if c in ("$", "(", "$(") and i + 1 < len(r):
                out.append((c + r[i + 1]).replace("$(", "($"))
                i += 2
                continue
            if c in (")", "%", ")%") and out:
                out[-1] = out[-1] + c
                i += 1
                continue
            out.append(c)
            i += 1
        merged.append(out)
    if not merged:
        return []
    width = max(len(r) for r in merged)
    merged = [r + [""] * (width - len(r)) for r in merged]
    keep = [j for j in range(width) if any(r[j] for r in merged)]
    return [[r[j] for j in keep] for r in merged]


_ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}(?:\.\d{2})?[A-C]?)\s*[.:\-–—]?\s*(.{0,160})$", re.I)
_PART_RE = re.compile(r"^\s*part\s+(I{1,3}|IV)\b", re.I)


def parse_filing_html(filing: RawFiling) -> ParsedDocument:
    """Split a 10-K/10-Q/8-K HTML document into Item sections with ordered blocks.

    Table-of-contents entries (usually inside a table, or a run of item headers without
    body) are discarded by keeping, for each item, the occurrence with the most content.
    """
    p = _BlockHTMLParser()
    p.feed(filing.content)
    p.close()
    sections: list[FilingSection] = []
    cur = FilingSection(item="PREAMBLE", title="Cover / Preamble", page_start=1)
    part = ""
    for b in p.blocks:
        if b.kind in ("heading", "paragraph") and len(b.text) < 200:
            pm = _PART_RE.match(b.text)
            if pm and len(b.text) < 60:
                part = pm.group(1).upper()
                continue
            m = _ITEM_RE.match(b.text)
            if m:
                sections.append(cur)
                item = m.group(1).upper()
                if filing.form_type == FormType.TEN_Q and part:
                    item = f"{part}-{item}"
                cur = FilingSection(item=item, title=m.group(2).strip(" .") or f"Item {item}", page_start=b.page)
                continue
        cur.blocks.append(b)
    sections.append(cur)
    best: dict[str, FilingSection] = {}
    order: list[str] = []
    for s in sections:
        size = sum(len(b.text) for b in s.blocks)
        if s.item not in best:
            order.append(s.item)
            best[s.item] = s
        elif size > sum(len(b.text) for b in best[s.item].blocks):
            best[s.item] = s
    final = [best[i] for i in order if best[i].blocks]
    return ParsedDocument(filing=filing, sections=final)


# ====================================================================== transcripts

_TURN_RE = re.compile(r"^\[(?P<speaker>[^|\]]+)\|\s*(?P<role>management|analyst|operator)\s*\]:\s*(?P<text>.+)$", re.I)


def parse_transcript(filing: RawFiling) -> ParsedDocument:
    """Parse a transcript in the normalized format::

        === PREPARED REMARKS ===
        [Chief Executive Officer | management]: ...
        === Q&A ===
        [Analyst, Firm A | analyst]: ...
        [Chief Financial Officer | management]: ...

    Vendor transcripts (e.g. JSON from a data provider) should be normalised into this
    form by an adapter so downstream chunking can pair each analyst question with the
    management answers that follow it.
    """
    sections: list[FilingSection] = []
    cur: FilingSection | None = None
    for line in filing.content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            name = line.strip("= ").upper()
            item = "Q&A" if "Q&A" in name or "QUESTION" in name else "PREPARED_REMARKS"
            cur = FilingSection(item=item, title=name.title())
            sections.append(cur)
            continue
        m = _TURN_RE.match(line)
        if cur is None:
            cur = FilingSection(item="PREPARED_REMARKS", title="Prepared Remarks")
            sections.append(cur)
        if m:
            cur.blocks.append(
                ContentBlock(kind="speaker_turn", text=m.group("text").strip(), speaker=m.group("speaker").strip(), speaker_role=m.group("role").lower())
            )
        elif cur.blocks:
            cur.blocks[-1].text += " " + line
        else:
            cur.blocks.append(ContentBlock(kind="paragraph", text=line))
    return ParsedDocument(filing=filing, sections=sections)


def parse_document(filing: RawFiling) -> ParsedDocument:
    if filing.form_type == FormType.TRANSCRIPT:
        return parse_transcript(filing)
    if filing.content_type == "html":
        return parse_filing_html(filing)
    # Plain text: paragraphs separated by blank lines, items detected the same way.
    html = "".join(f"<p>{unescape(par)}</p>" for par in re.split(r"\n\s*\n", filing.content) if par.strip())
    return parse_filing_html(filing.model_copy(update={"content": html, "content_type": "html"}))


def iter_tables(doc: ParsedDocument) -> Iterable[tuple[FilingSection, TableData]]:
    for s in doc.sections:
        for b in s.blocks:
            if b.kind == "table" and b.table is not None:
                yield s, b.table
