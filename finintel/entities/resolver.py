"""Entity resolution and ticker disambiguation.

Guards against "entity drift", the failure mode where facts about one legal entity or
instrument get attributed to another:

* **Subsidiary debt vs. parent equity**: a 13F/credit row keyed on a financing
  subsidiary's bond CUSIP resolves to the parent *issuer* for credit exposure, but is
  typed as debt so it never inflates equity ownership.
* **Share classes**: GOOGL/GOOG, BRK.A/BRK.B are distinct securities of one issuer;
  equity exposure is aggregated in economic-equivalent units via conversion ratios.
* **Cross-holdings**: issuer-to-issuer equity stakes are represented as Securities owned
  by an Issuer, so the graph can surface circular ownership.
* **Ticker reuse / formatting**: tickers are normalised (``BRK-B``→``BRK.B``) and every
  resolution reports its method and confidence so low-confidence matches can be held out.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable

from ..schemas.common import SecurityType
from ..schemas.entities import Issuer, ResolvedEntity, Security, normalize_cik, normalize_ticker

_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc",
    "llc", "lp", "holdings", "holding", "group", "the", "sa", "nv", "ag", "new", "del", "de",
    "com", "common", "stock", "shs", "cl", "class", "a", "b", "c",
}

_DEBT_HINTS = re.compile(r"\b(NOTE|NOTES|NT|NTS|SR|SUB|DEB|DEBT|BOND|BD|FRN|\d+(\.\d+)?%|DUE)\b")
_PREF_HINTS = re.compile(r"\b(PFD|PREF|PREFERRED|DEP SHS)\b")
_CONV_HINTS = re.compile(r"\b(CONV|CV|CONVERTIBLE)\b")
_WARRANT_HINTS = re.compile(r"\b(WT|WTS|WARRANT|WARRANTS)\b")
_ADR_HINTS = re.compile(r"\b(ADR|ADS|SPONSORED)\b")
_CLASS_RE = re.compile(r"\b(?:CL|CLASS|SER|SERIES)\s*([A-Z])\b")


def cusip_check_digit(base8: str) -> int:
    total = 0
    for i, ch in enumerate(base8.upper()):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch) - ord("A") + 10
        elif ch == "*":
            v = 36
        elif ch == "@":
            v = 37
        elif ch == "#":
            v = 38
        else:
            raise ValueError(f"invalid CUSIP character {ch!r}")
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return (10 - total % 10) % 10


def is_valid_cusip(cusip: str) -> bool:
    c = cusip.strip().upper()
    if len(c) != 9 or not c[8].isdigit():
        return False
    try:
        return cusip_check_digit(c[:8]) == int(c[8])
    except ValueError:
        return False


def cusip_issuer_prefix(cusip: str) -> str:
    """First six characters identify the issuer; chars 7-8 identify the issue."""
    return cusip.strip().upper()[:6]


def cusip_issue_is_equity(cusip: str) -> bool | None:
    """CUSIP issue-number convention: numeric issue codes (10..88) are typically equity,
    alphabetic or alphanumeric codes are typically fixed income. Heuristic only."""
    issue = cusip.strip().upper()[6:8]
    if issue.isdigit():
        return True
    if issue.isalpha() or any(ch.isalpha() for ch in issue):
        return False
    return None


def normalize_name(name: str) -> str:
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def infer_security_type(title_of_class: str, share_type: str = "SH", put_call: str | None = None) -> SecurityType:
    t = title_of_class.upper()
    if put_call:
        return SecurityType.OPTION_PUT if put_call.upper() == "PUT" else SecurityType.OPTION_CALL
    if _CONV_HINTS.search(t) and (share_type == "PRN" or _DEBT_HINTS.search(t)):
        return SecurityType.CONVERTIBLE_DEBT
    if share_type == "PRN" or _DEBT_HINTS.search(t):
        return SecurityType.SUBORDINATED_DEBT if re.search(r"\bSUB\b", t) else SecurityType.SENIOR_DEBT
    if _PREF_HINTS.search(t):
        return SecurityType.PREFERRED_EQUITY
    if _WARRANT_HINTS.search(t):
        return SecurityType.WARRANT
    if _ADR_HINTS.search(t):
        return SecurityType.ADR
    return SecurityType.COMMON_EQUITY


def share_class_of(title_of_class: str) -> str | None:
    m = _CLASS_RE.search(title_of_class.upper())
    return m.group(1) if m else None


class EntityRegistry:
    """In-memory master data for issuers and securities with multi-key indices."""

    def __init__(self) -> None:
        self.issuers: dict[str, Issuer] = {}
        self.securities_by_cusip: dict[str, Security] = {}
        self._ticker_to_security: dict[str, Security] = {}
        self._ticker_to_cik: dict[str, str] = {}
        self._name_index: dict[str, set[str]] = defaultdict(set)
        self._cusip6_to_cik: dict[str, str] = {}
        self.cross_holdings: list[tuple[str, str, float]] = []  # (holder_cik, held_cik, shares)

    # ------------------------------------------------------------------ registration
    def add_issuer(self, issuer: Issuer) -> None:
        self.issuers[issuer.cik] = issuer
        for t in issuer.tickers:
            self._ticker_to_cik[t] = issuer.cik
        for n in (issuer.name, *issuer.aliases):
            self._name_index[normalize_name(n)].add(issuer.cik)

    def add_security(self, sec: Security) -> None:
        if sec.cusip:
            c = sec.cusip.upper()
            self.securities_by_cusip[c] = sec
            if sec.security_type.is_equity_ownership or sec.security_type == SecurityType.PREFERRED_EQUITY:
                self._cusip6_to_cik.setdefault(cusip_issuer_prefix(c), sec.issuer_cik)
        if sec.ticker:
            self._ticker_to_security[sec.ticker] = sec
            self._ticker_to_cik.setdefault(sec.ticker, sec.issuer_cik)

    def add_cross_holding(self, holder_cik: str, held_cik: str, shares: float) -> None:
        self.cross_holdings.append((normalize_cik(holder_cik), normalize_cik(held_cik), shares))

    def bulk_load(self, issuers: Iterable[Issuer], securities: Iterable[Security]) -> "EntityRegistry":
        for i in issuers:
            self.add_issuer(i)
        for s in securities:
            self.add_security(s)
        return self

    # ------------------------------------------------------------------ lookups
    def ultimate_parent(self, cik: str) -> str:
        seen: set[str] = set()
        cur = normalize_cik(cik)
        while cur in self.issuers and self.issuers[cur].parent_cik and cur not in seen:
            seen.add(cur)
            cur = self.issuers[cur].parent_cik  # type: ignore[assignment]
        return cur

    def issuer_by_ticker(self, ticker: str) -> Issuer | None:
        cik = self._ticker_to_cik.get(normalize_ticker(ticker))
        return self.issuers.get(cik) if cik else None

    def ticker_for(self, cik: str) -> str | None:
        iss = self.issuers.get(normalize_cik(cik))
        return iss.primary_ticker if iss else None

    def equity_classes(self, cik: str) -> list[Security]:
        cik = normalize_cik(cik)
        return [
            s for s in self.securities_by_cusip.values()
            if s.issuer_cik == cik and s.security_type.is_equity_ownership
        ]

    # ------------------------------------------------------------------ resolution
    def resolve(self, raw: str) -> ResolvedEntity:
        """Resolve a free-form mention (CIK, CUSIP, ticker or company name)."""
        s = raw.strip()
        if not s:
            return ResolvedEntity(raw=raw, warnings=["empty mention"])
        if re.fullmatch(r"\d{1,10}", s):
            cik = normalize_cik(s)
            if cik in self.issuers:
                return ResolvedEntity(raw=raw, issuer_cik=cik, method="cik", confidence=1.0)
        if len(s) == 9 and re.fullmatch(r"[0-9A-Za-z*@#]{9}", s) and is_valid_cusip(s):
            return self._resolve_cusip(raw, s.upper(), title_of_class=None)
        tick = normalize_ticker(s)
        if re.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,2})?", tick):
            if tick in self._ticker_to_security:
                sec = self._ticker_to_security[tick]
                return ResolvedEntity(raw=raw, issuer_cik=sec.issuer_cik, security=sec, method="ticker", confidence=0.99)
            if tick in self._ticker_to_cik:
                return ResolvedEntity(raw=raw, issuer_cik=self._ticker_to_cik[tick], method="ticker", confidence=0.97)
        return self._resolve_name(raw, s)

    def resolve_holding(
        self,
        cusip: str,
        issuer_name: str,
        title_of_class: str,
        share_type: str = "SH",
        put_call: str | None = None,
    ) -> ResolvedEntity:
        """Resolve a 13F information-table row to (issuer, security, security type)."""
        res = self._resolve_cusip(issuer_name, cusip.upper(), title_of_class, share_type, put_call)
        if not res.resolved:
            by_name = self._resolve_name(issuer_name, issuer_name)
            if by_name.resolved and by_name.confidence >= 0.9:
                stype = infer_security_type(title_of_class, share_type, put_call)
                by_name.security = Security(
                    cusip=cusip.upper() if len(cusip) == 9 else None,
                    issuer_cik=by_name.issuer_cik,  # type: ignore[arg-type]
                    security_type=stype,
                    share_class=share_class_of(title_of_class),
                    description=title_of_class,
                )
                by_name.method = "name_fallback"
                by_name.confidence = min(by_name.confidence, 0.85)
                by_name.warnings.append(f"CUSIP {cusip} not in master; resolved by issuer name")
                if not is_valid_cusip(cusip):
                    by_name.warnings.append(f"CUSIP {cusip} fails check digit")
                return by_name
        return res

    def _resolve_cusip(
        self,
        raw: str,
        cusip: str,
        title_of_class: str | None,
        share_type: str = "SH",
        put_call: str | None = None,
    ) -> ResolvedEntity:
        warnings: list[str] = []
        if not is_valid_cusip(cusip):
            warnings.append(f"CUSIP {cusip} fails check digit")
        sec = self.securities_by_cusip.get(cusip)
        if sec is not None:
            if put_call:
                sec = sec.model_copy(update={"security_type": infer_security_type(title_of_class or "", share_type, put_call)})
            issuer = sec.issuer_cik
            if sec.obligor_cik and sec.obligor_cik != issuer:
                warnings.append(f"debt obligor {sec.obligor_cik} differs from issuer {issuer}")
            parent = self.ultimate_parent(issuer)
            if parent != issuer:
                warnings.append(f"issuer {issuer} is a subsidiary of {parent}; credit exposure rolls up, equity does not")
            return ResolvedEntity(raw=raw, issuer_cik=issuer, security=sec, method="cusip", confidence=1.0 if not warnings else 0.95, warnings=warnings)
        # Unknown issue under a known issuer prefix (e.g. a new bond of a known company).
        prefix_cik = self._cusip6_to_cik.get(cusip_issuer_prefix(cusip))
        if prefix_cik:
            stype = infer_security_type(title_of_class or "", share_type, put_call)
            if stype == SecurityType.COMMON_EQUITY and cusip_issue_is_equity(cusip) is False:
                stype = SecurityType.SENIOR_DEBT
                warnings.append("alphanumeric CUSIP issue code: treated as debt, excluded from equity ownership")
            sec = Security(
                cusip=cusip, issuer_cik=prefix_cik, security_type=stype,
                share_class=share_class_of(title_of_class or ""), description=title_of_class or "",
            )
            return ResolvedEntity(raw=raw, issuer_cik=prefix_cik, security=sec, method="cusip6_prefix", confidence=0.9, warnings=warnings)
        return ResolvedEntity(raw=raw, method="unresolved", confidence=0.0, warnings=warnings + [f"unknown CUSIP {cusip}"])

    def _resolve_name(self, raw: str, name: str) -> ResolvedEntity:
        norm = normalize_name(name)
        if not norm:
            return ResolvedEntity(raw=raw, warnings=["name normalises to empty"])
        exact = self._name_index.get(norm)
        if exact:
            if len(exact) == 1:
                return ResolvedEntity(raw=raw, issuer_cik=next(iter(exact)), method="name_exact", confidence=0.95)
            parents = {self.ultimate_parent(c) for c in exact}
            pick = sorted(exact)[0]
            return ResolvedEntity(
                raw=raw, issuer_cik=pick, method="name_exact_ambiguous", confidence=0.6,
                warnings=[f"name matches {len(exact)} issuers (parents: {sorted(parents)})"],
            )
        q_tokens = set(norm.split())
        scored: list[tuple[float, str]] = []
        for key, ciks in self._name_index.items():
            k_tokens = set(key.split())
            jacc = len(q_tokens & k_tokens) / max(len(q_tokens | k_tokens), 1)
            seq = SequenceMatcher(None, norm, key).ratio()
            score = 0.55 * seq + 0.45 * jacc
            for c in ciks:
                scored.append((score, c))
        if not scored:
            return ResolvedEntity(raw=raw, warnings=["empty registry"])
        scored.sort(reverse=True)
        best_score, best = scored[0]
        warnings: list[str] = []
        runner_up = next((s for s, c in scored[1:] if c != best), 0.0)
        if best_score < 0.72:
            return ResolvedEntity(raw=raw, method="unresolved", confidence=best_score, warnings=[f"best fuzzy match {best} below threshold ({best_score:.2f})"])
        conf = best_score
        if best_score - runner_up < 0.05:
            conf *= 0.75
            warnings.append(f"ambiguous fuzzy match (margin {best_score - runner_up:.3f})")
        return ResolvedEntity(raw=raw, issuer_cik=best, method="name_fuzzy", confidence=round(conf, 3), warnings=warnings)

    def mentions(self, text: str) -> list[str]:
        """Issuers mentioned in free text. Case-sensitive on purpose: "Target" the company
        vs. "target leverage"; tickers must be standalone upper-case tokens."""
        found: list[str] = []
        for cik, iss in self.issuers.items():
            hit = any(re.search(rf"\b{re.escape(t)}\b", text) for t in iss.tickers if len(t) >= 2)
            if not hit:
                for n in (iss.name, *iss.aliases):
                    core = " ".join(w for w in re.sub(r"[^\w&' -]", " ", n).split() if w.lower() not in _SUFFIXES)
                    if core and core[0].isupper() and re.search(rf"\b{re.escape(core)}(?:'s)?\b", text):
                        hit = True
                        break
            if hit and self.ultimate_parent(cik) not in found:
                found.append(self.ultimate_parent(cik))
        return found

    # ------------------------------------------------------------------ economics
    def economic_equity_shares(self, cusip: str, shares: float) -> float:
        """Convert a holding into the issuer's base-class economic share units."""
        sec = self.securities_by_cusip.get(cusip.upper())
        if sec is None or not sec.security_type.is_equity_ownership:
            return 0.0
        return shares * sec.economic_conversion_ratio
