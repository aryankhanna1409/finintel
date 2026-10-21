"""Finance-aware tokenization, sentence splitting and term normalization."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or of to in on for by with at from as is are was were be been being this that these those it its
    our we us you your they their them he she his her not no but if then than so such into over under about after before
    during per may might could would should will can do does did has have had which who whom whose what when where why how
    also any all each other some more most very there here""".split()
)

# Canonical forms for financial synonyms/abbreviations. Applied to both query and documents
# so lexical and hashed-dense channels see "buyback" and "repurchase" as the same concept.
CANONICAL = {
    "buyback": "repurchase", "buybacks": "repurchase", "repurchases": "repurchase", "repurchased": "repurchase",
    "repurchasing": "repurchase", "capex": "capital_expenditure", "expenditures": "expenditure",
    "ebit": "operating_income", "sga": "sga_expense", "eps": "earnings_per_share", "comps": "comparable_sales",
    "comp": "comparable_sales", "fcf": "free_cash_flow", "theft": "shrink", "shrinkage": "shrink",
    "tariffs": "tariff", "duties": "tariff", "levered": "leverage", "leveraged": "leverage", "indebtedness": "debt",
    "borrowings": "debt", "notes": "debt", "dividends": "dividend", "boycott": "boycott", "boycotts": "boycott",
    "activists": "activist", "guidance": "outlook", "guide": "outlook", "markdowns": "markdown", "clearance": "markdown",
    "rating": "credit_rating", "ratings": "credit_rating", "ceo": "chief_executive", "cfo": "chief_financial",
    "md&a": "mda", "10k": "10-k", "10q": "10-q", "8k": "8-k",
}

# Multi-word phrases collapsed to single tokens before tokenization.
PHRASES = [
    (re.compile(r"\bshare repurchases?\b", re.I), "repurchase"),
    (re.compile(r"\bstock buy-?backs?\b", re.I), "repurchase"),
    (re.compile(r"\bcapital expenditures?\b", re.I), "capital_expenditure"),
    (re.compile(r"\bfree cash flow\b", re.I), "free_cash_flow"),
    (re.compile(r"\bcomparable (?:store )?sales\b", re.I), "comparable_sales"),
    (re.compile(r"\bsame[- ]store sales\b", re.I), "comparable_sales"),
    (re.compile(r"\bidentical sales\b", re.I), "comparable_sales"),
    (re.compile(r"\boperating income\b", re.I), "operating_income"),
    (re.compile(r"\bearnings per share\b", re.I), "earnings_per_share"),
    (re.compile(r"\bgross margin\b", re.I), "gross_margin"),
    (re.compile(r"\boperating margin\b", re.I), "operating_margin"),
    (re.compile(r"\bcash conversion cycle\b", re.I), "cash_conversion_cycle"),
    (re.compile(r"\binvestment[- ]grade\b", re.I), "investment_grade"),
    (re.compile(r"\bcredit ratings?\b", re.I), "credit_rating"),
    (re.compile(r"\brisk factors?\b", re.I), "risk_factor"),
    (re.compile(r"\btrade policy\b", re.I), "tariff"),
    (re.compile(r"\borganized retail crime\b", re.I), "shrink"),
    (re.compile(r"\bspecial (?:cash )?dividends?\b", re.I), "special_dividend"),
    (re.compile(r"\bchief executive officer\b", re.I), "chief_executive"),
    (re.compile(r"\bchief financial officer\b", re.I), "chief_financial"),
    (re.compile(r"\bmanagement'?s discussion and analysis\b", re.I), "mda"),
    (re.compile(r"\bactivist (?:investors?|funds?|campaigns?)\b", re.I), "activist"),
    (re.compile(r"\bsuccession\b", re.I), "succession leadership_transition"),
]

_TOKEN_RE = re.compile(
    r"\d+[a-z]?-[a-z]{1,2}\b"  # form types: 10-k, 10-q, 8-k, 13f-hr
    r"|\$?\d+(?:[.,]\d+)*%?"  # numbers, currency, percents
    r"|[a-z][a-z0-9_]*(?:[-/&][a-z0-9_]+)*"  # words incl. 10-k style and ev/ebitda
)


def _stem(tok: str) -> str:
    """Conservative suffix stripper (Porter step 1a/1b subset). Applied identically to
    queries and documents, so it only needs to be consistent, not linguistically perfect."""
    if "_" in tok or len(tok) <= 4 or any(c.isdigit() for c in tok) or "-" in tok:
        return tok
    if tok.endswith("ss") or tok.endswith("us") or tok.endswith("is"):
        return tok
    if tok.endswith("ies") and len(tok) > 5:
        return tok[:-3] + "y"
    for suf in ("ing", "ed"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            base = tok[: -len(suf)]
            if base.endswith(("at", "iz", "bl", "ur", "lud", "uc", "ag", "ov", "rv")):
                base += "e"
            elif len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]
            return base
    if tok.endswith("es") and tok.endswith(("ches", "shes", "xes", "sses", "zes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


# Interrogative filler that carries no retrieval signal in analyst-style questions.
QUERY_FILLER = frozenset("much many say said tell talk describe explain mention mentioned discuss think know give show".split())

# Canonical finance concepts: up-weighted by the feature reranker relative to raw IDF.
FINANCE_CONCEPTS = frozenset(
    """repurchase dividend special_dividend tariff shrink leverage debt credit_rating investment_grade comparable_sales
    gross_margin operating_margin operating_income free_cash_flow capital_expenditure markdown inventory traffic
    activist boycott succession leadership_transition earnings_per_share outlook liquidity covenant acquisition
    merger pharmacy membership renewal cybersecurity competition pricing margin cash_conversion_cycle""".split()
)


def query_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in QUERY_FILLER]


def normalize_number(tok: str) -> str:
    t = tok.replace("$", "").replace(",", "")
    return t


def tokenize(text: str, stem: bool = True, keep_stopwords: bool = False) -> list[str]:
    s = text
    for rx, rep in PHRASES:
        s = rx.sub(f" {rep} ", s)
    s = s.lower()
    out: list[str] = []
    for m in _TOKEN_RE.finditer(s):
        tok = m.group(0)
        if "-" in tok and tok[0].isdigit():
            out.append(tok)
            continue
        if tok[0].isdigit() or tok[0] == "$":
            out.append(normalize_number(tok))
            continue
        if not keep_stopwords and tok in STOPWORDS:
            continue
        tok = CANONICAL.get(tok, tok)
        out.append(_stem(tok) if stem else tok)
    return out


_ABBREV = re.compile(r"\b(?:Inc|Corp|Co|Ltd|No|Nos|U\.S|vs|approx|Mr|Ms|Dr|St|e\.g|i\.e|Fig|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.$")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that respects decimals ($7.35), abbreviations (Inc., U.S.) and
    item references (Item 1A.)."""
    parts: list[str] = []
    buf = ""
    for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text.strip()):
        cand = (buf + " " + piece).strip() if buf else piece
        if _ABBREV.search(cand) or re.search(r"\bItem\s+\d+[A-C]?\.$", cand):
            buf = cand
            continue
        parts.append(cand)
        buf = ""
    if buf:
        parts.append(buf)
    return [p for p in parts if p]


NUMBER_RE = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?%?")


def extract_numbers(text: str) -> list[float]:
    vals = []
    for m in NUMBER_RE.finditer(text):
        raw = m.group(0)
        neg = (raw.startswith("(") and ")" in raw) or raw.startswith("-")
        core = raw.replace("(", "").replace(")", "").replace("$", "").replace(",", "").replace("%", "").lstrip("-")
        if not core or core == ".":
            continue
        try:
            v = float(core)
        except ValueError:
            continue
        vals.append(-v if neg else v)
    return vals
