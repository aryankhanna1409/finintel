"""SEC EDGAR client: polite, rate-limited, cached access to public EDGAR endpoints.

SEC fair-access policy: identify yourself with a descriptive User-Agent that includes a
contact email, and stay under 10 requests/second. This client enforces both, retries
transient failures with exponential backoff, and caches responses on disk so repeated
pipeline runs are cheap and reproducible.

Endpoints used:
* ``https://www.sec.gov/files/company_tickers.json``: ticker → CIK map
* ``https://data.sec.gov/submissions/CIK##########.json``: filing index per filer
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``: XBRL facts
* ``https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/index.json``: filing folder
* ``https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<doc>``: documents
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from ..schemas.common import Accession, DataProvenance, FiscalPeriod, FormType
from ..schemas.entities import normalize_cik
from ..schemas.filings import RawFiling

log = logging.getLogger(__name__)

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"

_FORM_MAP = {
    "10-K": FormType.TEN_K, "10-K/A": FormType.TEN_K, "10-Q": FormType.TEN_Q, "10-Q/A": FormType.TEN_Q,
    "8-K": FormType.EIGHT_K, "8-K/A": FormType.EIGHT_K, "13F-HR": FormType.THIRTEEN_F_HR,
    "13F-HR/A": FormType.THIRTEEN_F_HR, "4": FormType.FORM_4, "4/A": FormType.FORM_4,
    "SC 13D": FormType.SC_13D, "SC 13D/A": FormType.SC_13D, "SCHEDULE 13D": FormType.SC_13D,
    "SC 13G": FormType.SC_13G, "SC 13G/A": FormType.SC_13G, "SCHEDULE 13G": FormType.SC_13G,
    "DEF 14A": FormType.DEF_14A,
}


class EdgarError(RuntimeError):
    pass


class _RateLimiter:
    """Token-bucket limiter shared by all requests from one client."""

    def __init__(self, rate_per_sec: float) -> None:
        self.min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = self._next
            self._next = now + self.min_interval


@dataclass
class FilingRef:
    accession: str
    form: str
    filed: date
    report_date: date | None
    primary_document: str
    cik: str

    @property
    def folder_url(self) -> str:
        return f"{SEC_WWW}/Archives/edgar/data/{int(self.cik)}/{self.accession.replace('-', '')}"

    @property
    def primary_url(self) -> str:
        return f"{self.folder_url}/{self.primary_document}"


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path | None = None,
        rate_per_sec: float = 8.0,
        max_retries: int = 4,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with contact email, e.g. 'Acme Research ops@acme.com'")
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = _RateLimiter(rate_per_sec)
        self.max_retries = max_retries
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._ticker_map: dict[str, dict[str, Any]] | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / hashlib.sha256(url.encode()).hexdigest()[:32]

    def get_text(self, url: str, use_cache: bool = True) -> str:
        cp = self._cache_path(url)
        if use_cache and cp and cp.exists():
            return cp.read_text(encoding="utf-8")
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                resp = self._http.get(url)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise EdgarError(f"HTTP {resp.status_code} for {url}")
                if resp.status_code == 404:
                    raise FileNotFoundError(url)
                resp.raise_for_status()
                text = resp.text
                if cp:
                    cp.write_text(text, encoding="utf-8")
                return text
            except FileNotFoundError:
                raise
            except (httpx.HTTPError, EdgarError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                log.warning("EDGAR request failed (%s), retry %d in %.1fs", exc, attempt + 1, delay)
                time.sleep(delay)
                delay *= 2
        raise EdgarError(f"EDGAR request failed after {self.max_retries + 1} attempts: {url}") from last_exc

    def get_json(self, url: str, use_cache: bool = True) -> Any:
        return json.loads(self.get_text(url, use_cache=use_cache))

    # ------------------------------------------------------------------ discovery
    def ticker_map(self) -> dict[str, dict[str, Any]]:
        if self._ticker_map is None:
            raw = self.get_json(f"{SEC_WWW}/files/company_tickers.json")
            self._ticker_map = {
                str(v["ticker"]).upper().replace("-", "."): {"cik": normalize_cik(v["cik_str"]), "title": v["title"]}
                for v in raw.values()
            }
        return self._ticker_map

    def cik_for_ticker(self, ticker: str) -> str:
        t = ticker.upper().replace("-", ".")
        m = self.ticker_map()
        if t not in m:
            raise KeyError(f"ticker {ticker} not found in SEC company_tickers.json")
        return m[t]["cik"]

    def submissions(self, cik: str) -> dict[str, Any]:
        return self.get_json(f"{SEC_DATA}/submissions/CIK{normalize_cik(cik)}.json")

    def company_facts(self, cik: str) -> dict[str, Any]:
        return self.get_json(f"{SEC_DATA}/api/xbrl/companyfacts/CIK{normalize_cik(cik)}.json")

    def list_filings(
        self,
        cik: str,
        forms: set[str],
        since: date | None = None,
        until: date | None = None,
        limit: int = 20,
    ) -> list[FilingRef]:
        sub = self.submissions(cik)
        recent = sub.get("filings", {}).get("recent", {})
        out: list[FilingRef] = []
        n = len(recent.get("accessionNumber", []))
        for i in range(n):
            form = recent["form"][i]
            if form not in forms:
                continue
            filed = date.fromisoformat(recent["filingDate"][i])
            if since and filed < since:
                continue
            if until and filed > until:
                continue
            rd = recent.get("reportDate", [""] * n)[i]
            out.append(
                FilingRef(
                    accession=recent["accessionNumber"][i],
                    form=form,
                    filed=filed,
                    report_date=date.fromisoformat(rd) if rd else None,
                    primary_document=recent["primaryDocument"][i],
                    cik=normalize_cik(cik),
                )
            )
            if len(out) >= limit:
                break
        return out

    def filing_index(self, ref: FilingRef) -> list[dict[str, Any]]:
        idx = self.get_json(f"{ref.folder_url}/index.json")
        return idx.get("directory", {}).get("item", [])

    # ------------------------------------------------------------------ documents
    def fetch_filing(self, ref: FilingRef, company_name: str, ticker: str | None, period: FiscalPeriod | None) -> RawFiling:
        content = self.get_text(ref.primary_url)
        ctype = "xml" if ref.primary_document.lower().endswith(".xml") else ("text" if ref.primary_document.lower().endswith(".txt") else "html")
        return RawFiling(
            accession=Accession(value=ref.accession),
            cik=ref.cik,
            company_name=company_name,
            ticker=ticker,
            form_type=_FORM_MAP.get(ref.form, FormType.EIGHT_K),
            filed_date=ref.filed,
            period=period,
            primary_document_url=ref.primary_url,
            content_type=ctype,  # type: ignore[arg-type]
            content=content,
            provenance=DataProvenance(source="sec_edgar", url=ref.primary_url, retrieved_at=date.today()),
        )

    def thirteen_f_infotable_xml(self, ref: FilingRef) -> tuple[str, str]:
        """Return (url, xml) of the 13F information table inside a 13F-HR filing folder."""
        items = self.filing_index(ref)
        candidates = [
            it["name"] for it in items
            if it["name"].lower().endswith(".xml") and "primary_doc" not in it["name"].lower()
        ]
        # Prefer files explicitly named like info tables.
        candidates.sort(key=lambda n: (0 if ("info" in n.lower() or "table" in n.lower()) else 1, n))
        for name in candidates:
            url = f"{ref.folder_url}/{name}"
            xml = self.get_text(url)
            if "informationTable" in xml or "infoTable" in xml:
                return url, xml
        raise EdgarError(f"no information table found in {ref.folder_url}")

    def form4_xml(self, ref: FilingRef) -> tuple[str, str]:
        """Form 4 primary documents are often XSL-rendered HTML; the raw XML lives alongside."""
        if ref.primary_document.lower().endswith(".xml") and "/" not in ref.primary_document:
            url = ref.primary_url
            return url, self.get_text(url)
        for it in self.filing_index(ref):
            if it["name"].lower().endswith(".xml"):
                url = f"{ref.folder_url}/{it['name']}"
                xml = self.get_text(url)
                if "ownershipDocument" in xml:
                    return url, xml
        raise EdgarError(f"no ownershipDocument XML in {ref.folder_url}")
