"""Market and macro data adapters.

* :class:`StooqPriceProvider`: keyless daily quotes/history (``stooq.com`` CSV API).
* :class:`FredClient`: keyless FRED series via the public ``fredgraph.csv`` download:
  the 10Y Treasury (``DGS10``) and ICE BofA option-adjusted spreads by rating bucket.

Both implement small protocols so a licensed vendor feed (Bloomberg, FactSet,
Refinitiv) can be swapped in without touching analytics code.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from datetime import date, timedelta
from typing import Protocol

import httpx

from ..schemas.common import DataProvenance
from ..schemas.metrics import MacroSnapshot

log = logging.getLogger(__name__)

# ICE BofA US corporate OAS by rating (percent). AAA..CCC.
FRED_OAS_SERIES = {
    "AAA": "BAMLC0A1CAAA",
    "AA": "BAMLC0A2CAA",
    "A": "BAMLC0A3CA",
    "BBB": "BAMLC0A4CBBB",
    "BB": "BAMLH0A1HYBB",
    "B": "BAMLH0A2HYB",
    "CCC": "BAMLH0A3HYC",
}


class PriceProvider(Protocol):
    def last_close(self, ticker: str, as_of: date | None = None) -> tuple[date, float]: ...

    def history(self, ticker: str, start: date, end: date) -> list[tuple[date, float]]: ...


class StooqPriceProvider:
    BASE = "https://stooq.com/q/d/l/"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._http = client or httpx.Client(timeout=20.0, follow_redirects=True)

    @staticmethod
    def _symbol(ticker: str) -> str:
        return ticker.lower().replace(".", "-") + ".us"

    def history(self, ticker: str, start: date, end: date) -> list[tuple[date, float]]:
        params = {"s": self._symbol(ticker), "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d"), "i": "d"}
        resp = self._http.get(self.BASE, params=params)
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        out = []
        for r in rows:
            try:
                out.append((date.fromisoformat(r["Date"]), float(r["Close"])))
            except (KeyError, ValueError):
                continue
        if not out:
            raise ValueError(f"no price history from stooq for {ticker}")
        return out

    def last_close(self, ticker: str, as_of: date | None = None) -> tuple[date, float]:
        end = as_of or date.today()
        hist = self.history(ticker, end - timedelta(days=10), end)
        return hist[-1]


def total_return(history: list[tuple[date, float]], dividends_per_share: float = 0.0) -> float:
    """Approximate total return over the window from closes plus cash dividends."""
    if len(history) < 2:
        raise ValueError("need at least two observations")
    p0, p1 = history[0][1], history[-1][1]
    return (p1 + dividends_per_share) / p0 - 1.0


def beta_from_history(stock: list[tuple[date, float]], market: list[tuple[date, float]]) -> float:
    """OLS beta of weekly log returns (weekly sampling reduces non-synchronous noise)."""
    m = dict(market)
    common = [d for d, _ in stock if d in m]
    s_map = dict(stock)
    weekly = common[::5]
    rs, rm = [], []
    for a, b in zip(weekly, weekly[1:]):
        rs.append(math.log(s_map[b] / s_map[a]))
        rm.append(math.log(m[b] / m[a]))
    n = len(rs)
    if n < 20:
        raise ValueError("insufficient overlapping history for beta")
    mean_s, mean_m = sum(rs) / n, sum(rm) / n
    cov = sum((x - mean_s) * (y - mean_m) for x, y in zip(rs, rm)) / (n - 1)
    var = sum((y - mean_m) ** 2 for y in rm) / (n - 1)
    return cov / var


class FredClient:
    BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._http = client or httpx.Client(timeout=20.0, follow_redirects=True)

    def latest(self, series_id: str, as_of: date | None = None) -> tuple[date, float]:
        end = as_of or date.today()
        resp = self._http.get(self.BASE, params={"id": series_id, "cosd": (end - timedelta(days=30)).isoformat(), "coed": end.isoformat()})
        resp.raise_for_status()
        rows = list(csv.reader(io.StringIO(resp.text)))
        obs = []
        for r in rows[1:]:
            if len(r) >= 2 and r[1] not in (".", ""):
                try:
                    obs.append((date.fromisoformat(r[0]), float(r[1])))
                except ValueError:
                    continue
        if not obs:
            raise ValueError(f"no FRED observations for {series_id}")
        return obs[-1]

    def macro_snapshot(self, as_of: date | None = None, equity_risk_premium: float = 0.05, marginal_tax_rate: float = 0.25) -> MacroSnapshot:
        d, rf = self.latest("DGS10", as_of)
        spreads = {}
        for bucket, sid in FRED_OAS_SERIES.items():
            try:
                spreads[bucket] = self.latest(sid, as_of)[1] / 100.0
            except Exception as exc:  # noqa: BLE001 - one missing bucket shouldn't kill the snapshot
                log.warning("FRED %s unavailable: %s", sid, exc)
        if len(spreads) < 4:
            raise ValueError("too few FRED credit spread buckets available")
        return MacroSnapshot(
            as_of=d,
            risk_free_rate=rf / 100.0,
            equity_risk_premium=equity_risk_premium,
            credit_spreads=interpolate_spread_curve(spreads),
            marginal_tax_rate=marginal_tax_rate,
            provenance=DataProvenance(source="fred", url=self.BASE, retrieved_at=date.today()),
        )


RATING_SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC"]


def rating_bucket(rating: str) -> str:
    r = rating.upper().replace("+", "").replace("-", "")
    return r if r in FRED_OAS_SERIES else "BBB"


def interpolate_spread_curve(bucket_spreads: dict[str, float]) -> dict[str, float]:
    """Expand bucket OAS (AAA, AA, A, BBB...) to notched ratings by log-linear interpolation."""
    anchors = [(RATING_SCALE.index(b), s) for b, s in bucket_spreads.items() if b in RATING_SCALE and s > 0]
    anchors.sort()
    out: dict[str, float] = {}
    for i, notch in enumerate(RATING_SCALE):
        lower = max((a for a in anchors if a[0] <= i), default=anchors[0])
        upper = min((a for a in anchors if a[0] >= i), default=anchors[-1])
        if lower[0] == upper[0]:
            out[notch] = lower[1]
        else:
            w = (i - lower[0]) / (upper[0] - lower[0])
            out[notch] = math.exp(math.log(lower[1]) * (1 - w) + math.log(upper[1]) * w)
    return out
