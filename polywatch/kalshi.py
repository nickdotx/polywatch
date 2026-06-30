"""Kalshi read-only client (PUBLIC market data — no API key needed).

Kalshi's market-data endpoints are unauthenticated, so the cross-platform arb
SCANNER needs zero credentials (you only need API keys later, to place orders).
Base URL per the docs: https://external-api.kalshi.com/trade-api/v2

Orderbook quirk: Kalshi only returns BIDS (YES bids and NO bids), because a NO
bid at price p is economically a YES ask at (1 - p). So:
    best_yes_ask = 1 - best_no_bid     (cheapest someone will sell you YES)
    best_no_ask  = 1 - best_yes_bid
We expose those derived asks, which is exactly what arb needs (you buy at the ask).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import httpx

DEFAULT_BASE = "https://external-api.kalshi.com/trade-api/v2"


class _RateLimiter:
    def __init__(self, max_calls: int = 100, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self.period]
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = self.period - (now - self._calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)


class KalshiClient:
    def __init__(self, base: str = DEFAULT_BASE, timeout: float = 15.0,
                 retries: int = 3):
        self.base = base.rstrip("/")
        self.retries = retries
        self._client = httpx.Client(
            timeout=timeout, headers={"User-Agent": "polywatch/0.1 (+read-only)",
                                      "Accept": "application/json"})
        self._limiter = _RateLimiter()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _get(self, path: str, params=None) -> Optional[object]:
        url = f"{self.base}{path}"
        for attempt in range(self.retries):
            self._limiter.acquire()
            try:
                r = self._client.get(url, params=params)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError):
                time.sleep(0.5 * (2 ** attempt))
        return None

    # ---- market data (public) -----------------------------------------

    def get_markets(self, status: str = "open", limit: int = 200,
                    cursor: str = "", series_ticker: str = "",
                    event_ticker: str = "") -> tuple[list[dict], str]:
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        data = self._get("/markets", params)
        if not isinstance(data, dict):
            return [], ""
        return data.get("markets") or [], data.get("cursor") or ""

    def get_market(self, ticker: str) -> Optional[dict]:
        data = self._get(f"/markets/{ticker}")
        if isinstance(data, dict):
            return data.get("market") or data
        return None

    def get_events(self, category: str = "", status: str = "open",
                   limit: int = 200, cursor: str = "",
                   with_nested_markets: bool = True) -> tuple[list[dict], str]:
        params = {"limit": limit, "status": status,
                  "with_nested_markets": "true" if with_nested_markets else "false"}
        if category:
            params["category"] = category
        if cursor:
            params["cursor"] = cursor
        data = self._get("/events", params)
        if not isinstance(data, dict):
            return [], ""
        return data.get("events") or [], data.get("cursor") or ""

    def markets_by_categories(self, categories, per_cat: int = 300) -> list[dict]:
        """Open markets in the given categories, fetched via events-with-nested-
        markets. This AVOIDS the default market firehose, which during big sports
        seasons is flooded with KXMVE parlay combos. Each returned market dict is
        enriched with `_event_title` (the parent event's question) so terse
        per-candidate markets still carry context for matching."""
        out: list[dict] = []
        seen: set = set()
        for cat in categories:
            got = 0
            cursor = ""
            while got < per_cat:
                evs, cursor = self.get_events(category=cat, status="open",
                                              limit=200, cursor=cursor)
                if not evs:
                    break
                for e in evs:
                    etitle = e.get("title") or ""
                    for m in (e.get("markets") or []):
                        tk = str(m.get("ticker") or "")
                        if not tk or tk in seen:
                            continue
                        seen.add(tk)
                        mm = dict(m)
                        mm["_event_title"] = etitle
                        out.append(mm)
                        got += 1
                if not cursor:
                    break
        return out

    def get_orderbook(self, ticker: str) -> Optional[dict]:
        """Top-of-book derived asks for a Kalshi market.

        Returns {"yes_ask", "yes_ask_size", "no_ask", "no_ask_size",
                 "yes_bid", "no_bid"} in dollars (0-1). None on failure."""
        data = self._get(f"/markets/{ticker}/orderbook")
        if not isinstance(data, dict):
            return None
        ob = data.get("orderbook") or data.get("orderbook_fp") or {}

        def _best_bid(key_dollars, key_cents):
            levels = ob.get(key_dollars)
            scale = 1.0
            if levels is None:                      # older API: cents arrays
                levels = ob.get(key_cents)
                scale = 0.01
            best_p, best_sz = None, 0.0
            for lvl in levels or []:
                try:
                    p = float(lvl[0]) * scale
                    sz = float(lvl[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if best_p is None or p > best_p:    # bids: highest price is best
                    best_p, best_sz = p, sz
            return best_p, best_sz

        yes_bid, yes_bid_sz = _best_bid("yes_dollars", "yes")
        no_bid, no_bid_sz = _best_bid("no_dollars", "no")
        # reciprocal: buying YES means lifting the best NO bid, and vice versa
        yes_ask = (1.0 - no_bid) if no_bid is not None else None
        no_ask = (1.0 - yes_bid) if yes_bid is not None else None
        return {"yes_ask": yes_ask, "yes_ask_size": no_bid_sz,
                "no_ask": no_ask, "no_ask_size": yes_bid_sz,
                "yes_bid": yes_bid, "no_bid": no_bid}


def kalshi_fee(price: float, contracts: float = 1.0, rate: float = 0.07) -> float:
    """Kalshi trade fee estimate: rate * price * (1-price) per contract (default
    7%; some index markets are 3.5%). Highest near $0.50 (~1.75c/contract) — the
    main cost that eats thin cross-platform gaps. Exact fee has sub-cent rounding;
    this is the close-enough estimate for arb feasibility."""
    return rate * contracts * price * (1.0 - price)
