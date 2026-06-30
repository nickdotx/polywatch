"""Polymarket read-only API client (Data API + Gamma + CLOB).

All network access for v1 goes through here. Read-only — no keys, no orders.
A shared rate limiter keeps us under Polymarket's budget; `_get` retries
transient failures (429/5xx/timeouts) so a momentary blip doesn't look like
"this wallet has no trades".
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import httpx

from .models import Market, Position, TargetTrade

DEFAULT_DATA_API = "https://data-api.polymarket.com"
DEFAULT_GAMMA_API = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_API = "https://clob.polymarket.com"


class _RateLimiter:
    """At most `max_calls` per `period` seconds, shared across threads."""

    def __init__(self, max_calls: int = 400, period: float = 10.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        # Compute any required wait UNDER the lock, then sleep OUTSIDE it so
        # throttling one thread doesn't serialize all the others.
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


class PolymarketClient:
    def __init__(
        self,
        data_api: str = DEFAULT_DATA_API,
        gamma_api: str = DEFAULT_GAMMA_API,
        clob_api: str = DEFAULT_CLOB_API,
        timeout: float = 15.0,
        retries: int = 3,
    ):
        self.data_api = data_api.rstrip("/")
        self.gamma_api = gamma_api.rstrip("/")
        self.clob_api = clob_api.rstrip("/")
        self.retries = retries
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "polywatch/0.1 (+read-only)"},
        )
        self._limiter = _RateLimiter()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PolymarketClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get(self, url: str, params) -> Optional[object]:
        """GET with bounded retry/backoff on transient errors.

        Returns parsed JSON on success, or None after exhausting retries. A
        successful-but-empty response returns [] / {} (truthy distinction is
        left to callers); only genuine failures return None.
        """
        last_exc = None
        for attempt in range(self.retries):
            self._limiter.acquire()
            try:
                r = self._client.get(url, params=params)
                if r.status_code == 429 or r.status_code >= 500:
                    last_exc = httpx.HTTPStatusError(
                        "retryable", request=r.request, response=r
                    )
                    time.sleep(0.5 * (2 ** attempt))  # backoff: 0.5, 1, 2s
                    continue
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError) as e:
                last_exc = e
                time.sleep(0.5 * (2 ** attempt))
        return None

    # ---- Data API (per-wallet) ----------------------------------------

    def get_activity(self, wallet: str, limit: int = 50) -> list[TargetTrade]:
        """Recent activity for a wallet, filtered to TRADE events, newest first."""
        data = self._get(
            f"{self.data_api}/activity", {"user": wallet, "limit": limit}
        )
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            t = TargetTrade.from_activity(item)
            if t is not None:
                out.append(t)
        return out

    def get_positions(self, wallet: str, limit: int = 200) -> list[Position]:
        data = self._get(
            f"{self.data_api}/positions", {"user": wallet, "limit": limit}
        )
        if not isinstance(data, list):
            return []
        return [Position.from_api(i) for i in data]

    def get_closed_positions(self, wallet: str, limit: int = 500) -> list[dict]:
        """Resolved/closed positions for a wallet — used for skill scoring."""
        data = self._get(
            f"{self.data_api}/closed-positions", {"user": wallet, "limit": limit}
        )
        return data if isinstance(data, list) else []

    def get_value(self, wallet: str) -> float:
        """Current portfolio value (USDC) for a wallet."""
        data = self._get(f"{self.data_api}/value", {"user": wallet})
        if isinstance(data, list) and data:
            try:
                return float(data[0].get("value") or 0.0)
            except (TypeError, ValueError, AttributeError):
                return 0.0
        return 0.0

    def get_market_holders(self, condition_id: str, limit: int = 20) -> list[str]:
        """Top holder wallet addresses for a market (both outcome tokens)."""
        data = self._get(
            f"{self.data_api}/holders", {"market": condition_id, "limit": limit}
        )
        wallets: list[str] = []
        if isinstance(data, list):
            for token in data:
                if not isinstance(token, dict):
                    continue
                for h in token.get("holders", []) or []:
                    if not isinstance(h, dict):
                        continue
                    w = (h.get("proxyWallet") or "").lower()
                    if w:
                        wallets.append(w)
        return wallets

    def get_market_trades(self, condition_id: str, limit: int = 100) -> list[dict]:
        """Recent fills for a market (raw dicts)."""
        data = self._get(
            f"{self.data_api}/trades", {"market": condition_id, "limit": limit}
        )
        return data if isinstance(data, list) else []

    # ---- Gamma (market metadata) --------------------------------------

    def get_market(self, condition_id: str) -> Optional[Market]:
        # Gamma excludes closed/resolved markets by default, so try the plain
        # query first (open markets) and fall back to closed=true (resolved).
        for params in (
            {"condition_ids": condition_id},
            {"condition_ids": condition_id, "closed": "true"},
        ):
            data = self._get(f"{self.gamma_api}/markets", params)
            if isinstance(data, list) and data:
                return Market.from_gamma(data[0])
        return None

    def get_top_markets(self, limit: int = 150) -> list[dict]:
        """Active, open markets sorted by liquidity (desc). Raw gamma dicts.

        Paginates with offset (gamma caps ~100 per request), so asking for >100
        actually returns more — important for discovery, where the overlap with
        other venues lives outside the top-100-by-liquidity (currently sports)."""
        out: list[dict] = []
        offset = 0
        while len(out) < limit:
            data = self._get(
                f"{self.gamma_api}/markets",
                {"closed": "false", "active": "true", "limit": 100, "offset": offset},
            )
            if not isinstance(data, list) or not data:
                break
            out.extend(data)
            if len(data) < 100:
                break
            offset += 100

        def _liq(m):
            try:
                return float(m.get("liquidityNum") or m.get("liquidity") or 0)
            except (TypeError, ValueError):
                return 0.0

        return sorted(out, key=_liq, reverse=True)[:limit]

    def get_markets(self, condition_ids, closed=True, chunk: int = 40) -> dict:
        """Resolve many markets in few requests (repeated condition_ids param)."""
        out: dict = {}
        ids = [str(c) for c in condition_ids if c]
        for i in range(0, len(ids), chunk):
            part = ids[i:i + chunk]
            params = [("condition_ids", c) for c in part]
            params.append(("limit", str(len(part))))
            if closed is not None:
                params.append(("closed", "true" if closed else "false"))
            data = self._get(f"{self.gamma_api}/markets", params)
            if isinstance(data, list):
                for m in data:
                    cid = str(m.get("conditionId") or "")
                    if cid:
                        out[cid] = Market.from_gamma(m)
        return out

    # ---- CLOB (live price, read-only) ---------------------------------

    def get_price(self, token_id: str, side: str = "sell") -> Optional[float]:
        """Mid/last price for a CLOB token."""
        data = self._get(
            f"{self.clob_api}/price", {"token_id": token_id, "side": side}
        )
        if isinstance(data, dict) and "price" in data:
            try:
                return float(data["price"])
            except (TypeError, ValueError):
                return None
        return None

    def get_book(self, token_id: str) -> Optional[dict]:
        """CLOB order book for a token. Returns
        {"best_bid": float|None, "bid_size": float, "best_ask": float|None,
         "ask_size": float} — the top of book, which is what arb needs (you can
        only *buy* at the best ask). Raw asks/bids may be unsorted, so we take
        min(ask) / max(bid) defensively. None on failure."""
        data = self._get(f"{self.clob_api}/book", {"token_id": token_id})
        if not isinstance(data, dict):
            return None

        def _levels(key):
            out = []
            for lvl in data.get(key) or []:
                try:
                    out.append((float(lvl.get("price")), float(lvl.get("size"))))
                except (TypeError, ValueError, AttributeError):
                    continue
            return out

        asks = _levels("asks")
        bids = _levels("bids")
        best_ask = min(asks, key=lambda x: x[0]) if asks else (None, 0.0)
        best_bid = max(bids, key=lambda x: x[0]) if bids else (None, 0.0)
        return {"best_ask": best_ask[0], "ask_size": best_ask[1],
                "best_bid": best_bid[0], "bid_size": best_bid[1]}
