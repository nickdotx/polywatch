"""Shared data models — the contract every component codes against.

Shapes are derived from live Polymarket Data API responses (probed 2026-06-29):

  /activity?user=0x...  -> list of activity items (type == "TRADE" is what we want)
  /positions?user=0x... -> list of open positions
  /trades?market=0x...  -> list of fills
  gamma /markets        -> market metadata

Keep these models defensive: the API returns strings for some numbers and may
add/rename fields. Parse with the from_api() helpers, not by hand elsewhere.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _f(v, default: float = 0.0) -> float:
    """Coerce API value (which may be str/None) to float."""
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class TargetTrade(BaseModel):
    """One observed trade by a tracked wallet (from /activity, type=TRADE)."""

    trade_id: str               # idempotency key: tx_hash:asset:side
    wallet: str                 # proxyWallet (lowercased)
    timestamp: int              # unix seconds
    condition_id: str
    asset: str                  # ERC-1155 token id (the outcome token)
    side: str                   # "BUY" | "SELL"
    size: float                 # number of shares
    usdc_size: float            # USDC value of the trade
    price: float                # 0..1
    outcome: str                # e.g. "Yes" / "No"
    outcome_index: int
    title: str = ""
    slug: str = ""
    event_slug: str = ""
    tx_hash: str = ""
    name: str = ""              # display name / pseudonym of the wallet
    pseudonym: str = ""

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @classmethod
    def from_activity(cls, item: dict) -> Optional["TargetTrade"]:
        """Build from a /activity item. Returns None for non-TRADE items."""
        if (item.get("type") or "").upper() != "TRADE":
            return None
        wallet = (item.get("proxyWallet") or "").lower()
        asset = str(item.get("asset") or "")
        side = (item.get("side") or "").upper()
        tx = item.get("transactionHash") or ""
        trade_id = f"{tx}:{asset}:{side}"
        return cls(
            trade_id=trade_id,
            wallet=wallet,
            timestamp=int(item.get("timestamp") or 0),
            condition_id=str(item.get("conditionId") or ""),
            asset=asset,
            side=side,
            size=_f(item.get("size")),
            usdc_size=_f(item.get("usdcSize")),
            price=_f(item.get("price")),
            outcome=str(item.get("outcome") or ""),
            outcome_index=int(item.get("outcomeIndex") or 0),
            title=item.get("title") or "",
            slug=item.get("slug") or "",
            event_slug=item.get("eventSlug") or "",
            tx_hash=tx,
            name=item.get("name") or "",
            pseudonym=item.get("pseudonym") or "",
        )


class Position(BaseModel):
    """An open position for a wallet (from /positions)."""

    wallet: str
    asset: str
    condition_id: str
    size: float
    avg_price: float
    current_value: float
    cash_pnl: float
    percent_pnl: float
    realized_pnl: float
    cur_price: float
    outcome: str
    outcome_index: int
    title: str = ""
    end_date: str = ""
    redeemable: bool = False

    @classmethod
    def from_api(cls, item: dict) -> "Position":
        return cls(
            wallet=(item.get("proxyWallet") or "").lower(),
            asset=str(item.get("asset") or ""),
            condition_id=str(item.get("conditionId") or ""),
            size=_f(item.get("size")),
            avg_price=_f(item.get("avgPrice")),
            current_value=_f(item.get("currentValue")),
            cash_pnl=_f(item.get("cashPnl")),
            percent_pnl=_f(item.get("percentPnl")),
            realized_pnl=_f(item.get("realizedPnl")),
            cur_price=_f(item.get("curPrice")),
            outcome=str(item.get("outcome") or ""),
            outcome_index=int(item.get("outcomeIndex") or 0),
            title=item.get("title") or "",
            end_date=item.get("endDate") or "",
            redeemable=bool(item.get("redeemable") or False),
        )


class Market(BaseModel):
    """Market metadata (from gamma /markets)."""

    condition_id: str
    question: str = ""
    slug: str = ""
    closed: bool = False
    liquidity: float = 0.0
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list)
    clob_token_ids: list[str] = Field(default_factory=list)
    end_date: str = ""

    @classmethod
    def from_gamma(cls, item: dict) -> "Market":
        import json

        def _arr(key):
            raw = item.get(key)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, str) and raw:
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return []
            return []

        prices = [_f(p) for p in _arr("outcomePrices")]
        return cls(
            condition_id=str(item.get("conditionId") or ""),
            question=item.get("question") or "",
            slug=item.get("slug") or "",
            closed=bool(item.get("closed") or False),
            liquidity=_f(item.get("liquidityNum") or item.get("liquidity")),
            outcomes=[str(o) for o in _arr("outcomes")],
            outcome_prices=prices,
            clob_token_ids=[str(t) for t in _arr("clobTokenIds")],
            end_date=item.get("endDateIso") or item.get("endDate") or "",
        )

    @property
    def resolved(self) -> bool:
        """A closed market with a clear 0/1 price is effectively resolved."""
        if not self.closed:
            return False
        return any(p >= 0.99 or p <= 0.01 for p in self.outcome_prices)

    def resolved_outcome_index(self) -> Optional[int]:
        if not self.resolved:
            return None
        for i, p in enumerate(self.outcome_prices):
            if p >= 0.99:
                return i
        return None
