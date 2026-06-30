"""Realistic trading-cost model: slippage + fees.

Copying is never free or instant. By the time you copy a leader, the price has
usually moved against you (slippage), and the platform may charge a fee. This
turns an optimistic GROSS backtest into a realistic NET one — the honest number.

  - slippage_points: added to the entry price (you fill worse than the leader).
    e.g. leader bought at 0.40, slippage 0.02 -> you effectively pay 0.42, so you
    get fewer shares for the same stake, which lowers winning payouts.
  - fee_pct: fraction of notional charged per side (buy + payout). Polymarket has
    historically been ~0% but has introduced fees on some markets; keep it tunable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    slippage_points: float = 0.0   # probability points added to entry price
    fee_pct: float = 0.0           # fraction of notional, charged per side
    latency_points: float = 0.0    # extra adverse drift between leader's fill and
                                   # our copy (we see the trade late; the price has
                                   # already moved toward the informed side)

    def effective_entry(self, entry_price: float) -> float:
        return min(0.99, max(0.01, entry_price + self.slippage_points + self.latency_points))

    def net_pnl(self, entry_price: float, stake: float, won: bool) -> float:
        """Net P&L of copying one trade, after slippage and fees."""
        eff = self.effective_entry(entry_price)
        shares = stake / eff if eff > 0 else 0.0
        gross = (shares - stake) if won else -stake
        payout = shares if won else 0.0
        fee = self.fee_pct * (stake + payout)
        return gross - fee

    @property
    def is_zero(self) -> bool:
        return self.slippage_points == 0.0 and self.fee_pct == 0.0
