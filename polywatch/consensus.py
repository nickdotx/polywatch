"""K-of-N consensus tracker.

Fixes the "one wallet carries the whole edge" problem: a market/side only
becomes a signal once K DISTINCT tracked wallets have bought it within a rolling
time window. No single wallet's luck can trigger a copy on its own.

Shared by the live run loop (fed trades as they arrive) and the backtest (fed
historical trades in timestamp order).
"""

from __future__ import annotations


class ConsensusTracker:
    def __init__(self, k: int = 2, window_seconds: int = 86400):
        self.k = max(1, int(k))
        self.window = window_seconds
        self._seen: dict = {}        # (cid, idx) -> list[(ts, wallet)]
        self._triggered: set = set()

    def observe(self, cid: str, outcome_index: int, wallet: str, ts: int) -> bool:
        """Record a buy. Return True exactly once — on the event that first
        reaches K distinct wallets within the window for this (market, side)."""
        key = (cid, int(outcome_index))
        if key in self._triggered:
            return False
        lst = self._seen.setdefault(key, [])
        lst.append((int(ts), (wallet or "").lower()))
        right = max(t for t, _ in lst)                   # anchor on newest seen
        lo = right - self.window
        lst[:] = [(t, w) for (t, w) in lst if t >= lo]   # symmetric window prune
        if len({w for _, w in lst}) >= self.k:
            self._triggered.add(key)
            return True
        return False

    def distinct_in_window(self, cid: str, outcome_index: int) -> int:
        lst = self._seen.get((cid, int(outcome_index)), [])
        return len({w for _, w in lst})
