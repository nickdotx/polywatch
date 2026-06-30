"""Regression tests for the data-integrity hardening pass."""

from polywatch import scorer
from polywatch.backtest import backtest_copy
from polywatch.models import Market


def _ev(t, **kw):
    base = {"type": t, "conditionId": "A", "outcomeIndex": 0, "usdcSize": 0,
            "size": 0, "price": 0, "timestamp": 1700000000, "side": ""}
    base.update(kw)
    return base


class MergeClient:
    """Wallet that buys Yes+No then MERGEs back to collateral (net flat)."""
    data_api = "https://x"

    def _get(self, url, params):
        if params.get("offset", 0) != 0:
            return []
        return [
            _ev("TRADE", side="BUY", outcomeIndex=0, usdcSize=40, size=100, price=0.40),
            _ev("TRADE", side="BUY", outcomeIndex=1, usdcSize=60, size=100, price=0.60),
            _ev("MERGE", usdcSize=100, size=100),
        ]

    def get_market(self, cid):
        return None  # should not be needed; MERGE closes the position


def test_merge_event_keeps_position_and_nets_flat():
    recs = scorer.reconstruct(MergeClient(), "0xw", {}, max_events=10)
    assert len(recs) == 1                     # NOT dropped
    assert abs(recs[0]["realized"]) < 1e-6    # bought 100, merged 100 back -> flat


def _bt_ev(cid, idx, price, usd):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid, "outcomeIndex": idx,
            "price": price, "usdcSize": usd, "size": usd / price}


class DoubleBuyClient:
    data_api = "https://x"

    def _get(self, url, params):
        if params.get("offset", 0) != 0:
            return []
        # two BUYs in the SAME market+outcome (an add) -> should count as ONE copy
        return [_bt_ev("A", 0, 0.30, 200), _bt_ev("A", 0, 0.31, 200)]

    def get_market(self, cid):
        return Market(condition_id="A", closed=True, outcomes=["Y", "N"],
                      outcome_prices=[1.0, 0.0])


def test_backtest_dedupes_repeated_buys_per_market():
    r = backtest_copy(DoubleBuyClient(), ["0xw"], stake=50, min_usd=100)
    assert r.n == 1   # one copy per (market, outcome), not two
