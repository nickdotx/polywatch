"""Offline tests for the historical copy-trade backtester (no network)."""

from polywatch.backtest import backtest_copy, backtest_wallet
from polywatch.models import Market


def _ev(cid, idx, price, usd, side="BUY", typ="TRADE"):
    return {"type": typ, "side": side, "conditionId": cid, "outcomeIndex": idx,
            "price": price, "usdcSize": usd, "size": usd / price if price else 0}


class FakeClient:
    data_api = "https://x"

    def __init__(self, events, markets):
        self._events = events
        self._markets = markets

    def _get(self, url, params):
        return self._events if params.get("offset", 0) == 0 else []

    def get_market(self, cid):
        return self._markets.get(cid)


def test_backtest_wins_losses_and_filters():
    events = [
        _ev("A", 0, 0.50, 200),   # resolved winner -> +50
        _ev("B", 0, 0.50, 200),   # resolved loser  -> -50
        _ev("C", 0, 0.50, 200),   # unresolved      -> skipped
        _ev("A", 0, 0.50, 10),    # below min_usd   -> skipped
        _ev("A", 0, 0.50, 200, side="SELL"),  # not a BUY -> skipped
    ]
    markets = {
        "A": Market(condition_id="A", closed=True, outcomes=["Y", "N"], outcome_prices=[1.0, 0.0]),
        "B": Market(condition_id="B", closed=True, outcomes=["Y", "N"], outcome_prices=[0.0, 1.0]),
        "C": Market(condition_id="C", closed=False, outcomes=["Y", "N"], outcome_prices=[0.4, 0.6]),
    }
    client = FakeClient(events, markets)
    wb = backtest_wallet(client, "0xw", {}, stake=50, min_usd=100)
    assert wb.n == 2
    assert wb.wins == 1
    assert round(wb.pnl, 2) == 0.0          # +50 winner, -50 loser
    assert round(wb.stake_total, 2) == 100.0

    res = backtest_copy(client, ["0xw"], stake=50, min_usd=100)
    assert res.n == 2 and res.wins == 1
    assert round(res.pnl, 2) == 0.0
    assert res.win_rate == 0.5
