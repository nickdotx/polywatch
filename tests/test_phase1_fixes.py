"""Phase-1 correctness fixes (review-driven): negRisk conversions, resolution-
confirmed win metric, event dedup, two-sided (arb/MM) detection, and the
backtest's chronological first-entry rule."""

from polywatch import scorer
from polywatch.backtest import backtest_wallet
from polywatch.models import Market


def _client(events, win_idx=0):
    class C:
        data_api = "https://x"
        def _get(self, url, params):
            return events if params.get("offset", 0) == 0 else []
        def get_market(self, cid):
            prices = [1.0, 0.0] if win_idx == 0 else [0.0, 1.0]
            return Market(condition_id=cid, closed=True, outcomes=["Y", "N"], outcome_prices=prices)
    return C()


def _buy(cid, idx, usd, size, price, ts=1700000000):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid, "outcomeIndex": idx,
            "usdcSize": usd, "size": size, "price": price, "timestamp": ts}


def _sell(cid, idx, usd, size, ts=1700000001):
    return {"type": "TRADE", "side": "SELL", "conditionId": cid, "outcomeIndex": idx,
            "usdcSize": usd, "size": size, "timestamp": ts}


def _redeem(cid, usd, size, ts=1700000002):
    return {"type": "REDEEM", "conditionId": cid, "usdcSize": usd, "size": size, "timestamp": ts}


def _conv(cid, usd, size, ts=1700000003):
    return {"type": "CONVERSION", "conditionId": cid, "usdcSize": usd, "size": size, "timestamp": ts}


# --- negRisk conversions are dropped, not mis-settled --------------------

def test_conversion_market_is_dropped():
    ev = [_buy("A", 0, 20, 100, 0.20), _conv("A", 5, 50)]
    stats = {}
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {}, stats=stats)
    assert recs == []                      # no phantom P&L from a cross-market op
    assert stats["conv_dropped"] == 1


# --- resolution-confirmed correctness vs. mere profitability -------------

def test_held_longshot_win_is_resolution_confirmed():
    # buy 100 Yes @0.20, hold to resolution, Yes wins
    ev = [_buy("A", 0, 20, 100, 0.20)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    r = recs[0]
    assert r["won"] and r["resolved_known"] and r["predicted_correct"]
    assert round(r["avg_entry"], 2) == 0.20


def test_traded_out_longshot_is_not_a_resolution_win():
    # buy 100 Yes @0.20, SELL all @0.40 before resolution -> profit, but NOT a call
    ev = [_buy("A", 0, 20, 100, 0.20), _sell("A", 0, 40, 100)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    r = recs[0]
    assert r["won"]                        # made money
    assert not r["resolved_known"]         # ...but we never confirmed the outcome
    assert not r["predicted_correct"]


# --- paginated-overlap dedup --------------------------------------------

def test_pull_activity_dedupes_repeated_events():
    # each event duplicated; realized must reflect ONE buy + ONE redeem (+60, not +120)
    ev = [_buy("A", 0, 40, 100, 0.40), _buy("A", 0, 40, 100, 0.40),
          _redeem("A", 100, 100), _redeem("A", 100, 100)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    assert round(recs[0]["realized"], 2) == 60.0


# --- two-sided (arb / market-maker) tell --------------------------------

def test_two_sided_market_flagged():
    # bought BOTH legs of the same market -> two_sided
    ev = [_buy("A", 0, 30, 60, 0.50), _buy("A", 1, 20, 40, 0.50)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    assert recs[0]["two_sided"] is True


# --- backtest copies the EARLIEST entry, not the latest add -------------

class _BTClient:
    data_api = "https://x"
    def __init__(self, events, markets):
        self._events = events
        self._markets = markets
    def _get(self, url, params):
        return self._events if params.get("offset", 0) == 0 else []
    def get_market(self, cid):
        return self._markets.get(cid)


def test_backtest_uses_chronological_first_entry():
    # same (cid,idx) bought twice: cheap first (0.20), expensive later add (0.50).
    # Chronological rule must copy the 0.20 entry. Gross win = 10/0.20 - 10 = 40.
    events = [
        _buy("A", 0, 10, 50, 0.20, ts=100),    # earliest, cheap
        _buy("A", 0, 10, 20, 0.50, ts=200),    # later add, expensive
    ]
    markets = {"A": Market(condition_id="A", closed=True, outcomes=["Y", "N"],
                           outcome_prices=[1.0, 0.0])}
    wb = backtest_wallet(_BTClient(events, markets), "0xw", {}, stake=10, min_usd=1)
    assert wb.n == 1
    assert round(wb.pnl_gross, 2) == 40.0
