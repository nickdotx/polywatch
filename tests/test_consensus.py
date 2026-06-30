from polywatch.consensus import ConsensusTracker
from polywatch.backtest import backtest_consensus
from polywatch.models import Market


def test_tracker_triggers_on_kth_distinct_wallet():
    t = ConsensusTracker(k=2, window_seconds=3600)
    assert t.observe("A", 0, "0xw1", 1000) is False   # 1 distinct
    assert t.observe("A", 0, "0xw1", 1100) is False   # same wallet, still 1
    assert t.observe("A", 0, "0xw2", 1200) is True    # 2 distinct -> trigger
    assert t.observe("A", 0, "0xw3", 1300) is False   # already triggered


def test_tracker_window_prunes_old_buys():
    t = ConsensusTracker(k=2, window_seconds=60)
    assert t.observe("A", 0, "0xw1", 1000) is False
    # second wallet far outside the 60s window -> w1 pruned, still <2 distinct
    assert t.observe("A", 0, "0xw2", 5000) is False


def _ev(cid, idx, price, usd, ts, title="Will Trump win election"):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid, "outcomeIndex": idx,
            "price": price, "usdcSize": usd, "size": usd / price, "timestamp": ts,
            "title": title}


class ConsClient:
    data_api = "https://x"

    def __init__(self):
        # w1+w2 both buy market A (consensus); w3 alone buys B (no consensus)
        self._by_user = {
            "0xw1": [_ev("A", 0, 0.50, 200, 1000)],
            "0xw2": [_ev("A", 0, 0.50, 200, 2000)],
            "0xw3": [_ev("B", 0, 0.50, 200, 1000)],
        }

    def _get(self, url, params):
        if "activity" in url and params.get("offset", 0) == 0:
            return self._by_user.get(params.get("user"), [])
        return []

    def get_market(self, cid):
        return Market(condition_id=cid, closed=True, outcomes=["Y", "N"],
                      outcome_prices=[1.0, 0.0])  # idx0 wins


def test_backtest_consensus_requires_agreement():
    c = ConsClient()
    r = backtest_consensus(c, ["0xw1", "0xw2", "0xw3"], k=2, window_hours=24,
                           stake=50, min_usd=100)
    # only market A reached 2 wallets -> exactly one copy, and it won
    assert r.n == 1
    assert r.wins == 1
