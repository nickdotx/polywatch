from polywatch import scorer
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


def _split(cid, usd, size): return {"type": "SPLIT", "conditionId": cid, "usdcSize": usd, "size": size, "timestamp": 1700000000}
def _sell(cid, idx, usd, size): return {"type": "TRADE", "side": "SELL", "conditionId": cid, "outcomeIndex": idx, "usdcSize": usd, "size": size, "timestamp": 1700000001}
def _buy(cid, idx, usd, size, price): return {"type": "TRADE", "side": "BUY", "conditionId": cid, "outcomeIndex": idx, "usdcSize": usd, "size": size, "price": price, "timestamp": 1700000000}
def _redeem(cid, usd, size): return {"type": "REDEEM", "conditionId": cid, "usdcSize": usd, "size": size, "timestamp": 1700000002}


def test_split_sell_one_leg_hold_winner():
    # split $100 -> 100 Yes + 100 No; sell No for $40; Yes wins -> +40
    ev = [_split("A", 100, 100), _sell("A", 1, 40, 100)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    assert round(recs[0]["realized"], 2) == 40.0


def test_split_hold_both_to_resolution():
    # split $100, hold both legs, Yes wins -> realized 0 (paid 100, get 100)
    ev = [_split("A", 100, 100)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    assert abs(recs[0]["realized"]) < 1e-6


def test_buy_then_redeem_winner_still_correct():
    # buy 100 Yes @0.40 ($40), redeem 100 -> +60 (regression: old behavior preserved)
    ev = [_buy("A", 0, 40, 100, 0.40), _redeem("A", 100, 100)]
    recs = scorer.reconstruct(_client(ev, win_idx=0), "0xw", {})
    assert len(recs) == 1
    assert round(recs[0]["realized"], 2) == 60.0
