from polywatch.costs import CostModel
from polywatch.backtest import backtest_copy
from polywatch.models import Market


def test_costmodel_slippage_and_fee_reduce_pnl():
    zero = CostModel()
    assert zero.net_pnl(0.50, 100, True) == 100.0      # 200 shares -1.. (100 profit) gross
    assert zero.net_pnl(0.50, 100, False) == -100.0
    costly = CostModel(slippage_points=0.02, fee_pct=0.01)
    # win: fewer shares (entry 0.52) and a fee -> strictly less than gross 100
    assert costly.net_pnl(0.50, 100, True) < 100.0
    # loss is slightly worse than -stake due to entry-side fee
    assert costly.net_pnl(0.50, 100, False) <= -100.0


def _ev(cid, idx, price, usd):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid, "outcomeIndex": idx,
            "price": price, "usdcSize": usd, "size": usd / price}


class FakeClient:
    data_api = "https://x"
    def _get(self, url, params):
        return [_ev("A", 0, 0.50, 200), _ev("B", 0, 0.50, 200)] if params.get("offset", 0) == 0 else []
    def get_market(self, cid):
        prices = [1.0, 0.0] if cid == "A" else [0.0, 1.0]   # A wins, B loses
        return Market(condition_id=cid, closed=True, outcomes=["Y", "N"], outcome_prices=prices)


def test_backtest_net_below_gross_with_costs():
    client = FakeClient()
    r = backtest_copy(client, ["0xw"], stake=100, min_usd=100,
                      costs=CostModel(slippage_points=0.02, fee_pct=0.01))
    assert r.n == 2
    assert r.pnl < r.pnl_gross           # costs drag net below gross
