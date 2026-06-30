from polywatch.categories import categorize, passes_category_filter
from polywatch.backtest import backtest_copy
from polywatch.models import Market


def test_categorize_basics():
    assert categorize("Will Brazil win the 2026 FIFA World Cup?") == "sports"
    assert categorize("Bitcoin Up or Down - 5m") == "crypto"
    assert categorize("Will Trump win the 2028 election?") == "politics"
    assert categorize("Fed rate cut in March?") == "econ"
    assert categorize("Some unrelated topic here") == "other"


def test_passes_filter():
    assert passes_category_filter("sports", block=["sports"]) is False
    assert passes_category_filter("politics", block=["sports"]) is True
    assert passes_category_filter("politics", allow=["politics"]) is True
    assert passes_category_filter("sports", allow=["politics"]) is False


def _ev(cid, title, idx, price, usd):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid, "outcomeIndex": idx,
            "price": price, "usdcSize": usd, "size": usd / price, "title": title}


class CatClient:
    data_api = "https://x"

    def _get(self, url, params):
        if params.get("offset", 0) != 0:
            return []
        return [
            _ev("S", "Brazil vs Japan", 0, 0.50, 200),       # sports, wins
            _ev("P", "Will Trump win election", 0, 0.50, 200),  # politics, loses
        ]

    def get_market(self, cid):
        prices = [1.0, 0.0] if cid == "S" else [0.0, 1.0]
        return Market(condition_id=cid, closed=True, outcomes=["Y", "N"], outcome_prices=prices)


def test_backtest_per_category_and_block():
    full = backtest_copy(CatClient(), ["0xw"], stake=50, min_usd=100)
    assert full.n == 2
    assert set(full.by_category) == {"sports", "politics"}
    # blocking sports leaves only the politics copy
    blocked = backtest_copy(CatClient(), ["0xw"], stake=50, min_usd=100, block=["sports"])
    assert blocked.n == 1
    assert set(blocked.by_category) == {"politics"}
