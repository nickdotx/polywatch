"""Offline tests for the Kalshi client helpers + cross-platform arb math."""

from polywatch import crossarb
from polywatch.kalshi import KalshiClient, kalshi_fee


# --- Kalshi orderbook reciprocal (asks derived from bids) ---------------

def test_kalshi_orderbook_derives_asks_from_bids():
    class FakeHTTP:
        def get(self, url, params=None):
            class R:
                status_code = 200
                def raise_for_status(self): pass
                def json(self):
                    # YES bids best 0.45, NO bids best 0.50
                    return {"orderbook": {"yes_dollars": [[0.40, 10], [0.45, 5]],
                                          "no_dollars": [[0.50, 7], [0.30, 3]]}}
            return R()
    k = KalshiClient.__new__(KalshiClient)
    k._client = FakeHTTP()
    from polywatch.kalshi import _RateLimiter
    k._limiter = _RateLimiter()
    k.base = "x"
    k.retries = 1
    ob = k.get_orderbook("T")
    # yes_ask = 1 - best_no_bid(0.50) = 0.50 ; no_ask = 1 - best_yes_bid(0.45) = 0.55
    assert abs(ob["yes_ask"] - 0.50) < 1e-9
    assert abs(ob["no_ask"] - 0.55) < 1e-9
    assert ob["yes_ask_size"] == 7        # size of the NO bid we lift
    assert ob["no_ask_size"] == 5


def test_kalshi_fee_peaks_midbook():
    assert kalshi_fee(0.5) > kalshi_fee(0.1)        # biggest near 50c
    assert abs(kalshi_fee(0.5) - 0.0175) < 1e-9     # 0.07*0.25


# --- cross-platform arb math -------------------------------------------

class _Market:
    def __init__(self, toks):
        self.clob_token_ids = toks


class FakePoly:
    def __init__(self, books):
        self._b = books
    def get_market(self, cid):
        return _Market(["YES_TOK", "NO_TOK"])
    def get_book(self, tok):
        return self._b.get(tok)
    def close(self):
        pass


class FakeKalshi:
    def __init__(self, ob):
        self._ob = ob
    def get_orderbook(self, ticker):
        return self._ob
    def close(self):
        pass


def _pbook(ask, size=100):
    return {"best_ask": ask, "ask_size": size}


def test_cross_lock_found_when_venues_disagree():
    # Polymarket: YES ask 0.40, NO ask 0.62. Kalshi: YES ask 0.55, NO ask 0.46.
    # Cheapest YES = Polymarket 0.40 ; cheapest NO = Kalshi 0.46 -> cost 0.86 (+fees)
    poly = FakePoly({"YES_TOK": _pbook(0.40, 100), "NO_TOK": _pbook(0.62, 50)})
    kal = FakeKalshi({"yes_ask": 0.55, "yes_ask_size": 80,
                      "no_ask": 0.46, "no_ask_size": 30})
    pair = {"name": "Test", "polymarket_condition_id": "0x1",
            "polymarket_yes_index": 0, "kalshi_ticker": "T", "invert": False}
    o = crossarb.scan_pair(poly, kal, pair, poly_fee=0.0, kalshi_rate=0.07)
    assert o is not None
    assert o.yes_leg == "polymarket" and o.no_leg == "kalshi"
    # cost = 0.40 + (0.46 + 0.07*0.46*0.54)  ~= 0.40 + 0.4774 = 0.8774
    assert abs(o.cost - (0.40 + 0.46 + kalshi_fee(0.46))) < 1e-9
    assert o.edge_per_pair > 0.10
    assert o.size == 30                    # min(100 YES, 30 NO)


def test_no_lock_when_venues_agree():
    # both venues ~ same prices, YES+NO ~ 1 each -> no cross edge
    poly = FakePoly({"YES_TOK": _pbook(0.50), "NO_TOK": _pbook(0.50)})
    kal = FakeKalshi({"yes_ask": 0.50, "yes_ask_size": 100,
                      "no_ask": 0.50, "no_ask_size": 100})
    pair = {"polymarket_condition_id": "0x1", "kalshi_ticker": "T"}
    assert crossarb.scan_pair(poly, kal, pair) is None


def test_discover_matches_similar_titles_only():
    poly = [
        {"conditionId": "0xA", "question": "Will the Fed cut interest rates in July 2026?"},
        {"conditionId": "0xB", "question": "Will Bitcoin reach 200000 by December?"},
    ]
    kalshi = [
        {"ticker": "KXFED-JUL", "title": "Fed interest rate decision July",
         "subtitle": "25 bps cut", "event_ticker": "KXFED"},
        {"ticker": "KXNBA", "title": "NBA Finals champion", "subtitle": "",
         "event_ticker": "KXNBA"},
    ]
    cands = crossarb.discover_pairs(poly, kalshi, min_score=0.1, top=10)
    fed = [c for c in cands if c["poly_cid"] == "0xA"]
    assert fed and fed[0]["kalshi_ticker"] == "KXFED-JUL"   # Fed <-> Fed matched
    assert all(c["poly_cid"] != "0xB" for c in cands)        # bitcoin: no Kalshi match
    # the formatters don't crash and produce review output
    assert "REVIEW" in crossarb.format_discover(cands)
    assert "polymarket_condition_id" in crossarb.discover_yaml(cands)


def test_invert_swaps_kalshi_sides():
    # Polymarket-YES == Kalshi-NO. Kalshi NO ask is the cheap leg for our YES.
    poly = FakePoly({"YES_TOK": _pbook(0.70), "NO_TOK": _pbook(0.45)})
    kal = FakeKalshi({"yes_ask": 0.66, "yes_ask_size": 40,    # = our NO
                      "no_ask": 0.38, "no_ask_size": 25})     # = our YES
    pair = {"polymarket_condition_id": "0x1", "kalshi_ticker": "T", "invert": True}
    o = crossarb.scan_pair(poly, kal, pair, kalshi_rate=0.0)
    assert o is not None
    # our YES cheapest: kalshi(no_ask 0.38) vs poly(0.70) -> kalshi 0.38
    # our NO cheapest: poly(0.45) vs kalshi(yes_ask 0.66) -> poly 0.45
    assert o.yes_leg == "kalshi" and o.no_leg == "polymarket"
    assert abs(o.cost - (0.38 + 0.45)) < 1e-9
