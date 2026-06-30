"""Offline tests for the internal-arb scanner (no network)."""

from polywatch import arb


def _book(ask, ask_size=100.0, bid=None, bid_size=0.0):
    return {"best_ask": ask, "ask_size": ask_size,
            "best_bid": bid, "bid_size": bid_size}


class FakeBookClient:
    def __init__(self, markets, books):
        self._m = markets
        self._b = books

    def get_top_markets(self, limit=200):
        return self._m

    def get_book(self, token_id):
        return self._b.get(token_id)

    def close(self):
        pass


def _markets():
    return [
        {"conditionId": "A", "question": "Market A", "clobTokenIds": '["A_y","A_n"]'},
        {"conditionId": "B", "question": "Market B", "clobTokenIds": '["B_y","B_n"]'},
        {"conditionId": "C", "question": "Multi", "clobTokenIds": '["C1","C2","C3"]'},
    ]


def _books():
    return {
        "A_y": _book(0.40, 100), "A_n": _book(0.55, 80),   # sum 0.95 -> edge 0.05, size 80
        "B_y": _book(0.52, 100), "B_n": _book(0.50, 100),  # sum 1.02 -> no arb
        "C1": _book(0.3), "C2": _book(0.3), "C3": _book(0.3),  # 3 tokens -> skipped
    }


def test_parse_tokens():
    assert arb._parse_tokens('["x","y"]') == ["x", "y"]
    assert arb._parse_tokens(["x", "y"]) == ["x", "y"]
    assert arb._parse_tokens(None) == []
    assert arb._parse_tokens("not json") == []


def test_scan_finds_only_real_gap():
    c = FakeBookClient(_markets(), _books())
    opps = arb.scan_internal_arb(c, n_markets=10, fee=0.0, min_edge=0.0)
    assert len(opps) == 1                       # only A; B too expensive, C non-binary
    o = opps[0]
    assert o.cid == "A"
    assert abs(o.edge_per_pair - 0.05) < 1e-9
    assert abs(o.cost - 0.95) < 1e-9
    assert o.size == 80                         # min(100, 80)
    assert abs(o.max_profit - 0.05 * 80) < 1e-9


def test_fee_and_min_edge_filter():
    c = FakeBookClient(_markets(), _books())
    # a 6c fee wipes out the 5c edge -> nothing
    assert arb.scan_internal_arb(c, fee=0.06) == []
    # min-edge above the gap -> nothing
    assert arb.scan_internal_arb(c, min_edge=0.10) == []
    # small fee still leaves a gap
    assert len(arb.scan_internal_arb(c, fee=0.01)) == 1


def test_no_gaps_when_books_efficient():
    books = {"A_y": _book(0.50), "A_n": _book(0.50),
             "B_y": _book(0.60), "B_n": _book(0.45)}
    c = FakeBookClient(_markets()[:2], books)
    assert arb.scan_internal_arb(c) == []        # 1.00 and 1.05 -> no lock


def test_lifetime_tracker():
    t = arb.LifetimeTracker()

    class O:
        def __init__(self, cid, edge):
            self.cid = cid
            self.edge_per_pair = edge

    t.observe([O("A", 0.05), O("B", 0.02)], ts=100)
    t.observe([O("A", 0.07)], ts=130)            # A persists, B gone
    rows = {r["cid"]: r for r in t.rows()}
    assert rows["A"]["lifetime_s"] == 30
    assert rows["A"]["samples"] == 2
    assert abs(rows["A"]["best_edge"] - 0.07) < 1e-9
    assert rows["B"]["lifetime_s"] == 0          # seen once -> gone within the poll
