"""Offline tests for Phase-2 walk-forward validation (no network)."""

from polywatch import walkforward as wf
from polywatch.models import Market

DAY = 86400
T = 1_700_000_000  # the fold cutoff


def _buy(cid, idx, price, usd, ts):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid,
            "outcomeIndex": idx, "price": price, "usdcSize": usd,
            "size": usd / price, "timestamp": ts, "title": cid, "slug": cid}


def _mkt(cid, win_idx, end_iso):
    prices = [1.0, 0.0] if win_idx == 0 else [0.0, 1.0]
    return Market(condition_id=cid, closed=True, outcomes=["Y", "N"],
                  outcome_prices=prices, end_date=end_iso)


PRE = "2023-10-01T00:00:00Z"   # before T (T ~ 2023-11-14)
POST = "2024-02-01T00:00:00Z"  # after T


def _scenario():
    """Skilled wallet wins longshots in both train and test; bad wallet loses;
    leak wallet only has markets that resolve AFTER the cutoff."""
    ev = {}
    mk = {}

    # skilled wallet: longshot winners, train (<=T) and test (>T)
    sk = []
    for j in range(5):
        cid = f"SK_tr_{j}"
        sk.append(_buy(cid, 0, 0.20 + 0.02 * j, 200, T - (40 - j) * DAY))
        mk[cid] = _mkt(cid, 0, PRE)        # resolved before T -> usable in train
    for j in range(5):
        cid = f"SK_te_{j}"
        sk.append(_buy(cid, 0, 0.25 + 0.03 * j, 200, T + (5 + j) * DAY))
        mk[cid] = _mkt(cid, 0, POST)       # resolves after T -> test only
    ev["0xskill"] = sk

    # bad wallet: buys favorites that lose, both periods
    bad = []
    for j in range(5):
        cid = f"BD_tr_{j}"
        bad.append(_buy(cid, 0, 0.60, 200, T - (40 - j) * DAY))
        mk[cid] = _mkt(cid, 1, PRE)        # idx0 bought, idx1 wins -> loss
    for j in range(5):
        cid = f"BD_te_{j}"
        bad.append(_buy(cid, 0, 0.62, 200, T + (5 + j) * DAY))
        mk[cid] = _mkt(cid, 1, POST)
    ev["0xbad"] = bad

    # leak wallet: 'wins' but every market resolves AFTER T
    leak = []
    for j in range(6):
        cid = f"LK_tr_{j}"
        leak.append(_buy(cid, 0, 0.20, 200, T - (30 - j) * DAY))  # bought before T
        mk[cid] = _mkt(cid, 0, POST)       # but resolves AFTER T
    ev["0xleak"] = leak

    return ev, mk


def test_end_ts_parsing():
    m = _mkt("X", 0, "2024-02-01T00:00:00Z")
    assert wf._end_ts(m) == 1706745600
    assert wf._end_ts(_mkt("Y", 0, "")) is None


def test_train_cache_hides_post_cutoff_markets():
    _ev, mk = _scenario()
    view = wf._train_cache(mk, T)
    assert view["SK_tr_0"] is not None       # resolved pre-T -> visible
    assert view["SK_te_0"] is None           # resolves post-T -> hidden
    assert view["LK_tr_0"] is None           # leak market hidden in training


def test_leak_wallet_unscoreable_at_cutoff():
    # All of leak's markets resolve after T, so a point-in-time score sees
    # nothing settled -> None (no look-ahead credit).
    ev, mk = _scenario()
    from polywatch.scorer import score_wallet
    tcache = wf._train_cache(mk, T)
    ws = score_wallet(None, "0xleak", tcache, events=ev["0xleak"], cutoff_ts=T, now_ts=T)
    assert ws is None


def test_dedupe_portfolio_keeps_earliest():
    recs = [{"cid": "A", "idx": 0, "ts": 200, "net": 1},
            {"cid": "A", "idx": 0, "ts": 100, "net": 2},
            {"cid": "B", "idx": 1, "ts": 50, "net": 3}]
    out = wf._dedupe_portfolio(recs)
    assert len(out) == 2
    a = [r for r in out if r["cid"] == "A"][0]
    assert a["ts"] == 100   # earliest kept


def test_strategy_stats_basic():
    recs = [{"net": 10, "won": True, "alpha": 0.5, "cat": "crypto", "price": 0.5},
            {"net": -15, "won": False, "alpha": -0.3, "cat": "sports", "price": 0.3}]
    s = wf._strategy_stats("x", recs, stake=15.0)
    assert s.n == 2
    assert abs(s.total_net - (-5)) < 1e-9
    assert "crypto" in s.by_category and "sports" in s.by_category


def test_walk_forward_selects_skill_and_beats_favorite():
    ev, mk = _scenario()
    client = _FakeClient(ev, mk)
    targets = ["0xskill", "0xbad", "0xleak"]
    rep = wf.walk_forward(client, targets, cutoffs=[T], top_k=1,
                          min_train_settled=3, stake=15.0, min_usd=100.0,
                          entry_min=0.05, entry_max=0.95, workers=4,
                          cache_path=None)
    # skilled wallet should be the selection; leak wallet must not be
    assert rep.folds and "0xskill" in rep.folds[0].selected
    assert "0xleak" not in rep.folds[0].selected
    # out-of-sample copies exist and are net positive
    assert rep.selected.n > 0
    assert rep.selected.total_net > 0
    # beats the structural favorite baseline (driven by the losing favorites)
    assert rep.selected.roi > rep.favorite.roi
    # alpha vs price is positive for the skilled selection
    assert rep.selected.mean_alpha > 0
    # verdict dict is fully populated
    for k in ("t_gt_3", "dsr_gt_95", "beats_favorite_roi", "beats_random_ret",
              "positive_net", "PASS"):
        assert k in rep.verdict
    # DSR is now informative (var_sr estimated from longer in-sample tracks),
    # not pinned at None/0 by tiny-sample noise.
    assert rep.selected.dsr is not None
    assert rep.var_sr > 0


def test_csv_export(tmp_path):
    ev, mk = _scenario()
    rep = wf.walk_forward(_FakeClient(ev, mk), ["0xskill", "0xbad", "0xleak"],
                          cutoffs=[T], top_k=1, min_train_settled=3, min_usd=100.0)
    assert "selected" in rep.records and "favorite" in rep.records
    p = tmp_path / "out.csv"
    n = wf.write_walkforward_csv(rep, str(p))
    assert n > 0
    txt = p.read_text()
    assert txt.splitlines()[0].startswith("strategy,fold,wallet,cid,category")
    assert "selected" in txt


def test_auto_cutoffs_reserves_resolution_window():
    now = T + 100 * DAY
    # 3 old trades (70-90d ago) + 3 very recent (1-5d ago)
    ages = [90, 80, 70, 5, 3, 1]
    ev = {"w": [_buy(f"c{i}", 0, 0.30, 200, now - d * DAY) for i, d in enumerate(ages)]}
    cuts = wf._auto_cutoffs(ev, folds=2, entry_min=0.05, entry_max=0.95,
                            min_usd=100, now_ts=now, reserve_days=21)
    assert cuts                                   # produced cutoffs
    # none may fall inside the reserved (unresolvable) window near 'now'
    assert all(c <= now - 21 * DAY for c in cuts)


class _FakeClient:
    data_api = "https://x"

    def __init__(self, events_by_wallet, markets):
        self._e = events_by_wallet
        self._m = markets

    def _get(self, url, params):
        if params.get("offset", 0) != 0:
            return []
        return self._e.get(params.get("user"), [])

    def get_markets(self, cids, closed=True):
        return {c: self._m.get(c) for c in cids}

    def get_market(self, cid):
        return self._m.get(cid)


def test_cache_roundtrip_preserves_end_date(tmp_path):
    # Regression: _save_cache must persist end_date. Without it, a warm cache
    # rebuilds markets with end_date="" -> _train_cache hides every market ->
    # every training fold is silently empty.
    from polywatch.backtest import _save_cache, _load_cache
    p = str(tmp_path / "cache.json")
    _save_cache(p, {"X": _mkt("X", 0, "2024-02-01T00:00:00Z")})
    back = _load_cache(p)
    assert back["X"].end_date == "2024-02-01T00:00:00Z"
    assert wf._end_ts(back["X"]) == 1706745600


def test_warm_cache_still_selects(tmp_path):
    # A second run reusing the on-disk cache must select identically to the first.
    ev, mk = _scenario()
    common = dict(cutoffs=[T], top_k=1, min_train_settled=3, min_usd=100.0,
                  workers=4, cache_path=str(tmp_path / "markets_cache.json"))
    r1 = wf.walk_forward(_FakeClient(ev, mk), ["0xskill", "0xbad", "0xleak"], **common)
    r2 = wf.walk_forward(_FakeClient(ev, mk), ["0xskill", "0xbad", "0xleak"], **common)
    assert r1.selected.n > 0 and r2.selected.n == r1.selected.n
    assert "0xskill" in r2.folds[0].selected
