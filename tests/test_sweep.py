"""Offline tests for the strategy sweep (no network)."""

from polywatch import sweep
from polywatch.models import Market

DAY = 86400
T = 1_700_000_000


def _buy(cid, idx, price, usd, ts):
    return {"type": "TRADE", "side": "BUY", "conditionId": cid,
            "outcomeIndex": idx, "price": price, "usdcSize": usd,
            "size": usd / price, "timestamp": ts, "title": cid, "slug": cid}


def _mkt(cid, win_idx, end_iso):
    prices = [1.0, 0.0] if win_idx == 0 else [0.0, 1.0]
    return Market(condition_id=cid, closed=True, outcomes=["Y", "N"],
                  outcome_prices=prices, end_date=end_iso)


PRE = "2023-10-01T00:00:00Z"
POST = "2024-02-01T00:00:00Z"


def _scenario():
    ev, mk = {}, {}
    sk = []
    for j in range(5):
        cid = f"SK_tr_{j}"
        sk.append(_buy(cid, 0, 0.20 + 0.02 * j, 200, T - (40 - j) * DAY))
        mk[cid] = _mkt(cid, 0, PRE)
    for j in range(5):
        cid = f"SK_te_{j}"
        sk.append(_buy(cid, 0, 0.25 + 0.03 * j, 200, T + (5 + j) * DAY))
        mk[cid] = _mkt(cid, 0, POST)
    ev["0xskill"] = sk
    bad = []
    for j in range(5):
        cid = f"BD_tr_{j}"
        bad.append(_buy(cid, 0, 0.60, 200, T - (40 - j) * DAY))
        mk[cid] = _mkt(cid, 1, PRE)
    for j in range(5):
        cid = f"BD_te_{j}"
        bad.append(_buy(cid, 0, 0.62, 200, T + (5 + j) * DAY))
        mk[cid] = _mkt(cid, 1, POST)
    ev["0xbad"] = bad
    return ev, mk


class _FakeClient:
    data_api = "https://x"

    def __init__(self, ev, mk):
        self._e, self._m = ev, mk

    def _get(self, url, params):
        if params.get("offset", 0) != 0:
            return []
        return self._e.get(params.get("user"), [])

    def get_markets(self, cids, closed=True):
        return {c: self._m.get(c) for c in cids}

    def get_market(self, cid):
        return self._m.get(cid)


def test_consensus_needs_two_distinct_wallets():
    cid = "CONS_1"
    ev = {"0xa": [_buy(cid, 0, 0.40, 200, T + 10 * DAY)],
          "0xb": [_buy(cid, 0, 0.40, 200, T + 10 * DAY + 3600)]}
    mk = {cid: _mkt(cid, 0, POST)}
    recs = sweep.consensus_records(ev, mk, k=2, window_hours=24, min_usd=100.0)
    assert len(recs) == 1 and recs[0]["won"] is True          # K=2 reached once
    solo = sweep.consensus_records({"0xa": ev["0xa"]}, mk, k=2, min_usd=100.0)
    assert solo == []                                         # one wallet never triggers


def test_consensus_window_excludes_late_second_buy():
    cid = "CONS_2"
    ev = {"0xa": [_buy(cid, 0, 0.40, 200, T)],
          "0xb": [_buy(cid, 0, 0.40, 200, T + 5 * DAY)]}     # 5 days apart
    mk = {cid: _mkt(cid, 0, POST)}
    assert sweep.consensus_records(ev, mk, k=2, window_hours=24, min_usd=100.0) == []


def test_run_sweep_rows_and_writers(tmp_path):
    ev, mk = _scenario()
    res = sweep.run_sweep(_FakeClient(ev, mk), ["0xskill", "0xbad"],
                          cutoffs=[T], top_k=1, min_train_settled=3,
                          consensus_k=2, min_usd=100.0, workers=4, cache_path=None)
    assert [r.key for r in res.rows] == \
        ["copy_all", "selected", "consensus", "favorite", "random"]
    flags = {r.key: r.is_baseline for r in res.rows}
    assert flags["favorite"] and flags["random"]
    assert not flags["selected"] and not flags["consensus"]

    md = tmp_path / "RESULTS.md"
    csvp = tmp_path / "results.csv"
    sweep.write_results_md(res, str(md))
    assert sweep.write_sweep_csv(res, str(csvp)) == 5
    text = md.read_text()
    assert "## Strategies" in text and "Copy everything" in text
    assert csvp.read_text().splitlines()[0].startswith("strategy,key,is_baseline")
    assert "strategy sweep" in sweep.format_sweep(res)


def test_row_verdict_marks_baselines():
    ev, mk = _scenario()
    res = sweep.run_sweep(_FakeClient(ev, mk), ["0xskill", "0xbad"],
                          cutoffs=[T], top_k=1, min_train_settled=3, min_usd=100.0,
                          workers=4, cache_path=None)
    by_key = {r.key: sweep.row_verdict(r) for r in res.rows}
    assert by_key["favorite"] == "baseline" and by_key["random"] == "baseline"
    assert by_key["selected"] in ("PASS", "FAIL")
