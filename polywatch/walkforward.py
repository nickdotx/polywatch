"""Walk-forward validation (Phase 2) — the honest out-of-sample test.

The existing backtest is IN-SAMPLE / circular: wallets are picked because they
won, then "discovered" to win when copied. This module breaks that loop the way
Pardo's walk-forward analysis and Lopez de Prado's purged CV prescribe:

  1. Pull each wallet's full /activity ONCE.
  2. At each fold cutoff T, SCORE and SELECT wallets using only data <= T
     (and only markets that had RESOLVED by T -> no resolution look-ahead).
  3. Measure copy P&L only on those wallets' trades AFTER T (+ an optional
     embargo), against markets that have since resolved.
  4. Haircut the result for multiple testing (we score hundreds of wallets)
     with the Deflated Sharpe Ratio, and demand a t-stat > 3 (Harvey et al.).
  5. Require it to beat two skill-free baselines: a structural FAVORITE bet
     (favorite-longshot bias) and RANDOM wallet selection.

Output is a verdict: does selecting wallets by our scorer produce a real,
out-of-sample, cost-aware edge that survives selection bias and beats dumb
baselines? If not, no live money is justified.
"""

from __future__ import annotations

import itertools
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import stats
from .backtest import (_load_cache, _save_cache, _qualifying_buys,
                       copy_records, favorite_records)
from .costs import CostModel
from .scorer import _pull_activity, score_wallet

DAY = 86400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _end_ts(market) -> int | None:
    """Parse a Market.end_date (ISO) to a unix timestamp; None if unparseable.
    Used as a 'resolved by T' proxy for point-in-time training scores."""
    s = getattr(market, "end_date", "") or ""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def _pull_all(client, addrs, max_events, workers, progress=None):
    out = {}

    def _pull(addr):
        try:
            return addr, _pull_activity(client, addr, max_events)
        except Exception:
            return addr, []

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for addr, events in ex.map(_pull, addrs):
            out[addr] = events
            done += 1
            if progress:
                progress(done, len(addrs), phase="fetch")
    return out


def _resolve_markets(events_by_wallet, entry_min, entry_max, min_usd,
                     cache_path, client, workers=16, progress=None):
    """Resolve every qualifying-buy market once, in parallel.

    This is the only market-resolution network pass in the walk-forward: scoring
    runs OFFLINE against this cache, so we never fetch per wallet. Chunked 40/req
    and fanned out across `workers` so a big universe (thousands of markets) is
    network-latency-bound, not serial."""
    resolved = _load_cache(cache_path)
    needed = set()
    for events in events_by_wallet.values():
        for cid, _, _, _, _, _ in _qualifying_buys(events, entry_min, entry_max, min_usd):
            if cid not in resolved:
                needed.add(cid)
    needed = list(needed)
    if needed:
        if hasattr(client, "get_markets"):
            CH = 40
            chunks = [needed[i:i + CH] for i in range(0, len(needed), CH)]
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for part in ex.map(lambda c: client.get_markets(c, closed=True), chunks):
                    resolved.update(part or {})
                    done += 1
                    if progress:
                        progress(min(done * CH, len(needed)), len(needed), phase="resolve")
            for cid in needed:                 # cids with no gamma row -> mark None
                resolved.setdefault(cid, None)
        else:
            for cid in needed:
                resolved[cid] = client.get_market(cid)
        _save_cache(cache_path, resolved)
    return resolved


def _all_buy_ts(events_by_wallet, entry_min, entry_max, min_usd):
    ts = []
    for events in events_by_wallet.values():
        for _cid, _idx, _p, _t, _s, t in _qualifying_buys(events, entry_min, entry_max, min_usd):
            ts.append(t)
    return sorted(ts)


def _auto_cutoffs(events_by_wallet, folds, entry_min, entry_max, min_usd,
                  now_ts=None, reserve_days=21):
    """Pick `folds` interior cutoffs that split the trade timeline so each fold
    has a meaningful train and test side. Quantile-based on observed buy times.

    Critically, cutoffs are capped at `now_ts - reserve_days`: a cutoff placed
    too close to 'now' leaves no calendar time for post-cutoff trades to RESOLVE,
    so the fold's out-of-sample set is empty (the bug seen when folds landed ~2
    days before today). Reserving a window guarantees every fold can be scored."""
    ts = _all_buy_ts(events_by_wallet, entry_min, entry_max, min_usd)
    if now_ts is not None:
        cap = now_ts - reserve_days * DAY
        ts = [t for t in ts if t <= cap]
    if len(ts) < (folds + 1):
        return []
    cuts = []
    # interior quantiles: e.g. folds=3 -> 0.25, 0.50, 0.75 of the timeline
    for i in range(1, folds + 1):
        q = i / (folds + 1)
        idx = min(len(ts) - 1, int(q * len(ts)))
        cuts.append(ts[idx])
    seen = set()
    uniq = []
    for c in cuts:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _train_cache(resolved, cutoff):
    """A resolution view as of `cutoff`: markets whose end date is after the
    cutoff are hidden (mapped to None) so they cannot leak their outcome into a
    point-in-time training score. All touched cids remain keys -> no refetch."""
    view = {}
    for cid, m in resolved.items():
        if m is None:
            view[cid] = None
            continue
        e = _end_ts(m)
        view[cid] = m if (e is not None and e <= cutoff) else None
    return view


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    name: str
    n: int = 0
    total_net: float = 0.0
    total_stake: float = 0.0
    win_rate: float = 0.0
    mean_ret: float = 0.0
    sharpe: float = 0.0
    t_stat: float = 0.0
    mean_alpha: float = 0.0
    alpha_t: float = 0.0
    psr: float = 0.0
    dsr: float | None = None
    by_category: dict = field(default_factory=dict)

    @property
    def roi(self):
        return self.total_net / self.total_stake if self.total_stake else 0.0


@dataclass
class FoldResult:
    idx: int
    cutoff: int
    n_candidates: int
    n_selected: int
    selected: list = field(default_factory=list)
    oos_n: int = 0
    oos_net: float = 0.0
    oos_stake: float = 0.0

    @property
    def oos_roi(self):
        return self.oos_net / self.oos_stake if self.oos_stake else 0.0


@dataclass
class WFReport:
    selected: StrategyStats
    favorite: StrategyStats
    random_sel: StrategyStats
    all_copy: StrategyStats
    folds: list = field(default_factory=list)
    n_trials: int = 0
    var_sr: float = 0.0
    params: dict = field(default_factory=dict)
    verdict: dict = field(default_factory=dict)
    records: dict = field(default_factory=dict)   # raw per-copy rows, for CSV export


# ---------------------------------------------------------------------------
# stats assembly
# ---------------------------------------------------------------------------

def _strategy_stats(name, records, stake, *, n_trials=None, var_sr=None):
    s = StrategyStats(name=name)
    s.n = len(records)
    if not records:
        return s
    rets = [r["net"] / stake for r in records]
    alphas = [r["alpha"] for r in records]
    s.total_net = sum(r["net"] for r in records)
    s.total_stake = stake * len(records)
    s.win_rate = sum(1 for r in records if r["won"]) / len(records)
    ss = stats.SeriesStats.of(rets)
    s.mean_ret = ss.mean
    s.sharpe = ss.sharpe
    s.t_stat = ss.t_stat
    s.mean_alpha = stats.mean(alphas)
    s.alpha_t = stats.t_stat(alphas)
    s.psr = stats.probabilistic_sharpe_ratio(ss.sharpe, ss.n, ss.skew, ss.kurt, 0.0)
    if n_trials and var_sr is not None:
        s.dsr = stats.deflated_sharpe_ratio(ss.sharpe, ss.n, n_trials, var_sr,
                                            ss.skew, ss.kurt)
    cat = {}
    for r in records:
        d = cat.setdefault(r["cat"], {"n": 0, "wins": 0, "net": 0.0, "stake": 0.0, "rets": []})
        d["n"] += 1
        d["net"] += r["net"]
        d["stake"] += stake
        d["rets"].append(r["net"] / stake)
        if r["won"]:
            d["wins"] += 1
    for d in cat.values():
        d["t"] = stats.t_stat(d["rets"])   # per-category significance
        del d["rets"]
    s.by_category = cat
    return s


def _dedupe_portfolio(records):
    """One copy per (market, outcome) across the selected portfolio — mirror the
    live `dedupe_market_outcome` control. Keep the chronologically first."""
    best = {}
    for r in sorted(records, key=lambda x: x["ts"]):
        key = (r["cid"], r["idx"])
        if key not in best:
            best[key] = r
    return list(best.values())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def walk_forward(client, targets, *, folds=3, cutoffs=None, top_k=10,
                 min_train_settled=20, embargo_days=0, stake=15.0, costs=None,
                 entry_min=0.05, entry_max=0.95, min_usd=100.0, allow=None,
                 block=None, max_events=5000, workers=16, cache_path=None,
                 seed=0, reserve_days=21, now_ts=None, progress=None,
                 events_by_wallet=None, resolved=None):
    costs = costs or CostModel()
    embargo = int(embargo_days * DAY)
    now_ref = time.time() if now_ts is None else now_ts
    addrs = [getattr(t, "address", t) for t in targets]

    # Pull/resolve once. Callers (e.g. the sweep) can pass pre-fetched data so a
    # multi-strategy run doesn't hit the network twice.
    if events_by_wallet is None:
        events_by_wallet = _pull_all(client, addrs, max_events, workers, progress)
    if resolved is None:
        resolved = _resolve_markets(events_by_wallet, entry_min, entry_max, min_usd,
                                    cache_path, client, workers=workers, progress=progress)

    if cutoffs is None:
        cutoffs = _auto_cutoffs(events_by_wallet, folds, entry_min, entry_max,
                                min_usd, now_ts=now_ref, reserve_days=reserve_days)
    cutoffs = sorted(set(int(c) for c in cutoffs))

    rkw = dict(stake=stake, entry_min=entry_min, entry_max=entry_max,
               min_usd=min_usd, costs=costs, allow=allow, block=block)

    selected_records = []
    random_records = []
    fold_results = []
    candidates_scored = set()

    for i, cut in enumerate(cutoffs):
        test_lo = cut + embargo
        test_hi = cutoffs[i + 1] if i + 1 < len(cutoffs) else None
        tcache = _train_cache(resolved, cut)

        scored = []
        for j, addr in enumerate(addrs, 1):
            ev = events_by_wallet.get(addr, [])
            # OFFLINE: score against the pre-resolved cache only; never fetch per
            # wallet (that serial network storm is what makes big runs "hang").
            ws = score_wallet(client, addr, tcache, max_events,
                              events=ev, cutoff_ts=cut, now_ts=cut, offline=True)
            if ws is not None and ws.n_settled >= min_train_settled:
                scored.append((addr, ws))
                candidates_scored.add(addr)
            if progress and (j % 100 == 0 or j == len(addrs)):
                progress(j, len(addrs), phase=f"score fold {i}")
        scored.sort(key=lambda x: x[1].score, reverse=True)

        chosen = [a for a, _ in scored[:top_k]]
        # random baseline: pick top_k at random from the SAME eligible pool
        rng = random.Random(seed + i)
        pool = [a for a, _ in scored]
        rand_chosen = pool[:] if len(pool) <= top_k else rng.sample(pool, top_k)

        fold_net = 0.0
        fold_n = 0
        for addr in chosen:
            recs = copy_records(events_by_wallet.get(addr, []), resolved,
                                ts_lo=test_lo, ts_hi=test_hi, dedupe=True, **rkw)
            for r in recs:
                selected_records.append(dict(r, wallet=addr, fold=i))
                fold_net += r["net"]
                fold_n += 1
        for addr in rand_chosen:
            recs = copy_records(events_by_wallet.get(addr, []), resolved,
                                ts_lo=test_lo, ts_hi=test_hi, dedupe=True, **rkw)
            for r in recs:
                random_records.append(dict(r, wallet=addr, fold=i))

        fold_results.append(FoldResult(
            idx=i, cutoff=cut, n_candidates=len(scored), n_selected=len(chosen),
            selected=chosen, oos_n=fold_n, oos_net=fold_net,
            oos_stake=stake * fold_n))

    # Portfolio dedupe (one copy per market/outcome), mirroring live controls.
    sel = _dedupe_portfolio(selected_records)
    rnd = _dedupe_portfolio(random_records)

    # Baselines over the full OOS window/universe.
    oos_start = (cutoffs[0] + embargo) if cutoffs else None
    fav_records = []
    all_records = []
    if oos_start is not None:
        for k, addr in enumerate(addrs, 1):
            ev = events_by_wallet.get(addr, [])
            all_records.extend(copy_records(ev, resolved, ts_lo=oos_start,
                                            ts_hi=None, dedupe=True, **rkw))
            if progress and (k % 200 == 0 or k == len(addrs)):
                progress(k, len(addrs), phase="baseline copy-all")
        all_records = _dedupe_portfolio(all_records)
        if progress:
            progress(1, 1, phase="baseline favorite")
        # chain (don't materialize a merged list of millions of events); the
        # favorite baseline is a single O(events) pass.
        fav_records = favorite_records(
            itertools.chain.from_iterable(events_by_wallet.values()), resolved,
            stake=stake, min_usd=min_usd, costs=costs, allow=allow, block=block,
            ts_lo=oos_start, ts_hi=None)

    # Multiple-testing inputs for the Deflated Sharpe Ratio:
    #   breadth (n_trials) = number of wallets scored, and
    #   dispersion (var_sr) = variance of the per-wallet Sharpe ratios across the
    #     candidate pool, measured IN-SAMPLE (trades up to the last cutoff).
    # In-sample tracks are far longer than the thin OOS slices, so this estimate
    # is stable instead of being dominated by 2-3-bet noise (which previously
    # inflated var_sr and pinned DSR at 0). It is also the faithful Bailey/Lopez
    # de Prado quantity: the variance of the metric the selection RANKED on.
    n_trials = max(1, len(candidates_scored))
    train_hi = cutoffs[-1] if cutoffs else None
    trial_sharpes = []
    cand_list = list(candidates_scored)
    for k, addr in enumerate(cand_list, 1):
        recs = copy_records(events_by_wallet.get(addr, []), resolved,
                            ts_lo=None, ts_hi=train_hi, dedupe=True, **rkw)
        if len(recs) >= 5:
            trial_sharpes.append(stats.sharpe_ratio([r["net"] / stake for r in recs]))
        if progress and (k % 200 == 0 or k == len(cand_list)):
            progress(k, len(cand_list), phase="trial-variance")
    var_sr = stats.stdev(trial_sharpes) ** 2 if len(trial_sharpes) >= 2 else 0.0

    selected = _strategy_stats("selected (walk-forward)", sel, stake,
                               n_trials=n_trials, var_sr=var_sr)
    favorite = _strategy_stats("favorite baseline", fav_records, stake)
    random_sel = _strategy_stats("random-wallet baseline", rnd, stake)
    all_copy = _strategy_stats("copy-all baseline", all_records, stake)

    verdict = {
        "t_gt_3": selected.t_stat > 3.0,
        "dsr_gt_95": (selected.dsr is not None and selected.dsr > 0.95),
        "beats_favorite_roi": selected.roi > favorite.roi,
        "beats_random_ret": selected.mean_ret > random_sel.mean_ret,
        "positive_net": selected.total_net > 0,
    }
    verdict["PASS"] = all(verdict.values())

    return WFReport(
        selected=selected, favorite=favorite, random_sel=random_sel,
        all_copy=all_copy, folds=fold_results, n_trials=n_trials, var_sr=var_sr,
        params=dict(folds=len(cutoffs), top_k=top_k,
                    min_train_settled=min_train_settled,
                    embargo_days=embargo_days, stake=stake,
                    slippage=costs.slippage_points, fee=costs.fee_pct,
                    entry_min=entry_min, entry_max=entry_max, min_usd=min_usd,
                    allow=allow, block=block, cutoffs=cutoffs),
        verdict=verdict,
        records={"selected": sel, "favorite": fav_records,
                 "random": rnd, "all": all_records})


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _fmt_strategy(s: StrategyStats) -> list:
    dsr = "n/a" if s.dsr is None else f"{s.dsr:.2f}"
    return [
        f"  {s.name}",
        f"    copies {s.n} | win {s.win_rate:.0%} | net ${s.total_net:,.0f} | "
        f"ROI {s.roi:+.1%}",
        f"    mean ret/copy {s.mean_ret:+.3f} | Sharpe {s.sharpe:+.3f} | "
        f"t-stat {s.t_stat:+.2f} | PSR {s.psr:.2f} | DSR {dsr}",
        f"    alpha vs price {s.mean_alpha:+.3f} (t {s.alpha_t:+.2f})",
    ]


def format_walkforward(r: WFReport) -> str:
    p = r.params
    L = ["=== Polywatch Walk-Forward Validation (out-of-sample) ==="]
    L.append(
        f"folds {p['folds']} | top_k {p['top_k']} | min_train_settled "
        f"{p['min_train_settled']} | embargo {p['embargo_days']}d | "
        f"stake ${p['stake']:,.0f} | slippage {p['slippage']:.0%}pts fee {p['fee']:.1%}"
    )
    if p.get("allow") or p.get("block"):
        L.append(f"category allow={p.get('allow')} block={p.get('block')}")
    L.append(f"wallets tested (trials) {r.n_trials} | cross-wallet SR var {r.var_sr:.4f}")
    L.append("")

    L.append("--- Per fold (out-of-sample) ---")
    L.append(f"{'fold':>4}  {'cutoff(UTC)':>12}  {'cand':>5}  {'sel':>4}  {'oosN':>5}  {'oosROI':>7}")
    for f in r.folds:
        try:
            ct = datetime.fromtimestamp(f.cutoff, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            ct = str(f.cutoff)
        L.append(f"{f.idx:>4}  {ct:>12}  {f.n_candidates:>5}  {f.n_selected:>4}  "
                 f"{f.oos_n:>5}  {f.oos_roi:>+6.1%}")
    L.append("")

    L.append("--- Strategy vs baselines (aggregated OOS) ---")
    for s in (r.selected, r.favorite, r.random_sel, r.all_copy):
        L.extend(_fmt_strategy(s))
    L.append("")

    if r.selected.by_category:
        L.append("--- Selected strategy by category (net, with significance) ---")
        L.append(f"{'category':10}  {'n':>5}  {'win':>4}  {'net':>10}  {'ROI':>7}  {'t-stat':>7}")
        for c in sorted(r.selected.by_category,
                        key=lambda k: r.selected.by_category[k]["net"], reverse=True):
            d = r.selected.by_category[c]
            roi = d["net"] / d["stake"] if d["stake"] else 0.0
            wr = d["wins"] / d["n"] if d["n"] else 0.0
            L.append(f"{c:10}  {d['n']:>5}  {wr:>4.0%}  ${d['net']:>9,.0f}  {roi:>+6.1%}  {d.get('t', 0.0):>+7.2f}")
        L.append("  (per-category t is UNCORRECTED; ~6 categories were tested, so treat")
        L.append("   |t|>3 as the bar, not 2 — one positive category out of six is expected by chance)")
        L.append("")

    L.append("--- VERDICT ---")
    v = r.verdict

    def mark(b):
        return "PASS" if b else "FAIL"

    L.append(f"  t-stat > 3.0 (Harvey)            : {mark(v['t_gt_3'])}  ({r.selected.t_stat:+.2f})")
    dsr = "n/a" if r.selected.dsr is None else f"{r.selected.dsr:.2f}"
    L.append(f"  Deflated Sharpe > 0.95           : {mark(v['dsr_gt_95'])}  ({dsr})")
    L.append(f"  beats favorite baseline (ROI)    : {mark(v['beats_favorite_roi'])}  "
             f"({r.selected.roi:+.1%} vs {r.favorite.roi:+.1%})")
    L.append(f"  beats random-wallet selection    : {mark(v['beats_random_ret'])}  "
             f"({r.selected.mean_ret:+.3f} vs {r.random_sel.mean_ret:+.3f})")
    L.append(f"  positive net P&L                 : {mark(v['positive_net'])}  "
             f"(${r.selected.total_net:,.0f})")
    L.append("")
    L.append(f"  OVERALL: {'PASS - out-of-sample edge survives' if v['PASS'] else 'FAIL - edge not proven out-of-sample'}")
    if not v["PASS"]:
        L.append("  (do NOT enable live money; the selection edge is unproven.)")
    return "\n".join(L)


def write_walkforward_csv(report: WFReport, path: str) -> int:
    """Dump every per-copy row (all four strategies) to a CSV for your own
    slicing. One row per copied trade; columns include the alpha-vs-price and
    net P&L. Returns the number of rows written."""
    import csv
    stake = report.params.get("stake", 0.0)
    cols = ["strategy", "fold", "wallet", "cid", "category", "entry_price",
            "won", "stake", "gross_pnl", "net_pnl", "alpha", "ts", "title"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for strat, recs in report.records.items():
            for r in recs:
                w.writerow({
                    "strategy": strat,
                    "fold": r.get("fold", ""),
                    "wallet": r.get("wallet", ""),
                    "cid": r.get("cid", ""),
                    "category": r.get("cat", ""),
                    "entry_price": r.get("price", ""),
                    "won": int(bool(r.get("won"))),
                    "stake": stake,
                    "gross_pnl": round(r.get("gross", 0.0), 4),
                    "net_pnl": round(r.get("net", 0.0), 4),
                    "alpha": round(r.get("alpha", 0.0), 4),
                    "ts": r.get("ts", ""),
                    "title": (r.get("title") or "")[:120],
                })
                n += 1
    return n
