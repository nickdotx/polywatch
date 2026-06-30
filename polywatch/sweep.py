"""Strategy sweep — score every variant through one shared data pull.

Runs the copy-trade thesis (copy-all, scorer-selected, K-of-N consensus) against
its skill-free baselines (favourite-longshot, random selection) on the same
out-of-sample window, with a single network pass, and emits a publishable results
table (markdown + CSV). This is the "verified strategy database" view: one row per
strategy, each column an honest, cost-aware, out-of-sample statistic.

It reuses the walk-forward engine for the thesis rows and baselines (so the
selection logic and costs are identical), then adds the consensus row on the same
pre-fetched data. Nothing here trades; it measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import walkforward as wf
from .backtest import _qualifying_buys
from .categories import categorize, passes_category_filter
from .consensus import ConsensusTracker
from .costs import CostModel


# ---------------------------------------------------------------------------
# consensus strategy as per-copy records (so it scores like every other row)
# ---------------------------------------------------------------------------

def consensus_records(events_by_wallet, resolved, *, k=2, window_hours=24,
                      stake=15.0, entry_min=0.05, entry_max=0.95, min_usd=100.0,
                      costs=None, allow=None, block=None, ts_lo=None, ts_hi=None):
    """Per-copy records for the K-of-N consensus strategy over a time window.

    Flattens qualifying buys across all wallets, replays them in timestamp order
    through a ConsensusTracker, and emits ONE record the first time K distinct
    wallets agree on a (market, side). Same record shape as backtest.copy_records,
    so walkforward._strategy_stats scores it identically to every other row."""
    costs = costs or CostModel()
    flat = []
    for addr, events in events_by_wallet.items():
        for cid, idx, price, title, slug, ts in _qualifying_buys(
                events, entry_min, entry_max, min_usd):
            if ts_lo is not None and ts < ts_lo:
                continue
            if ts_hi is not None and ts > ts_hi:
                continue
            cat = categorize(title, slug)
            if not passes_category_filter(cat, allow, block):
                continue
            flat.append((ts, cid, idx, price, title, slug, cat, addr))
    flat.sort(key=lambda x: x[0])

    tracker = ConsensusTracker(k=k, window_seconds=window_hours * 3600)
    out = []
    for ts, cid, idx, price, title, slug, cat, addr in flat:
        market = resolved.get(cid)
        if market is None or not market.resolved:
            continue                       # only settle on resolved markets
        if not tracker.observe(cid, idx, addr, ts):
            continue                       # not yet K distinct wallets
        won = market.resolved_outcome_index() == idx
        gross = (stake / price - stake) if won else -stake
        net = costs.net_pnl(price, stake, won)
        alpha = (1.0 if won else 0.0) - price
        out.append({"cid": cid, "idx": idx, "price": price, "won": won,
                    "cat": cat, "ts": ts, "gross": gross, "net": net,
                    "alpha": alpha, "title": title, "slug": slug})
    return out


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class StrategyRow:
    key: str           # short id: copy_all | selected | consensus | favorite | random
    is_baseline: bool
    stats: object      # walkforward.StrategyStats


@dataclass
class SweepResult:
    rows: list = field(default_factory=list)   # ordered StrategyRow list
    report: object = None                       # the underlying WFReport
    params: dict = field(default_factory=dict)


def row_verdict(row: StrategyRow) -> str:
    """A thesis row must clear the same bar the walk-forward uses; baselines are
    just reference points, not strategies under test."""
    if row.is_baseline:
        return "baseline"
    s = row.stats
    t_ok = s.t_stat > 3.0
    roi_ok = s.roi > 0
    dsr_ok = (s.dsr is None) or (s.dsr > 0.95)
    return "PASS" if (t_ok and roi_ok and dsr_ok) else "FAIL"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run_sweep(client, targets, *, cutoffs=None, folds=3, top_k=10,
              min_train_settled=20, embargo_days=0, stake=15.0, costs=None,
              entry_min=0.05, entry_max=0.95, min_usd=100.0, allow=None, block=None,
              consensus_k=2, consensus_window_hours=24, max_events=5000, workers=16,
              cache_path=None, seed=0, reserve_days=21, now_ts=None, progress=None):
    costs = costs or CostModel()
    addrs = [getattr(t, "address", t) for t in targets]

    # One network pass, shared by every strategy below.
    events_by_wallet = wf._pull_all(client, addrs, max_events, workers, progress)
    resolved = wf._resolve_markets(events_by_wallet, entry_min, entry_max, min_usd,
                                   cache_path, client, workers=workers, progress=progress)

    report = wf.walk_forward(
        client, targets, folds=folds, cutoffs=cutoffs, top_k=top_k,
        min_train_settled=min_train_settled, embargo_days=embargo_days, stake=stake,
        costs=costs, entry_min=entry_min, entry_max=entry_max, min_usd=min_usd,
        allow=allow, block=block, max_events=max_events, workers=workers,
        cache_path=cache_path, seed=seed, reserve_days=reserve_days, now_ts=now_ts,
        progress=progress, events_by_wallet=events_by_wallet, resolved=resolved)

    cuts = report.params.get("cutoffs") or []
    oos_start = (cuts[0] + int(embargo_days * wf.DAY)) if cuts else None
    cons = consensus_records(events_by_wallet, resolved, k=consensus_k,
                             window_hours=consensus_window_hours, stake=stake,
                             entry_min=entry_min, entry_max=entry_max, min_usd=min_usd,
                             costs=costs, allow=allow, block=block, ts_lo=oos_start)
    cons_stats = wf._strategy_stats(f"consensus {consensus_k}-of-N", cons, stake)

    rows = [
        StrategyRow("copy_all", False, report.all_copy),
        StrategyRow("selected", False, report.selected),
        StrategyRow("consensus", False, cons_stats),
        StrategyRow("favorite", True, report.favorite),
        StrategyRow("random", True, report.random_sel),
    ]
    params = dict(report.params, consensus_k=consensus_k,
                  consensus_window_hours=consensus_window_hours,
                  n_trials=report.n_trials, var_sr=report.var_sr)
    return SweepResult(rows=rows, report=report, params=params)


# ---------------------------------------------------------------------------
# formatting / export
# ---------------------------------------------------------------------------

_LABELS = {"copy_all": "Copy everything", "selected": "Scorer-selected (top-k)",
           "consensus": "Consensus K-of-N", "favorite": "Favourite-longshot",
           "random": "Random selection"}


def _dsr_cell(s):
    return "n/a" if s.dsr is None else f"{s.dsr:.2f}"


def format_sweep(res: SweepResult) -> str:
    p = res.params
    L = ["=== Polywatch strategy sweep (out-of-sample) ==="]
    L.append(f"cutoffs {p.get('cutoffs')} | stake ${p.get('stake', 0):,.0f} | "
             f"slippage {p.get('slippage', 0):.0%}pts | wallets tested {p.get('n_trials', 0)}")
    L.append("")
    hdr = (f"{'strategy':24}  {'trades':>7}  {'win':>4}  {'netROI':>7}  "
           f"{'alpha':>6}  {'t-stat':>7}  {'DSR':>5}  verdict")
    L.append(hdr)
    L.append("-" * len(hdr))
    for row in res.rows:
        s = row.stats
        L.append(f"{_LABELS.get(row.key, row.key):24}  {s.n:>7,}  {s.win_rate:>4.0%}  "
                 f"{s.roi:>+6.1%}  {s.mean_alpha:>+6.3f}  {s.t_stat:>+7.2f}  "
                 f"{_dsr_cell(s):>5}  {row_verdict(row)}")
    L.append("")
    v = res.report.verdict
    L.append(f"OVERALL: {'PASS — out-of-sample edge survives' if v.get('PASS') else 'FAIL — no edge survives costs out-of-sample'}")
    return "\n".join(L)


def write_sweep_csv(res: SweepResult, path: str) -> int:
    """One summary row per strategy (the table behind RESULTS.md)."""
    import csv
    cols = ["strategy", "key", "is_baseline", "trades", "win_rate", "net_pnl",
            "net_roi", "mean_ret", "sharpe", "t_stat", "mean_alpha", "alpha_t",
            "psr", "dsr", "verdict"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in res.rows:
            s = row.stats
            w.writerow({
                "strategy": _LABELS.get(row.key, row.key), "key": row.key,
                "is_baseline": int(row.is_baseline), "trades": s.n,
                "win_rate": round(s.win_rate, 4), "net_pnl": round(s.total_net, 2),
                "net_roi": round(s.roi, 4), "mean_ret": round(s.mean_ret, 5),
                "sharpe": round(s.sharpe, 4), "t_stat": round(s.t_stat, 3),
                "mean_alpha": round(s.mean_alpha, 4), "alpha_t": round(s.alpha_t, 3),
                "psr": round(s.psr, 4),
                "dsr": "" if s.dsr is None else round(s.dsr, 4),
                "verdict": row_verdict(row)})
    return len(res.rows)


def write_results_md(res: SweepResult, path: str) -> None:
    """Render the publishable RESULTS.md table from a sweep."""
    p = res.params
    r = res.report
    out = []
    out.append("# Polywatch — strategy sweep results\n")
    out.append("_Generated by `python -m polywatch sweep`. Every row is scored on "
               "the same out-of-sample window with identical costs; baselines are "
               "the skill-free bars the thesis strategies must beat._\n")
    out.append(f"- Cutoffs: `{p.get('cutoffs')}`")
    out.append(f"- Stake/copy: ${p.get('stake', 0):,.0f} | slippage "
               f"{p.get('slippage', 0):.0%} pts | fee {p.get('fee', 0):.1%}")
    out.append(f"- Wallets tested (DSR trials): {p.get('n_trials', 0)} | "
               f"cross-wallet Sharpe variance: {p.get('var_sr', 0):.4f}\n")

    out.append("## Strategies\n")
    out.append("| Strategy | OOS trades | Win % | Net ROI | mean α | t-stat | DSR | Verdict |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in res.rows:
        s = row.stats
        out.append(f"| {_LABELS.get(row.key, row.key)} | {s.n:,} | {s.win_rate:.0%} | "
                   f"{s.roi:+.1%} | {s.mean_alpha:+.3f} | {s.t_stat:+.2f} | "
                   f"{_dsr_cell(s)} | {row_verdict(row)} |")
    out.append("")
    out.append("*α = outcome(1/0) − entry price (how much each copy beat the market's "
               "own implied probability). DSR = Deflated Sharpe Ratio, the "
               "multiple-testing haircut; it applies to the wallet-selected row, "
               "which searched over many candidates.*\n")

    if r.selected.by_category:
        out.append("## Scorer-selected, by category\n")
        out.append("| Category | n | Win % | Net ROI | t-stat |")
        out.append("|---|---:|---:|---:|---:|")
        cats = sorted(r.selected.by_category,
                      key=lambda k: r.selected.by_category[k]["net"], reverse=True)
        for c in cats:
            d = r.selected.by_category[c]
            roi = d["net"] / d["stake"] if d["stake"] else 0.0
            wr = d["wins"] / d["n"] if d["n"] else 0.0
            out.append(f"| {c} | {d['n']:,} | {wr:.0%} | {roi:+.1%} | {d.get('t', 0.0):+.2f} |")
        out.append("\n*Per-category t is uncorrected; with ~6 categories tested, treat "
                   "|t| > 3 as the bar — one positive category out of six is expected by "
                   "chance.*\n")

    v = r.verdict
    out.append("## Verdict (scorer-selected strategy)\n")

    def mk(b):
        return "✅" if b else "❌"

    out.append(f"- {mk(v.get('t_gt_3'))} t-stat > 3.0 (Harvey et al.): "
               f"`{r.selected.t_stat:+.2f}`")
    dsr = "n/a" if r.selected.dsr is None else f"{r.selected.dsr:.2f}"
    out.append(f"- {mk(v.get('dsr_gt_95'))} Deflated Sharpe > 0.95: `{dsr}`")
    out.append(f"- {mk(v.get('beats_favorite_roi'))} beats favourite baseline ROI: "
               f"`{r.selected.roi:+.1%}` vs `{r.favorite.roi:+.1%}`")
    out.append(f"- {mk(v.get('beats_random_ret'))} beats random selection: "
               f"`{r.selected.mean_ret:+.3f}` vs `{r.random_sel.mean_ret:+.3f}`")
    out.append(f"- {mk(v.get('positive_net'))} positive net P&L: "
               f"`${r.selected.total_net:,.0f}`")
    out.append("")
    out.append(f"**Overall: {'PASS — out-of-sample edge survives.' if v.get('PASS') else 'FAIL — no edge survives costs out-of-sample.'}**")
    out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
