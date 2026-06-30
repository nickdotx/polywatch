"""Command-line entry point.

Commands:
  targets      List the configured wallet pool.
  backtest     In-sample copy-trade backtest on resolved markets (gross vs net).
  walkforward  Out-of-sample walk-forward validation with a pass/fail verdict.
  arb          Scan Polymarket for internal arbitrage (ask(YES)+ask(NO) < $1).
  crossarb     Scan / discover cross-platform Polymarket<->Kalshi arbitrage.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

from . import __version__
from .config import Config
from .costs import CostModel
from .backtest import backtest_copy, backtest_consensus, format_backtest
from .walkforward import walk_forward, format_walkforward, write_walkforward_csv
from .arb import scan_internal_arb, format_arb, LifetimeTracker
from .crossarb import (scan_pairs, format_crossarb, discover_pairs,
                       format_discover, discover_yaml)
from .sweep import run_sweep, format_sweep, write_results_md, write_sweep_csv
from .polymarket import PolymarketClient


def _client(config: Config) -> PolymarketClient:
    return PolymarketClient(data_api=config.endpoints.data_api,
                            gamma_api=config.endpoints.gamma_api)


def cmd_targets(config: Config) -> int:
    if not config.targets:
        print("No targets configured.")
        return 0
    print(f"{len(config.targets)} target(s):")
    for t in config.targets:
        print(f"  {t.address}  {t.label}")
    return 0


def cmd_backtest(config: Config, args) -> int:
    client = _client(config)
    if not config.targets:
        print("No targets. Run scripts/pick_wallets.py --write first.")
        client.close()
        return 1
    try:
        def _progress(done, total, phase=""):
            if done % 10 == 0 or done == total:
                print(f"  fetching histories... {done}/{total}")

        costs = CostModel(
            slippage_points=args.slippage if args.slippage is not None else config.costs.slippage_points,
            fee_pct=args.fee if args.fee is not None else config.costs.fee_pct,
            latency_points=config.costs.latency_points,
        )
        kwargs = dict(
            stake=config.strategy.stake_usd, costs=costs,
            allow=args.allow if args.allow is not None else config.strategy.categories_allow,
            block=args.block if args.block is not None else config.strategy.categories_block,
            max_events=args.max_events, entry_min=config.strategy.entry_min_price,
            entry_max=config.strategy.entry_max_price, min_usd=config.strategy.min_copy_usd,
            max_wallets=args.max_wallets, workers=args.workers,
            cache_path="markets_cache.json", progress=_progress)
        if args.consensus and args.consensus > 1:
            result = backtest_consensus(client, config.targets, k=args.consensus,
                                        window_hours=args.window, **kwargs)
        else:
            result = backtest_copy(client, config.targets, **kwargs)
        print(format_backtest(result))
    finally:
        client.close()
    return 0


def _parse_cutoff(s: str) -> int:
    """Accept a unix timestamp or an ISO date (YYYY-MM-DD[...]) -> unix seconds."""
    s = s.strip()
    if s.isdigit():
        return int(s)
    import datetime as dt
    d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def cmd_walkforward(config: Config, args) -> int:
    client = _client(config)
    if not config.targets:
        print("No targets. Run scripts/pick_wallets.py --write first.")
        client.close()
        return 1
    try:
        costs = CostModel(
            slippage_points=args.slippage if args.slippage is not None else config.costs.slippage_points,
            fee_pct=args.fee if args.fee is not None else config.costs.fee_pct,
            latency_points=config.costs.latency_points,
        )
        cutoffs = [_parse_cutoff(c) for c in args.cutoffs] if args.cutoffs else None

        def _progress(done, total, phase=""):
            label = {"fetch": "fetching histories"}.get(phase, phase or "working")
            if done % 200 == 0 or done == total:
                print(f"  {label}... {done}/{total}")

        report = walk_forward(
            client, config.targets, folds=args.folds, cutoffs=cutoffs,
            top_k=args.top_k, min_train_settled=args.min_train_settled,
            embargo_days=args.embargo_days, stake=config.strategy.stake_usd,
            costs=costs, entry_min=config.strategy.entry_min_price,
            entry_max=config.strategy.entry_max_price, min_usd=config.strategy.min_copy_usd,
            allow=args.allow if args.allow is not None else config.strategy.categories_allow,
            block=args.block if args.block is not None else config.strategy.categories_block,
            max_events=args.max_events, workers=args.workers,
            cache_path="markets_cache.json", seed=args.seed,
            reserve_days=args.reserve_days, progress=_progress)
        print(format_walkforward(report))
        if args.csv:
            rows = write_walkforward_csv(report, args.csv)
            print(f"\nWrote {rows} per-copy rows to {args.csv}")
    finally:
        client.close()
    return 0


def cmd_sweep(config: Config, args) -> int:
    client = _client(config)
    if not config.targets:
        print("No targets. Run scripts/pick_wallets.py --write first.")
        client.close()
        return 1
    try:
        costs = CostModel(
            slippage_points=args.slippage if args.slippage is not None else config.costs.slippage_points,
            fee_pct=args.fee if args.fee is not None else config.costs.fee_pct,
            latency_points=config.costs.latency_points,
        )
        cutoffs = [_parse_cutoff(c) for c in args.cutoffs] if args.cutoffs else None

        def _progress(done, total, phase=""):
            if done % 200 == 0 or done == total:
                print(f"  {phase or 'working'}... {done}/{total}")

        res = run_sweep(
            client, config.targets, cutoffs=cutoffs, folds=args.folds,
            top_k=args.top_k, min_train_settled=args.min_train_settled,
            embargo_days=args.embargo_days, stake=config.strategy.stake_usd,
            costs=costs, entry_min=config.strategy.entry_min_price,
            entry_max=config.strategy.entry_max_price, min_usd=config.strategy.min_copy_usd,
            allow=args.allow if args.allow is not None else config.strategy.categories_allow,
            block=args.block if args.block is not None else config.strategy.categories_block,
            consensus_k=args.consensus_k, consensus_window_hours=args.window,
            max_events=args.max_events, workers=args.workers,
            cache_path="markets_cache.json", seed=args.seed,
            reserve_days=args.reserve_days, progress=_progress)
        print(format_sweep(res))
        write_results_md(res, args.out)
        rows = write_sweep_csv(res, args.csv)
        print(f"\nWrote {args.out} and {rows} summary rows to {args.csv}")
        if args.html or args.svg:
            from .report import rows_from_sweep, build_html, write_roi_svg
            rrows = rows_from_sweep(res)
            if args.html:
                build_html(rrows, res.params, args.html)
                print(f"Wrote {args.html}")
            if args.svg:
                write_roi_svg(rrows, args.svg)
                print(f"Wrote {args.svg}")
    finally:
        client.close()
    return 0


def cmd_report(config: Config, args) -> int:
    from .report import load_summary_csv, build_html, write_roi_svg
    rows = load_summary_csv(args.csv)
    if not rows:
        print(f"No rows in {args.csv}. Run `python -m polywatch sweep` first.")
        return 1
    build_html(rows, {"source": args.csv}, args.html)
    print(f"Wrote {args.html} ({len(rows)} strategies)")
    if args.svg:
        write_roi_svg(rows, args.svg)
        print(f"Wrote {args.svg}")
    return 0


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cmd_arb(config: Config, args) -> int:
    client = _client(config)
    try:
        if not args.watch:
            opps = scan_internal_arb(client, n_markets=args.markets, fee=args.fee,
                                     min_edge=args.min_edge, workers=args.workers)
            print(format_arb(opps, args.markets))
            if args.csv:
                _write_csv(args.csv,
                           ["cid", "question", "yes_ask", "no_ask", "size",
                            "edge_per_pair", "edge_pct", "max_profit", "ts"],
                           [[o.cid, (o.question or "")[:120], o.yes_ask, o.no_ask,
                             round(o.size, 2), round(o.edge_per_pair, 4),
                             round(o.edge_pct, 4), round(o.max_profit, 2), o.ts]
                            for o in opps])
                print(f"\nWrote {len(opps)} opportunities to {args.csv}")
            return 0

        tracker = LifetimeTracker()
        print(f"Watching {args.markets} markets every {args.watch}s. Ctrl-C to stop.")
        scans = 0
        try:
            end = time.time() + args.duration * 60 if args.duration else None
            while end is None or time.time() < end:
                opps = scan_internal_arb(client, n_markets=args.markets, fee=args.fee,
                                         min_edge=args.min_edge, workers=args.workers)
                tracker.observe(opps, int(time.time()))
                scans += 1
                print(f"  scan {scans}: {len(opps)} gap(s)")
                time.sleep(max(1, args.watch))
        except KeyboardInterrupt:
            print("\n(stopped)")
        rows = tracker.rows()
        print(f"\n=== Summary: {len(rows)} distinct gap(s) over {scans} scans ===")
        if rows:
            import statistics
            lifes = [r["lifetime_s"] for r in rows]
            print(f"gap lifetime (s): median {statistics.median(lifes):.0f} | "
                  f"max {max(lifes)} | min {min(lifes)}")
        if args.csv:
            _write_csv(args.csv,
                       ["cid", "lifetime_s", "samples", "best_edge", "first_seen", "last_seen"],
                       [[r["cid"], r["lifetime_s"], r["samples"], r["best_edge"],
                         r["first_seen"], r["last_seen"]] for r in rows])
            print(f"Wrote {len(rows)} gap lifetimes to {args.csv}")
    finally:
        client.close()
    return 0


def cmd_crossarb(config: Config, args) -> int:
    import yaml
    from .kalshi import KalshiClient
    poly = _client(config)
    kal = KalshiClient()
    try:
        if args.discover:
            cats = [c.strip() for c in args.kalshi_categories.split(",") if c.strip()]
            per_cat = max(100, args.kalshi_limit // max(1, len(cats)))
            print(f"Discovering candidates: {args.poly_markets} Polymarket markets x "
                  f"Kalshi categories {cats}...")
            pm = poly.get_top_markets(limit=args.poly_markets) or []
            km = kal.markets_by_categories(cats, per_cat=per_cat)
            print(f"  fetched {len(pm)} Polymarket + {len(km)} Kalshi markets")
            cands = discover_pairs(pm, km, min_score=args.min_score, top=args.top)
            print(format_discover(cands))
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(discover_yaml(cands))
                print(f"Wrote {len(cands)} candidates to {args.out} (review + verify).")
            return 0

        if not os.path.exists(args.pairs):
            print(f"Pairs file '{args.pairs}' not found. Copy pairs.example.yaml to "
                  f"'{args.pairs}' and add VERIFIED pairs, or run --discover.")
            return 1
        data = yaml.safe_load(open(args.pairs).read()) or {}
        pairs = data.get("pairs") if isinstance(data, dict) else data
        if not pairs:
            print(f"No pairs found in {args.pairs}.")
            return 1
        opps = scan_pairs(poly, kal, pairs, poly_fee=args.poly_fee,
                          kalshi_rate=args.kalshi_rate, min_edge=args.min_edge)
        print(format_crossarb(opps, len(pairs)))
        if args.csv:
            _write_csv(args.csv,
                       ["name", "edge_per_pair", "edge_pct", "size", "max_profit",
                        "yes_leg", "yes_price", "no_leg", "no_price", "poly_cid",
                        "kalshi_ticker", "ts"],
                       [[o.name[:120], round(o.edge_per_pair, 4), round(o.edge_pct, 4),
                         round(o.size, 2), round(o.max_profit, 2), o.yes_leg, o.yes_price,
                         o.no_leg, o.no_price, o.poly_cid, o.kalshi_ticker, o.ts]
                        for o in opps])
            print(f"\nWrote {len(opps)} rows to {args.csv}")
    finally:
        poly.close()
        kal.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polywatch",
                                     description="Polymarket smart-money research toolkit")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("targets", help="list configured wallets")

    bt = sub.add_parser("backtest", help="in-sample copy-trade backtest")
    bt.add_argument("--max-events", type=int, default=5000)
    bt.add_argument("--max-wallets", type=int, default=0)
    bt.add_argument("--workers", type=int, default=16)
    bt.add_argument("--slippage", type=float, default=None)
    bt.add_argument("--fee", type=float, default=None)
    bt.add_argument("--allow", nargs="*", default=None, metavar="CAT")
    bt.add_argument("--block", nargs="*", default=None, metavar="CAT")
    bt.add_argument("--consensus", type=int, default=1, help="require K wallets to agree")
    bt.add_argument("--window", type=int, default=24, help="consensus window (hours)")

    wf = sub.add_parser("walkforward", help="out-of-sample walk-forward validation")
    wf.add_argument("--folds", type=int, default=3)
    wf.add_argument("--cutoffs", nargs="*", default=None, metavar="T",
                    help="explicit cutoffs (unix ts or ISO dates); overrides --folds")
    wf.add_argument("--top-k", type=int, default=10)
    wf.add_argument("--min-train-settled", type=int, default=20)
    wf.add_argument("--embargo-days", type=float, default=0.0)
    wf.add_argument("--reserve-days", type=float, default=21.0)
    wf.add_argument("--max-events", type=int, default=5000)
    wf.add_argument("--workers", type=int, default=16)
    wf.add_argument("--slippage", type=float, default=None)
    wf.add_argument("--fee", type=float, default=None)
    wf.add_argument("--allow", nargs="*", default=None, metavar="CAT")
    wf.add_argument("--block", nargs="*", default=None, metavar="CAT")
    wf.add_argument("--seed", type=int, default=0)
    wf.add_argument("--csv", default=None, metavar="PATH")

    sw = sub.add_parser("sweep", help="score every strategy variant into RESULTS.md + CSV")
    sw.add_argument("--folds", type=int, default=3)
    sw.add_argument("--cutoffs", nargs="*", default=None, metavar="T",
                    help="explicit cutoffs (unix ts or ISO dates); overrides --folds")
    sw.add_argument("--top-k", type=int, default=10)
    sw.add_argument("--min-train-settled", type=int, default=20)
    sw.add_argument("--embargo-days", type=float, default=0.0)
    sw.add_argument("--reserve-days", type=float, default=21.0)
    sw.add_argument("--consensus-k", type=int, default=2, help="K-of-N consensus row")
    sw.add_argument("--window", type=int, default=24, help="consensus window (hours)")
    sw.add_argument("--max-events", type=int, default=5000)
    sw.add_argument("--workers", type=int, default=16)
    sw.add_argument("--slippage", type=float, default=None)
    sw.add_argument("--fee", type=float, default=None)
    sw.add_argument("--allow", nargs="*", default=None, metavar="CAT")
    sw.add_argument("--block", nargs="*", default=None, metavar="CAT")
    sw.add_argument("--seed", type=int, default=0)
    sw.add_argument("--out", default="RESULTS.md", help="markdown results file")
    sw.add_argument("--csv", default="results.csv", help="summary CSV file")
    sw.add_argument("--html", default=None, help="also write an HTML dashboard")
    sw.add_argument("--svg", default=None, help="also write a standalone ROI SVG")

    rp = sub.add_parser("report", help="render an HTML/SVG dashboard from a sweep CSV")
    rp.add_argument("--csv", default="results.csv", help="summary CSV from sweep")
    rp.add_argument("--html", default="report.html", help="output HTML dashboard")
    rp.add_argument("--svg", default=None, help="also write a standalone ROI bar SVG")

    ab = sub.add_parser("arb", help="scan Polymarket for internal arbitrage")
    ab.add_argument("--markets", type=int, default=200)
    ab.add_argument("--fee", type=float, default=0.0)
    ab.add_argument("--min-edge", type=float, default=0.0)
    ab.add_argument("--workers", type=int, default=16)
    ab.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    ab.add_argument("--duration", type=float, default=0, metavar="MIN")
    ab.add_argument("--csv", default=None)

    ca = sub.add_parser("crossarb", help="cross-platform Polymarket<->Kalshi arbitrage")
    ca.add_argument("--pairs", default="pairs.yaml")
    ca.add_argument("--poly-fee", type=float, default=0.0)
    ca.add_argument("--kalshi-rate", type=float, default=0.07)
    ca.add_argument("--min-edge", type=float, default=0.0)
    ca.add_argument("--csv", default=None)
    ca.add_argument("--discover", action="store_true",
                    help="fuzzy-match both venues into candidate pairs to verify")
    ca.add_argument("--poly-markets", type=int, default=300)
    ca.add_argument("--kalshi-limit", type=int, default=1500)
    ca.add_argument("--kalshi-categories",
                    default="Politics,Elections,Economics,Financials,Crypto,World,Companies")
    ca.add_argument("--min-score", type=float, default=0.5)
    ca.add_argument("--top", type=int, default=50)
    ca.add_argument("--out", default=None)

    sub.add_parser("version", help="print version")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command in (None, "version"):
        print(f"polywatch {__version__}")
        return 0
    config = Config.load(args.config)
    dispatch = {
        "targets": lambda: cmd_targets(config),
        "backtest": lambda: cmd_backtest(config, args),
        "walkforward": lambda: cmd_walkforward(config, args),
        "sweep": lambda: cmd_sweep(config, args),
        "report": lambda: cmd_report(config, args),
        "arb": lambda: cmd_arb(config, args),
        "crossarb": lambda: cmd_crossarb(config, args),
    }
    return dispatch[args.command]()


if __name__ == "__main__":
    sys.exit(main())
