"""Leaderboard-based wallet picker (parallelized, uses the informed-edge scorer).

Speed: candidate gathering and per-wallet scoring both run concurrently, and the
scorer's market lookups are cached across wallets. The work is network-bound, so
concurrency — not a faster language — is what makes it quick.

    python scripts/pick_wallets.py --markets 150 --max-candidates 400 --top 100 --write 100
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polywatch.polymarket import PolymarketClient  # noqa: E402
from polywatch import scorer as scoring             # noqa: E402
from polywatch.categories import categorize         # noqa: E402


def gather_candidates(client, n_markets, cap, workers=16, category=None):
    # When a category is requested, fetch many more top markets and keep only the
    # ones in that category, so we pull holders of (e.g.) crypto markets instead
    # of whatever is highest-liquidity right now (currently World Cup sports).
    fetch = max(n_markets * (8 if category else 2), 60)
    raw = client.get_top_markets(limit=fetch)
    markets = []
    for m in raw:
        if category:
            title = m.get("question") or m.get("title") or ""
            if categorize(title, m.get("slug") or "") != category:
                continue
        markets.append(m)
        if len(markets) >= n_markets:
            break
    cids = [str(m.get("conditionId") or "") for m in markets if m.get("conditionId")]

    seen: dict[str, None] = {}
    lock = threading.Lock()

    def _holders(cid):
        return client.get_market_holders(cid, limit=15)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for wallets in ex.map(_holders, cids):
            with lock:
                for w in wallets:
                    if w not in seen:
                        seen[w] = None
            if len(seen) >= cap:
                break
    return list(seen)[:cap]


def write_config(top, path="config.yaml"):
    import yaml
    p = Path(path)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    if not isinstance(data, dict):
        data = {}
    data["targets"] = [
        {"address": w.address,
         "label": (f"pnl=${w.realized_pnl:,.0f} roi={w.roi:.0%} "
                   f"n={w.n_settled} wr={w.win_rate:.0%} lsWR={w.longshot_win_rate:.0%}")}
        for w in top
    ]
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"\nWrote {len(top)} targets to {path}")


def write_addresses(addrs, path="config.yaml"):
    """Write a raw, UNSCORED candidate pool (just addresses). Used by
    --gather-only: walk-forward will do the point-in-time scoring, so we skip the
    expensive full-history reconstruction here — and writing the broad pool
    (rather than the top-N by full-history score) avoids selecting the pool with
    future information (the leakage that biases the walk-forward)."""
    import yaml
    p = Path(path)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    if not isinstance(data, dict):
        data = {}
    data["targets"] = [{"address": w, "label": "candidate (unscored pool)"} for w in addrs]
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"\nWrote {len(addrs)} unscored candidate targets to {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rank Polymarket wallets by informed-edge")
    ap.add_argument("--markets", type=int, default=30)
    ap.add_argument("--max-candidates", type=int, default=80)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--max-events", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=16, help="parallel scorers")
    ap.add_argument("--write", type=int, default=0, metavar="N")
    ap.add_argument("--gather-only", action="store_true",
                    help="collect candidate wallets from market holders and write "
                         "them WITHOUT scoring — fast way to assemble a big, "
                         "unbiased pool (walk-forward scores them point-in-time)")
    ap.add_argument("--category", default=None,
                    help="only gather holders of markets in this category "
                         "(crypto/sports/politics/econ/pop) — targets relevant wallets")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    client = PolymarketClient()
    market_cache: dict = {}
    try:
        print(f"Gathering candidates from top {args.markets} markets (parallel)...")
        candidates = gather_candidates(client, args.markets, args.max_candidates,
                                       args.workers, category=args.category)
        print(f"Gathered {len(candidates)} distinct candidate wallets"
              f"{' (' + args.category + ' markets)' if args.category else ''}.")

        if args.gather_only:
            # Fast path: write the broad pool unscored; walk-forward will score
            # point-in-time. Skips the expensive per-wallet history reconstruction.
            write_addresses(candidates, args.config)
            return 0

        print(f"Scoring {len(candidates)} wallets with {args.workers} workers "
              f"(reconstructing full P&L)...")

        scored = []
        done = 0
        lock = threading.Lock()

        def _score(addr):
            try:
                return scoring.score_wallet(client, addr, market_cache, args.max_events)
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ws in ex.map(_score, candidates):
                done += 1
                if ws is not None:
                    scored.append(ws)
                if done % 20 == 0 or done == len(candidates):
                    print(f"  ...{done}/{len(candidates)}")

        scored.sort(key=lambda w: w.score, reverse=True)
        top = scored[: args.top]

        hdr = (f"{'#':>2}  {'address':42}  {'realPnL':>11}  {'ROI':>6}  "
               f"{'n':>3}  {'win':>4}  {'lsWR':>5}  {'entwin':>6}  {'score':>11}")
        print("\n" + hdr)
        print("-" * len(hdr))
        for i, w in enumerate(top, 1):
            print(f"{i:>2}  {w.address:42}  {w.realized_pnl:>11,.0f}  {w.roi:>5.0%}  "
                  f"{w.n_settled:>3}  {w.win_rate:>4.0%}  {w.longshot_win_rate:>5.0%}  "
                  f"{w.avg_entry_win:>6.2f}  {w.score:>11,.0f}")
        print("\nlegend: realPnL=loss-inclusive realized $ | ROI=return on cost | "
              "n=settled markets | win=win rate | lsWR=longshot win rate | "
              "entwin=avg entry odds on wins (lower=edgier)")

        if args.write > 0 and top:
            write_config(top[: args.write], args.config)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
