"""Historical copy-trade backtester — fast, cost-aware, category-aware.

Parallel + batched + disk-cached for speed; CostModel for realism (gross vs net);
and a per-category breakdown so we can see WHERE the edge lives (sports vs
politics vs crypto ...). Supports allow/block category filters.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .categories import categorize, passes_category_filter
from .consensus import ConsensusTracker
from .costs import CostModel
from .models import Market
from .scorer import _pull_activity


@dataclass
class WalletBacktest:
    wallet: str
    label: str = ""
    n: int = 0
    wins: int = 0
    stake_total: float = 0.0
    pnl: float = 0.0          # NET (after costs)
    pnl_gross: float = 0.0
    longshot_n: int = 0
    longshot_wins: int = 0

    @property
    def win_rate(self): return self.wins / self.n if self.n else 0.0
    @property
    def roi(self): return self.pnl / self.stake_total if self.stake_total else 0.0
    @property
    def roi_gross(self): return self.pnl_gross / self.stake_total if self.stake_total else 0.0
    @property
    def longshot_win_rate(self): return self.longshot_wins / self.longshot_n if self.longshot_n else 0.0


@dataclass
class BacktestResult:
    wallets: list = field(default_factory=list)
    stake: float = 15.0
    costs: CostModel = field(default_factory=CostModel)
    by_category: dict = field(default_factory=dict)   # cat -> {n,wins,stake,pnl,pnl_gross}

    @property
    def n(self): return sum(w.n for w in self.wallets)
    @property
    def wins(self): return sum(w.wins for w in self.wallets)
    @property
    def pnl(self): return sum(w.pnl for w in self.wallets)
    @property
    def pnl_gross(self): return sum(w.pnl_gross for w in self.wallets)
    @property
    def stake_total(self): return sum(w.stake_total for w in self.wallets)
    @property
    def win_rate(self): return self.wins / self.n if self.n else 0.0
    @property
    def roi(self): return self.pnl / self.stake_total if self.stake_total else 0.0
    @property
    def roi_gross(self): return self.pnl_gross / self.stake_total if self.stake_total else 0.0


def _qualifying_buys(events, entry_min, entry_max, min_usd):
    for e in events:
        if (e.get("type") or "").upper() != "TRADE":
            continue
        if (e.get("side") or "").upper() != "BUY":
            continue
        try:
            price = float(e.get("price") or 0.0)
            usd = float(e.get("usdcSize") or 0.0)
        except (TypeError, ValueError):
            continue
        if usd < min_usd or price <= 0 or not (entry_min <= price <= entry_max):
            continue
        cid = str(e.get("conditionId") or "")
        if cid:
            yield (cid, int(e.get("outcomeIndex") or 0), price,
                   e.get("title") or "", e.get("slug") or "",
                   int(e.get("timestamp") or 0))


def _cat_add(acc, cat, *, stake, gross, net, won):
    d = acc.setdefault(cat, {"n": 0, "wins": 0, "stake": 0.0, "pnl": 0.0, "pnl_gross": 0.0})
    d["n"] += 1
    d["stake"] += stake
    d["pnl"] += net
    d["pnl_gross"] += gross
    if won:
        d["wins"] += 1


def _compute(wallet, events, resolved, *, label, stake, entry_min, entry_max,
             min_usd, costs, allow=None, block=None, cat_acc=None):
    res = WalletBacktest(wallet=wallet, label=label)
    seen = set()  # one copy per (market, outcome) — don't double-count adds
    # Sort chronologically so the FIRST entry per (cid,idx) is the genuine initial
    # entry, not whatever the API happened to return first (it returns newest-first,
    # which would otherwise copy the wallet's latest add at a worse, post-info price).
    buys = sorted(_qualifying_buys(events, entry_min, entry_max, min_usd), key=lambda x: x[5])
    for cid, idx, price, title, slug, _ts in buys:
        if (cid, idx) in seen:
            continue
        cat = categorize(title, slug)
        if not passes_category_filter(cat, allow, block):
            continue
        market = resolved.get(cid)
        if market is None or not market.resolved:
            continue
        seen.add((cid, idx))
        won = market.resolved_outcome_index() == idx
        shares_g = stake / price
        gross = (shares_g - stake) if won else -stake
        net = costs.net_pnl(price, stake, won)
        res.n += 1
        res.stake_total += stake
        res.pnl_gross += gross
        res.pnl += net
        if won:
            res.wins += 1
        if price <= 0.30:
            res.longshot_n += 1
            if won:
                res.longshot_wins += 1
        if cat_acc is not None:
            _cat_add(cat_acc, cat, stake=stake, gross=gross, net=net, won=won)
    return res


def copy_records(events, resolved, *, stake=15.0, entry_min=0.05, entry_max=0.95,
                 min_usd=100.0, costs=None, allow=None, block=None,
                 ts_lo=None, ts_hi=None, dedupe=True):
    """Yield ONE record per copied trade (not an aggregate) over an event list.

    This is the granular layer the walk-forward and the baselines share. Each
    record carries everything we need for honest statistics:

      price : the entry price we copied at (chronological first entry per side)
      won   : did the held side win at resolution
      net   : net P&L of the copy after CostModel (slippage/fees)
      gross : gross P&L before costs
      alpha : outcome(1/0) - entry_price  -> how much the trade beat the
              market's own implied probability (the cleaner edge signal than
              win-rate, per the scorer's open critique)
      ts/cat: timestamp + category, for windowing and per-category breakdowns

    `ts_lo`/`ts_hi` window by timestamp (walk-forward train/test split, with an
    embargo applied by the caller). `dedupe` keeps one copy per (market,outcome)
    within this event list, copying the chronological FIRST qualifying entry.
    """
    costs = costs or CostModel()
    out = []
    seen = set()
    buys = sorted(_qualifying_buys(events, entry_min, entry_max, min_usd),
                  key=lambda x: x[5])
    for cid, idx, price, title, slug, ts in buys:
        if ts_lo is not None and ts < ts_lo:
            continue
        if ts_hi is not None and ts > ts_hi:
            continue
        if dedupe and (cid, idx) in seen:
            continue
        cat = categorize(title, slug)
        if not passes_category_filter(cat, allow, block):
            continue
        market = resolved.get(cid)
        if market is None or not market.resolved:
            continue
        if dedupe:
            seen.add((cid, idx))
        won = market.resolved_outcome_index() == idx
        gross = (stake / price - stake) if won else -stake
        net = costs.net_pnl(price, stake, won)
        alpha = (1.0 if won else 0.0) - price
        out.append({"cid": cid, "idx": idx, "price": price, "won": won,
                    "cat": cat, "ts": ts, "gross": gross, "net": net,
                    "alpha": alpha, "title": title, "slug": slug})
    return out


def favorite_records(events, resolved, *, stake=15.0, entry_min=0.50,
                     entry_max=0.95, min_usd=100.0, costs=None, allow=None,
                     block=None, ts_lo=None, ts_hi=None):
    """Structural favorite-longshot baseline: one fixed-stake bet on each touched
    market's FAVORITE side (the in-band outcome the market priced highest, using
    observed entry prices as the tradeable proxy), held to resolution, net of
    costs. Skill-free; the copy strategy must beat it to justify any selection.

    SINGLE PASS, O(events): `events` may be ANY iterable (so the caller need not
    materialize a giant merged list). Per market we track the highest in-band buy
    price (the favorite side) and the earliest in-window timestamp. The previous
    implementation rescanned all events for every market -> O(markets x events),
    which is what made big runs hang.
    """
    costs = costs or CostModel()
    fav: dict = {}   # cid -> {idx, price, ts, title, slug}
    for e in events:
        if (e.get("type") or "").upper() != "TRADE":
            continue
        if (e.get("side") or "").upper() != "BUY":
            continue
        try:
            price = float(e.get("price") or 0.0)
            usd = float(e.get("usdcSize") or 0.0)
        except (TypeError, ValueError):
            continue
        if usd < min_usd or not (entry_min <= price <= entry_max):
            continue
        ts = int(e.get("timestamp") or 0)
        if ts_lo is not None and ts < ts_lo:
            continue
        if ts_hi is not None and ts > ts_hi:
            continue
        cid = str(e.get("conditionId") or "")
        if not cid:
            continue
        rec = fav.get(cid)
        if rec is None:
            rec = {"idx": int(e.get("outcomeIndex") or 0), "price": price,
                   "ts": ts, "title": e.get("title") or "", "slug": e.get("slug") or ""}
            fav[cid] = rec
        else:
            if ts < rec["ts"]:
                rec["ts"] = ts
            if price > rec["price"]:
                rec["price"] = price
                rec["idx"] = int(e.get("outcomeIndex") or 0)
    out = []
    for cid, rec in fav.items():
        market = resolved.get(cid)
        if market is None or not market.resolved:
            continue
        cat = categorize(rec["title"], rec["slug"])
        if not passes_category_filter(cat, allow, block):
            continue
        price = rec["price"]
        won = market.resolved_outcome_index() == rec["idx"]
        gross = (stake / price - stake) if won else -stake
        net = costs.net_pnl(price, stake, won)
        alpha = (1.0 if won else 0.0) - price
        out.append({"cid": cid, "idx": rec["idx"], "price": price, "won": won,
                    "cat": cat, "ts": rec["ts"], "gross": gross, "net": net,
                    "alpha": alpha, "title": rec["title"], "slug": rec["slug"]})
    return out


def _load_cache(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {cid: Market(condition_id=cid, closed=bool(m.get("closed")),
                        outcomes=m.get("outcomes") or [],
                        outcome_prices=m.get("outcome_prices") or [],
                        end_date=m.get("end_date") or "")
            for cid, m in raw.items()}


def _save_cache(path, resolved):
    if not path:
        return
    data = {cid: {"closed": m.closed, "outcome_prices": m.outcome_prices,
                  "outcomes": m.outcomes, "end_date": m.end_date}
            for cid, m in resolved.items() if m is not None}
    try:
        Path(path).write_text(json.dumps(data))
    except OSError:
        pass


def backtest_wallet(client, wallet, market_cache, *, label="", stake=15.0,
                    max_events=5000, entry_min=0.05, entry_max=0.95, min_usd=100.0,
                    costs=None, allow=None, block=None):
    costs = costs or CostModel()
    events = _pull_activity(client, wallet, max_events)
    cids = {cid for cid, _, _, _, _, _ in _qualifying_buys(events, entry_min, entry_max, min_usd)}
    for cid in cids:
        if cid not in market_cache:
            market_cache[cid] = client.get_market(cid)
    return _compute(wallet, events, market_cache, label=label, stake=stake,
                    entry_min=entry_min, entry_max=entry_max, min_usd=min_usd,
                    costs=costs, allow=allow, block=block)


def backtest_copy(client, targets, *, stake=15.0, max_events=5000,
                  entry_min=0.05, entry_max=0.95, min_usd=100.0, max_wallets=0,
                  workers=16, cache_path=None, costs=None, allow=None, block=None,
                  progress=None):
    costs = costs or CostModel()
    result = BacktestResult(stake=stake, costs=costs)
    items = list(targets)
    if max_wallets > 0:
        items = items[:max_wallets]
    addrs = [getattr(t, "address", t) for t in items]
    labels = {getattr(t, "address", t): getattr(t, "label", "") for t in items}

    events_by_wallet = {}

    def _pull(addr):
        try:
            return addr, _pull_activity(client, addr, max_events)
        except Exception:
            return addr, []

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for addr, events in ex.map(_pull, addrs):
            events_by_wallet[addr] = events
            done += 1
            if progress:
                progress(done, len(addrs), phase="fetch")

    resolved = _load_cache(cache_path)
    needed = set()
    for events in events_by_wallet.values():
        for cid, _, _, _, _, _ in _qualifying_buys(events, entry_min, entry_max, min_usd):
            if cid not in resolved:
                needed.add(cid)
    if needed:
        if hasattr(client, "get_markets"):
            fetched = client.get_markets(list(needed), closed=True)
        else:
            fetched = {c: client.get_market(c) for c in needed}
        for cid in needed:
            resolved[cid] = fetched.get(cid)
        _save_cache(cache_path, resolved)

    cat_acc: dict = {}
    for addr in addrs:
        result.wallets.append(_compute(
            addr, events_by_wallet.get(addr, []), resolved,
            label=labels.get(addr, ""), stake=stake, entry_min=entry_min,
            entry_max=entry_max, min_usd=min_usd, costs=costs,
            allow=allow, block=block, cat_acc=cat_acc))
    result.by_category = cat_acc
    result.wallets.sort(key=lambda w: w.pnl, reverse=True)
    return result


def format_backtest(result: BacktestResult) -> str:
    c = result.costs
    lines = ["=== Polywatch Copy-Trade Backtest (historical, resolved markets) ==="]
    lines.append(
        f"Stake/copy ${result.stake:,.0f} | settled {result.n} | win {result.win_rate:.0%} | "
        f"costs: slippage {c.slippage_points:.0%}pts, fee {c.fee_pct:.1%}"
    )
    lines.append(
        f"GROSS  P&L ${result.pnl_gross:,.0f}  ROI {result.roi_gross:+.1%}     "
        f"NET  P&L ${result.pnl:,.0f}  ROI {result.roi:+.1%}"
    )

    # Per-category breakdown — where does the edge actually live?
    if result.by_category:
        lines.append("")
        lines.append("--- By category (net) ---")
        ch = f"{'category':10}  {'n':>5}  {'win':>4}  {'netP&L':>10}  {'netROI':>7}"
        lines.append(ch)
        lines.append("-" * len(ch))
        for cat in sorted(result.by_category, key=lambda k: result.by_category[k]["pnl"], reverse=True):
            d = result.by_category[cat]
            roi = d["pnl"] / d["stake"] if d["stake"] else 0.0
            wr = d["wins"] / d["n"] if d["n"] else 0.0
            lines.append(f"{cat:10}  {d['n']:>5}  {wr:>4.0%}  ${d['pnl']:>9,.0f}  {roi:>+6.0%}")

    lines.append("")
    hdr = (f"{'wallet/label':32}  {'n':>4}  {'win':>4}  {'lsWR':>5}  {'staked':>9}  "
           f"{'grossROI':>8}  {'netP&L':>10}  {'netROI':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for w in result.wallets:
        if w.n == 0:
            continue
        name = (w.label or w.wallet)[:32]
        lines.append(
            f"{name:32}  {w.n:>4}  {w.win_rate:>4.0%}  {w.longshot_win_rate:>5.0%}  "
            f"${w.stake_total:>8,.0f}  {w.roi_gross:>+7.0%}  ${w.pnl:>9,.0f}  {w.roi:>+6.0%}"
        )
    if result.n == 0:
        lines.append("(no settled copy trades found — try more wallets or higher --max-events)")
    return "\n".join(lines)


def backtest_consensus(client, targets, *, k=2, window_hours=24, stake=15.0,
                       max_events=5000, entry_min=0.05, entry_max=0.95, min_usd=100.0,
                       max_wallets=0, workers=16, cache_path=None, costs=None,
                       allow=None, block=None, progress=None):
    """Backtest the K-of-N consensus strategy: copy a market/side once only,
    when K distinct tracked wallets have bought it within `window_hours`."""
    costs = costs or CostModel()
    result = BacktestResult(stake=stake, costs=costs)
    items = list(targets)
    if max_wallets > 0:
        items = items[:max_wallets]
    addrs = [getattr(t, "address", t) for t in items]

    events_by_wallet = {}

    def _pull(addr):
        try:
            return addr, _pull_activity(client, addr, max_events)
        except Exception:
            return addr, []

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for addr, events in ex.map(_pull, addrs):
            events_by_wallet[addr] = events
            done += 1
            if progress:
                progress(done, len(addrs), phase="fetch")

    # Flatten qualifying buys across ALL wallets, carrying ts + wallet.
    flat = []
    for addr in addrs:
        for e in events_by_wallet.get(addr, []):
            if (e.get("type") or "").upper() != "TRADE" or (e.get("side") or "").upper() != "BUY":
                continue
            try:
                price = float(e.get("price") or 0.0); usd = float(e.get("usdcSize") or 0.0)
            except (TypeError, ValueError):
                continue
            if usd < min_usd or price <= 0 or not (entry_min <= price <= entry_max):
                continue
            cid = str(e.get("conditionId") or "")
            if not cid:
                continue
            cat = categorize(e.get("title") or "", e.get("slug") or "")
            if not passes_category_filter(cat, allow, block):
                continue
            flat.append((int(e.get("timestamp") or 0), cid, int(e.get("outcomeIndex") or 0),
                         price, cat, addr))
    flat.sort(key=lambda x: x[0])

    # Resolve all involved markets in one batch.
    resolved = _load_cache(cache_path)
    needed = {cid for _, cid, _, _, _, _ in flat if cid not in resolved}
    if needed:
        if hasattr(client, "get_markets"):
            fetched = client.get_markets(list(needed), closed=True)
        else:
            fetched = {c: client.get_market(c) for c in needed}
        for cid in needed:
            resolved[cid] = fetched.get(cid)
        _save_cache(cache_path, resolved)

    tracker = ConsensusTracker(k=k, window_seconds=window_hours * 3600)
    agg = WalletBacktest(wallet="consensus", label=f"consensus K={k} / {window_hours}h")
    cat_acc: dict = {}
    for ts, cid, idx, price, cat, wallet in flat:
        market = resolved.get(cid)
        if market is None or not market.resolved:
            continue  # only settle on resolved markets
        if not tracker.observe(cid, idx, wallet, ts):
            continue  # not yet K-of-N
        won = market.resolved_outcome_index() == idx
        gross = (stake / price - stake) if won else -stake
        net = costs.net_pnl(price, stake, won)
        agg.n += 1; agg.stake_total += stake; agg.pnl_gross += gross; agg.pnl += net
        if won:
            agg.wins += 1
        if price <= 0.30:
            agg.longshot_n += 1
            if won:
                agg.longshot_wins += 1
        _cat_add(cat_acc, cat, stake=stake, gross=gross, net=net, won=won)

    result.wallets = [agg]
    result.by_category = cat_acc
    return result
