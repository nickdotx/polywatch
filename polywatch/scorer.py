"""Accurate, loss-inclusive wallet scorer with an "informed-edge" profile.

Reconstructs a wallet's realized P&L from its full /activity log. Tracks shares
PER OUTCOME across the full event vocabulary (BUY/SELL, REDEEM, SPLIT, MERGE,
CONVERSION) so split/arbitrage wallets settle correctly, and classifies edge by
the price of the side actually bought.

Settlement model (per market):
  cost   = buy_usdc + split_usdc                       (cash out)
  proceeds = sell_usdc + redeem_usdc + merge_usdc      (cash in)
  per-outcome shares: BUY/ SPLIT add (split adds to BOTH legs); SELL/MERGE
    remove (merge removes from BOTH legs); REDEEM removes from the winning leg.
  If shares are fully gone -> realized = proceeds - cost.
  Else look up resolution: winning held shares pay $1 each -> add to proceeds.

negRisk CONVERSION (NO of one market -> YES of all OTHER markets in a categorical
set + USDC) spans multiple conditionIds, so it can't be settled in this per-cid
model without phantom P&L. Any market touched by a conversion is DROPPED from the
records (and counted) rather than mis-settled.

"Won" is split into two distinct ideas, because they were conflated before:
  - profitable     : realized P&L > 0 (could be a mid-market trade-out, not a call)
  - predicted right: the wallet's dominant-bought outcome == the resolved outcome
The edge signals (longshot win rate, avg winning entry) now use *resolution-
confirmed* correctness, not mere profitability, so trading out of a longshot for a
gain no longer masquerades as an information edge.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .polymarket import PolymarketClient

NINETY_DAYS = 90 * 86400
EPS_SHARES = 1.0
FULL_HISTORY = 5000

MIN_SETTLED = 30          # below this, a fat-tailed P&L can't separate skill from luck
DOMINANCE_CAP = 0.55
LONGSHOT_MAX = 0.30
FAVORITE_MIN = 0.70
TWO_SIDED_CAP = 0.50      # > this share of markets traded two-sided -> looks like arb/MM
EXTREME_PRICE = 0.97      # near-certainty entries (favorite farming / spread capture)

_cache_lock = threading.Lock()


@dataclass
class _MarketAgg:
    cid: str
    title: str = ""
    buy_usdc: float = 0.0
    split_usdc: float = 0.0
    sell_usdc: float = 0.0
    redeem_usdc: float = 0.0
    merge_usdc: float = 0.0
    redeem_shares: float = 0.0          # winning-leg shares converted to cash
    redeemed: bool = False
    had_conversion: bool = False        # negRisk cross-market op: unreliable here -> drop
    last_ts: int = 0
    acq: dict = field(default_factory=dict)   # outcome_index -> shares acquired
    disp: dict = field(default_factory=dict)  # outcome_index -> shares disposed (sell/merge)
    buy_usdc_by_idx: dict = field(default_factory=dict)
    buy_shares_by_idx: dict = field(default_factory=dict)


@dataclass
class WalletScore:
    address: str
    realized_pnl: float
    roi: float
    n_settled: int
    win_rate: float
    avg_entry_win: float
    longshot_win_rate: float
    dominance: float
    recent: bool
    open_markets: int
    score: float
    predicted_win_rate: float = 0.0     # resolution-confirmed correctness rate
    two_sided_rate: float = 0.0         # share of markets traded on both legs (arb/MM tell)
    conv_dropped: int = 0              # markets dropped due to negRisk conversions
    label: str = ""


def _event_key(e: dict):
    """Stable identity for an activity event, so paginated overlap can't double-count.
    Two byte-identical events collapse; genuinely distinct fills differ in at least
    one field (tx hash, ts, size, usdc, side, outcome) and are kept."""
    return (
        e.get("transactionHash"),
        (e.get("type") or "").upper(),
        (e.get("side") or "").upper(),
        str(e.get("conditionId") or ""),
        e.get("outcomeIndex"),
        e.get("size"),
        e.get("usdcSize"),
        e.get("timestamp"),
    )


def _pull_activity(client: PolymarketClient, wallet: str, max_events: int) -> list[dict]:
    out: list[dict] = []
    seen: set = set()
    offset = 0
    page = 500
    while len(out) < max_events:
        data = client._get(
            f"{client.data_api}/activity",
            {"user": wallet, "limit": page, "offset": offset},
        )
        if not isinstance(data, list) or not data:
            break
        for e in data:
            if not isinstance(e, dict):
                continue
            k = _event_key(e)
            if k in seen:
                continue          # paginated duplicate — drop it
            seen.add(k)
            out.append(e)
        if len(data) < page:
            break
        offset += page
    return out[:max_events]


def _leftover(m: "_MarketAgg") -> float:
    """Net shares still held across both legs (before crediting winners)."""
    acq = sum(m.acq.values())
    disp = sum(m.disp.values()) + m.redeem_shares
    return acq - disp


def reconstruct(
    client: PolymarketClient, wallet: str, market_cache: dict,
    max_events: int = FULL_HISTORY, stats: dict | None = None,
    events: list[dict] | None = None, cutoff_ts: int | None = None,
    offline: bool = False,
) -> list[dict]:
    """Return settled-market records with realized P&L for a wallet.

    If `stats` is given, fills in {"conv_dropped": int, "markets_seen": int}.

    Walk-forward hooks (both optional, backward-compatible):
      - `events`: pre-fetched /activity events to reuse instead of fetching
        again (the walk-forward pulls each wallet's history exactly once).
      - `cutoff_ts`: if set, only events with timestamp <= cutoff_ts are used,
        so a wallet can be scored *as it would have looked at time T* — the
        in-sample half of a walk-forward split. (Resolution-by-T look-ahead is
        handled one layer up, in walkforward.py, using market end dates.)
    """
    if events is None:
        events = _pull_activity(client, wallet, max_events)
    if cutoff_ts is not None:
        events = [e for e in events if int(e.get("timestamp") or 0) <= cutoff_ts]
    markets: dict[str, _MarketAgg] = {}

    for e in events:
        etype = (e.get("type") or "").upper()
        if etype not in ("TRADE", "REDEEM", "SPLIT", "MERGE", "CONVERSION"):
            continue
        cid = str(e.get("conditionId") or "")
        if not cid:
            continue
        m = markets.get(cid)
        if m is None:
            m = _MarketAgg(cid=cid, title=e.get("title") or "")
            markets[cid] = m
        usdc = float(e.get("usdcSize") or 0.0)
        size = float(e.get("size") or 0.0)
        idx = int(e.get("outcomeIndex") or 0)
        m.last_ts = max(m.last_ts, int(e.get("timestamp") or 0))
        side = (e.get("side") or "").upper()

        if etype == "REDEEM":
            m.redeem_usdc += usdc
            m.redeem_shares += size
            m.redeemed = True
        elif etype == "SPLIT":
            m.split_usdc += usdc
            m.acq[0] = m.acq.get(0, 0.0) + size   # collateral -> Yes AND No
            m.acq[1] = m.acq.get(1, 0.0) + size
        elif etype == "MERGE":
            m.merge_usdc += usdc
            m.disp[0] = m.disp.get(0, 0.0) + size  # consumes Yes AND No
            m.disp[1] = m.disp.get(1, 0.0) + size
        elif etype == "CONVERSION":
            # negRisk: NO -> YES(all other markets) + USDC. Spans multiple cids;
            # can't be settled correctly here, so flag the market for dropping.
            m.had_conversion = True
        elif side == "BUY":
            m.buy_usdc += usdc
            m.acq[idx] = m.acq.get(idx, 0.0) + size
            m.buy_usdc_by_idx[idx] = m.buy_usdc_by_idx.get(idx, 0.0) + usdc
            m.buy_shares_by_idx[idx] = m.buy_shares_by_idx.get(idx, 0.0) + size
        elif side == "SELL":
            m.sell_usdc += usdc
            m.disp[idx] = m.disp.get(idx, 0.0) + size

    # Markets needing a resolution lookup (one batch): any with leftover held
    # shares, OR redeemed (so we can learn which outcome won and confirm the
    # call). Conversion-touched markets are dropped, so they never need it.
    need = []
    conv_dropped = 0
    for cid, m in markets.items():
        if m.had_conversion:
            conv_dropped += 1
            continue
        if (m.buy_usdc + m.split_usdc) <= 0:
            continue
        if _leftover(m) >= EPS_SHARES or m.redeemed:
            with _cache_lock:
                cached = cid in market_cache
            if not cached:
                need.append(cid)
    if need and not offline and hasattr(client, "get_markets"):
        fetched = client.get_markets(need, closed=True)
        with _cache_lock:
            for cid in need:
                market_cache[cid] = fetched.get(cid)

    if stats is not None:
        stats["conv_dropped"] = conv_dropped
        stats["markets_seen"] = len(markets)

    records = []
    for cid, m in markets.items():
        if m.had_conversion:
            continue                               # cross-market negRisk op — can't settle here
        cost_basis = m.buy_usdc + m.split_usdc
        if cost_basis <= 0:
            continue
        proceeds = m.sell_usdc + m.redeem_usdc + m.merge_usdc
        leftover = _leftover(m)

        if leftover < EPS_SHARES:
            settled = True                         # fully closed via cash events
        else:
            with _cache_lock:
                mk = market_cache.get(cid, "MISS")
            if mk == "MISS":
                if offline:
                    continue          # walk-forward: don't hit the network per wallet
                mk = client.get_market(cid)
                with _cache_lock:
                    market_cache[cid] = mk
            if mk is not None and mk.resolved:
                ridx = mk.resolved_outcome_index()
                if ridx is not None:
                    held_win = m.acq.get(ridx, 0.0) - m.disp.get(ridx, 0.0) - m.redeem_shares
                    if held_win > 0:
                        proceeds += held_win       # winning shares pay $1 each
                settled = True
            else:
                continue                           # genuinely open

        realized = proceeds - cost_basis

        if m.buy_shares_by_idx:
            dom = max(m.buy_shares_by_idx, key=m.buy_shares_by_idx.get)
            dom_shares = m.buy_shares_by_idx.get(dom, 0.0)
            avg_entry = (m.buy_usdc_by_idx.get(dom, 0.0) / dom_shares) if dom_shares > 0 else 0.0
        else:
            dom = None
            avg_entry = 0.0

        # Resolution-confirmed correctness (free read from cache when present).
        win_idx = None
        with _cache_lock:
            cmk = market_cache.get(cid)
        if cmk not in (None, "MISS") and getattr(cmk, "resolved", False):
            win_idx = cmk.resolved_outcome_index()
        resolved_known = win_idx is not None
        predicted_correct = bool(resolved_known and dom is not None and dom == win_idx)

        # Two-sided tell: bought both legs, or split (intrinsically both legs).
        bought_legs = sum(1 for s in m.buy_shares_by_idx.values() if s > 0)
        two_sided = bought_legs >= 2 or m.split_usdc > 0

        records.append({
            "cid": cid, "title": m.title, "realized": realized,
            "cost": cost_basis, "avg_entry": avg_entry, "last_ts": m.last_ts,
            "won": realized > 0,
            "resolved_known": resolved_known,
            "predicted_correct": predicted_correct,
            "two_sided": two_sided,
        })
    return records


def score_wallet(
    client: PolymarketClient, wallet: str, market_cache: dict,
    max_events: int = FULL_HISTORY,
    events: list[dict] | None = None, cutoff_ts: int | None = None,
    now_ts: float | None = None, offline: bool = False,
) -> "WalletScore | None":
    """Score a wallet's informed-edge profile.

    `events`/`cutoff_ts` enable point-in-time (walk-forward training) scoring;
    `now_ts` anchors the recency penalty to the cutoff instead of wall-clock now,
    so a wallet scored "as of T" isn't unfairly marked stale using today's date.
    """
    stats: dict = {}
    recs = reconstruct(client, wallet, market_cache, max_events, stats=stats,
                       events=events, cutoff_ts=cutoff_ts, offline=offline)
    if not recs:
        return None
    now_ref = time.time() if now_ts is None else now_ts

    n = len(recs)
    realized_total = sum(r["realized"] for r in recs)
    total_cost = sum(r["cost"] for r in recs)
    wins = [r for r in recs if r["won"]]                  # profitable markets (incl. trade-outs)
    win_rate = len(wins) / n if n else 0.0
    roi = realized_total / total_cost if total_cost > 0 else 0.0

    gross_profit = sum(r["realized"] for r in recs if r["realized"] > 0)
    best = max((r["realized"] for r in recs), default=0.0)
    dominance = (best / gross_profit) if gross_profit > 0 else 1.0

    # Edge signals now use RESOLUTION-CONFIRMED correctness, not mere profitability.
    resolved = [r for r in recs if r["resolved_known"]]
    correct = [r for r in resolved if r["predicted_correct"]]
    predicted_win_rate = (len(correct) / len(resolved)) if resolved else 0.0
    avg_entry_win = (sum(r["avg_entry"] for r in correct) / len(correct)) if correct else 0.0

    longshots = [r for r in resolved if 0 < r["avg_entry"] <= LONGSHOT_MAX]
    longshot_wins = [r for r in longshots if r["predicted_correct"]]
    longshot_win_rate = (len(longshot_wins) / len(longshots)) if longshots else 0.0

    two_sided_rate = sum(1 for r in recs if r["two_sided"]) / n if n else 0.0

    last_ts = max((r["last_ts"] for r in recs), default=0)
    recent = (now_ref - last_ts) < NINETY_DAYS if last_ts else False

    score = realized_total
    if n < MIN_SETTLED:
        score *= n / MIN_SETTLED
    if dominance > DOMINANCE_CAP and realized_total > 0:
        score *= max(0.1, 1 - (dominance - DOMINANCE_CAP))
    if not recent:
        score *= 0.5
    edge_mult = 1.0 + longshot_win_rate
    if correct and avg_entry_win >= FAVORITE_MIN:
        edge_mult *= 0.6
    score *= edge_mult
    score *= (0.5 + win_rate)
    # Arb/market-maker tell: mostly two-sided flow isn't a copyable directional edge.
    if two_sided_rate > TWO_SIDED_CAP and realized_total > 0:
        score *= 0.3

    return WalletScore(
        address=wallet, realized_pnl=realized_total, roi=roi, n_settled=n,
        win_rate=win_rate, avg_entry_win=avg_entry_win,
        longshot_win_rate=longshot_win_rate, dominance=dominance, recent=recent,
        open_markets=0, score=score, predicted_win_rate=predicted_win_rate,
        two_sided_rate=two_sided_rate, conv_dropped=stats.get("conv_dropped", 0),
    )
