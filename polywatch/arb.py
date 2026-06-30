"""Polymarket-internal arbitrage scanner (read-only, measurement-first).

The thesis to TEST before building any fast execution stack: can a
"buy YES + buy NO for less than $1" lock be captured at a latency a normal
operator can actually build, net of fees? On Polymarket a YES share + a NO share
of the same binary market can be **merged back into $1 USDC instantly** (you
don't even wait for resolution), so if best_ask(YES) + best_ask(NO) < $1 you have
a risk-free, capital-light lock — IF you can grab it before someone else does.

This module is read-only. It reads the CLOB order books of the top markets and
reports every market where that inequality holds, with the size available and
the dollar edge. The `arb --watch` CLI loop re-scans on an interval and logs how
LONG each gap survives — the single number that decides whether internal arb is
capturable by us or vanishes in milliseconds (in which case cross-platform arb,
not HFT, is the only viable route).

Nothing here places an order or needs a key. It only measures whether the
opportunity is real before a cent of engineering or capital goes into execution.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class ArbOpp:
    cid: str
    question: str
    yes_ask: float
    no_ask: float
    size: float            # min(ask sizes) — number of $1 pairs lockable now
    edge_per_pair: float   # 1 - (yes_ask + no_ask) - fee  (profit per pair)
    cost: float            # yes_ask + no_ask
    ts: int

    @property
    def edge_pct(self) -> float:
        return self.edge_per_pair / self.cost if self.cost else 0.0

    @property
    def max_profit(self) -> float:
        return self.edge_per_pair * self.size


def _parse_tokens(raw) -> list[str]:
    """clobTokenIds from gamma may be a JSON string or a list."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str) and raw:
        try:
            arr = json.loads(raw)
            return [str(t) for t in arr] if isinstance(arr, list) else []
        except json.JSONDecodeError:
            return []
    return []


def scan_internal_arb(client, n_markets=200, fee=0.0, min_edge=0.0,
                      workers=16, now=None):
    """Scan top `n_markets` for binary markets where ask(YES)+ask(NO) < 1 - fee.

    Returns a list of ArbOpp sorted by max lockable profit (desc). `min_edge` is
    the minimum per-pair edge to report (e.g. 0.005 = half a cent)."""
    now = int(now or time.time())
    try:
        markets = client.get_top_markets(limit=n_markets) or []
    except Exception:
        return []

    def _check(m):
        try:
            cid = str(m.get("conditionId") or "")
            toks = _parse_tokens(m.get("clobTokenIds"))
            if len(toks) != 2:          # only plain binary markets
                return None
            b0 = client.get_book(toks[0])
            b1 = client.get_book(toks[1])
            if not b0 or not b1:
                return None
            a0, a1 = b0.get("best_ask"), b1.get("best_ask")
            if a0 is None or a1 is None:
                return None
            cost = a0 + a1
            edge = (1.0 - cost) - fee
            if edge <= min_edge:
                return None
            size = min(b0.get("ask_size") or 0.0, b1.get("ask_size") or 0.0)
            return ArbOpp(cid=cid, question=m.get("question") or m.get("title") or "",
                          yes_ask=a0, no_ask=a1, size=size, edge_per_pair=edge,
                          cost=cost, ts=now)
        except Exception:
            return None

    opps = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_check, markets):
            if r is not None:
                opps.append(r)
    opps.sort(key=lambda o: o.max_profit, reverse=True)
    return opps


def format_arb(opps, n_scanned) -> str:
    lines = ["=== Polymarket internal-arb scan (YES_ask + NO_ask < $1) ==="]
    lines.append(f"scanned {n_scanned} markets | found {len(opps)} lockable gap(s)")
    if not opps:
        lines.append("(no risk-free internal-arb gaps right now — the usual case "
                     "in an efficient book; run --watch to sample over time)")
        return "\n".join(lines)
    hdr = (f"{'edge/pair':>9}  {'edge%':>6}  {'size':>9}  {'maxProfit':>9}  "
           f"{'YESask':>6}  {'NOask':>6}  market")
    lines.append("")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for o in opps:
        q = (o.question or o.cid)[:46]
        lines.append(f"${o.edge_per_pair:>8.3f}  {o.edge_pct:>5.1%}  {o.size:>9,.0f}  "
                     f"${o.max_profit:>8,.0f}  {o.yes_ask:>6.3f}  {o.no_ask:>6.3f}  {q}")
    lines.append("")
    lines.append("note: a lock = buy 1 YES + 1 NO, merge to $1 USDC. 'maxProfit' is "
                 "edge x top-of-book size, before slippage/latency. The real question "
                 "is how long these survive — use --watch.")
    return "\n".join(lines)


class LifetimeTracker:
    """Tracks how long each (market) arb gap persists across repeated scans, so we
    can measure whether gaps are capturable (seconds) or gone instantly (ms)."""

    def __init__(self):
        self.first_seen: dict = {}
        self.last_seen: dict = {}
        self.best_edge: dict = {}
        self.samples: dict = {}

    def observe(self, opps, ts):
        live = set()
        for o in opps:
            live.add(o.cid)
            self.first_seen.setdefault(o.cid, ts)
            self.last_seen[o.cid] = ts
            self.samples[o.cid] = self.samples.get(o.cid, 0) + 1
            self.best_edge[o.cid] = max(self.best_edge.get(o.cid, 0.0), o.edge_per_pair)
        return live

    def rows(self):
        """One row per gap ever seen: cid, lifetime_seconds, samples, best_edge."""
        out = []
        for cid in self.first_seen:
            life = self.last_seen[cid] - self.first_seen[cid]
            out.append({"cid": cid, "lifetime_s": life,
                        "samples": self.samples[cid],
                        "best_edge": round(self.best_edge[cid], 4),
                        "first_seen": self.first_seen[cid],
                        "last_seen": self.last_seen[cid]})
        out.sort(key=lambda r: r["lifetime_s"], reverse=True)
        return out
