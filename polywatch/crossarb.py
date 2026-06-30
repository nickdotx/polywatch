"""Cross-platform arbitrage scanner: Polymarket <-> Kalshi (read-only).

The latency-TOLERANT edge: because no single market-maker spans both venues, the
same real-world event can stay mispriced between Polymarket and Kalshi for
seconds-to-minutes — catchable without an HFT stack. If you can buy YES on the
cheaper venue and NO on the cheaper venue for a combined cost < $1 (net of both
venues' fees), you lock $1 of payout regardless of outcome.

SAFETY — this only scans HUMAN-VERIFIED event pairs. Auto-matching titles is the
#1 way to blow up at cross-platform arb: if the two contracts don't resolve by
*identical* rules, your "risk-free" lock has hidden risk. So you supply a pairs
file of pairs you've personally confirmed are the same contract (with an `invert`
flag when Polymarket-YES corresponds to Kalshi-NO). Nothing here places an order.

Capital note: when the YES leg is on one venue and the NO leg on the other, you
can't merge to cash instantly (that only works within Polymarket) — you hold both
to resolution and collect $1 from the winning side. The edge is locked, but the
capital is tied up until the market settles.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .categories import categorize
from .kalshi import kalshi_fee


@dataclass
class CrossArbOpp:
    name: str
    edge_per_pair: float    # 1 - total cost incl. fees
    cost: float
    size: float             # min lockable contracts across the two legs
    yes_leg: str            # venue to BUY YES on
    no_leg: str             # venue to BUY NO on
    yes_price: float
    no_price: float
    poly_cid: str
    kalshi_ticker: str
    ts: int

    @property
    def edge_pct(self) -> float:
        return self.edge_per_pair / self.cost if self.cost else 0.0

    @property
    def max_profit(self) -> float:
        return self.edge_per_pair * self.size


def _pick_leg(poly, kalshi, poly_fee, kalshi_rate):
    """Cheapest venue (incl. fee) to buy one leg. `poly`/`kalshi` = (price, size).
    Returns (venue, price, size, cost_incl_fee) or None."""
    opts = []
    if poly[0] is not None:
        opts.append(("polymarket", poly[0], poly[1] or 0.0, poly[0] + poly_fee))
    if kalshi[0] is not None:
        opts.append(("kalshi", kalshi[0], kalshi[1] or 0.0,
                     kalshi[0] + kalshi_fee(kalshi[0], 1.0, kalshi_rate)))
    return min(opts, key=lambda x: x[3]) if opts else None


def scan_pair(poly_client, kalshi_client, pair, *, poly_fee=0.0, kalshi_rate=0.07,
              min_edge=0.0, now=None):
    """Scan one verified pair. Returns a CrossArbOpp or None."""
    now = int(now or time.time())
    cid = str(pair.get("polymarket_condition_id") or "")
    yidx = int(pair.get("polymarket_yes_index", 0))
    if not cid or not pair.get("kalshi_ticker"):
        return None

    mk = poly_client.get_market(cid)
    toks = getattr(mk, "clob_token_ids", None) if mk is not None else None
    if not toks or len(toks) < 2:
        return None
    yes_tok = toks[yidx]
    no_tok = toks[1 - yidx]

    pby = poly_client.get_book(yes_tok)
    pbn = poly_client.get_book(no_tok)
    ob = kalshi_client.get_orderbook(pair["kalshi_ticker"])
    if not pby or not pbn or not ob:
        return None

    poly_yes = (pby.get("best_ask"), pby.get("ask_size"))
    poly_no = (pbn.get("best_ask"), pbn.get("ask_size"))
    if pair.get("invert"):       # Polymarket-YES == Kalshi-NO
        k_yes = (ob.get("no_ask"), ob.get("no_ask_size"))
        k_no = (ob.get("yes_ask"), ob.get("yes_ask_size"))
    else:
        k_yes = (ob.get("yes_ask"), ob.get("yes_ask_size"))
        k_no = (ob.get("no_ask"), ob.get("no_ask_size"))

    yes_pick = _pick_leg(poly_yes, k_yes, poly_fee, kalshi_rate)
    no_pick = _pick_leg(poly_no, k_no, poly_fee, kalshi_rate)
    if not yes_pick or not no_pick:
        return None

    cost = yes_pick[3] + no_pick[3]
    edge = 1.0 - cost
    if edge <= min_edge:
        return None
    size = min(yes_pick[2], no_pick[2])
    return CrossArbOpp(
        name=str(pair.get("name") or cid), edge_per_pair=edge, cost=cost, size=size,
        yes_leg=yes_pick[0], no_leg=no_pick[0], yes_price=yes_pick[1],
        no_price=no_pick[1], poly_cid=cid, kalshi_ticker=str(pair["kalshi_ticker"]),
        ts=now)


def scan_pairs(poly_client, kalshi_client, pairs, *, poly_fee=0.0, kalshi_rate=0.07,
               min_edge=0.0, now=None):
    out = []
    for pair in pairs:
        try:
            opp = scan_pair(poly_client, kalshi_client, pair, poly_fee=poly_fee,
                            kalshi_rate=kalshi_rate, min_edge=min_edge, now=now)
        except Exception:
            opp = None
        if opp is not None:
            out.append(opp)
    out.sort(key=lambda o: o.max_profit, reverse=True)
    return out


def format_crossarb(opps, n_pairs) -> str:
    L = ["=== Polymarket <-> Kalshi cross-platform arb scan ==="]
    L.append(f"checked {n_pairs} verified pair(s) | found {len(opps)} lock(s)")
    if not opps:
        L.append("(no cross-venue lock right now — normal; the venues agree. Keep "
                 "watching; gaps open when one venue moves and the other lags.)")
        return "\n".join(L)
    hdr = (f"{'edge/pair':>9}  {'edge%':>6}  {'size':>7}  {'maxProfit':>9}  "
           f"{'buy YES':>10}@{'':4} {'buy NO':>10}  event")
    L.append("")
    L.append(hdr)
    L.append("-" * len(hdr))
    for o in opps:
        L.append(f"${o.edge_per_pair:>8.3f}  {o.edge_pct:>5.1%}  {o.size:>7,.0f}  "
                 f"${o.max_profit:>8,.0f}  {o.yes_leg:>10}@{o.yes_price:<.2f} "
                 f"{o.no_leg:>10}@{o.no_price:<.2f}  {o.name[:40]}")
    L.append("")
    L.append("lock = buy YES on one venue + NO on the other; $1 payout at resolution "
             "regardless of outcome. Edge is net of est. fees but BEFORE slippage; "
             "capital is tied up until the market settles. VERIFY each pair resolves "
             "by identical rules before trusting it.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Candidate discovery: fuzzy-match both venues' open markets for HUMAN review.
# This NEVER trades — it just produces a shortlist of likely-same events so you
# don't have to hunt by hand. You still verify each pair resolves identically
# before adding it to pairs.yaml.
# ---------------------------------------------------------------------------

_STOP = {
    "will", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "by",
    "is", "be", "at", "as", "it", "this", "that", "with", "than", "more", "less",
    "before", "after", "during", "yes", "no", "market", "markets", "price", "win",
    "above", "below", "between", "end", "next", "who", "what", "when", "which",
    "2026", "2027", "2028",   # year is shared by ~everything -> not discriminating
}


def _tok(text: str) -> set[str]:
    out = set()
    for w in re.split(r"[^a-z0-9]+", (text or "").lower()):
        if len(w) >= 2 and w not in _STOP:
            out.add(w)
    return out


def _kalshi_text(m: dict) -> str:
    parts = [m.get("title") or "", m.get("subtitle") or "",
             m.get("yes_sub_title") or "", m.get("_event_title") or ""]
    return " ".join(p for p in parts if p)


def _kalshi_is_combo(text: str, ticker: str) -> bool:
    """Kalshi 'multivariate'/parlay markets have token-soup titles
    ("yes X advances, yes Y advances, ...") that falsely match everything. Skip
    them: they aren't a single contract you can arb against a Polymarket binary."""
    t = (text or "").lower()
    tk = (ticker or "").upper()
    if any(s in tk for s in ("MVE", "MULTIGAME", "CROSSCATEGORY", "MULTI")):
        return True
    if t.count(",") >= 2:
        return True
    if t.count("yes ") + t.count("no ") >= 2:
        return True
    return False


def discover_pairs(poly_markets, kalshi_markets, min_score=0.5, top=50):
    """Fuzzy-match Polymarket vs Kalshi open markets by title-token similarity.

    Scores with the OVERLAP COEFFICIENT (shared tokens / size of the smaller
    title) rather than Jaccard, so a short Kalshi title matching part of a long
    Polymarket question still scores high. Requires >= 2 shared meaningful tokens
    to avoid one-word coincidences. Inverted index keeps it fast over thousands of
    Kalshi markets. Returns candidate dicts sorted by similarity — NOT verified."""
    poly = []
    for m in poly_markets:
        q = m.get("question") or m.get("title") or ""
        cid = str(m.get("conditionId") or "")
        if categorize(q, m.get("slug") or "") == "sports":
            continue                 # Kalshi overlap lives in politics/econ/crypto
        toks = _tok(q)
        if cid and toks:
            poly.append((cid, q, toks))

    kal = []
    for m in kalshi_markets:
        text = _kalshi_text(m)
        tk = str(m.get("ticker") or "")
        if _kalshi_is_combo(text, tk):
            continue                 # drop parlay/combo token-soup markets
        toks = _tok(text)
        if tk and toks:
            kal.append((tk, text, toks, m))   # keep the raw market for rules/event

    index: dict[str, set[int]] = {}
    for i, (_tk, _t, toks, _m) in enumerate(kal):
        for w in toks:
            index.setdefault(w, set()).add(i)

    cands = []
    for cid, q, qt in poly:
        candidate_idx: set[int] = set()
        for w in qt:
            candidate_idx |= index.get(w, set())
        best = None
        for i in candidate_idx:
            tt = kal[i][2]
            inter = len(qt & tt)
            if inter < 2:                       # need >=2 shared meaningful tokens
                continue
            if inter / len(qt | tt) < 0.18:     # Jaccard floor: guard length mismatch
                continue
            score = inter / min(len(qt), len(tt))   # overlap coefficient
            if best is None or score > best[0]:
                best = (score, i)
        if best and best[0] >= min_score:
            s, i = best
            tk, t, _tt, m = kal[i]
            cands.append({"score": round(s, 3), "poly_cid": cid, "poly_q": q,
                          "kalshi_ticker": tk, "kalshi_title": m.get("title") or t,
                          "kalshi_event": m.get("_event_title") or "",
                          "kalshi_event_ticker": str(m.get("event_ticker") or ""),
                          "kalshi_rules": (m.get("rules_primary") or "")[:240]})
    cands.sort(key=lambda c: c["score"], reverse=True)
    # Collapse to one row per distinct Kalshi EVENT (its best Polymarket match),
    # so a single many-candidate event ("who will run...") can't bury everything.
    seen_ev = set()
    uniq = []
    for c in cands:
        ev = c.get("kalshi_event_ticker") or c["kalshi_ticker"]
        if ev in seen_ev:
            continue
        seen_ev.add(ev)
        uniq.append(c)
    return uniq[:top]


def filtered_titles(poly_markets, kalshi_markets):
    """Return (poly, kalshi) title lists after the same filtering discovery uses
    (drop Polymarket sports + Kalshi combo junk), so you can eyeball what each
    venue actually lists and spot real overlaps by hand."""
    poly = []
    for m in poly_markets:
        q = m.get("question") or m.get("title") or ""
        if not q:
            continue
        cat = categorize(q, m.get("slug") or "")
        if cat == "sports":
            continue
        poly.append((str(m.get("conditionId") or ""), cat, q))
    kal = []
    for m in kalshi_markets:
        text = _kalshi_text(m)
        tk = str(m.get("ticker") or "")
        if not text or _kalshi_is_combo(text, tk):
            continue
        kal.append((tk, text))
    return poly, kal


def format_discover(cands) -> str:
    L = ["=== Candidate Polymarket<->Kalshi pairs (REVIEW — not verified) ==="]
    if not cands:
        L.append("(no candidate matches — widen --poly-markets / --kalshi-limit "
                 "or lower --min-score)")
        return "\n".join(L)
    L.append(f"{len(cands)} candidate(s). For each: confirm BOTH resolve by the "
             "identical rule, then paste a verified entry into pairs.yaml.\n")
    for c in cands:
        L.append(f"[{c['score']:.2f}] PM : {c['poly_q'][:74]}")
        L.append(f"       KS : {c['kalshi_title'][:74]}  ({c['kalshi_ticker']})")
        if c.get("kalshi_rules"):
            L.append(f"       KS-rule: {c['kalshi_rules']}")
        L.append(f"       pm_cid: {c['poly_cid']}")
        L.append("")
    return "\n".join(L)


def discover_yaml(cands) -> str:
    """Paste-ready (commented) pairs.yaml block — uncomment after verifying."""
    out = ["# Candidate pairs from `crossarb --discover`. VERIFY identical "
           "resolution, then uncomment.", "pairs:"]
    for c in cands:
        out.append(f"  # score {c['score']:.2f}")
        out.append(f"  #   PM: {c['poly_q'][:80]}")
        out.append(f"  #   KS: {c['kalshi_title'][:80]}")
        out.append(f"  # - name: \"REVIEW: {c['poly_q'][:48]}\"")
        out.append(f"  #   polymarket_condition_id: \"{c['poly_cid']}\"")
        out.append("  #   polymarket_yes_index: 0")
        out.append(f"  #   kalshi_ticker: \"{c['kalshi_ticker']}\"")
        out.append("  #   invert: false")
    return "\n".join(out) + "\n"
