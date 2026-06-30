"""Lightweight market categorizer (word-boundary keyword matching).

Polymarket's API has no clean category field, so we infer one from the title.
Used to (a) filter which markets we copy and (b) break P&L down by category so
we can see WHERE the edge lives. Matching is word-boundary based (regex \\b) so
short tokens don't false-positive inside other words (e.g. "poll" must not match
"Apollo", "house" must not match "Full House", "match" must not match "rematch").

Categories: crypto | sports | politics | econ | pop | other
"""

from __future__ import annotations

import re

# Checked in priority order; first hit wins.
_RULES = [
    ("crypto", (
        "up or down", "updown", "bitcoin", "ethereum", "solana", "dogecoin",
        "btc", "eth", "sol", "xrp", "bnb", "hyperliquid", "hype",
        "price above", "price below", "crypto", "all-time high",
    )),
    ("sports", (
        "vs", "fifa", "world cup", "wimbledon", "atp", "wta", "nba", "nfl",
        "mlb", "nhl", "premier league", "la liga", "serie a", "bundesliga",
        "champions league", "ucl", "uefa", "exact score", "corners",
        "both teams to score", "o/u", "spread", "win on", "reach the semifinal",
        "reach the final", "top goalscorer", "golden boot", "super bowl",
        "playoffs", "grand prix", "f1", "tennis", "match", "relegated",
    )),
    ("econ", (
        "fed", "rate cut", "rate hike", "cpi", "inflation", "gdp", "recession",
        "interest rate", "unemployment", "jobs report", "jerome powell", "fomc",
    )),
    ("politics", (
        "president", "election", "nomination", "senate", "congress", "governor",
        "prime minister", "impeach", "democratic", "republican", "parliament",
        "resign", "putin", "trump", "xi jinping", "supreme court", "diplomatic",
        "ceasefire", "war", "sanction", "nominee", "cabinet",
    )),
    ("pop", (
        "album", "movie", "box office", "oscar", "grammy", "rotten tomatoes",
        "spotify", "tweet", "elon musk", "taylor swift", "netflix", "tiktok",
        "billboard",
    )),
]

# Pre-compile a word-boundary regex per (category, keyword).
_COMPILED = [
    (name, [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in kws])
    for name, kws in _RULES
]


def categorize(title: str, slug: str = "") -> str:
    text = f"{(title or '').lower()} {(slug or '').lower()}"
    for name, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(text):
                return name
    return "other"


def passes_category_filter(category: str, allow=None, block=None) -> bool:
    """True if `category` is permitted. allow (if non-empty) is a whitelist;
    block is a blacklist. allow takes precedence."""
    if allow:
        return category in allow
    if block:
        return category not in block
    return True
