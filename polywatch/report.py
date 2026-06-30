"""Self-contained HTML / SVG report for a strategy sweep — no third-party deps.

Reads the summary CSV written by `polywatch sweep` (or a SweepResult directly)
and renders a single standalone HTML file (inline CSS + inline SVG charts; no
network calls, no JavaScript libraries) plus an optional standalone SVG suited to
embedding in a README. This is presentation only: it plots the numbers it is
given and invents nothing.
"""

from __future__ import annotations

import csv
import html
from dataclasses import dataclass

# Colour by economic meaning, not decoration: losing strategies red, winning
# green, skill-free baselines grey. Kept readable on a white card (GitHub light
# and dark both render the card as drawn).
_LOSS = "#c0392b"
_GAIN = "#1e8f5a"
_BASE = "#8a8f98"
_INK = "#1b2430"
_MUTED = "#5b6675"
_GRID = "#dde3ea"


@dataclass
class Row:
    strategy: str
    key: str
    is_baseline: bool
    trades: int
    net_roi: float
    t_stat: float
    win_rate: float = 0.0
    mean_alpha: float = 0.0
    dsr: str = ""
    verdict: str = ""


def load_summary_csv(path: str) -> list:
    """Parse the summary CSV produced by sweep.write_sweep_csv."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            rows.append(Row(
                strategy=d.get("strategy", ""), key=d.get("key", ""),
                is_baseline=str(d.get("is_baseline", "")).strip() in ("1", "true", "True"),
                trades=int(float(d.get("trades") or 0)),
                net_roi=float(d.get("net_roi") or 0.0),
                t_stat=float(d.get("t_stat") or 0.0),
                win_rate=float(d.get("win_rate") or 0.0),
                mean_alpha=float(d.get("mean_alpha") or 0.0),
                dsr=str(d.get("dsr") or ""), verdict=d.get("verdict", "")))
    return rows


def rows_from_sweep(res) -> list:
    """Adapt a SweepResult into report Rows (skips a CSV round-trip)."""
    from .sweep import row_verdict, _LABELS
    out = []
    for r in res.rows:
        s = r.stats
        out.append(Row(
            strategy=_LABELS.get(r.key, r.key), key=r.key, is_baseline=r.is_baseline,
            trades=s.n, net_roi=s.roi, t_stat=s.t_stat, win_rate=s.win_rate,
            mean_alpha=s.mean_alpha, dsr=("" if s.dsr is None else f"{s.dsr:.2f}"),
            verdict=row_verdict(r)))
    return out


def _color(row: Row) -> str:
    if row.is_baseline:
        return _BASE
    return _GAIN if row.net_roi > 0 else _LOSS


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def diverging_bar_svg(items, *, title, value_fmt="{:+.1%}", width=720,
                      row_h=34, pad_left=200, pad_right=70) -> str:
    """Horizontal diverging bar chart around a zero axis.

    `items` = list of (label, value, color). Returns a standalone <svg> string
    (its own white card background) so it renders on its own or inline in HTML.
    """
    n = len(items)
    top = 56
    height = top + n * row_h + 24
    plot_w = width - pad_left - pad_right
    vals = [v for _, v, _ in items] + [0.0]
    vmin, vmax = min(vals), max(vals)
    span = (vmax - vmin) or 1.0

    def x(v):
        return pad_left + (v - vmin) / span * plot_w

    zero_x = x(0.0)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">']
    out.append(f'<rect x="0" y="0" width="{width}" height="{height}" rx="12" fill="#ffffff"/>')
    out.append(f'<text x="20" y="30" font-size="16" font-weight="700" '
               f'fill="{_INK}">{_esc(title)}</text>')
    out.append(f'<line x1="{zero_x:.1f}" y1="{top-6}" x2="{zero_x:.1f}" '
               f'y2="{top + n*row_h:.1f}" stroke="{_GRID}" stroke-width="1"/>')
    for i, (label, value, color) in enumerate(items):
        cy = top + i * row_h + row_h / 2
        bx = x(value)
        x0, x1 = (zero_x, bx) if bx >= zero_x else (bx, zero_x)
        bw = max(1.0, x1 - x0)
        out.append(f'<text x="{pad_left-12}" y="{cy+4:.1f}" text-anchor="end" '
                   f'font-size="13" fill="{_INK}">{_esc(label)}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{cy-9:.1f}" width="{bw:.1f}" height="18" '
                   f'rx="3" fill="{color}"/>')
        vx = (bx + 6) if value >= 0 else (bx - 6)
        anchor = "start" if value >= 0 else "end"
        out.append(f'<text x="{vx:.1f}" y="{cy+4:.1f}" text-anchor="{anchor}" '
                   f'font-size="12" font-weight="600" fill="{_MUTED}">'
                   f'{_esc(value_fmt.format(value))}</text>')
    out.append("</svg>")
    return "\n".join(out)


def roi_svg(rows) -> str:
    items = [(r.strategy, r.net_roi, _color(r)) for r in rows]
    return diverging_bar_svg(items, title="Net ROI by strategy (out-of-sample)",
                             value_fmt="{:+.1%}")


def write_roi_svg(rows, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(roi_svg(rows))


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

def _verdict_pill(v: str) -> str:
    color = {"PASS": _GAIN, "FAIL": _LOSS}.get(v, _BASE)
    return (f'<span style="background:{color};color:#fff;border-radius:999px;'
            f'padding:2px 10px;font-size:12px;font-weight:600">{_esc(v)}</span>')


def build_html(rows, meta=None, path="report.html") -> None:
    meta = meta or {}
    total_trades = sum(r.trades for r in rows)
    thesis = [r for r in rows if not r.is_baseline]
    any_pass = any(r.verdict == "PASS" for r in thesis)
    headline = ("An out-of-sample edge survives costs."
                if any_pass else
                "No strategy beats its costs out-of-sample.")

    def stat_card(label, value):
        return (f'<div class="card"><div class="cl">{_esc(label)}</div>'
                f'<div class="cv">{_esc(value)}</div></div>')

    cards = "".join([
        stat_card("OOS trades analysed", f"{total_trades:,}"),
        stat_card("Strategies compared", str(len(rows))),
        stat_card("Wallets tested", str(meta.get("n_trials", "-"))),
        stat_card("Verdict", "PASS" if any_pass else "FAIL"),
    ])

    body_rows = ""
    for r in rows:
        roi_c = _GAIN if r.net_roi > 0 else (_BASE if r.is_baseline and r.net_roi == 0 else _LOSS)
        body_rows += (
            f'<tr><td class="name">{_esc(r.strategy)}</td>'
            f'<td>{r.trades:,}</td>'
            f'<td>{r.win_rate:.0%}</td>'
            f'<td style="color:{roi_c};font-weight:600">{r.net_roi:+.1%}</td>'
            f'<td>{r.mean_alpha:+.3f}</td>'
            f'<td>{r.t_stat:+.2f}</td>'
            f'<td>{_esc(r.dsr or "n/a")}</td>'
            f'<td>{_verdict_pill(r.verdict)}</td></tr>')

    meta_line = ""
    if meta.get("cutoffs"):
        meta_line = (f'Cutoffs {_esc(meta.get("cutoffs"))} &nbsp;|&nbsp; '
                     f'slippage {meta.get("slippage", 0):.0%} pts &nbsp;|&nbsp; '
                     f'stake ${meta.get("stake", 0):,.0f}')

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polywatch — strategy sweep</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin:0; background:#f4f6f9; color:{_INK};
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:880px; margin:0 auto; padding:32px 20px 56px; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  .sub {{ color:{_MUTED}; font-size:14px; margin:0 0 8px; }}
  .headline {{ font-size:16px; font-weight:600; margin:14px 0 22px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:22px; }}
  .card {{ background:#fff; border:1px solid {_GRID}; border-radius:12px; padding:14px 16px; }}
  .cl {{ color:{_MUTED}; font-size:12px; }}
  .cv {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .panel {{ background:#fff; border:1px solid {_GRID}; border-radius:14px;
            padding:8px 10px; margin-bottom:22px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff;
           border:1px solid {_GRID}; border-radius:14px; overflow:hidden; }}
  th,td {{ text-align:right; padding:10px 14px; font-size:14px;
           border-bottom:1px solid {_GRID}; }}
  th {{ color:{_MUTED}; font-weight:600; font-size:12px; text-transform:uppercase;
        letter-spacing:.04em; background:#fbfcfe; }}
  td.name, th.name {{ text-align:left; }}
  tr:last-child td {{ border-bottom:none; }}
  .foot {{ color:{_MUTED}; font-size:12px; margin-top:18px; line-height:1.5; }}
  @media (max-width:640px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head>
<body><div class="wrap">
  <h1>Polywatch — strategy sweep</h1>
  <p class="sub">{meta_line or "Out-of-sample, cost-aware comparison of copy-trade strategies and skill-free baselines."}</p>
  <p class="headline">{_esc(headline)}</p>
  <div class="cards">{cards}</div>
  <div class="panel">{roi_svg(rows)}</div>
  <table>
    <thead><tr><th class="name">Strategy</th><th>Trades</th><th>Win%</th>
      <th>Net ROI</th><th>mean &#945;</th><th>t-stat</th><th>DSR</th><th>Verdict</th></tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
  <p class="foot">&#945; = outcome(1/0) &minus; entry price. DSR = Deflated Sharpe Ratio
  (the multiple-testing haircut; meaningful for the wallet-selected row, which searched
  over many candidates). Baselines are skill-free reference points, not strategies under
  test. Generated by <code>python -m polywatch report</code> from a sweep; no figures
  are synthetic.</p>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
