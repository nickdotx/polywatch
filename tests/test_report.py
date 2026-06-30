"""Offline tests for the HTML/SVG report renderer (no network, no deps)."""

import xml.dom.minidom as minidom

from polywatch import report
from polywatch.report import Row


def _rows():
    return [
        Row("Copy everything", "copy_all", False, 15763, -0.041, -4.46, 0.50, -0.02, "n/a", "FAIL"),
        Row("Scorer-selected (top-k)", "selected", False, 2248, -0.035, -1.37, 0.48, -0.01, "0.00", "FAIL"),
        Row("Favourite-longshot", "favorite", True, 8921, -0.019, -2.81, 0.70, -0.30, "n/a", "baseline"),
        Row("Random selection", "random", True, 1387, -0.056, -1.99, 0.40, -0.05, "n/a", "baseline"),
    ]


def test_roi_svg_is_wellformed_xml():
    svg = report.roi_svg(_rows())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    minidom.parseString(svg)                         # raises on malformed XML
    assert "Net ROI by strategy" in svg


def test_build_html_is_self_contained(tmp_path):
    p = tmp_path / "report.html"
    report.build_html(_rows(), {"cutoffs": [1, 2, 3], "n_trials": 803,
                                "slippage": 0.01, "stake": 15}, str(p))
    txt = p.read_text(encoding="utf-8")
    assert "<svg" in txt and "Copy everything" in txt and "803" in txt
    # no external/network resources beyond the SVG namespace URI
    assert "https://" not in txt
    assert "http://" not in txt.replace("http://www.w3.org/2000/svg", "")


def test_load_summary_csv_roundtrip(tmp_path):
    csvp = tmp_path / "results.csv"
    csvp.write_text(
        "strategy,key,is_baseline,trades,win_rate,net_pnl,net_roi,mean_ret,sharpe,"
        "t_stat,mean_alpha,alpha_t,psr,dsr,verdict\n"
        "Copy everything,copy_all,0,15763,0.5,-1000,-0.041,-0.001,-0.03,-4.46,-0.02,-3.1,0.1,,FAIL\n"
        "Favourite-longshot,favorite,1,8921,0.7,-500,-0.019,-0.0,-0.01,-2.81,-0.30,-2.0,0.2,,baseline\n",
        encoding="utf-8")
    rows = report.load_summary_csv(str(csvp))
    assert len(rows) == 2
    assert rows[0].strategy == "Copy everything" and rows[0].is_baseline is False
    assert rows[1].is_baseline is True


def test_rows_from_sweep_and_write_roi_svg(tmp_path):
    from polywatch import sweep
    from polywatch.walkforward import _strategy_stats
    recs = [{"net": 5.0, "won": True, "alpha": 0.1, "cat": "politics", "price": 0.4},
            {"net": -3.0, "won": False, "alpha": -0.2, "cat": "politics", "price": 0.6}]
    st = _strategy_stats("x", recs, stake=15.0)
    res = sweep.SweepResult(
        rows=[sweep.StrategyRow("selected", False, st),
              sweep.StrategyRow("random", True, st)],
        report=None, params={})
    rows = report.rows_from_sweep(res)
    assert [r.key for r in rows] == ["selected", "random"]
    assert rows[1].is_baseline and not rows[0].is_baseline
    p = tmp_path / "roi.svg"
    report.write_roi_svg(rows, str(p))
    txt = p.read_text(encoding="utf-8")
    assert txt.startswith("<svg") and "</svg>" in txt.rstrip()
