"""CR-DW Step 0 Q1 — reproduce the chart's yellow / darkest GEX levels; strike-space version.

Read-only. 2026 train sessions only (B4 stop: no 2024 / 2025 reads until Ryan approves).
"chart"  = the chart's own code (Ironbeam callbacks, Levels: Corrected, gross >= 40 $bn).
"repro"  = independent re-implementation of the same rule (corrected ES space) -> must equal chart.
"strike" = same rule with the level key = strike (kickoff: strike space), band and p95 as the chart.
"""
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "apps/web")]
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from packages.shared.level_placement import place_levels  # noqa: E402
from packages.shared.trading_calendar import prev_trading_day  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
GROSS_MIN = 40e9
HOLDOUT_START = dt.date(2026, 6, 6)


def es_bars(d):
    s, e = cb._session_window_pt(d, PT)
    sql = text(f"SELECT datetime, open, high, low, close FROM {cb.DB_TABLE_NAME} "
               "WHERE datetime >= :s AND datetime <= :e ORDER BY datetime")
    with cb.engine.connect() as con:
        return pd.read_sql(sql, con, params={"s": s.astimezone(ZoneInfo('UTC')).replace(tzinfo=None),
                                              "e": e.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)})


def select_large(df, band_min, band_max):
    """Chart rule: band, gross >= 40 $bn, greedy 5-pt spacing by gross (max 80), p95 |net| span."""
    df = df[(df.level >= band_min) & (df.level <= band_max)].copy()
    df["mag"] = df.call_gamma.abs() + df.put_gamma.abs()
    df = df[df.mag >= GROSS_MIN].sort_values("mag", ascending=False)
    kept = []
    for _, r in df.iterrows():
        if all(abs(r.level - k.level) >= cb.GEX_MIN_LEVEL_SPACING for k in kept):
            kept.append(r)
        if len(kept) >= cb.GEX_MAX_LEVELS_PER_DAY:
            break
    sel = pd.DataFrame(kept)
    if sel.empty:
        return sel, np.nan
    p95 = float(np.nanpercentile(np.abs(sel.net_gamma), cb.GEX_COLOR_PERCENTILE))
    sel["large"] = np.where(sel.net_gamma >= p95, "yellow", np.where(sel.net_gamma <= -p95, "darkest", ""))
    return sel.sort_values("level"), p95


def aggregate(rows, key):
    g = rows.groupby(key).agg(c=("gex_call", "sum"), p=("gex_put", "sum")).reset_index().rename(columns={key: "level"})
    g["call_gamma"] = g.c.astype(float)
    g["put_gamma"] = -g.p.abs().astype(float)
    g["net_gamma"] = g.call_gamma + g.put_gamma
    return g[["level", "call_gamma", "put_gamma", "net_gamma"]]


def run_day(d):
    assert d < HOLDOUT_START and d >= dt.date(2026, 1, 1), d
    bars = es_bars(d)
    tables = cb._corrected_gex_tables(d)
    chart = cb._build_gex_segments_for_session(d, bars, PT, gex_min_abs_b=40, gex_tables=tables[:2])
    net =np.array([s["net_gamma"] for s in chart])
    cmin, cmax = cb._compute_color_span(net)
    chart_large = [(s["level"], "yellow" if s["net_gamma"] >= cmax else "darkest")
                   for s in chart if s["net_gamma"] >= cmax or s["net_gamma"] <= cmin]

    rows = cb._fetch_gex_rows_for_placement(d)
    for c in ("gex_call", "gex_put", "strike", "stock_price"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows = rows.fillna({"gex_call": 0.0, "gex_put": 0.0})
    p_es = cb._fetch_es_snapshot_close(prev_trading_day(d))
    placed = place_levels(rows, p_es)
    placed["lvl_es"] = np.floor(placed["level_corrected"] + 0.5)  # bucket 1, half away from zero (>0)
    band_min, band_max = float(bars.low.min()) - cb.GEX_LEVEL_PADDING, float(bars.high.max()) + cb.GEX_LEVEL_PADDING
    repro, p95_r = select_large(aggregate(placed.dropna(subset=["lvl_es"]), "lvl_es"), band_min, band_max)
    repro_large = [(float(r.level), r.large) for _, r in repro.iterrows() if r.large]

    # strike space; S0 = stock_price of the nearest live expiry (CR-BU's cash anchor)
    s0 = float(rows.sort_values("expir_date").stock_price.iloc[0])
    basis = p_es - s0
    strike, p95_s = select_large(aggregate(rows, "strike"), band_min - basis, band_max - basis)
    # cash-space corrected peak per large strike: gross-weighted mean of K * S0 / F_e over its expiries
    rows["cash_peak"] = rows.strike * s0 / rows.stock_price
    rows["gross"] = rows.gex_call.abs() + rows.gex_put.abs()
    out_strike = []
    for _, r in strike.iterrows():
        if not r.large:
            continue
        sub = rows[rows.strike == r.level]
        cp = float((sub.cash_peak * sub.gross).sum() / sub.gross.sum())
        dte0 = float(sub[sub.dte == sub.dte.min()].gross.sum() / sub.gross.sum())
        out_strike.append(dict(strike=float(r.level), kind=r.large, net_bn=round(r.net_gamma / 1e9, 1),
                               gross_bn=round((abs(r.call_gamma) + abs(r.put_gamma)) / 1e9, 1),
                               cash_peak_minus_strike=round(cp - r.level, 1),
                               strike_plus_basis_es=round(r.level + basis, 1), share_nearest_expiry=round(dte0, 2)))
    return dict(
        date=d.isoformat(), n_chart_lines=len(chart), chart_p95_bn=round(cmax / 1e9, 1),
        chart_large=[(lv, k) for lv, k in sorted(chart_large)],
        repro_large=sorted(repro_large), repro_equals_chart=sorted(chart_large) == sorted(repro_large),
        repro_p95_bn=round(p95_r / 1e9, 1),
        p_es_1545=p_es, s0_nearest_expiry=s0, basis_es_minus_s0=round(basis, 2),
        expiries=int(rows.expir_date.nunique()), dte_max=int(rows.dte.max()),
        n_strike_lines=len(strike), strike_p95_bn=round(p95_s / 1e9, 1), strike_large=out_strike,
    )


if __name__ == "__main__":
    days = [dt.date.fromisoformat(a) for a in sys.argv[1:]]
    res = [run_day(d) for d in days]
    (ROOT / "scripts/logs/cr_dw/q1_examples.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
