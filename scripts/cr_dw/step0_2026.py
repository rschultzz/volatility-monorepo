"""CR-DW Step 0 Q1 (band / placement sweep), Q2 and Q3 on 2026 train sessions only.

B4 guard: dates are asserted in 2026-01-01 .. 2026-06-05 (no R-a / R-b / holdout reads until Ryan approves).
No outcome is read: settlement_prints is called only for the set of dates that HAVE a settlement (early-close
exclusion); its values are discarded. Q3's fairness check uses synthetic settles (open + IM * z), not real ones.
"""
import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "apps/web"), str(ROOT / "scripts/cr_dw")]
from q1_levels import GROSS_MIN, PT, aggregate, cb, select_large  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from packages.shared import spx_cash  # noqa: E402
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.level_placement import place_levels  # noqa: E402
from packages.shared.trading_calendar import prev_trading_day  # noqa: E402

D0, D1 = dt.date(2026, 1, 1), dt.date(2026, 6, 5)
UTC = ZoneInfo("UTC")
OUT = ROOT / "scripts/logs/cr_dw"


def third_friday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21


def load_universe():
    sql = text("SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features "
               "WHERE ticker='SPX' AND feature_version=:v AND active AND trade_date BETWEEN :a AND :b ORDER BY 1")
    with cb.engine.connect() as con:
        f = pd.read_sql(sql, con, params={"v": CANONICAL_FEATURE_VERSION, "a": D0, "b": D1})
    f["trade_date"] = pd.to_datetime(f.trade_date).dt.date
    raw = cb.engine.raw_connection()
    try:
        clean, _ = spx_cash.build_series(spx_cash.fetch_minutes(raw, D0, D1))
    finally:
        raw.close()
    opens = clean.sort_values("minute").groupby("session_date").agg(open=("px", "first"), open_min=("minute", "first"))
    has_settle = set(spx_cash.settlement_prints(clean).index)  # membership only; values not used
    f = f.merge(opens, left_on="trade_date", right_index=True, how="left")
    f["third_fri"] = f.trade_date.map(third_friday)
    f["has_settle"] = f.trade_date.isin(has_settle)
    assert f.trade_date.min() >= D0 and f.trade_date.max() <= D1
    return f


def es_window(d, end_pt):
    s, _ = cb._session_window_pt(d, PT)
    sql = text(f"SELECT high, low FROM {cb.DB_TABLE_NAME} WHERE datetime >= :s AND datetime < :e")
    with cb.engine.connect() as con:
        return pd.read_sql(sql, con, params={"s": s.astimezone(UTC).replace(tzinfo=None),
                                              "e": end_pt.astimezone(UTC).replace(tzinfo=None)})


def day_levels(d):
    rows = cb._fetch_gex_rows_for_placement(d)
    for c in ("gex_call", "gex_put", "strike", "stock_price"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows = rows.fillna({"gex_call": 0.0, "gex_put": 0.0}).dropna(subset=["strike", "stock_price"])
    p_es = cb._fetch_es_snapshot_close(prev_trading_day(d))
    s0 = float(rows.sort_values("expir_date").stock_price.iloc[0])
    pre = es_window(d, dt.datetime.combine(d, dt.time(6, 30), tzinfo=PT))
    full = es_window(d, dt.datetime.combine(d, dt.time(13, 1), tzinfo=PT))
    basis = (p_es - s0) if p_es else np.nan
    bands = {"pre": (pre.low.min() - 150 - basis, pre.high.max() + 150 - basis),
             "full": (full.low.min() - 150 - basis, full.high.max() + 150 - basis)}
    cashc = place_levels(rows, s0)  # CR-BU: SPX cash placement, P_target = S0
    cashc["lvl"] = np.floor(cashc["level_corrected"] + 0.5)
    tables = {"cash_corrected": aggregate(cashc.dropna(subset=["lvl"]), "lvl"), "strike": aggregate(rows, "strike")}
    out = {"s0": s0, "basis": basis}
    for pl, tab in tables.items():
        for bn, (lo, hi) in bands.items():
            sel, p95 = select_large(tab, lo, hi)
            big = sel[sel.large != ""] if not sel.empty else sel
            out[f"{pl}|{bn}"] = [(float(r.level), r.large) for _, r in big.iterrows()]
    return out


def nearest(levels, x):
    a = np.asarray(levels, float)
    return np.min(np.abs(a[None, :] - np.asarray(x, float)[:, None]), axis=1)


def fairness(days, rng, n_world=300, n_draw=1000):
    """No-pin worlds: settle* = open + IM*z, z ~ N(0, 0.75), independent of levels. Returns p-values of the
    primary statistic under null (a) uniform shift and null (c) move-swap, and (b)'s no-pin fraction."""
    pa, pc, fb = [], [], []
    ims = np.array([x["im"] for x in days])
    for _ in range(n_world):
        z = rng.normal(0, 0.75, len(days))
        obs = np.median([nearest(x["lv"], [x["open"] + x["im"] * zi])[0] / x["im"] for x, zi in zip(days, z)])
        # (a) shift whole set by U(-IM, IM)
        null_a = np.empty((n_draw, len(days)))
        for j, (x, zi) in enumerate(zip(days, z)):
            sh = rng.uniform(-x["im"], x["im"], n_draw)
            lv = np.asarray(x["lv"])[None, :] + sh[:, None]
            null_a[:, j] = np.min(np.abs(lv - (x["open"] + x["im"] * zi)), axis=1) / x["im"]
        pa.append((np.sum(np.median(null_a, axis=1) <= obs) + 1) / (n_draw + 1))
        # (c) move-swap: settle** = open + IM * z_k, z_k drawn from the pool of all days' moves
        null_c = np.empty((n_draw, len(days)))
        for j, x in enumerate(days):
            zk = rng.choice(z, n_draw)
            null_c[:, j] = nearest(x["lv"], x["open"] + x["im"] * zk) / x["im"]
        pc.append((np.sum(np.median(null_c, axis=1) <= obs) + 1) / (n_draw + 1))
        fb.append(np.mean([nearest(x["lv"], [x["open"] + x["im"] * zi])[0] < nearest(x["lv"], [x["open"]])[0]
                           for x, zi in zip(days, z)]))
    pa, pc = np.array(pa), np.array(pc)
    return {"n_worlds": n_world, "n_days": len(days),
            "null_a_reject_rate_p<0.05": float(np.mean(pa < 0.05)), "null_a_median_p": float(np.median(pa)),
            "null_c_reject_rate_p<0.05": float(np.mean(pc < 0.05)), "null_c_median_p": float(np.median(pc)),
            "b_fraction_no_pin_median": float(np.median(fb)), "b_fraction_no_pin_p05_p95": [float(np.percentile(fb, 5)), float(np.percentile(fb, 95))]}


if __name__ == "__main__":
    u = load_universe()
    excl = {"third_friday_am": u[u.third_fri].trade_date.astype(str).tolist(),
            "no_settlement_print": u[~u.has_settle & ~u.third_fri].trade_date.astype(str).tolist(),
            "no_im": u[u.im.isna() | (u.im <= 0)].trade_date.astype(str).tolist(),
            "no_open": u[u.open.isna()].trade_date.astype(str).tolist()}
    keep = u[~u.third_fri & u.has_settle & (u.im > 0) & u.open.notna()].copy()
    recs = []
    for _, r in keep.iterrows():
        lv = day_levels(r.trade_date)
        recs.append({"date": r.trade_date.isoformat(), "im": r.im, "open": r.open, **lv})
    (OUT / "step0_2026_days.json").write_text(json.dumps(recs, indent=0, default=str))

    variants = [k for k in recs[0] if "|" in k]
    q2 = {}
    for v in variants:
        n_lv = [len(x[v]) for x in recs]
        cond = [x for x in recs if x[v] and min(abs(l - x["open"]) for l, _ in x[v]) <= 0.5 * x["im"]]
        n_in = [sum(abs(l - x["open"]) <= 0.5 * x["im"] for l, _ in x[v]) for x in cond]
        n_in1 = [sum(abs(l - x["open"]) <= 1.0 * x["im"] for l, _ in x[v]) for x in recs]
        q2[v] = {"days": len(recs), "condition_days": len(cond),
                 "large_levels_per_day_dist": pd.Series(n_lv).value_counts().sort_index().to_dict(),
                 "large_within_0.5IM_on_cond_days_dist": pd.Series(n_in).value_counts().sort_index().to_dict(),
                 "large_within_1IM_all_days_dist": pd.Series(n_in1).value_counts().sort_index().to_dict(),
                 "cond_days_by_kind": {"yellow_only": sum(all(k == "yellow" for l, k in x[v] if abs(l - x["open"]) <= .5 * x["im"]) for x in cond),
                                       "darkest_only": sum(all(k == "darkest" for l, k in x[v] if abs(l - x["open"]) <= .5 * x["im"]) for x in cond)}}
    same = {pl: sum(sorted(x[f"{pl}|pre"]) == sorted(x[f"{pl}|full"]) for x in recs) for pl in ("cash_corrected", "strike")}
    ims = pd.Series([x["im"] for x in recs])
    rng = np.random.default_rng(20261031)
    q3 = {}
    for v in ("cash_corrected|pre", "strike|pre"):
        days = [{"im": x["im"], "open": x["open"], "lv": [l for l, _ in x[v]]} for x in recs
                if x[v] and min(abs(l - x["open"]) for l, _ in x[v]) <= 0.5 * x["im"]]
        q3[v] = fairness(days, rng)
    res = {"universe": {"rows": len(u), "kept": len(recs), "excluded": excl,
                        "im_pts_p10_p50_p90": [float(ims.quantile(q)) for q in (.1, .5, .9)]},
           "band_pre_equals_full_days": same, "q2": q2, "q3_fairness": q3}
    (OUT / "step0_2026.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
