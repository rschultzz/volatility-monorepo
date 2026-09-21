#!/usr/bin/env python3
"""CR-BH — diff v0.6.0-openiv (ES) against v0.6.0-openiv-spxcash (SPX cash). Read-only.

Reads both versions from bt_daily_outcomes, the ES control recompute and the SPX minute
series from scripts/.cache/cr_bh_rows.pkl (written by cr_bh_shadow_spxcash_outcomes.py),
and writes scripts/logs/cr_bh_diff.md. Every table is reported twice (spec A1.6):
full corpus, and the 'stock_price' segment only (trade_date >= 2023-11-09).

ES appears here only as (a) the basis used for bucketing — median over 06:58–07:02 PT of
es_minutes.close − SPX series — and (b) a per-day QA flag. It never alters an SPX price.

Usage:
    apps/web/.venv/bin/python scripts/cr_bh_diff.py
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from scripts.cr_bh_spx_cash import CUTOVER, is_lagged, series_segment  # noqa: E402

TICKER = "SPX"
SHADOW_VERSION = f"{CANONICAL_FEATURE_VERSION}-spxcash"
ROWS_CACHE = REPO_ROOT / "scripts" / ".cache" / "cr_bh_rows.pkl"
OUT = REPO_ROOT / "scripts" / "logs" / "cr_bh_diff.md"

BASIS_MIN, BASIS_MAX = -50.0, 120.0
BUCKETS = ("<20", "20-35", "35+")
QA_SPREAD, QA_CLOSE, QA_HILO = 15.0, 10.0, 10.0

_COLS = ("trade_date, outcome_status, regime_kind_at_classification, horizon_sessions, horizon_end_date, "
         "reached_touch, reached_close, days_to_reach, final_close_distance_from_target, "
         "session_open_t0, session_high_t0, session_low_t0, session_close_t0")


def _version(conn, version: str) -> pd.DataFrame:
    rows = conn.execute(
        f"SELECT {_COLS} FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s AND active ORDER BY 1",
        (TICKER, version)).fetchall()
    return pd.DataFrame(rows, columns=[c.strip() for c in _COLS.split(",")]).set_index("trade_date")


def bucket(b: float) -> str:
    if b is None or np.isnan(b) or b < BASIS_MIN or b > BASIS_MAX:
        return "no_basis"
    return "<20" if b < 20 else ("20-35" if b < 35 else "35+")


def two_prop_p(k1, n1, k2, n2) -> tuple[float, float]:
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k2 / n2 - k1 / n1) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) or v is None else str(v)
                                        for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def confusion(j: pd.DataFrame, col: str) -> dict:
    o, n = j[f"{col}_old"].astype(bool), j[f"{col}_new"].astype(bool)
    return {"n": len(j), "T→T": int((o & n).sum()), "T→F": int((o & ~n).sum()),
            "F→T": int((~o & n).sum()), "F→F": int((~o & ~n).sum()),
            "old %": round(100 * o.mean(), 1) if len(j) else np.nan,
            "new %": round(100 * n.mean(), 1) if len(j) else np.nan}


def main() -> None:
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    cache = pd.read_pickle(ROWS_CACHE)
    old, new = _version(conn, CANONICAL_FEATURE_VERSION), _version(conn, SHADOW_VERSION)
    ctl = cache["control"].set_index("trade_date")
    aux = cache["new"].set_index("trade_date")[["drift_target", "implied_move_1d"]]
    spx_daily, minutes = cache["spx_daily"], cache["spx_minutes"]
    assert len(new) and set(new.index) == set(old.index), "shadow version missing or date sets differ"

    # ── basis per day: median(es_minutes.close − SPX) over 06:58–07:02 PT ────
    es7 = pd.DataFrame(conn.execute(
        "SELECT ts_pt, close FROM es_minutes WHERE ts_pt::time BETWEEN '06:58' AND '07:02' AND trade_date >= %s",
        (min(old.index),)).fetchall(), columns=["minute", "es"])
    es7["minute"] = pd.to_datetime(es7["minute"])
    b = minutes.merge(es7, on="minute")
    b["basis"] = pd.to_numeric(b["es"]) - b["px"]
    basis = b.groupby("session_date")["basis"].median()

    # ── ES QA flag per day (flag only) ───────────────────────────────────────
    es1 = pd.DataFrame(conn.execute(
        """SELECT (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles') AS pt, high, low, close
           FROM ironbeam_es_1m_bars WHERE datetime >= %s""", (dt.datetime.combine(min(old.index), dt.time(0)),)
    ).fetchall(), columns=["minute", "high", "low", "close"])
    es1["minute"] = pd.to_datetime(es1["minute"])
    es1 = es1[(es1["minute"].dt.time >= dt.time(6, 33)) & (es1["minute"].dt.time <= dt.time(13, 0))]
    es_close_at = es1.assign(minute=es1["minute"] + pd.Timedelta(minutes=1))[["minute", "close"]]  # stamp at bar close
    q = minutes.merge(es_close_at, on="minute")
    q["basis"] = q["close"] - q["px"]
    qa_rows = []
    es_hl = es1.assign(d=es1["minute"].dt.date).groupby("d").agg(es_high=("high", "max"), es_low=("low", "min"))
    for d, g in q.groupby("session_date"):
        body = g[g["minute"].dt.time >= dt.time(6, 46)]["basis"]
        if len(body) < 30 or d not in spx_daily.index:
            continue
        med = body.median()
        spread = body.quantile(.95) - body.quantile(.05)
        close_dev = g.sort_values("minute")["basis"].iloc[-3:].median() - med
        hi_dev = (es_hl.loc[d, "es_high"] - med) - spx_daily.loc[d, "high"] if d in es_hl.index else np.nan
        lo_dev = (es_hl.loc[d, "es_low"] - med) - spx_daily.loc[d, "low"] if d in es_hl.index else np.nan
        why = []
        if spread > QA_SPREAD: why.append(f"spread {spread:.0f}")
        if abs(close_dev) > QA_CLOSE: why.append(f"close {close_dev:+.0f}")
        if abs(hi_dev) > QA_HILO: why.append(f"high {hi_dev:+.0f}")
        if abs(lo_dev) > QA_HILO: why.append(f"low {lo_dev:+.0f}")
        if why:
            qa_rows.append({"date": d, "segment": series_segment(d), "lagged": is_lagged(d),
                            "n_min": int(spx_daily.loc[d, "n_minutes"]), "first": spx_daily.loc[d, "first"],
                            "flags": "; ".join(why)})
    qa = pd.DataFrame(qa_rows)
    qa_days = set(qa["date"]) if len(qa) else set()
    lag_close = q[q["session_date"].map(is_lagged)].sort_values("minute").groupby("session_date").agg(
        close_basis=("basis", "last"))
    lag_med = q[q["session_date"].map(is_lagged) & (q["minute"].dt.time >= dt.time(6, 46))
                & (q["minute"].dt.time <= dt.time(12, 45))].groupby("session_date")["basis"].median()
    lag_err = (lag_close["close_basis"] - lag_med).dropna()

    # ── joined frame ─────────────────────────────────────────────────────────
    j = old.join(new, lsuffix="_old", rsuffix="_new").join(ctl.add_suffix("_ctl")).join(aux)
    j["basis"] = basis.reindex(j.index)
    j["bucket"] = j["basis"].map(bucket)
    j["series_segment"] = [series_segment(d) for d in j.index]
    j["lagged_day"] = [is_lagged(d) for d in j.index]
    j["horizon_crosses_cutover"] = [(d < CUTOVER) and (e is not None and not pd.isna(e) and e >= CUTOVER)
                                    for d, e in zip(j.index, j["horizon_end_date_new"])]
    j["qa_flag_t0"] = [d in qa_days for d in j.index]
    j["regime"] = j["regime_kind_at_classification_new"]
    j["horizon"] = j["horizon_sessions_new"]
    j.to_csv(OUT.with_name("cr_bh_diff_rows.csv"))

    out: list[str] = [f"# CR-BH diff — `{CANONICAL_FEATURE_VERSION}` (ES) vs `{SHADOW_VERSION}` (SPX cash)\n",
                      f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Rows: {len(j)} active. "
                      f"Segments: spot_shifted_2023 {int((j.series_segment == 'spot_shifted_2023').sum())}, "
                      f"stock_price {int((j.series_segment == 'stock_price').sum())}.\n"]

    # control
    both_ctl = j[(j.outcome_status_old == "computed") & (j.outcome_status_ctl == "computed")]
    ctl_mis = j[(j.outcome_status_old != j.outcome_status_ctl)
                | ((j.outcome_status_old == "computed") & ((j.reached_touch_old != j.reached_touch_ctl)
                                                            | (j.reached_close_old != j.reached_close_ctl)))]
    out.append("## 0. Control — stored canonical rows vs ES recompute through the same path today\n")
    out.append(f"{len(both_ctl)} rows computed in both; **{len(ctl_mis)} rows differ** in status / touch / close "
               "(differences here are not caused by the price series).\n")
    if len(ctl_mis):
        out.append(md_table(ctl_mis.reset_index()[["trade_date", "regime", "horizon", "outcome_status_old",
                                                   "outcome_status_ctl", "reached_touch_old", "reached_touch_ctl",
                                                   "reached_close_old", "reached_close_ctl"]]))

    for scope_name, s in (("Full corpus", j), ("`stock_price` segment only (trade_date ≥ 2023-11-09)",
                                                j[j.series_segment == "stock_price"])):
        out.append(f"\n## {scope_name} — {len(s)} rows\n")

        out.append("### Basis buckets (median 06:58–07:02 PT, `es_minutes.close` − SPX series)\n")
        bb = s.groupby("bucket").agg(days=("basis", "size"), mean_basis=("basis", "mean")).round(1).reset_index()
        out.append(md_table(bb))

        out.append("### Status transitions (old → new)\n")
        st = pd.crosstab(s.outcome_status_old, s.outcome_status_new).reset_index()
        out.append(md_table(st))
        moved = s[s.outcome_status_old != s.outcome_status_new]
        if len(moved):
            out.append(md_table(moved.reset_index()[["trade_date", "regime", "outcome_status_old", "outcome_status_new",
                                                     "horizon_end_date_old", "horizon_end_date_new"]]))
        hed = s[(s.outcome_status_old == "computed") & (s.outcome_status_new == "computed")
                & (s.horizon_end_date_old != s.horizon_end_date_new)]
        out.append(f"Rows computed in both whose `horizon_end_date` moved (ES counts market-holiday Globex "
                   f"sessions, SPX does not): **{len(hed)}**.\n")

        c = s[(s.outcome_status_old == "computed") & (s.outcome_status_new == "computed")]
        out.append(f"### Confusion matrices — {len(c)} rows computed in both (old → new)\n")
        rows = []
        for col in ("reached_touch", "reached_close"):
            rows.append({"label": col, "regime": "ALL", "horizon": "", "bucket": "", **confusion(c, col)})
            for (rg, hz), g in c.groupby(["regime", "horizon"]):
                rows.append({"label": col, "regime": rg, "horizon": int(hz), "bucket": "ALL", **confusion(g, col)})
                for bk in BUCKETS + ("no_basis",):
                    gb = g[g.bucket == bk]
                    if len(gb):
                        rows.append({"label": col, "regime": rg, "horizon": int(hz), "bucket": bk, **confusion(gb, col)})
        out.append(md_table(pd.DataFrame(rows)))

        out.append("### Pooled rates old vs new, by regime × horizon\n")
        rows = []
        for (rg, hz), g in c.groupby(["regime", "horizon"]):
            rows.append({"regime": rg, "horizon": int(hz), "n": len(g),
                         "touch old %": round(100 * g.reached_touch_old.astype(bool).mean(), 1),
                         "touch new %": round(100 * g.reached_touch_new.astype(bool).mean(), 1),
                         "close old %": round(100 * g.reached_close_old.astype(bool).mean(), 1),
                         "close new %": round(100 * g.reached_close_new.astype(bool).mean(), 1),
                         "days_to_reach old": round(g.days_to_reach_old.mean(), 1),
                         "days_to_reach new": round(g.days_to_reach_new.mean(), 1),
                         "final dist old": round(g.final_close_distance_from_target_old.mean(), 1),
                         "final dist new": round(g.final_close_distance_from_target_new.mean(), 1)})
        out.append(md_table(pd.DataFrame(rows)))

        out.append("### Confound test — touch by basis bucket, old vs new (same rows, same buckets)\n")
        rows = []
        for (rg, hz), g in c.groupby(["regime", "horizon"]):
            for ver in ("old", "new"):
                rec = {"regime": rg, "horizon": int(hz), "version": ver}
                ks = {}
                for bk in BUCKETS:
                    gb = g[g.bucket == bk]
                    k, n = int(gb[f"reached_touch_{ver}"].astype(bool).sum()), len(gb)
                    ks[bk] = (k, n)
                    rec[bk] = f"{k}/{n} = {100 * k / n:.1f}%" if n else ""
                z, p = two_prop_p(*ks["<20"], *ks["35+"])
                rec["z (<20 vs 35+)"], rec["p"] = round(z, 2), round(p, 4)
                rec["dtr <20"] = round(g[g.bucket == "<20"][f"days_to_reach_{ver}"].mean(), 1)
                rec["dtr 35+"] = round(g[g.bucket == "35+"][f"days_to_reach_{ver}"].mean(), 1)
                rows.append(rec)
        out.append(md_table(pd.DataFrame(rows)))
        ma5 = c[(c.regime == "magnet-above") & (c.horizon == 5) & (~c.qa_flag_t0)]
        ks = {bk: (int(ma5[ma5.bucket == bk].reached_touch_new.astype(bool).sum()), int((ma5.bucket == bk).sum()))
              for bk in BUCKETS}
        z, p = two_prop_p(*ks["<20"], *ks["35+"])
        out.append("Sensitivity (new version, magnet-above 5-session, excluding rows whose t0 is ES-QA-flagged): "
                   + ", ".join(f"{bk} {k}/{n}" for bk, (k, n) in ks.items()) + f"; z = {z:.2f}, p = {p:.4f}.\n")

        out.append("### Label flips (rows computed in both)\n")
        f = c[(c.reached_touch_old != c.reached_touch_new) | (c.reached_close_old != c.reached_close_new)].copy()
        f["touch"] = f.reached_touch_old.astype(str).str[0] + "→" + f.reached_touch_new.astype(str).str[0]
        f["close"] = f.reached_close_old.astype(str).str[0] + "→" + f.reached_close_new.astype(str).str[0]
        f["ctl touch/close"] = f.reached_touch_ctl.astype(str).str[0] + "/" + f.reached_close_ctl.astype(str).str[0]
        f["basis"] = f.basis.round(1)
        f["target"] = f.drift_target.round(1)
        f["dist old"] = f.final_close_distance_from_target_old.round(1)
        f["dist new"] = f.final_close_distance_from_target_new.round(1)
        out.append(f"{len(f)} rows flip: touch {int((f.reached_touch_old != f.reached_touch_new).sum())}, "
                   f"close {int((f.reached_close_old != f.reached_close_new).sum())}.\n")
        out.append(md_table(f.reset_index()[["trade_date", "regime", "horizon", "bucket", "basis", "series_segment",
                                             "lagged_day", "horizon_crosses_cutover", "qa_flag_t0", "target",
                                             "touch", "close", "ctl touch/close", "dist old", "dist new"]]))

    out.append(f"\n## ES QA flag (flag only — no price is changed) — {len(qa)} of {len(spx_daily)} SPX sessions\n")
    out.append(f"Rule: intraday basis spread p95−p5 (after 06:45) > {QA_SPREAD:.0f}, or |close basis − day median| > "
               f"{QA_CLOSE:.0f}, or |ES-implied high/low − SPX high/low| > {QA_HILO:.0f} pts.\n")
    if len(qa):
        out.append("By segment: " + ", ".join(f"{k} {v}" for k, v in qa.groupby("segment").size().items())
                   + f"; lagged days {int(qa.lagged.sum())}.\n")
        out.append(md_table(qa))
    if len(lag_err):
        out.append(f"\nLagged-day close (median `stock_price` 12:56–13:00) vs the day's own basis, {len(lag_err)} days: "
                   f"median |err| {lag_err.abs().median():.1f}, p90 {lag_err.abs().quantile(.9):.1f}, "
                   f"max {lag_err.abs().max():.1f}, mean {lag_err.mean():+.1f} pts.\n")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
