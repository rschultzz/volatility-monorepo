#!/usr/bin/env python3
"""CR-BL — backfill bt_vol_state_daily + bt_range_labels_daily (INSERT-only, backfill protocol).

Spec: specs/CR-BL-vol-state-range-labels.md (L1–L14, Amendment A1). SPX cash only, no ES.

Vol state for trade date t (all observable before the 06:30 PT open of t, L3):
  - implied: prior session's latest orats_monies_minute snapshot <= 12:46 PT; constant-maturity
    ATM IV at 7 and 30 calendar days from the snapshot date, linear in total variance
    (atmiv² × calendar days) between the bracketing PM expiries (A1.2: expiry_tod = 'am' and
    0-calendar-day expiries are not nodes); exact node used when one sits on the target.
  - realized: CR-BH hybrid SPX cash closes / highs / lows strictly before t; close-to-close
    (sample std of log returns) and Parkinson, 5 and 21 sessions, × sqrt(252).
  - vrp_7 = iv_7d − rv_5_cc, vrp_30 = iv_30d − rv_21_cc (vol points).
  - percentiles (A1.4): 100 × share of up to 250 prior non-NULL values <= x_t, t excluded, min 120.

Range labels, horizon h in {1, 5}, sessions t..t+h−1 (NYSE calendar), SPX cash RTH:
  range_over_implied     = (max high − min low) / implied_move_pts
  abs_close_over_implied = |close of t+h−1 − open of t| / implied_move_pts
  h=1: implied_move_pts = stored implied_move_1d of SOURCE_FEATURE_VERSION (A1.3: NULL ratios
       where none is stored; never recomputed).
  h=5: implied_move_pts = SPX open × sqrt(w / 365), w = total variance (atmiv² × days)
       interpolated to the calendar days from the t−1 snapshot date to the horizon-end date (A1.1).

Holdout (trade_date > 2026-06-05, or an h=5 horizon ending after it): rows are written but never
printed, summarised or read back. Smoke checks read train rows only.

Usage:
    apps/web/.venv/bin/python scripts/cr_bl_backfill.py --dry-run
    apps/web/.venv/bin/python scripts/cr_bl_backfill.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)
from packages.shared.spx_cash import load_spx_daily_bars  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

CR_ID = "CR-BL"
VERSION = "cr-bl-v1"
TICKER = "SPX"
SOURCE_FEATURE_VERSION = "v0.6.1-nyse-sessions"   # implied_move_1d, reused as stored (L4)
START = dt.date(2023, 5, 1)                        # first orats_monies_minute date
SPLIT = dt.date(2026, 6, 5)                        # train: trade_date (and h=5 horizon end) <= SPLIT
SNAP_CUTOFF = dt.time(12, 46)
HORIZONS = (1, 5)
PCT_WINDOW, PCT_MIN = 250, 120
PCT_FIELDS = ("iv_7d", "iv_30d", "rv_21_cc", "vrp_30", "vrp_7")

_SNAP_SQL = """
    SELECT max(snapshot_pt) FROM orats_monies_minute
    WHERE trade_date = %s AND snapshot_pt <= %s
"""
_NODES_SQL = """
    SELECT expir_date, atmiv FROM orats_monies_minute
    WHERE trade_date = %s AND snapshot_pt = %s AND ticker = 'SPX'
      AND expiry_tod = 'pm' AND atmiv > 0 AND expir_date > trade_date
"""
_IM1_SQL = """
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float
    FROM bt_daily_features
    WHERE ticker = %s AND feature_version = %s AND active
"""


# ── pure helpers ─────────────────────────────────────────────────────────────

def total_variance_at(nodes: list[tuple[int, float]], target: int):
    """nodes: sorted [(calendar days, atmiv)]. Returns (w = σ²·days at target, lo_days, hi_days) or None."""
    lo = [n for n in nodes if n[0] <= target]
    hi = [n for n in nodes if n[0] >= target]
    if not lo or not hi:
        return None
    (t1, v1), (t2, v2) = lo[-1], hi[0]
    w1, w2 = v1 * v1 * t1, v2 * v2 * t2
    w = w1 if t1 == t2 else w1 + (w2 - w1) * (target - t1) / (t2 - t1)
    return w, t1, t2


def trailing_pct(values: np.ndarray) -> np.ndarray:
    """A1.4: 100 × share of up to PCT_WINDOW prior non-NaN values <= x_t; t excluded; >= PCT_MIN priors."""
    out = np.full(len(values), np.nan)
    for i, x in enumerate(values):
        if np.isnan(x):
            continue
        w = values[max(0, i - PCT_WINDOW):i]
        w = w[~np.isnan(w)]
        if len(w) >= PCT_MIN:
            out[i] = 100.0 * float((w <= x).mean())
    return out


def nan_none(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return None
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ── load ─────────────────────────────────────────────────────────────────────

def load_snapshots(conn, sessions: list[dt.date]) -> dict:
    """{snapshot session date: (snapshot_pt, [(calendar days, atmiv)] sorted)}."""
    out = {}
    for d in sessions:
        sp = conn.execute(_SNAP_SQL, (str(d), dt.datetime.combine(d, SNAP_CUTOFF))).fetchone()[0]
        if sp is None:
            continue
        rows = conn.execute(_NODES_SQL, (str(d), sp)).fetchall()
        nodes = sorted(((dt.date.fromisoformat(e) - d).days, float(v)) for e, v in rows)
        out[d] = (sp, [n for n in nodes if n[0] >= 1])
    return out


def build(conn):
    last_snap = conn.execute("SELECT max(trade_date) FROM orats_monies_minute").fetchone()[0]
    end = dt.date.fromisoformat(last_snap)
    sessions = [d.date() for d in pd.date_range(START, end) if is_trading_day(d.date())]
    print(f"sessions {sessions[0]} → {sessions[-1]} ({len(sessions)})", flush=True)

    snaps = load_snapshots(conn, sessions)
    print(f"snapshots loaded: {len(snaps)}", flush=True)
    spx, _dropped, _clean = load_spx_daily_bars(conn, START, end)
    spx.index = [d for d in spx.index]
    print(f"SPX cash sessions: {len(spx)}", flush=True)
    im1 = {d: v for d, v in conn.execute(_IM1_SQL, (TICKER, SOURCE_FEATURE_VERSION)).fetchall()}

    # ── vol state ────────────────────────────────────────────────────────────
    c = spx.reindex(sessions)
    lr = np.log(c["close"] / c["close"].shift(1))
    pk = np.log(c["high"] / c["low"]) ** 2 / (4 * np.log(2))
    rv = {}
    for n in (5, 21):
        rv[f"rv_{n}_cc"] = (lr.rolling(n).std(ddof=1).shift(1) * np.sqrt(252) * 100).to_numpy()
        rv[f"rv_{n}_pk"] = (np.sqrt(pk.rolling(n).mean().shift(1) * 252) * 100).to_numpy()

    vs = []
    for i, d in enumerate(sessions):
        if i == 0:
            continue
        p = sessions[i - 1]
        row = {"trade_date": d, "rv_last_close_date": p,
               "sessions_in_7d": sum(is_trading_day(p + dt.timedelta(days=k)) for k in range(1, 8))}
        for k in rv:
            row[k] = rv[k][i]
        if p in snaps:
            sp, nodes = snaps[p]
            row.update(iv_snapshot_date=p, iv_snapshot_pt=sp)
            for T in (7, 30):
                r = total_variance_at(nodes, T)
                if r:
                    w, t1, t2 = r
                    row[f"iv_{T}d"] = np.sqrt(w / T) * 100
                    row[f"iv_{T}d_lo_expiry"] = p + dt.timedelta(days=t1)
                    row[f"iv_{T}d_hi_expiry"] = p + dt.timedelta(days=t2)
        vs.append(row)
    vs = pd.DataFrame(vs).set_index("trade_date")
    for col in ("iv_7d", "iv_30d", "iv_snapshot_date", "iv_snapshot_pt"):
        if col not in vs:
            vs[col] = np.nan
    vs["vrp_7"] = vs["iv_7d"] - vs["rv_5_cc"]
    vs["vrp_30"] = vs["iv_30d"] - vs["rv_21_cc"]
    for f in PCT_FIELDS:
        vs[f + "_pct"] = trailing_pct(vs[f].astype(float).to_numpy())

    # ── labels ───────────────────────────────────────────────────────────────
    pos = {d: i for i, d in enumerate(sessions)}
    labels = []
    for d in sessions:
        i = pos[d]
        for h in HORIZONS:
            if i + h - 1 >= len(sessions):
                continue
            win = sessions[i:i + h]
            if any(x not in spx.index for x in win):
                continue
            bars = spx.loc[win]
            o = float(bars["open"].iat[0])
            hi, lo, cl = float(bars["high"].max()), float(bars["low"].min()), float(bars["close"].iat[-1])
            row = {"trade_date": d, "horizon_sessions": h, "horizon_end_date": win[-1], "spx_open": o,
                   "horizon_high": hi, "horizon_low": lo, "horizon_close": cl,
                   "min_session_minutes": int(bars["n_minutes"].min()),
                   "implied_vol_h": None, "implied_maturity_days": None, "implied_move_pts": None}
            if h == 1:
                row["implied_move_basis"] = "implied_move_1d"
                v = im1.get(d)
                if v is not None and v > 0:
                    row["implied_move_pts"] = float(v)
            else:
                row["implied_move_basis"] = "total_var_to_horizon_end"
                p = sessions[i - 1] if i > 0 else None
                if p in snaps:
                    T = (win[-1] - p).days
                    r = total_variance_at(snaps[p][1], T)
                    if r:
                        w = r[0]
                        row["implied_maturity_days"] = T
                        row["implied_vol_h"] = float(np.sqrt(w / T) * 100)
                        row["implied_move_pts"] = float(o * np.sqrt(w / 365.0))
            im = row["implied_move_pts"]
            row["range_over_implied"] = (hi - lo) / im if im else None
            row["abs_close_over_implied"] = abs(cl - o) / im if im else None
            labels.append(row)
    labels = pd.DataFrame(labels)
    return vs, labels


def is_train_label(r) -> bool:
    return r.trade_date <= SPLIT and r.horizon_end_date <= SPLIT


# ── write ────────────────────────────────────────────────────────────────────

VS_COLS = ["iv_snapshot_date", "iv_snapshot_pt", "iv_7d", "iv_30d", "iv_7d_lo_expiry", "iv_7d_hi_expiry",
           "iv_30d_lo_expiry", "iv_30d_hi_expiry", "sessions_in_7d", "rv_last_close_date", "rv_5_cc", "rv_21_cc",
           "rv_5_pk", "rv_21_pk", "vrp_7", "vrp_30", "iv_7d_pct", "iv_30d_pct", "rv_21_cc_pct", "vrp_30_pct",
           "vrp_7_pct"]
LB_COLS = ["horizon_end_date", "spx_open", "horizon_high", "horizon_low", "horizon_close", "min_session_minutes",
           "implied_move_basis", "implied_vol_h", "implied_maturity_days", "implied_move_pts", "range_over_implied",
           "abs_close_over_implied"]


def insert_all(conn, run_id, vs: pd.DataFrame, labels: pd.DataFrame) -> tuple[int, int]:
    vs_sql = (f"INSERT INTO bt_vol_state_daily (ticker, trade_date, version, {', '.join(VS_COLS)}, backfill_run_id) "
              f"VALUES (%s, %s, %s, {', '.join(['%s'] * len(VS_COLS))}, %s)")
    lb_sql = (f"INSERT INTO bt_range_labels_daily (ticker, trade_date, horizon_sessions, version, {', '.join(LB_COLS)}, "
              f"backfill_run_id) VALUES (%s, %s, %s, %s, {', '.join(['%s'] * len(LB_COLS))}, %s)")
    n_vs = n_lb = 0
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(vs_sql, [
                (TICKER, d, VERSION, *[nan_none(r.get(c)) for c in VS_COLS], run_id) for d, r in vs.iterrows()])
            n_vs = len(vs)
            cur.executemany(lb_sql, [
                (TICKER, r["trade_date"], int(r["horizon_sessions"]), VERSION, *[nan_none(r[c]) for c in LB_COLS],
                 run_id)
                for r in labels.to_dict("records")])
            n_lb = len(labels)
    update_run_progress(conn, run_id, n_vs + n_lb)
    return n_vs, n_lb


# ── smoke (train rows only) ──────────────────────────────────────────────────

HAND_CHECKS = {   # Step-0 Q7 hand-checked values (vol points): iv_7d, iv_30d, rv_21_cc
    dt.date(2024, 8, 6): (31.47, 27.32, 18.52),
    dt.date(2025, 4, 7): (45.74, 33.31, 32.50),
    dt.date(2025, 4, 9): (61.18, 42.37, 31.42),
    dt.date(2024, 5, 17): (9.87, 10.31, 11.61),
    dt.date(2025, 6, 9): (13.29, 13.25, 15.91),
}


def smoke(conn, vs: pd.DataFrame, labels: pd.DataFrame, read_back: bool) -> dict:
    res = {}
    tv = vs[vs.index <= SPLIT]
    res["vol_state_train_rows"] = int(len(tv))
    res["iv_7d_pct_first_date"] = str(tv[tv["iv_7d_pct"].notna()].index.min())
    res["iv_7d_pct_train_n"] = int(tv["iv_7d_pct"].notna().sum())
    assert all(r.iv_snapshot_date < d for d, r in vs.dropna(subset=["iv_snapshot_date"]).iterrows()), "IV lookahead"
    assert all(r.rv_last_close_date < d for d, r in vs.iterrows()), "RV lookahead"
    assert max(pd.Timestamp(x).time() for x in vs["iv_snapshot_pt"].dropna()) <= SNAP_CUTOFF
    hc = {}
    for d, exp in HAND_CHECKS.items():
        got = tuple(round(float(vs.loc[d, c]), 2) for c in ("iv_7d", "iv_30d", "rv_21_cc"))
        hc[str(d)] = got
        assert all(abs(a - b) <= 0.011 for a, b in zip(got, exp)), f"hand check {d}: {got} vs {exp}"
    res["hand_checks_ok"] = hc

    train = labels[labels.apply(is_train_label, axis=1)]
    for h in HORIZONS:
        t = train[train.horizon_sessions == h]
        res[f"labels_train_h{h}_rows"] = int(len(t))
        res[f"labels_train_h{h}_with_implied"] = int(t["implied_move_pts"].notna().sum())
    t5 = train[(train.horizon_sessions == 5) & train.implied_move_pts.notna()].merge(
        vs[["iv_7d", "sessions_in_7d"]], left_on="trade_date", right_index=True)
    normal = t5[t5.sessions_in_7d == 5]
    old = normal.spx_open * normal.iv_7d / 100 * np.sqrt(5 / 252)
    res["h5_A1_vs_L8_normal_weeks_median_ratio"] = round(float((normal.implied_move_pts / old).median()), 4)
    res["h5_A1_vs_L8_normal_weeks_n"] = int(len(normal))

    if read_back:
        rb = {}
        rb["vol_state_train"] = conn.execute(
            "SELECT count(*) FROM bt_vol_state_daily WHERE version=%s AND trade_date<=%s", (VERSION, SPLIT)).fetchone()[0]
        for h in HORIZONS:
            rb[f"labels_train_h{h}"] = conn.execute(
                "SELECT count(*) FROM bt_range_labels_daily WHERE version=%s AND horizon_sessions=%s "
                "AND trade_date<=%s AND horizon_end_date<=%s", (VERSION, h, SPLIT, SPLIT)).fetchone()[0]
        assert rb["vol_state_train"] == res["vol_state_train_rows"]
        for h in HORIZONS:
            assert rb[f"labels_train_h{h}"] == res[f"labels_train_h{h}_rows"]
        res["read_back_train"] = rb
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute + smoke, no writes, no run row")
    args = ap.parse_args()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    existing = conn.execute("SELECT (SELECT count(*) FROM bt_vol_state_daily WHERE version=%s) + "
                            "(SELECT count(*) FROM bt_range_labels_daily WHERE version=%s)", (VERSION, VERSION)).fetchone()[0]
    if existing:
        sys.exit(f"ERROR: {existing} rows already exist for version {VERSION}; INSERT-only, bump VERSION to rerun.")

    vs, labels = build(conn)
    print(f"built: vol state {len(vs)} rows, labels {len(labels)} rows", flush=True)

    if args.dry_run:
        print(smoke(conn, vs, labels, read_back=False))
        return

    with backfill_run(conn, CR_ID) as run_id:
        n_vs, n_lb = insert_all(conn, run_id, vs, labels)
        print(f"inserted: bt_vol_state_daily {n_vs}, bt_range_labels_daily {n_lb}", flush=True)
        res = smoke(conn, vs, labels, read_back=True)
        res.update(inserted_vol_state=n_vs, inserted_labels=n_lb, version=VERSION)
        print(res, flush=True)
        update_run_smoke(conn, run_id, res,
                         f"inserted {n_vs} vol-state + {n_lb} label rows ({VERSION}); train read-back counts match; "
                         f"5 Step-0 hand checks match; no IV/RV lookahead")


if __name__ == "__main__":
    main()
