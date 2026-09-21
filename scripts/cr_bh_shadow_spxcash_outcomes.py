#!/usr/bin/env python3
"""CR-BH — shadow-recompute bt_daily_outcomes on SPX cash into a new feature_version.

Every active canonical row (ticker SPX, v0.6.0-openiv) is recomputed through the shared
packages/shared/outcomes_runner path with ONE change: the daily OHLC frame is SPX cash
(scripts/cr_bh_spx_cash, per-period hybrid series — spec Amendment A1) instead of
front-contract ES. Same targets, regime labels, horizons and tolerance rules.

Rows are INSERTed under feature_version 'v0.6.0-openiv-spxcash' (ON CONFLICT DO NOTHING).
Nothing existing is updated or deleted; the canonical version is not promoted; no
bt_daily_features rows are written (spec Step 0 Q1/Q2: no reader sees the new version).

The CR-G (session_*_t1/t5/t15) and CR-I (position_t*_post_touch) columns are computed here
from the same SPX frame, with those scripts' row scopes.

Control: every row is also recomputed on ES through the same path (read-only) so that
label changes not caused by the price series are separable in the diff. Both result sets
are written to scripts/.cache/cr_bh_rows.pkl for scripts/cr_bh_diff.py.

Usage:
    apps/web/.venv/bin/python scripts/cr_bh_shadow_spxcash_outcomes.py --dry-run
    apps/web/.venv/bin/python scripts/cr_bh_shadow_spxcash_outcomes.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.outcomes_runner import (  # noqa: E402
    CONTAINMENT_COLUMNS,
    compute_outcome_for_date,
    compute_session_containment,
)
from packages.shared.probability import classify_post_touch_positions  # noqa: E402
from scripts.cr_b_backfill_outcomes import _fetch_daily_bars, _fetch_landscape  # noqa: E402
from scripts.cr_bh_spx_cash import (  # noqa: E402
    build_series, daily_ohlc, fetch_minutes, is_lagged, normalize_minutes, series_segment,
)

CR_ID = "CR-BH"
TICKER = "SPX"
SHADOW_VERSION = f"{CANONICAL_FEATURE_VERSION}-spxcash"
CACHE_DIR = REPO_ROOT / "scripts" / ".cache"
LOG_DIR = REPO_ROOT / "scripts" / "logs"
ROWS_CACHE = CACHE_DIR / "cr_bh_rows.pkl"
MINUTES_CACHE = CACHE_DIR / "cr_bh_spot_stock_minutes.pkl"
TIMEFRAMES = (1, 5, 15)

OUTCOME_COLUMNS = (
    "regime_kind_at_classification", "dominant_bucket_at_classification",
    "horizon_sessions", "horizon_end_date", "outcome_status",
    "reached_touch", "reached_close", "days_to_reach",
    "max_excursion_in_direction", "final_close_distance_from_target", "actual_realized_em_pct",
)
TN_COLUMNS = tuple(f"session_{k}_t{n}" for n in TIMEFRAMES for k in ("open", "high", "low", "close"))
POS_COLUMNS = tuple(f"position_t{n}_post_touch" for n in TIMEFRAMES)
INSERT_COLUMNS = (("ticker", "trade_date", "feature_version") + OUTCOME_COLUMNS + ("session_open_t0",)
                  + CONTAINMENT_COLUMNS + TN_COLUMNS + POS_COLUMNS + ("backfill_run_id",))

_INSERT_SQL = (
    f"INSERT INTO bt_daily_outcomes ({', '.join(INSERT_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(INSERT_COLUMNS))}) "
    "ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING"
)

_SOURCE_ROWS_SQL = """
    SELECT o.trade_date, f.regime_at_classification, f.feature_vector
    FROM bt_daily_outcomes o
    JOIN bt_daily_features f
      ON f.ticker = o.ticker AND f.trade_date = o.trade_date
     AND f.feature_version = o.feature_version AND f.active = TRUE
    WHERE o.ticker = %s AND o.feature_version = %s AND o.active = TRUE
    ORDER BY o.trade_date
"""


def _implied_move(fv: dict):
    try:
        im = float(fv.get("implied_move_1d"))
    except (TypeError, ValueError):
        return None
    return im if im > 0 else None


def compute_row(trade_date: dt.date, regime: str, fv: dict, landscape: dict, bars: pd.DataFrame) -> dict:
    """One full outcome row on the given daily frame (no ticker / version / run id)."""
    fv = fv or {}
    walls = landscape.get("walls") or []
    outcome, open_t0 = compute_outcome_for_date(trade_date, regime, fv, landscape, bars)
    row = {**outcome, "session_open_t0": open_t0}
    row.update(compute_session_containment(trade_date, walls, bars, fv.get("implied_move_1d")))

    # CR-G scope: computed + na_regime rows; Nth session after trade_date
    row.update({c: None for c in TN_COLUMNS})
    if outcome["outcome_status"] in ("computed", "na_regime"):
        later = bars[bars.index > trade_date].sort_index()
        for n in TIMEFRAMES:
            if len(later) >= n:
                b = later.iloc[n - 1]
                for k in ("open", "high", "low", "close"):
                    row[f"session_{k}_t{n}"] = float(b[k])

    # CR-I scope: touched rows; close at days_to_reach + N vs target ± 0.25 × IM
    row.update({c: None for c in POS_COLUMNS})
    target, im = pick_drift_target(walls), _implied_move(fv)
    if outcome.get("reached_touch") and outcome.get("days_to_reach") is not None and target is not None and im:
        forward = bars[bars.index >= trade_date].sort_index().dropna(subset=["high", "low", "close"])
        pos = classify_post_touch_positions(
            days_to_reach=outcome["days_to_reach"], horizon_bars=forward,
            drift_target=float(target), tolerance=0.25 * im, timeframes_sessions=TIMEFRAMES)
        for n in TIMEFRAMES:
            row[f"position_t{n}_post_touch"] = pos.get(n)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="compute, print, cache; write nothing")
    ap.add_argument("--use-minutes-cache", action="store_true",
                    help=f"read {MINUTES_CACHE.name} instead of querying orats_monies_minute (dev only)")
    args = ap.parse_args()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    src = conn.execute(_SOURCE_ROWS_SQL, (TICKER, CANONICAL_FEATURE_VERSION)).fetchall()
    dates = [r[0] for r in src]
    print(f"=== {CR_ID} shadow recompute → {SHADOW_VERSION} ===")
    print(f"source rows: {len(src)} active {CANONICAL_FEATURE_VERSION} rows, {dates[0]} → {dates[-1]}")
    existing = conn.execute(
        "SELECT count(*) FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s",
        (TICKER, SHADOW_VERSION)).fetchone()[0]
    print(f"existing {SHADOW_VERSION} rows: {existing}")

    landscape = _fetch_landscape(conn, TICKER, dates)
    bar_from, bar_to = min(dates), max(dates) + dt.timedelta(days=90)
    es_daily = _fetch_daily_bars(conn, bar_from, bar_to)

    if args.use_minutes_cache and MINUTES_CACHE.exists():
        raw = normalize_minutes(pd.read_pickle(MINUTES_CACHE))
    else:
        raw = fetch_minutes(conn, bar_from, bar_to)
    clean, dropped = build_series(raw)
    spx_daily = daily_ohlc(clean)
    LOG_DIR.mkdir(exist_ok=True)
    dropped.to_csv(LOG_DIR / "cr_bh_dropped_minutes.csv", index=False)
    spx_daily.to_csv(LOG_DIR / "cr_bh_spx_daily.csv")
    print(f"SPX sessions: {len(spx_daily)} ({spx_daily.index.min()} → {spx_daily.index.max()}); "
          f"ES sessions: {len(es_daily)}; dropped minutes: {len(dropped)} "
          f"({dropped['reason'].value_counts().to_dict()})")
    only_es = sorted(set(es_daily.index) - set(spx_daily.index))
    print(f"ES 'sessions' with no SPX session ({len(only_es)}): {', '.join(map(str, only_es))}")
    print(f"outcome dates with no SPX session: {sorted(set(dates) - set(spx_daily.index))}")

    ohlc = spx_daily[["open", "high", "low", "close"]]
    new_rows, ctl_rows = [], []
    for trade_date, regime, fv in src:
        ls = landscape.get(trade_date, {})
        new_rows.append({"trade_date": trade_date, **compute_row(trade_date, regime, fv, ls, ohlc)})
        ctl_rows.append({"trade_date": trade_date, **compute_row(trade_date, regime, fv, ls, es_daily)})
        new_rows[-1].update({
            "drift_target": pick_drift_target(ls.get("walls") or []),
            "implied_move_1d": _implied_move(fv or {}),
            "series_segment": series_segment(trade_date), "lagged_day": is_lagged(trade_date),
        })
    new_df, ctl_df = pd.DataFrame(new_rows), pd.DataFrame(ctl_rows)
    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle({"new": new_df, "control": ctl_df, "spx_daily": spx_daily, "es_daily": es_daily,
                  "spx_minutes": clean}, ROWS_CACHE)

    print("\nstatus   new (SPX):", dict(Counter(new_df["outcome_status"])))
    print("status   control (ES):", dict(Counter(ctl_df["outcome_status"])))
    comp = new_df[new_df["outcome_status"] == "computed"]
    print(f"computed (SPX): touch {comp['reached_touch'].mean():.3f}  close {comp['reached_close'].mean():.3f}  n={len(comp)}")
    compc = ctl_df[ctl_df["outcome_status"] == "computed"]
    print(f"computed (ES) : touch {compc['reached_touch'].mean():.3f}  close {compc['reached_close'].mean():.3f}  n={len(compc)}")
    show = ["trade_date", "outcome_status", "horizon_sessions", "reached_touch", "reached_close",
            "session_open_t0", "session_close_t0", "final_close_distance_from_target"]
    print("\nsample (SPX):")
    print(new_df[show].iloc[:: max(1, len(new_df) // 12)].to_string(index=False))

    if args.dry_run:
        print("\n[DRY RUN — nothing written]")
        conn.close()
        return 0

    with backfill_run(conn, CR_ID) as run_id:
        params = [
            tuple([TICKER, r["trade_date"], SHADOW_VERSION] + [r[c] for c in INSERT_COLUMNS[3:-1]] + [run_id])
            for r in new_rows
        ]
        inserted = 0
        with conn.transaction():
            with conn.cursor() as cur:
                for p in params:
                    cur.execute(_INSERT_SQL, p)
                    inserted += cur.rowcount
        update_run_progress(conn, run_id, inserted)

        after = conn.execute(
            """SELECT count(*), count(*) FILTER (WHERE backfill_run_id = %s),
                      count(*) FILTER (WHERE outcome_status='computed'),
                      count(*) FILTER (WHERE outcome_status='pending_history'),
                      count(*) FILTER (WHERE outcome_status='na_regime'),
                      count(*) FILTER (WHERE outcome_status='na_data')
               FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s""",
            (run_id, TICKER, SHADOW_VERSION)).fetchone()
        canon = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE active), max(computed_at) FROM bt_daily_outcomes "
            "WHERE ticker=%s AND feature_version=%s", (TICKER, CANONICAL_FEATURE_VERSION)).fetchone()
        smoke = {
            "shadow_version": SHADOW_VERSION, "source_rows": len(src), "inserted": inserted,
            "shadow_total": after[0], "shadow_this_run": after[1],
            "computed": after[2], "pending_history": after[3], "na_regime": after[4], "na_data": after[5],
            "canonical_total": canon[0], "canonical_active": canon[1], "canonical_max_computed_at": str(canon[2]),
            "dropped_minutes": int(len(dropped)), "spx_sessions": int(len(spx_daily)),
        }
        ok = inserted == len(src) - existing and after[0] == len(src)
        update_run_smoke(conn, run_id, smoke,
                         "OK: one shadow row per active canonical row; canonical untouched" if ok
                         else "CHECK: inserted/total does not match source row count")
        print(f"\nrun {run_id}: inserted {inserted}; smoke {smoke}")
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
