#!/usr/bin/env python3
"""CR-024 (CR-D) — Vol surface feature backfill for bt_daily_features.

Populates five NULL vol surface columns for feature_version='v0.5.0-rebuilt':
    atm_iv_percentile    — percentile 0–100 vs trailing 60-session ATM IV distribution
    skew_percentile      — percentile 0–100 vs trailing 60-session put-call skew distribution
    term_structure_slope — raw IV points: near-30-DTE ATM IV − near-90-DTE ATM IV
    smile_convexity      — percentile 0–100 vs trailing 60-session wing-premium distribution
    vol_risk_premium     — raw vol points: 20-session realized ES vol − current ATM IV

Writes use COALESCE — existing non-NULL values are never overwritten.
backfill_run_id is not modified (CR-A attribution preserved).

Usage:
    python scripts/cr_d_backfill_vol_features.py
    python scripts/cr_d_backfill_vol_features.py --from-date 2023-05-05 --to-date 2023-05-05
    python scripts/cr_d_backfill_vol_features.py --dry-run

Exit: 0 on success; 1 if any date failed or full-run smoke flags stop condition.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.vol_features import (
    compute_atm_iv_percentile,
    compute_realized_vol_20d,
    compute_skew_percentile,
    compute_smile_convexity,
    compute_term_structure_slope,
    compute_vol_risk_premium,
    fetch_es_closes_before,
    fetch_iv_history_for_date,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURE_VERSION = "v0.5.0-rebuilt"

_TARGET_ROWS_SQL = """
    SELECT ticker, trade_date
    FROM bt_daily_features
    WHERE ticker = %s
      AND feature_version = %s
      AND active = TRUE
      AND (
          atm_iv_percentile    IS NULL
          OR skew_percentile   IS NULL
          OR term_structure_slope IS NULL
          OR smile_convexity   IS NULL
          OR vol_risk_premium  IS NULL
      )
    ORDER BY trade_date
"""

_UPDATE_VOL_SQL = """
    UPDATE bt_daily_features
    SET atm_iv_percentile    = COALESCE(atm_iv_percentile,    %s),
        skew_percentile      = COALESCE(skew_percentile,      %s),
        term_structure_slope = COALESCE(term_structure_slope, %s),
        smile_convexity      = COALESCE(smile_convexity,      %s),
        vol_risk_premium     = COALESCE(vol_risk_premium,     %s)
    WHERE ticker          = %s
      AND trade_date      = %s
      AND feature_version = %s
      AND (
          atm_iv_percentile    IS NULL
          OR skew_percentile   IS NULL
          OR term_structure_slope IS NULL
          OR smile_convexity   IS NULL
          OR vol_risk_premium  IS NULL
      )
"""


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _get_target_rows(
    conn,
    ticker: str,
    from_date: Optional[dt.date],
    to_date: Optional[dt.date],
    limit: Optional[int],
) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(_TARGET_ROWS_SQL, (ticker, FEATURE_VERSION))
        rows = cur.fetchall()
    if from_date:
        rows = [r for r in rows if r[1] >= from_date]
    if to_date:
        rows = [r for r in rows if r[1] <= to_date]
    if limit:
        rows = rows[:limit]
    return rows


def _compute_vol_features(
    conn,
    trade_date: dt.date,
    ticker: str,
) -> dict:
    """Compute all five vol features for a single trade_date.

    Returns dict with keys matching the 5 DB columns. Any value may be None
    if data is insufficient (< 60 prior sessions, no orats data, etc.).
    """
    history = fetch_iv_history_for_date(conn, trade_date, ticker)

    atm_iv_pct = compute_atm_iv_percentile(trade_date, history)
    skew_pct   = compute_skew_percentile(trade_date, history)
    conv_pct   = compute_smile_convexity(trade_date, history)

    # term_structure_slope: pass current-day scalars + prior-only slope history
    cur_rows = history.loc[history['trade_date'] == trade_date]
    if not cur_rows.empty:
        row0     = cur_rows.iloc[0]
        front_iv = float(row0['atm_iv'])     if pd.notna(row0['atm_iv'])     else None
        back_iv  = float(row0['far_atm_iv']) if pd.notna(row0['far_atm_iv']) else None
    else:
        front_iv = None
        back_iv  = None

    prior_slope = history.loc[
        history['trade_date'] < trade_date, ['trade_date', 'slope']
    ]
    raw_slope, _pct = compute_term_structure_slope(
        trade_date, front_iv, back_iv, prior_slope
    )

    # vol_risk_premium: ES 20-session realized vol vs current ATM IV
    closes       = fetch_es_closes_before(conn, trade_date, n=21)
    realized_vol = compute_realized_vol_20d(closes)
    vrp          = compute_vol_risk_premium(trade_date, realized_vol, front_iv)

    return {
        'atm_iv_percentile':    atm_iv_pct,
        'skew_percentile':      skew_pct,
        'term_structure_slope': raw_slope,
        'smile_convexity':      conv_pct,
        'vol_risk_premium':     vrp,
    }


def _update_vol_features(
    conn,
    ticker: str,
    trade_date: dt.date,
    features: dict,
) -> int:
    """Execute COALESCE UPDATE. Returns rowcount (0 = row already fully populated)."""
    with conn.cursor() as cur:
        cur.execute(_UPDATE_VOL_SQL, (
            features['atm_iv_percentile'],
            features['skew_percentile'],
            features['term_structure_slope'],
            features['smile_convexity'],
            features['vol_risk_premium'],
            ticker,
            trade_date,
            FEATURE_VERSION,
        ))
        return cur.rowcount


def _run_smoke(conn, ticker: str) -> dict:
    """Collect post-run stats for bt_backfill_runs.smoke_test_results."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                                    AS total,
                COUNT(*) FILTER (WHERE atm_iv_percentile    IS NULL
                                   AND skew_percentile      IS NULL
                                   AND term_structure_slope IS NULL
                                   AND smile_convexity      IS NULL
                                   AND vol_risk_premium     IS NULL)        AS null_all_five,
                COUNT(*) FILTER (WHERE atm_iv_percentile    IS NOT NULL)    AS have_atm_iv_pct,
                COUNT(*) FILTER (WHERE skew_percentile      IS NOT NULL)    AS have_skew_pct,
                COUNT(*) FILTER (WHERE term_structure_slope IS NOT NULL)    AS have_ts_slope,
                COUNT(*) FILTER (WHERE smile_convexity      IS NOT NULL)    AS have_convexity,
                COUNT(*) FILTER (WHERE vol_risk_premium     IS NOT NULL)    AS have_vrp,
                MIN(atm_iv_percentile),  MAX(atm_iv_percentile),
                MIN(skew_percentile),    MAX(skew_percentile),
                MIN(smile_convexity),    MAX(smile_convexity)
            FROM bt_daily_features
            WHERE ticker = %s AND feature_version = %s AND active = TRUE
        """, (ticker, FEATURE_VERSION))
        (total, null_all_five,
         have_atm, have_skew, have_ts, have_conv, have_vrp,
         atm_min, atm_max,
         skew_min, skew_max,
         conv_min, conv_max) = cur.fetchone()

    # v0.5.0 untouched: vol columns must remain NULL for original-version rows
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM bt_daily_features
            WHERE feature_version = 'v0.5.0'
              AND atm_iv_percentile IS NOT NULL
        """)
        v050_touched = int(cur.fetchone()[0])

    # Raw skew sign: vol75 > vol25 (put IV > call IV) should hold for most SPX EOD rows
    with conn.cursor() as cur:
        cur.execute("""
            WITH eod AS (
                SELECT trade_date, MAX(snapshot_pt) AS snap
                FROM orats_monies_minute
                WHERE ticker = %s
                  AND trade_date >= '2023-01-01'
                GROUP BY trade_date
            )
            SELECT
                COUNT(*) FILTER (WHERE o.vol75 > o.vol25) AS positive_skew,
                COUNT(*)                                   AS total_eod
            FROM eod
            JOIN orats_monies_minute o
              ON o.trade_date   = eod.trade_date
             AND o.ticker       = %s
             AND o.snapshot_pt  = eod.snap
             AND o.dte BETWEEN 20 AND 40
        """, (ticker, ticker))
        pos_skew, total_eod = cur.fetchone()

    def _f(v):
        return round(float(v), 4) if v is not None else None

    total_int         = int(total or 0)
    null_all_five_int = int(null_all_five or 0)
    pct_null_all      = null_all_five_int / max(total_int, 1) * 100

    return {
        "total_active_rows":     total_int,
        "null_all_five_count":   null_all_five_int,
        "null_all_five_pct":     round(pct_null_all, 2),
        "have_atm_iv_pct":       int(have_atm or 0),
        "have_skew_pct":         int(have_skew or 0),
        "have_ts_slope":         int(have_ts or 0),
        "have_convexity":        int(have_conv or 0),
        "have_vrp":              int(have_vrp or 0),
        "atm_iv_pct_range":      [_f(atm_min),  _f(atm_max)],
        "skew_pct_range":        [_f(skew_min), _f(skew_max)],
        "convexity_pct_range":   [_f(conv_min), _f(conv_max)],
        "v0_5_0_touched":        int(v050_touched),
        "raw_skew_positive_pct": round(
            int(pos_skew or 0) / max(int(total_eod or 0), 1) * 100, 1
        ),
    }


def _assess_smoke(smoke: dict, is_full_run: bool) -> tuple[str, str]:
    """Return (status_override, assessment). Empty string override → default 'completed'."""
    issues = []

    if is_full_run and smoke['null_all_five_pct'] > 5.0:
        issues.append(
            f"STOP: {smoke['null_all_five_pct']:.1f}% of rows still have all 5 vol "
            f"features NULL (threshold: 5%)"
        )

    for label, key in [
        ("atm_iv_percentile", "atm_iv_pct_range"),
        ("skew_percentile",   "skew_pct_range"),
        ("smile_convexity",   "convexity_pct_range"),
    ]:
        lo, hi = smoke[key]
        if lo is not None and lo < 0.0:
            issues.append(f"{label} min={lo} is below 0")
        if hi is not None and hi > 100.0:
            issues.append(f"{label} max={hi} is above 100")

    if smoke['v0_5_0_touched'] > 0:
        issues.append(f"v0.5.0 rows touched: {smoke['v0_5_0_touched']} (must be 0)")

    if smoke['raw_skew_positive_pct'] < 50.0:
        issues.append(
            f"raw skew positive only {smoke['raw_skew_positive_pct']}% of EOD rows "
            f"(expected >50% for SPX put skew)"
        )

    n_have  = smoke['have_atm_iv_pct']
    n_total = smoke['total_active_rows']
    partial = " (partial run — null counts expected high)" if not is_full_run else ""

    if issues:
        return "suspect", "ISSUES: " + "; ".join(issues)
    return "", (
        f"{n_have}/{n_total} rows populated{partial}; "
        f"percentile ranges OK; v0.5.0 untouched; "
        f"raw skew positive {smoke['raw_skew_positive_pct']}% — clean"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker",    default="SPX")
    ap.add_argument("--from-date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--to-date",   default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--limit",     type=int, default=None, metavar="N")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--batch",     type=int, default=30, metavar="N",
                    help="Progress heartbeat every N dates (default: 30)")
    args = ap.parse_args()

    ticker    = args.ticker
    from_date = dt.date.fromisoformat(args.from_date) if args.from_date else None
    to_date   = dt.date.fromisoformat(args.to_date)   if args.to_date   else None
    is_full_run = (from_date is None and to_date is None and args.limit is None)

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    target_rows = _get_target_rows(conn, ticker, from_date, to_date, args.limit)

    print(f"=== CR-024 Vol Surface Feature Backfill ===")
    print(f"Ticker: {ticker}   Feature version: {FEATURE_VERSION}")
    print(f"Target rows: {len(target_rows)}")
    if target_rows:
        print(f"Range: {target_rows[0][1]} → {target_rows[-1][1]}")
    if not is_full_run:
        print("[PARTIAL RUN — smoke stop-condition not enforced]")
    if args.dry_run:
        print("[DRY RUN — no writes]")
    print()

    if not target_rows:
        print("Nothing to do.")
        conn.close()
        sys.exit(0)

    if args.dry_run:
        for _, trade_date in target_rows:
            print(f"  [dry-run] {trade_date}")
        conn.close()
        sys.exit(0)

    n_updated = 0
    n_skipped = 0
    n_failed  = 0
    failed_dates: list[str] = []

    with backfill_run(conn, "CR-024") as run_id:
        print(f"Run ID: {run_id}")

        for i, (_, trade_date) in enumerate(target_rows, 1):
            try:
                features = _compute_vol_features(conn, trade_date, ticker)

                if all(v is None for v in features.values()):
                    # No usable orats/ES data for this date; skip UPDATE to avoid
                    # phantom rowcount from COALESCE(NULL, NULL).
                    n_skipped += 1
                    log.debug("skip %s: all features None", trade_date)
                else:
                    rowcount = _update_vol_features(conn, ticker, trade_date, features)
                    if rowcount > 0:
                        n_updated += 1
                    else:
                        n_skipped += 1

            except Exception as exc:
                n_failed += 1
                failed_dates.append(str(trade_date))
                log.error("ERROR %s: %s", trade_date, exc, exc_info=True)

            if i % args.batch == 0 or i == len(target_rows):
                update_run_progress(conn, run_id, n_updated)
                print(f"  [{i}/{len(target_rows)}] "
                      f"updated={n_updated} skipped={n_skipped} failed={n_failed}")

        smoke = _run_smoke(conn, ticker)
        smoke.update({
            "n_target":     len(target_rows),
            "n_updated":    n_updated,
            "n_skipped":    n_skipped,
            "n_failed":     n_failed,
            "failed_dates": failed_dates,
        })
        status_override, assessment = _assess_smoke(smoke, is_full_run)
        update_run_smoke(conn, run_id, smoke, assessment)

        print(f"\n=== DONE ===")
        print(f"updated={n_updated}  skipped={n_skipped}  failed={n_failed}")
        if failed_dates:
            print(f"  FAILED DATES: {', '.join(failed_dates)}")
        print(f"Rows with all 5 NULL: {smoke['null_all_five_count']} "
              f"({smoke['null_all_five_pct']}% of {smoke['total_active_rows']} active rows)")
        print(f"Coverage:  atm_iv_pct={smoke['have_atm_iv_pct']}  "
              f"skew_pct={smoke['have_skew_pct']}  "
              f"ts_slope={smoke['have_ts_slope']}  "
              f"convexity={smoke['have_convexity']}  "
              f"vrp={smoke['have_vrp']}")
        print(f"Pct ranges: atm={smoke['atm_iv_pct_range']}  "
              f"skew={smoke['skew_pct_range']}  "
              f"conv={smoke['convexity_pct_range']}")
        print(f"v0.5.0 rows touched:    {smoke['v0_5_0_touched']} (must be 0)")
        print(f"Raw skew positive:      {smoke['raw_skew_positive_pct']}%")
        print(f"Assessment: {assessment}")

    # backfill_run set status='completed'; override to 'suspect' if smoke flagged it
    if status_override == "suspect":
        conn.execute(
            "UPDATE bt_backfill_runs SET status = 'suspect' WHERE run_id = %s",
            (run_id,),
        )
        log.warning("Run marked suspect: %s", assessment)

    conn.close()
    sys.exit(1 if (n_failed > 0 or status_override == "suspect") else 0)


if __name__ == "__main__":
    main()
