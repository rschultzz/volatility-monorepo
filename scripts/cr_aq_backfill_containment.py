#!/usr/bin/env python3
"""CR-AQ — Backfill session containment columns on bt_daily_outcomes (null-fill).

For every active row at --feature-version with session_close_t0 IS NULL,
regardless of outcome_status, compute the CR-AQ containment outcome from:
  - the runner's RTH daily bars (same _RTH_BARS_SQL / _fetch_daily_bars the
    outcomes runner's callers use — one session source),
  - orats_gex_landscape.walls for the trade date (pre-open landscape for D),
  - implied_move_1d from bt_daily_features_active at that version,
and UPDATE the 11 columns. Sessions that have not closed are skipped.

Does NOT touch session_open_t0 or backfill_run_id (provenance of the earlier
CR-G / CR-I writes is kept; this run is recorded in bt_backfill_runs).

Data safety class: null_fill_update — eligible for unattended execution.

Usage:
    python -u scripts/cr_aq_backfill_containment.py [--feature-version V] [--dry-run] 2>&1 | tee scripts/logs/cr_aq_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from packages.shared.backfill_safety import (
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.outcomes_runner import CONTAINMENT_COLUMNS, compute_session_containment
from scripts.cr_b_backfill_outcomes import _fetch_daily_bars   # the runner callers' RTH session source

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

TICKER = "SPX"
BATCH = 50

_TARGETS_SQL = """
    SELECT o.trade_date, o.regime_kind_at_classification, o.outcome_status, f.feature_vector->>'implied_move_1d'
    FROM bt_daily_outcomes o
    LEFT JOIN bt_daily_features f
      ON f.ticker = o.ticker AND f.trade_date = o.trade_date
     AND f.feature_version = o.feature_version AND f.active = TRUE
    WHERE o.ticker = %s AND o.feature_version = %s AND o.active = TRUE
      AND o.session_close_t0 IS NULL
    ORDER BY o.trade_date
"""

_UPDATE_SQL = """
    UPDATE bt_daily_outcomes
    SET session_high_t0 = %s, session_low_t0 = %s, session_close_t0 = %s,
        wall_above_price = %s, wall_below_price = %s,
        contained_close = %s, contained_range = %s, close_pos_in_band = %s,
        range_over_im = %s, close_move_over_im = %s, breach_side = %s
    WHERE ticker = %s AND trade_date = %s AND feature_version = %s
      AND session_close_t0 IS NULL
"""


def _smoke(conn, fv: str) -> dict:
    q = lambda sql, *p: conn.execute(sql, p).fetchall()
    base = "FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s AND active"
    nn = q(f"SELECT count(*), count(session_close_t0), count(contained_close), count(range_over_im) {base}", TICKER, fv)[0]
    viol = q(f"SELECT count(*) {base} AND session_close_t0 IS NOT NULL AND NOT (session_low_t0 <= session_open_t0 AND session_open_t0 <= session_high_t0 AND session_low_t0 <= session_close_t0 AND session_close_t0 <= session_high_t0)", TICKER, fv)[0][0]
    by_regime = q(f"SELECT regime_kind_at_classification, count(*) FILTER (WHERE contained_close IS NOT NULL), count(*) FILTER (WHERE contained_close), count(*) FILTER (WHERE contained_range) {base} GROUP BY 1 ORDER BY 1", TICKER, fv)
    overall = q(f"SELECT count(*) FILTER (WHERE contained_close IS NOT NULL), count(*) FILTER (WHERE contained_close), count(*) FILTER (WHERE contained_range) {base}", TICKER, fv)[0]
    breach = q(f"SELECT breach_side, count(*) {base} AND breach_side IS NOT NULL GROUP BY 1 ORDER BY 1", TICKER, fv)
    checksum = q(f"SELECT md5(string_agg(session_open_t0::text, ',' ORDER BY trade_date)) {base}", TICKER, fv)[0][0]
    return {
        "rows": nn[0], "session_close_t0_nonnull": nn[1], "contained_close_nonnull": nn[2], "range_over_im_nonnull": nn[3],
        "ohlc_violations": viol,
        "contained_close_rate_overall": {"n": overall[0], "contained_close": overall[1], "contained_range": overall[2],
                                          "rate_close": round(overall[1] / overall[0], 4) if overall[0] else None},
        "contained_close_by_regime": {r[0]: {"n": r[1], "contained_close": r[2], "contained_range": r[3],
                                             "rate_close": round(r[2] / r[1], 4) if r[1] else None} for r in by_regime},
        "breach_side_dist": {r[0]: r[1] for r in breach},
        "session_open_t0_checksum": checksum,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feature-version", default=CANONICAL_FEATURE_VERSION,
                    help=f"bt_daily_outcomes.feature_version to fill (default: canonical = {CANONICAL_FEATURE_VERSION})")
    ap.add_argument("--dry-run", action="store_true", help="print target counts only; no writes, no run row")
    args = ap.parse_args(argv)
    fv = args.feature_version
    today_pt = dt.datetime.now(ZoneInfo("America/Los_Angeles")).date()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    log.info("Role verified: dash_backfill_writer ✓")
    log.info("Feature version: %s", fv)

    targets = conn.execute(_TARGETS_SQL, (TICKER, fv)).fetchall()
    targets = [t for t in targets if t[0] < today_pt]
    log.info("Target rows (session_close_t0 IS NULL, closed sessions): %d", len(targets))
    if args.dry_run:
        by_status = Counter(t[2] for t in targets)
        log.info("[dry-run] by outcome_status: %s", dict(by_status))
        log.info("[dry-run] date range: %s → %s", targets[0][0] if targets else None, targets[-1][0] if targets else None)
        return
    if not targets:
        log.info("Nothing to do."); return

    dates = [t[0] for t in targets]
    daily_bars = _fetch_daily_bars(conn, min(dates), max(dates))
    log.info("Loaded %d RTH daily sessions (%s → %s)", len(daily_bars), min(dates), max(dates))
    walls_by_date = {r[0]: (r[1] if isinstance(r[1], list) else []) for r in conn.execute(
        "SELECT trade_date, walls FROM orats_gex_landscape WHERE ticker=%s AND trade_date = ANY(%s)", (TICKER, dates)).fetchall()}
    log.info("Landscape walls loaded: %d dates", len(walls_by_date))

    with backfill_run(conn, "CR-AQ") as run_id:
        log.info("Run registered: run_id=%s", run_id)
        n_updated = n_no_bars = n_no_walls = n_failed = 0
        for i, (trade_date, regime, status, im) in enumerate(targets, 1):
            try:
                c = compute_session_containment(trade_date, walls_by_date.get(trade_date, []), daily_bars, im)
                if c["session_close_t0"] is None:
                    n_no_bars += 1
                    continue
                if c["contained_close"] is None:
                    n_no_walls += 1
                conn.execute(_UPDATE_SQL, tuple(c[k] for k in CONTAINMENT_COLUMNS) + (TICKER, trade_date, fv))
                n_updated += 1
            except Exception as exc:
                n_failed += 1
                log.error("FAILED %s: %s", trade_date, exc, exc_info=True)
            if i % BATCH == 0:
                update_run_progress(conn, run_id, n_updated)
                log.info("Progress: %d/%d  updated=%d no_bars=%d no_wall_side=%d failed=%d", i, len(targets), n_updated, n_no_bars, n_no_walls, n_failed)
        update_run_progress(conn, run_id, n_updated)
        log.info("Backfill complete: updated=%d no_bars=%d no_wall_side=%d failed=%d (of %d)", n_updated, n_no_bars, n_no_walls, n_failed, len(targets))

        smoke = _smoke(conn, fv)
        smoke.update({"n_targets": len(targets), "n_updated": n_updated, "n_no_bars": n_no_bars,
                      "n_no_wall_side": n_no_walls, "n_failed": n_failed, "feature_version": fv})
        log.info("Smoke: %s", smoke)
        assessment = (f"{n_updated}/{len(targets)} rows filled; {n_no_bars} no-bar dates left NULL; "
                      f"{n_no_walls} without a wall on each side (containment NULL); ohlc_violations={smoke['ohlc_violations']}; "
                      f"contained_close rate {smoke['contained_close_rate_overall']['rate_close']} over n={smoke['contained_close_rate_overall']['n']}")
        update_run_smoke(conn, run_id, smoke, assessment)
        log.info("Self-assessment: %s", assessment)
        if n_failed:
            conn.execute("UPDATE bt_backfill_runs SET status='suspect' WHERE run_id=%s", (run_id,))


if __name__ == "__main__":
    main()
