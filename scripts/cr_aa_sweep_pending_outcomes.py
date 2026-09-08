#!/usr/bin/env python3
"""CR-AA — Pending outcome sweep: promote matured pending_history rows to computed/na_data.

Finds pending_history rows in bt_daily_outcomes whose horizon window has closed
(enough RTH sessions now exist in ironbeam_es_1m_bars), re-runs compute_outcome
via the shared helper, and UPDATEs each row to its terminal status.

Design constraints:
  - Promotes pending_history ONLY (WHERE outcome_status = 'pending_history').
    Never touches computed, na_regime, or na_data.
  - Horizon-gated: only rows where the Nth session on/after trade_date exists
    and is <= the latest fully-closed RTH session in ironbeam_es_1m_bars.
    (horizon_end_date is NULL for pending rows; maturity is computed Python-side
    from the actual RTH session list — see Step-0 finding in CR-036 spec.)
  - Uses get_backfill_db_conn() + assert_role_or_die + backfill_run scaffolding.
  - UPDATE guard: WHERE outcome_status = 'pending_history' prevents clobbering
    a row that changed between SELECT and UPDATE.
  - Idempotent: a second immediate run promotes 0 rows.

Usage:
    python scripts/cr_aa_sweep_pending_outcomes.py
    python scripts/cr_aa_sweep_pending_outcomes.py --dry-run
    python scripts/cr_aa_sweep_pending_outcomes.py --limit 5
    python scripts/cr_aa_sweep_pending_outcomes.py --from-date 2026-03-01

CR-AU decision 2 (amendment A2): after the promotion run, a matured-trade
capture scans every post-split (> --split-date, default 2026-06-05)
magnet-above date with outcome_status = 'computed', plans the harness's debit
pair (payload drift target, snap_vertical_pair 'debit', expiry = 15 business
days), and fetches into orats_options_minute the windows that are due and not
yet covered in orats_options_fetched_windows:
  - settlement: expiry day 12:50–13:00 PT, once expiry <= the latest closed RTH session
  - touch: [touch_pt, touch_pt + 90 min] when detect_touch is rth_touch / gap_touch
Own run row (cr_id DAILY-CAPTURE-MATURE), created only when something is fetched.
Needs ORATS_API_KEY; without it the capture is skipped with a warning and the
sweep behaves exactly as before. --dry-run prints the windows without fetching.

Exit: 0 on success; 1 if any row failed or a capture fetch raised (non-404).
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
from packages.shared.buckets import bucket_sessions
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.options_cache.models import TimeRange
from packages.shared.options_cache.windows import find_gaps
from packages.shared.outcomes_runner import (
    CONTAINMENT_COLUMNS,
    compute_outcome_for_date,
    compute_session_containment,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Follow the canonical so openiv pending rows are swept after the Step-5 flip.
# NOTE: 12 v0.5.0-rebuilt pending_history rows existed as of 2026-06-06; run
# a one-off sweep pinned to v0.5.0-rebuilt before merging to avoid orphaning them,
# or accept that they remain pending (harmless — they never feed live proposals).
FEATURE_VERSION = CANONICAL_FEATURE_VERSION

_PENDING_ROWS_SQL = """
    SELECT o.trade_date,
           o.regime_kind_at_classification,
           o.dominant_bucket_at_classification,
           f.feature_vector
    FROM bt_daily_outcomes o
    JOIN bt_daily_features f
        ON f.ticker          = o.ticker
       AND f.trade_date      = o.trade_date
       AND f.feature_version = o.feature_version
    WHERE o.ticker          = %s
      AND o.feature_version = %s
      AND o.outcome_status  = 'pending_history'
    ORDER BY o.trade_date
"""

_LANDSCAPE_SQL = """
    SELECT trade_date, walls, table_spot
    FROM orats_gex_landscape
    WHERE ticker = %s
      AND trade_date = ANY(%s)
"""

_RTH_BARS_SQL = """
    WITH rth AS (
        SELECT
            (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date AS session_date,
            open, high, low, close,
            ROW_NUMBER() OVER (
                PARTITION BY
                    (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                ORDER BY datetime ASC
            ) AS rn_asc,
            ROW_NUMBER() OVER (
                PARTITION BY
                    (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                ORDER BY datetime DESC
            ) AS rn_desc
        FROM ironbeam_es_1m_bars
        WHERE
            (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
                BETWEEN '06:30:00' AND '13:00:00'
          AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                BETWEEN %s AND %s
    )
    SELECT
        session_date,
        MAX(CASE WHEN rn_asc  = 1 THEN open  END) AS open,
        MAX(high)                                   AS high,
        MIN(low)                                    AS low,
        MAX(CASE WHEN rn_desc = 1 THEN close END)  AS close
    FROM rth
    GROUP BY session_date
    ORDER BY session_date
"""

_RTH_SESSION_DATES_SQL = """
    SELECT DISTINCT (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
              BETWEEN '06:30:00' AND '13:00:00'
    ORDER BY 1
"""

# UPDATE guard: WHERE outcome_status = 'pending_history' prevents clobbering
# a row that changed between our SELECT and this UPDATE.
_UPDATE_OUTCOME_SQL = """
    UPDATE bt_daily_outcomes
    SET outcome_status                    = %s,
        regime_kind_at_classification     = %s,
        dominant_bucket_at_classification = %s,
        horizon_sessions                  = %s,
        horizon_end_date                  = %s,
        reached_touch                     = %s,
        reached_close                     = %s,
        days_to_reach                     = %s,
        max_excursion_in_direction        = %s,
        final_close_distance_from_target  = %s,
        actual_realized_em_pct            = %s,
        session_open_t0                   = %s,
        backfill_run_id                   = %s,
        computed_at                       = NOW()
    WHERE ticker          = %s
      AND trade_date      = %s
      AND feature_version = %s
      AND outcome_status  = 'pending_history'
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


def _fetch_session_dates(conn) -> list[dt.date]:
    """Return sorted list of all distinct RTH session dates in ironbeam_es_1m_bars."""
    rows = conn.execute(_RTH_SESSION_DATES_SQL).fetchall()
    return sorted(r[0] for r in rows)


def _expected_horizon_end(
    trade_date: dt.date,
    bucket: str,
    session_dates: list[dt.date],
) -> Optional[dt.date]:
    """Return the Nth session on/after trade_date; None if not enough sessions exist."""
    try:
        n = bucket_sessions(bucket)
    except KeyError:
        return None
    forward = [d for d in session_dates if d >= trade_date]
    if len(forward) >= n:
        return forward[n - 1]
    return None


def _fetch_landscape(conn, ticker: str, dates: list[dt.date]) -> dict[dt.date, dict]:
    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_SQL, (ticker, dates))
        rows = cur.fetchall()
    result = {}
    for (d, walls, table_spot) in rows:
        result[d] = {
            "walls":      walls if isinstance(walls, list) else [],
            "table_spot": float(table_spot) if table_spot is not None else None,
        }
    return result


_CONTAINMENT_TARGETS_SQL = """
    SELECT o.trade_date, o.feature_version, f.feature_vector
    FROM bt_daily_outcomes o
    JOIN bt_daily_features f
      ON f.ticker = o.ticker AND f.trade_date = o.trade_date
     AND f.feature_version = o.feature_version AND f.active = TRUE
    WHERE o.ticker = %s AND o.feature_version = %s AND o.active = TRUE
      AND o.session_close_t0 IS NULL
      AND o.trade_date >= %s AND o.trade_date <= %s
    ORDER BY o.trade_date
"""

_UPDATE_CONTAINMENT_SQL = """
    UPDATE bt_daily_outcomes
    SET session_high_t0 = %s, session_low_t0 = %s, session_close_t0 = %s,
        wall_above_price = %s, wall_below_price = %s,
        contained_close = %s, contained_range = %s, close_pos_in_band = %s,
        range_over_im = %s, close_move_over_im = %s, breach_side = %s
    WHERE ticker = %s AND trade_date = %s AND feature_version = %s
      AND session_close_t0 IS NULL
"""


def fill_session_containment(conn, ticker: str, feature_version: str,
                             daily_bars: pd.DataFrame, landscape_by_date: dict,
                             bar_from: dt.date, bar_to: dt.date) -> dict:
    """CR-AQ null-fill: rows with session_close_t0 IS NULL whose session has closed.

    Runs after the promotion loop on the same daily_bars / landscape the sweep
    already loaded (bounded to [bar_from, bar_to]); this is what makes the
    containment columns fill automatically for dates the 13:40 UTC insert saw
    as a partial session. Rows whose trade_date has no bar row stay NULL and
    are retried on the next sweep.
    """
    with conn.cursor() as cur:
        cur.execute(_CONTAINMENT_TARGETS_SQL, (ticker, feature_version, bar_from, bar_to))
        rows = cur.fetchall()
    n_filled = n_no_bars = 0
    for trade_date, fv_version, fv in rows:
        if trade_date not in daily_bars.index:
            n_no_bars += 1
            continue
        c = compute_session_containment(
            trade_date, (landscape_by_date.get(trade_date) or {}).get("walls") or [], daily_bars,
            (fv or {}).get("implied_move_1d"),
        )
        with conn.cursor() as cur:
            cur.execute(_UPDATE_CONTAINMENT_SQL,
                        tuple(c[k] for k in CONTAINMENT_COLUMNS) + (ticker, trade_date, fv_version))
            n_filled += cur.rowcount
    return {"containment_targets": len(rows), "containment_filled": n_filled, "containment_no_bars": n_no_bars}


def _fetch_daily_bars(conn, bar_from: dt.date, bar_to: dt.date) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(_RTH_BARS_SQL, (bar_from, bar_to))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["session_date", "open", "high", "low", "close"])
    df = df.set_index("session_date")
    df.index = [d for d in df.index]
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_smoke(conn, ticker: str, n_promoted: int) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE outcome_status = 'computed'),
                   COUNT(*) FILTER (WHERE outcome_status = 'pending_history'),
                   COUNT(*) FILTER (WHERE outcome_status = 'na_regime'),
                   COUNT(*) FILTER (WHERE outcome_status = 'na_data'),
                   MIN(trade_date), MAX(trade_date)
            FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
            """,
            (ticker, FEATURE_VERSION),
        )
        total, n_computed, n_pending, n_na_regime, n_na_data, min_d, max_d = cur.fetchone()

    # Rows where horizon window is closed but outcome is still pending_history
    # (should be 0 after a clean sweep run).
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX((datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date)
            FROM ironbeam_es_1m_bars
            """
        )
        latest_session = cur.fetchone()[0]

    # Count remaining pending rows whose horizon is still open (not yet matured).
    # These are expected — they just haven't had enough sessions yet.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND outcome_status = 'pending_history'
            """,
            (ticker, FEATURE_VERSION),
        )
        remaining_pending = int(cur.fetchone()[0])

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND backfill_run_id IS NULL
            """,
            (ticker, FEATURE_VERSION),
        )
        null_run_id = int(cur.fetchone()[0])

    return {
        "latest_session":       str(latest_session) if latest_session else None,
        "outcomes_total":       int(total) if total else 0,
        "outcomes_computed":    int(n_computed or 0),
        "outcomes_pending":     int(n_pending or 0),
        "outcomes_na_regime":   int(n_na_regime or 0),
        "outcomes_na_data":     int(n_na_data or 0),
        "n_promoted":           n_promoted,
        "remaining_pending":    remaining_pending,
        "null_run_id_count":    null_run_id,
        "date_min":             str(min_d) if min_d else None,
        "date_max":             str(max_d) if max_d else None,
    }


# ── CR-AU decision 2: matured-trade capture (amendment A2) ───────────────────

MATURE_CR_ID = "DAILY-CAPTURE-MATURE"
DEFAULT_SPLIT_DATE = dt.date(2026, 6, 5)
TOUCH_WINDOW_MIN = 90            # get_touch_pos_val: touch_datetime_pt → + 90 minutes
SETTLE_START = dt.time(12, 50)
SETTLE_END = dt.time(13, 0)
OPRA_ROOT = "SPX"

_HOLDOUT_MAGNET_COMPUTED_SQL = """
    SELECT o.trade_date
    FROM bt_daily_outcomes o
    JOIN bt_daily_features f
      ON f.ticker = o.ticker AND f.trade_date = o.trade_date
     AND f.feature_version = o.feature_version AND f.active
    WHERE o.ticker = %s AND o.feature_version = %s AND o.active
      AND o.trade_date > %s
      AND f.regime_at_classification = 'magnet-above'
      AND o.outcome_status = 'computed'
    ORDER BY o.trade_date
"""


def capture_enabled(env: Optional[dict] = None) -> bool:
    """The capture needs the ORATS token; the sweep cron may not carry it yet (Step 0)."""
    env = os.environ if env is None else env
    return bool((env.get("ORATS_API_KEY") or "").strip())


def plan_matured_windows(trade_date: dt.date, expiry: dt.date, touch_resolution: Optional[str],
                         touch_pt: Optional[dt.datetime], latest_session: dt.date) -> list[dict]:
    """Pure: the windows the harness reads for a matured debit trade, each tagged
    due / deferred. Settlement = expiry 12:50–13:00 PT (get_settlement_price's
    window); touch = [touch_pt, +90 min] for rth_touch / gap_touch (get_touch_pos_val).
    A window is due when its last day is on/before the latest fully closed RTH session."""
    out = []
    s0 = dt.datetime.combine(expiry, SETTLE_START)
    s1 = dt.datetime.combine(expiry, SETTLE_END)
    out.append({"label": "settlement", "start": s0, "end": s1, "due": expiry <= latest_session})
    if touch_pt is not None and touch_resolution in ("rth_touch", "gap_touch"):
        t1 = touch_pt + dt.timedelta(minutes=TOUCH_WINDOW_MIN)
        out.append({"label": f"touch ({touch_resolution})", "start": touch_pt, "end": t1,
                    "due": t1.date() <= latest_session})
    return out


def windows_to_fetch(opras: list[str], windows: list[dict], existing: dict[str, list]) -> tuple[list[tuple[str, dict]], int]:
    """Pure dedupe: (opra, window) pairs that are due and not fully covered in
    orats_options_fetched_windows; also returns the count already covered."""
    todo, covered = [], 0
    for w in windows:
        if not w["due"]:
            continue
        req = TimeRange(start_pt=w["start"], end_pt=w["end"])
        for o in opras:
            if find_gaps(req, existing.get(o, [])):
                todo.append((o, w))
            else:
                covered += 1
    return todo, covered


def capture_matured_trades(conn, ticker: str, latest_session: dt.date, *, split_date: dt.date = DEFAULT_SPLIT_DATE,
                           dry_run: bool = False, cr_id: str = MATURE_CR_ID) -> dict:
    """Decision 2 as amended (A2). Returns a stats dict; never raises past a fetch."""
    stats: dict = {"enabled": capture_enabled(), "dates": 0, "dates_unlistable": 0, "windows_due": 0,
                   "windows_deferred": 0, "legs_covered": 0, "legs_to_fetch": 0, "legs_fetched": 0,
                   "legs_404": 0, "legs_exception": 0, "bars_written": 0, "run_id": None,
                   "orats_404_detail": [], "exception_detail": []}
    print("\n=== CR-AU matured-trade capture (DAILY-CAPTURE-MATURE) ===")
    if not stats["enabled"]:
        log.warning("ORATS_API_KEY not set — matured-trade capture skipped (promotion unaffected). "
                    "Add ORATS_API_KEY to the sweep cron's env.")
        stats["skipped"] = "no ORATS_API_KEY"
        return stats
    os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]     # options_cache reads DATABASE_URL
    from packages.shared.options_cache import repository as repo
    from packages.shared.options_cache.opra import format_opra
    from packages.shared.options_cache.strikes import StructureNotListed, snap_vertical_pair
    from scripts.cr_ah_step4_analysis import DTE_TARGET, detect_touch, nth_business_day
    from scripts.cr_am_holdout_leg_capture import _payload_target

    dates = [r[0] for r in conn.execute(_HOLDOUT_MAGNET_COMPUTED_SQL, (ticker, FEATURE_VERSION, split_date)).fetchall()]
    stats["dates"] = len(dates)
    print(f"post-split magnet-above computed dates: {len(dates)} (> {split_date}); latest closed session {latest_session}"
          f"{'  [DRY RUN]' if dry_run else ''}")
    plans = []
    for td in dates:
        target = _payload_target(conn, td)
        row = conn.execute("SELECT table_spot FROM orats_gex_landscape WHERE ticker=%s AND trade_date=%s", (ticker, td)).fetchone()
        spot = float(row[0]) if row and row[0] is not None else None
        expiry = nth_business_day(td, DTE_TARGET)
        if target is None:
            stats["dates_unlistable"] += 1
            print(f"  {td}: no drift_target — skipped")
            continue
        try:
            d_ = snap_vertical_pair(target, 10.0, "debit", expiry, td, conn, toward=spot)
        except StructureNotListed as exc:
            stats["dates_unlistable"] += 1
            print(f"  {td}: UNLISTABLE — {exc}")
            continue
        opras = [format_opra(OPRA_ROOT, expiry, "C", k) for k in (d_.other, d_.anchor)]
        res, touch_pt = detect_touch(conn, td, expiry, target)
        windows = plan_matured_windows(td, expiry, res, touch_pt, latest_session)
        existing = {o: repo.get_windows_for_contract(o) for o in opras}
        todo, covered = windows_to_fetch(opras, windows, existing)
        stats["windows_due"] += sum(1 for w in windows if w["due"])
        stats["windows_deferred"] += sum(1 for w in windows if not w["due"])
        stats["legs_covered"] += covered
        stats["legs_to_fetch"] += len(todo)
        plans.append((td, opras, todo))
        print(f"  {td} target={target:.2f} debit {d_.other:g}/{d_.anchor:g} (w {d_.width_actual:g}) expiry={expiry} touch={res}"
              f"{f' @ {touch_pt:%m-%d %H:%M} PT' if touch_pt else ''}  windows: "
              + ", ".join(f"{w['label']} {w['start']:%m-%d %H:%M}–{w['end']:%H:%M} {'due' if w['due'] else 'deferred'}" for w in windows)
              + f"  → fetch {len(todo)} leg-windows, {covered} covered")
    if dry_run or not any(todo for _, _, todo in plans):
        print(f"matured capture: {'dry-run — ' if dry_run else ''}nothing to fetch ({stats['legs_covered']} leg-windows already covered,"
              f" {stats['windows_deferred']} windows deferred); no run row.")
        return stats

    from packages.shared.options_cache.fetcher import fetch_option_bars
    from packages.shared.options_cache.http_client import OratsPermanentError
    with backfill_run(conn, cr_id) as run_id:
        stats["run_id"] = run_id
        print(f"Run ID: {run_id}")
        for td, opras, todo in plans:
            for opra, w in todo:
                try:
                    r = fetch_option_bars([opra], w["start"], w["end"], source="historical_backfill", record_empty_windows=True)
                except OratsPermanentError as exc:
                    stats["legs_404"] += 1
                    stats["orats_404_detail"].append(f"{td} {w['label']} {opra}: {exc}")
                    print(f"    {td} {w['label']} {opra}: ORATS 4xx — {exc}")
                    continue
                except Exception as exc:                    # noqa: BLE001 — counted, reported, exit 1
                    stats["legs_exception"] += 1
                    stats["exception_detail"].append(f"{td} {w['label']} {opra}: {type(exc).__name__}: {exc}")
                    print(f"    {td} {w['label']} {opra}: EXCEPTION {type(exc).__name__}: {exc}")
                    continue
                stats["legs_fetched"] += 1
                stats["bars_written"] += r.bars_written
                print(f"    {td} {w['label']} {opra}: bars_written={r.bars_written} cache_hits={r.cache_hits}")
        summary = (f"{stats['dates']} dates; {stats['legs_fetched']} leg-windows fetched "
                   f"({stats['legs_covered']} covered, {stats['windows_deferred']} windows deferred, "
                   f"{stats['dates_unlistable']} unlistable); 404={stats['legs_404']} exceptions={stats['legs_exception']} "
                   f"bars_written={stats['bars_written']}; no P&L computed")
        update_run_smoke(conn, run_id, stats, summary)
        print(f"SUMMARY: {summary}")
    return stats


def _capture_then_exit(conn, ticker: str, latest_session: dt.date, dry_run: bool, code: int, split_date: dt.date) -> None:
    """Run the matured-trade capture after the promotion pass (whatever it did), then exit."""
    try:
        cap = capture_matured_trades(conn, ticker, latest_session, split_date=split_date, dry_run=dry_run)
    except Exception as exc:          # never let the capture change the promotion's outcome silently
        log.error("matured-trade capture failed: %s", exc, exc_info=True)
        cap = {"legs_exception": 1}
    conn.close()
    sys.exit(1 if (code or cap.get("legs_exception")) else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker",    default="SPX")
    ap.add_argument("--from-date", default=None, metavar="YYYY-MM-DD",
                    help="Only sweep pending rows on/after this trade_date.")
    ap.add_argument("--limit",     type=int, default=None, metavar="N",
                    help="Cap number of matured rows to promote (for testing).")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print what would be promoted without writing.")
    ap.add_argument("--split-date", default=DEFAULT_SPLIT_DATE.isoformat(), metavar="YYYY-MM-DD",
                    help="CR-AU: matured-trade capture scans magnet-above computed dates after this date.")
    args = ap.parse_args()
    split_date = dt.date.fromisoformat(args.split_date)

    ticker    = args.ticker
    from_date = dt.date.fromisoformat(args.from_date) if args.from_date else None

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    # Load all RTH session dates once (used for horizon-maturity check).
    session_dates = _fetch_session_dates(conn)
    if not session_dates:
        print("No RTH session dates found in ironbeam_es_1m_bars. Nothing to do.")
        conn.close()
        sys.exit(0)
    latest_session = session_dates[-1]

    # Fetch all pending_history rows (join with bt_daily_features for feature_vector).
    pending_rows = conn.execute(
        _PENDING_ROWS_SQL, (ticker, FEATURE_VERSION)
    ).fetchall()

    if from_date:
        pending_rows = [r for r in pending_rows if r[0] >= from_date]

    # Filter to matured rows: Nth session on/after trade_date exists and is <= latest_session.
    matured = []
    for trade_date, regime, bucket, fv in pending_rows:
        horizon_end = _expected_horizon_end(trade_date, bucket, session_dates)
        if horizon_end is not None and horizon_end <= latest_session:
            matured.append((trade_date, regime, bucket, fv))

    if args.limit:
        matured = matured[:args.limit]

    print("=== CR-AA Pending Outcome Sweep ===")
    print(f"Ticker: {ticker}   Feature version: {FEATURE_VERSION}")
    print(f"Latest RTH session: {latest_session}")
    print(f"Pending rows total: {len(pending_rows)}   Matured (window closed): {len(matured)}")
    if args.dry_run:
        print("[DRY RUN — no writes]")
    print()

    if not matured:
        print("No matured pending rows. Nothing to promote.")
        _capture_then_exit(conn, ticker, latest_session, args.dry_run, 0, split_date)

    dates = [r[0] for r in matured]

    if args.dry_run:
        for trade_date, regime, bucket, fv in matured:
            horizon_end = _expected_horizon_end(trade_date, bucket, session_dates)
            print(f"  [dry-run] {trade_date} regime={regime!r} bucket={bucket!r} "
                  f"horizon_end={horizon_end}")
        _capture_then_exit(conn, ticker, latest_session, True, 0, split_date)

    landscape_by_date = _fetch_landscape(conn, ticker, dates)

    # Load bars from min(trade_date) through today to cover all horizon windows.
    bar_from   = min(dates)
    bar_to     = latest_session
    daily_bars = _fetch_daily_bars(conn, bar_from, bar_to)
    log.info("Loaded %d RTH daily sessions (%s → %s)", len(daily_bars), bar_from, bar_to)

    n_promoted = 0
    n_skipped  = 0   # rows where compute_outcome still returned pending_history
    n_failed   = 0
    failed_dates: list[str] = []

    with backfill_run(conn, "CR-AA") as run_id:
        print(f"Run ID: {run_id}")

        for i, (trade_date, regime, bucket, fv) in enumerate(matured, 1):
            try:
                landscape = landscape_by_date.get(trade_date, {})
                outcome, session_open_t0 = compute_outcome_for_date(
                    trade_date     = trade_date,
                    regime         = regime,
                    feature_vector = fv or {},
                    landscape      = landscape,
                    daily_bars     = daily_bars,
                )

                new_status = outcome["outcome_status"]

                if new_status == "pending_history":
                    # horizon-gated filter said it was matured but compute_outcome
                    # disagrees — log and skip rather than leaving unchanged.
                    log.warning(
                        "SKIP %s: horizon-gated filter predicted matured but "
                        "compute_outcome returned pending_history (bars gap?)",
                        trade_date,
                    )
                    n_skipped += 1
                    continue

                with conn.cursor() as cur:
                    cur.execute(_UPDATE_OUTCOME_SQL, (
                        new_status,
                        outcome["regime_kind_at_classification"],
                        outcome["dominant_bucket_at_classification"],
                        outcome["horizon_sessions"],
                        outcome["horizon_end_date"],
                        outcome["reached_touch"],
                        outcome["reached_close"],
                        outcome["days_to_reach"],
                        outcome["max_excursion_in_direction"],
                        outcome["final_close_distance_from_target"],
                        outcome["actual_realized_em_pct"],
                        session_open_t0,
                        run_id,
                        ticker,
                        trade_date,
                        FEATURE_VERSION,
                    ))
                    if cur.rowcount == 1:
                        n_promoted += 1
                        log.info("PROMOTED %s: pending_history → %s", trade_date, new_status)
                    else:
                        # rowcount == 0: row was already promoted by a concurrent run
                        n_skipped += 1
                        log.info("SKIPPED %s: no longer pending_history (concurrent update?)", trade_date)

            except Exception as exc:
                n_failed += 1
                failed_dates.append(str(trade_date))
                log.error("ERROR %s: %s", trade_date, exc, exc_info=True)

            if i % 20 == 0 or i == len(matured):
                update_run_progress(conn, run_id, n_promoted)
                print(f"  [{i}/{len(matured)}] "
                      f"promoted={n_promoted} skipped={n_skipped} failed={n_failed}")

        # CR-AQ: fill session containment for closed sessions still NULL (bounded to the loaded bars)
        try:
            containment_stats = fill_session_containment(
                conn, ticker, FEATURE_VERSION, daily_bars, landscape_by_date, bar_from, bar_to,
            )
            print(f"  containment null-fill: {containment_stats}")
        except Exception as exc:   # never let the new pass break the promotion sweep
            log.error("containment null-fill failed: %s", exc, exc_info=True)
            containment_stats = {"containment_error": str(exc)}

        smoke = _run_smoke(conn, ticker, n_promoted)
        smoke.update(containment_stats)   # CR-AQ null-fill counters
        smoke.update({
            "n_matured":     len(matured),
            "n_skipped":     n_skipped,
            "n_failed":      n_failed,
            "failed_dates":  failed_dates,
        })
        assessment = (
            f"{n_promoted} rows promoted, {n_skipped} skipped, {n_failed} failed"
            + (f" — FAILURES: {', '.join(failed_dates)}" if failed_dates else " — clean run")
        )
        update_run_smoke(conn, run_id, smoke, assessment)

        print(f"\n=== DONE ===")
        print(f"promoted={n_promoted}  skipped={n_skipped}  failed={n_failed}")
        print(f"Latest RTH session:    {smoke['latest_session']}")
        print(f"Total outcome rows:    {smoke['outcomes_total']}")
        print(f"  computed={smoke['outcomes_computed']}  pending={smoke['outcomes_pending']}")
        print(f"  na_regime={smoke['outcomes_na_regime']}  na_data={smoke['outcomes_na_data']}")
        print(f"Remaining pending:     {smoke['remaining_pending']}")
        print(f"Null run_id count:     {smoke['null_run_id_count']}")

    # CR-AU decision 2: capture the matured trades' windows after the promotion run (own run row)
    _capture_then_exit(conn, ticker, latest_session, False, 1 if n_failed > 0 else 0, split_date)


if __name__ == "__main__":
    main()
