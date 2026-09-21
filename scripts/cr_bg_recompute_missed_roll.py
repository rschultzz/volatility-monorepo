#!/usr/bin/env python3
"""CR-BG — recompute the bt_daily_outcomes rows contaminated by the missed ES U26→Z26 roll.

Six rows (ticker SPX, feature_version v0.6.0-openiv): trade_date 2026-09-14..18, whose
session_*_t0 columns were computed on dying-contract ES U26 bars, and 2026-09-08, whose
5-session horizon ended 2026-09-14 on ES U26. CR-BF repaired the bars (ES Z26); this
script recomputes the rows from the repaired bars.

Amendment A1 (specs/CR-BG-missed-roll-derived-recompute.md): the PK
(ticker, trade_date, feature_version) excludes `active`, so deactivate-and-reinsert is
impossible for same-date rows. The rows are UPDATEd in place:
  - full-row recompute: every computed column is overwritten, explicit NULL where the new
    computation has no value (incl. the CR-G / CR-I t1/t5/t15 and post-touch columns,
    which this script does not compute);
  - the complete old rows are archived as JSON in bt_backfill_runs.smoke_test_results
    before the write;
  - active / deactivated_at / deactivated_reason are not touched — the repair is recorded
    via backfill_run_id and computed_at only;
  - each UPDATE is guarded on the archived computed_at / session_close_t0, and the six
    run in one transaction: a row changed since the archive aborts the whole write.

Computation is the shared packages/shared/outcomes_runner path (same as
cr_b_backfill_outcomes.py and cr_aa_sweep_pending_outcomes.py).

Usage:
    apps/web/.venv/bin/python scripts/cr_bg_recompute_missed_roll.py --dry-run
    apps/web/.venv/bin/python scripts/cr_bg_recompute_missed_roll.py

Exit: 0 on success; 1 if a guard failed (nothing written) or the after-state check failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import decimal
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.outcomes_runner import (  # noqa: E402
    CONTAINMENT_COLUMNS,
    compute_outcome_for_date,
    compute_session_containment,
)
from scripts.cr_b_backfill_outcomes import _fetch_daily_bars, _fetch_landscape  # noqa: E402

CR_ID = "CR-BG"
TICKER = "SPX"
FEATURE_VERSION = "v0.6.0-openiv"
TARGET_DATES = [
    dt.date(2026, 9, 8),
    dt.date(2026, 9, 14),
    dt.date(2026, 9, 15),
    dt.date(2026, 9, 16),
    dt.date(2026, 9, 17),
    dt.date(2026, 9, 18),
]

OUTCOME_COLUMNS = (
    "regime_kind_at_classification", "dominant_bucket_at_classification",
    "horizon_sessions", "horizon_end_date", "outcome_status",
    "reached_touch", "reached_close", "days_to_reach",
    "max_excursion_in_direction", "final_close_distance_from_target",
    "actual_realized_em_pct",
)
# CR-G / CR-I columns: not computed here, written as explicit NULL (A1 item 2).
NULLED_COLUMNS = tuple(
    [f"session_{f}_t{n}" for n in (1, 5, 15) for f in ("open", "high", "low", "close")]
    + [f"position_t{n}_post_touch" for n in (1, 5, 15)]
)
COMPUTED_COLUMNS = OUTCOME_COLUMNS + ("session_open_t0",) + CONTAINMENT_COLUMNS + NULLED_COLUMNS
UNTOUCHED_COLUMNS = ("ticker", "trade_date", "feature_version", "active", "deactivated_at", "deactivated_reason")

_SELECT_ROWS_SQL = """
    SELECT * FROM bt_daily_outcomes
    WHERE ticker = %s AND feature_version = %s AND trade_date = ANY(%s)
    ORDER BY trade_date
"""

_FEATURES_SQL = """
    SELECT trade_date, regime_at_classification, feature_vector
    FROM bt_daily_features
    WHERE ticker = %s AND feature_version = %s AND active = TRUE AND trade_date = ANY(%s)
"""

_UPDATE_SQL = (
    "UPDATE bt_daily_outcomes SET "
    + ", ".join(f"{c} = %s" for c in COMPUTED_COLUMNS)
    + ", backfill_run_id = %s, computed_at = NOW() "
    "WHERE ticker = %s AND trade_date = %s AND feature_version = %s "
    "AND computed_at IS NOT DISTINCT FROM %s "
    "AND session_close_t0 IS NOT DISTINCT FROM %s"
)


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    if isinstance(v, (uuid.UUID,)):
        return str(v)
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


def fetch_rows(conn) -> dict[dt.date, dict]:
    cur = conn.execute(_SELECT_ROWS_SQL, (TICKER, FEATURE_VERSION, TARGET_DATES))
    cols = [d.name for d in cur.description]
    return {r[cols.index("trade_date")]: dict(zip(cols, r)) for r in cur.fetchall()}


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        # session_* columns are REAL: compare at float32 resolution
        return abs(float(a) - float(b)) <= 1e-4 * max(1.0, abs(float(a)))
    return a == b


def diff_row(old: dict, new: dict) -> dict:
    """Every column whose value differs, old → new (all columns, not just session_*)."""
    return {c: {"old": _jsonable(old[c]), "new": _jsonable(new.get(c))}
            for c in old if not _same(old[c], new.get(c))}


def compute_new_values(conn, today: dt.date) -> dict[dt.date, dict]:
    cur = conn.execute(_FEATURES_SQL, (TICKER, FEATURE_VERSION, TARGET_DATES))
    features = {d: (regime, fv or {}) for d, regime, fv in cur.fetchall()}
    missing = [d for d in TARGET_DATES if d not in features]
    if missing:
        sys.exit(f"ERROR: no active {FEATURE_VERSION} feature row for {missing}")
    landscape_by_date = _fetch_landscape(conn, TICKER, TARGET_DATES)
    daily_bars = _fetch_daily_bars(conn, min(TARGET_DATES), max(TARGET_DATES) + dt.timedelta(days=90))

    out: dict[dt.date, dict] = {}
    for d in TARGET_DATES:
        assert d < today, f"{d} has not closed"
        regime, fv = features[d]
        landscape = landscape_by_date.get(d, {})
        outcome, session_open_t0 = compute_outcome_for_date(
            trade_date=d, regime=regime, feature_vector=fv,
            landscape=landscape, daily_bars=daily_bars,
        )
        containment = compute_session_containment(
            d, landscape.get("walls") or [], daily_bars, fv.get("implied_move_1d"),
        )
        vals = {c: outcome[c] for c in OUTCOME_COLUMNS}
        vals["session_open_t0"] = session_open_t0
        vals.update({c: containment[c] for c in CONTAINMENT_COLUMNS})
        vals.update({c: None for c in NULLED_COLUMNS})
        out[d] = vals
    return out


def print_diffs(old_rows: dict, new_rows: dict) -> None:
    for d in TARGET_DATES:
        o, n = old_rows[d], new_rows[d]
        print(f"\n  {d}  status {o['outcome_status']} → {n['outcome_status']}")
        print(f"    session_open_t0  {o['session_open_t0']} → {n['session_open_t0']}"
              + (f"   (Δ {n['session_open_t0'] - o['session_open_t0']:+.2f})"
                 if o['session_open_t0'] is not None and n['session_open_t0'] is not None else ""))
        print(f"    session_close_t0 {o['session_close_t0']} → {n['session_close_t0']}")
        print(f"    reached_touch {o['reached_touch']} → {n['reached_touch']}   "
              f"reached_close {o['reached_close']} → {n['reached_close']}")
        for c, ch in diff_row(o, n).items():
            if c not in ("session_open_t0", "session_close_t0", "outcome_status", "reached_touch", "reached_close"):
                print(f"    {c}: {ch['old']} → {ch['new']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CR-BG missed-roll outcome recompute (in-place UPDATE)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the old-row archive JSON and the old→new diff; write nothing, no run row")
    args = ap.parse_args(argv)
    assert FEATURE_VERSION == CANONICAL_FEATURE_VERSION, (FEATURE_VERSION, CANONICAL_FEATURE_VERSION)

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    from zoneinfo import ZoneInfo
    today = dt.datetime.now(ZoneInfo("America/Los_Angeles")).date()

    old_rows = fetch_rows(conn)
    if sorted(old_rows) != TARGET_DATES:
        sys.exit(f"ERROR: expected rows for {TARGET_DATES}, found {sorted(old_rows)}")
    archive = [{k: _jsonable(v) for k, v in old_rows[d].items()} for d in TARGET_DATES]
    new_vals = compute_new_values(conn, today)

    print(f"=== {CR_ID} missed-roll outcome recompute ===  dry_run={args.dry_run}")
    if args.dry_run:
        print("\n--- archive of the 6 old rows (JSON) ---")
        print(json.dumps(archive, indent=1))
        print("\n--- old → planned new (all differing columns) ---")
        print_diffs(old_rows, {d: {**old_rows[d], **new_vals[d]} for d in TARGET_DATES})
        print("\n[DRY RUN — nothing written, no run row]")
        conn.close()
        return 0

    ok = True
    with backfill_run(conn, CR_ID) as run_id:
        print(f"Run ID: {run_id}")
        smoke: dict = {"archived_old_rows": archive, "target_dates": [str(d) for d in TARGET_DATES]}
        update_run_smoke(conn, run_id, smoke, "archive of 6 old rows written; UPDATE pending")

        with conn.transaction():
            for d in TARGET_DATES:
                old = old_rows[d]
                cur = conn.execute(
                    _UPDATE_SQL,
                    tuple(new_vals[d][c] for c in COMPUTED_COLUMNS)
                    + (run_id, TICKER, d, FEATURE_VERSION, old["computed_at"], old["session_close_t0"]),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(f"guard failed for {d}: row changed since archive (rowcount={cur.rowcount}); "
                                       "transaction rolled back, nothing written")
        update_run_progress(conn, run_id, 0)  # UPDATE-only run: no rows inserted

        after = fetch_rows(conn)
        print_diffs(old_rows, after)
        problems = []
        for d in TARGET_DATES:
            for c in COMPUTED_COLUMNS:
                if not _same(after[d][c], new_vals[d][c]):
                    problems.append(f"{d}.{c}: stored {after[d][c]!r} != computed {new_vals[d][c]!r}")
            for c in UNTOUCHED_COLUMNS:
                if after[d][c] != old_rows[d][c]:
                    problems.append(f"{d}.{c}: changed {old_rows[d][c]!r} → {after[d][c]!r}")
            if str(after[d]["backfill_run_id"]) != run_id:
                problems.append(f"{d}: backfill_run_id not stamped")
        ok = not problems
        smoke.update({
            "rows_updated": len(TARGET_DATES),
            "diff_all_columns": {str(d): diff_row(old_rows[d], after[d]) for d in TARGET_DATES},
            "new_rows": [{k: _jsonable(v) for k, v in after[d].items()} for d in TARGET_DATES],
            "after_state_problems": problems,
        })
        opens = {str(d): (after[d]["session_open_t0"] or 0) - (old_rows[d]["session_open_t0"] or 0)
                 for d in TARGET_DATES if old_rows[d]["session_open_t0"] is not None}
        update_run_smoke(
            conn, run_id, smoke,
            f"CR-BG: 6 rows UPDATEd in place from ES Z26 bars; session_open_t0 deltas {opens}; "
            f"after-state {'OK' if ok else 'FAIL: ' + '; '.join(problems)}",
        )
    print(f"\nafter-state check: {'OK' if ok else 'FAIL'}")
    for p in ([] if ok else problems):
        print("  " + p)
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
