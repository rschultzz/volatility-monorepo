#!/usr/bin/env python3
"""CR-BI — backfill feature_version 'v0.6.1-nyse-sessions' (INSERT-only, backfill protocol).

What changes vs v0.6.0-openiv (spec Amendment A1): sessions come from the NYSE calendar
(packages/shared/trading_calendar) instead of ES bar presence, and the t1/t5/t15 session OHLC +
post-touch positions are computed on the same PT session window as t0 (packages/shared/sessions)
instead of CR-G / CR-I's fixed 13:30–20:00 UTC window. Outcomes stay ES-vs-B: same ES bars, same
targets, same regimes, same tolerance.

  1. bt_daily_features: verbatim copy of every ACTIVE v0.6.0-openiv row under the new version
     (one constant versions both tables; every outcome↔feature join is on feature_version).
  2. bt_daily_outcomes: every active v0.6.0-openiv row recomputed through the shared
     outcomes_runner path + sessions.session_ohlc_at / post_touch_positions.

Nothing existing is updated or deleted. The canonical constant is promoted in a separate commit.

Smoke (spec A1.2): t0 columns identical on every row; touch / close / days_to_reach / status
identical on every row whose old horizon contained no non-NYSE "session"; every other
difference listed with its cause.

Usage:
    apps/web/.venv/bin/python scripts/cr_bi_backfill_nyse_sessions.py --dry-run
    apps/web/.venv/bin/python scripts/cr_bi_backfill_nyse_sessions.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.outcomes_runner import (  # noqa: E402
    CONTAINMENT_COLUMNS, compute_outcome_for_date, compute_session_containment,
)
from packages.shared.sessions import (  # noqa: E402
    TIMEFRAMES, fetch_es_daily_bars, post_touch_positions, session_ohlc_at,
)
from packages.shared.trading_calendar import is_trading_day  # noqa: E402
from scripts.cr_b_backfill_outcomes import _fetch_landscape  # noqa: E402

CR_ID = "CR-BI"
TICKER = "SPX"
SOURCE_VERSION = "v0.6.0-openiv"
NEW_VERSION = "v0.6.1-nyse-sessions"
LOG_DIR = REPO_ROOT / "scripts" / "logs"

OUTCOME_COLUMNS = (
    "regime_kind_at_classification", "dominant_bucket_at_classification",
    "horizon_sessions", "horizon_end_date", "outcome_status",
    "reached_touch", "reached_close", "days_to_reach",
    "max_excursion_in_direction", "final_close_distance_from_target", "actual_realized_em_pct",
)
TN_COLUMNS = tuple(f"session_{k}_t{n}" for n in TIMEFRAMES for k in ("open", "high", "low", "close"))
POS_COLUMNS = tuple(f"position_t{n}_post_touch" for n in TIMEFRAMES)
ROW_COLUMNS = OUTCOME_COLUMNS + ("session_open_t0",) + CONTAINMENT_COLUMNS + TN_COLUMNS + POS_COLUMNS
INSERT_COLUMNS = ("ticker", "trade_date", "feature_version") + ROW_COLUMNS + ("backfill_run_id",)

_INSERT_OUTCOME_SQL = (
    f"INSERT INTO bt_daily_outcomes ({', '.join(INSERT_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(INSERT_COLUMNS))}) "
    "ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING"
)

# verbatim: every value column copied; only the version, run id and computed_at differ
_COPY_FEATURES_SQL = """
    INSERT INTO bt_daily_features
        (trade_date, ticker, feature_vector, feature_version, feature_config_hash,
         regime_at_classification, backfill_run_id,
         atm_iv_percentile, skew_percentile, term_structure_slope, smile_convexity, vol_risk_premium)
    SELECT trade_date, ticker, feature_vector, %s, feature_config_hash,
           regime_at_classification, %s,
           atm_iv_percentile, skew_percentile, term_structure_slope, smile_convexity, vol_risk_premium
    FROM bt_daily_features
    WHERE ticker = %s AND feature_version = %s AND active = TRUE
    ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING
"""

_SOURCE_ROWS_SQL = f"""
    SELECT o.trade_date, f.regime_at_classification, f.feature_vector,
           {', '.join('o.' + c for c in ROW_COLUMNS)}
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
    fv = fv or {}
    walls = landscape.get("walls") or []
    outcome, open_t0 = compute_outcome_for_date(trade_date, regime, fv, landscape, bars)
    row = {**outcome, "session_open_t0": open_t0}
    row.update(compute_session_containment(trade_date, walls, bars, fv.get("implied_move_1d")))
    # CR-G scope (computed + na_regime) and CR-I scope (touched rows), on the shared session frame
    row.update({c: None for c in TN_COLUMNS + POS_COLUMNS})
    if outcome["outcome_status"] in ("computed", "na_regime"):
        row.update(session_ohlc_at(bars, trade_date))
    im = _implied_move(fv)
    if outcome.get("reached_touch") and im:
        pos = post_touch_positions(bars, trade_date, outcome.get("days_to_reach"),
                                   pick_drift_target(walls), 0.25 * im)
        row.update({f"position_t{n}_post_touch": pos.get(n) for n in TIMEFRAMES})
    return row


def _same(a, b, tol=0.013) -> bool:
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return b is None or (isinstance(b, float) and pd.isna(b))
    if b is None or (isinstance(b, float) and pd.isna(b)):
        return False
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, (str, dt.date)):
        return a == b
    return abs(float(a) - float(b)) <= tol          # REAL columns round-trip at ~1e-3 relative


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    src = conn.execute(_SOURCE_ROWS_SQL, (TICKER, SOURCE_VERSION)).fetchall()
    dates = [r[0] for r in src]
    old = {r[0]: dict(zip(ROW_COLUMNS, r[3:])) for r in src}
    existing = conn.execute("SELECT count(*) FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s",
                            (TICKER, NEW_VERSION)).fetchone()[0]
    print(f"=== {CR_ID} → {NEW_VERSION} ===\nsource: {len(src)} active {SOURCE_VERSION} rows {dates[0]} → {dates[-1]}; "
          f"existing {NEW_VERSION} outcome rows: {existing}")

    landscape = _fetch_landscape(conn, TICKER, dates)
    bar_from, bar_to = min(dates), max(dates) + dt.timedelta(days=90)
    bars = fetch_es_daily_bars(conn, bar_from, bar_to)
    bars_all = fetch_es_daily_bars(conn, bar_from, bar_to, trading_days_only=False)
    non_nyse = sorted(set(bars_all.index) - set(bars.index))
    all_sessions = sorted(bars_all.index)
    print(f"ES RTH-window dates {len(bars_all)} → NYSE sessions {len(bars)}; dropped {len(non_nyse)}: "
          f"{', '.join(map(str, non_nyse))}")

    new = {d: compute_row(d, regime, fv, landscape.get(d, {}), bars) for d, regime, fv, *_ in src}

    # ── smoke vs the source version ──────────────────────────────────────────
    def old_window_has_non_nyse(d: dt.date, n_extra: int = 0) -> bool:
        hz = old[d]["horizon_sessions"]
        if not hz:
            return False
        fwd = [x for x in all_sessions if x >= d]      # compute_outcome's `forward` (trade_date may have no bars)
        return any(not is_trading_day(x) for x in fwd[:int(hz) + n_extra])

    t0_cols = ("session_open_t0",) + CONTAINMENT_COLUMNS
    label_cols = ("outcome_status", "reached_touch", "reached_close", "days_to_reach", "horizon_end_date",
                  "max_excursion_in_direction", "final_close_distance_from_target", "actual_realized_em_pct")
    # a source NULL that is now filled is a null-fill the sweep had not landed yet, not a change
    t0_diff = [d for d in dates if any(old[d][c] is not None and not _same(old[d][c], new[d][c]) for c in t0_cols)]
    t0_filled = [d for d in dates if any(old[d][c] is None and new[d][c] is not None for c in t0_cols)]
    lab_diff = [d for d in dates if any(not _same(old[d][c], new[d][c]) for c in label_cols)]
    lab_unexplained = [d for d in lab_diff if not old_window_has_non_nyse(d)]
    flips = [d for d in lab_diff if old[d]["outcome_status"] == "computed" and new[d]["outcome_status"] == "computed"
             and (old[d]["reached_touch"] != new[d]["reached_touch"] or old[d]["reached_close"] != new[d]["reached_close"])]
    status_moves = Counter((old[d]["outcome_status"], new[d]["outcome_status"]) for d in dates
                           if old[d]["outcome_status"] != new[d]["outcome_status"])
    tn = {}
    for n in TIMEFRAMES:
        c = f"session_close_t{n}"
        both = [d for d in dates if old[d][c] is not None and new[d][c] is not None]
        delta = pd.Series([float(new[d][c]) - float(old[d][c]) for d in both], index=both)
        tn[n] = {"old_nonnull": sum(old[d][c] is not None for d in dates), "new_nonnull": sum(new[d][c] is not None for d in dates),
                 "changed": int((delta.abs() > 0.013).sum()), "median_abs": round(float(delta.abs().median()), 2),
                 "p90_abs": round(float(delta.abs().quantile(.9)), 2), "max_abs": round(float(delta.abs().max()), 2)}
    pos = {n: {"old_nonnull": sum(old[d][f"position_t{n}_post_touch"] is not None for d in dates),
               "new_nonnull": sum(new[d][f"position_t{n}_post_touch"] is not None for d in dates),
               "changed_where_both": sum(1 for d in dates if old[d][f"position_t{n}_post_touch"] is not None
                                         and new[d][f"position_t{n}_post_touch"] is not None
                                         and old[d][f"position_t{n}_post_touch"] != new[d][f"position_t{n}_post_touch"])}
           for n in TIMEFRAMES}

    print(f"\nstatus old: {dict(Counter(old[d]['outcome_status'] for d in dates))}")
    print(f"status new: {dict(Counter(new[d]['outcome_status'] for d in dates))}   transitions: {dict(status_moves)}")
    print(f"t0 / containment columns differing: {len(t0_diff)} rows {t0_diff[:10]}; "
          f"source NULL → filled: {len(t0_filled)} rows {[str(x) for x in t0_filled]}")
    print(f"label columns differing: {len(lab_diff)} rows; of those with NO non-NYSE session in the old horizon "
          f"(must be 0): {len(lab_unexplained)} {lab_unexplained[:10]}")
    print(f"touch/close flips (computed in both): {len(flips)}")
    LOG_DIR.mkdir(exist_ok=True)
    rows_out = []
    for d in lab_diff:
        rows_out.append({"trade_date": d, **{f"{c}_old": old[d][c] for c in label_cols}, **{f"{c}_new": new[d][c] for c in label_cols}})
        if d in flips:
            print(f"  FLIP {d} hz={old[d]['horizon_sessions']} touch {old[d]['reached_touch']}→{new[d]['reached_touch']} "
                  f"close {old[d]['reached_close']}→{new[d]['reached_close']} end {old[d]['horizon_end_date']}→{new[d]['horizon_end_date']}")
    pd.DataFrame(rows_out).to_csv(LOG_DIR / "cr_bi_nyse_sessions_label_diffs.csv", index=False)
    print(f"T+N session close, old (UTC window, bar-present sessions) → new: {tn}")
    print(f"post-touch positions: {pos}")
    gate_ok = not t0_diff and not lab_unexplained
    print(f"SMOKE GATE: {'PASS' if gate_ok else 'FAIL'}")

    if args.dry_run or not gate_ok:
        print("\n[DRY RUN — nothing written]" if args.dry_run else "\nGate failed — nothing written.")
        conn.close()
        return 0 if gate_ok else 1

    with backfill_run(conn, CR_ID) as run_id:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(_COPY_FEATURES_SQL, (NEW_VERSION, run_id, TICKER, SOURCE_VERSION))
                n_feat = cur.rowcount
                n_out = 0
                for d in dates:
                    cur.execute(_INSERT_OUTCOME_SQL,
                                tuple([TICKER, d, NEW_VERSION] + [new[d][c] for c in ROW_COLUMNS] + [run_id]))
                    n_out += cur.rowcount
        update_run_progress(conn, run_id, n_feat + n_out)
        chk = conn.execute(
            """SELECT (SELECT count(*) FROM bt_daily_features WHERE ticker=%(t)s AND feature_version=%(n)s),
                      (SELECT count(*) FROM bt_daily_outcomes WHERE ticker=%(t)s AND feature_version=%(n)s),
                      (SELECT count(*) FROM bt_daily_features a JOIN bt_daily_features b USING (ticker, trade_date)
                        WHERE a.ticker=%(t)s AND a.feature_version=%(s)s AND a.active AND b.feature_version=%(n)s
                          AND a.feature_vector = b.feature_vector
                          AND a.regime_at_classification IS NOT DISTINCT FROM b.regime_at_classification
                          AND a.feature_config_hash IS NOT DISTINCT FROM b.feature_config_hash),
                      (SELECT count(*) FROM bt_daily_outcomes WHERE ticker=%(t)s AND feature_version=%(s)s),
                      (SELECT max(computed_at) FROM bt_daily_outcomes WHERE ticker=%(t)s AND feature_version=%(s)s)""",
            {"t": TICKER, "n": NEW_VERSION, "s": SOURCE_VERSION}).fetchone()
        ok = n_out == len(dates) - existing and chk[1] == len(dates) and chk[0] == chk[2] == len(dates)
        smoke = {"new_version": NEW_VERSION, "source_rows": len(dates), "features_copied": n_feat, "outcomes_inserted": n_out,
                 "features_total": chk[0], "outcomes_total": chk[1], "features_identical_to_source": chk[2],
                 "source_outcomes_total": chk[3], "source_max_computed_at": str(chk[4]),
                 "non_nyse_dates_dropped": [str(x) for x in non_nyse], "t0_rows_differing": len(t0_diff), "t0_rows_null_filled": [str(x) for x in t0_filled],
                 "label_rows_differing": len(lab_diff), "label_rows_unexplained": len(lab_unexplained),
                 "touch_close_flips": [str(x) for x in flips], "status_transitions": {f"{a}->{b}": n for (a, b), n in status_moves.items()},
                 "tN_close": tn, "post_touch": pos}
        update_run_smoke(conn, run_id, smoke,
                         "OK: features verbatim; t0 identical; label diffs all explained by non-NYSE sessions" if ok
                         else "CHECK: counts do not reconcile")
        print(f"\nrun {run_id}: features {n_feat}, outcomes {n_out}; check {chk}")
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
