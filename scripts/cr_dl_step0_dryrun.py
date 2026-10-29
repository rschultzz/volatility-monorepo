#!/usr/bin/env python3
"""CR-DL Step 0 (read-only) — which derived rows did the CR-DJ bar patch change, and can the
code that wrote them recompute them unchanged?

CR-DJ (run 2adcc833) replaced 2026-03-15 22:00 → 03-20 20:59 UTC in ironbeam_es_1m_bars /
es_minutes (ES H26 → Databento ES M26). For every active row that could read those bars, this
script computes the row twice with the code that wrote it:

  pre   on the bars as they were when the row was computed — the current table with every
        logged repair undone (earliest old_* per minute from ironbeam_es_1m_bars_repair_log:
        CR-BZ 2026-10-25 and CR-DJ 2026-10-29) and CR-DJ's 124 inserted minutes removed;
  mid   with only CR-DJ undone (CR-BZ's fragment repair kept);
  post  on the bars as they are now.

pre == stored is the reproduction check (the code is the writer); mid → post is the CR-DJ
effect, pre → mid the CR-BZ effect (outside this CR's kickoff, reported separately). Nothing is written: read-only transaction, no run row.

Writers, per feature_version (Step 0a):
  v0.6.1-nyse-sessions  CR-BI compute_row (current shared session frame: PT window, NYSE days)
  v0.6.0-openiv         outcomes_runner + CR-AQ containment on the pre-CR-BI daily frame
                        (same RTH SQL, bar-present days); CR-G / CR-I columns on their pre-CR-BI
                        fixed 13:30–20:00 UTC frames (verbatim from 5531453^)
  v0.5.0-rebuilt        as v0.6.0-openiv with its own features; containment never filled
  v0.6.0-openiv-spxcash SPX cash frame (packages/shared/spx_cash) — reads no ES bar; not run
  bt_daily_features.vol_risk_premium (v0.5.0-rebuilt only): CR-D _compute_vol_features

B4: every bar read is capped at 2026-06-05 (train end); candidate rows are trade_date ≤ 03-20.

Usage:
    apps/web/.venv/bin/python scripts/cr_dl_step0_dryrun.py
Writes scripts/logs/cr_dl/step0_dryrun.json and prints the summary.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.outcomes_runner import (  # noqa: E402
    CONTAINMENT_COLUMNS, compute_outcome_for_date, compute_session_containment,
)
from packages.shared.probability import classify_post_touch_positions  # noqa: E402
from packages.shared.sessions import fetch_es_daily_bars  # noqa: E402
from scripts.cr_b_backfill_outcomes import _fetch_landscape  # noqa: E402
from scripts.cr_bi_backfill_nyse_sessions import compute_row as cr_bi_compute_row  # noqa: E402
from scripts.cr_d_backfill_vol_features import _compute_vol_features  # noqa: E402

TICKER = "SPX"
CRDJ_RUN = "2adcc833-5f7f-4406-b645-0ce1b7d61bbc"
TRAIN_END = dt.date(2026, 6, 5)
ROWS_FROM, ROWS_TO = dt.date(2025, 11, 1), dt.date(2026, 3, 20)
VRP_FROM, VRP_TO = dt.date(2026, 2, 1), dt.date(2026, 5, 22)
BAR_WINDOW_CALENDAR_DAYS = 90
TIMEFRAMES = (1, 5, 15)
LOG_DIR = REPO_ROOT / "scripts" / "logs" / "cr_dl"

OUTCOME_COLUMNS = (
    "regime_kind_at_classification", "dominant_bucket_at_classification",
    "horizon_sessions", "horizon_end_date", "outcome_status",
    "reached_touch", "reached_close", "days_to_reach",
    "max_excursion_in_direction", "final_close_distance_from_target", "actual_realized_em_pct",
)
TN_COLUMNS = tuple(f"session_{k}_t{n}" for n in TIMEFRAMES for k in ("open", "high", "low", "close"))
POS_COLUMNS = tuple(f"position_t{n}_post_touch" for n in TIMEFRAMES)
OWNED = {   # columns each version's writers filled (Step 0a); the rest stay as stored
    "v0.6.1-nyse-sessions": OUTCOME_COLUMNS + ("session_open_t0",) + CONTAINMENT_COLUMNS + TN_COLUMNS + POS_COLUMNS,
    "v0.6.0-openiv": OUTCOME_COLUMNS + ("session_open_t0",) + CONTAINMENT_COLUMNS + TN_COLUMNS + POS_COLUMNS,
    "v0.5.0-rebuilt": OUTCOME_COLUMNS + ("session_open_t0",) + TN_COLUMNS + POS_COLUMNS,
}

# Bars as they were before the logged repairs of {runs}: earliest old_* per minute; CR-DJ inserts removed.
OVERLAY_SQL = """(
    WITH first_old AS (
        SELECT DISTINCT ON (bar_ts) bar_ts, old_open, old_high, old_low, old_close, old_volume
        FROM ironbeam_es_1m_bars_repair_log
        WHERE target_table = 'ironbeam_es_1m_bars' AND run_id = ANY(ARRAY[{runs}]::uuid[])
        ORDER BY bar_ts, logged_at
    )
    SELECT b.datetime,
           COALESCE(o.old_open, b.open) AS open, COALESCE(o.old_high, b.high) AS high,
           COALESCE(o.old_low, b.low) AS low, COALESCE(o.old_close, b.close) AS close,
           COALESCE(o.old_volume, b.volume) AS volume
    FROM ironbeam_es_1m_bars b
    LEFT JOIN first_old o ON o.bar_ts = b.datetime AT TIME ZONE 'UTC'
    WHERE b.datetime >= '2025-09-01' AND b.datetime < '2026-06-06'   -- every read below is inside this
      AND NOT EXISTS (
        SELECT 1 FROM ironbeam_es_1m_bars_repair_log l
        WHERE l.run_id = '{crdj}' AND l.target_table = 'ironbeam_es_1m_bars'
          AND l.bar_ts = b.datetime AT TIME ZONE 'UTC' AND l.old_close IS NULL)
) ironbeam_es_1m_bars"""
CRBZ_RUN = "88139854-3605-4d94-ab03-7bc34af5da5c"


def _overlay(runs) -> str:
    return OVERLAY_SQL.format(runs=", ".join(f"'{r}'" for r in runs), crdj=CRDJ_RUN)


def _rewrite(sql: str, overlay: str) -> str:
    out = sql.replace("FROM ironbeam_es_1m_bars\n", "FROM @PRE@\n").replace("FROM ironbeam_es_1m_bars ", "FROM @PRE@ ")
    assert "ironbeam_es_1m_bars" not in out, sql
    return out.replace("@PRE@", overlay)


class _PreCursor:
    def __init__(self, cur, overlay):
        self._cur, self._overlay = cur, overlay

    def execute(self, sql, params=None):
        return self._cur.execute(_rewrite(sql, self._overlay), params)

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()


class PreConn:
    """The writers' own code, reading ironbeam_es_1m_bars as it was before the repairs."""
    def __init__(self, conn, runs):
        self._conn, self._overlay = conn, _overlay(runs)

    def execute(self, sql, params=None):
        return self._conn.execute(_rewrite(sql, self._overlay), params)

    def cursor(self, *a, **k):
        return _PreCursor(self._conn.cursor(*a, **k), self._overlay)


# ── pre-CR-BI frames, verbatim from 5531453^ (CR-G / CR-I scripts) ──────────────────────────────
RTH_START_UTC = dt.time(13, 30)
RTH_END_UTC = dt.time(20, 0)


def _fetch_rth_minute_bars(conn, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    start_ts = dt.datetime.combine(start_date, RTH_START_UTC, tzinfo=dt.timezone.utc)
    end_ts = dt.datetime.combine(end_date, RTH_END_UTC, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=1)
    rows = conn.execute(
        """
        SELECT datetime, open, high, low, close
        FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime < %s
        ORDER BY datetime
        """,
        (start_ts, end_ts),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["session_date", "datetime", "open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    hm = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df = df[(hm >= 13 * 60 + 30) & (hm < 20 * 60)].copy()
    df["session_date"] = df["datetime"].dt.date.astype(str)
    return df


def _compute_session_ohlc(bars: pd.DataFrame, trade_date: dt.date) -> dict[str, Optional[float]]:
    none = {c: None for c in TN_COLUMNS}
    if bars.empty:
        return none
    session_dates = sorted(bars["session_date"].unique())
    try:
        origin_idx = session_dates.index(trade_date.isoformat())
    except ValueError:
        return none
    result = {}
    for tf in TIMEFRAMES:
        target_idx = origin_idx + tf
        if target_idx >= len(session_dates):
            result.update({f"session_{c}_t{tf}": None for c in ("open", "high", "low", "close")})
            continue
        sb = bars[bars["session_date"] == session_dates[target_idx]]
        result[f"session_open_t{tf}"] = float(sb["open"].iloc[0])
        result[f"session_high_t{tf}"] = float(sb["high"].max())
        result[f"session_low_t{tf}"] = float(sb["low"].min())
        result[f"session_close_t{tf}"] = float(sb["close"].iloc[-1])
    return result


def _rth_daily_closes(minutes: pd.DataFrame) -> pd.DataFrame:
    """Old CR-I _fetch_rth_daily_bars on the same minute frame: last close per UTC date."""
    if minutes.empty:
        return pd.DataFrame(columns=["trade_date", "close"])
    df = minutes.rename(columns={"session_date": "trade_date"})
    return df.groupby("trade_date")["close"].last().reset_index()


# ── per-version computation ──────────────────────────────────────────────────────────────────
def _implied_move(fv: dict):
    try:
        im = float((fv or {}).get("implied_move_1d"))
    except (TypeError, ValueError):
        return None
    return im if im > 0 else None


def compute_legacy(conn, version, rows, features, landscape, stored) -> dict:
    """v0.6.0-openiv / v0.5.0-rebuilt writers on the pre-CR-BI frames."""
    dates = [r for r in rows]
    bar_to = min(max(dates) + dt.timedelta(days=BAR_WINDOW_CALENDAR_DAYS), TRAIN_END)
    daily = fetch_es_daily_bars(conn, min(dates), bar_to, trading_days_only=False)
    minutes_all = _fetch_rth_minute_bars(conn, min(dates), bar_to)
    out = {}
    for d in dates:
        regime, fv = features[d]
        walls = (landscape.get(d) or {}).get("walls") or []
        outcome, open_t0 = compute_outcome_for_date(d, regime, fv or {}, landscape.get(d, {}), daily)
        row = {c: outcome[c] for c in OUTCOME_COLUMNS}
        row["session_open_t0"] = open_t0
        if version == "v0.5.0-rebuilt":
            # CR-G Step 2.5a / 2.5a-bis wrote this version's open: first bar in [13:30, 20:00) UTC on d
            day = minutes_all[minutes_all["datetime"].dt.date == d]
            row["session_open_t0"] = float(day["open"].iloc[0]) if not day.empty else None
        if version == "v0.6.0-openiv":
            row.update(compute_session_containment(d, walls, daily, (fv or {}).get("implied_move_1d")))
        end = d + dt.timedelta(days=BAR_WINDOW_CALENDAR_DAYS)
        lo = pd.Timestamp(dt.datetime.combine(d, RTH_START_UTC, tzinfo=dt.timezone.utc))
        hi = pd.Timestamp(dt.datetime.combine(end, RTH_END_UTC, tzinfo=dt.timezone.utc)) + pd.Timedelta(minutes=1)
        mins = minutes_all[(minutes_all["datetime"] >= lo) & (minutes_all["datetime"] < hi)]
        # CR-G scope: computed + na_regime rows (stored status — the row the script saw)
        row.update({c: None for c in TN_COLUMNS})
        if stored[d]["outcome_status"] in ("computed", "na_regime"):
            row.update(_compute_session_ohlc(mins, d))
        # CR-I scope: reached_touch rows, target + tolerance from the trade-date landscape / features
        row.update({c: None for c in POS_COLUMNS})
        target, im = pick_drift_target(walls), _implied_move(fv)
        if row["reached_touch"] and row["days_to_reach"] is not None and target is not None and im:
            pos = classify_post_touch_positions(
                days_to_reach=row["days_to_reach"], horizon_bars=_rth_daily_closes(mins),
                drift_target=target, tolerance=0.25 * im, timeframes_sessions=TIMEFRAMES)
            row.update({f"position_t{n}_post_touch": pos.get(n) for n in TIMEFRAMES})
        out[d] = row
    return out


def compute_nyse(conn, rows, features, landscape) -> dict:
    """v0.6.1-nyse-sessions writer: CR-BI compute_row on the shared session frame."""
    dates = list(rows)
    bars = fetch_es_daily_bars(conn, min(dates), min(max(dates) + dt.timedelta(days=90), TRAIN_END))
    return {d: cr_bi_compute_row(d, features[d][0], features[d][1] or {}, landscape.get(d, {}), bars)
            for d in dates}


def _same(a, b, tol=0.013) -> bool:
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return b is None or (isinstance(b, float) and pd.isna(b))
    if b is None or (isinstance(b, float) and pd.isna(b)):
        return False
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, (str, dt.date)):
        return a == b
    return abs(float(a) - float(b)) <= tol


def _j(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def _diff(cols, a: dict, b: dict) -> dict:
    return {c: {"from": _j(a[c]), "to": _j(b[c])} for c in cols if not _same(a[c], b[c])}


def main() -> int:
    logging.disable(logging.WARNING)
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    conn.read_only = True
    states = {   # bars as when the rows were computed / with CR-BZ only / now
        "pre": PreConn(conn, [CRBZ_RUN, CRDJ_RUN]),
        "mid": PreConn(conn, [CRDJ_RUN]),
        "post": conn,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "versions": {}}

    for version, cols in OWNED.items():
        cur = conn.execute(
            "SELECT * FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s AND active "
            "AND trade_date BETWEEN %s AND %s ORDER BY trade_date", (TICKER, version, ROWS_FROM, ROWS_TO))
        names = [c.name for c in cur.description]
        stored = {r[names.index("trade_date")]: dict(zip(names, r)) for r in cur.fetchall()}
        assert all(s["horizon_end_date"] is None or s["horizon_end_date"] <= TRAIN_END for s in stored.values())
        feats = {d: (reg, fv) for d, reg, fv in conn.execute(
            "SELECT trade_date, regime_at_classification, feature_vector FROM bt_daily_features "
            "WHERE ticker=%s AND feature_version=%s AND active AND trade_date = ANY(%s)",
            (TICKER, version, list(stored))).fetchall()}
        missing = [d for d in stored if d not in feats]
        assert not missing, (version, missing)
        landscape = _fetch_landscape(conn, TICKER, list(stored))
        comp = {}
        for k, c in states.items():
            comp[k] = (compute_nyse(c, stored, feats, landscape) if version == "v0.6.1-nyse-sessions"
                       else compute_legacy(c, version, stored, feats, landscape, stored))
        rows = {}
        for d, s in stored.items():
            rows[str(d)] = {
                "status": s["outcome_status"],
                "repro_fail": _diff(cols, s, comp["pre"][d]),
                "crbz": _diff(cols, comp["pre"][d], comp["mid"][d]),
                "crdj": _diff(cols, comp["mid"][d], comp["post"][d]),
                "stored_to_post": _diff(cols, s, comp["post"][d]),
            }
        v = {"rows_checked": len(stored), "rows": rows,
             "repro_fail": sorted(d for d, r in rows.items() if r["repro_fail"]),
             "crdj_rows": sorted(d for d, r in rows.items() if r["crdj"]),
             "crbz_rows": sorted(d for d, r in rows.items() if r["crbz"])}
        report["versions"][version] = v
        print(f"{version}: {len(stored)} rows {min(stored)} → {max(stored)}; reproduction fails "
              f"{len(v['repro_fail'])}; CR-DJ changes {len(v['crdj_rows'])}; CR-BZ changes {len(v['crbz_rows'])}",
              flush=True)

    # vol_risk_premium (v0.5.0-rebuilt only)
    vrows = conn.execute(
        "SELECT trade_date, vol_risk_premium FROM bt_daily_features WHERE ticker=%s AND "
        "feature_version='v0.5.0-rebuilt' AND active AND vol_risk_premium IS NOT NULL "
        "AND trade_date BETWEEN %s AND %s ORDER BY trade_date", (TICKER, VRP_FROM, VRP_TO)).fetchall()
    vrp = {"rows_checked": len(vrows), "rows": {}}
    for d, stored_v in vrows:
        x = {k: _compute_vol_features(c, d, TICKER)["vol_risk_premium"] for k, c in states.items()}
        vrp["rows"][str(d)] = {"stored": stored_v, **x, "repro_ok": _same(stored_v, x["pre"], 1e-9),
                               "crbz": not _same(x["pre"], x["mid"], 1e-9),
                               "crdj": not _same(x["mid"], x["post"], 1e-9)}
    report["vol_risk_premium"] = vrp
    r = vrp["rows"].values()
    print(f"vol_risk_premium: {len(vrows)} rows {VRP_FROM} → {VRP_TO}; reproduction fails "
          f"{sum(not x['repro_ok'] for x in r)}; CR-DJ changes {sum(x['crdj'] for x in r)}; "
          f"CR-BZ changes {sum(x['crbz'] for x in r)}")

    (LOG_DIR / "step0_dryrun.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {LOG_DIR / 'step0_dryrun.json'}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
