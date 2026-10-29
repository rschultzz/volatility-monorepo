#!/usr/bin/env python3
"""CR-DO Step 0 (read-only) — which derived rows did CR-BZ's fragment-bar repair change, and can
the code that wrote them recompute them unchanged?

CR-BZ (run 88139854, 2026-10-25) repaired 783 fragment minutes (2026-01-02 → 10-22) in
ironbeam_es_1m_bars / es_minutes. For every active row of the three ES-based versions with
trade_date ≥ 2025-09-01 (earliest that can reach a 2026 bar over a 60-session horizon), and every
non-null vol_risk_premium of 2026, this script computes the row with the code that wrote it
(CR-DL's writers: scripts/cr_dl_step0_dryrun.py) on two bar states:

  pre   CR-BZ undone: earliest old_* per minute from ironbeam_es_1m_bars_repair_log for the
        CR-BZ run, except minutes CR-DJ later rewrote (they keep CR-DJ's values, which CR-DL
        already carried into the derived rows);
  post  the bars now.

Reproduction: rows computed before CR-BZ (or stamped by CR-DL, which wrote only CR-DJ columns)
must equal pre; rows computed after CR-BZ must equal post. The CR-BZ effect is pre → post on the
columns a row stores (stored NULL = never filled by its writer; left as stored, CR-DL rule).

B4 regimes (row-level):
  train    trade_date ≤ 2026-06-05 and the row computes identically with every bar read capped
           at 2026-06-05 (both states) — i.e. it reads no holdout bar;
  holdout  trade_date in 2026-06-06 → 10-26, or a train-dated row that reads a holdout bar;
  rb       trade_date in 2025;  fwd  trade_date ≥ 2026-10-27.
Train rows are reported with values. Holdout, R-b and forward rows are BLIND: only row counts,
column names, per-column change counts and md5 checksums leave this process (stdout and JSON);
their values are never printed or written. md5 keys / before / after cover the planned write
(CR-BZ columns that reproduce); md5 all_post covers every filled column of the group. Run it twice (--tag run1 / run2) and compare the
checksums.

Nothing is written to the DB: read-only transaction, no run row. DB errors print the type only.

Usage:
    apps/web/.venv/bin/python scripts/cr_do_step0_dryrun.py --tag run1
Writes scripts/logs/cr_do/step0_<tag>.json (train values + blind counts / checksums).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.outcomes_runner import compute_outcome_for_date, compute_session_containment  # noqa: E402
from packages.shared.probability import classify_post_touch_positions  # noqa: E402
from packages.shared.sessions import fetch_es_daily_bars  # noqa: E402
from scripts.cr_b_backfill_outcomes import _fetch_landscape  # noqa: E402
from scripts.cr_bi_backfill_nyse_sessions import compute_row as cr_bi_compute_row  # noqa: E402
from scripts.cr_d_backfill_vol_features import _compute_vol_features  # noqa: E402
from scripts.cr_dl_step0_dryrun import (  # noqa: E402
    BAR_WINDOW_CALENDAR_DAYS, OUTCOME_COLUMNS, OWNED, POS_COLUMNS, RTH_END_UTC, RTH_START_UTC, TIMEFRAMES, TN_COLUMNS,
    _compute_session_ohlc, _fetch_rth_minute_bars, _implied_move, _j, _rewrite, _rth_daily_closes, _same,
)

TICKER = "SPX"
CRBZ_RUN = "88139854-3605-4d94-ab03-7bc34af5da5c"
CRDJ_RUN = "2adcc833-5f7f-4406-b645-0ce1b7d61bbc"
CRDL_RUN = "9531e3de-8a14-464d-a3fc-cc25a3dce7ef"
TRAIN_END = dt.date(2026, 6, 5)
HOLDOUT_END = dt.date(2026, 10, 26)
RB_FROM, RB_TO = dt.date(2025, 1, 1), dt.date(2025, 12, 31)
ROWS_FROM = dt.date(2025, 9, 1)
VRP_FROM = dt.date(2026, 1, 1)
LOG_DIR = REPO_ROOT / "scripts" / "logs" / "cr_do"
LABELS = ("outcome_status", "reached_touch", "reached_close", "days_to_reach",
          "max_excursion_in_direction", "horizon_sessions", "horizon_end_date") + POS_COLUMNS

# Bars with CR-BZ undone; minutes CR-DJ rewrote afterwards keep their current (CR-DJ) values.
PRE_SQL = f"""(
    WITH bz AS (
        SELECT DISTINCT ON (bar_ts) bar_ts, old_open, old_high, old_low, old_close, old_volume
        FROM ironbeam_es_1m_bars_repair_log
        WHERE target_table = 'ironbeam_es_1m_bars' AND run_id = '{CRBZ_RUN}'
          AND bar_ts NOT IN (SELECT bar_ts FROM ironbeam_es_1m_bars_repair_log
                             WHERE target_table = 'ironbeam_es_1m_bars' AND run_id = '{CRDJ_RUN}')
        ORDER BY bar_ts, logged_at
    )
    SELECT b.datetime,
           COALESCE(o.old_open, b.open) AS open, COALESCE(o.old_high, b.high) AS high,
           COALESCE(o.old_low, b.low) AS low, COALESCE(o.old_close, b.close) AS close,
           COALESCE(o.old_volume, b.volume) AS volume
    FROM ironbeam_es_1m_bars b
    LEFT JOIN bz o ON o.bar_ts = b.datetime AT TIME ZONE 'UTC'
    WHERE b.datetime >= '2025-07-01'   -- every read below starts after this
) ironbeam_es_1m_bars"""


class _PreCursor:
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        return self._cur.execute(_rewrite(sql, PRE_SQL), params)

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()


class PreConn:
    """The writers' own code, reading ironbeam_es_1m_bars as it was before CR-BZ."""
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        return self._conn.execute(_rewrite(sql, PRE_SQL), params)

    def cursor(self, *a, **k):
        return _PreCursor(self._conn.cursor(*a, **k))


# ── writers (CR-DL's, with the bar-read cap as a parameter instead of a fixed TRAIN_END) ──────
def compute_legacy(conn, version, rows, features, landscape, stored, cap: Optional[dt.date]) -> dict:
    """v0.6.0-openiv / v0.5.0-rebuilt writers on the pre-CR-BI frames (cr_dl_step0_dryrun.compute_legacy)."""
    dates = list(rows)
    bar_to = max(dates) + dt.timedelta(days=BAR_WINDOW_CALENDAR_DAYS)
    bar_to = min(bar_to, cap) if cap else bar_to
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
            day = minutes_all[minutes_all["datetime"].dt.date == d]
            row["session_open_t0"] = float(day["open"].iloc[0]) if not day.empty else None
        if version == "v0.6.0-openiv":
            row.update(compute_session_containment(d, walls, daily, (fv or {}).get("implied_move_1d")))
        end = d + dt.timedelta(days=BAR_WINDOW_CALENDAR_DAYS)
        lo = pd.Timestamp(dt.datetime.combine(d, RTH_START_UTC, tzinfo=dt.timezone.utc))
        hi = pd.Timestamp(dt.datetime.combine(end, RTH_END_UTC, tzinfo=dt.timezone.utc)) + pd.Timedelta(minutes=1)
        mins = minutes_all[(minutes_all["datetime"] >= lo) & (minutes_all["datetime"] < hi)]
        row.update({c: None for c in TN_COLUMNS})
        if stored[d]["outcome_status"] in ("computed", "na_regime"):
            row.update(_compute_session_ohlc(mins, d))
        row.update({c: None for c in POS_COLUMNS})
        target, im = pick_drift_target(walls), _implied_move(fv)
        if row["reached_touch"] and row["days_to_reach"] is not None and target is not None and im:
            pos = classify_post_touch_positions(
                days_to_reach=row["days_to_reach"], horizon_bars=_rth_daily_closes(mins),
                drift_target=target, tolerance=0.25 * im, timeframes_sessions=TIMEFRAMES)
            row.update({f"position_t{n}_post_touch": pos.get(n) for n in TIMEFRAMES})
        out[d] = row
    return out


def compute_nyse(conn, rows, features, landscape, cap: Optional[dt.date]) -> dict:
    """v0.6.1-nyse-sessions writer: CR-BI compute_row on the shared session frame."""
    dates = list(rows)
    bar_to = max(dates) + dt.timedelta(days=90)
    bars = fetch_es_daily_bars(conn, min(dates), min(bar_to, cap) if cap else bar_to)
    return {d: cr_bi_compute_row(d, features[d][0], features[d][1] or {}, landscape.get(d, {}), bars)
            for d in dates}


def compute(conn, version, rows, feats, landscape, stored, cap=None) -> dict:
    if version == "v0.6.1-nyse-sessions":
        return compute_nyse(conn, rows, feats, landscape, cap)
    return compute_legacy(conn, version, rows, feats, landscape, stored, cap)


# ── blind summaries ───────────────────────────────────────────────────────────────────────────
def _canon(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NULL"
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, (dt.date, dt.datetime, str)):
        return str(_j(v))
    try:
        return f"{float(v):.4f}"   # stored session_* / *_over_im are float4: compare at 4 dp
    except (TypeError, ValueError):
        return str(v)


def md5(lines) -> str:
    return hashlib.md5("\n".join(sorted(lines)).encode()).hexdigest()


def _differs(a, b) -> bool:
    return not _same(a, b, 1e-6) and _canon(a) != _canon(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run label (run1 / run2) for the checksum comparison")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    try:
        conn = get_backfill_db_conn()
        assert_role_or_die(conn)
    except Exception as e:  # never echo the connection string
        print(f"DB connect failed: {type(e).__name__}")
        return 1
    conn.read_only = True
    pre_conn = PreConn(conn)
    crbz_done = conn.execute("SELECT completed_at FROM bt_backfill_runs WHERE run_id = %s", (CRBZ_RUN,)).fetchone()[0]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "tag": args.tag,
                    "crbz_completed_at": _j(crbz_done), "versions": {}}

    for version, cols in OWNED.items():
        cur = conn.execute(
            "SELECT o.*, (o.computed_at < %s::timestamptz OR o.backfill_run_id = %s::uuid) AS expect_pre "
            "FROM bt_daily_outcomes o WHERE ticker=%s AND feature_version=%s AND active "
            "AND trade_date >= %s ORDER BY trade_date",
            (crbz_done, CRDL_RUN, TICKER, version, ROWS_FROM))
        names = [c.name for c in cur.description]
        stored = {r[names.index("trade_date")]: dict(zip(names, r)) for r in cur.fetchall()}
        feats = {d: (reg, fv) for d, reg, fv in conn.execute(
            "SELECT trade_date, regime_at_classification, feature_vector FROM bt_daily_features "
            "WHERE ticker=%s AND feature_version=%s AND active AND trade_date = ANY(%s)",
            (TICKER, version, list(stored))).fetchall()}
        assert not [d for d in stored if d not in feats], version
        landscape = _fetch_landscape(conn, TICKER, list(stored))
        comp = {"pre": compute(pre_conn, version, stored, feats, landscape, stored),
                "post": compute(conn, version, stored, feats, landscape, stored)}
        train_dated = {d: s for d, s in stored.items() if d <= TRAIN_END}
        capped = {"pre": compute(pre_conn, version, train_dated, feats, landscape, stored, TRAIN_END),
                  "post": compute(conn, version, train_dated, feats, landscape, stored, TRAIN_END)}

        def regime(d) -> str:
            if RB_FROM <= d <= RB_TO:
                return "rb"
            if d > HOLDOUT_END:
                return "fwd"
            if d > TRAIN_END:
                return "holdout"
            reads_holdout = any(_differs(comp[k][d][c], capped[k][d][c]) for k in comp for c in cols)
            return "holdout" if reads_holdout else "train"

        train_rows: dict = {}
        blind: dict = {g: {"rows": 0, "repro_fail_rows": 0, "repro_fail_cols": Counter(), "changed_rows": 0,
                           "changed_cols": Counter(), "label_changes": Counter(), "crdl_rows": 0,
                           "crdl_rows_only_crbz_cols": 0, "planned_rows": 0, "planned_cols": Counter(),
                           "excluded_by_repro": 0, "keys": [], "before": [], "after": [], "all_post": []}
                       for g in ("holdout", "rb", "fwd")}
        for d, s in stored.items():
            g = regime(d)
            want = comp["pre"][d] if s["expect_pre"] else comp["post"][d]
            # stored NULL = never filled by the writer (CR-DL rule): not compared, not written
            filled = [c for c in cols if s[c] is not None]
            repro_fail = {c: {"stored": _j(s[c]), "recomputed": _j(want[c])}
                          for c in filled if _differs(s[c], want[c])}
            crbz = ({c: {"from": _j(s[c]), "to": _j(comp["post"][d][c])}
                     for c in filled if _differs(comp["pre"][d][c], comp["post"][d][c])}
                    if s["expect_pre"] else {})
            # write plan: CR-BZ columns that reproduce; a label that doesn't reproduce leaves the row as stored
            planned = [] if set(repro_fail) & set(LABELS) else [c for c in crbz if c not in repro_fail]
            is_crdl = str(s["backfill_run_id"]) == CRDL_RUN
            # CR-DL rows: stored vs post must differ only in CR-BZ columns
            crdl_extra = sorted(set(c for c in filled if _differs(s[c], comp["post"][d][c])) - set(crbz)) if is_crdl else []
            if g == "train":
                train_rows[str(d)] = {"status": s["outcome_status"], "computed_at": _j(s["computed_at"]),
                                      "expect": "pre" if s["expect_pre"] else "post", "crdl_row": is_crdl,
                                      "repro_fail": repro_fail, "crbz": crbz, "planned": planned,
                                      "never_filled": [c for c in cols if s[c] is None],
                                      "crdl_extra_diff": crdl_extra}
                continue
            b = blind[g]
            b["rows"] += 1
            b["repro_fail_rows"] += bool(repro_fail)
            b["repro_fail_cols"].update(list(repro_fail))
            b["changed_rows"] += bool(crbz)
            b["changed_cols"].update(list(crbz))
            b["label_changes"].update(c for c in crbz if c in LABELS)
            if is_crdl:
                b["crdl_rows"] += 1
                b["crdl_rows_only_crbz_cols"] += not crdl_extra
            b["planned_rows"] += bool(planned)
            b["planned_cols"].update(planned)
            b["excluded_by_repro"] += len(crbz) - len(planned)
            for c in planned:
                b["keys"].append(f"{version}|{d}|{c}")
                b["before"].append(f"{version}|{d}|{c}|{_canon(s[c])}")
                b["after"].append(f"{version}|{d}|{c}|{_canon(comp['post'][d][c])}")
            b["all_post"].extend(f"{version}|{d}|{c}|{_canon(comp['post'][d][c])}" for c in filled)

        out_blind = {g: {"rows": b["rows"], "repro_fail_rows": b["repro_fail_rows"],
                         "repro_fail_cols": dict(b["repro_fail_cols"]), "changed_rows": b["changed_rows"],
                         "changed_cols": dict(b["changed_cols"]), "label_changes": dict(b["label_changes"]),
                         "crdl_rows": b["crdl_rows"], "crdl_rows_only_crbz_cols": b["crdl_rows_only_crbz_cols"],
                         "planned_rows": b["planned_rows"], "planned_cols": dict(b["planned_cols"]),
                         "excluded_by_repro": b["excluded_by_repro"], "n_planned_values": len(b["keys"]), "md5_keys": md5(b["keys"]),
                         "md5_before": md5(b["before"]), "md5_after": md5(b["after"]),
                         "md5_all_post": md5(b["all_post"])}
                     for g, b in blind.items()}
        tr = train_rows.values()
        v = {"rows_checked": len(stored),
             "train": {"rows": len(train_rows),
                       "repro_fail_rows": sum(bool(r["repro_fail"]) for r in tr),
                       "changed_rows": sum(bool(r["crbz"]) for r in tr),
                       "changed_cols": dict(Counter(c for r in tr for c in r["crbz"])),
                       "planned_rows": sum(bool(r["planned"]) for r in tr),
                       "planned_values": sum(len(r["planned"]) for r in tr),
                       "label_changes": dict(Counter(c for r in tr for c in r["crbz"] if c in LABELS)),
                       "rows_detail": train_rows},
             **out_blind}
        report["versions"][version] = v
        print(f"{version}: {len(stored)} rows from {ROWS_FROM}")
        t = v["train"]
        print(f"  train   rows {t['rows']}; reproduction fails {t['repro_fail_rows']}; CR-BZ changes "
              f"{t['changed_rows']} rows; planned {t['planned_rows']} rows / {t['planned_values']} values; "
              f"columns {t['changed_cols']}; label changes {t['label_changes']}")
        for g in ("holdout", "rb", "fwd"):
            o = out_blind[g]
            print(f"  {g:7} rows {o['rows']}; reproduction fails {o['repro_fail_rows']} {o['repro_fail_cols']}; "
                  f"CR-BZ changes {o['changed_rows']} rows; planned {o['planned_rows']} rows / {o['n_planned_values']} values "
                  f"(excluded by reproduction {o['excluded_by_repro']}); columns "
                  f"{o['changed_cols']}; label changes {o['label_changes']}; CR-DL rows {o['crdl_rows']} "
                  f"(only CR-BZ cols {o['crdl_rows_only_crbz_cols']}); md5 keys {o['md5_keys'][:12]} "
                  f"before {o['md5_before'][:12]} after {o['md5_after'][:12]} all_post {o['md5_all_post'][:12]}",
                  flush=True)
        (LOG_DIR / f"step0_{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))

    # vol_risk_premium (v0.5.0-rebuilt only; every non-null 2026 row is ≤ 2026-05-22, train).
    # Fresh connection: the outcome passes run long enough for the server to drop the first one.
    conn.close()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    conn.read_only = True
    pre_conn = PreConn(conn)
    vrows = conn.execute(
        "SELECT trade_date, vol_risk_premium FROM bt_daily_features WHERE ticker=%s AND "
        "feature_version='v0.5.0-rebuilt' AND active AND vol_risk_premium IS NOT NULL "
        "AND trade_date >= %s ORDER BY trade_date", (TICKER, VRP_FROM)).fetchall()
    assert all(d <= TRAIN_END for d, _ in vrows)
    vrp = {"rows_checked": len(vrows), "rows": {}}
    for d, stored_v in vrows:
        x = {k: _compute_vol_features(c, d, TICKER)["vol_risk_premium"] for k, c in (("pre", pre_conn), ("post", conn))}
        vrp["rows"][str(d)] = {"stored": stored_v, **x, "repro_ok": _same(stored_v, x["pre"], 1e-9),
                               "crbz": not _same(x["pre"], x["post"], 1e-9)}
    report["vol_risk_premium"] = vrp
    r = vrp["rows"].values()
    print(f"vol_risk_premium: {len(vrows)} rows {VRP_FROM} → {TRAIN_END}; reproduction fails "
          f"{sum(not x['repro_ok'] for x in r)}; CR-BZ changes {sum(x['crbz'] for x in r)}")

    (LOG_DIR / f"step0_{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {LOG_DIR / f'step0_{args.tag}.json'}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
