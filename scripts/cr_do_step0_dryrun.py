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

Amendment 1 (Ryan, 2026-10-29): every comparison is exact in the column's own type — `real` as
float4, `double precision` as float8, the rest as values; checksums hash the same canonical form.
D4: the 10 CR-DJ ratio values CR-DL's 0.013 tolerance hid (RESIDUAL) are written too; each must
equal, exactly, a recompute on bars with both CR-DJ and CR-BZ undone.

B4 regimes (row-level):
  train    trade_date ≤ 2026-06-05 and the row computes identically with every bar read capped
           at 2026-06-05 (both states) — i.e. it reads no holdout bar;
  holdout  trade_date in 2026-06-06 → 10-26, or a train-dated row that reads a holdout bar;
  rb       trade_date in 2025;  fwd  trade_date ≥ 2026-10-27.
Train rows are reported with values. Holdout, R-b and forward rows are BLIND: only row counts,
column names, per-column change counts and md5 checksums leave this process (stdout and JSON);
their values are never printed or written. Run it twice (--tag) and compare the checksums.

analyse() and measure() are shared with scripts/cr_do_recompute_derived.py (preflight and smoke).

Nothing is written to the DB: read-only transaction, no run row. DB errors print the type only.

Usage:
    apps/web/.venv/bin/python scripts/cr_do_step0_dryrun.py --tag run4
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

import numpy as np  # noqa: E402
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
    _compute_session_ohlc, _fetch_rth_minute_bars, _implied_move, _j, _rewrite, _rth_daily_closes,
)
from scripts.cr_dl_step0_dryrun import PreConn as CrDlPreConn  # noqa: E402

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
GROUPS = ("train", "holdout", "rb", "fwd")
BLIND = ("holdout", "rb", "fwd")
LABELS = ("outcome_status", "reached_touch", "reached_close", "days_to_reach",
          "max_excursion_in_direction", "horizon_sessions", "horizon_end_date") + POS_COLUMNS
FLIP_LABELS = ("outcome_status", "reached_touch", "reached_close", "days_to_reach")   # Ryan's "labels"
STAMP_COLUMNS = ("backfill_run_id", "computed_at")
# D4: CR-DJ changes to the t0 ratios that CR-DL's 0.013 tolerance hid (train, CR-DL rows)
RESIDUAL = {(v, dt.date(2026, 3, d)): cols
            for v in ("v0.6.1-nyse-sessions", "v0.6.0-openiv")
            for d, cols in ((16, ("range_over_im",)), (18, ("range_over_im", "close_move_over_im")),
                            (19, ("range_over_im", "close_move_over_im")))}

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


# ── exact, in-type comparison and canonical form (Amendment 1) ─────────────────────────────────
def column_types(conn, table="bt_daily_outcomes") -> dict:
    return dict(conn.execute("SELECT column_name, data_type FROM information_schema.columns "
                             "WHERE table_schema = 'public' AND table_name = %s", (table,)).fetchall())


def _is_null(v) -> bool:
    return v is None or v is pd.NaT or (isinstance(v, (float, np.floating)) and np.isnan(v))


def canon(v, typ: str) -> str:
    """The value as the column stores it: `real` → float4, `double precision` → float8."""
    if _is_null(v):
        return "NULL"
    if typ == "real":
        return repr(float(np.float32(float(v))))
    if typ == "double precision":
        return repr(float(v))
    if typ == "boolean":
        return "t" if bool(v) else "f"
    if typ in ("smallint", "integer", "bigint"):
        return str(int(v))
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if typ == "date" and isinstance(v, dt.datetime):
        v = v.date()
    return str(_j(v))


def md5(lines) -> str:
    return hashlib.md5("\n".join(sorted(lines)).encode()).hexdigest()


# ── per-version analysis (shared with the write script) ────────────────────────────────────────
def fetch_stored(conn, version) -> dict:
    cur = conn.execute(
        "SELECT * FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s AND active "
        "AND trade_date >= %s ORDER BY trade_date", (TICKER, version, ROWS_FROM))
    names = [c.name for c in cur.description]
    return {r[names.index("trade_date")]: dict(zip(names, r)) for r in cur.fetchall()}


def analyse(conn, version, types, crbz_done, stored=None, run_id: Optional[str] = None) -> dict:
    """Every row of `version` from ROWS_FROM: regime, stored, pre / post recompute and the plan.

    run_id=None (Step 0 / preflight): the plan is derived from reproduction.
    run_id set (smoke after the write): rows stamped with run_id are the written rows; the plan is
    re-derived from the bars alone — (CR-BZ columns ∪ RESIDUAL) where stored now equals post.
    """
    cols = OWNED[version]
    stored = stored if stored is not None else fetch_stored(conn, version)
    feats = {d: (reg, fv) for d, reg, fv in conn.execute(
        "SELECT trade_date, regime_at_classification, feature_vector FROM bt_daily_features "
        "WHERE ticker=%s AND feature_version=%s AND active AND trade_date = ANY(%s)",
        (TICKER, version, list(stored))).fetchall()}
    assert not [d for d in stored if d not in feats], version
    landscape = _fetch_landscape(conn, TICKER, list(stored))
    pre_conn = PreConn(conn)
    comp = {"pre": compute(pre_conn, version, stored, feats, landscape, stored),
            "post": compute(conn, version, stored, feats, landscape, stored)}
    train_dated = {d: s for d, s in stored.items() if d <= TRAIN_END}
    capped = {"pre": compute(pre_conn, version, train_dated, feats, landscape, stored, TRAIN_END),
              "post": compute(conn, version, train_dated, feats, landscape, stored, TRAIN_END)}
    resid_dates = {d: s for (v, d), _ in RESIDUAL.items() if v == version for s in [stored[d]]}
    pre_both = (compute(CrDlPreConn(conn, [CRDJ_RUN, CRBZ_RUN]), version, resid_dates, feats, landscape, stored)
                if resid_dates else {})

    def eq(c, a, b) -> bool:
        return canon(a, types[c]) == canon(b, types[c])

    def regime(d) -> str:
        if RB_FROM <= d <= RB_TO:
            return "rb"
        if d > HOLDOUT_END:
            return "fwd"
        if d > TRAIN_END:
            return "holdout"
        reads_holdout = any(not eq(c, comp[k][d][c], capped[k][d][c]) for k in comp for c in cols)
        return "holdout" if reads_holdout else "train"

    rows = {}
    for d, s in stored.items():
        filled = [c for c in cols if s[c] is not None]   # stored NULL = never filled: not compared, not written
        pre, post = comp["pre"][d], comp["post"][d]
        crbz = [c for c in filled if not eq(c, pre[c], post[c])]
        resid = list(RESIDUAL.get((version, d), ()))
        r = {"regime": regime(d), "stored": s, "pre": pre, "post": post, "filled": filled, "resid": resid,
             "crdl_row": str(s["backfill_run_id"]) == CRDL_RUN}
        if run_id is None:
            expect_pre = s["computed_at"] < crbz_done or r["crdl_row"]
            want = pre if expect_pre else post
            repro_fail = [c for c in filled if not eq(c, s[c], want[c])]
            crbz = crbz if expect_pre else []
            resid_ok = all(c in repro_fail and c not in crbz and eq(c, s[c], pre_both[d][c]) and not eq(c, s[c], post[c])
                           for c in resid)
            # a label that doesn't reproduce leaves the row whole as stored; other failing columns stay as stored
            if (set(repro_fail) - set(resid)) & set(LABELS):
                planned = []
            else:
                planned = sorted(set(c for c in crbz if c not in repro_fail) | set(resid))
            r.update(expect_pre=expect_pre, repro_fail=repro_fail, crbz=crbz, resid_ok=resid_ok, planned=planned)
        else:
            written = str(s["backfill_run_id"]) == run_id
            planned = sorted(c for c in set(crbz) | set(resid) if eq(c, s[c], post[c])) if written else []
            r.update(written=written, crbz=crbz, planned=planned)
        r["differs"] = [c for c in filled if not eq(c, s[c], post[c])]   # stored vs post now
        rows[d] = r
    return rows


def _row_lines(version, d, s, types, skip=()) -> list:
    return [f"{version}|{d}|{c}|{canon(v, types.get(c, ''))}" for c, v in s.items() if c not in skip]


def measure(version, rows, types) -> dict:
    """Per-group counts and checksums (no values). Identical in Step 0, preflight and smoke if the
    write did exactly the plan: `md5_after` / `md5_keys` are of the planned (or re-derived written)
    cells; `md5_differs` of the cells where stored ≠ post (expected after the write: the excluded
    ones); `md5_untouched` of every cell the write must not change (unplanned rows whole, planned rows
    minus written columns and stamps); `md5_labels` of the stored status / touch / reached / days."""
    out = {}
    for g in GROUPS:
        rs = {d: r for d, r in rows.items() if r["regime"] == g}
        keys, before, after, all_post, untouched, labels = [], [], [], [], [], []
        differs_after_write = []
        for d, r in rs.items():
            s, post = r["stored"], r["post"]
            for c in r["planned"]:
                keys.append(f"{version}|{d}|{c}")
                before.append(f"{version}|{d}|{c}|{canon(s[c], types[c])}")
                after.append(f"{version}|{d}|{c}|{canon(post[c], types[c])}")
            all_post.extend(f"{version}|{d}|{c}|{canon(post[c], types[c])}" for c in r["filled"])
            skip = set(r["planned"]) | (set(STAMP_COLUMNS) if r["planned"] else set())
            untouched.extend(_row_lines(version, d, s, types, skip))
            labels.extend(f"{version}|{d}|{c}|{canon(s[c], types[c])}" for c in FLIP_LABELS)
            # cells that differ from post once the plan is written (Step 0: differs minus planned)
            differs_after_write.extend(f"{version}|{d}|{c}|{canon(s[c], types[c])}"
                                       for c in r["differs"] if c not in r["planned"])
        out[g] = {
            "rows": len(rs), "planned_rows": sum(bool(r["planned"]) for r in rs.values()),
            "planned_values": len(keys), "planned_cols": dict(Counter(c for r in rs.values() for c in r["planned"])),
            "label_changes": dict(Counter(c for r in rs.values() for c in r["planned"] if c in LABELS)),
            "flip_label_changes": sum(c in FLIP_LABELS for r in rs.values() for c in r["planned"]),
            "differs_after_write": len(differs_after_write),
            "md5_keys": md5(keys), "md5_before": md5(before), "md5_after": md5(after),
            "md5_all_post": md5(all_post), "md5_untouched": md5(untouched), "md5_labels": md5(labels),
            "md5_differs_after_write": md5(differs_after_write),
        }
    return out


def step0_summary(version, rows, types) -> dict:
    """Step 0 view: measure() plus reproduction counts; train rows with values."""
    m = measure(version, rows, types)
    for g in GROUPS:
        rs = [r for r in rows.values() if r["regime"] == g]
        m[g].update({
            "repro_fail_rows": sum(bool(r["repro_fail"]) for r in rs),
            "repro_fail_cols": dict(Counter(c for r in rs for c in r["repro_fail"])),
            "changed_rows": sum(bool(r["crbz"]) for r in rs),
            "changed_cols": dict(Counter(c for r in rs for c in r["crbz"])),
            "crbz_label_changes": dict(Counter(c for r in rs for c in r["crbz"] if c in LABELS)),
            "excluded_crbz_values": sum(len(set(r["crbz"]) - set(r["planned"])) for r in rs),
            "resid_values": sum(len(r["resid"]) for r in rs),
            "resid_ok": all(r["resid_ok"] for r in rs),
            "crdl_rows": sum(r["crdl_row"] for r in rs),
            "crdl_rows_only_crbz_cols": sum(r["crdl_row"] and not (set(r["differs"]) - set(r["crbz"]) - set(r["resid"]))
                                            for r in rs),
        })
    detail = {}
    for d, r in rows.items():
        if r["regime"] != "train":
            continue
        s = r["stored"]
        detail[str(d)] = {
            "status": s["outcome_status"], "computed_at": _j(s["computed_at"]),
            "expect": "pre" if r["expect_pre"] else "post", "crdl_row": r["crdl_row"],
            "repro_fail": {c: {"stored": canon(s[c], types[c]), "recomputed":
                               canon((r["pre"] if r["expect_pre"] else r["post"])[c], types[c])} for c in r["repro_fail"]},
            "crbz": {c: {"from": canon(s[c], types[c]), "to": canon(r["post"][c], types[c])} for c in r["crbz"]},
            "resid": {c: {"from": canon(s[c], types[c]), "to": canon(r["post"][c], types[c])} for c in r["resid"]},
            "planned": {c: {"from": canon(s[c], types[c]), "to": canon(r["post"][c], types[c])} for c in r["planned"]},
            "never_filled": [c for c in OWNED[version] if s[c] is None],
        }
    m["train"]["rows_detail"] = detail
    return m


def _print_group(version, g, o) -> None:
    extra = (f"; reproduction fails {o['repro_fail_rows']} rows {o['repro_fail_cols']}; CR-BZ changes "
             f"{o['changed_rows']} rows {o['changed_cols']}; CR-BZ label changes {o['crbz_label_changes']}; "
             f"excluded CR-BZ values {o['excluded_crbz_values']}; residual {o['resid_values']} ok={o['resid_ok']}; "
             f"CR-DL rows {o['crdl_rows']} (only CR-BZ / residual cols {o['crdl_rows_only_crbz_cols']})"
             if "repro_fail_rows" in o else "")
    print(f"  {version} {g:7} rows {o['rows']}; planned {o['planned_rows']} rows / {o['planned_values']} values "
          f"{o['planned_cols']}; label cols planned {o['label_changes']} (status/touch/reached/days "
          f"{o['flip_label_changes']}); differ from post after write "
          f"{o['differs_after_write']}{extra}", flush=True)
    print(f"    md5 keys {o['md5_keys']} before {o['md5_before']} after {o['md5_after']} all_post {o['md5_all_post']}\n"
          f"    md5 untouched {o['md5_untouched']} labels {o['md5_labels']} differs_after_write "
          f"{o['md5_differs_after_write']}", flush=True)


def connect():
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    return conn


def vrp_rows(conn) -> list:
    return conn.execute(
        "SELECT trade_date, vol_risk_premium FROM bt_daily_features WHERE ticker=%s AND "
        "feature_version='v0.5.0-rebuilt' AND active AND vol_risk_premium IS NOT NULL "
        "AND trade_date >= %s ORDER BY trade_date", (TICKER, VRP_FROM)).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run label for the checksum comparison")
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    try:
        conn = connect()
    except Exception as e:  # never echo the connection string
        print(f"DB connect failed: {type(e).__name__}")
        return 1
    conn.read_only = True
    types = column_types(conn)
    crbz_done = conn.execute("SELECT completed_at FROM bt_backfill_runs WHERE run_id = %s", (CRBZ_RUN,)).fetchone()[0]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "tag": args.tag,
                    "comparison": "exact in column type (Amendment 1)", "crbz_completed_at": _j(crbz_done),
                    "versions": {}}
    for version in OWNED:
        rows = analyse(conn, version, types, crbz_done)
        report["versions"][version] = step0_summary(version, rows, types)
        print(f"{version}: {len(rows)} rows from {ROWS_FROM}")
        for g in GROUPS:
            _print_group(version, g, report["versions"][version][g])
        (LOG_DIR / f"step0_{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))
        conn.close()   # fresh connection per version: long passes outlive the server's idle limit
        conn = connect()
        conn.read_only = True

    # vol_risk_premium (v0.5.0-rebuilt only; every non-null 2026 row is ≤ 2026-06-05, train)
    vtype = column_types(conn, "bt_daily_features")["vol_risk_premium"]
    pre_conn = PreConn(conn)
    vrows = vrp_rows(conn)
    assert all(d <= TRAIN_END for d, _ in vrows)
    vrp = {"rows_checked": len(vrows), "type": vtype, "rows": {}}
    for d, stored_v in vrows:
        x = {k: canon(_compute_vol_features(c, d, TICKER)["vol_risk_premium"], vtype)
             for k, c in (("pre", pre_conn), ("post", conn))}
        vrp["rows"][str(d)] = {"stored": canon(stored_v, vtype), **x}
    r = vrp["rows"].values()
    vrp.update(repro_fail=sum(x["stored"] != x["pre"] for x in r), crbz=sum(x["pre"] != x["post"] for x in r),
               md5_stored=md5(f"{d}|{x['stored']}" for d, x in vrp["rows"].items()))
    report["vol_risk_premium"] = vrp
    print(f"vol_risk_premium ({vtype}): {len(vrows)} rows {VRP_FROM} → {TRAIN_END}; reproduction fails "
          f"{vrp['repro_fail']}; CR-BZ changes {vrp['crbz']}; md5 stored {vrp['md5_stored']}")
    (LOG_DIR / f"step0_{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {LOG_DIR / f'step0_{args.tag}.json'}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
