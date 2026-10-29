#!/usr/bin/env python3
"""CR-DJ I2 — replace the dying ES H26 bars of the March 2026 roll window with Databento ES M26.

Window (Amendment 1 D1, CR-BF calendar rule): bar starts 2026-03-15 22:00 -> 2026-03-20 20:59 UTC,
6,900 Globex minutes. `ironbeam_es_1m_bars` / `es_minutes` hold ESH6 to 03-20 13:29 UTC and
ESM6 after; every minute of the window becomes Databento ESM6 (`source = 'databento'`), as
CR-BF did for 2026-09-13 -> 09-18.

Input: scripts/logs/cr_dj/databento_esm6_window.parquet (cr_dj_fetch_databento.py; the billed
fetch is not repeated here).

Write (only without --dry-run, after the run-only grants of Amendment 1 D2 are applied):
  - every minute goes first to ironbeam_es_1m_bars_repair_log, one row per (table, minute),
    symbol 'ESM6' (the raw symbol); old values NULL for inserted minutes;
  - stored minutes: UPDATE OHLCV + source, guarded on the logged old values;
  - missing minutes: INSERT into es_minutes first (trade_date / is_rth as the trigger computes
    them), then into ironbeam_es_1m_bars (its AFTER INSERT trigger then hits ON CONFLICT);
  - RTH bar_index renumbered 0..n-1 by time on the PT dates that received an RTH minute;
  - all of it in one transaction; any guard miss rolls back everything;
  - backfill protocol: BACKFILL_DATABASE_URL, assert_role_or_die, backfill_run, update_run_smoke.
    No DELETE.

Smoke (same run, after the write): both tables equal Databento minute for minute; tables equal
each other; RTH bar_index = rank; 03-18 11:00-11:15 PT range; CR-CZ box edges recomputed from
the patched closes; no 'databento' rows in 2026 outside the CR-BF and CR-DJ windows.

Usage:
    apps/web/.venv/bin/python scripts/cr_dj_roll_patch.py --dry-run
    apps/web/.venv/bin/python scripts/cr_dj_roll_patch.py --grants-applied-at "2026-10-28T18:05Z"

Exit: 0 on success; 1 if a precondition, guard or smoke check failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_smoke,
)

CR_ID = "CR-DJ"
LO = dt.datetime(2026, 3, 15, 22, 0)  # naive UTC bar starts, as ironbeam_es_1m_bars.datetime
HI = dt.datetime(2026, 3, 20, 21, 0)  # exclusive
N_MINUTES = 6900
RAW_SYMBOL = "ESM6"
CRBF_LO, CRBF_HI = dt.datetime(2026, 9, 13, 22, 0), dt.datetime(2026, 9, 18, 21, 0)
OHLCV = ["open", "high", "low", "close", "volume"]
LOG_TABLE = "ironbeam_es_1m_bars_repair_log"
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_dj"
DB_FILE = OUT_DIR / "databento_esm6_window.parquet"
CARDS = [f"CR-CZ:cr-cz:cp-ranges:{i}" for i in range(49, 61)]
PRIVS = [  # (table, column, privilege) the write needs
    *[(t, c, "UPDATE") for t in ("ironbeam_es_1m_bars", "es_minutes") for c in OHLCV + ["source"]],
    ("es_minutes", "bar_index", "UPDATE"),
    *[(t, c, "INSERT") for t in ("ironbeam_es_1m_bars", "es_minutes") for c in OHLCV + ["source"]],
]

SELECT_STORED = """
SELECT b.datetime, b.open AS bar_open, b.high AS bar_high, b.low AS bar_low, b.close AS bar_close,
       b.volume AS bar_volume, b.source AS bar_source,
       (e.ts_utc IS NOT NULL) AS em_present,
       e.open AS em_open, e.high AS em_high, e.low AS em_low, e.close AS em_close,
       e.volume AS em_volume, e.source AS em_source
FROM ironbeam_es_1m_bars b
LEFT JOIN es_minutes e ON e.ts_utc = b.datetime AT TIME ZONE 'UTC'
WHERE b.datetime >= %(lo)s AND b.datetime < %(hi)s
ORDER BY b.datetime
"""
COUNT_EM = """SELECT count(*) FROM es_minutes
WHERE ts_utc >= %(lo)s::timestamp AT TIME ZONE 'UTC' AND ts_utc < %(hi)s::timestamp AT TIME ZONE 'UTC'"""

LOG_INSERT = f"""
INSERT INTO {LOG_TABLE} (run_id, target_table, bar_ts,
    old_open, old_high, old_low, old_close, old_volume,
    new_open, new_high, new_low, new_close, new_volume, symbol)
VALUES (%(run_id)s, %(target_table)s, %(datetime)s::timestamp AT TIME ZONE 'UTC',
    %(old_open)s, %(old_high)s, %(old_low)s, %(old_close)s, %(old_volume)s,
    %(new_open)s, %(new_high)s, %(new_low)s, %(new_close)s, %(new_volume)s, %(symbol)s)
"""
# Guarded on the logged old values and source: a row changed since selection matches nothing.
UPDATE_BARS = """
UPDATE ironbeam_es_1m_bars
SET open = %(new_open)s, high = %(new_high)s, low = %(new_low)s, close = %(new_close)s,
    volume = %(new_volume)s, source = 'databento'
WHERE datetime = %(datetime)s AND source = %(old_source)s
  AND open = %(old_open)s AND high = %(old_high)s AND low = %(old_low)s AND close = %(old_close)s
  AND volume IS NOT DISTINCT FROM %(old_volume)s
"""
UPDATE_ES_MINUTES = """
UPDATE es_minutes
SET open = %(new_open)s, high = %(new_high)s, low = %(new_low)s, close = %(new_close)s,
    volume = %(new_volume)s, source = 'databento'
WHERE ts_utc = %(datetime)s::timestamp AT TIME ZONE 'UTC' AND source IS NOT DISTINCT FROM %(old_source)s
  AND open = %(old_open)s AND high = %(old_high)s AND low = %(old_low)s AND close = %(old_close)s
  AND volume IS NOT DISTINCT FROM %(old_volume)s
"""
# trade_date / is_rth exactly as trigger es_minutes_from_ironbeam computes them; bar_index set below.
INSERT_ES_MINUTES = """
WITH t AS (SELECT %(datetime)s::timestamp AT TIME ZONE 'UTC' AS ts_utc)
INSERT INTO es_minutes (ts_utc, ts_pt, trade_date, bar_index, is_rth, open, high, low, close, volume, source)
SELECT ts_utc, ts_utc AT TIME ZONE 'America/Los_Angeles', (ts_utc AT TIME ZONE 'America/Los_Angeles')::date, -1,
       (extract(hour FROM ts_utc AT TIME ZONE 'America/Los_Angeles') * 60
        + extract(minute FROM ts_utc AT TIME ZONE 'America/Los_Angeles')) BETWEEN 390 AND 779,
       %(new_open)s, %(new_high)s, %(new_low)s, %(new_close)s, %(new_volume)s, 'databento'
FROM t
"""
INSERT_BARS = """
INSERT INTO ironbeam_es_1m_bars (datetime, open, high, low, close, volume, source)
VALUES (%(datetime)s, %(new_open)s, %(new_high)s, %(new_low)s, %(new_close)s, %(new_volume)s, 'databento')
"""
RENUMBER = """
UPDATE es_minutes e SET bar_index = r.rn
FROM (SELECT ts_utc, (row_number() OVER (PARTITION BY trade_date ORDER BY ts_utc) - 1)::int AS rn
      FROM es_minutes WHERE is_rth AND trade_date = ANY(%(dates)s)) r
WHERE e.ts_utc = r.ts_utc AND e.bar_index IS DISTINCT FROM r.rn
"""


def load_databento() -> pd.DataFrame:
    df = pd.read_parquet(DB_FILE)
    df["datetime"] = pd.to_datetime(df["ts_event"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[(df["datetime"] >= LO) & (df["datetime"] < HI)].sort_values("datetime").reset_index(drop=True)
    for f in OHLCV:
        df[f] = df[f].astype(float)
    return df


def select_stored(conn) -> pd.DataFrame:
    cur = conn.execute(SELECT_STORED, {"lo": LO, "hi": HI})
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def plan(db: pd.DataFrame, stored: pd.DataFrame) -> dict:
    m = db.merge(stored, on="datetime", how="left", indicator=True)
    new = {f"new_{f}": m[f] for f in OHLCV}
    upd = m[m["_merge"] == "both"]
    ins = m[m["_merge"] == "left_only"]
    base = lambda d: pd.DataFrame({"datetime": d["datetime"], "symbol": RAW_SYMBOL,  # noqa: E731
                                   **{k: v.loc[d.index] for k, v in new.items()}})
    bars_upd, em_upd = base(upd), base(upd)
    for f in OHLCV:
        bars_upd[f"old_{f}"] = upd[f"bar_{f}"]
        em_upd[f"old_{f}"] = upd[f"em_{f}"]
    bars_upd["old_source"], em_upd["old_source"] = upd["bar_source"], upd["em_source"]
    inserts = base(ins)
    for f in OHLCV:
        inserts[f"old_{f}"] = None
    pt = inserts["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
    rth = (pt.dt.hour * 60 + pt.dt.minute).between(390, 779)
    return {
        "bars_upd": bars_upd.reset_index(drop=True),
        "em_upd": em_upd.reset_index(drop=True),
        "inserts": inserts.reset_index(drop=True),
        "rth_dates": sorted({d for d in pt[rth].dt.date}),
        "rth_inserts": int(rth.sum()),
        "stored_not_in_databento": int(len(set(stored["datetime"]) - set(db["datetime"]))),
        "stored_without_es_minutes": int((~stored["em_present"]).sum()),
        "es_minutes_differs_from_bar": int(sum(
            (stored[f"em_{f}"] != stored[f"bar_{f}"]).sum() for f in OHLCV)),
    }


def summarize(db: pd.DataFrame, stored: pd.DataFrame, p: dict) -> dict:
    hour21 = int((stored["datetime"].dt.hour == 21).sum())
    pt_days = stored["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.date
    return {
        "window_utc": [str(LO), str(HI - dt.timedelta(minutes=1))],
        "databento_minutes": int(len(db)),
        "databento_symbols": sorted(db["symbol"].unique().tolist()),
        "databento_volume": int(db["volume"].sum()),
        "stored_minutes": int(len(stored)),
        "stored_source_counts": {str(k): int(v) for k, v in stored["bar_source"].value_counts(dropna=False).items()},
        "stored_by_pt_date": {str(k): int(v) for k, v in pt_days.value_counts().sort_index().items()},
        "stored_in_21utc_hour": hour21,
        "to_update_per_table": int(len(p["bars_upd"])),
        "to_insert_per_table": int(len(p["inserts"])),
        "rth_inserts": p["rth_inserts"],
        "bar_index_renumber_dates": [str(d) for d in p["rth_dates"]],
        "stored_not_in_databento": p["stored_not_in_databento"],
        "stored_without_es_minutes": p["stored_without_es_minutes"],
        "es_minutes_differs_from_bar": p["es_minutes_differs_from_bar"],
    }


def check_preconditions(conn, db: pd.DataFrame, s: dict) -> list[str]:
    problems = []
    if conn.execute("SELECT to_regclass(%s)", (LOG_TABLE,)).fetchone()[0] is None:
        problems.append(f"{LOG_TABLE} does not exist")
    for table, col, priv in PRIVS:
        if not conn.execute("SELECT has_column_privilege(current_user, %s, %s, %s)", (table, col, priv)).fetchone()[0]:
            problems.append(f"no {priv} ({col}) on {table}")
    if s["databento_minutes"] != N_MINUTES or s["databento_symbols"] != [RAW_SYMBOL]:
        problems.append(f"Databento file: {s['databento_minutes']} minutes, symbols {s['databento_symbols']}")
    if db["datetime"].dt.hour.eq(21).any():
        problems.append("Databento bars in the 21:00 UTC maintenance hour")
    if s["stored_not_in_databento"] or s["stored_without_es_minutes"] or s["es_minutes_differs_from_bar"]:
        problems.append("stored minutes not covered by Databento, or es_minutes not equal to the bars")
    if conn.execute(COUNT_EM, {"lo": LO, "hi": HI}).fetchone()[0] != s["stored_minutes"]:
        problems.append("es_minutes holds minutes the bars table does not")
    return problems


def _rows(df: pd.DataFrame, **extra) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        r = {k: (None if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in r.items()}
        r["datetime"] = pd.Timestamp(r["datetime"]).to_pydatetime()
        out.append({**r, **extra})
    return out


def write(conn, run_id: str, p: dict) -> int:
    """Log, update, insert, renumber — one transaction. Raises (rolls back) on any guard miss."""
    with conn.transaction():
        for table, frame in (("ironbeam_es_1m_bars", p["bars_upd"]), ("es_minutes", p["em_upd"]),
                             ("ironbeam_es_1m_bars", p["inserts"]), ("es_minutes", p["inserts"])):
            for r in _rows(frame, run_id=run_id, target_table=table):
                conn.execute(LOG_INSERT, r)
        for sql, frame, name in ((UPDATE_BARS, p["bars_upd"], "ironbeam_es_1m_bars"),
                                 (UPDATE_ES_MINUTES, p["em_upd"], "es_minutes")):
            for r in _rows(frame):
                n = conn.execute(sql, r).rowcount
                if n != 1:
                    raise RuntimeError(f"guard miss on {name} {r['datetime']}: {n} rows matched")
        for r in _rows(p["inserts"]):
            for sql, name in ((INSERT_ES_MINUTES, "es_minutes"), (INSERT_BARS, "ironbeam_es_1m_bars")):
                if conn.execute(sql, r).rowcount != 1:
                    raise RuntimeError(f"insert failed on {name} {r['datetime']}")
        return conn.execute(RENUMBER, {"dates": p["rth_dates"]}).rowcount if p["rth_dates"] else 0


def smoke(conn, db: pd.DataFrame) -> dict:
    q = lambda sql, prm=None: conn.execute(sql, prm or {}).fetchall()  # noqa: E731
    out = {}
    bars = pd.DataFrame(q("SELECT datetime, open, high, low, close, volume, source FROM ironbeam_es_1m_bars "
                          "WHERE datetime >= %(lo)s AND datetime < %(hi)s", {"lo": LO, "hi": HI}),
                        columns=["datetime", *OHLCV, "source"])
    em = pd.DataFrame(q("SELECT ts_utc AT TIME ZONE 'UTC', open, high, low, close, volume, source FROM es_minutes "
                        "WHERE ts_utc >= %(lo)s::timestamp AT TIME ZONE 'UTC' AND ts_utc < %(hi)s::timestamp AT TIME ZONE 'UTC'",
                        {"lo": LO, "hi": HI}), columns=["datetime", *OHLCV, "source"])
    for name, t in (("ironbeam_es_1m_bars", bars), ("es_minutes", em)):
        t["datetime"] = pd.to_datetime(t["datetime"])
        m = db.merge(t, on="datetime", how="outer", suffixes=("_db", ""), indicator=True)
        both = m[m["_merge"] == "both"]
        out[name] = {
            "rows": int(len(t)),
            "only_databento": int((m["_merge"] == "left_only").sum()),
            "only_table": int((m["_merge"] == "right_only").sum()),
            "ohlcv_mismatch_minutes": int(sum((both[f] != both[f"{f}_db"]) for f in OHLCV).astype(bool).sum()),
            "sources": {str(k): int(v) for k, v in t["source"].value_counts(dropna=False).items()},
        }
    ok = all(v["rows"] == N_MINUTES and v["only_databento"] == 0 and v["only_table"] == 0
             and v["ohlcv_mismatch_minutes"] == 0 and v["sources"] == {"databento": N_MINUTES}
             for v in (out["ironbeam_es_1m_bars"], out["es_minutes"]))

    bi = q("""SELECT trade_date, count(*), sum((bar_index <> rn)::int) FROM (
                SELECT trade_date, bar_index, (row_number() OVER (PARTITION BY trade_date ORDER BY ts_utc) - 1) rn
                FROM es_minutes WHERE is_rth AND trade_date BETWEEN '2026-03-16' AND '2026-03-20') x
              GROUP BY 1 ORDER BY 1""")
    out["rth_bar_index"] = {str(d): {"rows": int(n), "not_rank": int(bad)} for d, n, bad in bi}
    ok &= all(v["rows"] == 390 and v["not_rank"] == 0 for v in out["rth_bar_index"].values())

    lo, hi = q("SELECT min(low), max(high) FROM es_minutes WHERE ts_pt >= '2026-03-18 11:00' AND ts_pt < '2026-03-18 11:16'")[0]
    out["chart_2026_03_18_1100_1115_pt"] = {"low": lo, "high": hi}

    cards = []
    for cid, meta in q("SELECT id, meta FROM setup_candidates WHERE id = ANY(%(ids)s) ORDER BY session_date, entry_ts_pt",
                       {"ids": CARDS}):
        rng = meta.get("range") or {}
        a, b = rng.get("formed_ts_pt"), rng.get("qualified_ts_pt")
        top = bot = None
        if a and b:
            # qualified_ts_pt is the qualifying bar's close (cr_cz_build_gallery: ts(d, r.q, close=True)),
            # so the box covers bar starts [formed, qualified).
            top, bot = q("SELECT max(close), min(close) FROM es_minutes WHERE ts_pt >= %(a)s AND ts_pt < %(b)s",
                         {"a": a, "b": b})[0]
        cards.append({"id": cid, "card_top": rng.get("top"), "card_bottom": rng.get("bottom"),
                      "chart_close_top": top, "chart_close_bottom": bot,
                      "max_abs_diff": None if top is None else max(abs(top - rng["top"]), abs(bot - rng["bottom"]))})
    out["cr_cz_cards"] = cards
    ok &= all(c["max_abs_diff"] is not None and c["max_abs_diff"] <= 0.25 for c in cards)  # B5 one-tick tolerance

    stray = q("""SELECT count(*) FROM ironbeam_es_1m_bars WHERE source = 'databento' AND datetime >= '2026-01-01'
                 AND NOT (datetime >= %(a)s AND datetime < %(b)s) AND NOT (datetime >= %(c)s AND datetime < %(d)s)""",
              {"a": LO, "b": HI, "c": CRBF_LO, "d": CRBF_HI})[0][0]
    out["databento_rows_2026_outside_windows"] = int(stray)
    ok &= stray == 0
    out["ok"] = bool(ok)
    return out


def smoke_only(conn, run_id: str) -> int:
    """Re-run the smoke on the patched tables; keep the run's original results, add the re-run beside them."""
    row = conn.execute("SELECT smoke_test_results, self_assessment FROM bt_backfill_runs WHERE run_id = %s AND cr_id = %s",
                       (run_id, CR_ID)).fetchone()
    if row is None:
        print(f"[{CR_ID}] no {CR_ID} run {run_id}")
        return 1
    original, assessment = row
    rerun = smoke(conn, load_databento())
    rerun["at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rerun["why"] = ("in-run smoke rebuilt CR-CZ boxes over [formed, qualified] inclusive; qualified_ts_pt is the "
                    "qualifying bar's close, so the box is [formed, qualified) — cards 50 (0.75) and 54 (8.0) failed "
                    "on the smoke's rule, not the data")
    update_run_smoke(conn, run_id, {**original, "smoke_rerun": rerun},
                     f"{assessment} | smoke re-run {rerun['at']} (card box rule fixed): {'OK' if rerun['ok'] else 'FAILED'}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT_DIR / f"smoke_rerun_{stamp}.json").write_text(json.dumps({"run_id": run_id, **rerun}, indent=2, default=str))
    print(json.dumps(rerun, indent=1, default=str))
    return 0 if rerun["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="plan and report only; no writes")
    ap.add_argument("--grants-applied-at", help="UTC time the run-only grants were applied (recorded in notes)")
    ap.add_argument("--smoke-only", metavar="RUN_ID",
                    help="re-run the smoke (read-only) and add it to RUN_ID's smoke results as 'smoke_rerun'")
    args = ap.parse_args()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke_only:
        return smoke_only(conn, args.smoke_only)
    if args.dry_run:
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")

    db = load_databento()
    stored = select_stored(conn)
    p = plan(db, stored)
    summary = summarize(db, stored, p)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "dryrun" if args.dry_run else "run"
    stored.to_csv(OUT_DIR / f"stored_before_{tag}_{stamp}.csv", index=False)

    print(f"[{CR_ID}] window {LO} -> {HI} UTC (exclusive), {RAW_SYMBOL}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    problems = check_preconditions(conn, db, summary)
    print(f"  preconditions for the write: {'OK' if not problems else problems}")

    if args.dry_run:
        before = smoke(conn, db)
        print(f"[{CR_ID}] smoke on the current (unpatched) tables, for reference:")
        print(json.dumps(before, indent=1, default=str))
        print(f"[{CR_ID}] dry run: nothing written.")
        return 0
    if problems:
        print(f"[{CR_ID}] preconditions failed; nothing written.")
        return 1
    if not args.grants_applied_at:
        print(f"[{CR_ID}] --grants-applied-at is required for the write; nothing written.")
        return 1

    with backfill_run(conn, CR_ID) as run_id:
        conn.execute("UPDATE bt_backfill_runs SET notes = %s WHERE run_id = %s",
                     (f"Run-only grants (Amendment 1 D2) applied {args.grants_applied_at}: UPDATE(source) on "
                      "ironbeam_es_1m_bars, es_minutes; UPDATE(bar_index) on es_minutes. Revoke: pending.", run_id))
        renumbered = write(conn, run_id, p)
        n_log = conn.execute(f"SELECT count(*) FROM {LOG_TABLE} WHERE run_id = %s", (run_id,)).fetchone()[0]
        result = smoke(conn, db)
        result.update(summary)
        result.update({"repair_log_rows": int(n_log), "bar_index_rows_renumbered": int(renumbered)})
        ok = result["ok"] and n_log == 2 * N_MINUTES
        update_run_smoke(conn, run_id, result,
                         f"{len(p['bars_upd'])} updated + {len(p['inserts'])} inserted minutes per table, "
                         f"ESH6 -> Databento ESM6 {LO} -> {HI} UTC; log rows {n_log}; "
                         f"bar_index renumbered {renumbered}; smoke {'OK' if ok else 'FAILED'}")
        (OUT_DIR / f"smoke_{stamp}.json").write_text(json.dumps({"run_id": run_id, **result}, indent=2, default=str))
        print(f"[{CR_ID}] run {run_id}: log rows {n_log}, renumbered {renumbered}, smoke {'OK' if ok else 'FAILED'}")
        print(json.dumps({k: result[k] for k in ("ironbeam_es_1m_bars", "es_minutes", "rth_bar_index",
                                                  "chart_2026_03_18_1100_1115_pt", "databento_rows_2026_outside_windows")},
                         indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
