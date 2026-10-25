#!/usr/bin/env python3
"""CR-BZ — repair 2026 ES minute bars that are fragments of the minute (Amendment A1.3).

A fragment bar is a `ironbeam_es_1m_bars` row whose flow-built minute (from
`ironbeam_es_flow_1s`: first open, max high, min low, last close, summed volume) differs
by more than one tick in high, low or close, AND whose bar volume is below the flow
volume — the bar holds only part of the minute's trades (Step 0 Q3: HEAD/TAIL fragments).
Where bar volume >= flow volume the flow is the series missing trades; those minutes are
not touched.

Scope: bars from 2026-01-01 up to --end (default: today 00:00 UTC), excluding the CR-BF
patch window 2026-09-13 22:00 -> 2026-09-18 21:00 UTC (source 'databento', Z26 by design).
Minutes whose flow holds more than one symbol are skipped (none in Step 0).

New values: OHLC from the flow; volume = flow volume. The flow volume is a LOWER BOUND:
on clean minutes bar volume runs 2-7 % above flow volume (Step 0 Q3).

Write (only without --dry-run, after Ryan's go):
  - old and new values of every changed row go first to ironbeam_es_1m_bars_repair_log,
    one row per (target table, minute): `ironbeam_es_1m_bars` and the matching
    `es_minutes` row (A1.3, A1.4 — es_minutes has an insert-only trigger, so it is
    updated here explicitly);
  - each UPDATE is guarded on the logged old OHLCV; all rows run in one transaction, and
    any guard miss aborts the whole write (nothing changes);
  - backfill protocol: BACKFILL_DATABASE_URL, assert_role_or_die, backfill_run,
    update_run_smoke. Needs scripts/cr_bz_ddl.sql applied (repair log + column grants).

Usage:
    apps/web/.venv/bin/python scripts/cr_bz_repair_fragment_bars.py --dry-run
    apps/web/.venv/bin/python scripts/cr_bz_repair_fragment_bars.py

Exit: 0 on success; 1 if a precondition or guard failed (nothing written) or the
after-state check failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
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

CR_ID = "CR-BZ"
START = dt.datetime(2026, 1, 1)
PATCH_LO = dt.datetime(2026, 9, 13, 22, 0)  # CR-BF window, naive UTC bar starts
PATCH_HI = dt.datetime(2026, 9, 18, 21, 0)
TICK = 0.25
OHLCV = ["open", "high", "low", "close", "volume"]
LOG_TABLE = "ironbeam_es_1m_bars_repair_log"
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bz"

# One row per fragment bar: the bar, its flow-built minute, and the matching es_minutes row.
SELECT_SQL = """
WITH f AS (
    SELECT date_trunc('minute', ts_utc) AS m,
           count(DISTINCT symbol) AS nsym, min(symbol) AS symbol,
           (array_agg(open ORDER BY ts_utc))[1] AS open,
           max(high) AS high, min(low) AS low,
           (array_agg(close ORDER BY ts_utc DESC))[1] AS close,
           sum(volume)::double precision AS volume,
           count(*) AS n_sec
    FROM ironbeam_es_flow_1s
    WHERE ts_utc >= %(start)s::timestamp AT TIME ZONE 'UTC'
      AND ts_utc <  %(end)s::timestamp AT TIME ZONE 'UTC'
    GROUP BY 1
)
SELECT b.datetime,
       b.open AS bar_open, b.high AS bar_high, b.low AS bar_low, b.close AS bar_close, b.volume AS bar_volume,
       b.source AS bar_source,
       f.open AS new_open, f.high AS new_high, f.low AS new_low, f.close AS new_close, f.volume AS new_volume,
       f.symbol, f.nsym, f.n_sec,
       (e.ts_utc IS NOT NULL) AS em_present,
       e.open AS em_open, e.high AS em_high, e.low AS em_low, e.close AS em_close, e.volume AS em_volume,
       e.is_rth AS em_is_rth
FROM ironbeam_es_1m_bars b
JOIN f ON f.m = b.datetime AT TIME ZONE 'UTC'
LEFT JOIN es_minutes e ON e.ts_utc = b.datetime AT TIME ZONE 'UTC'
WHERE b.datetime >= %(start)s AND b.datetime < %(end)s
  AND NOT (b.datetime >= %(patch_lo)s AND b.datetime < %(patch_hi)s)
  AND greatest(abs(b.high - f.high), abs(b.low - f.low), abs(b.close - f.close)) > %(tick)s
  AND b.volume < f.volume
ORDER BY b.datetime
"""

LOG_INSERT = f"""
INSERT INTO {LOG_TABLE} (run_id, target_table, bar_ts,
    old_open, old_high, old_low, old_close, old_volume,
    new_open, new_high, new_low, new_close, new_volume, symbol)
VALUES (%(run_id)s, %(target_table)s, %(bar_ts)s,
    %(old_open)s, %(old_high)s, %(old_low)s, %(old_close)s, %(old_volume)s,
    %(new_open)s, %(new_high)s, %(new_low)s, %(new_close)s, %(new_volume)s, %(symbol)s)
"""

# Guarded on the logged old values: a row changed since selection matches nothing.
UPDATE_BARS = """
UPDATE ironbeam_es_1m_bars
SET open = %(new_open)s, high = %(new_high)s, low = %(new_low)s, close = %(new_close)s, volume = %(new_volume)s
WHERE datetime = %(datetime)s
  AND open = %(old_open)s AND high = %(old_high)s AND low = %(old_low)s AND close = %(old_close)s
  AND volume IS NOT DISTINCT FROM %(old_volume)s
"""
UPDATE_ES_MINUTES = """
UPDATE es_minutes
SET open = %(new_open)s, high = %(new_high)s, low = %(new_low)s, close = %(new_close)s, volume = %(new_volume)s
WHERE ts_utc = %(datetime)s::timestamp AT TIME ZONE 'UTC'
  AND open = %(old_open)s AND high = %(old_high)s AND low = %(old_low)s AND close = %(old_close)s
  AND volume IS NOT DISTINCT FROM %(old_volume)s
"""


def select_fragments(conn, end: dt.datetime) -> pd.DataFrame:
    cur = conn.execute(SELECT_SQL, {
        "start": START, "end": end, "patch_lo": PATCH_LO, "patch_hi": PATCH_HI, "tick": TICK,
    })
    cols = [d.name for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    pt = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
    df["rth"] = (pt.dt.hour * 60 + pt.dt.minute).between(390, 779)
    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    df["em_equals_bar"] = df["em_present"] & df.apply(
        lambda r: all(r[f"em_{f}"] == r[f"bar_{f}"] for f in OHLCV), axis=1
    )
    return df


def plan(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bars rows to change, es_minutes rows to change), each with old_* / new_* columns."""
    use = df[df["nsym"] == 1]
    bars = pd.DataFrame({"datetime": use["datetime"], "symbol": use["symbol"]})
    for f in OHLCV:
        bars[f"old_{f}"] = use[f"bar_{f}"]
        bars[f"new_{f}"] = use[f"new_{f}"]
    em_src = use[use["em_present"]]
    em = pd.DataFrame({"datetime": em_src["datetime"], "symbol": em_src["symbol"]})
    for f in OHLCV:
        em[f"old_{f}"] = em_src[f"em_{f}"]
        em[f"new_{f}"] = em_src[f"new_{f}"]
    return bars.reset_index(drop=True), em.reset_index(drop=True)


def summarize(df: pd.DataFrame, bars: pd.DataFrame, em: pd.DataFrame) -> dict:
    if df.empty:
        return {"fragments": 0}
    err = pd.concat([(df["bar_high"] - df["new_high"]).abs(), (df["bar_low"] - df["new_low"]).abs(),
                     (df["bar_close"] - df["new_close"]).abs()], axis=1).max(axis=1)
    by_month = df.groupby(["month", "rth"]).size().unstack(fill_value=0).rename(columns={True: "rth", False: "eth"})
    return {
        "fragments": int(len(df)),
        "rth": int(df["rth"].sum()),
        "eth": int((~df["rth"]).sum()),
        "skipped_multi_symbol": int((df["nsym"] > 1).sum()),
        "bars_to_update": int(len(bars)),
        "es_minutes_to_update": int(len(em)),
        "es_minutes_missing": int((~df["em_present"]).sum()),
        "es_minutes_differs_from_bar": int((df["em_present"] & ~df["em_equals_bar"]).sum()),
        "bar_source_counts": {str(k): int(v) for k, v in df["bar_source"].value_counts(dropna=False).items()},
        "price_err_pts": {"median": float(err.median()), "p90": float(err.quantile(0.9)), "max": float(err.max())},
        "bar_over_flow_volume_median": float((df["bar_volume"] / df["new_volume"]).median()),
        "flow_seconds_median": float(df["n_sec"].median()),
        "first": str(df["datetime"].min()),
        "last": str(df["datetime"].max()),
        "by_month": {m: {k: int(v) for k, v in row.items()} for m, row in by_month.iterrows()},
    }


def check_preconditions(conn) -> list[str]:
    problems = []
    if conn.execute("SELECT to_regclass(%s)", (LOG_TABLE,)).fetchone()[0] is None:
        problems.append(f"{LOG_TABLE} does not exist (apply scripts/cr_bz_ddl.sql)")
    for table in ("ironbeam_es_1m_bars", "es_minutes"):
        missing = [
            col for col in OHLCV
            if not conn.execute(
                "SELECT has_column_privilege(current_user, %s, %s, 'UPDATE')", (table, col)
            ).fetchone()[0]
        ]
        if missing:
            problems.append(f"no UPDATE on {table} ({', '.join(missing)}) for current_user (apply scripts/cr_bz_ddl.sql)")
    return problems


def _rows(df: pd.DataFrame, **extra) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        r = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        r["datetime"] = pd.Timestamp(r["datetime"]).to_pydatetime()
        out.append({**r, **extra})
    return out


def write(conn, run_id: str, bars: pd.DataFrame, em: pd.DataFrame) -> None:
    """Log, then update, both tables in one transaction. Raises (rolls back) on any guard miss."""
    with conn.transaction():
        for table, frame in (("ironbeam_es_1m_bars", bars), ("es_minutes", em)):
            for r in _rows(frame, run_id=run_id, target_table=table):
                conn.execute(LOG_INSERT, {**r, "bar_ts": r["datetime"].replace(tzinfo=dt.timezone.utc)})
        for sql, frame, name in ((UPDATE_BARS, bars, "ironbeam_es_1m_bars"), (UPDATE_ES_MINUTES, em, "es_minutes")):
            for r in _rows(frame):
                n = conn.execute(sql, r).rowcount
                if n != 1:
                    raise RuntimeError(f"guard miss on {name} {r['datetime']}: {n} rows matched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count and report only; no writes")
    ap.add_argument("--end", help="exclusive upper bound, naive UTC (default: today 00:00 UTC)")
    args = ap.parse_args()
    end = (dt.datetime.fromisoformat(args.end) if args.end
           else dt.datetime.combine(dt.datetime.now(dt.timezone.utc).date(), dt.time()))

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    df = select_fragments(conn, end)
    bars, em = plan(df)
    summary = summarize(df, bars, em)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df.to_csv(OUT_DIR / f"fragments_{'dryrun' if args.dry_run else 'run'}_{stamp}.csv", index=False)

    print(f"[{CR_ID}] window {START} -> {end} UTC (CR-BF patch window excluded)")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    problems = check_preconditions(conn)
    print(f"  preconditions for the write: {'OK' if not problems else problems}")

    if args.dry_run:
        print(f"[{CR_ID}] dry run: nothing written. Plan CSV in {OUT_DIR}")
        return 0
    if problems:
        print(f"[{CR_ID}] preconditions failed; nothing written.")
        return 1
    if bars.empty:
        print(f"[{CR_ID}] no fragment bars; nothing to do.")
        return 0

    with backfill_run(conn, CR_ID) as run_id:
        write(conn, run_id, bars, em)
        after = select_fragments(conn, end)
        n_log = conn.execute(f"SELECT count(*) FROM {LOG_TABLE} WHERE run_id = %s", (run_id,)).fetchone()[0]
        remaining = int(len(after))
        smoke = {
            **summary,
            "repair_log_rows": int(n_log),
            "fragments_remaining_after": remaining,
            "end": str(end),
        }
        ok = remaining == 0 and n_log == len(bars) + len(em)
        update_run_smoke(conn, run_id, smoke,
                         f"{len(bars)} bars + {len(em)} es_minutes rows repaired from flow; "
                         f"{remaining} fragments remain; log rows {n_log}"
                         + ("" if ok else " — AFTER-STATE CHECK FAILED"))
        print(f"[{CR_ID}] run {run_id}: log rows {n_log}, fragments remaining {remaining}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
