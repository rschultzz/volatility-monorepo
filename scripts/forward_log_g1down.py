"""CR-CX — forward log of the G1-down signal (B4: sessions from 2026-10-27, written once, never looked at early).

Spec: specs/CR-CX-forward-log-g1down.md (K1-K9, rulings RL1-RL5, choices Y1-Y12). Daily, from the Render cron
at 11:00 UTC Tue-Sat, for the previous session:

  1. guard (Y1): every date < 2026-10-27 is refused before any DB or network call; so is a non-trading day, and
     a session whose Databento data is not yet available (window end + 8 h).
  2. Databento (RL1, Y2): the session's ES trades [06:15, 13:00) PT for every CR-CA contract, cost-checked
     (<= $1), aggregated to 1 s as CR-CA into es_flow_1s_databento, rolled up into es_flow_1m_databento.
  3. frozen code (Y3): the CR-CJ loader SQL for that one session, then cr_cl_lib.build_legs with the CR-CO A1
     vol-scaled swing (12 x IM_d / IM_train) and the CR-CK noise band as shipped.
  4. log (Y4-Y7), one transaction per session: a forward_log_sessions row (logged / skipped + reason); every
     primary, ratio-defined leg in forward_log_events with its CR-CL match keys; outcomes for G1-down and
     base-down legs (h = up) in forward_log_outcomes. Catch-up: earlier eligible legs missing an outcome.

Prints COUNTS ONLY (B4, K6, Y9): sessions, legs per cell, outcomes inserted. Never a result, rate or move.
Backfill protocol: dash_backfill_writer, backfill_run(conn, "CR-CX").

Usage (apps/web/.venv/bin/python locally; python on Render):
  python scripts/forward_log_g1down.py                 # every unlogged session from 2026-10-27 to yesterday
  python scripts/forward_log_g1down.py --session 2026-10-27
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cr_cj_lib as L  # noqa: E402  (puts the repo root on sys.path; reads .env if present)
import cr_cl_lib as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared import flow_1m  # noqa: E402
from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

FORWARD_START = date(2026, 10, 27)              # B4 as frozen 2026-10-27
DEFINITION = "G1-down v1"                       # X1: CR-CO A1 primary
IM_TRAIN = 65.71410292087306                    # median y26 implied_move_1d (CR-CO O8; re-derived in the tests)
TERCILE_CUTS = (21.25, 30.333333333333343)      # Y5: y26 vol-scaled primary ratio-defined legs (C6b; tests)
EARLY_CLOSES = {date(2026, 11, 27), date(2026, 12, 24), date(2027, 11, 26)}
COST_CAP_USD = 1.0                              # RL1
AVAIL_LAG = timedelta(hours=8)                  # Step 0b: rolling 8 h historical delay
OUTCOME_CELLS = ("G1-down", "base-down")        # RL2; h = +1 for both
H_UP = 1
CR_ID = "CR-CX"
T_1S = "es_flow_1s_databento"

# CR-CJ load_skew / load_es queries, restricted to one session (Y3). Same columns, filters and aggregation.
SKEW_SQL = f"""
    SELECT trade_date, expir_date, expiry_tod, snapshot_pt, snap_shot_date, quote_date, stock_price, atmiv,
           {", ".join(L.VOLS)}
    FROM orats_monies_minute
    WHERE ticker = 'SPX' AND trade_date = %s
      AND expir_date = trade_date AND expiry_tod = 'pm'
      AND snapshot_pt::time >= '06:30' AND snapshot_pt::time < '13:00'
    ORDER BY trade_date, snapshot_pt
"""
ES_SQL = """
    WITH s AS (
        SELECT symbol, ts_utc, open, high, low, close, volume,
               (ts_utc AT TIME ZONE 'America/Los_Angeles') AS ts_pt
        FROM es_flow_1s_databento
        WHERE ts_utc >= %s AND ts_utc < %s
          AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
          AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    )
    SELECT ts_pt::date AS d, symbol, date_trunc('minute', ts_pt) AS m_pt,
           (array_agg(open ORDER BY ts_utc))[1]::float AS o, max(high)::float AS h, min(low)::float AS l,
           (array_agg(close ORDER BY ts_utc DESC))[1]::float AS c, sum(volume)::float AS vol
    FROM s GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
"""
IM_SQL = """SELECT (feature_vector->>'implied_move_1d')::float FROM bt_daily_features
            WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date = %s"""


class Refused(Exception):
    """A date the forward log must not process (B4 / Y1). Raised before any DB or network call."""


# ------------------------------------------------------------------------------------------------- guard (Y1)
def databento_window_utc(d: date) -> tuple[datetime, datetime]:
    """CR-CA window [06:15, 13:00) PT (cr_ca_step0.window_utc), without importing the Databento module."""
    return (L.pt(d, 6, 15).astimezone(timezone.utc), L.pt(d, 13, 0).astimezone(timezone.utc))


def check_date(d: date, now: datetime) -> None:
    if d < FORWARD_START:
        raise Refused(f"{d} is before the forward log start {FORWARD_START} (B4: train / holdout are not logged)")
    if not is_trading_day(d):
        raise Refused(f"{d} is not an NYSE trading day")
    if now < databento_window_utc(d)[1] + AVAIL_LAG:
        raise Refused(f"{d}: Databento data not available before {databento_window_utc(d)[1] + AVAIL_LAG:%Y-%m-%d %H:%M} UTC")


def default_targets(done: set, now: datetime) -> list[date]:
    """Every trading day from FORWARD_START whose data is available and that has no session row (Y1 catch-up)."""
    out, d = [], FORWARD_START
    while d <= now.date():
        try:
            check_date(d, now)
            if d not in done:
                out.append(d)
        except Refused:
            pass
        d += timedelta(days=1)
    return out


def code_sha() -> str:
    sha = os.environ.get("RENDER_GIT_COMMIT", "").strip()
    if not sha:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    if not sha:
        raise RuntimeError("no code SHA (RENDER_GIT_COMMIT unset and git unavailable)")
    return sha


# -------------------------------------------------------------------------------------------- Databento (Y2)
def pull_databento(conn, d: date) -> dict:
    """Load the session's ES trades into es_flow_1s_databento (+ 1-min rollup). Skips if every contract is there."""
    from cr_ca_databento_flow import insert as insert_1s          # CR-CA insert (ON CONFLICT DO NOTHING)
    from cr_ca_step0 import DATASET, aggregate_1s, client, contracts_for, in_window, raw_symbol
    import databento as db

    a, b = databento_window_utc(d)
    want = {raw_symbol(*c): c for c in contracts_for(d)}
    have = {r[0] for r in conn.execute(f"SELECT DISTINCT raw_symbol FROM {T_1S} WHERE ts_utc >= %s AND ts_utc < %s",
                                       (a, b)).fetchall()}
    if set(want) <= have:
        return dict(rows=0, usd=0.0, reused=True, contracts=sorted(want))
    cl = client()
    end = pd.Timestamp(cl.metadata.get_dataset_range(dataset=DATASET)["schema"]["trades"]["end"])
    if end < pd.Timestamp(b):
        raise RuntimeError(f"{d}: Databento trades end {end} is before the window end {b}")
    kw = dict(dataset=DATASET, schema="trades", stype_in="raw_symbol", start=a.isoformat(), end=b.isoformat())
    usd = sum(float(cl.metadata.get_cost(symbols=[s], **kw)) for s in want)
    if usd > COST_CAP_USD:
        raise RuntimeError(f"{d}: Databento cost ${usd:.2f} > cap ${COST_CAP_USD:.2f} (RL1)")
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for sym, (y, m) in want.items():
            p = Path(tmp) / f"{sym}.dbn.zst"
            cl.timeseries.get_range(symbols=[sym], path=str(p), **kw)
            tr = db.DBNStore.from_file(str(p)).to_df(pretty_ts=True, map_symbols=True)
            if tr.empty:
                continue
            tr = tr.reset_index()
            if set(tr["symbol"].unique()) - {sym} or not (tr["action"] == "T").all():
                raise RuntimeError(f"{sym} {d}: unexpected symbols or non-trade actions")
            f = aggregate_1s(tr, y, m)
            frames.append(f[in_window(f["ts_utc"])])
    if not frames:
        return dict(rows=0, usd=usd, reused=False, contracts=sorted(want))
    n = insert_1s(conn, pd.concat(frames, ignore_index=True))
    conn.execute(flow_1m.insert_sql(T_1S, "ts_utc >= %(a)s AND ts_utc < %(b)s", on_conflict="nothing"),
                 {"a": a, "b": b})
    return dict(rows=int(n), usd=usd, reused=False, contracts=sorted(want))


# ----------------------------------------------------------------------------------- frozen definition (Y3)
def qdf(conn, sql, params) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])


def load_session(conn, d: date) -> dict:
    """One session's inputs, as the CR-CJ / CR-CL loaders build them (no period guard: Y1 is the guard)."""
    sk = qdf(conn, SKEW_SQL, (d.isoformat(),))
    es = qdf(conn, ES_SQL, (L.pt(d, 6, 30), L.pt(d, 13, 0)))
    row = conn.execute(IM_SQL, (CANONICAL_FEATURE_VERSION, d)).fetchone()
    D = dict(MIN={}, BARS={}, SYM={}, SESS=[], PX={}, IM=None if row is None else row[0], N_SKEW=0)
    if len(sk):
        sk["d"] = pd.to_datetime(sk["trade_date"]).dt.date
        for c in ["stock_price", "atmiv"] + L.VOLS:
            sk[c] = sk[c].astype(float)
        D["MIN"] = L.skew_minutes(sk)
        D["N_SKEW"] = int(np.isfinite(D["MIN"][d]["put"]).sum()) if d in D["MIN"] else 0
    if len(es):
        D["SYM"], D["BARS"] = L.front_bars(es)
    if d in D["MIN"] and d in D["BARS"]:
        D["SESS"] = [d]
        D["PX"] = {d: L.es_close_at_snapshot(D["BARS"], d, D["MIN"])}
    return D


def skip_reason(D: dict, d: date) -> str | None:
    """Y4 completeness, checked in this order; None = complete."""
    if d in EARLY_CLOSES:
        return "early close"
    if d not in D["BARS"]:
        return "no Databento ES bars"
    B = D["BARS"][d]
    n = int((np.isfinite(B["h"]) & np.isfinite(B["l"])).sum())
    if n < L.NB:
        return f"incomplete ES bars ({n}/{L.NB})"
    if d not in D["MIN"] or D["N_SKEW"] == 0:
        return "no 0DTE PM skew snapshots"
    im = D["IM"]
    if im is None or not np.isfinite(im) or im <= 0:
        return "no implied_move_1d"
    return None


def session_legs(D: dict, d: date) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(all legs in the CR-CL domain, logged legs = primary and ratio-defined, swing). CR-CO A1 per session."""
    swing = 12.0 * float(D["IM"]) / IM_TRAIN
    W = C.build_legs(dict(D, SESS=[d]), swing)
    if W.empty:
        return W, W, swing
    Q = W[(~W["late"]) & (W["reason"] == "ok")].copy()
    Q["cell"] = Q["grp"] + np.where(Q["dir"] == 1, "-up", "-down")
    Q["tercile"] = np.digitize(Q["size"], TERCILE_CUTS)
    return W, Q.reset_index(drop=True), swing


def outcome(B: dict, ci: int, p0: float) -> dict:
    """CR-CL C9 / C10 (RL4), h = up. NaN moves past 12:59 become NULL."""
    def clean(v):
        return None if v is None or not np.isfinite(v) else float(v)
    return dict(race_10=C.race(B, ci, p0, H_UP, 10.0), race_20=C.race(B, ci, p0, H_UP, 20.0),
                move_30=clean(C.move_at(B, ci, p0, H_UP, 30)), move_60=clean(C.move_at(B, ci, p0, H_UP, 60)),
                mfe_before_m10=float(C.mfe_before(B, ci, p0, H_UP, 10.0)))


# --------------------------------------------------------------------------------------------- writes (Y7)
def ts_pt(d: date, i: int) -> datetime:
    return datetime.combine(d, time(6, 30)) + timedelta(minutes=int(i))


EV_COLS = ["signal", "grp", "dir", "definition_version", "session_date", "leg_start_ts_pt", "leg_end_ts_pt",
           "confirm_ts_pt", "entry_price", "start_px", "end_px", "swing_pts", "size_pts", "size_tercile", "cbucket",
           "edge", "ratio", "actual_pp", "expected_pp", "band_pp", "beyond_noise", "front_symbol", "code_sha",
           "run_id"]
OUT_COLS = ["event_id", "race_10", "race_20", "move_30", "move_60", "mfe_before_m10", "code_sha", "run_id"]
SESS_COLS = ["definition_version", "session_date", "status", "skip_reason", "front_symbol", "rth_bars",
             "skew_snapshots", "im_pts", "swing_pts", "legs_total", "legs_primary", "legs_logged",
             "databento_rows", "databento_usd", "code_sha", "run_id"]


def _ins(conn, table: str, cols: list, vals: list, conflict: str) -> int:
    sql = (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT {conflict} DO NOTHING")
    return conn.execute(sql, vals).rowcount


def insert_outcomes(conn, d: date, B: dict, sha: str, run_id) -> int:
    """Outcomes for this session's eligible legs that have none (current run or catch-up)."""
    rows = conn.execute("""
        SELECT e.id, e.confirm_ts_pt, e.entry_price FROM forward_log_events e
        LEFT JOIN forward_log_outcomes o ON o.event_id = e.id
        WHERE o.id IS NULL AND e.definition_version = %s AND e.session_date = %s AND e.signal = ANY(%s)
    """, (DEFINITION, d, list(OUTCOME_CELLS))).fetchall()
    n = 0
    for eid, cts, p0 in rows:
        ci = int((cts - datetime.combine(d, time(6, 30))).total_seconds() // 60)
        o = outcome(B, ci, float(p0))
        n += _ins(conn, "forward_log_outcomes", OUT_COLS,
                  [eid, o["race_10"], o["race_20"], o["move_30"], o["move_60"], o["mfe_before_m10"], sha, run_id],
                  "(event_id)")
    return n


def log_session(conn, d: date, D: dict, sha: str, run_id, pulled: dict | None = None) -> dict:
    """Y4-Y7 for one session, in one transaction. Returns counts only."""
    pulled = pulled or dict(rows=0, usd=0.0)
    why = skip_reason(D, d)
    sym = D["SYM"].get(d)
    nbar = 0 if d not in D["BARS"] else int((np.isfinite(D["BARS"][d]["h"]) & np.isfinite(D["BARS"][d]["l"])).sum())
    base = dict(definition_version=DEFINITION, session_date=d, front_symbol=sym, rth_bars=nbar,
                skew_snapshots=D["N_SKEW"], im_pts=D["IM"], databento_rows=pulled["rows"],
                databento_usd=pulled["usd"], code_sha=sha, run_id=run_id)
    res = dict(session=str(d), status="skipped" if why else "logged", skip_reason=why, legs_inserted=0,
               outcomes_inserted=0, cells={})
    with conn.transaction():
        if why:
            vals = dict(base, status="skipped", skip_reason=why, swing_pts=None, legs_total=None, legs_primary=None,
                        legs_logged=None)
            res["session_row_inserted"] = _ins(conn, "forward_log_sessions", SESS_COLS, [vals[c] for c in SESS_COLS],
                                               "(definition_version, session_date)")
            return res
        W, Q, swing = session_legs(D, d)
        sym_raw = D["SYM"][d]
        cells = Counter()                                  # legs inserted by this run, per cell
        for r in Q.itertuples():
            vals = dict(signal=r.cell, grp=r.grp, dir=int(r.dir), definition_version=DEFINITION, session_date=d,
                        leg_start_ts_pt=ts_pt(d, r.s - 1), leg_end_ts_pt=ts_pt(d, r.e - 1),
                        confirm_ts_pt=ts_pt(d, r.ci), entry_price=float(r.p0), start_px=float(r.start_px),
                        end_px=float(r.end_px), swing_pts=swing, size_pts=float(r.size), size_tercile=int(r.tercile),
                        cbucket=int(r.cbucket), edge=r.edge, ratio=float(r.ratio), actual_pp=float(r.actual),
                        expected_pp=float(r.expected), band_pp=float(r.band), beyond_noise=bool(r.beyond),
                        front_symbol=sym_raw, code_sha=sha, run_id=run_id)
            k = _ins(conn, "forward_log_events", EV_COLS, [vals[c] for c in EV_COLS],
                     "(definition_version, session_date, leg_end_ts_pt, dir)")
            res["legs_inserted"] += k
            if k:
                cells[r.cell] += 1
        res["cells"] = dict(cells)
        res["outcomes_inserted"] = insert_outcomes(conn, d, D["BARS"][d], sha, run_id)
        vals = dict(base, status="logged", skip_reason=None, swing_pts=swing, legs_total=len(W),
                    legs_primary=int((~W["late"]).sum()) if len(W) else 0, legs_logged=len(Q))
        res["session_row_inserted"] = _ins(conn, "forward_log_sessions", SESS_COLS, [vals[c] for c in SESS_COLS],
                                           "(definition_version, session_date)")
    return res


def catch_up_outcomes(conn, sha: str, run_id) -> int:
    """Eligible legs on earlier sessions still missing an outcome (e.g. a crash between commits)."""
    days = [r[0] for r in conn.execute("""
        SELECT DISTINCT e.session_date FROM forward_log_events e
        LEFT JOIN forward_log_outcomes o ON o.event_id = e.id
        WHERE o.id IS NULL AND e.definition_version = %s AND e.signal = ANY(%s)
    """, (DEFINITION, list(OUTCOME_CELLS))).fetchall()]
    n = 0
    for d in days:
        D = load_session(conn, d)
        if skip_reason(D, d) is None:
            with conn.transaction():
                n += insert_outcomes(conn, d, D["BARS"][d], sha, run_id)
    return n


# --------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=date.fromisoformat)
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    if args.session is not None:
        try:
            check_date(args.session, now)
        except Refused as e:
            print(f"REFUSED: {e}")
            return 2
    sha = code_sha()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    with backfill_run(conn, CR_ID) as run_id:
        if args.session is not None:
            targets = [args.session]
        else:
            done = {r[0] for r in conn.execute(
                "SELECT session_date FROM forward_log_sessions WHERE definition_version = %s", (DEFINITION,))}
            targets = default_targets(done, now)
        print(f"CR-CX forward log ({DEFINITION}), code {sha[:10]}, run {run_id}: {len(targets)} session(s) to process")
        results = []
        for d in targets:
            pulled = pull_databento(conn, d)
            r = log_session(conn, d, load_session(conn, d), sha, run_id, pulled)
            r["databento"] = pulled
            results.append(r)
            cells = ", ".join(f"{k} {v}" for k, v in sorted(r["cells"].items())) or "none"
            print(f"  {d}: {r['status']}{' (' + r['skip_reason'] + ')' if r['skip_reason'] else ''}; "
                  f"Databento rows {pulled['rows']} (${pulled['usd']:.2f}{', reused' if pulled.get('reused') else ''}); "
                  f"legs logged {r['legs_inserted']} [{cells}]; outcomes logged {r['outcomes_inserted']}")
        caught = catch_up_outcomes(conn, sha, run_id)
        smoke = dict(definition=DEFINITION, code_sha=sha, sessions=len(results),
                     logged=sum(r["status"] == "logged" for r in results),
                     skipped={r["session"]: r["skip_reason"] for r in results if r["skip_reason"]},
                     legs_inserted=sum(r["legs_inserted"] for r in results),
                     g1down_inserted=sum(r["cells"].get("G1-down", 0) for r in results),
                     outcomes_inserted=sum(r["outcomes_inserted"] for r in results) + caught,
                     outcomes_caught_up=caught,
                     databento_usd=round(sum(r["databento"]["usd"] for r in results), 4))
        update_run_smoke(conn, run_id, smoke, f"forward log: {smoke['logged']} logged, {len(smoke['skipped'])} "
                                               f"skipped, {smoke['legs_inserted']} legs, counts only (B4)")
        print(f"Done: sessions {smoke['sessions']} (logged {smoke['logged']}, skipped {len(smoke['skipped'])}); "
              f"legs logged {smoke['legs_inserted']} (G1-down {smoke['g1down_inserted']}); "
              f"outcomes logged {smoke['outcomes_inserted']} (catch-up {caught})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
