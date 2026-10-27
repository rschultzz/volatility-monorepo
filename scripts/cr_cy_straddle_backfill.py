#!/usr/bin/env python3
"""CR-CY — backfill 1-DTE SPX ATM straddle quotes at the close and the open. NO READ of outcomes.

Spec: specs/CR-CY-straddle-backfill.md.

For each CR-BO ES-clean night (t, t+1) in scripts/cr_cy_nights.csv (t+1 <= 2026-06-05):
  expiry  = SPX PM expiry on t+1 (root SPX; PM = expiry_tod 'pm' at read time)
  close   = t   12:50-13:00 PT (15:50-16:00 ET), strikes K0-5, K0, K0+5 around S_c
  open    = t+1 06:30-06:45 PT (09:30-09:45 ET), strikes K0-5, K0, K0+5 around S_o
  S_c     = ES close of the bar ending 15:55 ET  - basis_t
  S_o     = ES close of the bar ending 09:30 ET on t+1 - basis_t
  basis_t = median(ES - SPX cash) over 12:30-12:54 PT on t (prints available before 15:55 ET)

Fetch: options_cache.fetch_option_bars(record_empty_windows=False) — an empty ORATS response is
never recorded as a fetched window; it goes to scripts/logs/cr_cy/missing.csv.

Modes:
  --mode step0  the 5 trial nights; gates G1-G4; writes scripts/logs/cr_cy/step0_gates.json
  --mode full   every night; refuses to start unless step0_gates.json says pass
  --dry-run     plan only (strikes, windows), no fetch, no run row

Usage:
    PYTHONUNBUFFERED=1 /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u \
        scripts/cr_cy_straddle_backfill.py --mode step0 --vpx <path to cr_bp/vpx_primary.parquet>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ── ENV (before any import that reads DATABASE_URL) ──────────────────────────
def _find_dotenv() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        c = current / ".env"
        if c.exists():
            return c
        current = current.parent
    return None

_env_path = _find_dotenv()
if _env_path:
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = _bak_url   # options_cache repo reads DATABASE_URL

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import pandas as pd

from packages.shared import spx_cash
from packages.shared.backfill_safety import (
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError, OratsTransientError
from packages.shared.options_cache.opra import format_opra

CR_ID = "CR-CY"
ROOT = "SPX"
LAST_T1 = date(2026, 6, 5)                          # B4: nothing after the train end
NIGHTS_CSV = REPO_ROOT / "scripts" / "cr_cy_nights.csv"
LOG_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cy"
STEP0_NIGHTS = [date(2023, 7, 17), date(2024, 5, 13), date(2025, 2, 11), date(2025, 10, 6), date(2026, 1, 12)]

PT = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")
CLOSE_WIN = (time(12, 50), time(13, 0))             # 15:50-16:00 ET
OPEN_WIN = (time(6, 30), time(6, 45))               # 09:30-09:45 ET
BASIS_WIN = (time(12, 30), time(12, 54))            # print stamps, PT
LAG_MIN = 15                                        # spx_cash lagged-day shift
CLOSE_ES_BAR = time(12, 54)                         # bar start; ends 15:55 ET
OPEN_ES_BAR = time(6, 29)                           # bar start; ends 09:30 ET
OPEN_ES_FALLBACK_MIN = 5
STRIKE_STEP = 5.0
G3_MINUTE = time(6, 35)                             # 09:35 ET
G3_BAND = (0.5, 2.0)
MAX_CALLS_PER_MIN = 80
TRANSIENT_SLEEP_S = 60
TRANSIENT_MAX = 5


# ── strike placement ─────────────────────────────────────────────────────────

def _pt_to_utc_naive(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=PT).astimezone(UTC).replace(tzinfo=None)


def _es_close(conn, d: date, bar_start_pt: time, fallback_min: int = 0) -> float | None:
    """Close of the ES bar starting at bar_start_pt (PT) on d; else the latest within fallback_min."""
    hi = _pt_to_utc_naive(d, bar_start_pt)
    lo = hi - timedelta(minutes=fallback_min)
    r = conn.execute(
        "SELECT close FROM ironbeam_es_1m_bars WHERE datetime BETWEEN %s AND %s ORDER BY datetime DESC LIMIT 1",
        (lo, hi)).fetchone()
    return float(r[0]) if r else None


def _es_closes(conn, d: date, start_pt: time, end_pt: time) -> dict[datetime, float]:
    """{bar END minute in naive PT: close} for bars starting in [start_pt - 1, end_pt - 1] on d."""
    lo = _pt_to_utc_naive(d, start_pt) - timedelta(minutes=1)
    hi = _pt_to_utc_naive(d, end_pt) - timedelta(minutes=1)
    rows = conn.execute(
        "SELECT datetime, close FROM ironbeam_es_1m_bars WHERE datetime BETWEEN %s AND %s", (lo, hi)).fetchall()
    out = {}
    for ts, c in rows:
        end_pt_ = ts.replace(tzinfo=UTC).astimezone(PT).replace(tzinfo=None) + timedelta(minutes=1)
        out[end_pt_] = float(c)
    return out


def load_cash(conn, ts: list[date]) -> pd.DataFrame:
    """CR-BH SPX cash series covering each t (with a 7-day lead for the gross-guard reference)."""
    raws = []
    if len(ts) <= 20:
        for t in ts:
            raws.append(spx_cash.fetch_minutes(conn, t - timedelta(days=7), t))
    else:
        raws.append(spx_cash.fetch_minutes(conn, min(ts) - timedelta(days=7), max(ts)))
    raw = pd.concat(raws, ignore_index=True).drop_duplicates("minute").sort_values("minute")
    clean, _ = spx_cash.build_series(raw.reset_index(drop=True))
    return clean


def basis_on(conn, cash: pd.DataFrame, t: date) -> tuple[float | None, int]:
    """median(ES - SPX cash) over prints stamped 12:30-12:54 PT on t. Returns (basis, n)."""
    end = datetime.combine(t, BASIS_WIN[1])
    if spx_cash.is_lagged(t):
        end -= timedelta(minutes=LAG_MIN)               # shifted prints were stamped 15 min later
    start = datetime.combine(t, BASIS_WIN[0])
    c = cash[(cash["session_date"] == t) & (cash["minute"] >= start) & (cash["minute"] <= end)]
    if c.empty:
        return None, 0
    es = _es_closes(conn, t, BASIS_WIN[0], BASIS_WIN[1])
    diffs = [es[m.to_pydatetime()] - px for m, px in zip(c["minute"], c["px"]) if m.to_pydatetime() in es]
    if not diffs:
        return None, 0
    return float(pd.Series(diffs).median()), len(diffs)


def strikes_around(s: float) -> list[float]:
    k0 = STRIKE_STEP * round(s / STRIKE_STEP)
    return [k0 - STRIKE_STEP, k0, k0 + STRIKE_STEP]


def plan_night(conn, cash: pd.DataFrame, t: date, t1: date) -> dict:
    p = {"t": t, "t1": t1, "basis": None, "basis_n": 0, "S_c": None, "S_o": None,
         "K_c": [], "K_o": [], "plan_error": None}
    basis, n = basis_on(conn, cash, t)
    p["basis"], p["basis_n"] = basis, n
    if basis is None:
        p["plan_error"] = "no basis (no SPX cash or ES prints 12:30-12:54 PT on t)"
        return p
    es_c = _es_close(conn, t, CLOSE_ES_BAR)
    es_o = _es_close(conn, t1, OPEN_ES_BAR, OPEN_ES_FALLBACK_MIN)
    if es_c is not None:
        p["S_c"] = es_c - basis
        p["K_c"] = strikes_around(p["S_c"])
    if es_o is not None:
        p["S_o"] = es_o - basis
        p["K_o"] = strikes_around(p["S_o"])
    if es_c is None or es_o is None:
        p["plan_error"] = f"ES bar missing (close={es_c is not None}, open={es_o is not None})"
    return p


# ── fetch ────────────────────────────────────────────────────────────────────

class Pacer:
    def __init__(self, per_min: int):
        self.gap = 60.0 / per_min
        self.last = 0.0

    def wait(self):
        dt_ = _time.monotonic() - self.last
        if dt_ < self.gap:
            _time.sleep(self.gap - dt_)
        self.last = _time.monotonic()


def _pm_rows(conn, opra: str, start: datetime, end: datetime) -> tuple[int, int]:
    """(pm rows, non-pm rows) for opra in [start, end]."""
    r = conn.execute(
        "SELECT count(*) FILTER (WHERE expiry_tod = 'pm'), count(*) FILTER (WHERE expiry_tod IS DISTINCT FROM 'pm') "
        "FROM orats_options_minute WHERE opra_symbol = %s AND snapshot_pt BETWEEN %s AND %s",
        (opra, start, end)).fetchone()
    return int(r[0]), int(r[1])


def fetch_window(conn, pacer: Pacer, t1: date, d: date, win: tuple[time, time], strikes: list[float],
                 counters: dict, missing: list[dict], label: str, night: tuple[date, date]) -> dict:
    """Fetch one window for K0-5/K0/K0+5. Returns {opra: pm_rows} for the 6 contracts."""
    start, end = datetime.combine(d, win[0]), datetime.combine(d, win[1])
    n_min = int((end - start).total_seconds() // 60) + 1
    result = {}
    for k in strikes:
        call, put = format_opra(ROOT, t1, "C", k), format_opra(ROOT, t1, "P", k)
        have = {o: _pm_rows(conn, o, start, end)[0] for o in (call, put)}
        err = None
        if not all(v >= n_min for v in have.values()):
            for attempt in range(TRANSIENT_MAX + 1):
                try:
                    pacer.wait()
                    s = fetch_option_bars([call, put], start, end,
                                          source="historical_backfill", record_empty_windows=False)
                    counters["orats_calls"] += s.gaps_filled
                    counters["rows_inserted"] += s.bars_written
                    counters["cache_hit_contracts"] += s.cache_hits
                    break
                except OratsTransientError as e:
                    counters["transient_errors"] += 1
                    if attempt == TRANSIENT_MAX:
                        err = f"transient_exhausted: {str(e)[:80]}"
                        break
                    print(f"    transient ({e!s:.80}); sleeping {TRANSIENT_SLEEP_S}s", flush=True)
                    _time.sleep(TRANSIENT_SLEEP_S)
                except OratsPermanentError as e:
                    counters["orats_calls"] += 1
                    counters["permanent_errors"] += 1
                    msg = str(e)
                    err = "http_404" if "404" in msg else f"http_error: {msg[:80]}"
                    break
        else:
            counters["skipped_cached_windows"] += 1
        for o in (call, put):
            pm, other = _pm_rows(conn, o, start, end)
            result[o] = pm
            if pm == 0:
                if err:
                    reason = err
                elif other:
                    reason = "am_only"
                else:
                    covered = conn.execute(
                        "SELECT count(*) FROM orats_options_fetched_windows WHERE opra_symbol = %s "
                        "AND window_start_pt <= %s AND window_end_pt >= %s AND row_count > 0",
                        (o, start, end)).fetchone()[0]
                    reason = "covered_no_rows" if covered else "empty_response"
                missing.append({"t": night[0].isoformat(), "t1": night[1].isoformat(), "window": label,
                                "strike": k, "side": o[-9], "opra": o, "reason": reason})
    return result


# ── quote checks (G2 / G3) ───────────────────────────────────────────────────

def quote_check(conn, opras: list[str], start: datetime, end: datetime) -> dict:
    r = conn.execute(
        "SELECT count(*), "
        "count(*) FILTER (WHERE bid_price IS NULL OR ask_price IS NULL), "
        "count(*) FILTER (WHERE bid_price < 0 OR ask_price <= 0 OR bid_price > ask_price) "
        "FROM orats_options_minute WHERE opra_symbol = ANY(%s) AND expiry_tod = 'pm' "
        "AND snapshot_pt BETWEEN %s AND %s", (opras, start, end)).fetchone()
    return {"rows": int(r[0]), "null_quote": int(r[1]), "outside": int(r[2])}


def straddle_mid(conn, t1: date, k: float, at: datetime) -> float | None:
    rows = conn.execute(
        "SELECT option_type, bid_price, ask_price FROM orats_options_minute "
        "WHERE opra_symbol = ANY(%s) AND snapshot_pt = %s AND expiry_tod = 'pm'",
        ([format_opra(ROOT, t1, "C", k), format_opra(ROOT, t1, "P", k)], at)).fetchall()
    mids = {ot: (float(b) + float(a)) / 2 for ot, b, a in rows if b is not None and a is not None}
    return mids["C"] + mids["P"] if {"C", "P"} <= mids.keys() else None


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_straddle(f: float, k: float, v: float) -> float:
    """Black (r = 0) call + put at total variance v."""
    sv = math.sqrt(v)
    d1 = (math.log(f / k) + v / 2) / sv
    d2 = d1 - sv
    call = f * _ncdf(d1) - k * _ncdf(d2)
    put = k * _ncdf(-d2) - f * _ncdf(-d1)
    return call + put


def g3_rows(conn, plans: list[dict], vpx: pd.DataFrame) -> list[dict]:
    ref = {pd.Timestamp(r["d"]).date(): r for r in vpx.to_dict("records") if r["in_universe"]}
    out = []
    for p in plans:
        r = ref.get(p["t1"])
        if r is None or not p["K_o"]:
            continue
        k0 = p["K_o"][1]
        mine = straddle_mid(conn, p["t1"], k0, datetime.combine(p["t1"], G3_MINUTE))
        v = r.get("vpx_06:35")
        refpx = black_straddle(float(r["S"]), float(r["K"]), float(v)) if v and v > 0 else None
        ratio = mine / refpx if mine and refpx else None
        out.append({"t1": p["t1"].isoformat(), "K0": k0, "S_o": round(p["S_o"], 2), "crbp_K": float(r["K"]),
                    "crbp_S": float(r["S"]), "mine": mine, "crbp_ref": refpx, "ratio": ratio,
                    "same_strike": float(r["K"]) == k0})
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def load_nights() -> list[tuple[date, date]]:
    with open(NIGHTS_CSV) as f:
        nights = [(date.fromisoformat(r["t"]), date.fromisoformat(r["t1"])) for r in csv.DictReader(f)]
    assert all(t1 <= LAST_T1 for _, t1 in nights), "night after 2026-06-05 in the list"
    return nights


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CR-CY 1-DTE SPX ATM straddle quote backfill")
    ap.add_argument("--mode", choices=["step0", "full"], required=True)
    ap.add_argument("--vpx", default=str(REPO_ROOT / "scripts" / "output" / "cr_bp" / "vpx_primary.parquet"),
                    help="CR-BP vpx_primary.parquet (G3 reference)")
    ap.add_argument("--limit", type=int, default=None, help="full mode: first N nights only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_nights = load_nights()
    if args.mode == "step0":
        nights = [n for n in all_nights if n[1] in STEP0_NIGHTS]
        assert len(nights) == len(STEP0_NIGHTS), "a trial night is not in the universe"
    else:
        gates = LOG_DIR / "step0_gates.json"
        if not args.dry_run and not (gates.exists() and json.loads(gates.read_text()).get("pass") is True):
            sys.exit("ERROR: full mode needs scripts/logs/cr_cy/step0_gates.json with pass = true.")
        nights = all_nights[: args.limit] if args.limit else all_nights
    vpx = pd.read_parquet(args.vpx)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    print(f"CR-CY  mode={args.mode}  nights={len(nights)}  dry_run={args.dry_run}", flush=True)

    t_load = _time.monotonic()
    cash = load_cash(conn, sorted({t for t, _ in nights}))
    print(f"SPX cash loaded: {len(cash)} prints in {_time.monotonic() - t_load:.0f}s", flush=True)
    plans = [plan_night(conn, cash, t, t1) for t, t1 in nights]
    plan_path = LOG_DIR / f"{args.mode}_plan.csv"
    pd.DataFrame([{**p, "K_c": "/".join(f"{k:g}" for k in p["K_c"]), "K_o": "/".join(f"{k:g}" for k in p["K_o"])}
                  for p in plans]).to_csv(plan_path, index=False)
    n_err = sum(1 for p in plans if p["plan_error"])
    print(f"plan written: {plan_path.name}  plan errors={n_err}", flush=True)
    if args.dry_run:
        for p in plans[:10]:
            print(f"  {p['t']} -> {p['t1']}  basis={p['basis']}  S_c={p['S_c']}  K_c={p['K_c']}  "
                  f"S_o={p['S_o']}  K_o={p['K_o']}  {p['plan_error'] or ''}")
        return

    counters = {"orats_calls": 0, "rows_inserted": 0, "cache_hit_contracts": 0, "skipped_cached_windows": 0,
                "transient_errors": 0, "permanent_errors": 0}
    missing: list[dict] = []
    per_window: list[dict] = []
    pacer = Pacer(MAX_CALLS_PER_MIN)
    run_start = conn.execute("SELECT now()").fetchone()[0]

    with backfill_run(conn, CR_ID) as run_id:
        print(f"run_id={run_id}", flush=True)
        t0 = _time.monotonic()
        for i, p in enumerate(plans, 1):
            night = (p["t"], p["t1"])
            for label, d, win, ks in (("close", p["t"], CLOSE_WIN, p["K_c"]), ("open", p["t1"], OPEN_WIN, p["K_o"])):
                if not ks:
                    per_window.append({"t": p["t"].isoformat(), "t1": p["t1"].isoformat(), "window": label,
                                       "contracts_with_rows": 0, "rows": 0, "null_quote": 0, "outside": 0,
                                       "note": p["plan_error"]})
                    continue
                res = fetch_window(conn, pacer, p["t1"], d, win, ks, counters, missing, label, night)
                q = quote_check(conn, list(res), datetime.combine(d, win[0]), datetime.combine(d, win[1]))
                per_window.append({"t": p["t"].isoformat(), "t1": p["t1"].isoformat(), "window": label,
                                   "strikes": "/".join(f"{k:g}" for k in ks),
                                   "contracts_with_rows": sum(1 for v in res.values() if v > 0), **q, "note": ""})
            if i % 10 == 0 or i == len(plans):
                update_run_progress(conn, run_id, counters["rows_inserted"])
                el = _time.monotonic() - t0
                print(f"  [{i}/{len(plans)}] {p['t1']}  calls={counters['orats_calls']}  rows={counters['rows_inserted']}"
                      f"  missing={len(missing)}  {el / 60:.1f} min", flush=True)

        pd.DataFrame(per_window).to_csv(LOG_DIR / f"{args.mode}_windows.csv", index=False)
        mpath = LOG_DIR / ("missing.csv" if args.mode == "full" else "step0_missing.csv")
        pd.DataFrame(missing, columns=["t", "t1", "window", "strike", "side", "opra", "reason"]).to_csv(mpath, index=False)

        # ── gates / smoke ──
        w = pd.DataFrame(per_window)
        g1_fail = w[w["contracts_with_rows"] < 5]
        g2_outside = int(w["outside"].sum())
        g3 = g3_rows(conn, plans, vpx)
        g3_bad = [r for r in g3 if r["ratio"] is None or not (G3_BAND[0] <= r["ratio"] <= G3_BAND[1])]
        g4_empty = conn.execute(
            "SELECT count(*) FROM orats_options_fetched_windows WHERE row_count = 0 AND fetched_at >= %s",
            (run_start,)).fetchone()[0]
        by_night = w.groupby(["t", "t1"])["contracts_with_rows"].agg(["sum", "min", "max"])
        complete = int((by_night["sum"] == 12).sum())
        none_ = int((by_night["max"] == 0).sum())
        partial = len(by_night) - complete - none_
        n_am_only = sum(1 for r in missing if r["reason"] == "am_only")
        smoke = {
            "mode": args.mode, "nights": len(plans), "plan_errors": n_err,
            "nights_complete": complete, "nights_partial": partial, "nights_missing": none_,
            "windows": len(w), "windows_lt5_contracts": len(g1_fail),
            "G1_pass": len(g1_fail) == 0,
            "G2_rows_checked": int(w["rows"].sum()), "G2_null_quote_rows": int(w["null_quote"].sum()),
            "G2_outside_rows": g2_outside, "G2_pass": g2_outside == 0,
            "G3_overlap_nights": len(g3), "G3_out_of_band": len(g3_bad),
            "G3_ratio_median": float(pd.Series([r["ratio"] for r in g3 if r["ratio"]]).median()) if g3 else None,
            "G3_pass": len(g3) > 0 and not g3_bad,
            "G4_empty_windows_written": int(g4_empty), "G4_pass": g4_empty == 0,
            "missing_contract_windows": len(missing),
            "missing_by_reason": pd.Series([m["reason"] for m in missing]).value_counts().to_dict() if missing else {},
            "am_only": n_am_only,
            **counters,
            "run_id": run_id,
        }
        smoke["pass"] = all(smoke[g] for g in ("G1_pass", "G2_pass", "G3_pass", "G4_pass"))
        (LOG_DIR / f"{args.mode}_g3.json").write_text(json.dumps(g3, indent=2, default=str))
        if args.mode == "step0":
            (LOG_DIR / "step0_gates.json").write_text(json.dumps(smoke, indent=2, default=str))
        else:
            (LOG_DIR / "full_smoke.json").write_text(json.dumps(smoke, indent=2, default=str))
        verdict = "all gates pass" if smoke["pass"] else "GATE FAIL: " + ",".join(
            g for g in ("G1_pass", "G2_pass", "G3_pass", "G4_pass") if not smoke[g])
        update_run_smoke(conn, run_id, smoke,
                         f"CR-CY {args.mode}: {complete}/{len(plans)} nights complete, {partial} partial, "
                         f"{none_} missing; {counters['orats_calls']} ORATS calls; {verdict}")
        print(json.dumps(smoke, indent=2, default=str), flush=True)
        for r in g3:
            print(f"  G3 {r['t1']}  K0={r['K0']:g} (CR-BP K {r['crbp_K']:g})  mine={r['mine']}  "
                  f"ref={r['crbp_ref'] and round(r['crbp_ref'], 3)}  ratio={r['ratio'] and round(r['ratio'], 3)}")


if __name__ == "__main__":
    main()
