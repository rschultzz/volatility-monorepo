"""CR-DQ Step 0 — baseline for the box panel's percentile badges (spec item 5): source, coverage, length buckets,
time-of-day split, and the 3-window source comparison (5d). Read-only (PGOPTIONS read-only transaction).

Windows match the box readout (`flowV3.anchorReadout`): anchor minute a, length L minutes; Σ delta and Σ volume over
minutes a+1 .. a+L (delta = buy − sell, volume = `volume` incl. unknown side, as the v2 rows); progress = close[a+L] −
close[a]. |Δ share| = |Σ delta| ÷ Σ volume; contracts/pt = Σ volume ÷ max(|progress|, 0.25) (one tick floor, so a flat
window is the most effortful, not undefined). A window counts only when every minute a .. a+L is present.

Train sessions only (B4): NYSE days 2026-01-02 → 2026-06-05 (Ironbeam 1-minute flow starts 2025-12-26; R-a / R-b /
holdout / forward log not read). Left out: FOMC decision days (A22) and the March 2026 expiry week 03-16 → 03-20
(Ironbeam flow stayed on the dying H26 contract). Per session and source, the contract with the larger RTH volume.

Usage: apps/web/.venv/bin/python scripts/cr_dq_step0.py [--out scripts/logs/cr-dq/step0.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
sys.path[:0] = [REPO, os.path.join(REPO, "apps", "web")]

import psycopg  # noqa: E402

from modules.SetupReview import routes as review_routes  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
FIRST, LAST = dt.date(2026, 1, 2), dt.date(2026, 6, 5)
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
EXCLUDE = {dt.date.fromisoformat(d) for d in FOMC_2026} | {dt.date(2026, 3, 16) + dt.timedelta(days=k) for k in range(5)}
TABLES = {"ironbeam": "ironbeam_es_flow_1m", "databento": "es_flow_1m_databento"}
RTH_MIN = 390
MAX_L = 240
BUCKETS = [("5", 3, 7), ("10", 8, 12), ("15", 13, 22), ("30", 23, 45), ("60", 46, 75), ("90+", 76, MAX_L)]
TOD = [("open 06:30–07:30", 0, 60), ("mid 07:30–11:00", 60, 270), ("late 11:00–13:00", 270, RTH_MIN)]
TICK = 0.25
# 5d: three fixed train windows (anchor PT, minutes), chosen before looking at any value.
EXAMPLES = [("2026-02-10", "07:05", 25), ("2026-04-15", "09:12", 12), ("2026-05-20", "10:40", 55)]
# CR-CA measurement (open question ironbeam-es-flow-1s-undercounts-trades): Ironbeam / Databento per minute.
IB_DELTA_SLOPE, IB_VOL_RATIO = 0.71, 0.943

SQL = """
SELECT symbol, (minute_utc AT TIME ZONE 'America/Los_Angeles')::date AS d,
       extract(epoch FROM minute_utc)::bigint AS t, close::float8, (buy_vol - sell_vol)::float8, volume::float8
FROM {table}
WHERE minute_utc >= %s AND minute_utc < %s
  AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
  AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
"""


def sessions():
    d, out = FIRST, []
    while d <= LAST:
        if is_trading_day(d) and d not in EXCLUDE:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def rth_open_utc(d):
    return int(dt.datetime.combine(d, dt.time(6, 30), PT).timestamp())


def load(conn, source, days):
    lo = dt.datetime.combine(FIRST, dt.time(0), PT)
    hi = dt.datetime.combine(LAST + dt.timedelta(days=1), dt.time(0), PT)
    rows = conn.execute(SQL.format(table=TABLES[source]), (lo, hi)).fetchall()
    by = {}
    for sym, d, t, c, dl, v in rows:
        by.setdefault(d, {}).setdefault(sym, []).append((t, c, dl, v))
    out = {}
    for d in days:
        syms = by.get(d)
        if not syms:
            continue
        sym = max(syms, key=lambda s: sum(r[3] for r in syms[s]))
        arr = np.full((4, RTH_MIN), np.nan)
        t0 = rth_open_utc(d)
        for t, c, dl, v in syms[sym]:
            i = (t - t0) // 60
            if 0 <= i < RTH_MIN:
                arr[:, i] = (1, c, dl, v)
        out[d] = (sym, arr)
    return out


def windows(arr):
    """All (anchor, L) windows with every minute present: arrays anchor, L, |share|, cpp, share."""
    present = ~np.isnan(arr[0])
    c, dl, v = np.nan_to_num(arr[1]), np.nan_to_num(arr[2]), np.nan_to_num(arr[3])
    cd, cv = np.concatenate([[0], np.cumsum(dl)]), np.concatenate([[0], np.cumsum(v)])
    cp = np.concatenate([[0], np.cumsum(~present)])
    A, Ls, S, C = [], [], [], []
    for L in range(BUCKETS[0][1], MAX_L + 1):
        a = np.arange(0, RTH_MIN - L)
        ok = (cp[a + L + 1] - cp[a]) == 0
        a = a[ok]
        if not len(a):
            continue
        sd = cd[a + L + 1] - cd[a + 1]
        sv = cv[a + L + 1] - cv[a + 1]
        good = sv > 0
        a, sd, sv = a[good], sd[good], sv[good]
        dp = c[a + L] - c[a]
        A.append(a)
        Ls.append(np.full(len(a), L))
        S.append(sd / sv)
        C.append(sv / np.maximum(np.abs(dp), TICK))
    if not A:
        return None
    return np.concatenate(A), np.concatenate(Ls), np.abs(np.concatenate(S)), np.concatenate(C), np.concatenate(S)


def bucket_of(minutes):
    for name, lo, hi in BUCKETS:
        if lo <= minutes <= hi:
            return name
    return BUCKETS[-1][0] if minutes > MAX_L else None


def pct(grid, v):
    """Percentile of v in a 101-point quantile grid (0..100), interpolated; the badge's lookup."""
    return float(np.interp(v, grid, np.arange(101)))


def window_stats(arr, a, L):
    seg = arr[:, a:a + L + 1]
    if np.isnan(seg[0]).any():
        return None
    sd, sv = seg[2, 1:].sum(), seg[3, 1:].sum()
    dp = seg[1, -1] - seg[1, 0]
    return {"delta": sd, "vol": sv, "dp": dp, "share": sd / sv, "cpp": sv / max(abs(dp), TICK)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "scripts", "logs", "cr-dq", "step0.json"))
    args = ap.parse_args()
    days = sessions()
    try:
        with psycopg.connect(review_routes._normalize_db_url(os.environ["BACKFILL_DATABASE_URL"]), connect_timeout=20) as conn:
            data = {s: load(conn, s, days) for s in TABLES}
    except Exception as e:  # never print the connection string
        print("DB error:", type(e).__name__)
        sys.exit(1)

    report = {"sessions_eligible": len(days), "excluded": sorted(str(d) for d in EXCLUDE if FIRST <= d <= LAST), "sources": {}}
    grids = {}
    for src, per in data.items():
        allw = {k: [] for k in ("a", "L", "abs", "cpp")}
        missing = []
        for d, (sym, arr) in sorted(per.items()):
            missing.append(int(np.isnan(arr[0]).sum()))
            w = windows(arr)
            if w is None:
                continue
            for k, x in zip(("a", "L", "abs", "cpp"), w[:4]):
                allw[k].append(x)
        allw = {k: np.concatenate(v) for k, v in allw.items()}
        info = {"sessions": len(per), "first": str(min(per)), "last": str(max(per)),
                "missing_minutes_per_session": {"median": float(np.median(missing)), "max": int(max(missing)),
                                                "sessions_with_any": int(sum(m > 0 for m in missing))},
                "buckets": {}}
        grids[src] = {}
        for name, lo, hi in BUCKETS:
            m = (allw["L"] >= lo) & (allw["L"] <= hi)
            g_abs = np.percentile(allw["abs"][m], np.arange(101))
            g_cpp = np.percentile(allw["cpp"][m], np.arange(101))
            grids[src][name] = (g_abs, g_cpp)
            tod = {}
            for tname, t0, t1 in TOD:
                mt = m & (allw["a"] >= t0) & (allw["a"] < t1)
                tod[tname] = {
                    "n": int(mt.sum()),
                    "abs_share_p50_p80_p95": [round(float(x), 4) for x in np.percentile(allw["abs"][mt], [50, 80, 95])],
                    "cpp_p50_p80_p95": [round(float(x)) for x in np.percentile(allw["cpp"][mt], [50, 80, 95])],
                    # where the pooled p80 / p95 cut-offs land inside this time-of-day group
                    "pooled_p80_p95_abs_share_in_group": [round(float((allw["abs"][mt] <= g_abs[q]).mean() * 100), 1) for q in (80, 95)],
                    "pooled_p80_p95_cpp_in_group": [round(float((allw["cpp"][mt] <= g_cpp[q]).mean() * 100), 1) for q in (80, 95)],
                }
            info["buckets"][name] = {
                "lengths": [lo, hi], "n_windows": int(m.sum()),
                "abs_share_p50_p80_p95": [round(float(g_abs[q]), 4) for q in (50, 80, 95)],
                "cpp_p50_p80_p95": [round(float(g_cpp[q])) for q in (50, 80, 95)],
                "time_of_day": tod,
            }
        report["sources"][src] = info

    # 5d: same window, both sources. Percentile vs own-source baseline; Ironbeam vs the Databento baseline raw and
    # adjusted by CR-CA's ratios (share × 0.943 / 0.71, contracts/pt ÷ 0.943).
    ex = []
    for ds, hhmm, L in EXAMPLES:
        d = dt.date.fromisoformat(ds)
        a = (int(hhmm[:2]) * 60 + int(hhmm[3:])) - (6 * 60 + 30)
        b = bucket_of(L)
        row = {"window": f"{ds} {hhmm} + {L} m", "bucket": b}
        for src in TABLES:
            st = window_stats(data[src][d][1], a, L) if d in data[src] else None
            if st is None:
                row[src] = None
                continue
            ga, gc = grids[src][b]
            row[src] = {"delta": round(st["delta"]), "vol": round(st["vol"]), "pts": st["dp"], "share_pct": round(st["share"] * 100, 2),
                        "cpp": round(st["cpp"]), "share_p": round(pct(ga, abs(st["share"])), 1), "cpp_p": round(pct(gc, st["cpp"]), 1)}
        ib, db = row.get("ironbeam"), row.get("databento")
        if ib and db:
            ga, gc = grids["databento"][b]
            s_ib = ib["share_pct"] / 100
            row["ironbeam_vs_databento_baseline_raw"] = {"share_p": round(pct(ga, abs(s_ib)), 1), "cpp_p": round(pct(gc, ib["cpp"]), 1)}
            row["ironbeam_vs_databento_baseline_adjusted"] = {
                "share_p": round(pct(ga, abs(s_ib) * IB_VOL_RATIO / IB_DELTA_SLOPE), 1),
                "cpp_p": round(pct(gc, ib["cpp"] / IB_VOL_RATIO), 1)}
        ex.append(row)
    report["examples_5d"] = ex

    # 5d, all windows: how far an Ironbeam box's percentile moves if read against the Databento baseline (raw /
    # adjusted) instead of its own-source baseline. Common sessions, 1 in 7 anchors, lengths at each bucket's ends + mid.
    diffs = {"own_vs_db_raw_share": [], "own_vs_db_raw_cpp": [], "own_vs_db_adj_share": [], "own_vs_db_adj_cpp": [],
             "ib_own_vs_db_own_share": [], "ib_own_vs_db_own_cpp": []}
    for d in sorted(set(data["ironbeam"]) & set(data["databento"])):
        ia, da = data["ironbeam"][d][1], data["databento"][d][1]
        for name, lo, hi in BUCKETS:
            for L in sorted({lo, (lo + hi) // 2, hi}):
                for a in range(0, RTH_MIN - L, 7):
                    si, sd = window_stats(ia, a, L), window_stats(da, a, L)
                    if si is None or sd is None or si["vol"] <= 0 or sd["vol"] <= 0:
                        continue
                    gia, gic = grids["ironbeam"][name]
                    gda, gdc = grids["databento"][name]
                    own_s, own_c = pct(gia, abs(si["share"])), pct(gic, si["cpp"])
                    diffs["own_vs_db_raw_share"].append(pct(gda, abs(si["share"])) - own_s)
                    diffs["own_vs_db_raw_cpp"].append(pct(gdc, si["cpp"]) - own_c)
                    diffs["own_vs_db_adj_share"].append(pct(gda, abs(si["share"]) * IB_VOL_RATIO / IB_DELTA_SLOPE) - own_s)
                    diffs["own_vs_db_adj_cpp"].append(pct(gdc, si["cpp"] / IB_VOL_RATIO) - own_c)
                    diffs["ib_own_vs_db_own_share"].append(pct(gda, abs(sd["share"])) - own_s)
                    diffs["ib_own_vs_db_own_cpp"].append(pct(gdc, sd["cpp"]) - own_c)
    report["percentile_shift_5d"] = {k: {"n": len(v), "mean": round(float(np.mean(v)), 1), "median_abs": round(float(np.median(np.abs(v))), 1),
                                         "p90_abs": round(float(np.percentile(np.abs(v), 90)), 1)} for k, v in diffs.items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
