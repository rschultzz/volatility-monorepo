#!/usr/bin/env python3
"""CR-BP Step 0 — diagnosis + the P3 gate for T-S1a′. READ-ONLY: SELECTs only, no run row, no API calls.

Spec: specs/CR-BP-session-premium-live-open.md (Step 0 list items 1–11; P3 (a)–(g) and its decision rule).

  1   CR-BO Q1–Q7 re-run with the CR-BO script logic into scripts/output/cr_bp/; gate: identical step0_pairs.csv
  2   P3(a)(b)(g) CR-AS capture coverage, primary subset (same-strike C+P within 10 pts of S), median |K − S|
  3   P3(c) secondary set (±0.5 IM short put + short call strangle)
  4   P3(d)(g) atmiv²τ / V_px and vol50²τ / V_px at 09:31 / 09:33 / 09:35 / 09:45 ET; mid_iv and closed-form diagnostics
  5   P3(e) close side: atmiv / vol50 at 15:59 by bucket
  6   P3(f) atmiv minute-to-minute 09:31 → 09:40
  7   vol50 / atmiv per minute on five random days (rng 17)
  8   P3 decision line
  9   P4 bid–ask lower-bound proxy
  10  P6 priors vs CR-BO
  11  C6 defaults

V_px (P3(b)(c)(g)): Black (F = S, r = q = 0 — the harness prices from quote mids and settles on intrinsic,
it carries no r or q) in total variance v = σ²τ, one unknown:
  primary   call(K, v) + put(K, v) = straddle mid      (K-aware; closed form kept as a cross-check, exact at K = S)
  secondary put(K_p, v) + call(K_c, v) = strangle mid
S = CR-AS's placement spot: `spot_price` of the first `orats_monies_minute` row ≥ 06:33 PT (dte ascending), ≤ 06:40
(scripts/cr_as_capture_0dte_condor_legs.py SPX_OPEN_SQL / OPEN_SNAPSHOT_CEIL).

Usage: apps/web/.venv/bin/python -u scripts/cr_bp_step0.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
sys.path.insert(0, str(REPO_ROOT))

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from packages.shared.backtest.quote_validity import leg_quote_is_valid
from packages.shared.spx_cash import is_lagged
from scripts import cr_bo_common as C
from scripts import cr_bo_step0 as BO

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bp"
CRAS_JSON = REPO_ROOT / "scripts" / "logs" / "cr_as_analysis_full_fee0.65.json"
MINUTES_PT = ["06:31", "06:33", "06:35", "06:45"]          # 09:31 / 09:33 / 09:35 / 09:45 ET
PRIMARY_MAX_DIST = 10.0
PRIMARY_MIN_N = 100
TOL = 0.10
PRIORS = {"S1": 0.05, "S2": 0.0, "S7": 0.14}

LEGS_SQL = """
SELECT trade_date_d, strike, option_type, snapshot_pt, bid_price, ask_price, mid_iv
FROM orats_options_minute
WHERE ticker = 'SPX' AND expiry_tod = 'pm' AND expir_date_d = trade_date_d
  AND trade_date_d = ANY(%s) AND snapshot_pt::time BETWEEN '06:30' AND '06:45'
"""
SPOT_SQL = """
SELECT DISTINCT ON (trade_date) trade_date, spot_price, stock_price, snapshot_pt
FROM orats_monies_minute
WHERE ticker = 'SPX' AND trade_date = ANY(%s) AND snapshot_pt::time >= '06:33' AND spot_price IS NOT NULL
ORDER BY trade_date, snapshot_pt ASC, dte ASC
"""
OPEN_MONIES_SQL = """
SELECT trade_date, expir_date, expiry_tod, dte, snapshot_pt, atmiv, vol50
FROM orats_monies_minute
WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND dte BETWEEN 1 AND 16
  AND (snapshot_pt::time BETWEEN '06:30' AND '06:50' OR snapshot_pt::time BETWEEN '12:50' AND '12:59')
"""


def black(F, K, v, cp):
    if v <= 0:
        return max(F - K, 0.0) if cp == "C" else max(K - F, 0.0)
    s = np.sqrt(v); d1 = (np.log(F / K) + v / 2) / s; d2 = d1 - s
    return F * norm.cdf(d1) - K * norm.cdf(d2) if cp == "C" else K * norm.cdf(-d2) - F * norm.cdf(-d1)


def solve_v(price, F, legs):
    """Total variance v with Σ black(F, K, v, cp) = price; NaN if price ≤ intrinsic or no root."""
    f = lambda v: sum(black(F, K, v, cp) for K, cp in legs) - price
    if f(1e-12) >= 0:
        return np.nan
    try:
        return brentq(f, 1e-12, 0.5)
    except ValueError:
        return np.nan


def closed_form_v(straddle, S):
    x = (1 + straddle / (2 * S)) / 2
    return (2 * norm.ppf(x)) ** 2 if 0.5 < x < 1 else np.nan


def q(s, a=.25, b=.75):
    s = pd.Series(s).dropna()
    return (s.median(), s.quantile(a), s.quantile(b), len(s)) if len(s) else (np.nan, np.nan, np.nan, 0)


def fq(t, d=3):
    m, a, b, n = t
    return "—" if n == 0 else f"{m:.{d}f} [{a:.{d}f}–{b:.{d}f}] (n {n})"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------ item 1: CR-BO Q1–Q7 re-run, same logic, new output dir
    BO.OUT = OUT_DIR / "step0_crbo_rerun.md"
    BO.main()
    a = pd.read_csv(REPO_ROOT / "scripts" / "output" / "cr_bo" / "step0_pairs.csv")
    b = pd.read_csv(OUT_DIR / "step0_crbo_rerun_pairs.csv") if (OUT_DIR / "step0_crbo_rerun_pairs.csv").exists() else pd.read_csv(OUT_DIR / "step0_pairs.csv")
    gate1 = a.equals(b)
    b.to_csv(OUT_DIR / "step0_pairs.csv", index=False)

    L: list[str] = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    say(f"# CR-BP Step 0 — {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Read-only, SELECTs only. Train only; DAILY-CAPTURE / CR-AU rows (≥ 2026-09-08) not read (C2). Times ET unless tagged.\n")
    say("## Item 1 — CR-BO Q1–Q7 re-run (gate)\n")
    fin = b[b.keep]
    say(f"Re-run of `scripts/cr_bo_step0.py` logic → `scripts/output/cr_bp/step0_crbo_rerun.md`. `step0_pairs.csv` identical to CR-BO's: **{gate1}**. "
        f"ES-clean nights {len(fin)}; S1 {int(fin.S1_ok.sum())} / S2 {int(fin.S2_ok.sum())} / S7 {int(fin.S7_ok.sum())} "
        f"(CR-BO: 621; 500 / 498 / 517).\n")
    if not gate1:
        say("**GATE 1 FAILED — STOP.**"); (OUT_DIR / "step0.md").write_text("\n".join(L) + "\n"); sys.exit(1)

    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    cras = json.load(open(CRAS_JSON))["results"]
    boxes = {}
    for r in cras:
        d = date.fromisoformat(r["date"])
        if d <= C.TRAIN_END and r["box"] == "pm0.5":
            k = r["strikes"]; boxes[d] = dict(Kp=k[1], Kc=k[2], im=r["im"])
    cras_dates = sorted({date.fromisoformat(r["date"]) for r in cras if date.fromisoformat(r["date"]) <= C.TRAIN_END})
    print(f"CR-AS train dates {len(cras_dates)}", flush=True)
    legs = pd.read_sql(LEGS_SQL, conn, params=(cras_dates,))
    legs["hm"] = legs.snapshot_pt.dt.strftime("%H:%M")
    spot = pd.read_sql(SPOT_SQL, conn, params=([d.isoformat() for d in cras_dates],))
    spot["d"] = pd.to_datetime(spot.trade_date).dt.date
    spot = spot[spot.snapshot_pt.dt.time <= time(6, 40)].set_index("d")
    print(f"legs rows {len(legs)}, spots {len(spot)}", flush=True)

    mon = pd.concat([pd.read_sql(OPEN_MONIES_SQL, conn, params=(lo.isoformat(), hi.isoformat()))
                     for lo, hi in BO.year_chunks(C.TRAIN_START, C.TRAIN_END)], ignore_index=True)
    mon = mon[mon.expiry_tod == "pm"].copy()
    mon["td"] = pd.to_datetime(mon.trade_date).dt.date; mon["ex"] = pd.to_datetime(mon.expir_date).dt.date
    mon["hm"] = mon.snapshot_pt.dt.strftime("%H:%M")
    mkey = {(r.td, r.ex, r.hm): r for r in mon.itertuples()}

    # quote lookup
    Q = {}
    for r in legs.itertuples():
        Q[(r.trade_date_d, float(r.strike), r.option_type, r.hm)] = r

    def ok(d, K, cp, hm):
        r = Q.get((d, float(K), cp, hm)); return r is not None and leg_quote_is_valid(r.bid_price, r.ask_price)

    def mid(d, K, cp, hm):
        r = Q[(d, float(K), cp, hm)]; return (r.bid_price + r.ask_price) / 2

    universe_t1 = set(pd.to_datetime(fin[fin.S1_ok].t1).dt.date)
    rows_p, rows_s = [], []
    strikes_by_day = legs.groupby("trade_date_d").strike.apply(lambda s: sorted(set(map(float, s))))
    for d in cras_dates:
        if d not in spot.index:
            continue
        S = float(spot.loc[d, "spot_price"]); S_alt = float(spot.loc[d, "stock_price"])
        # primary: same-strike C+P within 10 pts of S, valid at all four minutes; nearest such strike
        cands = [K for K in strikes_by_day.get(d, []) if abs(K - S) <= PRIMARY_MAX_DIST
                 and all(ok(d, K, "C", m) and ok(d, K, "P", m) for m in MINUTES_PT)]
        if cands:
            K = min(cands, key=lambda k: abs(k - S))
            rec = dict(d=d, S=S, S_alt=S_alt, K=K, dist=K - S, lagged=is_lagged(d), in_universe=d in universe_t1)
            for m in MINUTES_PT:
                st = mid(d, K, "C", m) + mid(d, K, "P", m)
                rec[f"vpx_{m}"] = solve_v(st, S, [(K, "C"), (K, "P")])
                rec[f"vcf_{m}"] = closed_form_v(st, S)
                rec[f"vpx_alt_{m}"] = solve_v(st, S_alt, [(K, "C"), (K, "P")])
                rec[f"midiv_{m}"] = np.nanmean([Q[(d, K, "C", m)].mid_iv or np.nan, Q[(d, K, "P", m)].mid_iv or np.nan])
                rec[f"w_{m}"] = (Q[(d, K, "C", m)].ask_price - Q[(d, K, "C", m)].bid_price) + (Q[(d, K, "P", m)].ask_price - Q[(d, K, "P", m)].bid_price)
            rows_p.append(rec)
        bx = boxes.get(d)
        if bx and all(ok(d, bx["Kp"], "P", m) and ok(d, bx["Kc"], "C", m) for m in MINUTES_PT):
            rec = dict(d=d, S=S, Kp=bx["Kp"], Kc=bx["Kc"], dp=bx["Kp"] - S, dc=bx["Kc"] - S, im=bx["im"], lagged=is_lagged(d), in_universe=d in universe_t1)
            for m in MINUTES_PT:
                st = mid(d, bx["Kp"], "P", m) + mid(d, bx["Kc"], "C", m)
                rec[f"vpx_{m}"] = solve_v(st, S, [(bx["Kp"], "P"), (bx["Kc"], "C")])
                rec[f"midiv_{m}"] = np.nanmean([Q[(d, bx["Kp"], "P", m)].mid_iv or np.nan, Q[(d, bx["Kc"], "C", m)].mid_iv or np.nan])
                rec[f"w_{m}"] = (Q[(d, bx["Kp"], "P", m)].ask_price - Q[(d, bx["Kp"], "P", m)].bid_price) + (Q[(d, bx["Kc"], "C", m)].ask_price - Q[(d, bx["Kc"], "C", m)].bid_price)
            rows_s.append(rec)
    P = pd.DataFrame(rows_p); Sx = pd.DataFrame(rows_s)

    # monies ratios
    def add_ratios(df):
        for m in MINUTES_PT:
            et = (datetime.combine(date(2000, 1, 1), time.fromisoformat(m)) + timedelta(hours=3)).time()
            a_, v_, n_ = [], [], []
            for r in df.to_dict("records"):
                mr = mkey.get((r["d"], r["d"], m))
                tau = C.tau_years(datetime.combine(r["d"], et), r["d"])
                vp = r[f"vpx_{m}"]
                if mr is None or not (vp > 0):
                    a_.append(np.nan); v_.append(np.nan); n_.append(np.nan); continue
                a_.append(mr.atmiv ** 2 * tau / vp if mr.atmiv and mr.atmiv > 0 else np.nan)
                v_.append(mr.vol50 ** 2 * tau / vp if mr.vol50 and mr.vol50 > 0 else np.nan)
                mi = r[f"midiv_{m}"]
                n_.append(mi ** 2 * tau / vp if mi and mi > 0 else np.nan)
            df[f"r_atm_{m}"] = a_; df[f"r_v50_{m}"] = v_; df[f"r_mid_{m}"] = n_
        return df
    if len(P):
        P = add_ratios(P)
        for m in MINUTES_PT:
            P[f"r_cf_{m}"] = P[f"vcf_{m}"] / P[f"vpx_{m}"]
    if len(Sx):
        Sx = add_ratios(Sx)

    say("## Item 2 — P3(a)(b)(g): CR-AS capture coverage and the primary subset\n")
    say(f"- **S** = `orats_monies_minute.spot_price` at the first row ≥ 06:33 PT (dte ascending), ≤ 06:40 PT — `scripts/cr_as_capture_0dte_condor_legs.py` `SPX_OPEN_SQL` / `OPEN_SNAPSHOT_CEIL` (CR-AS Step 0 amendment 1). "
        f"S found on {len(spot)} of {len(cras_dates)} CR-AS train dates (source of the date list: `{CRAS_JSON.relative_to(REPO_ROOT)}`, run `85b690ec`).")
    say(f"- **r, q** = 0: the harness (`packages/shared/backtest/harness.py`) prices from quote mids and settles on intrinsic; it carries no rate or dividend. F = S.")
    say(f"- 0DTE PM leg rows in 06:30–06:45 PT on those dates: {len(legs)} (both C and P are stored per fetched strike — the ORATS option endpoint returns one row with both sides).")
    say(f"- **Primary subset** (same-strike C + P, |K − S| ≤ {PRIMARY_MAX_DIST:.0f}, CR-AN-valid at 06:31 / 06:33 / 06:35 / 06:45 PT): **n = {len(P)}**; "
        f"in the CR-BO universe as t+1 (S1 ok): {int(P.in_universe.sum()) if len(P) else 0}; on a 2023 `spot_price`-lagged day: {int(P.lagged.sum()) if len(P) else 0}.")
    if len(P):
        say(f"- **median |K − S| = {P.dist.abs().median():.2f} pts** (IQR {P.dist.abs().quantile(.25):.2f}–{P.dist.abs().quantile(.75):.2f}; K − S median {P.dist.median():+.2f}). "
            f"By year: {dict(Counter(d.year for d in P.d))}. IM on primary days (CR-AS `im`) median {np.median([boxes[d]['im'] for d in P.d if d in boxes]):.1f} vs all CR-AS days {np.median([v['im'] for v in boxes.values()]):.1f} pts.")
        nan_p = int(P[[f'vpx_{m}' for m in MINUTES_PT]].isna().any(axis=1).sum())
        say(f"- V_px solve failures (mid ≤ intrinsic or no root) on primary rows: {nan_p}.")
    say("")

    say("## Item 3 — P3(c): secondary set (±0.5 IM strangle)\n")
    say(f"- All CR-AS train dates with both ±0.5 IM short legs CR-AN-valid at the four minutes: **n = {len(Sx)}**; in the CR-BO universe as t+1 (S1 ok): {int(Sx.in_universe.sum()) if len(Sx) else 0}; lagged-spot days {int(Sx.lagged.sum()) if len(Sx) else 0}.")
    if len(Sx):
        say(f"- K_p − S median {Sx.dp.median():+.1f}, K_c − S median {Sx.dc.median():+.1f} pts. Wing-strike implied variance (carries the smile at ±0.5 IM; not ATM). "
            f"Solve failures: {int(Sx[[f'vpx_{m}' for m in MINUTES_PT]].isna().any(axis=1).sum())}.")
    say("")

    say("## Item 4 — P3(d)(g): monies variance / V_px at the open (0DTE; only 0DTE is captured)\n")
    say("τ per CR-BO L3 (calendar to 16:00 ET / 365) for the monies side only; V_px carries no τ. Median [IQR] (n).\n")
    for name, df, extra in [("primary", P, True), ("secondary", Sx, False)]:
        if not len(df):
            say(f"_{name}: n = 0_\n"); continue
        say(f"**{name}**\n")
        say("| minute (ET) | atmiv²τ / V_px | vol50²τ / V_px | mid_iv²τ / V_px (diagnostic) |" + (" closed-form V / V_px (diagnostic) | V_px(S = stock_price) / V_px (diagnostic) |" if extra else ""))
        say("|---|---|---|---|" + ("---|---|" if extra else ""))
        for m in MINUTES_PT:
            et = f"09:{m[3:]}"
            row = f"| {et} | {fq(q(df[f'r_atm_{m}']))} | {fq(q(df[f'r_v50_{m}']))} | {fq(q(df[f'r_mid_{m}']))} |"
            if extra:
                row += f" {fq(q(df[f'r_cf_{m}']))} | {fq(q(df[f'vpx_alt_{m}'] / df[f'vpx_{m}']))} |"
            say(row)
        say("")

    # ------------------------------------------------ item 5 close side
    say("## Item 5 — P3(e): close side, atmiv / vol50 at 15:59 ET by bucket\n")
    U = fin.copy()
    for c_ in ["t", "t1", "S1_ex", "S2_ex", "S7_ex"]:
        U[c_] = pd.to_datetime(U[c_]).dt.date
    say("| bucket | 15:59 on t (v_a) | 15:59 on t+1 (v_c) | flag |")
    say("|---|---|---|---|")
    flags = {}
    for bk in ("S1", "S2", "S7"):
        ra, rc = [], []
        for r in U[U[f"{bk}_ok"]].itertuples():
            e = getattr(r, f"{bk}_ex")
            x = mkey.get((r.t, e, "12:59")); y = mkey.get((r.t1, e, "12:59"))
            if x is not None and x.vol50:
                ra.append(x.atmiv / x.vol50)
            if y is not None and y.vol50:
                rc.append(y.atmiv / y.vol50)
        ma = np.median(ra); mc = np.median(rc)
        flag = "unadjudicated" if not (0.95 <= ma <= 1.05) else "ok"
        flags[bk] = flag
        say(f"| {bk} | {fq(q(ra))} | {fq(q(rc))} | {flag} (v_a) |")
    say("\nThe flag is read on v_a (the 15:59 row on t). v_c is shown for completeness: for S1 it is the 0DTE row one minute before expiry, where iv²τ ≈ 0 under either field.\n")

    # ------------------------------------------------ item 6 atmiv minute-to-minute
    say("## Item 6 — P3(f): atmiv minute-to-minute 09:31 → 09:40\n")
    steps = [f"06:{m:02d}" for m in range(31, 41)]
    say("| contract at the open | n days | share changed at step 09:31→32, 32→33, …, 39→40 | first change minute (ET) median [IQR] |")
    say("|---|---|---|---|")
    for label, sel in [("0DTE", lambda r: (r.t1, r.t1)), ("S1 open-side (= 0DTE of t+1)", lambda r: (r.t1, r.S1_ex)), ("S2 open-side", lambda r: (r.t1, r.S2_ex))]:
        ch = np.zeros(len(steps) - 1); n = 0; firsts = []
        base = U[U.S1_ok] if "S1" in label or label == "0DTE" else U[U.S2_ok]
        for r in base.itertuples():
            dd, ee = sel(r)
            vals = [mkey.get((dd, ee, s)) for s in steps]
            if any(v is None for v in vals):
                continue
            a_ = [v.atmiv for v in vals]; n += 1
            diff = [a_[i + 1] != a_[i] for i in range(len(a_) - 1)]
            ch += np.array(diff, dtype=float)
            fc = next((i + 1 for i, x in enumerate(diff) if x), None)
            if fc is not None:
                firsts.append(31 + fc)
        shares = ", ".join(f"{100*c/n:.0f}%" for c in ch) if n else "—"
        fm = f"09:{int(np.median(firsts)):02d} [09:{int(np.percentile(firsts,25)):02d}–09:{int(np.percentile(firsts,75)):02d}] ({len(firsts)} of {n} change by 09:40)" if firsts else "—"
        say(f"| {label} | {n} | {shares} | {fm} |")
    say("")

    # ------------------------------------------------ item 7 five random days
    say("## Item 7 — vol50 / atmiv per minute on five random CR-BO-universe nights (numpy default_rng(17))\n")
    rng = np.random.default_rng(17)
    pick = U[U.S1_ok & U.S2_ok & U.S7_ok].iloc[sorted(rng.choice(int((U.S1_ok & U.S2_ok & U.S7_ok).sum()), 5, replace=False))]
    for r in pick.itertuples():
        say(f"**t = {r.t}, t+1 = {r.t1}** — S1 {r.S1_ex}, S2 {r.S2_ex}, S7 {r.S7_ex}\n")
        say("| minute (ET) | " + " | ".join(f"{b} atmiv / vol50" for b in ("S1", "S2", "S7")) + " |")
        say("|---|---|---|---|")
        for day, mins in [(r.t, [f"12:{m}" for m in range(50, 60)]), (r.t1, [f"06:{m:02d}" for m in range(30, 51)])]:
            for mm in mins:
                cells = []
                for e in (r.S1_ex, r.S2_ex, r.S7_ex):
                    x = mkey.get((day, e, mm))
                    cells.append("—" if x is None else f"{x.atmiv:.4f} / {x.vol50:.4f}")
                hh = int(mm[:2]) + 3
                say(f"| {day} {hh:02d}:{mm[3:]} | " + " | ".join(cells) + " |")
        say("")

    # ------------------------------------------------ item 8 decision
    say("## Item 8 — P3 decision line\n")
    use_primary = len(P) >= PRIMARY_MIN_N
    D = P if use_primary else Sx
    ra = np.nanmedian(D["r_atm_06:33"]) if len(D) else np.nan
    rv = np.nanmedian(D["r_v50_06:33"]) if len(D) else np.nan
    qa = abs(ra - 1) <= TOL; qv = abs(rv - 1) <= TOL
    ext = "atmiv" if qa else ("vol50" if qv else None)
    if qa and qv:
        ext = "atmiv"
    dec = dict(subset="primary" if use_primary else "secondary", n_primary=len(P), n_secondary=len(Sx), r_atm_0933=ra, r_v50_0933=rv,
               atm_qualifies=bool(qa), v50_qualifies=bool(qv), extension_field=ext, close_flags=flags)
    say(f"- Deciding subset: **{dec['subset']}** (primary n {len(P)} {'≥' if use_primary else '<'} {PRIMARY_MIN_N}).")
    say(f"- 09:33 ET medians on the deciding subset: atmiv²τ / V_px **{ra:.3f}** ({'within' if qa else 'outside'} 10 %), vol50²τ / V_px **{rv:.3f}** ({'within' if qv else 'outside'} 10 %).")
    say(f"- **Extension field: {ext or 'none — P1 runs on the decision-subset days only'}.**")
    dset = set(D.d) if len(D) else set()
    dvalid = set(D[D["vpx_06:31"] > 0].d) if len(D) else set()
    cov = {}
    for bk in ("S1", "S2", "S7"):
        base = U[U[f"{bk}_ok"]]
        nv = int(sum(1 for t1 in base.t1 if t1 in dvalid)) if bk == "S1" else 0
        cov[bk] = dict(vpx=nv, extended=(len(base) - nv) if ext else 0, dropped=0 if ext else len(base) - nv)
    dec["coverage"] = cov
    say("- P1 open-side coverage by bucket (nights):\n")
    say("| bucket | v_b = V_px (09:31) | v_b = extension field²τ | dropped |")
    say("|---|---|---|---|")
    for bk, c_ in cov.items():
        say(f"| {bk} | {c_['vpx']} | {c_['extended']} | {c_['dropped']} |")
    say("\nOnly 0DTE contracts are captured, so V_px can serve only S1's open-side contract; S2 and S7 rest on the extension field (P3(d)).\n")

    # ------------------------------------------------ item 9 cost proxy
    say("## Item 9 — P4 cost: lower-bound proxy (CR-AS 0DTE bid–ask widths, 09:31–09:33 ET)\n")
    w = legs[legs.hm.isin(["06:31", "06:32", "06:33"])]
    w = w[[leg_quote_is_valid(b_, a_) for b_, a_ in zip(w.bid_price, w.ask_price)]]
    ww = w.ask_price - w.bid_price
    say(f"- Per leg (all captured CR-AS 0DTE contracts, CR-AN-valid): median {ww.median():.3f} pts, IQR {ww.quantile(.25):.3f}–{ww.quantile(.75):.3f} (n {len(ww)}).")
    if len(P):
        x = pd.concat([P['w_06:31'], P['w_06:33']]); say(f"- Two-leg straddle (primary): median {x.median():.3f}, IQR {x.quantile(.25):.3f}–{x.quantile(.75):.3f}.")
    if len(Sx):
        x = pd.concat([Sx['w_06:31'], Sx['w_06:33']]); say(f"- Two-leg strangle (secondary): median {x.median():.3f}, IQR {x.quantile(.25):.3f}–{x.quantile(.75):.3f}.")
    say("- **Label: lower-bound proxy; the real cost test is Gate B after quote capture.** No 1-DTE train quotes exist in `orats_options_minute` (CR-BO Step 0 Q3: no 1–3 DTE pairs cached). Nothing further is estimated.\n")

    # ------------------------------------------------ item 10 / 11
    say("## Item 10 — P6 priors vs CR-BO\n")
    s = json.load(open(REPO_ROOT / "scripts" / "output" / "cr_bo" / "summary.json"))
    say("| bucket | CR-BO P_on mean (atmiv) | P6 prior |")
    say("|---|---|---|")
    for bk in ("S1", "S2", "S7"):
        say(f"| {bk} | {s['headline'][bk]['P_on_mean'][0]*1e4:+.3f} | {PRIORS[bk]:+.2f} |")
    say("\n## Item 11 — remaining defaults (C6)\n")
    say("- (i) P2 read: RV_on 16:00 ET t → 09:50 ET t+1 and RV_rth 09:50 → 16:00 ET t+1; the P1 read keeps L4's windows.")
    say("- (ii) If any L8 cell has n < 40 the verdict is printed with a 'cell n < 40' note naming it; the verdict is computed exactly as L8 says.\n")

    (OUT_DIR / "step0.md").write_text("\n".join(L) + "\n")
    P.to_parquet(OUT_DIR / "vpx_primary.parquet", index=False)
    Sx.to_parquet(OUT_DIR / "vpx_secondary.parquet", index=False)
    (OUT_DIR / "p3_decision.json").write_text(json.dumps(dec, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'step0.md'}")


if __name__ == "__main__":
    main()
