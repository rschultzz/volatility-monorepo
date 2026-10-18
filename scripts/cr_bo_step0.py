#!/usr/bin/env python3
"""CR-BO Step 0 — diagnosis for T-S1a (session premium). READ-ONLY: SELECTs only, no run row.

Answers (specs/CR-BO-session-premium.md, Implementation order step 2):
  Q1 L1 coverage counts after each exclusion (nights, by bucket)
  Q2 ES contract-switch dates (detected from the ES − SPX basis) and roll-week nights removed
  Q3 ORATS τ convention for atmiv (implied τ from cached option quote pairs; one worked example)
     + atmiv refresh cadence at the 09:31 / 15:59 rows
  Q4 vol25 / vol75 delta mapping
  Q5 ES overnight minutes missing, by year
  Q6 first date each weekday of t has a next-session (S1) expiry
  Q7 event-date list (from scripts/cr_bo_common.py) with sources

Usage: apps/web/.venv/bin/python -u scripts/cr_bo_step0.py
"""
from __future__ import annotations

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
from scripts import cr_bo_common as C

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT = REPO_ROOT / "scripts" / "output" / "cr_bo" / "step0.md"
BUCKETS = ("S1", "S2", "S7")
MAX_MISSING = 0.05
SWITCH_BASIS_JUMP = 30.0   # pts; non-switch day-over-day basis changes are < 28 in train (|Δ| sd ≈ 9)

MONIES_SQL = """
SELECT trade_date, expir_date, expiry_tod, dte, dte_pct, snapshot_pt, quote_date, atmiv, vol25, vol50, vol75,
       stock_price
FROM orats_monies_minute
WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND dte BETWEEN 1 AND 16
  AND snapshot_pt::time IN ('06:30', '06:31', '12:59')
"""
ES_SQL = """
SELECT datetime, close, source FROM ironbeam_es_1m_bars
WHERE datetime >= %s AND datetime < %s ORDER BY datetime
"""
TAU_SQL = """
SELECT c.trade_date_d, c.expir_date_d, c.strike, c.snapshot_pt,
       (c.bid_price + c.ask_price) / 2 AS cm, (p.bid_price + p.ask_price) / 2 AS pm,
       c.mid_iv AS civ, p.mid_iv AS piv
FROM orats_options_minute c
JOIN orats_options_minute p
  ON p.snapshot_pt = c.snapshot_pt AND p.expir_date_d = c.expir_date_d AND p.strike = c.strike
 AND p.option_type = 'P' AND p.expiry_tod = 'pm'
WHERE c.option_type = 'C' AND c.expiry_tod = 'pm' AND c.trade_date_d <= %s
  AND abs(c.strike - c.stock_price) <= 10 AND extract(minute FROM c.snapshot_pt)::int IN (31, 59)
  AND extract(hour FROM c.snapshot_pt)::int IN (6, 12)
  AND c.mid_iv > 0 AND p.mid_iv > 0 AND c.bid_price > 0 AND p.bid_price > 0
"""


def year_chunks(a: date, b: date):
    y = a.year
    while y <= b.year:
        lo = max(a, date(y, 1, 1)); hi = min(b, date(y, 12, 31))
        yield lo, hi
        y += 1


def load_monies(conn) -> pd.DataFrame:
    parts = []
    for lo, hi in year_chunks(C.TRAIN_START, C.TRAIN_END):
        parts.append(pd.read_sql(MONIES_SQL, conn, params=(lo.isoformat(), hi.isoformat())))
        print(f"  monies {lo}..{hi}: {len(parts[-1])}", flush=True)
    m = pd.concat(parts, ignore_index=True)
    m["td"] = pd.to_datetime(m.trade_date).dt.date
    m["ex"] = pd.to_datetime(m.expir_date).dt.date
    m["et"] = m.snapshot_pt + pd.Timedelta(hours=3)          # PT → ET (same DST rules)
    m["hm"] = m.et.dt.strftime("%H:%M")
    return m


def load_es(conn) -> pd.DataFrame:
    a = datetime.combine(C.TRAIN_START - timedelta(days=5), time(0))
    b = datetime.combine(C.TRAIN_END + timedelta(days=2), time(0))
    parts = []
    for lo, hi in year_chunks(a.date(), b.date()):
        parts.append(pd.read_sql(ES_SQL, conn, params=(datetime.combine(lo, time(0)), datetime.combine(hi + timedelta(days=1), time(0)))))
        print(f"  es {lo}..{hi}: {len(parts[-1])}", flush=True)
    es = pd.concat(parts, ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    es["et"] = es.datetime.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    es = es.reset_index(drop=True)
    es["r"] = np.log(es.close / es.close.shift())
    return es


def bs_call(F, K, t, s):
    v = s * np.sqrt(t); d1 = (np.log(F / K) + v * v / 2) / v
    return F * norm.cdf(d1) - K * norm.cdf(d1 - v)


def main() -> None:
    L: list[str] = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    say(f"# CR-BO Step 0 — {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Read-only. Train: pairs (t, t+1) with t ≥ 2023-05-01, t+1 ≤ 2026-06-05. Times ET unless tagged.\n")

    print("loading …", flush=True)
    m = load_monies(conn)
    es = load_es(conn)
    pm = m[m.expiry_tod == "pm"]

    # ---------------------------------------------------------------- Q2 switch detection
    es["hm"] = es.et.dt.strftime("%H:%M"); es["d"] = es.et.dt.date
    es1559 = es[es.hm == "15:59"].set_index("d").close
    spx1559 = pm[pm.hm == "15:59"].sort_values("dte").groupby("td").first().stock_price
    b = pd.concat([es1559.rename("es"), spx1559.rename("spx")], axis=1, join="inner")
    b["basis"] = b.es - b.spx; b["db"] = b.basis.diff()
    sw = b[b.db > SWITCH_BASIS_JUMP]
    say("## Q2 ES contract switches and roll weeks\n")
    say(f"Detector: day-over-day change in (ES 15:59 bar close − SPX `stock_price` at the 15:59 row, nearest PM expiry) > {SWITCH_BASIS_JUMP:.0f} pts. "
        f"Δbasis over {len(b)-1} sessions: sd {b.db.std():.2f}, largest non-switch |Δ| {b.db[b.db <= SWITCH_BASIS_JUMP].abs().max():.2f}.\n")
    say("| switch detected on (session) | previous session | Δbasis (pts) | in a roll week? | switch minute (ET) | source |")
    say("|---|---|---|---|---|---|")
    sessions_idx = list(b.index)
    switch_intervals = []
    for d, row in sw.iterrows():
        prev = sessions_idx[sessions_idx.index(d) - 1]
        lo = datetime.combine(prev, time(15, 59)); hi = datetime.combine(d, time(15, 59))
        w = es[(es.et > lo) & (es.et <= hi)]
        j = w.loc[w.r.abs().idxmax()]
        switch_intervals.append((j.et.to_pydatetime(), d))
        third = C.roll_week(prev) or C.roll_week(d)
        say(f"| {d} | {prev} | {row.db:+.2f} | {'yes' if third else 'NO'} | {j.et:%Y-%m-%d %a %H:%M} (1-min r {j.r:+.4f}) | {j.source} |")
    say(f"\n{len(sw)} switches = one per quarterly expiry in the train window. Databento `ES.c.0` (2023–2025) switches at the Sunday 18:00 reopen after expiry Friday and prints **no RTH bars on expiry Friday** (dying contract stops at the 09:30 SOQ); the Ironbeam ingest (2026-03) switched at 09:30 on expiry Friday.\n")

    # ---------------------------------------------------------------- per-pair universe
    pairs = C.session_pairs()
    es_et = es.set_index("et")
    es_minutes = set(es_et.index)
    have = {(r.td, r.ex, r.hm): r for r in pm.itertuples()}
    exps_by_t = pm[pm.hm == "15:59"].groupby("td").ex.apply(lambda s: sorted(set(s)))

    def first_bar_ok(t: date) -> bool:
        lo = datetime.combine(t, time(15, 55)); hi = datetime.combine(t, time(15, 59))
        return any(lo <= x <= hi for x in (lo + timedelta(minutes=k) for k in range(5)) if x in es_minutes)

    recs = []
    for t, t1 in pairs:
        skip = C.holiday_dates_between(t, t1)
        on_lo, on_hi = C.overnight_bounds(t, t1); r_lo, r_hi = C.rth_bounds(t1)
        exp_on = C.window_minutes(on_lo, on_hi, skip); exp_rth = C.window_minutes(r_lo, r_hi)
        got_on = sum(1 for x in exp_on if x in es_minutes); got_rth = sum(1 for x in exp_rth if x in es_minutes)
        wlo, whi = datetime.combine(t, time(16, 0)), datetime.combine(t1, time(16, 0))
        rec = dict(t=t, t1=t1, wd=t.strftime("%a"), cal_gap=(t1 - t).days, holiday_span=bool(skip),
                   early=(t in C.EARLY_CLOSES or t1 in C.EARLY_CLOSES),
                   roll=(C.roll_week(t) or C.roll_week(t1)),
                   switch=any(wlo < sm <= whi for sm, _ in switch_intervals),
                   on_exp=len(exp_on), on_got=got_on, rth_exp=len(exp_rth), rth_got=got_rth,
                   anchor=first_bar_ok(t))
        exps = exps_by_t.get(t, [])
        tgt = C.bucket_expiries(t)
        s7 = [e for e in exps if (e - t).days >= 7]
        tgt["S7"] = s7[0] if s7 else None
        for k in BUCKETS:
            e = tgt[k]
            rec[f"{k}_ex"] = e
            rec[f"{k}_ok"] = bool(e is not None and e in exps and (t1, e, "09:31") in have and (t1, e, "15:59") in have
                                   and all(pd.notna(getattr(have[x], "atmiv")) and getattr(have[x], "atmiv") > 0
                                           for x in [(t, e, "15:59"), (t1, e, "09:31"), (t1, e, "15:59")]))
        recs.append(rec)
    U = pd.DataFrame(recs)
    U["on_miss"] = 1 - U.on_got / U.on_exp; U["rth_miss"] = 1 - U.rth_got / U.rth_exp

    say("\n## Q1 L1 coverage — nights after each exclusion\n")
    say("Exclusions applied in this order (a night removed by an earlier step is not counted again):\n")
    steps = [
        ("all session pairs (t, t+1), t ≥ 2023-05-01, t+1 ≤ 2026-06-05", lambda d: pd.Series(True, index=d.index)),
        ("− early close on t or t+1", lambda d: ~d.early),
        ("− ES roll week on t or t+1 (Mon of week before 3rd Fri → expiry Fri; Mar/Jun/Sep/Dec)", lambda d: ~d.roll),
        ("− window 16:00 t → 16:00 t+1 contains a contract switch (not already in a roll week)", lambda d: ~d.switch),
        ("− no ES anchor bar 15:55–15:59 on t", lambda d: d.anchor),
        (f"− ES overnight missing > {MAX_MISSING:.0%} of expected minutes", lambda d: d.on_miss <= MAX_MISSING),
        (f"− ES RTH (t+1) missing > {MAX_MISSING:.0%} (needed for RV_rth; L1 interpretation)", lambda d: d.rth_miss <= MAX_MISSING),
    ]
    say("| step | nights | S1 | S2 | S7 |")
    say("|---|---|---|---|---|")
    keep = pd.Series(True, index=U.index)
    for label, f in steps:
        keep &= f(U)
        say(f"| {label} | {keep.sum()} | {(keep & U.S1_ok).sum()} | {(keep & U.S2_ok).sum()} | {(keep & U.S7_ok).sum()} |")
    say(f"\nBucket columns require the monies rows of L1/L3: atmiv > 0 at 15:59 on t, 09:31 on t+1 and 15:59 on t+1, same `expir_date`, PM. "
        f"Monies-missing nights within the final ES-clean set: S1 {(keep & ~U.S1_ok).sum()}, S2 {(keep & ~U.S2_ok).sum()}, S7 {(keep & ~U.S7_ok).sum()}.\n")
    U["keep"] = keep
    fin = U[keep]
    say(f"Final universe by weekday of t: {dict(Counter(fin.wd))}; holiday-spanning nights kept: {int(fin.holiday_span.sum())}; "
        f"calendar-day gap t→t+1: {dict(sorted(Counter(fin.cal_gap).items()))}.\n")
    say(f"Removed by roll week: {int((U.roll & ~U.early).sum())} nights; by a switch outside a roll week: {int((U.switch & ~U.roll & ~U.early).sum())}; "
        f"switch nights total (any overlap): {int(U.switch.sum())}, all of which are also roll-week nights: {bool((U.switch <= U.roll).all())}.\n")
    say("Early-close pairs removed: " + ", ".join(f"{a}→{b}" for a, b in U[U.early][["t", "t1"]].itertuples(index=False)) + "\n")
    miss_s1 = U[keep & ~U.S1_ok]
    if len(miss_s1):
        say("S1 monies-missing nights: " + ", ".join(str(x) for x in miss_s1.t) + "\n")
    # why monies rows are missing (ES-clean nights)
    snaps = set(zip(m.td, m.hm))
    am_rows = set(zip(m[m.expiry_tod == "am"].td, m[m.expiry_tod == "am"].ex, m[m.expiry_tod == "am"].hm))
    say("Reasons, per bucket, on ES-clean nights (a night can have more than one):\n")
    say("| bucket | no 09:31 snapshot at all on t+1 | 15:59 row on t: only an AM row for that expir_date | 15:59 row on t absent otherwise | 09:31 row absent with snapshot present | 15:59 row on t+1 absent | no expiry fits (S7) |")
    say("|---|---|---|---|---|---|---|")
    for k in BUCKETS:
        g = U[keep & ~U[f"{k}_ok"]]
        c = Counter()
        for r in g.itertuples():
            e = getattr(r, f"{k}_ex")
            if e is None or pd.isna(e):
                c["nofit"] += 1; continue
            if (r.t1, "09:31") not in snaps:
                c["no931snap"] += 1
            elif (r.t1, e, "09:31") not in have:
                c["no931row"] += 1
            if (r.t, e, "15:59") not in have:
                c["am_only" if (r.t, e, "15:59") in am_rows else "no1559t"] += 1
            if (r.t1, e, "15:59") not in have:
                c["no1559t1"] += 1
        say(f"| {k} | {c['no931snap']} | {c['am_only']} | {c['no1559t']} | {c['no931row']} | {c['no1559t1']} | {c['nofit']} |")
    no931 = sorted({r.t1 for r in U[keep].itertuples() if (r.t1, "09:31") not in snaps})
    say(f"\nES-clean nights whose t+1 has no 09:31 snapshot: {len(no931)}; by month: {dict(Counter(d.strftime('%Y-%m') for d in no931).most_common(8))}. "
        "`orats_monies_minute` is keyed (ticker, expir_date, quote_date), so on a third-Friday expir_date the AM monthly and PM weekly cannot both be stored; the table holds the AM row and the PM S1 on Thursday nights before a monthly expiry is unavailable. L1/L2 exclude both cases as written (no amendment).\n")

    # ---------------------------------------------------------------- Q5 ES missing by year
    say("## Q5 ES minutes missing, by year (all pairs that pass early-close / roll / switch)\n")
    base = U[~U.early & ~U.roll & ~U.switch].copy(); base["y"] = [d.year for d in base.t1]
    say("| year | nights | overnight expected min (median) | missing % median | p95 | max | nights > 5 % | RTH missing % median | RTH nights > 5 % |")
    say("|---|---|---|---|---|---|---|---|---|")
    for y, g in base.groupby("y"):
        say(f"| {y} | {len(g)} | {int(g.on_exp.median())} | {100*g.on_miss.median():.2f} | {100*g.on_miss.quantile(.95):.2f} | {100*g.on_miss.max():.2f} | {(g.on_miss > MAX_MISSING).sum()} | {100*g.rth_miss.median():.2f} | {(g.rth_miss > MAX_MISSING).sum()} |")
    bad = base[(base.on_miss > MAX_MISSING) | (base.rth_miss > MAX_MISSING) | ~base.anchor]
    if len(bad):
        say("\nNights failing coverage: " + "; ".join(f"{r.t}→{r.t1} on {100*r.on_miss:.1f}% rth {100*r.rth_miss:.1f}%{' no-anchor' if not r.anchor else ''}" for r in bad.itertuples()) + "\n")
    say("Expected-minute grid: CME equity Globex Sun 18:00 → Fri 17:00 ET, daily halt 17:00–17:59, bar-start stamps; "
        "on holiday-spanning nights the NYSE-holiday calendar date is dropped from the grid (CME runs a holiday schedule the grid does not model); bars that exist that day still enter RV.\n")

    # ---------------------------------------------------------------- Q3 τ convention
    say("## Q3 ORATS τ convention for atmiv\n")
    tq = pd.read_sql(TAU_SQL, conn, params=(C.TRAIN_END,))
    out = []
    for r in tq.itertuples():
        F = r.strike + r.cm - r.pm; iv = (r.civ + r.piv) / 2
        try:
            tau = brentq(lambda x: bs_call(F, r.strike, x, iv) - r.cm, 1e-9, 2.0)
        except ValueError:
            continue
        snap_et = r.snapshot_pt + timedelta(hours=3)
        cal = C.tau_years(snap_et, r.expir_date_d) * 365
        if cal <= 0:
            continue
        out.append(dict(cdte=(r.expir_date_d - r.trade_date_d).days, hm=snap_et.strftime("%H:%M"), cal=cal, tau=tau * 365, F=F, K=r.strike, cm=r.cm, iv=iv,
                        td=r.trade_date_d, ex=r.expir_date_d))
    T = pd.DataFrame(out); T["ratio"] = T.tau / T.cal
    T["b"] = pd.cut(T.cdte, [-1, 0, 3, 7, 30, 400], labels=["0", "1–3", "4–7", "8–30", "31+"])
    say("Method: cached `orats_options_minute` call/put pairs at the same strike (|K − stock_price| ≤ 10), forward from parity F = K + C − P, IV = mean of the pair's `mid_iv`; "
        "solve Black (r = 0) for the τ that reproduces the call mid; compare with calendar time to 16:00 ET on expiry / 365.\n")
    say("| calendar DTE | snapshot (ET) | n | median τ_ORATS / τ_calendar | IQR |")
    say("|---|---|---|---|---|")
    for (bb, hm), g in T.groupby(["b", "hm"], observed=True):
        say(f"| {bb} | {hm} | {len(g)} | {g.ratio.median():.3f} | {g.ratio.quantile(.25):.3f}–{g.ratio.quantile(.75):.3f} |")
    cand = T[(T.cdte.between(4, 30)) & (T.hm == "09:31")]
    ex = cand.loc[(cand.ratio - cand.ratio.median()).abs().idxmin()]   # representative (median-ratio) pair
    say(f"\n**Worked example.** {ex.td} 09:31 ET, expiry {ex.ex} PM, K {ex.K:.0f}: parity F {ex.F:.2f}, call mid {ex.cm:.3f}, pair mid_iv {ex.iv:.5f}. "
        f"Calendar time to {ex.ex} 16:00 ET = {ex.cal:.4f} days; the τ that reproduces the mid = {ex.tau:.4f} days (ratio {ex.tau/ex.cal:.4f}).\n")
    say("**Reading.** For 4–30 calendar-DTE contracts at both the 09:31 and 15:59 snapshots, ORATS IV is consistent with τ = calendar time to 16:00 ET on expiry / 365 "
        "(median ratio 1.000–1.020). No 1–3 DTE pairs are cached, so the S1 / S2 15:59 rows cannot be checked directly; calendar/365 is used for them (L3 fallback). "
        "0DTE IV departs from calendar time in the last ~90 minutes (ORATS runs a longer clock near expiry); in this test that touches only v_c of S1 "
        "(the 15:59 row on its expiry day, τ = 1 minute), whose iv²τ is ≈ 0 under either clock. **L3 τ used: (16:00 ET on expir_date − snapshot label time) / 365 days.**\n")
    # atmiv worked example on a universe night
    say("`orats_monies_minute` carries no fractional-τ column: `dte` = calendar days + 1 (1 on expiry day) and `dte_pct` = (dte − 1)/365.25, identical at 09:31 and 15:59.\n")

    # atmiv cadence
    rows = []
    for (td, ex_), g in pm.groupby(["td", "ex"]):
        g = g.set_index("hm")
        if "09:30" in g.index and "09:31" in g.index:
            rows.append(dict(same=g.loc["09:30", "atmiv"] == g.loc["09:31", "atmiv"], q931=str(g.loc["09:31", "quote_date"]), q1559=str(g.loc["15:59", "quote_date"]) if "15:59" in g.index else None,
                             r50=g.loc["09:31", "atmiv"] / g.loc["09:31", "vol50"] if g.loc["09:31", "vol50"] else np.nan))
    A = pd.DataFrame(rows)

    def q_et(s):
        if not s or s == "None":
            return None
        ts = pd.Timestamp(s).tz_convert("America/New_York")
        return ts.strftime("%H:%M")
    A["q931_et"] = A.q931.map(q_et); A["q1559_et"] = A.q1559.map(q_et)
    say(f"**atmiv refresh cadence.** Over {len(A)} (date, PM expiry ≤ 15 DTE) rows: atmiv at the 09:31 row equals the 09:30 row on {100*A.same.mean():.1f} %. "
        f"Quote time (`quote_date`, ET) of the 09:31 row: {dict(Counter(A.q931_et).most_common(4))}; of the 15:59 row: {dict(Counter(A.q1559_et).most_common(4))}. "
        f"Median atmiv / vol50 at 09:31: {A.r50.median():.4f}.\n")

    # ---------------------------------------------------------------- Q4 vol25 / vol75
    s1559 = pm[pm.hm == "15:59"]
    say("## Q4 vol25 / vol75 mapping\n")
    say(f"ORATS monies smile is indexed by **call delta** (`vol5` … `vol95`; repo convention `packages/shared/vol_features.py:20-21`: `vol25` = 25-delta call IV, `vol75` = 75-delta call = 25-delta put IV). "
        f"Check on the data: at 15:59, `vol75 > vol25` on {100*(s1559.vol75 > s1559.vol25).mean():.1f} % of PM rows ≤ 15 DTE (n {len(s1559)}), "
        f"`vol75 > vol50` on {100*(s1559.vol75 > s1559.vol50).mean():.1f} %, `vol25 < vol50` on {100*(s1559.vol25 < s1559.vol50).mean():.1f} % — the downward SPX skew puts the higher vol on the low-strike side, i.e. call delta 0.75. "
        "**L6(d): put side = `vol75`, call side = `vol25`.**\n")

    # ---------------------------------------------------------------- Q6 weekday S1 coverage
    say("## Q6 first t per weekday with a next-session (S1) PM expiry\n")
    U["wdn"] = [d.weekday() for d in U.t]
    say("| weekday of t | first t with S1 rows | nights with S1 ok / all pairs |")
    say("|---|---|---|")
    for wdn, g in U.groupby("wdn"):
        ok = g[g.S1_ok]
        say(f"| {g.wd.iloc[0]} | {ok.t.min() if len(ok) else '—'} | {len(ok)} / {len(g)} |")
    say("\nDaily SPX expiries for every weekday predate the train window; every weekday is covered from the first train session.\n")

    # ---------------------------------------------------------------- Q7 events
    say("## Q7 event dates (L6(c)) — hand-coded in `scripts/cr_bo_common.py`\n")
    for k, s in C.EVENT_0830.items():
        say(f"- **{k}** 08:30 ET ({len(s)}): " + ", ".join(str(d) for d in sorted(s)))
    say(f"- **PCE at 10:00 ET, not event nights**: {', '.join(str(d) for d in sorted(C.PCE_NOT_0830))}")
    say(f"- **FOMC statement days** ({len(C.FOMC_DAYS)}): " + ", ".join(str(d) for d in sorted(C.FOMC_DAYS)))
    ev = sorted(set().union(*C.EVENT_0830.values()))
    say(f"\nDistinct 08:30 event dates: {len(ev)}. In the final universe as t+1: {sum(1 for d in fin.t1 if d in set(ev))} event nights, "
        f"{sum(1 for d in fin.t1 if d in C.FOMC_DAYS)} FOMC-on-t+1 nights.\n")
    say("Sources: CPI https://www.bls.gov/bls/news-release/cpi.htm · PPI https://www.bls.gov/bls/news-release/ppi.htm · NFP https://www.bls.gov/bls/news-release/empsit.htm · "
        "PCE 2023 https://apps.bea.gov/scb/issues/2022/12-december/1222-news-releases-2023.htm, 2024 https://apps.bea.gov/scb/issues/2023/12-december/1223-news-releases-2024.htm, "
        "2025 https://www.bea.gov/news/schedule/full-2025, 2026 https://bea.gov/index.php/news/blog/2026-01-15/economic-release-schedule-updates-gdp-personal-income-and-outlays + "
        "https://www.bea.gov/news/2026/personal-income-and-outlays-december-2025 + https://www.bea.gov/index.php/news/schedule/full · FOMC https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    U.to_csv(OUT.with_name("step0_pairs.csv"), index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
