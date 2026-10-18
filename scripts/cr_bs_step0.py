#!/usr/bin/env python3
"""CR-BS Step 0 — diagnosis for T-S3a. READ-ONLY: SELECTs only, no run row, no API calls. Holdout never read.

Spec: specs/CR-BS-morning-surface.md (Step 0 list items 1–6).

  1  L1 counts after each exclusion, by era
  2  07:30 PT atmiv / vol50 on 0DTE and the next-session expiry, by era; field(10:30)/field(10:50) stability
  3  0DTE dte convention + one worked day
  4  vol25 / vol75 mapping
  5  morning-RV availability (ES bars 09:30 → 10:30 ET)
  6  walk-forward window sizes

Gate (item 2), fixed here before the run. "The E0 depression seen at 09:31 persists at 10:30" means ANY of:
  (a) E0 median atmiv/vol50 on 0DTE at the snapshot lies outside 0.95–1.05 (the CR-BP C3 band) while E1's is inside;
  (b) |E0 median / E1 median − 1| > 0.05 for atmiv/vol50 on 0DTE;
  (c) E0 median vol50(snapshot)/vol50(10:50 ET) on 0DTE lies outside 0.95–1.05 while E1's is inside
      (CR-BP's 09:33/09:50 form of the depression: E0 0.523, E1 1.035).
If the gate trips, the script prints STOP and the CR stops for a spec amendment.

Usage: apps/web/.venv/bin/python -u scripts/cr_bs_step0.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date, datetime, time
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

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from scripts import cr_bs_common as S
from scripts.cr_bo_common import EARLY_CLOSES, FOMC_DAYS, event_tags, roll_week
from scripts.cr_bo_step0 import load_es

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bs"
WF_START = 250


def q(x, p):
    return float(np.nanquantile(x, p)) if len(x) else float("nan")


def med_iqr(x) -> str:
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return "— (n 0)"
    return f"{np.median(x):.3f} (IQR {q(x, .25):.3f}–{q(x, .75):.3f}, n {len(x)})"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    say(f"# CR-BS Step 0 — {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Read-only. Train t ≤ 2026-06-05. Era E0 t < 2024-03-01, E1 t ≥ 2024-03-01. Times ET unless tagged PT.\n")

    print("loading monies …", flush=True)
    m = S.load_monies(conn)
    print(f"  monies rows {len(m)}", flush=True)
    print("loading ES …", flush=True)
    es = load_es(conn)
    snaps = S.snapshot_rows(m)
    m50 = {d: g for d, g in m[m.hm == "07:50"].groupby("td")}

    # ---------------------------------------------------------------- item 3 (dte convention) — first, it defines L1(i)
    say("## 3. 0DTE dte convention\n")
    pm = m[(m.expiry_tod == "pm") & m.hm.isin(["07:30", "07:31", "07:32", "07:33"])]
    same = pm[pm.ex == pm.td]
    say(f"- Rows with dte = 0 in the 07:30–07:33 PT window (all expiries, AM and PM): **{int((m[m.hm != '07:50'].dte == 0).sum())}**.")
    say(f"- Same-day PM rows (expir_date = trade_date): dte values {dict(Counter(same.dte.astype(int)))}.")
    nxt = pm[pm.ex == pm.td.map(S.next_session)]
    say(f"- Next-session PM rows: dte values {dict(Counter(nxt.dte.astype(int)))} (calendar gap + 1 on Fridays / before holidays).")
    say(f"- `dte_pct` on same-day PM rows at 07:30 PT: {med_iqr(same[same.hm == '07:30'].dte_pct.to_numpy())} (ORATS' fraction field, diagnostic only; L3 fixes the clock).")
    wd = date(2025, 3, 4)
    say(f"\nWorked day {wd} (Tue), first snapshot ≥ 07:30 PT:\n")
    say("| snapshot_pt (PT) | quote_date (UTC) | expir_date | tod | dte | dte_pct | atmiv | vol25 | vol50 | vol75 | stock_price |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in snaps[wd].sort_values("dte").itertuples():
        say(f"| {r.snapshot_pt:%H:%M} | {r.quote_date} | {r.ex} | {r.expiry_tod} | {r.dte} | {r.dte_pct:.5f} | {r.atmiv:.4f} | {r.vol25:.4f} | {r.vol50:.4f} | {r.vol75:.4f} | {r.stock_price:.2f} |")
    say("\nReading: ORATS stores the same-day expiry with **dte = 1** (dte counts the expiry day). L1(i)'s \"dte = 0 (ORATS dte "
        "convention — confirm it is the same-day expiry)\" is implemented as `expir_date = trade_date`, PM-settled, which is "
        "CR-BP's \"smallest dte > 0\" contract. The next-session expiry is `expir_date = next_trading_day(t)`, PM-settled.\n")

    # ---------------------------------------------------------------- item 1 (L1 counts)
    say("## 1. L1 universe — counts after each exclusion, by era\n")
    days = S.train_days()
    steps = ["train sessions", "− early closes", "− roll weeks", "snapshot 07:30–07:33 PT", "0DTE PM row with smile",
             "next-session PM row with atmiv", f"ES RTH bars ≥ {S.MIN_RTH_BARS}", "morning bars ≥ 1 and remaining bars ≥ 1"]
    cnt = {s: Counter() for s in steps}
    lost_tf, uni = [], []
    for t in days:
        e = S.era(t)
        cnt[steps[0]][e] += 1
        if t in EARLY_CLOSES:
            continue
        cnt[steps[1]][e] += 1
        if roll_week(t):
            continue
        cnt[steps[2]][e] += 1
        rows = snaps.get(t)
        if rows is None:
            continue
        cnt[steps[3]][e] += 1
        r0 = S.pick(rows, t)
        if not S.smile_ok(r0):
            if t.weekday() == 4 and 15 <= t.day <= 21:
                lost_tf.append(t)
            continue
        cnt[steps[4]][e] += 1
        r1 = S.pick(rows, S.next_session(t))
        if r1 is None or not (pd.notna(r1.atmiv) and r1.atmiv > 0):
            continue
        cnt[steps[5]][e] += 1
        w = S.rth_frame(es, t)
        if len(w) < S.MIN_RTH_BARS:
            continue
        cnt[steps[6]][e] += 1
        mo = S.rv(w, datetime.combine(t, time(9, 30)), datetime.combine(t, time(10, 30)))
        re = S.rv(w, datetime.combine(t, time(10, 30)), datetime.combine(t, time(16, 0)))
        if mo[3] < 1 or re[3] < 1:
            continue
        cnt[steps[7]][e] += 1
        uni.append(dict(t=t, era=e, snap=rows.snapshot_pt.iloc[0], r0=r0, r1=r1, n_morn=mo[3], n_rem=re[3], n_rth=len(w),
                        fomc=t in FOMC_DAYS, ev=",".join(event_tags(t))))
    say("| step | E0 | E1 | total |")
    say("|---|---|---|---|")
    for s in steps:
        say(f"| {s} | {cnt[s]['E0']} | {cnt[s]['E1']} | {cnt[s]['E0'] + cnt[s]['E1']} |")
    U = pd.DataFrame(uni)
    say(f"\n- Third Fridays lost at the 0DTE step (the table keeps only the AM monthly row): {len(lost_tf)} — {[str(d) for d in lost_tf]}")
    say(f"- Snapshot minute used: {dict(Counter(U.snap.dt.strftime('%H:%M')))} (PT).")
    say(f"- Flags in the universe: FOMC days {int(U.fomc.sum())} (E0 {int(U[U.era == 'E0'].fomc.sum())}, E1 {int(U[U.era == 'E1'].fomc.sum())}); "
        f"08:30 release days {int((U.ev != '').sum())} (E0 {int(((U.ev != '') & (U.era == 'E0')).sum())}, E1 {int(((U.ev != '') & (U.era == 'E1')).sum())}). Both stay in.")
    miss = [t for t in days if t not in snaps and t not in EARLY_CLOSES and not roll_week(t)]
    by_m = Counter(f"{d:%Y-%m}" for d in miss)
    say(f"- Non-roll, non-early-close days with no 07:30–07:33 PT snapshot: {len(miss)}; by month {dict(sorted(by_m.items()))}.")

    # ---------------------------------------------------------------- item 2 (era ratio) + gate
    say("\n## 2. `atmiv` / `vol50` at 10:30 ET by era (no price-based adjudication exists at 10:30)\n")
    say("No captured option quotes exist at 10:30 ET in train (`orats_options_minute` holds 06:30–06:45 PT condor legs and ≤ 2026-06-05 verticals only), "
        "so neither field can be adjudicated against a traded price here. The comparison is field vs field, plus each field against itself 20 minutes later.\n")
    rows_out = []
    for u in uni:
        r0, r1 = u["r0"], u["r1"]
        g50 = m50.get(u["t"])
        a0 = S.pick(g50, u["t"]) if g50 is not None else None
        a1 = S.pick(g50, S.next_session(u["t"])) if g50 is not None else None
        rows_out.append(dict(
            t=u["t"], era=u["era"],
            r0=r0.atmiv / r0.vol50, r1=(r1.atmiv / r1.vol50) if (pd.notna(r1.vol50) and r1.vol50 > 0) else np.nan,
            v50_0=(r0.vol50 / a0.vol50) if (a0 is not None and a0.vol50 > 0) else np.nan,
            atm_0=(r0.atmiv / a0.atmiv) if (a0 is not None and a0.atmiv > 0) else np.nan,
            v50_1=(r1.vol50 / a1.vol50) if (a1 is not None and pd.notna(r1.vol50) and a1.vol50 > 0) else np.nan,
            atm_1=(r1.atmiv / a1.atmiv) if (a1 is not None and a1.atmiv > 0) else np.nan,
            k=(r0.vol75 - r0.vol25) / r0.vol50,
        ))
    R = pd.DataFrame(rows_out)
    say("| quantity | E0 | E1 |")
    say("|---|---|---|")
    labels = [("r0", "atmiv / vol50, 0DTE"), ("r1", "atmiv / vol50, next-session expiry"),
              ("v50_0", "vol50(10:30) / vol50(10:50), 0DTE"), ("atm_0", "atmiv(10:30) / atmiv(10:50), 0DTE"),
              ("v50_1", "vol50(10:30) / vol50(10:50), next-session"), ("atm_1", "atmiv(10:30) / atmiv(10:50), next-session")]
    for c, lab in labels:
        say(f"| {lab} | {med_iqr(R[R.era == 'E0'][c].to_numpy())} | {med_iqr(R[R.era == 'E1'][c].to_numpy())} |")
    R["ym"] = pd.to_datetime(R.t).dt.strftime("%Y-%m")
    mon = R[(R.ym >= "2023-11") & (R.ym <= "2024-06")].groupby("ym").r0.median()
    say(f"\nMonthly median atmiv/vol50 on 0DTE around the break: {', '.join(f'{k} {v:.3f}' for k, v in mon.items())}.")
    say("\nPrior (CR-BP Step 0): at 09:33 ET, 0DTE vol50(09:33)/vol50(09:50) E0 0.523, E1 1.035; atmiv E0 0.622, E1 0.625; "
        "CR-BO median atmiv/vol50 at 09:31 on 0DTE 0.706.\n")
    e0, e1 = R[R.era == "E0"], R[R.era == "E1"]
    band = lambda x: 0.95 <= x <= 1.05
    g_a = (not band(e0.r0.median())) and band(e1.r0.median())
    g_b = abs(e0.r0.median() / e1.r0.median() - 1) > 0.05
    g_c = (not band(e0.v50_0.median())) and band(e1.v50_0.median())
    say(f"**Gate:** (a) {g_a} — E0 {e0.r0.median():.3f}, E1 {e1.r0.median():.3f} vs band 0.95–1.05; "
        f"(b) {g_b} — E0/E1 − 1 = {e0.r0.median() / e1.r0.median() - 1:+.3f}; "
        f"(c) {g_c} — vol50 10:30/10:50 E0 {e0.v50_0.median():.3f}, E1 {e1.v50_0.median():.3f}.")
    tripped = g_a or g_b or g_c
    say(f"\n**{'STOP — the E0 depression persists at 10:30; spec amendment required.' if tripped else 'Gate PASS — the E0 depression does not persist at 10:30.'}**\n")

    # ---------------------------------------------------------------- item 4 (vol25 / vol75)
    say("## 4. `vol25` / `vol75` mapping\n")
    say("- Code: `vol_features.py:21-22` — `vol25` = 25Δ call, `vol75` = 25Δ put (the delta grid is call delta; call Δ 0.75 sits at a strike below spot, i.e. the put wing). CR-BO / CR-BP Step 0: `vol75` = put side, `vol25` = call side.")
    say(f"- Data, 0DTE at the snapshot: vol75 > vol25 on {(R.k > 0).mean() * 100:.1f} % of universe days; K = (vol75 − vol25)/vol50 by era: E0 {med_iqr(e0.k.to_numpy())}, E1 {med_iqr(e1.k.to_numpy())}. "
        "Positive K = put wing richer, consistent with the mapping (index skew).")
    say("- So L2(b) K = (vol75 − vol25)/vol50 is put-side minus call-side, as written.\n")

    # ---------------------------------------------------------------- item 5 (morning RV)
    say("## 5. Morning-RV availability (ES `ironbeam_es_1m_bars`, bar-start ET)\n")
    for e in ("E0", "E1"):
        x = U[U.era == e]
        say(f"- {e}: morning bars 09:30 → 10:30 {med_iqr(x.n_morn.to_numpy())} (expected 60; min {x.n_morn.min()}); "
            f"remaining bars 10:30 → 16:00 {med_iqr(x.n_rem.to_numpy())} (expected 330; min {x.n_rem.min()}); "
            f"days with < 57 morning bars: {int((x.n_morn < 57).sum())}.")
    say("- Returns are log close / prior stored close on the continuous series; roll weeks are excluded (L1(iii)), so no contract switch falls inside an RTH window.\n")

    # ---------------------------------------------------------------- item 6 (walk-forward)
    say("## 6. Walk-forward window sizes\n")
    n = len(U)
    U = U.sort_values("t").reset_index(drop=True)
    oos = U.iloc[WF_START:]
    say(f"- Universe n = {n} (E0 {int((U.era == 'E0').sum())}, E1 {int((U.era == 'E1').sum())}).")
    if n > WF_START:
        say(f"- First fit window: days 1–{WF_START}, {U.t.iloc[0]} → {U.t.iloc[WF_START - 1]}. First prediction {U.t.iloc[WF_START]}.")
        say(f"- OOS predictions: {len(oos)} (E0 {int((oos.era == 'E0').sum())}, E1 {int((oos.era == 'E1').sum())}). Final fit window {n - 1} days.")
        say(f"- Expanding-rank quintiles start after 20 predictions: first ranked day {U.t.iloc[WF_START + 19]}; ranked OOS days {max(len(oos) - 19, 0)}.")
        if int((oos.era == 'E0').sum()) < 40:
            say("- **E0 OOS n < 40**: the by-era E0 Q5 − Q1 carries a \"cell n < 40\" note (spec reading).")

    (OUT_DIR / "step0.md").write_text("\n".join(L) + "\n")
    pd.DataFrame([dict(t=u["t"], era=u["era"], snap=u["snap"], fomc=u["fomc"], ev=u["ev"]) for u in uni]).to_csv(OUT_DIR / "step0_universe.csv", index=False)
    (OUT_DIR / "step0_gate.json").write_text(json.dumps(dict(gate_a=bool(g_a), gate_b=bool(g_b), gate_c=bool(g_c), stop=bool(tripped),
                                                              e0_r0=float(e0.r0.median()), e1_r0=float(e1.r0.median()),
                                                              e0_v50=float(e0.v50_0.median()), e1_v50=float(e1.v50_0.median()), n=n), indent=1))
    print(f"\nwrote {OUT_DIR / 'step0.md'}", flush=True)


if __name__ == "__main__":
    main()
