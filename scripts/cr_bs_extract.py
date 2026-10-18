#!/usr/bin/env python3
"""CR-BS extraction — one row per universe day: the 10:30 ET state, the remaining-day outcome on both clocks,
side semivariances, realized skew, and the t−1 dealer-loss proxy.
READ-ONLY: SELECTs only, no run row, no API calls. Holdout never read (t ≤ 2026-06-05).

Spec: specs/CR-BS-morning-surface.md (L1–L3, L5 secondaries, readings fixed at the freeze).
Universe gate: the day set must equal scripts/output/cr_bs/step0_universe.csv exactly.

Per day t (snapshot = first monies minute in [07:30, 07:33] PT; r0 = same-day PM expiry, r1 = next-session PM expiry):
  L  = log(atmiv0²)                         K = (vol75 − vol25)/vol50 (0DTE)       C = (vol25 + vol75 − 2·vol50)/vol50 (0DTE)
  T  = log(atmiv0² / atmiv1²)               M = log(RV_09:30→10:30 / (atmiv0² · τ_morn))
  Y  = log(RV_10:30→16:00 / (atmiv0² · τ_rem))   τ session clock (5.5/6.5)/252, (1/6.5)/252; Y_cal, M_cal on 1/365
  *_v50 columns: the same with vol50 in place of atmiv (reported beside, never gated)
  IV_up / IV_dn: 2∫_F^∞ C/K² dK, 2∫_0^F P/K² dK on the 0DTE smile (delta grid → strikes by Black, F = stock_price,
                 r = q = 0, τ = τ_rem session); RV_up / RV_dn = Σ r² over r > 0 / r < 0, 10:30 → 16:00
  rskew = √N Σr³ / (Σr²)^{3/2}, 1-min, 10:30 → 16:00
  loss_prev = |close_move_over_im| − 0.798 on the previous session (bt_daily_outcomes; 0.798 = √(2/π), the ATM
              straddle in IM units), > 0 = a short straddle at the implied move lost

Usage: apps/web/.venv/bin/python -u scripts/cr_bs_extract.py
"""
from __future__ import annotations

import json
import os
import sys
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
from scipy.stats import norm

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from packages.shared.trading_calendar import prev_trading_day
from scripts import cr_bs_common as S
from scripts.cr_bo_common import EARLY_CLOSES, FOMC_DAYS, event_tags, roll_week
from scripts.cr_bo_step0 import load_es

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bs"
DELTAS = list(range(5, 100, 5))
STRADDLE_IM = float(np.sqrt(2 / np.pi))

OUTCOMES_SQL = """
SELECT trade_date, horizon_sessions, close_move_over_im, range_over_im
FROM bt_daily_outcomes
WHERE active AND ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND close_move_over_im IS NOT NULL
ORDER BY trade_date, horizon_sessions
"""


def black(F, K, v, call):
    s = np.sqrt(v)
    d1 = (np.log(F / K) + v / 2) / s
    if call:
        return F * norm.cdf(d1) - K * norm.cdf(d1 - s)
    return K * norm.cdf(s - d1) - F * norm.cdf(-d1)


def semivariances(r: pd.Series, tau: float) -> tuple[float, float]:
    """Model-free up/down implied variance from the delta-grid smile (spec reading L5(i))."""
    F = float(r.stock_price)
    ks, vs = [], []
    for d in DELTAS:
        sig = r[f"vol{d}"]
        if not (pd.notna(sig) and sig > 0):
            continue
        sd = sig * np.sqrt(tau)
        ks.append(F * np.exp(-sd * norm.ppf(d / 100) + sd * sd / 2)); vs.append(sig)
    if len(ks) < 5:
        return np.nan, np.nan
    o = np.argsort(ks); ks, vs = np.asarray(ks)[o], np.asarray(vs)[o]
    sd0 = float(r.atmiv) * np.sqrt(tau)
    grid_up = F * np.exp(np.linspace(0, 10 * sd0, 2001)); grid_dn = F * np.exp(np.linspace(-10 * sd0, 0, 2001))
    iv = lambda K: np.interp(K, ks, vs)          # linear in strike, flat beyond the grid
    cu = black(F, grid_up, iv(grid_up) ** 2 * tau, True)
    pdn = black(F, grid_dn, iv(grid_dn) ** 2 * tau, False)
    up = 2 * np.trapz(cu / grid_up ** 2, grid_up); dn = 2 * np.trapz(pdn / grid_dn ** 2, grid_dn)
    return float(up), float(dn)


def main() -> None:
    gate = json.loads((OUT_DIR / "step0_gate.json").read_text())
    assert not gate["stop"], "Step 0 gate tripped — no extraction"
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    print("loading …", flush=True)
    m = S.load_monies(conn); es = load_es(conn)
    snaps = S.snapshot_rows(m)
    oc = pd.read_sql(OUTCOMES_SQL, conn, params=(S.TRAIN_START.isoformat(), S.TRAIN_END.isoformat()))
    oc["td"] = pd.to_datetime(oc.trade_date).dt.date
    oc = oc.drop_duplicates("td")      # close_move_over_im is a t0-session column, identical across horizons
    cmo = dict(zip(oc.td, oc.close_move_over_im))

    out = []
    for t in S.train_days():
        if t in EARLY_CLOSES or roll_week(t) or t not in snaps:
            continue
        rows = snaps[t]
        r0, r1 = S.pick(rows, t), S.pick(rows, S.next_session(t))
        if not S.smile_ok(r0) or r1 is None or not (pd.notna(r1.atmiv) and r1.atmiv > 0):
            continue
        w = S.rth_frame(es, t)
        if len(w) < S.MIN_RTH_BARS:
            continue
        a, b, c = datetime.combine(t, time(9, 30)), datetime.combine(t, time(10, 30)), datetime.combine(t, time(16, 0))
        mo, mo_up, mo_dn, n_mo = S.rv(w, a, b)
        re, re_up, re_dn, n_re = S.rv(w, b, c)
        if n_mo < 1 or n_re < 1:
            continue
        x = w[(w.et >= b) & (w.et < c)].r.to_numpy(); x = x[np.isfinite(x)]
        rskew = float(np.sqrt(len(x)) * (x ** 3).sum() / (x ** 2).sum() ** 1.5)
        v0, v50 = r0.atmiv ** 2, r0.vol50 ** 2
        v1 = r1.atmiv ** 2
        v1_50 = r1.vol50 ** 2 if (pd.notna(r1.vol50) and r1.vol50 > 0) else np.nan
        iv_up, iv_dn = semivariances(r0, S.TAU_REM_SESSION)
        tp = prev_trading_day(t)
        cm = cmo.get(tp)
        out.append(dict(
            t=t, era=S.era(t), snap_pt=rows.snapshot_pt.iloc[0], weekday=t.strftime("%a"),
            fomc=t in FOMC_DAYS, ev0830=",".join(event_tags(t)),
            atmiv0=r0.atmiv, vol25_0=r0.vol25, vol50_0=r0.vol50, vol75_0=r0.vol75, slope0=r0.slope, atmiv1=r1.atmiv, vol50_1=r1.vol50,
            next_expiry=r1.ex, next_cal_gap=(r1.ex - t).days, stock_price=r0.stock_price,
            L=np.log(v0), K=(r0.vol75 - r0.vol25) / r0.vol50, C=(r0.vol25 + r0.vol75 - 2 * r0.vol50) / r0.vol50,
            T=np.log(v0 / v1), M=np.log(mo / (v0 * S.TAU_MORN_SESSION)),
            Y=np.log(re / (v0 * S.TAU_REM_SESSION)), Y_cal=np.log(re / (v0 * S.TAU_REM_CAL)), M_cal=np.log(mo / (v0 * S.TAU_MORN_CAL)),
            L_v50=np.log(v50), T_v50=np.log(v50 / v1_50), M_v50=np.log(mo / (v50 * S.TAU_MORN_SESSION)),
            Y_v50=np.log(re / (v50 * S.TAU_REM_SESSION)),
            RV_morn=mo, RV_rem=re, RV_up=re_up, RV_dn=re_dn, n_morn=n_mo, n_rem=n_re, n_rth=len(w),
            IV_up=iv_up, IV_dn=iv_dn, IV_rem=v0 * S.TAU_REM_SESSION,
            Y_up=np.log(re_up / iv_up) if iv_up > 0 and re_up > 0 else np.nan,
            Y_dn=np.log(re_dn / iv_dn) if iv_dn > 0 and re_dn > 0 else np.nan,
            rskew=rskew, prev_session=tp, cmo_prev=cm,
            loss_prev=(abs(cm) - STRADDLE_IM) if cm is not None and pd.notna(cm) else np.nan,
        ))
    D = pd.DataFrame(out).sort_values("t").reset_index(drop=True)
    u0 = pd.read_csv(OUT_DIR / "step0_universe.csv")
    same = sorted(pd.to_datetime(u0.t).dt.date) == list(D.t)
    print(f"universe gate (== step0_universe.csv): {'PASS' if same else 'FAIL'} — {len(D)} vs {len(u0)}", flush=True)
    assert same, "extraction universe differs from Step 0"
    D.to_parquet(OUT_DIR / "days.parquet", index=False)
    # A1.4: does the 0DTE atmiv repeat across adjacent minutes at 07:30 PT?
    z = m[(m.expiry_tod == "pm") & (m.ex == m.td) & m.hm.isin(["07:30", "07:31", "07:32", "07:33"]) & m.td.isin(set(D.t))]
    pv = z.pivot_table(index="td", columns="hm", values="atmiv", aggfunc="first")
    rep = {}
    for a, b in (("07:30", "07:31"), ("07:31", "07:32"), ("07:32", "07:33")):
        ok = pv[[a, b]].dropna()
        rep[f"{a}={b}"] = dict(share=float((ok[a] == ok[b]).mean()), n=len(ok))
    ok = pv[["07:30", "07:31", "07:32", "07:33"]].dropna()
    rep["07:30..07:33 all equal"] = dict(share=float(ok.nunique(axis=1).eq(1).mean()), n=len(ok))
    v50 = z.pivot_table(index="td", columns="hm", values="vol50", aggfunc="first")[["07:30", "07:31"]].dropna()
    rep["vol50 07:30=07:31"] = dict(share=float((v50["07:30"] == v50["07:31"]).mean()), n=len(v50))
    for e, days in (("E0", set(D[D.era == "E0"].t)), ("E1", set(D[D.era == "E1"].t))):
        s = pv[pv.index.isin(days)][["07:30", "07:31"]].dropna()
        rep[f"07:30=07:31 {e}"] = dict(share=float((s["07:30"] == s["07:31"]).mean()), n=len(s))
    meta = dict(n=len(D), n_E0=int((D.era == "E0").sum()), n_E1=int((D.era == "E1").sum()),
                loss_prev_n=int(D.loss_prev.notna().sum()), semivar_n=int(D.IV_up.notna().sum()),
                implied_share_up_median=float((D.IV_up / (D.IV_up + D.IV_dn)).median()),
                implied_sum_over_atm_median=float(((D.IV_up + D.IV_dn) / D.IV_rem).median()),
                atmiv_repeat_0dte=rep, extracted_at=f"{datetime.now():%Y-%m-%d %H:%M} PT")
    (OUT_DIR / "extract_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
