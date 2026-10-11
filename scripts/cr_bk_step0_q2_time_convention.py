#!/usr/bin/env python3
"""CR-BK Step 0 (read-only, train only) — see specs/CR-BK-corrected-gamma-profile-test.md. Run: apps/web/.venv/bin/python scripts/cr_bk_step0_q2_time_convention.py"""
import sys, datetime as dt; from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
import numpy as np, pandas as pd
from scipy.stats import norm
from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.trading_calendar import prev_trading_day
c = get_backfill_db_conn(); c.execute("SET default_transaction_read_only = on"); c.execute("SET statement_timeout = '120s'")
q = lambda s, a=None: c.execute(s, a).fetchall()
DIAG = []
DELTAS = list(range(5, 100, 5))
dates = [r[0] for r in q("select distinct trade_date from orats_oi_gamma where ticker='SPX' and trade_date between '2023-05-02' and '2026-06-05' order by 1")]
rng = np.random.default_rng(20261011)
sample = sorted(rng.choice(dates, 20, replace=False))

def smile_strikes(m, F, T):
    """per delta point: strike for call forward-delta Δ with vol σ_Δ."""
    ks, vs = [], []
    for d in DELTAS:
        v = m[f"vol{d}"]
        if v is None or v <= 0: continue
        d1 = norm.ppf(d / 100.0)
        ks.append(F * np.exp(-d1 * v * np.sqrt(T) + 0.5 * v * v * T)); vs.append(v)
    o = np.argsort(ks); return np.array(ks)[o], np.array(vs)[o]

def run(snap_time, variant):
    errs, werrs, wts, nrow = [], [], [], 0
    for sd in sample:
        api = prev_trading_day(sd)
        g = pd.DataFrame(q("""select expir_date, strike::float8, stock_price::float8, gamma, call_oi, put_oi, short_rate, div_yield
                              from orats_oi_gamma where ticker='SPX' and trade_date=%s and expir_date>=%s and gamma>0""", (sd, sd)),
                         columns="exp K S gamma coi poi r q".split())
        snap = q("select max(snapshot_pt) from orats_monies_minute where trade_date=%s and snapshot_pt::time <= %s", (api.isoformat(), snap_time))[0][0]
        mon = q(f"""select expir_date, stock_price, risk_free_rate, yield_rate, {",".join(f"vol{d}" for d in DELTAS)}
                    from orats_monies_minute where trade_date=%s and snapshot_pt=%s and ticker='SPX'""", (api.isoformat(), snap))
        cols = ["exp", "S_m", "r_m", "q_m"] + [f"vol{d}" for d in DELTAS]
        mon = {dt.date.fromisoformat(r[0]): dict(zip(cols, r)) for r in mon}
        snap_dt = dt.datetime.combine(api, dt.time(12, 46))   # ORATS EOD snapshot (PT)
        for e, ge in g.groupby("exp"):
            m = mon.get(e)
            if m is None: continue
            T = ((dt.datetime.combine(e, dt.time(13, 0)) - snap_dt).total_seconds() / 86400) / 365.0
            if T <= 0: continue
            S = ge.S.iloc[0]; r = ge.r.iloc[0] or 0.0; qy = ge.q.iloc[0] or 0.0
            F = S if variant.startswith("fwd=S") else S * np.exp((r - qy) * T)
            ks, vs = smile_strikes(m, F, T)
            if len(ks) < 3: continue
            sig = np.interp(ge.K.to_numpy(), ks, vs)   # flat beyond Δ5/Δ95
            K = ge.K.to_numpy()
            d1 = (np.log(F / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
            gam = np.exp(-qy * T) * norm.pdf(d1) / (S * sig * np.sqrt(T)) if variant.endswith("bs") else norm.pdf(d1) / (S * sig * np.sqrt(T))
            pe = np.abs(gam / ge.gamma.to_numpy() - 1)
            w = ge.gamma.to_numpy() * (ge.coi + ge.poi).to_numpy() * S * S
            oi = (ge.coi + ge.poi).to_numpy() > 0
            z = np.log(K / F) / (sig * np.sqrt(T)); inside = (K >= ks[0]) & (K <= ks[-1])
            DIAG.append(pd.DataFrame({'pe': pe[oi], 'w': w[oi], 'z': z[oi], 'dte': (e - sd).days, 'inside': inside[oi], 'oi': (ge.coi + ge.poi).to_numpy()[oi], 'ratio': (gam / ge.gamma.to_numpy())[oi], 'cal': (e - api).days, 'bus': np.busday_count(api, e), 'Tc': T, 'sig': sig[oi]}))
            errs.append(pe[oi]); werrs.append(pe[oi]); wts.append(w[oi]); nrow += oi.sum()
    e = np.concatenate(errs); w = np.concatenate(wts)
    o = np.argsort(e); cw = np.cumsum(w[o]) / w.sum()
    wmed = e[o][np.searchsorted(cw, .5)]; wp90 = e[o][np.searchsorted(cw, .9)]
    print(f"snap≤{snap_time} {variant:14s} rows {nrow:6d} | median {100*np.median(e):6.2f}% p90 {100*np.quantile(e,.9):7.2f}% | GEX-weighted median {100*wmed:5.2f}% p90 {100*wp90:6.2f}%", flush=True)


run("12:46", "fwd=S,black")
d = pd.concat(DIAG)

a = d[d.z.abs() <= 0.5]
g = a.groupby(["cal", "bus"]).agg(n=("ratio", "size"), med_ratio=("ratio", "median"), sig=("sig", "median")).reset_index()
g["T_orats/T_ours (=ratio^2)"] = g.med_ratio ** 2
g["implied_T_days"] = g["T_orats/T_ours (=ratio^2)"] * (g.cal + 14/1440)
print(g[g.cal <= 40].round(4).to_string())
