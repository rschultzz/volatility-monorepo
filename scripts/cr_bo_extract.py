#!/usr/bin/env python3
"""CR-BO extraction — monies snapshots + ES overnight / RTH sums, one row per night × bucket × side.
READ-ONLY: SELECTs only, no run row, no API calls. Holdout never read (t+1 ≤ 2026-06-05).

Spec: specs/CR-BO-session-premium.md (L1–L5, L6 flags, L9 placebo input, L10 output).

Per night (t, t+1) in the L1 universe and per bucket S1 / S2 / S7 (L2) and side
atm (`atmiv`) / put (`vol75`) / call (`vol25`) (L6(d), Step 0 Q4):

  v_a = iv²τ at the 15:59 ET row on t        τ = calendar time to 16:00 ET on expir_date / 365 (Step 0 Q3)
  v_b = iv²τ at the 09:31 ET row on t+1
  v_c = iv²τ at the 15:59 ET row on t+1
  D_on  = v_a − v_b            RV_on  = Σ r²  over 1-min ES log returns, bars 16:00 t … 09:29 t+1
  D_rth = v_b − v_c            RV_rth = Σ r²  over bars 09:30 … 15:59 t+1
  P_on  = D_on − RV_on         P_rth  = D_rth − RV_rth        (total variance, no annualization — L4)

r_i = ln(close_i / close_{i−1}) with close_{i−1} the previous bar in the table (bar-start stamps, so bar
15:59's close is the 16:00 print); the 17:00–17:59 halt and weekends contribute only the reopen gap,
which belongs to the first bar after them. RV_on_next = RV_on of the night (t+1, t+2) when that night
passes the ES-side exclusions (L9 placebo input).

Universe: rebuilt here with the Step 0 definitions (scripts/cr_bo_common.py) and checked against
scripts/output/cr_bo/step0_pairs.csv (gate: identical keep set).

Usage: apps/web/.venv/bin/python -u scripts/cr_bo_extract.py
"""
from __future__ import annotations

import os
import sys
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

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from scripts import cr_bo_common as C
from scripts.cr_bo_step0 import BUCKETS, MAX_MISSING, SWITCH_BASIS_JUMP, load_es, load_monies

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bo"
SIDES = {"atm": "atmiv", "put": "vol75", "call": "vol25"}


def detect_switches(es: pd.DataFrame, pm: pd.DataFrame) -> list[datetime]:
    """Same detector as Step 0 Q2: basis jump > SWITCH_BASIS_JUMP, switch minute = largest |r| between
    the two 15:59 bars."""
    es1559 = es[es.hm == "15:59"].set_index("d").close
    spx1559 = pm[pm.hm == "15:59"].sort_values("dte").groupby("td").first().stock_price
    b = pd.concat([es1559.rename("es"), spx1559.rename("spx")], axis=1, join="inner")
    b["db"] = (b.es - b.spx).diff()
    idx = list(b.index); out = []
    for d in b.index[b.db > SWITCH_BASIS_JUMP]:
        prev = idx[idx.index(d) - 1]
        w = es[(es.et > datetime.combine(prev, time(15, 59))) & (es.et <= datetime.combine(d, time(15, 59)))]
        out.append(w.loc[w.r.abs().idxmax()].et.to_pydatetime())
    return out


def es_side(es: pd.DataFrame, t: date, t1: date, switches: list[datetime], minutes: set) -> dict:
    """ES-side flags and sums for one night. Exclusion flags match Step 0."""
    skip = C.holiday_dates_between(t, t1)
    on_lo, on_hi = C.overnight_bounds(t, t1); r_lo, r_hi = C.rth_bounds(t1)
    exp_on = C.window_minutes(on_lo, on_hi, skip); exp_rth = C.window_minutes(r_lo, r_hi)
    on_miss = 1 - sum(1 for x in exp_on if x in minutes) / len(exp_on)
    rth_miss = 1 - sum(1 for x in exp_rth if x in minutes) / len(exp_rth)
    anchor = any((datetime.combine(t, time(15, 55)) + timedelta(minutes=k)) in minutes for k in range(5))
    wlo, whi = datetime.combine(t, time(16, 0)), datetime.combine(t1, time(16, 0))
    w_on = es[(es.et >= on_lo) & (es.et < on_hi)]
    w_rth = es[(es.et >= r_lo) & (es.et < r_hi)]
    return dict(
        early=(t in C.EARLY_CLOSES or t1 in C.EARLY_CLOSES), roll=(C.roll_week(t) or C.roll_week(t1)),
        switch=any(wlo < s <= whi for s in switches), anchor=anchor, on_miss=on_miss, rth_miss=rth_miss,
        RV_on=float((w_on.r ** 2).sum()), RV_rth=float((w_rth.r ** 2).sum()), n_on=len(w_on), n_rth=len(w_rth),
        es_close_t=float(es.close.iloc[es.et.searchsorted(datetime.combine(t, time(15, 59)), side="right") - 1]),
    )


def es_ok(x: dict) -> bool:
    return (not x["early"] and not x["roll"] and not x["switch"] and x["anchor"]
            and x["on_miss"] <= MAX_MISSING and x["rth_miss"] <= MAX_MISSING)


def main() -> None:
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    print("loading …", flush=True)
    m = load_monies(conn); es = load_es(conn)
    es["hm"] = es.et.dt.strftime("%H:%M"); es["d"] = es.et.dt.date
    pm = m[m.expiry_tod == "pm"]
    switches = detect_switches(es, pm)
    print(f"switches: {[s.isoformat() for s in switches]}", flush=True)
    minutes = set(es.et)
    rows = {(r.td, r.ex, r.hm): r for r in pm.itertuples()}
    exps_by_t = pm[pm.hm == "15:59"].groupby("td").ex.apply(lambda s: sorted(set(s)))

    pairs = C.session_pairs()
    es_info = {(t, t1): es_side(es, t, t1, switches, minutes) for t, t1 in pairs}
    nxt = {t: t1 for t, t1 in pairs}

    out = []
    keep_nights = set()
    for t, t1 in pairs:
        x = es_info[(t, t1)]
        if not es_ok(x):
            continue
        keep_nights.add(t)
        t2 = nxt.get(t1)
        xn = es_info.get((t1, t2)) if t2 else None
        rv_next = xn["RV_on"] if (xn is not None and es_ok(xn)) else np.nan
        exps = exps_by_t.get(t, [])
        tgt = C.bucket_expiries(t)
        s7 = [e for e in exps if (e - t).days >= 7]
        tgt["S7"] = s7[0] if s7 else None
        ev = C.event_tags(t1)
        for bk in BUCKETS:
            e = tgt[bk]
            ka, kb, kc = (t, e, "15:59"), (t1, e, "09:31"), (t1, e, "15:59")
            if e is None or e not in exps or kb not in rows or kc not in rows:
                continue
            ra, rb, rc = rows[ka], rows[kb], rows[kc]
            if not all(pd.notna(r.atmiv) and r.atmiv > 0 for r in (ra, rb, rc)):
                continue
            ta = C.tau_years(ra.et.to_pydatetime(), e); tb = C.tau_years(rb.et.to_pydatetime(), e)
            tc = C.tau_years(rc.et.to_pydatetime(), e)
            for side, col in SIDES.items():
                ia, ib, ic = getattr(ra, col), getattr(rb, col), getattr(rc, col)
                va, vb, vc = ia * ia * ta, ib * ib * tb, ic * ic * tc
                out.append(dict(
                    t=t, t1=t1, weekday=t.strftime("%a"), cal_gap=(t1 - t).days, holiday_span=bool(C.holiday_dates_between(t, t1)),
                    bucket=bk, expir_date=e, cal_dte=(e - t).days, side=side,
                    iv_a=ia, iv_b=ib, iv_c=ic, tau_a=ta, tau_b=tb, tau_c=tc, v_a=va, v_b=vb, v_c=vc,
                    D_on=va - vb, D_rth=vb - vc, RV_on=x["RV_on"], RV_rth=x["RV_rth"],
                    P_on=(va - vb) - x["RV_on"], P_rth=(vb - vc) - x["RV_rth"], RV_on_next=rv_next,
                    n_on=x["n_on"], n_rth=x["n_rth"], on_miss=x["on_miss"], rth_miss=x["rth_miss"],
                    stock_price_t=ra.stock_price, es_close_t=x["es_close_t"],
                    q_a=str(ra.quote_date), q_b=str(rb.quote_date), q_c=str(rc.quote_date),
                    event_0830=bool(ev), event_tags=",".join(ev), fomc_t1=(t1 in C.FOMC_DAYS),
                ))
    df = pd.DataFrame(out)

    # gate: universe identical to Step 0
    s0 = pd.read_csv(OUT_DIR / "step0_pairs.csv", parse_dates=["t"])
    s0_keep = set(s0[s0.keep].t.dt.date)
    if s0_keep != keep_nights:
        raise SystemExit(f"universe mismatch vs Step 0: only-here {sorted(keep_nights - s0_keep)[:10]} only-step0 {sorted(s0_keep - keep_nights)[:10]}")
    for bk in BUCKETS:
        n0 = int((s0.keep & s0[f"{bk}_ok"]).sum()); n1 = df[(df.bucket == bk) & (df.side == "atm")].t.nunique()
        if n0 != n1:
            raise SystemExit(f"bucket {bk}: step0 {n0} vs extract {n1}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "nights.parquet", index=False)
    print(f"universe gate PASS ({len(keep_nights)} ES-clean nights); rows {len(df)}; "
          f"per bucket (atm): {df[df.side == 'atm'].groupby('bucket').size().to_dict()}")
    print(f"wrote {OUT_DIR / 'nights.parquet'}")


if __name__ == "__main__":
    main()
