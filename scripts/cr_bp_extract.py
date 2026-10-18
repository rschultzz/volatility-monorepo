#!/usr/bin/env python3
"""CR-BP extraction — CR-BO's night × bucket × side rows with the open-side variance v_b per P1, plus the P2 read.
READ-ONLY: SELECTs only, no run row, no API calls. Holdout never read (t+1 ≤ 2026-06-05).

Spec: specs/CR-BP-session-premium-live-open.md (P1, P2, L1–L10) as amended by A1 (era-aware P3; decision via
scripts/output/cr_bp/p3_decision_a1.json). Era of a night: E0 t < 2024-03-01, E1 t ≥ 2024-03-01.

Per night (t, t+1) in the CR-BO universe (ES side identical — gate) and bucket S1 / S2 / S7:
  atm side (gated):
    v_a = atmiv²τ at 15:59 ET on t            (C3, unchanged from CR-BO)
    v_b = P1 open-side variance at 09:31 ET on t+1:
          V_px (09:31, primary subset, K-aware solve — A1.2) of t+1's 0DTE contract (S1 only — the only captured contract);
          else the extension field²τ of the night's era × bucket cell (A1.3); else P1 is NaN for that row (p1 = False).
          P2 columns are filled on every CR-BO-universe night (A1 reading r3).
    v_c = atmiv²τ at 15:59 ET on t+1          (unchanged)
  put / call sides: exactly CR-BO's (vol75 / vol25 at all three snapshots; not gated).
  P2 (atm rows): v_b2 = atmiv²τ at 09:50 ET; RV_on2 = Σ r² 16:00 t → 09:50 t+1, RV_rth2 = Σ r² 09:50 → 16:00 t+1 (C6(i)).
RV_on / RV_rth (P1) = CR-BO's L4 windows (16:00 → 09:30, 09:30 → 16:00).

Usage: apps/web/.venv/bin/python -u scripts/cr_bp_extract.py
"""
from __future__ import annotations

import json
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
from scripts.cr_bo_extract import detect_switches, es_ok, es_side
from scripts.cr_bo_step0 import BUCKETS, load_es, year_chunks

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bp"
SIDES = {"atm": "atmiv", "put": "vol75", "call": "vol25"}

MONIES_SQL = """
SELECT trade_date, expir_date, expiry_tod, dte, snapshot_pt, atmiv, vol25, vol50, vol75, stock_price, quote_date
FROM orats_monies_minute
WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND dte BETWEEN 1 AND 16
  AND snapshot_pt::time IN ('06:31', '06:50', '12:59')
"""


def load_monies(conn) -> pd.DataFrame:
    parts = [pd.read_sql(MONIES_SQL, conn, params=(lo.isoformat(), hi.isoformat())) for lo, hi in year_chunks(C.TRAIN_START, C.TRAIN_END)]
    m = pd.concat(parts, ignore_index=True)
    m = m[m.expiry_tod == "pm"].copy()
    m["td"] = pd.to_datetime(m.trade_date).dt.date; m["ex"] = pd.to_datetime(m.expir_date).dt.date
    m["et"] = m.snapshot_pt + pd.Timedelta(hours=3); m["hm"] = m.et.dt.strftime("%H:%M")
    return m


def rv(es: pd.DataFrame, lo: datetime, hi: datetime) -> float:
    w = es[(es.et >= lo) & (es.et < hi)]
    return float((w.r ** 2).sum())


def main() -> None:
    dec = json.loads((OUT_DIR / "p3_decision_a1.json").read_text())
    cut = date.fromisoformat(dec["cut"])
    vp = pd.read_parquet(OUT_DIR / "vpx_primary.parquet")   # A1.2: V_px from the primary subset in both eras
    vpx = {d: v for d, v in zip(pd.to_datetime(vp.d).dt.date, vp["vpx_06:31"]) if v > 0}
    print(f"A1: cut {cut}, cells {{k: v['extension'] for k, v in dec['cells'].items()}}, V_px days {len(vpx)}", flush=True)

    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    m = load_monies(conn); es = load_es(conn)
    es["hm"] = es.et.dt.strftime("%H:%M"); es["d"] = es.et.dt.date
    # switch detector needs the 15:59 rows only
    switches = detect_switches(es, m.rename(columns={}))
    minutes = set(es.et)
    rows = {(r.td, r.ex, r.hm): r for r in m.itertuples()}
    exps_by_t = m[m.hm == "15:59"].groupby("td").ex.apply(lambda s: sorted(set(s)))

    pairs = C.session_pairs()
    es_info = {(t, t1): es_side(es, t, t1, switches, minutes) for t, t1 in pairs}
    nxt = {t: t1 for t, t1 in pairs}
    out, keep_nights, drops = [], set(), {b: 0 for b in BUCKETS}
    for t, t1 in pairs:
        x = es_info[(t, t1)]
        if not es_ok(x):
            continue
        keep_nights.add(t)
        t2 = nxt.get(t1); xn = es_info.get((t1, t2)) if t2 else None
        rv_next = xn["RV_on"] if (xn is not None and es_ok(xn)) else np.nan
        on2 = rv(es, datetime.combine(t, time(16, 0)), datetime.combine(t1, time(9, 50)))
        rth2 = rv(es, datetime.combine(t1, time(9, 50)), datetime.combine(t1, time(16, 0)))
        exps = exps_by_t.get(t, [])
        tgt = C.bucket_expiries(t); s7 = [e for e in exps if (e - t).days >= 7]; tgt["S7"] = s7[0] if s7 else None
        ev = C.event_tags(t1)
        for bk in BUCKETS:
            e = tgt[bk]
            ka, kb, kc, kb2 = (t, e, "15:59"), (t1, e, "09:31"), (t1, e, "15:59"), (t1, e, "09:50")
            if e is None or e not in exps or kb not in rows or kc not in rows:
                continue
            ra, rb, rc = rows[ka], rows[kb], rows[kc]
            if not all(pd.notna(r.atmiv) and r.atmiv > 0 for r in (ra, rb, rc)):
                continue   # CR-BO bucket rule (L1/L3 rows present) — identical universe per bucket
            ta = C.tau_years(ra.et.to_pydatetime(), e); tb = C.tau_years(rb.et.to_pydatetime(), e); tc = C.tau_years(rc.et.to_pydatetime(), e)
            # P1 open-side variance (A1.2 / A1.3)
            era = "E0" if t < cut else "E1"
            ext = dec["cells"][f"{era}_{bk}"]["extension"]
            if bk == "S1" and t1 in vpx:
                vb_atm, src = float(vpx[t1]), "vpx"
            elif ext and pd.notna(getattr(rb, ext)) and getattr(rb, ext) > 0:
                vb_atm, src = getattr(rb, ext) ** 2 * tb, ext
            else:
                vb_atm, src = np.nan, "none"
                drops[bk] += 1
            p1 = src != "none"
            rb2 = rows.get(kb2)
            for side, col in SIDES.items():
                ia, ic = getattr(ra, col), getattr(rc, col)
                va, vc = ia * ia * ta, ic * ic * tc
                vb = (vb_atm if side == "atm" else getattr(rb, col) ** 2 * tb) if p1 else np.nan
                rec = dict(
                    t=t, t1=t1, weekday=t.strftime("%a"), cal_gap=(t1 - t).days, holiday_span=bool(C.holiday_dates_between(t, t1)),
                    bucket=bk, expir_date=e, cal_dte=(e - t).days, side=side, era=era, p1=p1, vb_source=src if side == "atm" else col,
                    v_a=va, v_b=vb, v_c=vc, v_b_atmiv=rb.atmiv ** 2 * tb, v_b_vol50=(rb.vol50 ** 2 * tb) if rb.vol50 else np.nan,
                    D_on=va - vb, D_rth=vb - vc, RV_on=x["RV_on"], RV_rth=x["RV_rth"],
                    P_on=(va - vb) - x["RV_on"], P_rth=(vb - vc) - x["RV_rth"], RV_on_next=rv_next,
                    stock_price_t=ra.stock_price, event_0830=bool(ev), event_tags=",".join(ev), fomc_t1=(t1 in C.FOMC_DAYS),
                )
                if side == "atm":
                    if rb2 is not None and pd.notna(rb2.atmiv) and rb2.atmiv > 0:
                        tb2 = C.tau_years(rb2.et.to_pydatetime(), e); vb2 = rb2.atmiv ** 2 * tb2
                        rec.update(v_b2=vb2, D_on2=va - vb2, D_rth2=vb2 - vc, RV_on2=on2, RV_rth2=rth2,
                                   P_on2=(va - vb2) - on2, P_rth2=(vb2 - vc) - rth2)
                    else:
                        rec.update(v_b2=np.nan, D_on2=np.nan, D_rth2=np.nan, RV_on2=on2, RV_rth2=rth2, P_on2=np.nan, P_rth2=np.nan)
                out.append(rec)
    df = pd.DataFrame(out)

    # gates: ES universe identical to Step 0; per-bucket P1 coverage identical to the Step 0 decision line
    s0 = pd.read_csv(OUT_DIR / "step0_pairs.csv", parse_dates=["t"])
    if set(s0[s0.keep].t.dt.date) != keep_nights:
        raise SystemExit("ES universe mismatch vs Step 0")
    cov = dec["coverage"]
    for key, c_ in cov.items():
        e_, bk = key.split("_")
        got = df[(df.side == "atm") & (df.bucket == bk) & (df.era == e_)]
        if len(got) != c_["universe"] or int(got.p1.sum()) != c_["vpx"] + c_["extended"] or int((got.vb_source == "vpx").sum()) != c_["vpx"]:
            raise SystemExit(f"{key}: extract n {len(got)} / p1 {int(got.p1.sum())} / vpx {int((got.vb_source == 'vpx').sum())} vs Step 0 A1 {c_}")
    df.to_parquet(OUT_DIR / "nights.parquet", index=False)
    a = df[df.side == "atm"]
    print(f"gates PASS; rows {len(df)}; atm per bucket {a.groupby('bucket').size().to_dict()}; v_b source {a.groupby(['era','bucket','vb_source']).size().to_dict()}; P1-NaN rows {drops}")
    print(f"wrote {OUT_DIR / 'nights.parquet'}")


if __name__ == "__main__":
    main()
