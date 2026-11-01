#!/usr/bin/env python3
"""CR-CI Step 0 for Amendment A2 — direction rule reads, no outcomes. READ-ONLY.

Spec: specs/CR-CI-wide-stop-touch-trade.md — A2.0–A2.8, C14–C26. Machinery: scripts/cr_ci_lib.py.
Computed (no price after a touch bar's close is read for any statistic here):
  - approach-leg counts at swing 8 / 12 / 16 with reasons for no leg; push > 0 share; effort ratio distribution and
    ABSORBED / FREE / neutral counts;
  - skew coverage at leg start and touch (C17), fallback-expiry sessions, residual and raw change distributions,
    divergent / confirming counts, residual vs raw sign agreement;
  - n per call (primary, delta-only, skew-only, raw-skew) per swing size and per CR-CE cell;
  - the any-price population (C24): event counts and n per call at swing 12;
  - C20: snap_shot_date − snapshot_pt (hours) on the loaded rows;
  - C25: R-a availability (counts only).

Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_ci_step0_a2.py | tee scripts/logs/cr_ci/step0_a2.log
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_ci_lib as L  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_ci"
RES: dict = {}
E = L.R[L.R["D"] > 0].copy()
A = L.a2_reads(E)
assert len(A) == 803


def calls(df, col):
    v = df[col].value_counts()
    return {k: int(v.get(k, 0)) for k in ("FADE", "BREAK", "NO TRADE")}


def summarize(A):
    out = {}
    for sw in L.SWINGS:
        t = f"s{int(sw)}"
        ok = A[A[f"{t}_why"] == "ok"]
        blk = dict(n=int(len(A)), approach_leg=int(len(ok)), no_leg=A[f"{t}_why"].value_counts().drop("ok", errors="ignore").to_dict())
        if len(ok):
            blk.update(
                push_gt0=int((ok[f"{t}_push"] > 0).sum()),
                leg_pts=L.pts(ok[f"{t}_pts"]), leg_minutes=L.pts(ok[f"{t}_mins"]),
                ratio_defined=int(np.isfinite(ok[f"{t}_ratio"]).sum()),
                ratio_pcts=L.pts(ok[f"{t}_ratio"].dropna()),
                effort=ok[f"{t}_eff"].value_counts().to_dict(),
                effort_push_gt0=ok[ok[f"{t}_push"] > 0][f"{t}_eff"].value_counts().to_dict(),
                skew=ok[f"{t}_skew"].value_counts().to_dict(),
                skew_why=ok[f"{t}_skew_why"].value_counts().to_dict(),
                residual_pcts=L.pts(ok[f"{t}_res"].dropna()), raw_pcts=L.pts(ok[f"{t}_raw"].dropna()),
                residual_raw_sign_agree=float((np.sign(ok[f"{t}_res"]) == np.sign(ok[f"{t}_raw"]))[
                    np.isfinite(ok[f"{t}_res"]) & np.isfinite(ok[f"{t}_raw"])].mean()),
                residual_raw_corr=float(ok[[f"{t}_res", f"{t}_raw"]].dropna().corr().iloc[0, 1]))
        blk["calls"] = {c: calls(A, f"{t}_{c}") for c in ("call", "call_delta", "call_skew", "call_raw")}
        if "cell" in A and A["cell"].isin(L.CELLS).any():
            blk["calls_by_cell"] = {c: calls(A[A["cell"] == c], f"{t}_call") for c in L.CELLS}
        blk["calls_first_touch"] = calls(A[A["first"]], f"{t}_call")
        out[t] = blk
    return out


RES["primary_803"] = summarize(A)
RES["entry_minus_level"] = L.pts((A["entry"] - A["L"]) * A["side"])

# skew coverage at touch timestamps
cov_touch = np.mean([L.snap_at(x.d, x.k) is not None for x in A.itertuples()])
RES["skew_coverage"] = dict(touch_snapshot=float(cov_touch),
                            sessions_without_rows=[str(d) for d in L.USABLE if L.SKROWS[d].empty],
                            fallback_expiry_sessions=[[str(d), L.SKEXP[d][0]] for d in L.USABLE if L.SKEXP[d][1]],
                            rth_snapshots_per_session=L.pts([len(L.SKROWS[d]) for d in L.USABLE]))

# C20 snap_shot_date clock
off = []
for d in L.USABLE:
    r = L.SKROWS[d]
    if len(r) and "snap_shot_date" in r:
        x = (pd.to_datetime(r["snap_shot_date"], utc=True).dt.tz_localize(None) - r["snap_pt"]).dt.total_seconds() / 3600.0
        qlag = (pd.to_datetime(r["snap_shot_date"], utc=True) - pd.to_datetime(r["quote_date"], utc=True)).dt.total_seconds()
        off.append(pd.DataFrame(dict(d=d, off=x, qlag=qlag)))
off = pd.concat(off)
off["pt_to_utc"] = [-(L.pt(d, 12, 0).astimezone(L.PT).utcoffset().total_seconds() / 3600.0) for d in off["d"]]
off["dev_min"] = (off["off"] - off["pt_to_utc"]) * 60.0
RES["C20_snap_shot_date"] = dict(n=int(len(off)), dev_minutes_pcts=L.pts(off["dev_min"].dropna()),
                                 share_within_1min=float((off["dev_min"].abs() <= 1.0).mean()),
                                 offset_hours_values=off["off"].round(3).value_counts().head(5).to_dict(),
                                 quote_lag_seconds_pcts=L.pts(off["qlag"].dropna()))

# any-price population (C24)
AP, fun = L.build_anyprice()
AA = L.a2_reads(AP)
RES["anyprice"] = dict(funnel=fun, events=int(len(AP)), by_cell=AP["cell"].value_counts().to_dict(),
                       first_touch=int(AP["first"].sum()), sessions=int(AP["d"].nunique()),
                       s12=summarize(AA)["s12"])

# R-a availability (C25), counts only
ra_days = [d for d in L.trading_days(date(2024, 4, 3), date(2024, 12, 31))]
fl = L.q("""SELECT trade_date, count(*) AS n FROM es_flow_1s_databento
            WHERE ts_utc >= '2024-04-03' AND ts_utc < '2025-01-01'
              AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
              AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
            GROUP BY 1""")
sk = L.q("""SELECT trade_date, count(DISTINCT snapshot_pt) AS n FROM orats_monies_minute
            WHERE ticker = 'SPX' AND trade_date >= '2024-04-03' AND trade_date <= '2024-12-31'
              AND expir_date >= trade_date AND lower(expiry_tod) = 'pm'
              AND snapshot_pt::time >= '06:30' AND snapshot_pt::time < '13:00'
            GROUP BY 1""")
flow_days = {pd.Timestamp(x).date() for x, n in zip(fl["trade_date"], fl["n"]) if n > 0}
skew_days = {date.fromisoformat(str(x)[:10]) for x, n in zip(sk["trade_date"], sk["n"]) if n >= 300}
RES["R_a_availability"] = dict(nyse_sessions=len(ra_days), flow_sessions=len(set(ra_days) & flow_days),
                               skew_sessions_ge300_snapshots=len(set(ra_days) & skew_days),
                               both=len(set(ra_days) & flow_days & skew_days))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "step0_a2.json").write_text(json.dumps(RES, indent=1, default=str))
A.to_pickle(OUT / "step0_a2_reads.pkl")
print(json.dumps(RES, indent=1, default=str))
