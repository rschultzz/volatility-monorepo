#!/usr/bin/env python3
"""CR-BP Step 0 addendum — era split of the P3 result. READ-ONLY: SELECTs only, no run row, no API calls.

Why: the P3 ratios (scripts/output/cr_bp/step0.md item 4) are bimodal. This script splits them by era
(< 2024-03-01 vs ≥ 2024-03-01) and measures, for every PM expiry ≤ 15 DTE on every train day, the ratio of each
monies field at 09:31 / 09:33 / 09:35 ET to its own value at 09:50 ET, by calendar DTE at the open.
Inputs: scripts/output/cr_bp/vpx_{primary,secondary}.parquet (Step 0) + orats_monies_minute (SELECT).

Usage: apps/web/.venv/bin/python -u scripts/cr_bp_step0_era.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, time
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
from scripts.cr_bo_step0 import year_chunks

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bp"
ERA_CUT = pd.Timestamp("2024-03-01")
SQL = """
SELECT trade_date, expir_date, snapshot_pt, atmiv, vol50
FROM orats_monies_minute
WHERE ticker = 'SPX' AND expiry_tod = 'pm' AND trade_date BETWEEN %s AND %s AND dte BETWEEN 1 AND 16
  AND snapshot_pt::time IN ('06:31', '06:33', '06:35', '06:50')
"""


def main() -> None:
    L = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    say("# CR-BP Step 0 addendum — era split of P3\n")
    say(f"Era cut {ERA_CUT.date()} (first month in which the secondary-set vol50²τ / V_px at 09:33 has no low-mode days).\n")

    say("## A. P3 ratios at 09:33 ET by era and month (from Step 0 parquets)\n")
    for name in ("primary", "secondary"):
        d = pd.read_parquet(OUT_DIR / f"vpx_{name}.parquet"); d["dt"] = pd.to_datetime(d.d)
        d["era"] = np.where(d.dt < ERA_CUT, "2023-05 → 2024-02", "2024-03 → 2026-06")
        say(f"**{name}** — median [IQR] (n)\n")
        say("| era | atmiv²τ / V_px @09:33 | vol50²τ / V_px @09:33 | vol50²τ / V_px @09:45 | share of days with vol50²τ / V_px < 0.5 @09:33 |")
        say("|---|---|---|---|---|")
        for era, g in d.groupby("era"):
            f = lambda s: f"{s.median():.3f} [{s.quantile(.25):.3f}–{s.quantile(.75):.3f}] ({s.notna().sum()})"
            say(f"| {era} | {f(g['r_atm_06:33'])} | {f(g['r_v50_06:33'])} | {f(g['r_v50_06:45'])} | {100*(g['r_v50_06:33'] < 0.5).mean():.0f} % |")
        if name == "secondary":
            d["m"] = d.dt.dt.to_period("M").astype(str)
            low = d.groupby("m").apply(lambda g: (g["r_v50_06:33"] < 0.5).mean())
            say("\nShare of low-mode days by month: " + ", ".join(f"{k} {100*v:.0f}%" for k, v in low.items()) + "\n")
            tau = np.array([C.tau_years(datetime.combine(x, time(9, 33)), x) for x in d.d])
            d["vol_px"] = np.sqrt(d["vpx_06:33"] / tau); d["vol50"] = np.sqrt(d["r_v50_06:33"] * d["vpx_06:33"] / tau)
            say("Annualized vols at 09:33 ET (secondary set, medians): " + "; ".join(
                f"{era}: strangle-implied {g.vol_px.median():.3f}, vol50 {g.vol50.median():.3f}, leg-pair bid–ask {g['w_06:33'].median():.2f} pts"
                for era, g in d.groupby("era")) + ".\n")

    say("## B. Monies fields at the open relative to their own 09:50 ET value, by era and calendar DTE (all train days, PM, ≤ 15 DTE)\n")
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    m = pd.concat([pd.read_sql(SQL, conn, params=(lo.isoformat(), hi.isoformat())) for lo, hi in year_chunks(C.TRAIN_START, C.TRAIN_END)], ignore_index=True)
    m["td"] = pd.to_datetime(m.trade_date); m["ex"] = pd.to_datetime(m.expir_date)
    m["hm"] = m.snapshot_pt.dt.strftime("%H:%M"); m["cd"] = (m.ex - m.td).dt.days
    m["grp"] = pd.cut(m.cd, [-1, 0, 1, 3, 7, 15], labels=["0", "1", "2–3", "4–7", "8–15"])
    m["era"] = np.where(m.td < ERA_CUT, "2023-05 → 2024-02", "2024-03 → 2026-06")
    for col in ("vol50", "atmiv"):
        z = m.pivot_table(index=["td", "ex", "grp", "era"], columns="hm", values=col, observed=True).reset_index()
        for hm, et in (("06:31", "09:31"), ("06:33", "09:33"), ("06:35", "09:35")):
            z["r"] = z[hm] / z["06:50"]
            t = z.groupby(["era", "grp"], observed=True).r.median().unstack()
            say(f"**{col}({et}) / {col}(09:50)** — median by era × calendar DTE at the open\n")
            say("| era | " + " | ".join(t.columns.astype(str)) + " |")
            say("|---|" + "---|" * len(t.columns))
            for era, row in t.iterrows():
                say(f"| {era} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
            say("")
    (OUT_DIR / "step0_era.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_DIR / 'step0_era.md'}")


if __name__ == "__main__":
    main()
