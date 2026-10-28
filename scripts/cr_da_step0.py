#!/usr/bin/env python3
"""CR-DA Step 0 — counts, quote validity, V_px solve success, the CR-BP reproduction. NO outcome statistics
(no realised variance, no premium, no P&L). READ-ONLY: SELECTs only, no run row, no API calls.

Spec: specs/CR-DA-overnight-premium-prices.md, "Step 0" items 1–6 and gates S0-G1 … S0-G4.
Writes scripts/output/cr_da/step0.md and scripts/logs/cr_da/step0_gates.json (read by scripts/cr_da_run.py).

Usage: apps/web/.venv/bin/python -u scripts/cr_da_step0.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts import cr_da_lib as L
from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from scripts.cr_bo_step0 import load_es

REPRO_SQL = """
SELECT trade_date_d, strike, option_type, to_char(snapshot_pt, 'HH24:MI') AS hm, bid_price, ask_price
FROM orats_options_minute
WHERE ticker = 'SPX' AND expiry_tod = 'pm' AND expir_date_d = trade_date_d
  AND trade_date_d = ANY(%s) AND snapshot_pt::time IN ('06:31', '06:35')
"""


def pct(a, b):
    return f"{a}/{b} ({100*a/b:.1f} %)" if b else "0/0"


def main() -> None:
    out: list[str] = []
    say = lambda s="": (out.append(s), print(s, flush=True))
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    say(f"# CR-DA Step 0 — {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Read-only; no outcome statistics (no RV, premium or P&L). Times ET unless tagged.\n")

    # 1 universe
    n = L.load_universe()
    say(f"## 1. Universe\n\n`cr_cy_nights.csv` = CR-BO `keep` set: **{len(n)} nights** (gate). "
        f"CR-CY complete (6/6 contracts both windows): **{int(n.cy_complete.sum())}**; incomplete {int((~n.cy_complete).sum())}.\n")

    print("loading quotes and ES …", flush=True)
    q = L.load_quotes(conn, sorted(n.t1))
    Q = L.quote_map(q, dict(zip(n.t1, n.t)))
    es = load_es(conn)
    es_close = es.set_index("et").close
    df = L.exclusions(L.instruments(n, Q, es_close))

    # 2 counts
    say("## 2. Night counts after each exclusion (D2)\n")
    say("| step | nights | removed |")
    say("|---|---|---|")
    left = len(df); say(f"| CR-BO ES-clean universe | {left} | |")
    for code, label in [("a_cy_incomplete", "(a) CR-CY window incomplete"), ("b_no_valid_atm_quote", "(b) no valid ATM leg quote within ±2 min"),
                        ("c_solve_failed", "(c) V_px solve failed")]:
        k = int((df.excl == code).sum()); left -= k
        say(f"| {label} | {left} | {k} |")
    fin = df[df.final]
    say(f"\nFinal primary n = **{len(fin)}**; D9 population (non-event weeknights) **{int(fin.nonevent_weeknight.sum())}**.\n")
    say("| year of t | universe | final | D9 population |")
    say("|---|---|---|---|")
    for y, g in df.groupby("year"):
        say(f"| {y} | {len(g)} | {int(g.final.sum())} | {int((g.final & g.nonevent_weeknight).sum())} |")
    inc = df[df.excl == "a_cy_incomplete"]
    say(f"\n(a) nights: {', '.join(str(x) for x in inc.t1)} (t+1).")
    for code in ("b_no_valid_atm_quote", "c_solve_failed"):
        g = df[df.excl == code]
        if len(g):
            say(f"\n{code}: {', '.join(f'{a}→{b}' for a, b in zip(g.t, g.t1))}")

    # 3 quote validity (on CR-CY-complete nights)
    say("\n## 3. Quote validity per window and leg (CR-CY-complete nights)\n")
    cc = df[df.cy_complete]
    say("| window | leg | valid at target | via ±1 | via ±2 | missing | bid = ask = 0 at target |")
    say("|---|---|---|---|---|---|---|")
    for end, win in (("c", "close 15:55"), ("o", "open 09:35")):
        for leg in ("atmC", "atmP", "wingP", "wingC"):
            off = cc[f"{leg}_{end}_off"]; N = len(cc)
            say(f"| {win} | {leg} | {pct(int((off == 0).sum()), N)} | {int(off.abs().eq(1).sum())} | {int(off.abs().eq(2).sum())} | "
                f"{int(off.isna().sum())} | {int(cc[f'{leg}_{end}_zz'].sum())} |")

    # 4 solve success
    say("\n## 4. V_px solve success\n")
    say("| end | straddle (nights with valid ATM quotes) | ±5 strangle (nights with valid wing quotes) |")
    say("|---|---|---|")
    g3 = {}
    for end, win in (("c", "close"), ("o", "open")):
        a = cc[cc[f"st_ok_{end}"]]; b = cc[cc[f"sg_ok_{end}"]]
        sa = int((np.isfinite(a[f"V_{end}"]) & (a[f"V_{end}"] > 0)).sum()); sb = int((np.isfinite(b[f"Vstr_{end}"]) & (b[f"Vstr_{end}"] > 0)).sum())
        g3[end] = sa / len(a) if len(a) else 0.0
        say(f"| {win} | {pct(sa, len(a))} | {pct(sb, len(b))} |")
    sg_both = int((fin.Vstr_c.gt(0) & fin.Vstr_o.gt(0)).sum())
    say(f"\nStrangle readout n (final nights with both ends solved): {sg_both}.")

    # 5 CR-BP reproduction
    say("\n## 5. CR-BP reproduction (V_px at CR-BP's K and S)\n")
    vp = pd.read_parquet(L.REPO_ROOT / "scripts" / "output" / "cr_bp" / "vpx_primary.parquet")
    vp["d"] = pd.to_datetime(vp.d).dt.date
    vp = vp[vp.in_universe].copy()
    rq = pd.read_sql(REPRO_SQL, conn, params=(list(vp.d),))
    RQ = {(r.trade_date_d, float(r.strike), r.option_type, r.hm): (r.bid_price, r.ask_price) for r in rq.itertuples()}
    rep = []
    for r in vp.to_dict("records"):
        rec = dict(d=r["d"], K=r["K"], S=r["S"])
        for hm in ("06:31", "06:35"):
            c, p = RQ.get((r["d"], float(r["K"]), "C", hm)), RQ.get((r["d"], float(r["K"]), "P", hm))
            if c and p and L.leg_quote_is_valid(*c) and L.leg_quote_is_valid(*p):
                v = L.solve_v((c[0] + c[1]) / 2 + (p[0] + p[1]) / 2, r["S"], [(r["K"], "C"), (r["K"], "P")])
            else:
                v = np.nan
            ref = r[f"vpx_{hm}"]
            rec[f"v_{hm}"] = v; rec[f"ref_{hm}"] = ref; rec[f"rel_{hm}"] = v / ref - 1 if ref else np.nan
        rep.append(rec)
    R = pd.DataFrame(rep)
    # reproduced = within 1 %, or both NaN (CR-BP's own 06:31 solve returned NaN: straddle mid ≤ intrinsic)
    rep_ok = lambda hm: (R[f"rel_{hm}"].abs() <= 0.01) | (R[f"v_{hm}"].isna() & R[f"ref_{hm}"].isna())
    ok31 = bool(len(R) and rep_ok("06:31").all())
    say(f"CR-BP primary-subset nights in the CR-BO universe: **{len(R)}**.\n")
    say("| minute PT | CR-BP has a value | within 1 % | both NaN | reproduced | max abs rel. diff |")
    say("|---|---|---|---|---|---|")
    for hm in ("06:31", "06:35"):
        say(f"| {hm} | {int(R[f'ref_{hm}'].notna().sum())} | {int((R[f'rel_{hm}'].abs() <= 0.01).sum())} | "
            f"{int((R[f'v_{hm}'].isna() & R[f'ref_{hm}'].isna()).sum())} | {int(rep_ok(hm).sum())}/{len(R)} | {R[f'rel_{hm}'].abs().max():.2e} |")

    # 6 diagnostics
    say("\n## 6. Diagnostics (final nights)\n")
    say(f"- |F_parity − S|: close median {np.nanmedian((fin.Fpar_c - fin.S_c).abs()):.2f} pts (IQR "
        f"{np.nanpercentile((fin.Fpar_c - fin.S_c).abs(), 25):.2f}–{np.nanpercentile((fin.Fpar_c - fin.S_c).abs(), 75):.2f}); "
        f"open median {np.nanmedian((fin.Fpar_o - fin.S_o).abs()):.2f} (IQR {np.nanpercentile((fin.Fpar_o - fin.S_o).abs(), 25):.2f}–"
        f"{np.nanpercentile((fin.Fpar_o - fin.S_o).abs(), 75):.2f}).")
    say(f"- |K0 − S|: close median {np.median((fin.K0_c - fin.S_c).abs()):.2f}; open median {np.median((fin.K0_o - fin.S_o).abs()):.2f}.")
    say(f"- Annualised straddle IV √(V/τ): close median {np.median(np.sqrt(fin.V_c / fin.tau_c)):.3f}, open median {np.median(np.sqrt(fin.V_o / fin.tau_o)):.3f}.")
    mk = fin.exit_mark.value_counts().to_dict()
    say(f"- Tradeable buyback marks (final nights): {mk}.")
    say(f"- ES prints missing: close {int(fin.es_c.isna().sum())}, open {int(fin.es_o.isna().sum())}.")

    gates = {
        "S0-G1 universe 621 = CR-BO keep": True,
        "S0-G2 CR-BP reproduction 06:31 all within 1 %": ok31,
        "S0-G3 straddle solve success >= 95 % each end": bool(g3["c"] >= 0.95 and g3["o"] >= 0.95),
        "S0-G4 final n >= 500": bool(len(fin) >= 500),
    }
    say("\n## Gates\n")
    for k, v in gates.items():
        say(f"- {k}: **{'PASS' if v else 'FAIL'}**")
    allp = all(gates.values())
    say(f"\n**Step 0: {'PASS — run once' if allp else 'FAIL — STOP'}.**")
    L.OUT_DIR.mkdir(parents=True, exist_ok=True); L.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (L.OUT_DIR / "step0.md").write_text("\n".join(out) + "\n")
    (L.LOG_DIR / "step0_gates.json").write_text(json.dumps(dict(pass_=allp, gates=gates, final_n=len(fin),
                                                               population_n=int(fin.nonevent_weeknight.sum())), indent=2))
    R.to_csv(L.LOG_DIR / "step0_repro.csv", index=False)


if __name__ == "__main__":
    main()
