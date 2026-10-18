#!/usr/bin/env python3
"""CR-BP Step 0 addendum A1 — era × bucket table (A1.5), per-cell decision (A1.3), P1 coverage. READ-ONLY, no DB access.

Inputs: scripts/output/cr_bp/{vpx_primary,vpx_secondary}.parquet and step0_pairs.csv (Step 0).
Era of a capture day d = era of its night t = prev NYSE session of d (E0: t < 2024-03-01, E1: t ≥ 2024-03-01; A1.1).
Only 0DTE contracts are captured, i.e. S1's open-side contract; S2 / S7 cells have no V_px (A1 reading r1).
Outputs: scripts/output/cr_bp/step0_a1.md, p3_decision_a1.json.

Usage: apps/web/.venv/bin/python -u scripts/cr_bp_step0_a1.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from packages.shared.trading_calendar import prev_trading_day

OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bp"
CUT = date(2024, 3, 1)
ALT_CUTS = (date(2024, 2, 1), date(2024, 4, 1))
MINUTES = [("06:31", "09:31"), ("06:33", "09:33"), ("06:35", "09:35"), ("06:45", "09:45")]
TOL = 0.10


def era(t: date, cut: date = CUT) -> str:
    return "E0" if t < cut else "E1"


def stat(s: pd.Series, root: bool = False) -> str:
    s = s.dropna()
    s = s[s > 0] if root else s
    if not len(s):
        return "— (n 0)"
    x = np.sqrt(s) if root else s
    return f"{x.median():.3f} [{x.quantile(.25):.3f}–{x.quantile(.75):.3f}] (n {len(x)})"


def main() -> None:
    L = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    sets = {}
    for name in ("primary", "secondary"):
        d = pd.read_parquet(OUT_DIR / f"vpx_{name}.parquet")
        d["d"] = pd.to_datetime(d.d).dt.date
        d["t"] = [prev_trading_day(x) for x in d.d]
        d["era"] = [era(t) for t in d.t]
        for c in ALT_CUTS:
            d[f"era_{c:%Y%m%d}"] = [era(t, c) for t in d.t]
        sets[name] = d
    U = pd.read_csv(OUT_DIR / "step0_pairs.csv")
    U = U[U.keep].copy()
    for c in ("t", "t1"):
        U[c] = pd.to_datetime(U[c]).dt.date
    U["era"] = [era(t) for t in U.t]

    say("# CR-BP Step 0 addendum A1 — era × bucket (A1.5), per-cell decision (A1.3)\n")
    say(f"Era by night t (E0: t < {CUT}, E1: t ≥ {CUT}; in-sample choice, A1.1). A capture day d is the open of night t = prev session(d). "
        "Ratios: monies field²τ (τ per CR-BO L3) / V_px; √ form = implied-vol ratio. Only 0DTE (S1's open-side contract) is captured; "
        "S2 / S7 rows are n 0 by construction (A1 reading r1). No 1–3 DTE captures exist, so the 15:59 rows are n 0.\n")

    say("## A1.5 — ratios by era, bucket, set and minute\n")
    say("| set | era | bucket | minute (ET) | atmiv²τ / V_px | √ (vol ratio) | vol50²τ / V_px | √ (vol ratio) |")
    say("|---|---|---|---|---|---|---|---|")
    for name, d in sets.items():
        for e in ("E0", "E1"):
            g = d[d.era == e]
            for bk in ("S1", "S2", "S7"):
                for pt, et in MINUTES + [("12:59", "15:59")]:
                    if bk != "S1" or pt == "12:59":
                        say(f"| {name} | {e} | {bk} | {et} | — (n 0) | — | — (n 0) | — |")
                        continue
                    a, v = g[f"r_atm_{pt}"], g[f"r_v50_{pt}"]
                    say(f"| {name} | {e} | {bk} | {et} | {stat(a)} | {stat(a, True)} | {stat(v)} | {stat(v, True)} |")
    say("")

    say("## A1.1 sensitivity (diagnostic column, not a selection) — S1, 09:33 ET medians under alternative cuts\n")
    say("| set | cut | E0 atmiv²τ/V_px | E0 vol50²τ/V_px | E1 atmiv²τ/V_px | E1 vol50²τ/V_px |")
    say("|---|---|---|---|---|---|")
    for name, d in sets.items():
        for c, col in [(CUT, "era")] + [(c, f"era_{c:%Y%m%d}") for c in ALT_CUTS]:
            cells = []
            for e in ("E0", "E1"):
                g = d[d[col] == e]
                cells += [stat(g["r_atm_06:33"]), stat(g["r_v50_06:33"])]
            say(f"| {name} | {c} | " + " | ".join(cells) + " |")
    say("")

    # A1.3 decision per era × bucket on the deciding set (C1: secondary, primary n < 100 in both eras)
    P, S = sets["primary"], sets["secondary"]
    say("## A1.3 — decision per era × bucket\n")
    n_prim = {e: int((P.era == e).sum()) for e in ("E0", "E1")}
    deciding = {e: ("primary" if n_prim[e] >= 100 else "secondary") for e in ("E0", "E1")}
    say(f"Deciding set per C1: E0 {deciding['E0']} (primary n {n_prim['E0']}), E1 {deciding['E1']} (primary n {n_prim['E1']}).\n")
    say("| era | bucket | atmiv²τ/V_px @09:33 | vol50²τ/V_px @09:33 | atmiv passes | vol50 passes | extension field |")
    say("|---|---|---|---|---|---|---|")
    cells = {}
    for e in ("E0", "E1"):
        D = sets[deciding[e]]; g = D[D.era == e]
        for bk in ("S1", "S2", "S7"):
            if bk == "S1":
                ra, rv = g["r_atm_06:33"].median(), g["r_v50_06:33"].median()
                pa, pv = abs(ra - 1) <= TOL, abs(rv - 1) <= TOL
            else:
                ra = rv = np.nan; pa = pv = False
            ext = "atmiv" if pa else ("vol50" if pv else None)
            cells[f"{e}_{bk}"] = dict(r_atm=None if np.isnan(ra) else float(ra), r_v50=None if np.isnan(rv) else float(rv), atm=bool(pa), v50=bool(pv), extension=ext)
            say(f"| {e} | {bk} | {'—' if np.isnan(ra) else f'{ra:.3f}'} | {'—' if np.isnan(rv) else f'{rv:.3f}'} | {pa} | {pv} | {ext or 'none → primary subset only'} |")
    say("")

    # coverage
    vpx_days = set(P[P["vpx_06:31"] > 0].d)
    say("## P1 coverage per era × bucket (CR-BO-universe nights)\n")
    say("| era | bucket | universe nights | v_b = V_px (primary, 09:31) | v_b = extension field²τ | dropped |")
    say("|---|---|---|---|---|---|")
    cov = {}
    for e in ("E0", "E1"):
        for bk in ("S1", "S2", "S7"):
            base = U[(U.era == e) & U[f"{bk}_ok"]]
            nv = int(sum(1 for t1 in base.t1 if t1 in vpx_days)) if bk == "S1" else 0
            ext = cells[f"{e}_{bk}"]["extension"]
            ne = (len(base) - nv) if ext else 0
            cov[f"{e}_{bk}"] = dict(universe=len(base), vpx=nv, extended=ne, dropped=len(base) - nv - ne)
            say(f"| {e} | {bk} | {len(base)} | {nv} | {ne} | {len(base) - nv - ne} |")
    say("")

    # A1.8
    say("## A1.8 — live-card implied_move_1d accuracy (S1, 09:33 ET, atmiv²τ / V_px, primary set = ATM straddle)\n")
    line = []
    for e in ("E1", "E0"):
        g = P[P.era == e]["r_atm_06:33"].dropna()
        line.append(f"{e} median {g.median():.3f} (√ {np.sqrt(g.median()):.3f}), n {len(g)}")
    gs = {e: S[S.era == e]["r_atm_06:33"].dropna() for e in ("E1", "E0")}
    say("**" + "; ".join(line) + ".** Secondary (±0.5 IM strangle, wing): "
        + "; ".join(f"{e} {g.median():.3f} (√ {np.sqrt(g.median()):.3f}), n {len(g)}" for e, g in gs.items()) + ".\n")

    # implied_move_1d uses atmiv × √(1/252), not CR-BO's calendar τ; restate the ratio on that formula's own clock
    from datetime import datetime, time
    from scripts import cr_bo_common as C
    say("**On `implied_move_1d`'s own formula** (`table_spot × atmiv × √(1/252)`, `day_features.py`): the variance it implies is atmiv² / 252, "
        "not atmiv²·τ_calendar. Ratio atmiv²·(1/252) / V_px at 09:33 ET (S1 contract), median [IQR] (n), with √:\n")
    say("| set | era | atmiv²/252 ÷ V_px | √ |")
    say("|---|---|---|---|")
    for name, d in sets.items():
        tau = np.array([C.tau_years(datetime.combine(x, time(9, 33)), x) for x in d.d])
        r = d["r_atm_06:33"] / tau * (1 / 252)
        for e in ("E1", "E0"):
            g = r[d.era == e]
            say(f"| {name} | {e} | {stat(g)} | {stat(g, True)} |")
    say("\nCalendar τ at 09:33 to 16:00 ET is 6.45 h / 8760 h ≈ 0.000736 yr vs 1/252 ≈ 0.00397 yr (×5.39). "
        "The A1.8 line above uses CR-BO's calendar τ as specified; this table is the formula-consistent reading for the live card. Both are reported; no cause is inferred.\n")

    (OUT_DIR / "step0_a1.md").write_text("\n".join(L) + "\n")
    (OUT_DIR / "p3_decision_a1.json").write_text(json.dumps(dict(cut=str(CUT), deciding=deciding, cells=cells, coverage=cov), indent=2))
    print(f"wrote {OUT_DIR / 'step0_a1.md'}")


if __name__ == "__main__":
    main()
