#!/usr/bin/env python3
"""CR-CI Step 0 — counts, placebo / random coverage after EDGE matching, A/A on the centre cell, fixture reproduction.
READ-ONLY. No outcome at a real level is computed (real trades get entries, EDGE tags and random-draw validity only).

Spec: specs/CR-CI-wide-stop-touch-trade.md (Step 0, C1–C10). Machinery: scripts/cr_ci_lib.py.
  - counts: CR-CE events (D > 0) -> setup_candidates confirm-bar trades (C1), per CR-CE cell, real and placebo. The trade
    set does not depend on the exit cell, so the count is the same in all 6 grid cells (and the IM readout).
  - C2: trades where the CR-CC builder's direction (1-min close 5 bars before the touch) disagrees with the CR-BW side.
  - coverage (C6, C7): share of real trades whose cell × EDGE stratum holds >= 5 placebo trades; share with >= 1 valid
    random draw. Stop for Ryan if either is < 80 % in any cell.
  - EDGE share real vs placebo, raw and after matching.
  - A/A (C9): placebo trades at S = 15 / T = 20, placebo levels split A / B per session by default_rng(20261026);
    A − B with B post-stratified to A; CI from 10 000 session-cluster resamples (seed 20260920). Stop if it excludes 0.
    Readout: 200 splits, 2 000 resamples each.
  - A1: opposite-direction trades share each real trade's entry, so their coverage is 100 % by construction.
  - CR_CI_BARS=databento (default, C3) | ironbeam (the setup_candidates series) selects the minute bars.
  - fixtures (L6): research/gallery/cc-fixture.json recomputed on the Databento minute bars: confirm-bar entry time and
    price vs the fixture row; heat (max adverse from entry before the +20 bar) and the +20 bar start (UTC); max
    favourable to 12:59 PT if +20 is never reached. These are the five published paths Ryan quoted.

Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_ci_step0.py | tee scripts/logs/cr_ci/step0.log
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_ci_lib as L  # noqa: E402
import numpy as np  # noqa: E402

OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_ci"
TAG = "" if L.BAR_SOURCE == "databento" else "_ironbeam"
TR, TP = L.TR, L.TP
RES: dict = dict(info=L.INFO)

# counts
RES["counts"] = {c: dict(real_events=int((L.R[(L.R["D"] > 0)]["cell"] == c).sum()), real_trades=int((TR["cell"] == c).sum()),
                         placebo_events=int((L.P[(L.P["D"] > 0)]["cell"] == c).sum()),
                         placebo_trades=int((TP["cell"] == c).sum()))
                 for c in L.CELLS}
RES["counts"]["pooled"] = dict(real_events=803, real_trades=int(len(TR)), placebo_events=int((L.P["D"] > 0).sum()),
                               placebo_trades=int(len(TP)))
RES["counts_first_touch"] = dict(real=int(TR["first"].sum()), placebo=int(TP["first"].sum()))
top3 = TR.groupby("d").size().sort_values(ascending=False)
RES["top3_session_share"] = dict(share=float(top3.iloc[:3].sum() / len(TR)), sessions=[[str(d), int(n)] for d, n in top3.iloc[:3].items()],
                                 sessions_with_trades=int(len(top3)))
RES["C2_direction_disagree"] = dict(n=int(((TR["cc_dir"] != 0) & (TR["cc_dir"] != TR["side"])).sum()),
                                    cc_undefined=int((TR["cc_dir"] == 0).sum()), of=int(len(TR)))
RES["entry_distance_from_level"] = dict(real=L.pts(np.abs(TR["entry"] - TR["L"])), placebo=L.pts(np.abs(TP["entry"] - TP["L"])))

# coverage + EDGE shares
draws = L.random_draws(TR)
TR["rnd_valid"] = [sum(r is not None for r in rs) if any(r is not None for r in rs) else np.nan for rs in draws]
cov, edge = {}, {}
for c in L.CELLS + ["pooled"]:
    er = TR if c == "pooled" else TR[TR["cell"] == c]
    epl = TP if c == "pooled" else TP[TP["cell"] == c]
    cov[c] = L.coverage(er, epl, "rnd_valid")
    edge[c] = L.edge_shares(er, epl)
RES["coverage"], RES["edge_share"] = cov, edge
RES["A1_opposite_coverage"] = dict(n=int(len(TR)), share=1.0, note="same entry as each real trade (C11); defined for every trade")
RES["strata_counts"] = {k: dict(real=int((TR["stratum"] == k).sum()), placebo=int((TP["stratum"] == k).sum()))
                        for k in L.STRATA}
RES["random_draws_valid"] = dict(total=int(sum(r is not None for rs in draws for r in rs)), of=int(5 * len(TR)))
STOP = [c for c in L.CELLS + ["pooled"] if cov[c]["placebo_covered"] < 0.8 or cov[c]["random_covered"] < 0.8]

# A/A on placebo, centre cell
CEN = L.cfg_name(L.CENTRE)
L.add_trades(TP, [CEN])
col = f"pnl_{CEN}"
levels = sorted({(d, x) for d, x in zip(TP["d"], TP["L"])})
rng = np.random.default_rng(L.SEED_AA)
W = L.weights(L.N_BOOT, L.SEED_BOOT)


def split(coins):
    m = dict(zip(levels, coins))
    a = np.array([m[(d, x)] for d, x in zip(TP["d"], TP["L"])])
    return TP[a == 0], TP[a == 1]


A, B = split(rng.integers(0, 2, len(levels)))
aa = L.matched_diff(A, B, col, W)
aa["A_covered"] = float(A["stratum"].isin(L.covered_strata(B)).mean())
RES["AA_primary"] = aa
W2 = L.weights(2000, L.SEED_BOOT)
sp = []
for _ in range(200):
    A2, B2 = split(rng.integers(0, 2, len(levels)))
    sp.append(L.matched_diff(A2, B2, col, W2))
sp = [x for x in sp if "lo" in x]
RES["AA_200"] = dict(n=len(sp), share_ci_excl_0=float(np.mean([x["lo"] > 0 or x["hi"] < 0 for x in sp])),
                     median_diff=float(np.median([x["est"] for x in sp])), diff_pcts=L.pts([x["est"] for x in sp]))
RES["placebo_centre_descriptive"] = dict(n=int(np.isfinite(TP[col]).sum()), mean=float(TP[col].mean()),
                                         kinds=TP[f"kind_{CEN}"].value_counts().to_dict())
if aa["lo"] > 0 or aa["hi"] < 0:
    STOP.append("A/A")

# fixtures
fx = json.loads(L.FIXTURE.read_text())["events"]
FX = []
for e in fx:
    d = datetime.fromisoformat(e["session_date"]).date()
    s = 1 if e["direction"] == "long" else -1
    t = int(datetime.fromisoformat(e["meta"]["touch_utc"]).replace(tzinfo=timezone.utc).timestamp())
    Lv = float(e["level_price"])
    row = dict(id=e["id"], d=str(d), direction=e["direction"], L=Lv, in_universe=d in L.USABLE)
    m = TR[(TR["d"] == d) & (TR["L"] == Lv) & (TR["t"] == t)]
    row["in_trade_set"] = bool(len(m))
    row["side_matches"] = bool(len(m) and int(m.iloc[0]["side"]) == s) if len(m) else None
    r = L.confirm(d, t, Lv, s) if d in L.BARS else "no bars"
    if isinstance(r, str):
        row["confirm"] = r
    else:
        j, entry = r
        B_ = L.BARS[d]
        ets = datetime.fromtimestamp(B_["t0"] + 60 * (j + 1), L.UTC).astimezone(L.PT).replace(tzinfo=None)
        row.update(entry_ts_pt=ets.isoformat(), entry=entry, fixture_entry_ts_pt=e["entry_ts_pt"],
                   fixture_entry=e["entry_price"], entry_matches=bool(ets.isoformat() == e["entry_ts_pt"]
                                                                       and abs(entry - e["entry_price"]) < 1e-9))
        h, l = B_["h"][j + 1:], B_["l"][j + 1:]
        fav = (h - entry) if s == 1 else (entry - l)
        adv = (entry - l) if s == 1 else (h - entry)
        k = np.nonzero(fav >= 20)[0]
        if len(k):
            k = int(k[0])
            row["heat"] = float(np.nanmax(adv[:k])) if k else 0.0
            row["plus20_utc"] = datetime.fromtimestamp(B_["t0"] + 60 * (j + 1 + k), L.UTC).strftime("%H:%MZ")
        else:
            row["heat"], row["plus20_utc"], row["max_fav"] = float(np.nanmax(adv)), None, float(np.nanmax(fav))
    FX.append(row)
RES["fixtures"] = FX
EXPECT = {1: (15.75, "19:00Z"), 2: (4.75, "17:05Z"), 3: (10.25, "16:09Z"), 4: (15.75, "18:18Z"), 5: (23.75, None)}
RES["fixtures_reproduce"] = all(abs(r.get("heat", -1) - EXPECT[r["id"]][0]) < 1e-9 and r.get("plus20_utc", "x") == EXPECT[r["id"]][1]
                                for r in FX) and abs(FX[4].get("max_fav", 0) - 16.5) < 1e-9
if not RES["fixtures_reproduce"]:
    STOP.append("fixtures")
RES["stop_for_review"] = STOP

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"step0{TAG}.json").write_text(json.dumps(RES, indent=1, default=str))
print(json.dumps({k: v for k, v in RES.items() if k != "info"}, indent=1, default=str))
print("funnel", json.dumps(L.INFO["trade_funnel"]), "missing minutes", L.INFO["bars"])
print("STOP FOR REVIEW:" if STOP else "Step 0 gates pass.", STOP)
