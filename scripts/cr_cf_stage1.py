#!/usr/bin/env python3
"""CR-CF Stage 1 — screen on train (2026-01-06 .. 2026-06-05), single run. READ-ONLY; price paths only.

Spec: specs/CR-CF-which-levels-price-respects.md (L1–L12, Amendment A1; A1.6 is this run). Construction from
scripts/cr_cf_lib.py, unchanged since the Step 0 A1 run (commit b9414a0) whose A/A gate passed in all 24 Holm cells.
Committed before its single run; every statistic is written to JSON before anything is printed.

Statistic (copied from scripts/cr_cb_t0_ra_a17.py, A17.3): per (session, arm) mean of the measure; arm estimate =
mean over sessions with >= 1 test; diff = real − placebo. Bootstrap: one 10,000 × n session-multinomial weight matrix
(default_rng(20260920)) applied to both arms; 95 % percentile CI; one-sided p = share of resamples with diff <= 0
(floor 1e-4) — H1 real escapes more slowly. Holm step-down at one-sided α 0.025 over the 24 hypotheses of A1.3
(F1, F2, F3, F4, F5_100, F5_50, F5_25, F7 × pooled / below / above). Verdict per hypothesis: pass = Holm-rejected and
CI lower bound > 0; refuted = CI upper bound < 0; else inconclusive. A family is shortlisted (L9) if any of its
three hypotheses passes. F6 gets the same statistics with no gate.

Secondary (no gate): dwell (s within level ± max(3, 0.05·IM) over 60 min) and lvr (log path-length ratio, 5 min
after / before; NaN rows dropped). Readouts (no gate), primary only unless stated:
  - first side only (the L5 test): real first tests vs placebo first tests;
  - excluding IM < 25 sessions (weights zeroed);
  - L11 GEX overlap: real − placebo within overlap tests and within non-overlap tests; real overlap − real
    non-overlap; pooled sides;
  - L11 time of day: real − placebo before / after 10:00 PT; pooled sides.
Reference row (L3): Corrected GEX, CR-CB A17 numbers quoted, not re-run.

Usage: apps/web/.venv/bin/python -u scripts/cr_cf_stage1.py | tee scripts/logs/cr_cf/stage1.log
"""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
signal.alarm(3 * 3600)
import numpy as np  # noqa: E402

import cr_cf_lib as C  # noqa: E402

HOLM_FAMS = ["F1", "F2", "F3", "F4", "F5_100", "F5_50", "F5_25", "F7"]
READOUT_FAMS = ["F6"]
N_BOOT, ALPHA = 10_000, 0.025
R, P = C.build(measure_real=True)
nS, USABLE, SIDX = C.nS, C.USABLE, C.SIDX
W = np.random.default_rng(C.SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)
FLAG = np.array([C.IM[x] < C.IM_FLAG for x in USABLE])


def sess_means(E, m):
    E = E[np.isfinite(E[m].to_numpy(float))]
    s_ = np.zeros((nS, 2))
    np.add.at(s_, ([SIDX[x] for x in E["d"]], 0), E[m].to_numpy(float))
    np.add.at(s_, ([SIDX[x] for x in E["d"]], 1), 1)
    pres = s_[:, 1] > 0
    return np.where(pres, s_[:, 0] / np.maximum(s_[:, 1], 1), 0.0), pres.astype(float), int(len(E))


def compare(A, B, m, mask=None):
    keep = np.ones(nS) if mask is None else mask.astype(float)
    (ma, pa, na), (mb, pb, nb) = sess_means(A, m), sess_means(B, m)
    pa, pb = pa * keep, pb * keep
    if pa.sum() == 0 or pb.sum() == 0:
        return dict(n_a=na, n_b=nb, n_sessions_a=int(pa.sum()), n_sessions_b=int(pb.sum()), note="empty arm",
                    lo=np.nan, hi=np.nan, p_one_sided=1.0)
    Wk = W * keep
    with np.errstate(invalid="ignore", divide="ignore"):
        bd = (Wk @ (ma * pa)) / (Wk @ pa) - (Wk @ (mb * pb)) / (Wk @ pb)
    av, bv = (ma * pa).sum() / pa.sum(), (mb * pb).sum() / pb.sum()
    lo, hi = np.nanpercentile(bd, [2.5, 97.5])
    return dict(n_a=na, n_b=nb, n_sessions_a=int(pa.sum()), n_sessions_b=int(pb.sum()),
                a=round(float(av), 4), b=round(float(bv), 4), diff=round(float(av - bv), 4),
                lo=round(float(lo), 4), hi=round(float(hi), 4),
                p_one_sided=max(float(np.nanmean(bd <= 0)), 1.0 / N_BOOT))


def cells(f):
    Rf, Pf = R[R["fam"] == f], P[P["fam"] == f]
    return {"pooled": (Rf, Pf), "below": (Rf[Rf["side"] == 1], Pf[Pf["side"] == 1]),
            "above": (Rf[Rf["side"] == -1], Pf[Pf["side"] == -1])}


def holm(block):
    order = sorted(block, key=lambda k: block[k]["p_one_sided"])
    stop = False
    for i, k in enumerate(order):
        thr = ALPHA / (len(order) - i)
        rej = (not stop) and block[k]["p_one_sided"] <= thr
        stop = stop or not rej
        b = block[k]
        b.update(holm_threshold=thr, holm_rejected=bool(rej), passed=bool(rej and b["lo"] > 0),
                 verdict="pass" if (rej and b["lo"] > 0) else ("refuted" if b["hi"] < 0 else "inconclusive"))


PRIMARY = {f"{f}/{c}": compare(a, b, "esc") for f in HOLM_FAMS for c, (a, b) in cells(f).items()}
holm(PRIMARY)
SHORTLIST = {f: any(PRIMARY[f"{f}/{c}"]["passed"] for c in ("pooled", "below", "above")) for f in HOLM_FAMS}

ALLF = HOLM_FAMS + READOUT_FAMS
RES = dict(
    sessions=dict(n=nS, first=str(USABLE[0]), last=str(USABLE[-1])),
    tests=dict(real=len(R), placebo=len(P), real_censored=int(R["censored"].sum()),
               placebo_censored=int(P["censored"].sum()), dropped_placebo=C.DROPPED),
    primary=PRIMARY, shortlist=SHORTLIST, shortlisted=[f for f, v in SHORTLIST.items() if v],
    f6_readout={f"F6/{c}": compare(a, b, "esc") for c, (a, b) in cells("F6").items()},
    secondary=dict(dwell={f"{f}/{c}": compare(a, b, "dwell") for f in ALLF for c, (a, b) in cells(f).items()},
                   lvr={f"{f}/{c}": compare(a, b, "lvr") for f in ALLF for c, (a, b) in cells(f).items()}),
    readout_first_side_only={f"{f}/{c}": compare(a[a["first"]], b[b["first"]], "esc")
                             for f in ALLF for c, (a, b) in cells(f).items()},
    readout_excl_low_im=dict(sessions=[str(x) for x in USABLE if C.IM[x] < C.IM_FLAG],
                             primary={f"{f}/{c}": compare(a, b, "esc", ~FLAG)
                                      for f in ALLF for c, (a, b) in cells(f).items()}),
    readout_gex_overlap={f: dict(
        overlap=compare(cells(f)["pooled"][0][cells(f)["pooled"][0]["ovl"]], cells(f)["pooled"][1][cells(f)["pooled"][1]["ovl"]], "esc"),
        non_overlap=compare(cells(f)["pooled"][0][~cells(f)["pooled"][0]["ovl"]], cells(f)["pooled"][1][~cells(f)["pooled"][1]["ovl"]], "esc"),
        real_overlap_minus_real_non=compare(cells(f)["pooled"][0][cells(f)["pooled"][0]["ovl"]],
                                            cells(f)["pooled"][0][~cells(f)["pooled"][0]["ovl"]], "esc"))
        for f in ALLF},
    readout_time_of_day={f: dict(
        before_10=compare(cells(f)["pooled"][0][cells(f)["pooled"][0]["pre10"]], cells(f)["pooled"][1][cells(f)["pooled"][1]["pre10"]], "esc"),
        after_10=compare(cells(f)["pooled"][0][~cells(f)["pooled"][0]["pre10"]], cells(f)["pooled"][1][~cells(f)["pooled"][1]["pre10"]], "esc"))
        for f in ALLF},
    reference_corrected_gex="CR-CB A17 (R-a, 147 sessions, first test per level): pooled −5.98 [−13.35, +1.21] min, "
                            "every cell inconclusive; CR-CB Step 0 v3 (train, every armed test): pooled −4.7 [−11.9, +2.0]",
)
C.OUT_DIR.mkdir(parents=True, exist_ok=True)
(C.OUT_DIR / "stage1.json").write_text(json.dumps(RES, indent=1, default=str))


def line(k, v):
    if "note" in v:
        return f"  {k:16s} {v['note']} (n {v['n_a']}/{v['n_b']})"
    return (f"  {k:16s} sess {v['n_sessions_a']}/{v['n_sessions_b']} tests {v['n_a']}/{v['n_b']}  {v['a']:.2f} vs "
            f"{v['b']:.2f}  diff {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]  p {v['p_one_sided']:.4f}"
            + (f"  Holm {v['holm_threshold']:.4f} {v['verdict']}" if "verdict" in v else ""))


print("tests", RES["tests"])
print("\nPRIMARY time to escape 0.25·IM, session-weighted (min, real − placebo), Holm over 24:")
for k, v in PRIMARY.items():
    print(line(k, v))
print("shortlisted:", RES["shortlisted"])
for title, blk in (("F6 readout", RES["f6_readout"]), ("SECONDARY dwell (s)", RES["secondary"]["dwell"]),
                   ("SECONDARY lvr", RES["secondary"]["lvr"]), ("first side only", RES["readout_first_side_only"]),
                   ("excl IM < 25", RES["readout_excl_low_im"]["primary"])):
    print(f"\n{title}:")
    for k, v in blk.items():
        print(line(k, v))
print("\nGEX overlap (pooled):")
for f, v in RES["readout_gex_overlap"].items():
    for k2, v2 in v.items():
        print(line(f"{f} {k2}", v2))
print("\ntime of day (pooled):")
for f, v in RES["readout_time_of_day"].items():
    for k2, v2 in v.items():
        print(line(f"{f} {k2}", v2))
