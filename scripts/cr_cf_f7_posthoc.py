#!/usr/bin/env python3
"""CR-CF post-hoc F7 diagnostic (Amendment A2, D1–D4) — POST-HOC, NOT A TEST. READ-ONLY; train only.

Spec: specs/CR-CF-which-levels-price-respects.md, Amendment A2 (decision rule recorded before this run).
Construction: scripts/cr_cf_lib.py, unchanged (as in the Stage 1 run). F7 only. No Holm, no p-values. The dropped
A1.1 placebo tests are recovered by calling the unchanged cr_cf_lib.run with family label "F7_nodrop", which
bypasses its `f == "F7"` drop; dropped = recovered tests whose (key, side) is not in the kept set.

Statistic: session-weighted mean per arm, diff = arm A − arm B, CI from one 10,000 × n session-multinomial weight
matrix (default_rng(20260920)), 95 % percentile — copied from scripts/cr_cf_stage1.py. A cell is flagged
"underpowered" if either arm has < 30 tests or < 15 sessions.

EDGE (D3): L >= running RTH high − 2 or L <= running RTH low + 2, running extremes of the 1-s flow from 06:30 PT
through the test second inclusive; else INTERIOR.

Decision (A2 ruling 3): CANDIDATE if D2 pooled lo > 0 AND D3 EDGE pooled (kept placebo) diff >= 4.11 with lo > 0;
else "placebo artifact likely".

Usage: apps/web/.venv/bin/python -u scripts/cr_cf_f7_posthoc.py | tee scripts/logs/cr_cf/f7_posthoc.log
"""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
signal.alarm(3600)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cr_cf_lib as C  # noqa: E402

LABEL = "post-hoc, not a test"
STAGE1_F7 = 8.22
N_BOOT, MIN_TESTS, MIN_SESS = 10_000, 30, 15
nS, USABLE, SIDX = C.nS, C.USABLE, C.SIDX
W = np.random.default_rng(C.SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)

rr, kept, allp = [], [], []
for d in USABLE:
    F = C.FLOW[d]
    n_ = len(F["t"])
    for lv in C.LEV[d]["F7"]:
        rr += C.run(d, "F7", lv, np.full(n_, lv["L"]), None, True)
    for lv in C.PLV[d]["F7"]:
        Lv = np.full(n_, lv["L"])
        kept += C.run(d, "F7", lv, Lv, (str(d), "F7", lv["L"]), True)
        allp += C.run(d, "F7_nodrop", lv, Lv, (str(d), "F7", lv["L"]), True)
R, K, A = pd.DataFrame(rr), pd.DataFrame(kept), pd.DataFrame(allp)
kk = set(zip(K["key"], K["side"]))
A["dropped"] = [(k, s) not in kk for k, s in zip(A["key"], A["side"])]
D = A[A["dropped"]].copy()
assert len(R) == 1228 and len(K) == 324 and len(D) == 282, (len(R), len(K), len(D))


def edge(E):
    out = []
    for d, t0, L in zip(E["d"], E["t"], E["L"]):
        F = C.FLOW[d]
        a, _ = C.rth_slice(F, d)
        j = int(np.searchsorted(F["t"], t0))
        hi, lo = F["h"][a:j + 1].max(), F["l"][a:j + 1].min()
        out.append(bool(L >= hi - C.BAND or L <= lo + C.BAND))
    return np.array(out, dtype=bool)


for E in (R, K, A, D):
    E["edge"] = edge(E)


def sess_means(E, m):
    s_ = np.zeros((nS, 2))
    np.add.at(s_, ([SIDX[x] for x in E["d"]], 0), E[m].to_numpy(float))
    np.add.at(s_, ([SIDX[x] for x in E["d"]], 1), 1)
    pres = s_[:, 1] > 0
    return np.where(pres, s_[:, 0] / np.maximum(s_[:, 1], 1), 0.0), pres.astype(float)


def compare(Ea, Eb, m="esc"):
    (ma, pa), (mb, pb) = sess_means(Ea, m), sess_means(Eb, m)
    out = dict(n_a=int(len(Ea)), n_b=int(len(Eb)), s_a=int(pa.sum()), s_b=int(pb.sum()), label=LABEL)
    out["underpowered"] = bool(min(out["n_a"], out["n_b"]) < MIN_TESTS or min(out["s_a"], out["s_b"]) < MIN_SESS)
    if pa.sum() == 0 or pb.sum() == 0:
        return out | dict(note="empty arm")
    with np.errstate(invalid="ignore", divide="ignore"):
        bd = (W @ (ma * pa)) / (W @ pa) - (W @ (mb * pb)) / (W @ pb)
    av, bv = (ma * pa).sum() / pa.sum(), (mb * pb).sum() / pb.sum()
    lo, hi = np.nanpercentile(bd, [2.5, 97.5])
    return out | dict(a=round(float(av), 2), b=round(float(bv), 2), diff=round(float(av - bv), 2),
                      lo=round(float(lo), 2), hi=round(float(hi), 2))


def sides(E):
    return {"pooled": E, "below": E[E["side"] == 1], "above": E[E["side"] == -1]}


def descr(E):
    v = E["esc"].to_numpy(float)
    return dict(n=int(len(v)), sessions=int(E["d"].nunique()), median=round(float(np.median(v)), 2),
                p25=round(float(np.percentile(v, 25)), 2), p75=round(float(np.percentile(v, 75)), 2),
                mean=round(float(v.mean()), 2), censored=int(E["censored"].sum()),
                at_cap_120=int((v >= C.ESC_CAP - 1e-9).sum()), label=LABEL)


def block(Rx, Kx, Ax):
    rs, ks, as_ = sides(Rx), sides(Kx), sides(Ax)
    return dict(
        D2_restored={c: compare(rs[c], as_[c]) for c in rs},
        stage1_kept_check={c: compare(rs[c], ks[c]) for c in rs},
        D3_edge_kept={c: compare(rs[c][rs[c]["edge"]], ks[c][ks[c]["edge"]]) for c in rs},
        D3_interior_kept={c: compare(rs[c][~rs[c]["edge"]], ks[c][~ks[c]["edge"]]) for c in rs},
        D3_edge_restored={c: compare(rs[c][rs[c]["edge"]], as_[c][as_[c]["edge"]]) for c in rs},
        D3_interior_restored={c: compare(rs[c][~rs[c]["edge"]], as_[c][~as_[c]["edge"]]) for c in rs},
        edge_share=dict(real=round(float(Rx["edge"].mean()), 3), placebo_kept=round(float(Kx["edge"].mean()), 3),
                        placebo_restored=round(float(Ax["edge"].mean()), 3),
                        placebo_dropped=round(float(Ax[Ax["dropped"]]["edge"].mean()), 3) if Ax["dropped"].any() else None),
    )


RES = dict(label=LABEL,
           D1=dict(dropped=descr(D), kept=descr(K), dropped_minus_kept={c: compare(sides(D)[c], sides(K)[c]) for c in ("pooled", "below", "above")}),
           all_tests=block(R, K, A),
           D4_first_side_only=block(R[R["first"]], K[K["first"]], A[A["first"]]))
b = RES["all_tests"]
d2, d3 = b["D2_restored"]["pooled"], b["D3_edge_kept"]["pooled"]
cand = bool(d2.get("lo", -1) > 0 and d3.get("diff", -1) >= 0.5 * STAGE1_F7 and d3.get("lo", -1) > 0)
RES["decision"] = dict(D2_pooled_lo=d2.get("lo"), D3_edge_pooled_diff=d3.get("diff"), D3_edge_pooled_lo=d3.get("lo"),
                       threshold=round(0.5 * STAGE1_F7, 2), D3_edge_underpowered=d3["underpowered"],
                       verdict="CANDIDATE" if cand else "placebo artifact likely")
C.OUT_DIR.mkdir(parents=True, exist_ok=True)
(C.OUT_DIR / "f7_posthoc.json").write_text(json.dumps(RES, indent=1, default=str))


def ln(k, v):
    if "note" in v:
        return f"  {k:28s} {v['note']} (n {v['n_a']}/{v['n_b']})"
    return (f"  {k:28s} tests {v['n_a']}/{v['n_b']} sess {v['s_a']}/{v['s_b']}  {v['a']:.2f} vs {v['b']:.2f}  "
            f"diff {v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" + ("  UNDERPOWERED" if v["underpowered"] else ""))


print(f"=== CR-CF F7 diagnostic — {LABEL.upper()} ===")
print("D1 dropped:", RES["D1"]["dropped"])
print("D1 kept:   ", RES["D1"]["kept"])
for c, v in RES["D1"]["dropped_minus_kept"].items():
    print(ln(f"D1 dropped−kept {c}", v))
for name, blk in (("ALL TESTS", RES["all_tests"]), ("D4 FIRST SIDE ONLY", RES["D4_first_side_only"])):
    print(f"\n{name}  edge share {blk['edge_share']}")
    for part in ("stage1_kept_check", "D2_restored", "D3_edge_kept", "D3_interior_kept", "D3_edge_restored",
                 "D3_interior_restored"):
        for c, v in blk[part].items():
            print(ln(f"{part} {c}", v))
print("\nDecision (A2 ruling 3):", RES["decision"])
