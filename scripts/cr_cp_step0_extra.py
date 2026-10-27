#!/usr/bin/env python3
"""CR-CP Step 0 extra — how many ranges D1's "from the leg's end" costs when a new extreme is made inside a box.
Counts only, no outcome. Writes scripts/logs/cr_cp/step0_extra.json.

Diagnostic rule (not a definition): the leg-end anchor e moves only when a new extreme beats the anchor by > 2 pts
(D1's void tolerance), so marginal new highs inside a box don't restart the hold. Everything else is the primary.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

import cr_cp_lib as C


def find_range_tol(B, lg, p, tol=2.0):
    H, L = B["h"], B["l"]
    s, dr = lg["start_i"], lg["dir"]
    best_px, anchor_px, e = (-np.inf if dr == 1 else np.inf), None, None
    for k in range(s, min(lg["c_next"], C.NB)):
        x = H[k] if dr == 1 else L[k]
        if (dr == 1 and x > best_px) or (dr == -1 and x < best_px):
            best_px = float(x)
            if anchor_px is None or abs(best_px - anchor_px) > tol:
                anchor_px, e = best_px, k
        r0 = k - p.hold + 1
        if k < lg["c_s"] or e is None or not (e <= r0 <= e + p.start_win):
            continue
        hi, lo = float(np.max(H[r0:k + 1])), float(np.min(L[r0:k + 1]))
        if hi - lo <= float(p.width):
            return dict(r0=r0, q=k, hi=hi, lo=lo)
    return None


m1, im = C.load_minutes(), C.load_im()
D = C.build(m1, im)
p = C.PRIMARY
cnt = Counter()
for d in D["SESS"]:
    B = D["BARS"][d]
    for lg in C.approach_legs(B["h"], B["l"], C.swing_of(D, d, p)):
        a, b = C.find_range(B, lg, p), find_range_tol(B, lg, p)
        cnt["legs"] += 1
        cnt[f"primary {'yes' if a else 'no'} / tol2 {'yes' if b else 'no'}"] += 1
        if a and b and a["q"] != b["q"]:
            cnt["both, different qualifying bar"] += 1
out = dict(cnt)
print(out)
(C.OUT / "step0_extra.json").write_text(json.dumps(out, indent=1))
