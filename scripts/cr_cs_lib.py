#!/usr/bin/env python3
"""CR-CS shared definitions — stop run at a range edge, fade the failed sweep. READ-ONLY, train only.

Spec: specs/CR-CS-range-stop-run.md (kickoff, C1-C15). Ranges are CR-CP's after A0 / A1 exactly (`cr_cp_lib.PRIMARY`,
`find_range`, one per approach leg, close box frozen at q). Imported by scripts/cr_cs_step0.py and scripts/cr_cs_run.py.

  break (C3): first bar b > q with H >= hi + size (mirror), start < 12:30 PT; each edge's first break only.
  burst (C4): mean 1-min volume over the sweep window b..x (x = extreme over [b, min(r, b + 4)]) >= vr x median
              1-min volume of r0..q.
  failure (C5): first r in [b, b + rwin - 1] closing at / inside the swept edge; none -> breakout (range disarmed).
  entry (C6): fade at c[r]; continuation = same price, other side. Dedupe (C8) on (session, r, side).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

assert os.environ.setdefault("CR_CP_PERIOD", "train") == "train"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cr_cp_lib as C  # noqa: E402

assert C.PERIOD == "train" and C.OUT.name == "cr_cp" and C.D_FROM.isoformat() == "2026-01-06"
assert C.CAP.isoformat() == "2026-06-05T21:00:00+00:00"

NB = C.NB
LAST_BREAK = 360                            # C2: break bar starts before 12:30 PT (06:30 + 360 min)
SWIN = 5                                    # C4: sweep window <= 5 bars
OUT = C.REPO_ROOT / "scripts" / "logs" / "cr_cs"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SP:
    size: float = 2.0                       # C3: break by >= size pts beyond the close edge
    vr: float = 2.0                         # C4: burst ratio
    rwin: int = 5                           # C5: re-entry bars [b, b + rwin - 1]


PRIMARY = SP()
VARIANTS = {
    "primary": PRIMARY,
    "size1": replace(PRIMARY, size=1.0), "size4": replace(PRIMARY, size=4.0),
    "vr15": replace(PRIMARY, vr=1.5), "vr3": replace(PRIMARY, vr=3.0),
    "reentry3": replace(PRIMARY, rwin=3), "reentry10": replace(PRIMARY, rwin=10),
}


def load():
    m1, im = C.load_minutes(), C.load_im()
    D = C.build(m1, im)
    assert len(D["SESS"]) == 101 and abs(D["MED_IM"] - 65.71) < 0.01, (len(D["SESS"]), D["MED_IM"])
    return m1, D


def ranges(D) -> list:
    """C1: CR-CP primary ranges, (d, range) in session / leg order."""
    out = []
    for d in D["SESS"]:
        B = D["BARS"][d]
        for lg in C.approach_legs(B["h"], B["l"], D["SW"][d]):
            rg = C.find_range(B, lg, C.PRIMARY)
            if rg is not None:
                out.append((d, rg))
    return out


def events(D, RG, sp: SP) -> pd.DataFrame:
    """Every edge break after q (C2-C5), one row per break, with burst / failure / breakout. No outcome."""
    rows = []
    for ri, (d, rg) in enumerate(RG):
        B = D["BARS"][d]
        H, L, Cl, V = B["h"], B["l"], B["c"], B["vol"]
        hi, lo, q, r0 = rg["hi"], rg["lo"], rg["q"], rg["r0"]
        med = float(np.median(V[r0:q + 1]))
        armed = {1: True, -1: True}
        k = q + 1
        while k < min(LAST_BREAK, NB) and (armed[1] or armed[-1]):
            up = armed[1] and H[k] >= hi + sp.size
            dn = armed[-1] and L[k] <= lo - sp.size
            if up and dn:
                rows.append(dict(ri=ri, d=d, q=q, r0=r0, hi=hi, lo=lo, dir=0, b=k, burst=False, kind="both edges"))
                break
            if not (up or dn):
                k += 1
                continue
            dr = 1 if up else -1
            armed[dr] = False
            b = k
            r = None
            for j in range(b, min(b + sp.rwin, NB)):
                if (Cl[j] <= hi) if dr == 1 else (Cl[j] >= lo):
                    r = j
                    break
            wend = min(b + SWIN - 1, r if r is not None else NB - 1, NB - 1)
            seg = H[b:wend + 1] if dr == 1 else L[b:wend + 1]
            x = b + int(np.argmax(seg) if dr == 1 else np.argmin(seg))
            ext = float(H[x] if dr == 1 else L[x])
            vmean = float(np.mean(V[b:x + 1]))
            vsum = float(np.sum(V[b:x + 1]))
            dshare = dr * float(np.sum(B["buy"][b:x + 1] - B["sell"][b:x + 1])) / vsum if vsum > 0 else 0.0
            burst = med > 0 and vmean >= sp.vr * med
            bo = b + sp.rwin - 1
            rows.append(dict(ri=ri, d=d, q=q, r0=r0, hi=hi, lo=lo, dir=dr, b=b, x=x, r=r, ext=ext,
                             size=float((H[b] - hi) if dr == 1 else (lo - L[b])), med_vol=med, vmean=vmean,
                             vratio=vmean / med if med > 0 else np.nan, dshare=dshare, burst=bool(burst),
                             kind="failure" if r is not None else "breakout",
                             bo_j=bo if (r is None and bo < NB - 1) else None))
            if r is None:                                     # C2: a breakout disarms the range
                break
            k = r + 1
    return pd.DataFrame(rows)


def trades(D, EV: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """C6 / C8: fades = burst failures, entry c[r], h = -dir; dedupe on (d, r, h). Returns (trades, n_dropped)."""
    if EV.empty:
        return EV, 0
    T = EV[(EV["kind"] == "failure") & EV["burst"]].copy()
    T["r"] = T["r"].astype(int)
    T["h"] = -T["dir"].astype(int)
    T["px"] = [float(D["BARS"][d]["c"][r]) for d, r in zip(T["d"], T["r"])]
    n0 = len(T)
    T = T.sort_values(["d", "q", "ri", "b"]).drop_duplicates(["d", "r", "h"], keep="first")
    T["edge"] = [C.edge_tag(D["BARS"][d], r, px, h, include_j=True)
                 for d, r, px, h in zip(T["d"], T["r"], T["px"], T["h"])]
    T["hour"] = (390 + T["r"]) // 60
    return T.reset_index(drop=True), n0 - len(T)


def tag_levels(T: pd.DataFrame, LV) -> pd.DataFrame:
    """C13: 'beyond' (a strong line in (edge, extreme]), else 'at edge' (strong line within 2 pts of the swept edge),
    else 'no strong line'; 'no levels' outside CR-CM's universe. Kind = put / call by the line's net sign."""
    pos, kind = [], []
    for r in T.itertuples():
        if r.d not in LV["STRONG"]:
            pos.append("no levels"), kind.append("no levels")
            continue
        E = r.hi if r.dir == 1 else r.lo
        st = LV["STRONG"][r.d]
        bey = [x for x in st if (E < x["L"] <= r.ext if r.dir == 1 else r.ext <= x["L"] < E)]
        at = [x for x in st if abs(x["L"] - E) <= 2.0]
        if bey:
            x = min(bey, key=lambda z: abs(z["L"] - E))
            pos.append("sweep beyond a strong line")
        elif at:
            x = min(at, key=lambda z: abs(z["L"] - E))
            pos.append("edge at a strong line")
        else:
            pos.append("no strong line"), kind.append("no strong line")
            continue
        kind.append("put" if x["net"] < 0 else "call")
    T = T.copy()
    T["lv_pos"], T["lv_kind"] = pos, kind
    return T
