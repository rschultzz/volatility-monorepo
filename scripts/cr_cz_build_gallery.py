#!/usr/bin/env python3
"""CR-CZ — build the blind review gallery from CR-CP's flow-called ranges. READ-ONLY (DB reads via cr_cp_lib), train only.

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_cz_build_gallery.py

Set (spec: specs/CR-CZ-review-cp-ranges.md): the 235 CR-CP A1-primary D2 calls on 2026 train, frozen in
scripts/logs/cr_cp/run_primary_trades.parquet. The schedule is rebuilt here from cr_cp_lib (bars, box, decision, fills)
and asserted equal to the frozen file, including its ±10 race results. Sample: 60 short calls + 60 long calls, seed
20261027, without replacement.

Card (L10 / CR-CC A1.2): entry_ts_pt = the close of the decision bar t, so the replay ends on bar t; direction = the
D2 side; entry_price = the D4 limit at the called edge (box high − 1 short, box low + 1 long), not a fill; no stop
(stop-free); extreme_price = the approach leg's running extreme at the qualifying bar; confirm_ts_pt = the close of the
qualifying bar q. `meta` holds only what is known at the close of t. `outcome` holds both sides' D4 fills and their ±10
and ±20 races (C9 / C10), hidden until reveal; pnl_pts / exit_* are the called side's ±10 race.

Writes scripts/logs/cr_cz/cr-cz-cp-ranges.json, git-ignored because it holds the outcomes (load with scripts/load_setup_candidates.py --run-cr CR-CZ).
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

import cr_cp_lib as C

SEED, N_PER_SIDE = 20261027, 60
TRAIN_END = date(2026, 6, 5)
CR_ID, SET_NAME = "CR-CZ", "cr-cz:cp-ranges"
FROZEN = C.OUT / "run_primary_trades.parquet"
GALLERY = C.REPO_ROOT / "scripts" / "logs" / "cr_cz" / "cr-cz-cp-ranges.json"   # git-ignored: holds the outcomes
KEY = ["d", "dir", "start_i", "r0", "q", "t", "side"]
SIDE = {1: "long", -1: "short"}


def ts(d, i, close=False) -> str:
    """Naive PT ISO of RTH bar i's start (or close)."""
    return (datetime.combine(d, time(6, 30)) + timedelta(minutes=int(i) + int(close))).isoformat()


def r2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def race_exit(B, j, px, h, X):
    """C10 with the resolving bar: (result, bar). Same logic as cr_cp_lib.race_from (fill-bar adverse, then CR-CL race)."""
    if (px - B["l"][j] if h == 1 else B["h"][j] - px) >= X:
        return "loss", j
    for k in range(j + 1, C.NB):
        fav = (B["h"][k] - px) if h == 1 else (px - B["l"][k])
        adv = (px - B["l"][k]) if h == 1 else (B["h"][k] - px)
        if fav >= X and adv >= X:
            return "tie", k
        if fav >= X:
            return "win", k
        if adv >= X:
            return "loss", k
    return "unresolved", None


def cpp(B, r0, t):
    """C12 contracts / pt of the in-range up and down bars over r0..t, and up ÷ down."""
    sl = slice(r0, t + 1)
    o, c, v = B["o"][sl], B["c"][sl], B["vol"][sl]
    up, dn = c > o, c < o
    cu = float(np.sum(v[up]) / np.sum(np.abs(c - o)[up])) if up.any() else None
    cd = float(np.sum(v[dn]) / np.sum(np.abs(c - o)[dn])) if dn.any() else None
    return cu, cd, (cu / cd if cu is not None and cd else None)


def side_outcome(B, d, status, j, px, h):
    out = dict(side=SIDE[h], fill_status=status)
    if status != "fill":
        return out
    j = int(j)
    out.update(fill_ts_pt=ts(d, j), fill_price=r2(px))
    for X in (10.0, 20.0):
        res, k = race_exit(B, j, px, h, X)
        assert X != 10.0 or res == C.race_from(B, j, px, h, X)
        r = dict(result=res)
        if res in ("win", "loss"):
            r.update(exit_ts_pt=ts(d, k), exit_price=r2(px + h * X * (1 if res == "win" else -1)))
        out[f"race_{int(X)}"] = r
    return out


def text(o):
    if o["fill_status"] != "fill":
        return f"{o['side']} {o['fill_status']}"
    return f"{o['side']} ±10 {o['race_10']['result']}, ±20 {o['race_20']['result']}"


def frozen_schedule():
    """The 235 CR-CP calls rebuilt from cr_cp_lib and asserted equal to the frozen file: (D, a, FZ)."""
    m1, im = C.load_minutes(), C.load_im()
    D = C.build(m1, im)
    assert len(D["SESS"]) == 101 and abs(D["MED_IM"] - 65.71) < 0.01
    S = C.schedule(D, C.PRIMARY)
    S = S[S["dstatus"] == "call"].reset_index(drop=True)

    F = pd.read_parquet(FROZEN)
    assert len(S) == len(F) == 235, (len(S), len(F))
    assert (S["side"] == -1).sum() == 108 and (S["side"] == 1).sum() == 127
    a, b = S.sort_values(KEY).reset_index(drop=True), F.sort_values(KEY).reset_index(drop=True)
    assert (a[KEY].astype(str).values == b[KEY].astype(str).values).all()
    for c in ("hi", "lo", "ext_px", "leg_pts", "share", "prog"):
        assert np.allclose(a[c].astype(float), b[c].astype(float)), c
    for nm in ("ch", "op"):
        assert (a[f"{nm}_status"].values == b[f"{nm}_status"].values).all(), nm
        assert np.allclose(a[f"{nm}_px"].astype(float), b[f"{nm}_px"].astype(float), equal_nan=True), nm
    assert max(a["d"]) <= TRAIN_END                                     # B4

    FZ = {tuple(str(x[k]) for k in KEY): x for x in b.to_dict("records")}
    assert len(FZ) == len(b)
    return D, a, FZ


def sample_index(a):
    """CR-CZ's sample: 60 short calls, then 60 long calls, seed 20261027, without replacement (index labels of `a`)."""
    rng = np.random.default_rng(SEED)
    pick = []
    for sd in (-1, 1):                                                  # short calls, then long calls
        idx = a.index[a["side"] == sd].to_numpy()
        pick += list(rng.choice(idx, size=N_PER_SIDE, replace=False))
    return pick


def card(n, r, D, FZ):
    """One review card (event) for schedule row `r`, numbered n."""
    d, B, h = r.d, D["BARS"][r.d], int(r.side)
    fz = FZ[tuple(str(getattr(r, k)) for k in KEY)]
    ch = side_outcome(B, d, r.ch_status, r.ch_j, r.ch_px, h)
    op = side_outcome(B, d, r.op_status, r.op_j, r.op_px, -h)
    for o, frozen in ((ch, fz["ch_res"]), (op, fz["op_res"])):
        assert (o.get("race_10") or {}).get("result") == frozen         # frozen ±10 results reproduced
    lg = next(x for x in C.approach_legs(B["h"], B["l"], r.swing) if x["start_i"] == r.start_i and x["dir"] == r.dir)
    cu, cd, ratio = cpp(B, int(r.r0), int(r.t))
    limit = r.hi - 1.0 if h == -1 else r.lo + 1.0
    r10 = ch.get("race_10") or {}
    return dict(
        id=n,
        session_date=str(d),
        cell=f"{SIDE[h]} call",
        direction=SIDE[h],
        entry_ts_pt=ts(d, r.t, close=True),
        entry_price=r2(limit),
        extreme_price=r2(r.ext_px),
        confirm_ts_pt=ts(d, r.q, close=True),
        meta=dict(
            cell=f"{SIDE[h]} call",
            range=dict(top=r2(r.hi), bottom=r2(r.lo), height=r2(r.width), formed_ts_pt=ts(d, r.r0),
                       qualified_ts_pt=ts(d, r.q, close=True)),
            zone=dict(lo=r2(r.lo), hi=r2(r.hi)),
            box_basis="highest / lowest 1-min close over range start .. qualifying bar (CR-CP A1.1)",
            approach="up" if r.dir == 1 else "down",
            approach_start_ts_pt=ts(d, r.start_i),
            approach_start_px=r2(lg["start_px"]),
            approach_confirmed_ts_pt=ts(d, r.c_s, close=True),
            approach_anchor_ts_pt=ts(d, r.e),
            approach_extreme_px=r2(r.ext_px),
            approach_pts=r2(r.leg_pts),
            swing_pts=r2(r.swing),
            range_in_session=int(r.n_in_session),
            delta_share_pct=r2(100 * r.share),
            progress_pts=r2(r.prog),
            cpp_up=r2(cu), cpp_down=r2(cd), cpp_ratio=r2(ratio),
            entry_rule="D4: limit at the called edge (short box high − 1, long box low + 1) after the decision bar",
        ),
        outcome=dict(
            pnl_pts=(10.0 if r10.get("result") == "win" else -10.0 if r10.get("result") == "loss" else None),
            result=f"called {text(ch)} · opposite {text(op)}",
            exit_ts_pt=r10.get("exit_ts_pt"),
            exit_price=r10.get("exit_price"),
            called=ch,
            opposite=op,
            basis="CR-CP C9 / C10: stop-free races from the D4 edge fill to 12:59 PT; a direction test, not Ryan's exit (B6)",
        ),
    )


def main():
    D, a, FZ = frozen_schedule()
    P = a.loc[sample_index(a)].sort_values(["d", "t"]).reset_index(drop=True)
    events = [card(n, r, D, FZ) for n, r in enumerate(P.itertuples(), start=1)]

    GALLERY.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(cr_id=CR_ID, set_name=SET_NAME, level_frame="es", generated_at=datetime.now(timezone.utc).isoformat(),
               source="CR-CP A1 primary D2 calls (scripts/logs/cr_cp/run_primary_trades.parquet), Databento 1-min bars",
               sample=dict(seed=SEED, n_short=N_PER_SIDE, n_long=N_PER_SIDE, population_short=108, population_long=127),
               events=events)
    GALLERY.write_text(json.dumps(doc, indent=1))
    n_s = sum(e["direction"] == "short" for e in events)
    print(f"wrote {GALLERY}: {len(events)} events (short {n_s}, long {len(events) - n_s}), "
          f"{len({e['session_date'] for e in events})} sessions, {events[0]['session_date']} .. {events[-1]['session_date']}")


if __name__ == "__main__":
    main()
