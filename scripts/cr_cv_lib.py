#!/usr/bin/env python3
"""CR-CV shared definitions — today's setups at scalping scale, after costs. READ-ONLY, train only.

Spec: specs/CR-CV-scalping-scale.md (kickoff, C1-C14). Imported by scripts/cr_cv_step0.py and scripts/cr_cv_run.py so
both use one definition of every quantity. Importing computes no outcome.

  events (C1):  rebuilt through each CR's own library, unchanged — S1 cr_cl_lib (G1-down), S2 cr_cp_lib (schedule),
                S3 cr_cs_lib (fades), S4 cr_ct_lib (FAST / SLOW first touches, frozen cuts).
  1-s (C2):     es_flow_1s_databento o/h/l/c, each setup's front symbol, 06:30:00 .. 12:59:59 PT; every read guarded to
                [2026-01-01, 2026-06-05 21:00 UTC); cr_cp_lib's read-only connection (role + read-only asserted) is
                the only connection, including the CR-CJ loader's.
  fills (C3, C5), race (C4), statistic (C6, C7).
Second index = seconds since 06:30:00 PT of the session.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

assert os.environ.setdefault("CR_CP_PERIOD", "train") == "train"
assert os.environ.setdefault("CR_CL_PERIOD", "y26") == "y26"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_cp_lib as C  # noqa: E402  (loads .env and the repo path)
import cr_cj_lib as CJ  # noqa: E402

CJ.conn = C.conn                            # C2: one read-only connection for every loader
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TICK = 0.25
NB = C.NB
NSEC = 60 * NB                              # 13:00:00 PT
SIZES, REF = (2.0, 4.0), 10.0
COST, COSTS_READ = 0.50, (0.25, 0.75)
SEED, N_BOOT, MIN_N, ALPHA, M_HOLM = 20260920, 10_000, 40, 0.05, 16
FLOOR, CAP = C.FLOOR, C.CAP
assert str(FLOOR) == "2026-01-01 00:00:00+00:00" and str(CAP) == "2026-06-05 21:00:00+00:00"
OUT = C.REPO_ROOT / "scripts" / "logs" / "cr_cv"
OUT.mkdir(parents=True, exist_ok=True)
REF_DIR = OUT / "ref"


# ------------------------------------------------------------------------------------------------ 1-s data (C2)
def load_1s(pairs: list[tuple]) -> pd.DataFrame:
    """(d, symbol) pairs -> 1-s o/h/l/c rows, 06:30 .. 13:00 PT (cached in s1_bars.parquet)."""
    path = OUT / "s1_bars.parquet"
    if path.exists():
        s = pd.read_parquet(path)
        have = set(zip(s["d"], s["sym"]))
        assert set(pairs) <= have, sorted(set(pairs) - have)[:5]
        return s
    parts = []
    for d, sym in sorted(set(pairs)):
        a, b = C.pt(d, 6, 30), C.pt(d, 13, 0)
        assert FLOOR <= a < b <= CAP, (a, b)
        r = C.q("""SELECT ts_utc, open::float AS o, high::float AS h, low::float AS l, close::float AS c
                   FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s ORDER BY ts_utc""",
                (sym, a, b))
        assert r.empty or (r["ts_utc"].min() >= a and r["ts_utc"].max() < b)
        r["d"], r["sym"] = d, sym
        parts.append(r)
    s = pd.concat(parts, ignore_index=True)
    s.to_parquet(path)
    return s


def sec_arrays(s: pd.DataFrame) -> dict:
    out = {}
    for (d, sym), g in s.groupby(["d", "sym"]):
        t = pd.to_datetime(g["ts_utc"], utc=True)
        sec = (t - pd.Timestamp(C.pt(d, 6, 30))).dt.total_seconds().to_numpy().astype(np.int64)
        assert (np.diff(sec) > 0).all() and sec.min() >= 0 and sec.max() < NSEC
        out[(d, sym)] = dict(sec=sec, **{k: g[k].to_numpy(float) for k in "ohlc"})
    return out


def minute_agg(S: dict) -> dict:
    """1-s -> (NB,) minute o (first) / h (max) / l (min) / c (last); NaN where a minute has no trade."""
    m = S["sec"] // 60
    out = {k: np.full(NB, np.nan) for k in "ohlc"}
    first = np.r_[True, m[1:] != m[:-1]]
    last = np.r_[m[1:] != m[:-1], True]
    out["o"][m[first]] = S["o"][first]
    out["c"][m[last]] = S["c"][last]
    out["h"][:] = np.nan
    hh = np.full(NB, -np.inf)
    ll = np.full(NB, np.inf)
    np.maximum.at(hh, m, S["h"])
    np.minimum.at(ll, m, S["l"])
    have = np.isfinite(out["o"])
    out["h"][have], out["l"][have] = hh[have], ll[have]
    return out


def minute_slice(S: dict, j: int) -> tuple[int, int]:
    return int(np.searchsorted(S["sec"], 60 * j)), int(np.searchsorted(S["sec"], 60 * (j + 1)))


# ------------------------------------------------------------------------------------------------ race (C4)
def race(S: dict, k0: int, fill: float, h: int, T: float, entry_k: int | None = None) -> str:
    """Seconds k0 .. end: 'win' / 'loss' / 'tie' / 'unresolved'. entry_k: the entry second's adverse extreme beyond the
    fill >= T is a loss (its favourable side not credited) — the primary model's entry-second rule (C4)."""
    if entry_k is not None:
        adv0 = (fill - S["l"][entry_k]) if h == 1 else (S["h"][entry_k] - fill)
        if adv0 >= T:
            return "loss"
    hh, ll = S["h"][k0:], S["l"][k0:]
    fav = (hh - fill) if h == 1 else (fill - ll)
    adv = (fill - ll) if h == 1 else (hh - fill)
    fw, aw = fav >= T, adv >= T
    i_f = int(np.argmax(fw)) if fw.any() else None
    i_a = int(np.argmax(aw)) if aw.any() else None
    if i_f is None and i_a is None:
        return "unresolved"
    if i_a is None or (i_f is not None and i_f < i_a):
        return "win"
    if i_f is None or i_a < i_f:
        return "loss"
    return "tie"


# ------------------------------------------------------------------------------------------------ fills (C3, C5)
def market_fill(S: dict, m: int, h: int, slip: float = TICK):
    """C3: next trade after the close of minute m, at its open + slip adverse. -> (entry second index, fill) or None."""
    k = int(np.searchsorted(S["sec"], 60 * (m + 1)))
    if k >= len(S["sec"]):
        return None
    return k, float(S["o"][k] + h * slip)


def bar_close_start(S: dict, m: int):
    """C5 touch-fill for bar-close entries: race from the first second of minute m + 1 (index) or None."""
    k = int(np.searchsorted(S["sec"], 60 * (m + 1)))
    return None if k >= len(S["sec"]) else k


def edge_fill(B: dict, S: dict, hi: float, lo: float, t: int, h: int, through: float, p=C.PRIMARY):
    """S2: CR-CP `fill` (close box, void checked first in each minute), but a minute fills only if one of its seconds
    reaches the limit by `through` (TICK = trade-through, 0 = touch). -> (status, minute, second index, price)."""
    assert p.box == "close" and p.entry == "edge"
    px = lo + p.touch if h == 1 else hi - p.touch
    for j in range(t + 1, NB):
        far_break = (B["c"][j] > hi + p.void) if h == 1 else (B["c"][j] < lo - p.void)
        if far_break:
            return "no fill: box broke", j, None, None
        a, b = minute_slice(S, j)
        seg = (S["l"][a:b] <= px - through) if h == 1 else (S["h"][a:b] >= px + through)
        if seg.any():
            return "fill", j, a + int(np.argmax(seg)), float(px)
    return "no fill: 13:00", None, None, None


def level_fill(S: dict, j: int, L: float, b: int, h: int, through: float):
    """S4: within the touch minute j, the first second reaching L by `through` in the break direction b. Fade side
    (h = -b) fills at L; break side (h = b) at L + b*through (stop one tick through; C3). -> (second, fill) or None."""
    a, e = minute_slice(S, j)
    seg = (S["h"][a:e] >= L + through) if b == 1 else (S["l"][a:e] <= L - through)
    if not seg.any():
        return None
    k = a + int(np.argmax(seg))
    return k, float(L if h == -b else L + b * through)


# ------------------------------------------------------------------------------------------------ statistic (C6, C7)
def expectancy(p, T, c):
    return (2.0 * np.asarray(p) - 1.0) * T - c


def break_even(T, c):
    return (1.0 + c / T) / 2.0


def boot_weights(n_sessions: int, n_draws: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_sessions, size=(n_draws, n_sessions))
    Wt = np.zeros((n_draws, n_sessions))
    np.add.at(Wt, (np.repeat(np.arange(n_draws), n_sessions), idx.ravel()), 1)
    return Wt


def sess_counts(rows, SI: dict):
    """rows: (d, outcome) -> per-session resolved n and wins."""
    n, w = np.zeros(len(SI)), np.zeros(len(SI))
    for d, x in rows:
        if x in ("win", "loss"):
            n[SI[d]] += 1
            w[SI[d]] += x == "win"
    return n, w


def boot_p(Wt, n, w):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Wt @ w) / (Wt @ n)


def two_sided_p(E_draws: np.ndarray) -> float:
    e = E_draws[np.isfinite(E_draws)]
    return float(min(1.0, 2.0 * min((e <= 0).mean(), (e >= 0).mean())))


def pct_ci(x: np.ndarray, level: float) -> list:
    x = x[np.isfinite(x)]
    a = (1.0 - level) / 2.0
    return [float(np.percentile(x, 100 * a)), float(np.percentile(x, 100 * (1 - a)))]


def holm(cells: list[dict]) -> list[dict]:
    """C7: cells carry 'p', 'n', 'E_draws'. Adds rank, p_holm, holm_level, ci_holm, verdict (m = 16 always)."""
    m = len(cells)
    assert m == M_HOLM, m
    order = sorted(range(m), key=lambda i: cells[i]["p"])
    run = 0.0
    for rank, i in enumerate(order, start=1):
        c = cells[i]
        run = max(run, min(1.0, (m - rank + 1) * c["p"]))
        level = 1.0 - ALPHA / (m - rank + 1)
        c.update(rank=rank, p_holm=run, holm_level=level, ci_holm=pct_ci(c["E_draws"], level))
    for c in cells:
        if c["n"] < MIN_N:
            c["verdict"] = "UNDERPOWERED"
        elif c["p_holm"] < ALPHA and c["ci_holm"][0] > 0:
            c["verdict"] = "PASS"
        elif c["ci95"][1] < 0:
            c["verdict"] = "REFUTED"
        else:
            c["verdict"] = "INCONCLUSIVE"
    return cells


# ------------------------------------------------------------------------------------------------ events (C1)
def s1_events():
    """CR-CL A2.1 cell: swing-12 primary legs, G1, down. Returns (D, legs, covered mask, P12)."""
    import cr_cl_lib as CL
    D = CL.load()
    W, _ = CL.add_terciles(CL.build_legs(D, 12.0))
    assert len(W) == 1233 and int((W["reason"] == "ok").sum()) == 764   # CR-CL Step 0 parity
    P12 = W[~W["late"]]
    G = P12[CL.group_mask(P12, "G1", -1)].copy()
    return D, G.reset_index(drop=True), P12


def s1_covered(D, G, P12):
    """CR-CL C6 coverage: the leg's stratum has >= 1 resolved in-noise down leg (±10 race, h = +1, minute bars)."""
    import cr_cl_lib as CL
    base = P12[(P12["grp"] == "base") & (P12["dir"] == -1)]
    ok = set()
    for r in base.itertuples():
        if CL.race(D["BARS"][r.d], r.ci, r.p0, 1, 10.0) in ("win", "loss"):
            ok.add(CL.stratum(r._asdict()))
    return np.array([CL.stratum(r._asdict()) in ok for r in G.itertuples()])


def cp_data():
    m1, im = C.load_minutes(), C.load_im()
    D = C.build(m1, im)
    assert len(D["SESS"]) == 101 and abs(D["MED_IM"] - 65.71) < 0.01
    return D


def s2_events(D):
    S = C.schedule(D, C.PRIMARY)
    assert len(S) == 470
    calls = S[S["dstatus"] == "call"].reset_index(drop=True)
    assert len(calls) == 235
    return calls


def s3_events(D):
    import cr_cs_lib as CS
    RG = CS.ranges(D)
    assert len(RG) == 470
    EV = CS.events(D, RG, CS.PRIMARY)
    T, nd = CS.trades(D, EV)
    assert len(T) == 58 and T["d"].nunique() == 41 and nd == 1
    return T


def s4_events():
    """CR-CT primary: first touches, not duplicate, with a 1-s touch second; classes from the frozen step0 cuts."""
    import json
    import cr_ct_lib as CT
    S0 = json.loads((CT.OUT / "step0.json").read_text())
    D = CT.build()
    assert len(D["SESS"]) == S0["n_sessions"] == 101
    S1 = CT.load_1s(D)
    ev, _ = CT.events(D, S1)
    P = ev[(ev["touch"] == 1) & ~ev["dup"] & ev["k1s"].notna()].copy()
    assert len(P) == S0["primary_events"]["n"] == 422
    c, n = CT.cuts(P, "speed10", "relvol10")
    assert np.allclose(c, S0["cuts"]["10"]["cuts"]) and n == S0["cuts"]["10"]["n"], (c, S0["cuts"]["10"])
    P["cls10"] = CT.classify(P, "speed10", "relvol10", S0["cuts"]["10"]["cuts"])
    P["k1s"] = P["k1s"].astype(int)
    return D, S1, P.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ schedule (no outcome)
def build_all() -> dict:
    """Every setup's events, the 1-s arrays, and one row per (setup, side, event, fill model) with its fill. No outcome.
    Row: setup, side ('hyp' / 'opp'), d, ev (event index), h, model ('primary' / 'touch'), status, k_entry, k0 (first
    raced second), fill, entry_rule (C4 entry-second rule applies), sub (S1: covered / not; S4: FAST / SLOW)."""
    D1, G1, P12 = s1_events()
    cov1 = s1_covered(D1, G1, P12)
    Dcp = cp_data()
    S2 = s2_events(Dcp)
    S3 = s3_events(Dcp)
    D4, S1ct, P4 = s4_events()
    P4 = P4[P4["cls10"].isin(["FAST", "SLOW"])].reset_index(drop=True)
    assert (P4["cls10"] == "FAST").sum() == 85 and (P4["cls10"] == "SLOW").sum() == 75
    assert all(D4["SYM"][d] == Dcp["SYM"][d] for d in D4["SESS"])
    pairs = [(d, D1["SYM"][d]) for d in sorted(set(G1["d"]))] + [(d, Dcp["SYM"][d]) for d in Dcp["SESS"]]
    S1S = sec_arrays(load_1s(pairs))
    rows = []

    def add(setup, side, d, ev, h, model, st, k_entry, k0, fill, rule, sub):
        rows.append(dict(setup=setup, side=side, d=d, ev=ev, h=int(h), model=model, status=st, k_entry=k_entry,
                         k0=k0, fill=fill, entry_rule=rule, sub=sub))

    def bar_close(setup, d, ev, m, px, h, side, sym, sub):
        S = S1S[(d, sym)]
        mf = market_fill(S, m, h)
        if mf is None:
            add(setup, side, d, ev, h, "primary", "no fill: 13:00", None, None, None, True, sub)
        else:
            add(setup, side, d, ev, h, "primary", "fill", mf[0], mf[0] + 1, mf[1], True, sub)
        k0 = bar_close_start(S, m)
        add(setup, side, d, ev, h, "touch", "fill" if k0 is not None else "no fill: 13:00", None, k0, float(px),
            False, sub)

    for i, r in enumerate(G1.itertuples()):                      # S1: long = hypothesis
        for side, h in (("hyp", 1), ("opp", -1)):
            bar_close("S1", r.d, i, int(r.ci), r.p0, h, side, D1["SYM"][r.d], "covered" if cov1[i] else "uncovered")
    for i, r in enumerate(S2.itertuples()):                      # S2: chosen edge = hypothesis
        B, S = Dcp["BARS"][r.d], S1S[(r.d, Dcp["SYM"][r.d])]
        for side, h in (("hyp", int(r.side)), ("opp", -int(r.side))):
            for model, thr, rule in (("primary", TICK, True), ("touch", 0.0, True)):
                st, j, k, px = edge_fill(B, S, float(r.hi), float(r.lo), int(r.t), h, thr)
                add("S2", side, r.d, i, h, model, st, k, None if k is None else k + 1, px, rule, f"j={j}")
    for i, r in enumerate(S3.itertuples()):                      # S3: fade = hypothesis
        for side, h in (("hyp", int(r.h)), ("opp", -int(r.h))):
            bar_close("S3", r.d, i, int(r.r), r.px, h, side, Dcp["SYM"][r.d], "")
    for i, r in enumerate(P4.itertuples()):                      # S4: FAST -> break, SLOW -> fade = hypothesis
        S, b = S1S[(r.d, Dcp["SYM"][r.d])], int(r.b)
        hyp = b if r.cls10 == "FAST" else -b
        for side, h in (("hyp", hyp), ("opp", -hyp)):
            lf = level_fill(S, int(r.j), float(r.L), b, h, TICK)
            if lf is None:
                add("S4", side, r.d, i, h, "primary", "no fill: no trade-through in the touch minute", None, None,
                    None, True, r.cls10)
            else:
                add("S4", side, r.d, i, h, "primary", "fill", lf[0], lf[0] + 1, lf[1], True, r.cls10)
            add("S4", side, r.d, i, h, "touch", "fill", int(r.k1s), int(r.k1s) + 1, float(r.L), False, r.cls10)
    TR = pd.DataFrame(rows)
    SYMS = {"S1": D1["SYM"], "S2": Dcp["SYM"], "S3": Dcp["SYM"], "S4": Dcp["SYM"]}
    UNIV = {"S1": list(D1["SESS"]), "S2": list(Dcp["SESS"]), "S3": list(Dcp["SESS"]), "S4": list(Dcp["SESS"])}
    return dict(TR=TR, S1S=S1S, SYMS=SYMS, UNIV=UNIV, D1=D1, G1=G1, P12=P12, cov1=cov1, Dcp=Dcp, S2=S2, S3=S3,
                D4=D4, S1ct=S1ct, P4=P4)


def outcome(A: dict, r, T: float) -> str:
    """Race one schedule row (C4)."""
    if r.status != "fill":
        return "no fill"
    S = A["S1S"][(r.d, A["SYMS"][r.setup][r.d])]
    return race(S, int(r.k0), float(r.fill), int(r.h), T, int(r.k_entry) if r.entry_rule else None)
