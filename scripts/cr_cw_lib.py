#!/usr/bin/env python3
"""CR-CW shared definitions — clean-path approach (H1) and snap-point hold (H2). READ-ONLY, train only.

Spec: specs/CR-CW-momentum-path-snap.md (C1-C16). Imported by scripts/cr_cw_step0.py and scripts/cr_cw_run.py.
Everything CR-CT defines (universe, levels, touches, touch second, race, MFE, random baseline) is imported from
scripts/cr_ct_lib.py unchanged. New here: the 1-s front-symbol rows 06:15 .. 13:00 PT with volume (C1), volume at
price (C3), path efficiency and retrace count (C4-C7), and the H2 break / snap band / pullback / turn (C8-C13).
Importing computes no outcome; `race1s` / `mfe1s` are only called by the run script.

1-s arrays here: sec = seconds since 06:15 PT (CR-CT's 1-s arrays count from 06:30, so CR-CT sec + 900).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_ct_lib as T  # noqa: E402  (also cr_cp_lib, .env, repo path)
import cr_cp_lib as C  # noqa: E402
from packages.shared.flow_features import swing_legs  # noqa: E402

PT, UTC, PRE, NB = T.PT, T.UTC, T.PRE, T.NB
OFF = 60 * PRE                                # 900 s: 06:15 -> 06:30
TICK = 0.25
S_1230, S_1245, S_1300 = 60 * (PRE + 360), 60 * (PRE + 375), 60 * (PRE + 390)
T_MAX = 374                                   # C11: turn bar must close by 12:45 (the 12:44 bar, RTH index 374)
W_PRIMARY, W_READ = 10, (5, 20)               # C4
RETRACE = 1.0                                 # C6
RETRACE_BANDS = ((0, 1), (2, 4), (5, 10 ** 6))
BRK, PB = 6.0, 3.0                            # C8, C10
BRK_READ, PB_READ = (4.0, 8.0), (2.0, 4.0)    # C13(c)
X1, X2 = T.X1, T.X2
SEED, N_BOOT, SEED_AA, N_AA = T.SEED, T.N_BOOT, T.SEED_AA, T.N_AA
MIN_CELL, MIN_SESS, ALPHA, M_HOLM = 40, 30, 0.05, 2
OUT = C.REPO_ROOT / "scripts" / "logs" / "cr_cw"
OUT.mkdir(parents=True, exist_ok=True)
CT_STEP0 = C.REPO_ROOT / "scripts" / "logs" / "ref_cr_ct" / "step0.json"   # CR-CT's frozen cuts (copied, C7)


# ------------------------------------------------------------------------------------------------ data (C1)
def load_1s(D) -> dict:
    """Front-symbol 1-s o/h/l/c/volume 06:15 .. 13:00 PT per universe session (cached in s1_vol.parquet)."""
    path = OUT / "s1_vol.parquet"
    if path.exists():
        s = pd.read_parquet(path)
    else:
        parts = []
        for d in D["SESS"]:
            a, b = T.guard(C.pt(d, 6, 15), C.pt(d, 13, 0))
            r = C.q("""SELECT ts_utc, open::float AS o, high::float AS h, low::float AS l, close::float AS c,
                              volume::float AS v
                       FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
                       ORDER BY ts_utc""", (D["SYM"][d], a, b))
            assert r.empty or (r["ts_utc"].min() >= a and r["ts_utc"].max() < b)
            r["d"] = d
            parts.append(r)
        s = pd.concat(parts, ignore_index=True)
        s.to_parquet(path)
    out = {}
    for d, g in s.groupby("d"):
        t = pd.to_datetime(g["ts_utc"], utc=True)
        sec = ((t - pd.Timestamp(C.pt(d, 6, 15))).dt.total_seconds()).to_numpy().astype(np.int64)
        out[d] = dict(sec=sec, o=g["o"].to_numpy(float), h=g["h"].to_numpy(float), l=g["l"].to_numpy(float),
                      c=g["c"].to_numpy(float), v=g["v"].to_numpy(float))
    return out


def rth_view(S: dict) -> dict:
    """The 06:30 .. 13:00 slice with CR-CT's clock (sec since 06:30), for CR-CT's race1s / mfe1s / touch_second."""
    i = int(np.searchsorted(S["sec"], OFF))
    return {k: (S[k][i:] - OFF if k == "sec" else S[k][i:]) for k in S}, i


def ticks(px, L, b):
    """C3: b-oriented tick offset of a price from L (close rounded to 0.25)."""
    return np.rint(b * (np.asarray(px, float) - L) / TICK).astype(np.int64)


# ------------------------------------------------------------------------------------------------ H1 (C4-C7)
def window_rows(S, lo_t, hi_t):
    """Anchor index (last row before lo_t) and end index (first row at/after hi_t)."""
    i0 = int(np.searchsorted(S["sec"], lo_t)) - 1
    i1 = int(np.searchsorted(S["sec"], hi_t))
    return i0, i1


def path_stats(S, i0, i1, b):
    """C4 / C6 over closes c[i0 .. i1-1] (i0 = anchor): efficiency, net, path, retrace count."""
    c = S["c"][i0:i1]
    if i0 < 0 or len(c) < 2:
        return None
    net = b * (c[-1] - c[0])
    path = float(np.abs(np.diff(c)).sum())
    if path <= 0:
        return None
    n_rt, up, ext = 0, True, b * c[0]                # zigzag at 1 pt: count each -b leg once
    for x in b * c[1:]:
        if up:
            if x > ext:
                ext = x
            elif ext - x >= RETRACE:
                n_rt, up, ext = n_rt + 1, False, x
        else:
            if x < ext:
                ext = x
            elif x - ext >= RETRACE:
                up, ext = True, x
    return dict(eff=float(net / path), net=float(net), path=path, retr=n_rt)


def a7_start(D, d, j, b):
    """CR-CT C13 A7 leg start bar s (RTH index), or None (same logic as cr_ct_lib.a7_approach)."""
    B = D["BARS"][d]
    if j < 2:
        return None
    H, Lo = B["h"][:j], B["l"][:j]
    legs = [lg for lg in swing_legs(H, Lo, D["SW"][d]) if lg["dir"] == b]
    p = legs[-1]["end_i"] if legs else 0
    seg = Lo[p:j] if b == 1 else H[p:j]
    s = p + int(np.argmin(seg) if b == 1 else np.argmax(seg))
    return s if j - s >= 1 else None


def h1_features(D, S, r) -> dict:
    """Path features for one CR-CT touch row r (j = RTH touch bar, b)."""
    J, out = int(r.j) + PRE, {}
    for w in (W_PRIMARY,) + W_READ:
        st = path_stats(S, *window_rows(S, 60 * (J - w), 60 * J), r.b) if J - w >= 1 else None
        for k in ("eff", "net", "path", "retr"):
            out[f"{k}{w}"] = st[k] if st else None
    s = a7_start(D, r.d, int(r.j), r.b)
    st = path_stats(S, *window_rows(S, 60 * (s + PRE), 60 * J), r.b) if s is not None else None
    out["effA7"] = st["eff"] if st else None
    return out


def tercile_cuts(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return [float(np.percentile(x, 100 / 3)), float(np.percentile(x, 200 / 3))], int(len(x))


def h1_class(eff, c):
    out = np.full(len(eff), "MIDDLE", dtype=object)
    e = pd.Series(eff, dtype=float).to_numpy()
    out[e >= c[1]] = "CLEAN"
    out[e <= c[0]] = "CHOPPY"
    out[~np.isfinite(e)] = "UNDEFINED"
    return out


def retr_band(n):
    if n is None or not np.isfinite(n):
        return "undefined"
    for lo, hi in RETRACE_BANDS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}"


# ------------------------------------------------------------------------------------------------ H2 (C8-C13)
def h2_event(S, B, k, L, b, brk=BRK, pb=PB) -> dict:
    """C8-C13 for one first touch. S = CR-CW 1-s arrays (sec since 06:15), k = touch-second index into S,
    B = CR-CT front 1-min RTH bars. Returns a dict with 'stage' = no_break / no_pullback / no_turn / trade."""
    sec, h, lo, c, v = S["sec"], S["h"], S["l"], S["c"], S["v"]
    fx = b * ((h if b == 1 else lo) - L)              # b-favourable extreme of each row, from L
    ax = b * ((lo if b == 1 else h) - L)              # b-adverse extreme of each row, from L
    hit = np.where((fx[k:] >= brk) & (sec[k:] < S_1230))[0]
    if not hit.size:
        return dict(stage="no_break")
    x = k + int(hit[0])
    E = float(fx[x])
    # C9 snap band over the leg k .. x (closes, volume at price)
    ti = ticks(c[k:x + 1], L, b)
    vv = v[k:x + 1]
    nb = int(np.floor(E + 1e-9))
    band_vol = np.zeros(nb)
    m = (ti >= 0) & (ti < 4 * nb)
    np.add.at(band_vol, ti[m] // 4, vv[m])
    snap = int(nb - 1 - np.argmin(band_vol[::-1]))    # lowest volume; ties -> farthest from L
    far = snap + 0.75
    snap_x = int(nb - 2 - np.argmin(band_vol[:-1][::-1]))   # C13(f): final band excluded
    far_x = snap_x + 0.75
    # C13(b) bottom-quartile ticks L .. E inclusive (zero-volume ticks included)
    nt = int(np.floor(E / TICK + 1e-9))
    tick_vol = np.zeros(nt + 1)
    mt = (ti >= 0) & (ti <= nt)
    np.add.at(tick_vol, ti[mt], vv[mt])
    q_far = int(np.max(np.where(tick_vol <= np.percentile(tick_vol, 25))[0])) * TICK
    # C13(d) session profile 06:30 .. x on the same L-anchored 1-pt bins
    i30 = int(np.searchsorted(sec, OFF))
    st = ticks(c[i30:x + 1], L, b) // 4
    bins, cnt = np.unique(st, return_inverse=True)
    prof = np.bincount(cnt, weights=v[i30:x + 1])
    nz = prof[prof > 0]
    snap_cum = float(prof[bins == snap].sum()) if (bins == snap).any() else 0.0
    light = bool(snap_cum < float(np.median(nz))) if nz.size else None
    ev = dict(x=x, E=E, nb=nb, snap=snap, far=far, snap_x=snap_x, far_x=far_x, band_vol=band_vol.tolist(), q_far=q_far, light=light,
              snap_cum=snap_cum, prof_median=float(np.median(nz)) if nz.size else None)
    # C10 pullback: running b-extreme P from x; first row with P - adverse >= pb, before 12:45
    P, pe, r = E, x, None
    for i in range(x, len(sec)):
        if sec[i] >= S_1245:
            break
        if fx[i] > P:
            P, pe = float(fx[i]), i
        if P - ax[i] >= pb:
            r = i
            break
    if r is None:
        return dict(ev, stage="no_pullback")
    # C11 turn bar
    m_r = int(sec[r] // 60) - PRE
    Cl = B["c"]
    t = next((mm for mm in range(max(m_r, 1), T_MAX + 1) if b * (Cl[mm] - Cl[mm - 1]) > 0), None)
    if t is None:
        return dict(ev, r=r, pe=pe, P=P, stage="no_turn")
    end = int(np.searchsorted(sec, 60 * (t + PRE + 1)))
    pb_ext = float(np.min(ax[pe:end]))
    return dict(ev, r=r, pe=pe, P=P, t=t, entry=float(Cl[t]), pb_ext=pb_ext, end=end, stage="trade",
                cls="HOLD" if pb_ext > far else "FILL", cls_q="HOLD" if pb_ext > q_far else "FILL", cls_x="HOLD" if pb_ext > far_x else "FILL")


def h2_events(D, S1, P, brk=BRK, pb=PB) -> pd.DataFrame:
    """H2 over CR-CT's primary first touches P (with column k_cw = touch-second index into S1). Dedupe C12."""
    rows = []
    for r in P.itertuples():
        e = h2_event(S1[r.d], D["BARS"][r.d], int(r.k_cw), r.L, int(r.b), brk, pb)
        e.update(d=r.d, fam=r.fam, kind=r.kind, L=r.L, b=int(r.b), j=int(r.j), k_cw=int(r.k_cw))
        rows.append(e)
    H = pd.DataFrame(rows)
    H["dup"] = False
    tr = H[H["stage"] == "trade"].assign(pr=lambda z: z["fam"].map(T.FAM_PRIORITY)).sort_values(["d", "t", "b", "pr"])
    H.loc[tr.index[tr.duplicated(["d", "t", "b"])], "dup"] = True
    if len(tr):
        H["edge"] = None
        for i in tr.index:
            H.at[i, "edge"] = T.edge_at(D["BARS"][H.at[i, "d"]], int(H.at[i, "t"]), H.at[i, "entry"])
    return H


def h2_race(S, e, X):
    """C12: race from entry c[t] over 1-s rows from the start of bar t+1 (CR-CT race1s semantics)."""
    return T.race1s(S, int(e["end"]) - 1, float(e["entry"]), int(e["b"]), X)


def h2_mfe(S, e):
    return T.mfe1s(S, int(e["end"]) - 1, float(e["entry"]), int(e["b"]))


# ------------------------------------------------------------------------------------------------ stats (C14, C15)
def two_sided_p(x):
    x = x[np.isfinite(x)]
    return float(min(1.0, 2.0 * min((x <= 0).mean(), (x >= 0).mean())))


def pct_ci(x, level):
    x = x[np.isfinite(x)]
    a = (1.0 - level) / 2.0
    return [float(np.percentile(x, 100 * a)), float(np.percentile(x, 100 * (1 - a)))]


def holm(cells: dict) -> dict:
    """C14: cells name -> dict(p, na, nb, draws, ci95, ran). m = 2 fixed; a cell not run keeps its slot (p = 1)."""
    names = sorted(cells, key=lambda k: cells[k]["p"] if cells[k]["ran"] else 1.0)
    run = 0.0
    for rank, k in enumerate(names, start=1):
        c = cells[k]
        p = c["p"] if c["ran"] else 1.0
        run = max(run, min(1.0, (M_HOLM - rank + 1) * p))
        level = 1.0 - ALPHA / (M_HOLM - rank + 1)
        c.update(rank=rank, p_holm=run, holm_level=level)
        if not c["ran"]:
            c["verdict"] = c.get("verdict", "UNDERPOWERED")
            continue
        c["ci_holm"] = pct_ci(c["draws"], level)
        if min(c["na"], c["nb"]) < MIN_CELL:
            c["verdict"] = "UNDERPOWERED"
        elif run < ALPHA and c["ci_holm"][0] > 0:
            c["verdict"] = "PASS"
        elif c["ci_holm"][1] < 0:
            c["verdict"] = "REFUTED"
        else:
            c["verdict"] = "INCONCLUSIVE"
    return cells


def aa(keys, BT, sess_list):
    """CR-CT C12 A/A: 200 session half-splits (seed 20261025); keys = (m, b, edge)."""
    rng = np.random.default_rng(SEED_AA)
    sess = np.array(sess_list, dtype=object)
    grp = {k: g for k, g in BT.groupby(["m", "h", "edge"])}
    diffs, excl = [], []
    for _ in range(N_AA):
        perm = rng.permutation(len(sess))
        A = set(sess[perm[: len(sess) // 2]])
        vals, nx = [], 0
        for k in keys:
            g = grp.get(k)
            if g is None:
                nx += 1
                continue
            inA = g["d"].isin(A).to_numpy()
            ga, gb = g[inA], g[~inA]
            if not len(ga) or not len(gb):
                nx += 1
                continue
            vals.append(float(ga["w"].iloc[rng.integers(len(ga))]) - float(gb["w"].mean()))
        diffs.append(float(np.mean(vals)) if vals else np.nan)
        excl.append(nx / max(len(keys), 1))
    d = np.array(diffs)
    return dict(n_splits=N_AA, keys=len(keys), mean_diff_pp=100 * float(np.nanmean(d)),
                median_diff_pp=100 * float(np.nanmedian(d)), sd_diff_pp=100 * float(np.nanstd(d)),
                p5_pp=100 * float(np.nanpercentile(d, 5)), p95_pp=100 * float(np.nanpercentile(d, 95)),
                mean_excluded_share=float(np.mean(excl)))


# ------------------------------------------------------------------------------------------------ shared build
def build_all():
    """Universe, CR-CT primary first touches (asserted 422), CR-CW 1-s arrays, k_cw, H1 features. No outcome."""
    D = T.build()
    S_ct = T.load_1s(D)
    ev, _ = T.events(D, S_ct)
    ev["no1s"] = ev["k1s"].isna()
    P = ev[(ev["touch"] == 1) & ~ev["dup"] & ~ev["no1s"]].copy()
    assert len(P) == 422, len(P)
    S1 = load_1s(D)
    kc = []
    for r in P.itertuples():
        t = S_ct[r.d]["sec"][int(r.k1s)] + OFF
        i = int(np.searchsorted(S1[r.d]["sec"], t))
        assert S1[r.d]["sec"][i] == t, (r.d, t)
        kc.append(i)
    P["k_cw"] = kc
    F = pd.DataFrame([h1_features(D, S1[r.d], r) for r in P.itertuples()], index=P.index)
    P = pd.concat([P, F], axis=1)
    return D, S_ct, S1, P
