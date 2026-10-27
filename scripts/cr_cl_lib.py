"""CR-CL shared definitions — legs, skew reads, groups, strata, race, matched estimator. READ-ONLY.

Spec: specs/CR-CL-skew-beyond-noise.md (K1-K10, C1-C16, A1, Approvals). Imported by scripts/cr_cl_step0.py and
scripts/cr_cl_run.py so both use one definition of every quantity.

  data (C1):    CR-CJ loaders, period `y26` only (2026-01-02 .. 2026-06-05); R-a, 2025 and the holdout are never loaded.
  legs (C2):    flow_features.swing_legs on Databento bars; skew window s = start_i + 1, e = end_i + 1; s >= 07:00 PT;
                primary e <= 11:00 PT, late 11:00 < e <= 12:45 PT.
  read (C3):    skew_expected.stickiness_read exactly as CR-CK Part 1, band = the shipped NOISE_BAND_PP.
  groups (C4/C5): G1 ratio < 0 (h = -dir), G2 ratio > 1 (h = +dir), beyond noise; baseline = ratio defined, in noise.
  strata (C6/C7): direction x size tercile x confirmation 30-min bucket x EDGE / INTERIOR.
  race (C9):    from the close of bar confirm_i; same-minute tie and unresolved excluded and counted.
"""
from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

import cr_cj_lib as L
from packages.shared import skew_expected as se
from packages.shared.flow_features import swing_legs
from packages.shared.level_placement import bucket_levels, es_snapshot_bar_start_utc, place_levels
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day

SEED = 20260920
RTH0 = 6 * 60 + 30                         # minute of day of index 0 (06:30 PT)
DOM_LO = 30                                # s >= 07:00 PT
PRIMARY_END = 270                          # e <= 11:00 PT
LATE_END = L.NB - 15                       # e <= 12:45 PT (CR-CK domain)
EDGE_PTS = 2.0                             # C7
GAP_MAX = 5                                # C9: a missing run > 5 bars inside the race drops the leg
MIN_CELL = 40                              # C12
CELLS = [("G1", 1), ("G1", -1), ("G2", 1), ("G2", -1)]
PERIOD = os.environ.get("CR_CL_PERIOD", "y26")   # CR-CN N2: "ra" re-runs CR-CL unchanged on R-a
OUT = L.REPO_ROOT / "scripts" / "logs" / ("cr_cl" if PERIOD == "y26" else "cr_cn")
OUT.mkdir(parents=True, exist_ok=True)

# GEX readout (C14): CR-CF select_levels constants
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N, INTO_PTS = 30, 40e9, 5.0, 1.0, 6, 5.0


def load():
    mins = L.skew_minutes(L.load_skew(PERIOD))
    sym, bars = L.front_bars(L.load_es(PERIOD))
    sess = [d for d in L.sessions(PERIOD) if d in mins and d in bars]
    px = {d: L.es_close_at_snapshot(bars, d, mins) for d in sess}
    return dict(MIN=mins, BARS=bars, SYM=sym, SESS=sess, PX=px)


def vols_at(D, d, i):
    j = D["MIN"][d]["row"][i]
    if j < 0:
        return None
    row = D["MIN"][d]["rows"].iloc[j]
    return {n: float(row[f"vol{n}"]) for n in range(5, 100, 5)}


def read(D, d, s, e):
    """CR-CK Part 1 `read`, unchanged (band = shipped NOISE_BAND_PP)."""
    put, px = D["MIN"][d]["put"], D["PX"][d]
    v = vols_at(D, d, s)
    if v is None or not (np.isfinite(px[s]) and np.isfinite(px[e])) or not np.isfinite(put[s]):
        return None
    idx = np.arange(s, e + 1)
    ok = np.isfinite(put[idx])
    return se.stickiness_read((idx[ok] + RTH0).tolist(), put[idx][ok].tolist(), int(s + RTH0), int(e + RTH0),
                              v, float(px[s]), float(px[e]))


def build_legs(D, swing: float) -> pd.DataFrame:
    """Every leg with s >= 07:00 and e <= 12:45 PT, its skew read, group, strata keys (size tercile added later)."""
    rows = []
    for d in D["SESS"]:
        B = D["BARS"][d]
        H, Lo = B["h"], B["l"]
        if not (np.isfinite(H).all() and np.isfinite(Lo).all()):
            continue
        for lg in swing_legs(H, Lo, swing):
            s, e, ci = lg["start_i"] + 1, lg["end_i"] + 1, lg["confirm_i"]
            if s < DOM_LO or e > LATE_END:
                continue
            rd = read(D, d, s, e)
            reason = "no read" if rd is None else rd.reason
            dr = int(lg["dir"])
            runhi, runlo = float(np.max(H[:ci + 1])), float(np.min(Lo[:ci + 1]))
            edge = (lg["end_px"] >= runhi - EDGE_PTS) if dr == 1 else (lg["end_px"] <= runlo + EDGE_PTS)
            ratio = None if rd is None else rd.ratio
            beyond = None if rd is None else rd.beyond_noise
            if reason == "ok" and beyond:
                grp = "G1" if ratio < 0 else ("G2" if ratio > 1 else "neither")
            elif reason == "ok" and beyond is False:
                grp = "base"
            else:
                grp = "excl"
            rows.append(dict(d=d, swing=swing, dir=dr, s=s, e=e, ci=int(ci), start_px=float(lg["start_px"]),
                             end_px=float(lg["end_px"]), size=abs(float(lg["end_px"]) - float(lg["start_px"])),
                             p0=float(B["c"][ci]), cbucket=int(ci) // 30, sbucket=s // 30,
                             edge="EDGE" if edge else "INTERIOR", late=e > PRIMARY_END,
                             reason=reason, ratio=ratio, beyond=beyond,
                             actual=None if rd is None else rd.actual_pp, expected=None if rd is None else rd.expected_pp,
                             band=None if rd is None else rd.band_pp, grp=grp))
    return pd.DataFrame(rows)


def add_terciles(W: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """C6b: tercile cut points of leg size over primary legs with a ratio (directions pooled), per swing."""
    W = W.copy()
    ref = W[(~W["late"]) & (W["reason"] == "ok")]["size"]
    cuts = [float(np.percentile(ref, 100 / 3)), float(np.percentile(ref, 200 / 3))]
    W["tercile"] = np.digitize(W["size"], cuts)
    return W, cuts


def stratum(r) -> tuple:
    return (int(r["dir"]), int(r["tercile"]), int(r["cbucket"]), r["edge"])


def group_mask(W, g, dr, ratio_lo=0.0, ratio_hi=1.0):
    """Cell membership; the K9 ratio-cut readout passes ratio_lo = -0.5, ratio_hi = 1.5."""
    ok = (W["reason"] == "ok") & (W["beyond"] == True) & (W["dir"] == dr)  # noqa: E712
    rr = W["ratio"].astype(float)
    return ok & ((rr < ratio_lo) if g == "G1" else (rr > ratio_hi))


def h_of(g, dr):
    return -dr if g == "G1" else dr


# ------------------------------------------------------------------------------------------------ outcomes
def race(B, ci, p0, h, X):
    """C9: 'win' / 'loss' / 'tie' / 'unresolved' / 'gap'. Bars ci+1 .. 389."""
    H, Lo = B["h"], B["l"]
    run = 0
    for j in range(ci + 1, L.NB):
        if not (np.isfinite(H[j]) and np.isfinite(Lo[j])):
            run += 1
            if run > GAP_MAX:
                return "gap"
            continue
        run = 0
        fav = (H[j] - p0) if h == 1 else (p0 - Lo[j])
        adv = (p0 - Lo[j]) if h == 1 else (H[j] - p0)
        if fav >= X and adv >= X:
            return "tie"
        if fav >= X:
            return "win"
        if adv >= X:
            return "loss"
    return "unresolved"


def mfe_before(B, ci, p0, h, adv_x=10.0):
    """C10: largest favourable excursion before the first -adv_x touch (or 13:00). Same-bar: the bar's favourable
    extreme is not credited when the bar also touches -adv_x (conservative)."""
    H, Lo = B["h"], B["l"]
    best = 0.0
    for j in range(ci + 1, L.NB):
        if not (np.isfinite(H[j]) and np.isfinite(Lo[j])):
            continue
        fav = (H[j] - p0) if h == 1 else (p0 - Lo[j])
        adv = (p0 - Lo[j]) if h == 1 else (H[j] - p0)
        if adv >= adv_x:
            break
        best = max(best, fav)
    return best


def move_at(B, ci, p0, h, k):
    j = ci + k
    return h * (B["c"][j] - p0) if j < L.NB and np.isfinite(B["c"][j]) else np.nan


# ------------------------------------------------------------------------------------- matched estimator
def strata_matrices(sessions, grp_rows, base_rows):
    """Per-session counts by stratum: Gn, Gw (group legs, wins) and Bn, Bw (baseline). rows: (d, stratum, win)."""
    keys = sorted({k for _, k, _ in grp_rows} | {k for _, k, _ in base_rows}, key=str)
    ki = {k: i for i, k in enumerate(keys)}
    si = {d: i for i, d in enumerate(sessions)}
    S, K = len(sessions), len(keys)
    M = {n: np.zeros((S, K)) for n in ("Gn", "Gw", "Bn", "Bw")}
    for d, k, w in grp_rows:
        M["Gn"][si[d], ki[k]] += 1
        M["Gw"][si[d], ki[k]] += w
    for d, k, w in base_rows:
        M["Bn"][si[d], ki[k]] += 1
        M["Bw"][si[d], ki[k]] += w
    return M, keys


def matched_diff(Wt, M):
    """Wt: (n_draws, S) session weights (ones for the point estimate). Returns (diff, n_covered, coverage) arrays."""
    Gn, Gw, Bn, Bw = (Wt @ M[k] for k in ("Gn", "Gw", "Bn", "Bw"))
    cov = Bn > 0
    br = np.divide(Bw, Bn, out=np.zeros_like(Bw), where=cov)
    gn_c = (Gn * cov).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = ((Gw * cov).sum(1) - (Gn * br * cov).sum(1)) / gn_c
        coverage = gn_c / Gn.sum(1)
    return diff, gn_c, coverage


def boot_weights(n_sessions, n_draws, rng):
    idx = rng.integers(0, n_sessions, size=(n_draws, n_sessions))
    Wt = np.zeros((n_draws, n_sessions))
    np.add.at(Wt, (np.repeat(np.arange(n_draws), n_sessions), idx.ravel()), 1)
    return Wt


# --------------------------------------------------------------------------------------- GEX levels (C14)
def gex_levels(D) -> tuple[dict, dict]:
    """Corrected levels per session (CR-CF select_levels, copied). Sessions whose prior session is not a y26 session
    with the same front symbol (roll weeks, 2026-01-02), or without P_target / IM / gamma rows, get none (counted)."""
    from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
    lo, hi = L.PERIODS[PERIOD]
    L.guard(lo, hi)
    oig = L.q("""
        SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
        FROM orats_oi_gamma
        WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
          AND expir_date >= trade_date AND expir_date <= trade_date + %s
    """, (lo, hi, DTE_MAX))
    oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
    oig = oig[(oig["d"] >= lo) & (oig["d"] <= hi)]
    oig = oig[[d_ <= e <= d_ + timedelta(days=DTE_MAX) for e, d_ in zip(oig["expir_date"], oig["d"])]]
    oig["call"] = oig["gex_call"].astype(float)
    oig["put"] = -oig["gex_put"].abs().astype(float)
    feat = L.q("""
        SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
        WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
    """, (CANONICAL_FEATURE_VERSION, lo, hi)).set_index("trade_date")["im"]
    feat = feat[~feat.index.duplicated()]
    starts = {}
    for d in D["SESS"]:
        a = prev_trading_day(d)
        if a in D["SYM"] and D["SYM"][a] == D["SYM"][d]:
            starts[es_snapshot_bar_start_utc(a)] = d
    assert all(lo <= starts[k] <= hi for k in starts)
    es = L.q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)", (list(starts),))
    ptgt = {starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
            if c is not None and np.isfinite(c) and c > 0}
    out, why = {}, {}
    gby = dict(tuple(oig.groupby("d")))
    for d in D["SESS"]:
        o = D["BARS"][d]["o"][0]
        if d not in ptgt:
            why[d] = "no P_target (roll / first session / missing bar)"
            continue
        if not (d in feat.index and pd.notna(feat[d]) and feat[d] > 0):
            why[d] = "no IM"
            continue
        if d not in gby or not np.isfinite(o):
            why[d] = "no gamma rows / no open"
            continue
        g = place_levels(gby[d], ptgt[d])
        g = g[g["level_corrected"].notna()].copy()
        g["level"] = bucket_levels(g["level_corrected"].to_numpy(), 1.0)
        bk = g.groupby("level").agg(call=("call", "sum"), put=("put", "sum")).reset_index()
        bk["mag"] = bk["call"].abs() + bk["put"].abs()
        bk["net"] = bk["call"] + bk["put"]
        bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
        kept = []
        for r in bk.itertuples():
            if all(abs(r.level - k.level) >= SPACING for k in kept):
                kept.append(r)
        win = [k for k in kept if abs(k.level - o) <= WIN_IM * float(feat[d])]
        out[d] = [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:TOP_N]]
    return out, why


def tag_gex(W, levels):
    """C14: 'put' / 'call' if the leg's extreme is within 5 pts of a level (nearest wins), 'none', or 'no levels'."""
    tags = []
    for r in W.itertuples():
        lv = levels.get(r.d)
        if lv is None:
            tags.append("no levels")
            continue
        near = [x for x in lv if abs(r.end_px - x["L"]) <= INTO_PTS]
        if not near:
            tags.append("none")
        else:
            x = min(near, key=lambda z: abs(r.end_px - z["L"]))
            tags.append("put" if x["net"] < 0 else "call")
    W = W.copy()
    W["gex"] = tags
    return W
