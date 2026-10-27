#!/usr/bin/env python3
"""CR-CT shared definitions — speed into the level. READ-ONLY, train only (2026-01-06 .. 2026-06-05).

Spec: specs/CR-CT-approach-speed.md (E-R2, C1-C15). Imported by scripts/cr_ct_step0.py and scripts/cr_ct_run.py so
both use one definition of every quantity. Importing computes no outcome; `race1s`, `move_at`, `mfe1s` and
`baseline_table` are only called by the run script (and the baseline by the Step 0 A/A, C12).

  data (C1, C2): es_flow_1m_databento 06:15 .. 12:59 PT, all outright symbols (price = the session's most-RTH-volume
                 symbol, volume = all symbols); es_flow_1s_databento front symbol 06:30 .. 13:00 PT; F2 from
                 ironbeam_es_1m_bars. Universe = CR-CP's funnel (cr_cp_lib.build). Every read asserted inside
                 [2026-01-01 00:00 UTC, 2026-06-05 21:00 UTC).
  levels (C3), arming / touch (C4), pooled events (C5), touch second (C6), approach (C7, C8, C13, C14).
Ext arrays index 0 = 06:15 PT; RTH bar k = ext index k + 15.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_cp_lib as C  # noqa: E402  (also loads .env and the repo path)
from packages.shared.flow_features import swing_legs  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, prev_trading_day  # noqa: E402

assert C.PERIOD == "train"
PT, UTC = C.PT, C.UTC
D_FROM, D_TO = date(2026, 1, 6), date(2026, 6, 5)
PRIOR_FROM = date(2026, 1, 2)                 # C1: priors only (F1, trailing volume); 2025 never read
FLOOR, CAP = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
NB, PRE = C.NB, 15                            # RTH bars; ext offset (06:15 -> 06:30)
NE = NB + PRE
ARM_IM, ARM_MIN = 0.10, 4.0                   # C4
WIN_IM = 1.0                                  # C3 (F1 / F2 window)
MERGE_FAM, MERGE_POOL = 0.5, 1.0              # C3, C5
RUN_AGE = 30                                  # C3
W_PRIMARY, W_READ = 10, (5, 20)               # C7, C13
N_PRIOR, MIN_PRIOR = 20, 10                   # C8
EDGE_PTS = 2.0                                # C12
BSG_SWING, BSG_WIN = 3.0, 60                  # C14
STRONG_PTS = 3.0                              # C13
FIRST_HOUR_END = 60                           # C13: touch bar < 07:30 PT
X1, X2 = 10.0, 20.0
SEED, N_BOOT, SEED_AA, N_AA = 20260920, 10_000, 20261025, 200
MIN_CELL, MIN_SESS = 40, 30
FAM_PRIORITY = {"F1": 0, "F2": 1, "RUN": 2}
OUT = C.REPO_ROOT / "scripts" / "logs" / "cr_ct"
OUT.mkdir(parents=True, exist_ok=True)


def guard(lo, hi):
    assert FLOOR <= lo <= hi <= CAP, (lo, hi)
    return lo, hi


# ------------------------------------------------------------------------------------------------ data (C1)
def load_m1() -> pd.DataFrame:
    path = OUT / "m1_ext.parquet"
    if path.exists():
        return pd.read_parquet(path)
    a, b = guard(C.pt(PRIOR_FROM, 6, 15), C.pt(D_TO, 13, 0))
    r = C.q("""
        SELECT minute_utc, symbol, open::float AS o, high::float AS h, low::float AS l, close::float AS c,
               volume::float AS vol, buy_vol::float AS buy, sell_vol::float AS sell
        FROM es_flow_1m_databento
        WHERE minute_utc >= %s AND minute_utc < %s
          AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:15'
          AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        ORDER BY minute_utc, symbol
    """, (a, b))
    assert r["minute_utc"].min() >= a and r["minute_utc"].max() < b
    r.to_parquet(path)
    return r


def ext_arrays(m1: pd.DataFrame):
    """Per date: front symbol (most RTH volume), raw front OHLC (NE,), flat-filled front OHLC, total volume (NE,)."""
    m1 = m1.copy()
    ts = pd.to_datetime(m1["minute_utc"], utc=True).dt.tz_convert(PT)
    m1["d"] = ts.dt.date
    m1["e"] = (ts.dt.hour * 60 + ts.dt.minute - 375).astype(int)
    rth = m1[m1["e"] >= PRE]
    v = rth.groupby(["d", "symbol"])["vol"].sum().reset_index().sort_values(["d", "vol"], ascending=[True, False])
    SYM = {d: g.iloc[0]["symbol"] for d, g in v.groupby("d")}
    RAW, FILL, TOT, LAST = {}, {}, {}, {}
    for d, g in m1.groupby("d"):
        tot = np.zeros(NE)
        np.add.at(tot, g["e"].to_numpy(), g["vol"].to_numpy(float))
        f = g[g["symbol"] == SYM[d]]
        raw = {k: np.full(NE, np.nan) for k in ("o", "h", "l", "c")}
        for k in raw:
            raw[k][f["e"].to_numpy()] = f[k].to_numpy(float)
        have = np.isfinite(raw["c"])
        fl = {k: raw[k].copy() for k in raw}
        if have.any():
            first, last = int(np.argmax(have)), int(NE - 1 - np.argmax(have[::-1]))
            for j in range(first + 1, last + 1):
                if not have[j]:
                    for k in fl:
                        fl[k][j] = fl["c"][j - 1]
            LAST[d] = last
        RAW[d], FILL[d], TOT[d] = raw, fl, tot
    return dict(SYM=SYM, RAW=RAW, FILL=FILL, TOT=TOT, LAST=LAST)


def load_1s(D) -> dict:
    """C6 / C10: front-symbol 1-s bars 06:30 .. 13:00 PT per universe session (cached)."""
    path = OUT / "s1_front.parquet"
    if path.exists():
        s = pd.read_parquet(path)
    else:
        parts = []
        for d in D["SESS"]:
            a, b = guard(C.pt(d, 6, 30), C.pt(d, 13, 0))
            r = C.q("""SELECT ts_utc, high::float AS h, low::float AS l, close::float AS c FROM es_flow_1s_databento
                       WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s ORDER BY ts_utc""", (D["SYM"][d], a, b))
            r["d"] = d
            parts.append(r)
        s = pd.concat(parts, ignore_index=True)
        s.to_parquet(path)
    out = {}
    for d, g in s.groupby("d"):
        t = pd.to_datetime(g["ts_utc"], utc=True)
        sec = ((t - pd.Timestamp(C.pt(d, 6, 30))).dt.total_seconds()).to_numpy().astype(np.int64)
        out[d] = dict(sec=sec, h=g["h"].to_numpy(float), l=g["l"].to_numpy(float), c=g["c"].to_numpy(float))
    return out


def overnight(d, prev, raw_d) -> tuple:
    """C3 F2 (CR-CF `overnight`, Databento 06:29 check against the ext raw close)."""
    t0, t1 = guard(C.pt(prev, 15, 0), C.pt(d, 6, 30))
    r = C.q("""SELECT max(high)::float h, min(low)::float l, count(*) n FROM ironbeam_es_1m_bars
               WHERE datetime >= %s AND datetime < %s""", (t0.replace(tzinfo=None), t1.replace(tzinfo=None)))
    if not r["n"][0]:
        return None, "no Ironbeam overnight bars"
    ib = C.q("SELECT close::float c FROM ironbeam_es_1m_bars WHERE datetime = %s",
             (C.pt(d, 6, 29).astimezone(UTC).replace(tzinfo=None),))
    dc = raw_d["c"][PRE - 1]
    if not len(ib) or not np.isfinite(dc):
        return None, "no 06:29 bar"
    if abs(ib["c"][0] - dc) > 2.0:
        return None, f"06:29 Ironbeam {ib['c'][0]} vs Databento {dc}"
    return dict(h=float(r["h"][0]), l=float(r["l"][0])), None


def build():
    """Universe (CR-CP funnel on RTH minutes of 2026-01-06 .. 06-05), ext arrays, prior lists, IM, swing_d."""
    m1 = load_m1()
    E = ext_arrays(m1)
    ts = pd.to_datetime(m1["minute_utc"], utc=True).dt.tz_convert(PT)
    rth = m1[(ts.dt.time >= pd.Timestamp("06:30").time()) & (ts.dt.date >= D_FROM)]
    im = C.load_im()
    D = C.build(rth, im)
    for d in D["SESS"]:                       # ext RTH slice == CR-CP bars (same symbol, same fill)
        assert D["SYM"][d] == E["SYM"][d]
        for k in ("h", "l", "c"):
            a, b = D["BARS"][d][k], E["FILL"][d][k][PRE:]
            ok = np.isfinite(a)
            assert np.allclose(a[ok], b[ok]), (d, k)
    alld = sorted(d for d in E["RAW"] if is_trading_day(d))
    # C8: priors = trading dates with Databento minutes whose last front bar is >= 12:55 PT (not an early close)
    PRI = [d for d in alld if E["LAST"].get(d, -1) >= C.EARLY_LAST + PRE]
    D.update(E=E, ALLD=alld, PRI=PRI)
    return D


# ---------------------------------------------------------------------------------------------- levels (C3)
def merge(lv, tol):
    out = []
    for x in sorted(lv, key=lambda z: z["L"]):
        if out and abs(x["L"] - out[-1]["L"]) < tol and x["fam"] == out[-1]["fam"]:
            out[-1]["kind"] += "+" + x["kind"]
            continue
        out.append(dict(x))
    return out


def run_levels(H, Lo):
    """C3 RUN: every running-high (low) value whose last print is >= 30 bars old; known bar = last print + 30."""
    out = []
    for sgn, A in ((1, H), (-1, Lo)):
        v, lp, done = None, None, set()
        for k in range(NB):
            x = A[k]
            if v is None or sgn * (x - v) > 0:
                v, lp = float(x), k
            elif x == v:
                lp = k
            if k - lp == RUN_AGE and v not in done:
                done.add(v)
                out.append(dict(fam="RUN", kind="run_high" if sgn == 1 else "run_low", L=v, known=k))
    return out


def levels(D, d, why: dict):
    E, B = D["E"], D["BARS"][d]
    o, im = float(B["o"][0]), D["IM"][d]
    lv = []
    prev = prev_trading_day(d)
    prev = prev if prev in E["RAW"] else None
    if prev is None or E["SYM"].get(prev) != E["SYM"][d]:
        why.setdefault("F1", []).append((str(d), "prior session missing or other front symbol"))
    else:
        R = E["RAW"][prev]
        hh, ll, cc = R["h"][PRE:], R["l"][PRE:], R["c"][PRE:]
        last = int(np.where(np.isfinite(cc))[0][-1])
        for kind, L_ in (("prior_high", np.nanmax(hh)), ("prior_low", np.nanmin(ll)), ("prior_close", cc[last])):
            lv.append(dict(fam="F1", kind=kind, L=float(L_), known=0))
    if prev is not None:
        on, err = overnight(d, prev, E["RAW"][d])
        if on is None:
            why.setdefault("F2", []).append((str(d), err))
        else:
            lv += [dict(fam="F2", kind="on_high", L=on["h"], known=0), dict(fam="F2", kind="on_low", L=on["l"], known=0)]
    lv = [x for x in lv if abs(x["L"] - o) <= WIN_IM * im]
    lv = merge([x for x in lv if x["fam"] == "F1"], MERGE_FAM) + merge([x for x in lv if x["fam"] == "F2"], MERGE_FAM)
    return lv + run_levels(B["h"], B["l"])


def arm_of(D, d):
    return max(ARM_IM * D["IM"][d], ARM_MIN)


def touches(B, L_, known, arm, n_max=2):
    """C4: [(touch bar j, armed side s, arming bar a)] — first and second touch."""
    H, Lo, Cl = B["h"], B["l"], B["c"]
    out, side, abar = [], None, None
    for k in range(known, NB):
        if side is not None and k > abar:
            if (side == -1 and H[k] >= L_) or (side == 1 and Lo[k] <= L_):
                out.append((k, side, abar))
                side = None
                if len(out) >= n_max:
                    break
                continue
        if Cl[k] <= L_ - arm and side != -1:
            side, abar = -1, k
        elif Cl[k] >= L_ + arm and side != 1:
            side, abar = 1, k
    return out


def touch_second(S1, j, L_, b):
    """C6: index into the 1-s arrays of the first second in bar j that reaches L, or None."""
    lo, hi = np.searchsorted(S1["sec"], 60 * j), np.searchsorted(S1["sec"], 60 * (j + 1))
    if hi <= lo:
        return None
    seg = S1["h"][lo:hi] >= L_ if b == 1 else S1["l"][lo:hi] <= L_
    return lo + int(np.argmax(seg)) if seg.any() else None


# --------------------------------------------------------------------------------------------- approach (C7)
def prior_median(D, d, a, b):
    """C8: median total volume over ext [a, b) on the up-to-20 most recent priors (>= 10 required)."""
    pri = [p for p in D["PRI"] if p < d][-N_PRIOR:]
    if len(pri) < MIN_PRIOR:
        return None, len(pri)
    return float(np.median([D["E"]["TOT"][p][a:b].sum() for p in pri])), len(pri)


def approach(D, d, j, b, w):
    """C7 / C8 for window w minutes ending at bar j - 1. Returns (speed, relvol, why)."""
    F, T = D["E"]["FILL"][d], D["E"]["TOT"][d]
    J = j + PRE
    if J - 1 - w < 0 or not np.isfinite(F["c"][J - 1 - w]):
        return None, None, "window before 06:15 / no bar"
    speed = b * (F["c"][J - 1] - F["c"][J - 1 - w]) / w / D["IM"][d]
    med, _ = prior_median(D, d, J - w, J)
    if med is None:
        return float(speed), None, "< 10 prior sessions"
    return float(speed), float(T[J - w:J].sum() / med) if med > 0 else None, None


def a7_approach(D, d, j, b):
    """C13 A7 leg: vol-scaled swing; start = most b-adverse extreme since the last confirmed b-direction extreme."""
    B = D["BARS"][d]
    if j < 2:
        return None, None, "touch at the open"
    H, Lo, Cl = B["h"][:j], B["l"][:j], B["c"]
    legs = [lg for lg in swing_legs(H, Lo, D["SW"][d]) if lg["dir"] == b]
    p = legs[-1]["end_i"] if legs else 0
    seg = Lo[p:j] if b == 1 else H[p:j]
    s = p + int(np.argmin(seg) if b == 1 else np.argmax(seg))
    px = Lo[s] if b == 1 else H[s]
    mins = j - s
    if mins < 1:
        return None, None, "no leg"
    speed = b * (Cl[j - 1] - px) / mins / D["IM"][d]
    med, _ = prior_median(D, d, s + PRE, j + PRE)
    if med is None or med <= 0:
        return float(speed), None, "< 10 prior sessions"
    return float(speed), float(D["E"]["TOT"][d][s + PRE:j + PRE].sum() / med), None


def before_she_goes(B, j, L_, b):
    """C14: >= 2 pullbacks (the -b swing legs at 3 pts) in the prior 60 min, depths strictly decreasing."""
    w0 = max(0, j - BSG_WIN)
    if j - w0 < 3:
        return False, 0
    legs = [lg for lg in swing_legs(B["h"][w0:j], B["l"][w0:j], BSG_SWING) if lg["dir"] == -b]
    dep = [b * (L_ - lg["end_px"]) for lg in legs]
    ok = len(dep) >= 2 and all(dep[k + 1] < dep[k] for k in range(len(dep) - 1))
    return bool(ok), len(dep)


def edge_at(B, j, px):
    """C12: within 2 pts of the running RTH high or low through bar j."""
    hi, lo = float(np.max(B["h"][:j + 1])), float(np.min(B["l"][:j + 1]))
    return "EDGE" if (px >= hi - EDGE_PTS or px <= lo + EDGE_PTS) else "INTERIOR"


# --------------------------------------------------------------------------------------------- events
def events(D, S1):
    """All first and second touches per level, with features. No outcome."""
    rows, why = [], {}
    for d in D["SESS"]:
        B, arm = D["BARS"][d], arm_of(D, d)
        for lv in levels(D, d, why):
            for n, (j, s, a) in enumerate(touches(B, lv["L"], lv["known"], arm)):
                b = -s
                k = touch_second(S1[d], j, lv["L"], b) if d in S1 else None
                r = dict(d=d, fam=lv["fam"], kind=lv["kind"], L=lv["L"], known=lv["known"], touch=n + 1, j=j,
                         b=b, arm_bar=a, k1s=k, edge=edge_at(B, j, lv["L"]), first_hour=j < FIRST_HOUR_END)
                for w in (W_PRIMARY,) + W_READ:
                    sp, rv, wy = approach(D, d, j, b, w)
                    r[f"speed{w}"], r[f"relvol{w}"], r[f"why{w}"] = sp, rv, wy
                sp, rv, wy = a7_approach(D, d, j, b)
                r["speedA7"], r["relvolA7"], r["whyA7"] = sp, rv, wy
                r["bsg"], r["n_pullbacks"] = before_she_goes(B, j, lv["L"], b)
                rows.append(r)
    ev = pd.DataFrame(rows)
    ev["dup"] = False
    keep = {}
    order = ev.assign(pr=ev["fam"].map(FAM_PRIORITY)).sort_values(["d", "touch", "j", "pr"]).index
    for i in order:                                          # C5: pooled dedupe, per touch number
        r = ev.loc[i]
        key = (r["d"], r["touch"], r["j"], r["b"])
        if any(abs(r["L"] - x) <= MERGE_POOL for x in keep.get(key, [])):
            ev.at[i, "dup"] = True
        else:
            keep.setdefault(key, []).append(r["L"])
    return ev, why


def cuts(ev, sp, rv):
    m = ev[sp].notna() & ev[rv].notna()
    x = ev.loc[m, sp].to_numpy(float)
    return [float(np.percentile(x, 100 / 3)), float(np.percentile(x, 200 / 3))], int(m.sum())


def classify(ev, sp, rv, c):
    lo, hi = c
    out = np.full(len(ev), "NEITHER", dtype=object)
    s, v = ev[sp].astype(float), ev[rv].astype(float)
    out[(s >= hi) & (v >= 1)] = "FAST"
    out[(s <= lo) & (v < 1)] = "SLOW"
    out[s.isna() | v.isna()] = "UNDEFINED"
    return out


def tag_strong(ev, LV):
    near, kind = [], []
    for r in ev.itertuples():
        st = LV["STRONG"].get(r.d) if r.d in LV["STRONG"] else None
        if st is None:
            near.append("no levels"), kind.append("no levels")
            continue
        x = [z for z in st if abs(z["L"] - r.L) <= STRONG_PTS]
        if not x:
            near.append("none"), kind.append("none")
            continue
        z = min(x, key=lambda q: abs(q["L"] - r.L))
        near.append("strong"), kind.append("put" if z["net"] < 0 else "call")
    ev = ev.copy()
    ev["strong"], ev["a8"] = near, kind
    return ev


# --------------------------------------------------------------------------------------------- outcomes
def race1s(S1, k, L_, b, X):
    """C10: from second k + 1 to 13:00. 'win' / 'loss' / 'tie' / 'unresolved'."""
    h, lo = S1["h"][k + 1:], S1["l"][k + 1:]
    fav = (h - L_) if b == 1 else (L_ - lo)
    adv = (L_ - lo) if b == 1 else (h - L_)
    fw, aw = fav >= X, adv >= X
    i_f = int(np.argmax(fw)) if fw.any() else None
    i_a = int(np.argmax(aw)) if aw.any() else None
    if i_f is None and i_a is None:
        return "unresolved"
    if i_a is None or (i_f is not None and i_f < i_a):
        return "win"
    if i_f is None or i_a < i_f:
        return "loss"
    return "tie"


def mfe1s(S1, k, L_, b, adv_x=X1):
    """Largest favourable excursion before the first -adv_x second (that second's favourable side not credited)."""
    h, lo = S1["h"][k + 1:], S1["l"][k + 1:]
    fav = (h - L_) if b == 1 else (L_ - lo)
    adv = (L_ - lo) if b == 1 else (h - L_)
    aw = adv >= adv_x
    stop = int(np.argmax(aw)) if aw.any() else len(fav)
    return float(max(0.0, np.max(fav[:stop]))) if stop > 0 else 0.0


def move_at(S1, k, L_, b, mins):
    t = S1["sec"][k] + 60 * mins
    if t >= 60 * NB:
        return np.nan
    i = int(np.searchsorted(S1["sec"], t, side="right")) - 1
    return float(b * (S1["c"][i] - L_))


def baseline_table(D, X=X1) -> pd.DataFrame:
    """C12: every 1-min close m (0 .. NB-2), both b, raced on 1-min bars from m + 1 (CR-CL `race`, vectorised)."""
    rows = []
    for d in D["SESS"]:
        B = D["BARS"][d]
        H, Lo, Cl = B["h"], B["l"], B["c"]
        hi_run, lo_run = np.maximum.accumulate(H), np.minimum.accumulate(Lo)
        m = np.arange(NB - 1)
        later = np.arange(NB)[None, :] > m[:, None]
        edge = np.where((Cl[m] >= hi_run[m] - EDGE_PTS) | (Cl[m] <= lo_run[m] + EDGE_PTS), "EDGE", "INTERIOR")
        for b in (1, -1):
            fav = (H[None, :] - Cl[m, None]) if b == 1 else (Cl[m, None] - Lo[None, :])
            adv = (Cl[m, None] - Lo[None, :]) if b == 1 else (H[None, :] - Cl[m, None])
            fw, aw = (fav >= X) & later, (adv >= X) & later
            i_f = np.where(fw.any(1), fw.argmax(1), NB + 1)
            i_a = np.where(aw.any(1), aw.argmax(1), NB + 1)
            res = np.where((i_f > NB) & (i_a > NB), "unresolved",
                           np.where(i_f < i_a, "win", np.where(i_a < i_f, "loss", "tie")))
            rows.append(pd.DataFrame(dict(d=d, m=m, h=b, edge=edge, res=res)))
    return pd.concat(rows, ignore_index=True)
