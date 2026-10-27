#!/usr/bin/env python3
"""CR-CP shared definitions — range edge, order flow picks the side. READ-ONLY, train only.

Spec: specs/CR-CP-range-edge-flow.md (D0-D5, C1-C18, Approvals). Imported by scripts/cr_cp_step0.py and the later run
script so both use one definition of every quantity.

  data (C3, C4, C17, C18): es_flow_1m_databento RTH minutes, the session's most-RTH-volume symbol; canonical IM;
                every read asserted >= 2026-01-01 and < 2026-06-05 21:00 UTC. Missing minutes = flat bars.
  legs (C1, C2, C5): flow_features.swing_legs at swing_d = 12 x IM_d / IM_train (A0: CR-CO's); a leg is an approach from the bar
                that confirms its start (price >= swing_d from the prior extreme) until the next leg's start is
                confirmed. One range per leg.
  range (C6), side (C7, C8), entry (C9), race (C10: CR-CL `race` plus the fill bar's adverse extreme).
  baseline (C13): entry at the close of minute m, stratum (m, h, EDGE).
  levels (C14): CR-CM `chart_levels` via cr_cm_lib.load("train"); skew (C15): CR-CL `read` on the approach leg.
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.flow_features import swing_legs  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2026, 1, 6), date(2026, 6, 5)
FLOOR, CAP = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
NB = 390                                    # RTH minutes 06:30 .. 12:59 PT
EARLY_LAST = 385                            # C4: last RTH bar before 12:55 PT -> early close
MISS_RUN_MAX = 5                            # C18
SWING_REF = 12.0                            # A7 / C2
EDGE_PTS = 2.0                              # C13
SEED, N_BOOT = 20260920, 10_000
SEED_AA, N_AA = 20261025, 200
MIN_TRADES = 40
OUT = REPO_ROOT / "scripts" / "logs" / "cr_cp"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class P:
    """One definition set. Primary = the defaults; each readout swaps one field (C12)."""
    swing: object = "vol"                   # "vol" (C2) or fixed pts
    width: object = 12.0                    # pts, or "half" = 0.5 x approach leg
    hold: int = 15                          # bars
    start_win: int = 5                      # r0 in [e, e + 5]
    void: float = 2.0
    rule: str = "delta"                     # "delta" (D2) or "cpp" (C12)
    thr: float = 0.02
    prog: float = 2.0
    cpp_ratio: float = 1.5
    dec_max: int = 60                       # decision bar <= r0 + 60
    touch: float = 1.0
    entry: str = "edge"                     # "edge" (D4) or "close" (decision-bar close)
    X: float = 10.0


PRIMARY = P()
VARIANTS = {
    "primary": PRIMARY,
    "swing_fixed12": replace(PRIMARY, swing=12.0), "swing_fixed8": replace(PRIMARY, swing=8.0),
    "swing_fixed16": replace(PRIMARY, swing=16.0),
    "width8": replace(PRIMARY, width=8.0), "width16": replace(PRIMARY, width=16.0),
    "width_half_leg": replace(PRIMARY, width="half"),
    "hold10": replace(PRIMARY, hold=10), "hold30": replace(PRIMARY, hold=30),
    "thr1": replace(PRIMARY, thr=0.01), "thr3": replace(PRIMARY, thr=0.03),
    "prog1": replace(PRIMARY, prog=1.0), "prog4": replace(PRIMARY, prog=4.0),
    "cpp_pushes": replace(PRIMARY, rule="cpp"),
    "entry_close": replace(PRIMARY, entry="close"),
}

_CONN = None


def conn():
    global _CONN
    if _CONN is None:
        from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
        _CONN = get_backfill_db_conn()
        assert_role_or_die(_CONN)
        _CONN.execute("SET default_transaction_read_only = on")
        _CONN.execute("SET statement_timeout = '1200s'")
        assert _CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"
    return _CONN


def q(sql, params=None) -> pd.DataFrame:
    cur = conn().execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def pt(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def guard_ts(lo, hi):
    assert FLOOR <= lo <= hi <= CAP, (lo, hi)


def sessions():
    out, d = [], D_FROM
    while d <= D_TO:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------------------------------------------- data
def load_minutes() -> pd.DataFrame:
    path = OUT / "m1_train.parquet"
    if path.exists():
        return pd.read_parquet(path)
    a, b = pt(D_FROM, 6, 30), pt(D_TO, 13, 0)
    guard_ts(a, b)
    r = q("""
        SELECT minute_utc, symbol, open::float AS o, high::float AS h, low::float AS l, close::float AS c,
               volume::float AS vol, buy_vol::float AS buy, sell_vol::float AS sell
        FROM es_flow_1m_databento
        WHERE minute_utc >= %s AND minute_utc < %s
          AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
          AND (minute_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        ORDER BY minute_utc, symbol
    """, (a, b))
    assert r["minute_utc"].min() >= a and r["minute_utc"].max() < b
    r.to_parquet(path)
    return r


IM_LO = date(2026, 1, 2)                    # A0: CR-CO's IM_train is the median over the CR-CJ y26 sessions


def load_im() -> pd.Series:
    guard_ts(pt(IM_LO, 0, 0), pt(D_TO, 13, 0))
    f = q("""
        SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
        WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
    """, (CANONICAL_FEATURE_VERSION, IM_LO, D_TO))
    f = f.drop_duplicates("trade_date").set_index("trade_date")["im"]
    return f


def im_train(im: pd.Series) -> float:
    """A0 / C2: as cr_cl_lib.vol_scaled_legs (CR-CO A1.1)."""
    import cr_cj_lib as CJ
    return float(im[im.index.isin(CJ.sessions("y26"))].dropna().median())


def build(m1: pd.DataFrame, im: pd.Series) -> dict:
    """Per-session (NB,) bars for the front symbol, the C4 / C18 funnel, and swing_d (C2)."""
    m1 = m1.copy()
    ts = pd.to_datetime(m1["minute_utc"], utc=True).dt.tz_convert(PT)
    m1["d"] = ts.dt.date
    m1["i"] = (ts.dt.hour * 60 + ts.dt.minute - 390).astype(int)
    vol = m1.groupby(["d", "symbol"])["vol"].sum().reset_index().sort_values(["d", "vol"], ascending=[True, False])
    SYM = {d: g.iloc[0]["symbol"] for d, g in vol.groupby("d")}
    BARS, MISS = {}, {}
    for d, g in m1.groupby("d"):
        g = g[g["symbol"] == SYM[d]]
        arr = {k: np.full(NB, np.nan) for k in ("o", "h", "l", "c", "vol", "buy", "sell")}
        for k in arr:
            arr[k][g["i"].to_numpy()] = g[k].to_numpy(float)
        have = np.isfinite(arr["c"])
        first, last = int(np.argmax(have)), int(NB - 1 - np.argmax(have[::-1]))
        miss = ~have
        miss[:first] = False
        run, best = 0, 0
        for x in miss[first:last + 1]:
            run = run + 1 if x else 0
            best = max(best, run)
        for j in range(first + 1, last + 1):              # C18 flat fill inside [first, last]
            if miss[j]:
                px = arr["c"][j - 1]
                for k in ("o", "h", "l", "c"):
                    arr[k][j] = px
                for k in ("vol", "buy", "sell"):
                    arr[k][j] = 0.0
        BARS[d] = arr
        MISS[d] = dict(first=first, last=last, n_missing=int(miss[first:last + 1].sum()), max_run=best)
    U, funnel = sessions(), []
    for nm, rule in [
        ("no Databento RTH flow", lambda d: d not in BARS),
        ("starts after 06:31 PT", lambda d: MISS[d]["first"] > 1),
        ("early close (last RTH bar < 12:55 PT)", lambda d: MISS[d]["last"] < EARLY_LAST),
        ("missing-minute run > 5 (C18)", lambda d: MISS[d]["max_run"] > MISS_RUN_MAX),
        ("no canonical IM / IM <= 0", lambda d: not (d in im.index and pd.notna(im[d]) and im[d] > 0)),
    ]:
        rm = [d for d in U if rule(d)]
        U = [d for d in U if not rule(d)]
        funnel.append([nm, len(rm), len(U), [str(x) for x in rm]])
    med = im_train(im)
    SW = {d: SWING_REF * float(im[d]) / med for d in U}
    return dict(SESS=U, BARS={d: BARS[d] for d in U}, SYM=SYM, MISS=MISS, IM={d: float(im[d]) for d in U},
                MED_IM=med, SW=SW, funnel=funnel)


# ------------------------------------------------------------------------------------------------ legs
def approach_legs(H, L, swing):
    """C1 / C5: every leg whose start is confirmed, with c_s (start confirmation) and c_next (the next leg's start
    confirmation, or NB). From swing_legs: leg k's start is confirmed at leg k-1's confirm_i; the first leg's at the
    first bar after its start that is >= swing beyond it; the leg in progress starts at the last leg's end."""
    legs = swing_legs(H, L, swing)
    out = []
    for k, lg in enumerate(legs):
        if k == 0:
            s = lg["start_i"]
            if lg["dir"] == 1:
                cs = s + 1 + int(np.argmax(H[s + 1:] >= lg["start_px"] + swing))
            else:
                cs = s + 1 + int(np.argmax(L[s + 1:] <= lg["start_px"] - swing))
        else:
            cs = legs[k - 1]["confirm_i"]
        out.append(dict(dir=int(lg["dir"]), start_i=int(lg["start_i"]), start_px=float(lg["start_px"]),
                        c_s=int(cs), c_next=int(lg["confirm_i"])))
    if legs:
        lg = legs[-1]
        out.append(dict(dir=-int(lg["dir"]), start_i=int(lg["end_i"]), start_px=float(lg["end_px"]),
                        c_s=int(lg["confirm_i"]), c_next=NB))
    return out


def swing_of(D, d, p: P):
    return D["SW"][d] if p.swing == "vol" else float(p.swing)


def find_range(B, lg, p: P):
    """C6: first qualifying bar k in [c_s, c_next) with r0 = k - hold + 1 in [e(k), e(k) + start_win] and box width
    <= p.width. Returns None or the range dict."""
    H, L = B["h"], B["l"]
    s, dr = lg["start_i"], lg["dir"]
    hold = p.hold
    best_px, e = (-np.inf if dr == 1 else np.inf), None
    for k in range(s, min(lg["c_next"], NB)):
        x = H[k] if dr == 1 else L[k]
        if (dr == 1 and x > best_px) or (dr == -1 and x < best_px):
            best_px, e = float(x), k
        r0 = k - hold + 1
        if k < lg["c_s"] or e is None or not (e <= r0 <= e + p.start_win):
            continue
        hi, lo = float(np.max(H[r0:k + 1])), float(np.min(L[r0:k + 1]))
        leg_pts = abs(best_px - lg["start_px"])
        wmax = 0.5 * leg_pts if p.width == "half" else float(p.width)
        if hi - lo <= wmax:
            return dict(dir=dr, start_i=s, c_s=lg["c_s"], e=e, ext_px=best_px, leg_pts=leg_pts, r0=r0, q=k,
                        hi=hi, lo=lo, width=hi - lo)
    return None


def side_call(B, r0, t, p: P):
    """C7 / C12 at bar t: (side, share, prog_buy) with side +1 = long, -1 = short, 0 = none."""
    sl = slice(r0, t + 1)
    vol = float(np.sum(B["vol"][sl]))
    net = float(np.sum(B["buy"][sl]) - np.sum(B["sell"][sl]))
    share = net / vol if vol > 0 else 0.0
    chg = float(B["c"][t] - B["o"][r0])                         # + = price advanced (buying direction)
    if p.rule == "delta":
        if share >= p.thr and chg <= p.prog:
            return -1, share, chg
        if share <= -p.thr and -chg <= p.prog:
            return 1, share, chg
        return 0, share, chg
    o, c, v = B["o"][sl], B["c"][sl], B["vol"][sl]
    up, dn = c > o, c < o
    cu = np.sum(v[up]) / np.sum(np.abs(c - o)[up]) if up.any() else np.nan
    cd = np.sum(v[dn]) / np.sum(np.abs(c - o)[dn]) if dn.any() else np.nan
    if np.isfinite(cu) and np.isfinite(cd) and cd > 0:
        rr = cu / cd
        if rr >= p.cpp_ratio and chg <= p.prog:
            return -1, share, chg
        if rr <= 1.0 / p.cpp_ratio and -chg <= p.prog:
            return 1, share, chg
    return 0, share, chg


def broken(B, j, rg, p: P):
    return B["h"][j] > rg["hi"] + p.void or B["l"][j] < rg["lo"] - p.void


def decide(B, rg, p: P):
    """C8: first t in [q, r0 + dec_max] with a side; a void (bar after q) first ends the range."""
    for t in range(rg["q"], min(rg["r0"] + p.dec_max, NB - 1) + 1):
        if t > rg["q"] and broken(B, t, rg, p):
            return dict(status="void before decision", t=t)
        side, share, chg = side_call(B, rg["r0"], t, p)
        if side != 0:
            return dict(status="call", t=t, side=side, share=share, prog=chg)
    t = min(rg["r0"] + p.dec_max, NB - 1)
    side, share, chg = side_call(B, rg["r0"], t, p)
    return dict(status="no call", t=t, side=0, share=share, prog=chg)


def fill(B, rg, t, h, p: P):
    """C9: h = +1 long at lo + touch, h = -1 short at hi - touch; first bar after t. Returns (status, j, price)."""
    if p.entry == "close":
        return ("fill", t, float(B["c"][t])) if t < NB - 1 else ("no fill: 13:00", None, None)
    px = rg["lo"] + p.touch if h == 1 else rg["hi"] - p.touch
    for j in range(t + 1, NB):
        touch = B["l"][j] <= px if h == 1 else B["h"][j] >= px
        far_break = (B["h"][j] > rg["hi"] + p.void) if h == 1 else (B["l"][j] < rg["lo"] - p.void)
        if far_break:                                          # broke the other edge (same-bar touch -> void)
            return "no fill: box broke", j, None
        if touch:
            return "fill", j, px
    return "no fill: 13:00", None, None


def edge_tag(B, j, px, h, include_j=False):
    """C13: EDGE if the entry is within 2 pts of the running RTH extreme on its side (bars before j, or through j)."""
    stop = j + 1 if include_j else j
    if h == -1:
        ref = max(float(np.max(B["h"][:stop])) if stop > 0 else -np.inf, px)
        return "EDGE" if px >= ref - EDGE_PTS else "INTERIOR"
    ref = min(float(np.min(B["l"][:stop])) if stop > 0 else np.inf, px)
    return "EDGE" if px <= ref + EDGE_PTS else "INTERIOR"


# ---------------------------------------------------------------------------------------------- outcomes
def race_from(B, j, px, h, X, fill_bar=True):
    """C10: CR-CL `race` from bar j + 1, plus (edge fills) the fill bar's adverse extreme as a loss."""
    import cr_cl_lib as CL
    if fill_bar:
        adv = (px - B["l"][j]) if h == 1 else (B["h"][j] - px)
        if adv >= X:
            return "loss"
    return CL.race(B, j, px, h, X)


def baseline_table(D, X=10.0) -> pd.DataFrame:
    """C13: every (session, minute m, h): entry at the close of m, race from m + 1, EDGE through m."""
    import cr_cl_lib as CL
    rows = []
    for d in D["SESS"]:
        B = D["BARS"][d]
        for m in range(0, NB - 1):
            for h in (1, -1):
                px = float(B["c"][m])
                rows.append((d, m, h, edge_tag(B, m, px, h, include_j=True), CL.race(B, m, px, h, X)))
    return pd.DataFrame(rows, columns=["d", "m", "h", "edge", "res"])


# ------------------------------------------------------------------------------------------- the schedule
def schedule(D, p: P, with_opposite=True) -> pd.DataFrame:
    """Every range under definition p with its decision and fills. No outcome is computed here."""
    rows = []
    for d in D["SESS"]:
        B = D["BARS"][d]
        sw = swing_of(D, d, p)
        legs = approach_legs(B["h"], B["l"], sw)
        n_rng = 0
        for lg in legs:
            rg = find_range(B, lg, p)
            if rg is None:
                continue
            n_rng += 1
            dc = decide(B, rg, p)
            row = dict(d=d, swing=sw, n_in_session=n_rng, **rg, dstatus=dc["status"], t=dc["t"],
                       side=dc.get("side", 0), share=dc.get("share"), prog=dc.get("prog"))
            for nm, h in (("ch", dc.get("side", 0)), ("op", -dc.get("side", 0))):
                if dc["status"] != "call" or (nm == "op" and not with_opposite):
                    row.update({f"{nm}_status": None, f"{nm}_j": None, f"{nm}_px": None, f"{nm}_edge": None})
                    continue
                st, j, px = fill(B, rg, dc["t"], h, p)
                row.update({f"{nm}_status": st, f"{nm}_j": j, f"{nm}_px": px,
                            f"{nm}_edge": edge_tag(B, j, px, h) if st == "fill" else None})
            if dc["status"] == "no call":                    # readout: no-trade ranges at both edges (after r0+60)
                for nm, h in (("nl", 1), ("ns", -1)):
                    st, j, px = fill(B, rg, dc["t"], h, replace(p, entry="edge"))
                    row.update({f"{nm}_status": st, f"{nm}_j": j, f"{nm}_px": px})
            rows.append(row)
    return pd.DataFrame(rows)


def leg_counts(D, p: P) -> dict:
    return {d: len(approach_legs(D["BARS"][d]["h"], D["BARS"][d]["l"], swing_of(D, d, p))) for d in D["SESS"]}


# --------------------------------------------------------------------------------------------- readouts
def load_levels():
    """C14: CR-CM's train load (chart lines at the RTH open, strong = p95, SETS['S'] within +-1 IM). Cached."""
    path = OUT / "levels_train.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    import cr_cm_lib as CM
    Dm = CM.load("train")
    lv = dict(USABLE=list(Dm.USABLE), CHART=Dm.CHART, STRONG=Dm.SETS["S"], OPEN=Dm.OPEN, IM=Dm.IM,
              funnel=Dm.funnel)
    path.write_bytes(pickle.dumps(lv))
    return lv


def tag_levels(S: pd.DataFrame, LV) -> pd.DataFrame:
    pos, strength, kind = [], [], []
    for r in S.itertuples():
        if r.d not in LV["STRONG"]:
            pos.append("no levels"), strength.append("no levels"), kind.append("no levels")
            continue
        open_, im = LV["OPEN"][r.d], LV["IM"][r.d]

        def dist(L_):
            return 0.0 if r.lo <= L_ <= r.hi else min(abs(L_ - r.lo), abs(L_ - r.hi))
        st = [x for x in LV["STRONG"][r.d] if dist(x["L"]) <= 10.0]
        disp = [x for x in LV["CHART"][r.d] if abs(x["L"] - open_) <= im and dist(x["L"]) <= 10.0]
        strength.append("strong" if st else ("weak" if disp else "none"))
        if not st:
            pos.append("no strong line"), kind.append("no strong line")
            continue
        x = min(st, key=lambda z: (dist(z["L"]), abs(z["L"] - 0.5 * (r.lo + r.hi))))
        L_ = x["L"]
        if r.lo <= L_ <= r.hi:
            pos.append("straddling")
        elif (L_ > r.hi) == (r.dir == 1):
            pos.append("short of")
        else:
            pos.append("beyond")
        kind.append("put" if x["net"] < 0 else "call")
    S = S.copy()
    S["lv_pos"], S["lv_strength"], S["lv_kind"] = pos, strength, kind
    return S


def tag_skew(S: pd.DataFrame) -> pd.DataFrame:
    """C15: CR-CL `read` on the approach leg, s = start + 1, e = leg end (running extreme at q) + 1."""
    import cr_cl_lib as CL
    Ds = CL.load()
    tags, g1 = [], []
    for r in S.itertuples():
        s, e = r.start_i + 1, r.e + 1
        if r.d not in Ds["MIN"] or r.d not in Ds["PX"]:
            tags.append("no skew data"), g1.append(False)
            continue
        if s < CL.DOM_LO or e > CL.LATE_END:
            tags.append("outside 07:00-12:45"), g1.append(False)
            continue
        rd = CL.read(Ds, r.d, s, e)
        if rd is None or rd.reason != "ok":
            tags.append("no read" if rd is None else rd.reason), g1.append(False)
            continue
        tags.append("beyond noise" if rd.beyond_noise else "in noise")
        g1.append(bool(rd.beyond_noise and rd.ratio < 0 and r.dir == -1 and e <= CL.PRIMARY_END))
    S = S.copy()
    S["skew"], S["g1down_like"] = tags, g1
    return S
