"""Shared order-flow building blocks (CR-CD A1.2) and the Flow v2 pane series (CR-CD L4/L5, A1.3).

Part A — CR-CB building blocks, copied unchanged from `scripts/cr_cb_step0_v2.py` in the CR-CB
worktree (sha256 954c7deeeac3b906bd2d4f4e46787225ba6da93c59aa4712d5026e1d362bb522). The arithmetic
is identical; the only edits are that CR-CB's module globals (BD, SIG5, the session list) are
explicit parameters. `apps/web/modules/Ironbeam/tests/test_flow_features.py` proves identical
outputs against a vendored copy of the CR-CB source. Inputs are 1-second flow arrays
F = {t (epoch s, int64), buy, sell, vol (int64), o, h, l, c (float)}, sorted by t.

Part B — the Flow v2 pane: rolling, causal, 1-minute series and markers. These are visual aids
(A1.2), not CR-CB's event features: they reuse Part A's primitives where the definitions match and
are otherwise the CR-CD A1 operational readings (spec, "A1 operational readings"). Every value at
bar t uses bars <= t only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc

# CR-CB constants (cr_cb_step0_v2.py:159, :164)
GAP_MAX, RETREAT = 15, 2.0
BASE_N, BASE_MIN, BASE_W, PUSH_W = 20, 5, 300, 120


# ============================================================================ Part A: CR-CB blocks
def pt(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def ep(x: datetime) -> int:
    return int(x.timestamp())


def space(F, side):
    return (F["h"], F["l"], F["c"], 1.0) if side == 1 else (-F["l"], -F["h"], -F["c"], -1.0)


def max_gap(F, d, t0, t1):
    """Largest RTH flow gap (s) within [t0, t1], clipped to [06:30, 13:00] PT."""
    f = F["t"]
    t0 = max(t0, ep(pt(d, 6, 30)))
    t1 = min(t1, ep(pt(d, 13, 0)))
    if t1 <= t0:
        return 0
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(np.diff(np.concatenate([[t0], f[a:b], [t1]])).max())


def flow_sum(F, t0, t1):
    """Net aggressor volume (buy - sell) and volume over [t0, t1)."""
    a, b = np.searchsorted(F["t"], t0), np.searchsorted(F["t"], t1)
    return int(F["buy"][a:b].sum() - F["sell"][a:b].sum()), int(F["vol"][a:b].sum()), a, b


def delta_share(net, vol):
    """Net aggressor volume as a share of volume; NaN when there is no volume."""
    return net / vol if vol > 0 else np.nan


def impact_windows(F, d):
    """Non-overlapping RTH 300 s windows with no flow gap > 15 s (CR-CB `windows`)."""
    t0 = ep(pt(d, 6, 30))
    for k in range(78):
        ws, we = t0 + BASE_W * k, t0 + BASE_W * (k + 1)
        a, b = np.searchsorted(F["t"], ws), np.searchsorted(F["t"], we)
        if a == 0 or b <= a or max_gap(F, d, ws, we) > GAP_MAX:
            continue
        yield a, b


def session_impact_samples(F, d):
    """Per-window |Δclose| and signed impact sign(net)·Δclose/(|net|/1000) (CR-CB `sess_stats`, flow part)."""
    dc, imp = [], []
    for a, b in impact_windows(F, d):
        dc.append(abs(F["c"][b - 1] - F["c"][a - 1]))
        net = int(F["buy"][a:b].sum() - F["sell"][a:b].sum())
        if net:
            imp.append(np.sign(net) * (F["c"][b - 1] - F["c"][a - 1]) / (abs(net) / 1000.0))
    return np.array(dc), np.array(imp)


def impact_baseline(samples, sessions, d):
    """σ5_d and B_d from the prior >= 5 (up to 20) sessions (CR-CB SIG5 / BD). None when too few.

    samples: {session: (dc, imp)}; sessions: the usable sessions, sorted.
    """
    prior = [x for x in sessions if x < d][-BASE_N:]
    if len(prior) < BASE_MIN:
        return None
    return (float(np.median(np.concatenate([samples[x][0] for x in prior]))),
            float(np.median(np.concatenate([samples[x][1] for x in prior]))))


def push_nets(F, set_i, ti, edge_px, tol, edge, side, real_dir):
    """Net aggressor volume of each push into an edge (CR-CB `edge_features`, push loop).

    A push starts when price is within tol of the edge and ends when it retreats more than
    tol + RETREAT from it; its net is over the PUSH_W seconds before its first bar.
    """
    t = F["t"]
    Hs, Los, _, _ = space(F, side)
    pushes, inpush = [], False
    for k in range(set_i, ti + 1):
        near = (Hs[k] >= edge_px - tol) if edge == "top" else (Los[k] <= edge_px + tol)
        far = (Hs[k] < edge_px - tol - RETREAT) if edge == "top" else (Los[k] > edge_px + tol + RETREAT)
        if near and not inpush:
            n_, _, _, _ = flow_sum(F, int(t[k]) - PUSH_W, int(t[k]))
            pushes.append(real_dir * n_)
            inpush = True
        elif far:
            inpush = False
    return pushes


def push_decay(pushes):
    """PD: OLS slope of push net on push index ÷ mean push net; NaN unless >= 2 pushes and mean > 0."""
    return (float(np.polyfit(np.arange(len(pushes)), pushes, 1)[0]) / np.mean(pushes)
            if len(pushes) >= 2 and np.mean(pushes) > 0 else np.nan)


def edge_flow_features(F, d, e, a, edge, B_d):
    """CR-CB A4.3 / A5.1 flow features at one edge event (short space; positive = toward the edge).

    e: {side, L}; a: {top_i|bot_i, hi_i|pl_i, H|PL, tol, c}; B_d: the session's impact baseline or None.
    Returns P, A_rel, Q, n_push, PD, range_s, LE, RA, DV, SS_share, SS_vwap.
    """
    side, L = e["side"], e["L"]
    _, _, C, sg = space(F, side)
    t = F["t"]
    ti = a["top_i"] if edge == "top" else a["bot_i"]
    te = int(t[ti])
    u = 1 if edge == "top" else -1
    real_dir = sg * u
    x = dict(edge=edge, real_edge="top" if real_dir == 1 else "bottom", t=te)
    net, vol, i0, _ = flow_sum(F, te - PUSH_W, te)
    x["P"] = real_dir * net
    mv = u * (C[ti - 1] - C[max(i0 - 1, 0)]) if ti > 0 else np.nan
    x["A_rel"] = (mv / (x["P"] / 1000.0) / B_d) if (x["P"] > 0 and B_d is not None) else np.nan
    set_i = a["hi_i"] if edge == "top" else a["pl_i"]
    ns, _, _, _ = flow_sum(F, int(t[set_i]) - PUSH_W, int(t[set_i]))
    pset = real_dir * ns
    x["Q"] = x["P"] / pset if pset > 0 else np.nan
    edge_px = a["H"] if edge == "top" else a["PL"]
    pushes = push_nets(F, set_i, ti, edge_px, a["tol"], edge, side, real_dir)
    x["n_push"] = len(pushes)
    x["PD"] = push_decay(pushes)
    tc = int(t[a["c"]])
    x["range_s"] = te - tc
    th = (te - tc) / 3.0
    n1, v1, _, _ = flow_sum(F, tc, tc + th)
    n3, v3, _, _ = flow_sum(F, te - th, te)
    x["LE"] = (real_dir * n3 / v3 - real_dir * n1 / v1) if v1 > 0 and v3 > 0 else np.nan
    nr, vr, ra, rb = flow_sum(F, tc, te)
    netr = real_dir * nr
    mvr = u * (C[ti - 1] - C[a["c"]]) if ti > a["c"] else 0.0
    x["RA"] = (mvr / (netr / 1000.0) / B_d) if (netr > 0 and B_d is not None) else np.nan
    x["DV"] = netr / vr if vr > 0 else np.nan
    seg = sg * F["c"][ra:rb]
    x["SS_share"] = float(np.mean(seg > sg * L)) if rb > ra else np.nan
    w = F["vol"][ra:rb]
    x["SS_vwap"] = (float(np.sum(seg * w) / w.sum()) - sg * L) if rb > ra and w.sum() else np.nan
    return x


# ============================================================================ Part B: Flow v2 pane
IMPACT_BARS = 5             # L4(d): rolling 5 min
ABSORB_PCT = 20             # L4(d): lowest 20 %
ABSORB_MIN_SHARE = 0.15     # L4(d): |delta share| >= 0.15
SHARE_SPAN = 3              # L4(c): EMA over 3 bars
TEST_BAND = 2.0             # A1.3: a test is within 2 pts of the swing
DECAY_RATIO = 0.6           # L5: push delta <= 0.6 x the previous
DIV_FRAC = 0.10             # L5: >= 10 % of the CVD range
FADE_BARS = 30              # L5: trailing 30 min, thirds of 10
PUSH_BARS = PUSH_W // 60    # CR-CB's 120 s push window at 1-minute resolution


@dataclass(frozen=True)
class MarkerParams:
    """Marker settings (A1.3 as amended by A2: defaults = Step 1 setting F; settable in the pane).

    A1 as written is MarkerParams(swing=4, min_prev_push=0, min_div_cvd=0, fade_min_share=0, gap_min=15).
    """
    swing: float = 8.0              # swing confirmed by a retrace of >= this many points
    min_prev_push: float = 500.0    # push decay: the previous push must exceed this net volume
    min_div_cvd: float = 300.0      # divergence: CVD differs from the swing by >= this (and >= 10 % of range)
    fade_min_share: float = 0.10    # fading: dominant side's trailing 30-min |delta share| >= this
    gap_min: int = 30               # at most one marker of a kind per this many minutes

    @property
    def test_retreat(self) -> float:
        return TEST_BAND + RETREAT  # between tests price retreats > 4 pts (2 + CR-CB's 2-pt retreat)


A1_AS_WRITTEN = MarkerParams(swing=4.0, min_prev_push=0.0, min_div_cvd=0.0, fade_min_share=0.0, gap_min=15)

COLUMNS = ("o", "h", "l", "c", "buy", "sell", "vol")


def minutes_from_seconds(F) -> pd.DataFrame:
    """Flow-built 1-minute bars from 1-second arrays: first open, max high, min low, last close, sums."""
    df = pd.DataFrame({k: F[k] for k in ("o", "h", "l", "c", "buy", "sell", "vol")},
                      index=(F["t"] // 60) * 60)
    g = df.groupby(level=0)
    out = pd.DataFrame({"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last(),
                        "buy": g["buy"].sum().astype(float), "sell": g["sell"].sum().astype(float),
                        "vol": g["vol"].sum().astype(float)})
    out.index = out.index.astype(np.int64)
    return out


def complete_minutes(rows: pd.DataFrame, start_epoch: int, end_epoch: int) -> pd.DataFrame:
    """Reindex flow-built minutes to every minute of [start, end); quiet minutes get zero volume and
    the prior close as O/H/L/C. rows: index = bar-start epoch seconds, columns COLUMNS."""
    idx = np.arange(int(start_epoch), int(end_epoch), 60, dtype=np.int64)
    m = rows.reindex(idx)
    m[["buy", "sell", "vol"]] = m[["buy", "sell", "vol"]].fillna(0.0)
    m["c"] = m["c"].ffill()
    for k in ("o", "h", "l"):
        m[k] = m[k].fillna(m["c"])
    m.index.name = "time"
    return m


def rolling_abs_impact(m: pd.DataFrame) -> pd.Series:
    """|Δclose over 5 bars| ÷ (|net over the same 5 bars| / 1000); NaN where net is 0."""
    net5 = (m["buy"] - m["sell"]).rolling(IMPACT_BARS).sum()
    mv5 = (m["c"] - m["c"].shift(IMPACT_BARS)).abs()
    return (mv5 / (net5.abs() / 1000.0)).where(net5 != 0)


def absorption_baseline(session_minutes: list[pd.DataFrame]) -> dict | None:
    """Median and p20 of the pooled rolling 5-min impact over prior sessions' RTH minutes (D4)."""
    pool = pd.concat([rolling_abs_impact(m).dropna() for m in session_minutes]) if session_minutes else None
    if pool is None or pool.empty:
        return None
    return {"abs_median": float(pool.median()), "abs_p20": float(np.percentile(pool, ABSORB_PCT)),
            "n_sessions": len(session_minutes), "n_windows": int(len(pool))}


def pane_series(m: pd.DataFrame, anchor_epoch: int, baseline: dict | None, absorption: bool) -> pd.DataFrame:
    """Delta, CVD (anchored at anchor_epoch), delta share (EMA 3) and, if asked, absorption."""
    out = pd.DataFrame(index=m.index)
    out["buy"], out["sell"], out["vol"] = m["buy"], m["sell"], m["vol"]
    out["delta"] = m["buy"] - m["sell"]
    out["cvd"] = out["delta"].where(m.index >= anchor_epoch, 0.0).cumsum()
    raw = (out["delta"] / m["vol"]).where(m["vol"] > 0)
    out["share"] = raw.ewm(span=SHARE_SPAN, adjust=False, ignore_na=True).mean()
    out["imp_rel"] = np.nan
    out["absorb"] = False
    if absorption and baseline:
        imp = rolling_abs_impact(m)
        out["imp_rel"] = imp / baseline["abs_median"]
        out["absorb"] = ((imp <= baseline["abs_p20"]) & (out["share"].abs() >= ABSORB_MIN_SHARE)).fillna(False)
    return out[out.index >= anchor_epoch]


def _n(x) -> str:
    return f"{abs(int(round(x))):,}"


def divergence_markers(H, L, cvd, times, start_i, p: MarkerParams):
    """A1.3 divergence against the last confirmed swing (>= 4-pt retrace), one per swing."""
    out = []
    for sg in (1, -1):                       # 1: swing lows / bullish; -1: swing highs / bearish
        X = L if sg == 1 else -H
        Y = H if sg == 1 else -L
        run, run_i, swing = np.inf, None, None
        for i in range(start_i, len(times)):
            if swing is not None and not swing["fired"] and X[i] < swing["px"]:
                k = swing["i"]
                rng = cvd[k:i + 1].max() - cvd[k:i + 1].min()
                if rng > 0 and sg * (cvd[i] - cvd[k]) >= max(DIV_FRAC * rng, p.min_div_cvd):
                    swing["fired"] = True
                    px = sg * swing["px"]
                    if sg == 1:
                        text = (f"Sellers fading: price made a new low below the {px:.2f} swing low, but "
                                f"cumulative delta is {_n(cvd[i] - cvd[k])} higher than at that low")
                    else:
                        text = (f"Buyers fading: price made a new high above the {px:.2f} swing high, but "
                                f"cumulative delta is {_n(cvd[i] - cvd[k])} lower than at that high")
                    out.append(dict(i=i, kind="divergence", side="bull" if sg == 1 else "bear", text=text,
                                    detail={"swing_time": int(times[k]), "swing_price": px,
                                            "cvd_change": float(cvd[i] - cvd[k]), "cvd_range": float(rng)}))
            if X[i] < run:
                run, run_i = X[i], i
            if run_i is not None and run_i < i and Y[i] >= run + p.swing:
                swing = {"px": run, "i": run_i, "fired": False}
                run, run_i = np.inf, None
    return out


def push_decay_markers(H, L, delta, times, start_i, p: MarkerParams):
    """A1.3 push decay at repeated tests (within 2 pts) of the last confirmed swing high/low."""
    out = []
    for sg in (1, -1):                       # 1: swing highs (buyers pushing up); -1: swing lows
        X = H if sg == 1 else -L             # extreme in the push direction
        Y = L if sg == 1 else -H             # retrace side
        run, run_i = -np.inf, None
        sw = None
        for i in range(start_i, len(times)):
            if sw is not None:
                if X[i] > sw["px"] + TEST_BAND:
                    sw = None
                elif X[i] >= sw["px"] - TEST_BAND and sw["armed"]:
                    push = sg * float(delta[max(i - PUSH_BARS, 0):i].sum())
                    prev = sw["push"]
                    if prev > max(0.0, p.min_prev_push) and push <= DECAY_RATIO * prev:
                        who, where = ("Buyers", "high") if sg == 1 else ("Sellers", "low")
                        out.append(dict(i=i, kind="push_decay", side="bear" if sg == 1 else "bull",
                                        text=(f"{who} weaker on the retest of the {sg * sw['px']:.2f} swing "
                                              f"{where}: {_n(push) if push > 0 else 'no'} net "
                                              f"{'buying' if sg == 1 else 'selling'} into it, vs "
                                              f"{_n(prev)} on the previous push"),
                                        detail={"swing_price": sg * sw["px"], "push": push, "prev_push": prev}))
                    sw["push"], sw["armed"] = push, False
                if sw is not None and Y[i] < sw["px"] - p.test_retreat:
                    sw["armed"] = True
            if X[i] > run:
                run, run_i = X[i], i
            if run_i is not None and run_i < i and Y[i] <= run - p.swing:
                k = run_i
                sw = {"px": run, "i": k, "armed": Y[i] < run - p.test_retreat,
                      "push": sg * float(delta[max(k - PUSH_BARS, 0):k].sum())}
                run, run_i = -np.inf, None
    return out


def fading_markers(delta, vol, start_i, p: MarkerParams):
    """L5 fading: the dominant side's last-third minus first-third delta share over the trailing
    30 bars crosses below 0."""
    out, prev = [], None
    for i in range(start_i + FADE_BARS - 1, len(delta)):
        a, b = slice(i - 29, i - 19), slice(i - 9, i + 1)
        if vol[a].sum() <= 0 or vol[b].sum() <= 0:
            prev = None
            continue
        s = np.sign(delta[i - 29:i + 1].sum()) or 1.0
        if abs(delta[i - 29:i + 1].sum()) < p.fade_min_share * vol[i - 29:i + 1].sum():
            prev = None
            continue
        le = s * (delta[b].sum() / vol[b].sum() - delta[a].sum() / vol[a].sum())
        if prev is not None and prev >= 0 > le:
            who = "Buyers" if s > 0 else "Sellers"
            out.append(dict(i=i, kind="fading", side="bear" if s > 0 else "bull",
                            text=(f"{who} fading: their share of volume in the last 10 minutes is lower "
                                  f"than in the first 10 of the last 30"),
                            detail={"late_minus_early": float(le)}))
        prev = le
    return out


def cap_markers(marks, gap_min: int):
    """D3: at most one marker of a kind per gap_min minutes (earliest kept)."""
    kept, last = [], {}
    for mk in sorted(marks, key=lambda x: (x["i"], x["kind"])):
        if mk["i"] - last.get(mk["kind"], -10**9) >= gap_min:
            kept.append(mk)
            last[mk["kind"]] = mk["i"]
    return kept


def pane_markers(m: pd.DataFrame, series: pd.DataFrame, start_epoch: int,
                 p: MarkerParams = MarkerParams()) -> list[dict]:
    """Divergence, push decay and fading on bars >= start_epoch (windows never reach earlier)."""
    sel = m.index >= start_epoch
    mm, ss = m[sel], series.reindex(m.index[sel])
    times = mm.index.to_numpy()
    H, L = mm["h"].to_numpy(float), mm["l"].to_numpy(float)
    delta, vol, cvd = ss["delta"].to_numpy(float), mm["vol"].to_numpy(float), ss["cvd"].to_numpy(float)
    marks = (divergence_markers(H, L, cvd, times, 0, p) + push_decay_markers(H, L, delta, times, 0, p)
             + fading_markers(delta, vol, 0, p))
    return [dict(time=int(times[mk["i"]]), kind=mk["kind"], side=mk["side"], text=mk["text"],
                 detail=mk["detail"]) for mk in cap_markers(marks, p.gap_min)]
