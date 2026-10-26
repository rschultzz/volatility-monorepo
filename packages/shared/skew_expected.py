"""Fitted expected curve for intraday 0DTE SPX put skew and ATM vol vs the ES move (CR-CJ).

Pure functions, no I/O. The fitted curves are not wired into the Dash chart (CR-CJ K9); the Skew module's
sticky-strike "Expected SS" block (`apps/web/modules/Skew/callbacks.py`) is unchanged. The CR-CK stickiness functions
back the Overview panel's optional "Stickiness" rows (`apps/web/modules/Ironbeam/skew_stickiness.py`).

Inputs, as CR-CJ measured them:
  put skew  = (vol75 - vol50) * 100 vol pts on the 0DTE PM expiry (`orats_monies_minute`; vol75 = 25-delta put)
  ATM       = vol50 * 100 vol pts on the same row
  ES move   = 100 * (es_now / es_prev - 1) %, ES front-month close at the snapshot minute (Databento)
  times     = snapshot minutes, PT wall clock; RTH is 06:30-13:00 PT

Model, vol pts, over a window from snapshot t_prev to t_now:
  expected change = beta(window, sign of move) * ES % move + sum over 30-min buckets b of theta_b * minutes in b
where beta(window) is the minute-weighted mean of the per-bucket betas the window spans. The residual is the
actual change minus the expected change. CR-CJ fitted it on 07:00-12:45 PT (`in_domain`); outside that the
parameters are extrapolated (open-time ORATS quirks, 0DTE degenerating near the close).

Standing ruling A6 read: price up with put skew rising vs expected, or price down with it falling, is
"divergent"; the opposite is "confirming".

Stickiness ratio (CR-CK, standing ruling A14 as rewritten after CR-CJ), over a window [t_start, t_end]:
  actual   = OLS slope of put skew on snapshot minute x window length (A16: a trend, not endpoints)
  expected = sticky-strike change for the same ES move: the start smile held fixed in strike, spot moved by the
             ES move, T held at the start (so expected = 0 for no move)
  ratio    = actual / expected, only when |ES move| >= 0.15 %
  noise    = |actual| > the endpoint-change SD for that window length and start bucket (2026 train, CR-CK C6)
Reading: ~0 normal re-forming; ~1 skew relaxed as sticky strike implies; < 0 put skew steepened against the move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from typing import Mapping, Optional, Sequence, Union

from packages.shared.surface_compare import k_for_abs_delta

RTH_OPEN_MIN = 6 * 60 + 30            # 06:30 PT, minutes after midnight
RTH_CLOSE_MIN = 13 * 60               # 13:00 PT
BUCKET_MIN = 30
N_BUCKETS = (RTH_CLOSE_MIN - RTH_OPEN_MIN) // BUCKET_MIN      # 13: 06:30, 07:00, ..., 12:30
DOMAIN_START_MIN = 7 * 60             # fit domain 07:00 .. 12:45 PT
DOMAIN_END_MIN = 12 * 60 + 45

TimeLike = Union[time, datetime, int]


@dataclass(frozen=True)
class CurveParams:
    """Per 30-min bucket (06:30 first): beta for up and down ES moves (vol pts per 1 % ES move) and the
    drift theta (vol pts per minute)."""
    beta_up: Sequence[float]
    beta_down: Sequence[float]
    theta: Sequence[float]

    def __post_init__(self):
        for f in (self.beta_up, self.beta_down, self.theta):
            if len(f) != N_BUCKETS:
                raise ValueError(f"need {N_BUCKETS} buckets, got {len(f)}")


def _flat(x: float) -> tuple:
    return (x,) * N_BUCKETS


# CR-CJ recommendation (A2.4): expected put-skew change = 0. The median put-skew response to the ES move is about 0
# (LAD beta -0.32 vol pts per 1 %) and flips sign between subsamples (2026 H1 +0.58, H2 -0.65), and no fitted curve
# stayed balanced out of sample. So the residual is the raw change. (The Skew module's sticky-strike curve implies
# -5.7 and reads "divergent" on 76-78 % of windows and legs.)
PUT_SKEW_PARAMS = CurveParams(beta_up=_flat(0.0), beta_down=_flat(0.0), theta=_flat(0.0))

# Reference only, NOT recommended: the A2 median fit on all 293 sessions (swing-12 legs, joint drift), balanced in
# sample (0.50 / 0.51) but 0.43 on legs out of sample. theta[0] (06:30) is copied from 07:00 (out of domain).
PUT_SKEW_LAD_A2_PARAMS = CurveParams(
    beta_up=_flat(-0.316), beta_down=_flat(-0.316),
    theta=(+0.00129, +0.00129, -0.00376, +0.00478, -0.00297, +0.00768, -0.00579,
           +0.00188, +0.00481, -0.00484, +0.00894, -0.02665, -0.02410),
)

# ATM vol50: A2 median fit on all 293 sessions; beta stable across subsamples (-5.8 .. -9.0). Reported, not gated
# by an out-of-sample rule (CR-CJ A2.6). The Skew module's curve implies about -9.4 (sticky strike plus the 4.5 add-on).
ATM_PARAMS = CurveParams(
    beta_up=_flat(-7.017), beta_down=_flat(-7.017),
    theta=(-0.03389, -0.03389, -0.02433, -0.01232, -0.01651, +0.00407, +0.00967,
           -0.00797, +0.00360, -0.00480, -0.00680, -0.13465, -0.34396),
)


def minute_of_day(t: TimeLike) -> int:
    """PT wall-clock minute after midnight; an int is taken as already in minutes."""
    if isinstance(t, int):
        return t
    if isinstance(t, datetime):
        t = t.time()
    return t.hour * 60 + t.minute


def minutes_by_bucket(t_prev: TimeLike, t_now: TimeLike) -> list[int]:
    """Minutes of the window spent in each 30-min RTH bucket; minute m -> m + 1 counts in the bucket of m.
    Minutes outside 06:30-13:00 are dropped."""
    a, b = minute_of_day(t_prev), minute_of_day(t_now)
    if b < a:
        raise ValueError("t_now is before t_prev")
    out = [0] * N_BUCKETS
    for m in range(max(a, RTH_OPEN_MIN), min(b, RTH_CLOSE_MIN)):
        out[(m - RTH_OPEN_MIN) // BUCKET_MIN] += 1
    return out


def in_domain(t_prev: TimeLike, t_now: TimeLike) -> bool:
    """True when the window lies inside the CR-CJ fit domain (07:00 <= t_prev, t_now <= 12:45 PT)."""
    return DOMAIN_START_MIN <= minute_of_day(t_prev) <= minute_of_day(t_now) <= DOMAIN_END_MIN


def es_move_pct(es_prev: float, es_now: float) -> float:
    return 100.0 * (es_now / es_prev - 1.0)


def expected_change_pp(ret_pct: float, t_prev: TimeLike, t_now: TimeLike, params: CurveParams) -> float:
    """Expected change (vol pts) over the window for an ES move of ret_pct %."""
    w = minutes_by_bucket(t_prev, t_now)
    n = sum(w)
    if n == 0:
        return 0.0
    betas = params.beta_up if ret_pct >= 0 else params.beta_down
    beta = sum(wi * bi for wi, bi in zip(w, betas)) / n
    drift = sum(wi * th for wi, th in zip(w, params.theta))
    return beta * ret_pct + drift


def residual_pp(prev_pp: float, now_pp: float, es_prev: float, es_now: float,
                t_prev: TimeLike, t_now: TimeLike, params: CurveParams) -> float:
    """Actual change minus expected change, vol pts."""
    return (now_pp - prev_pp) - expected_change_pp(es_move_pct(es_prev, es_now), t_prev, t_now, params)


def put_skew_residual_pp(put_skew_prev_pp: float, put_skew_now_pp: float, es_prev: float, es_now: float,
                         t_prev: TimeLike, t_now: TimeLike, params: CurveParams | None = None) -> float:
    return residual_pp(put_skew_prev_pp, put_skew_now_pp, es_prev, es_now, t_prev, t_now, params or PUT_SKEW_PARAMS)


def atm_residual_pp(atm_prev_pp: float, atm_now_pp: float, es_prev: float, es_now: float,
                    t_prev: TimeLike, t_now: TimeLike, params: CurveParams | None = None) -> float:
    return residual_pp(atm_prev_pp, atm_now_pp, es_prev, es_now, t_prev, t_now, params or ATM_PARAMS)


def divergence_read(put_skew_residual: float, es_move: float) -> str:
    """A6: 'divergent' when the residual moves with price, 'confirming' when against, 'none' if either is 0."""
    if put_skew_residual == 0 or es_move == 0 or put_skew_residual != put_skew_residual:
        return "none"
    return "divergent" if (put_skew_residual > 0) == (es_move > 0) else "confirming"


# ------------------------------------------------------------------------------------------------ stickiness (CR-CK)
MIN_ES_MOVE_PCT = 0.15                # ratio defined only for |ES move| >= 0.15 % (A14 / CR-CK K2)
EPS_T = 1e-4                          # T floor, years (as the Skew module)
TREND_MIN_COVERAGE = 0.8              # CR-CK C3: >= 80 % of the window's minutes carry a snapshot
LATE_DAY_MIN = 11 * 60                # CR-CK A1.7: a window ending after 11:00 PT is flagged low reliability
TREND_MIN_POINTS = 5

# CR-CK C6: SD of the endpoint change in put skew (vol pts) by window length (min) and 30-min start bucket (index from
# 06:30; 1 = 07:00 .. 12 = 12:30), 2026 train (01-02 .. 06-05), windows inside 07:00-12:45 PT on a 5-min start grid.
# Lengths a bucket cannot reach before 12:45 (or with < 30 windows) carry its longest measured value. The 06:30 bucket
# is not measured (no flag). Source: scripts/cr_ck_stickiness.py -> scripts/logs/cr_ck/results.json.
NOISE_BAND_LENGTHS: tuple = (5, 10, 15, 20, 30, 45, 60, 90, 120)
NOISE_BAND_PP: Optional[Mapping[int, tuple]] = {
     1: (0.727, 0.904, 1.127, 1.255, 1.259, 1.051, 1.072, 1.047, 1.015),   # 07:00
     2: (0.889, 1.220, 1.255, 1.174, 1.186, 1.206, 1.213, 1.198, 1.394),   # 07:30
     3: (0.654, 0.804, 0.853, 0.925, 0.876, 0.903, 0.882, 1.207, 1.289),   # 08:00
     4: (0.572, 0.742, 0.823, 0.822, 0.816, 0.888, 1.180, 1.226, 1.060),   # 08:30
     5: (0.583, 0.699, 0.793, 0.985, 1.109, 1.197, 1.170, 1.051, 1.270),   # 09:00
     6: (0.908, 1.105, 1.436, 1.414, 1.361, 1.343, 1.334, 1.528, 2.590),   # 09:30
     7: (0.911, 1.199, 1.059, 1.110, 1.194, 1.253, 1.408, 2.579, 3.022),   # 10:00
     8: (0.787, 0.915, 0.940, 1.177, 1.312, 2.627, 2.534, 2.919, 4.484),   # 10:30
     9: (1.433, 1.549, 2.559, 2.480, 2.502, 2.253, 3.016, 4.598, 4.598),   # 11:00
    10: (2.973, 2.576, 3.045, 3.065, 3.691, 4.707, 5.229, 5.229, 5.229),   # 11:30
    11: (3.140, 4.764, 5.131, 5.057, 5.029, 1.467, 1.467, 1.467, 1.467),   # 12:00
    12: (5.342, 6.545, 8.831, 8.831, 8.831, 8.831, 8.831, 8.831, 8.831),   # 12:30
}


def years_to_close_0dte(t: TimeLike) -> float:
    """T in years to 13:00 PT (16:00 ET) on a 365-day clock, floored at EPS_T, from the snapshot minute."""
    return max((RTH_CLOSE_MIN - minute_of_day(t)) / (365.0 * 24 * 60), EPS_T)


def _abs_delta_is_put(n: int) -> tuple:
    if n > 50:
        return (100 - n) / 100.0, True
    return n / 100.0, False


def smile_k_grid(vols: Mapping[int, float], T: float) -> tuple:
    """(k, sigma) arrays of a smile given as {call-delta bucket n: vol (fraction)}, n in 1..99 with 50 = ATM.
    Delta -> log-moneyness with the smile's own ATM vol and T (`k_for_abs_delta`), the same map as the Skew module's
    `_prev_smile_interp`; nodes whose k does not increase are dropped."""
    if 50 not in vols:
        raise ValueError("smile needs the ATM (50) bucket")
    atm = float(vols[50])
    puts = sorted((n for n in vols if n >= 50), reverse=True)
    calls = sorted((n for n in vols if n < 50), reverse=True)
    k, s = [], []
    for n in puts + calls:
        v = vols[n]
        if v is None or v != v:
            continue
        if n == 50:
            kn = 0.0
        else:
            p, is_put = _abs_delta_is_put(n)
            kn = k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        if k and kn <= k[-1] + 1e-12:
            continue
        k.append(kn)
        s.append(float(v))
    if len(k) < 3:
        raise ValueError("smile k-grid degenerate")
    return k, s


def interp_linear_extrap(kq: float, k: Sequence[float], s: Sequence[float]) -> float:
    """Linear interpolation inside the grid, linear extrapolation from the end segments (the Skew module's rule)."""
    if kq <= k[0]:
        return s[0] + (s[1] - s[0]) * (kq - k[0]) / (k[1] - k[0])
    if kq >= k[-1]:
        return s[-1] + (s[-1] - s[-2]) * (kq - k[-1]) / (k[-1] - k[-2])
    for i in range(1, len(k)):
        if kq <= k[i]:
            return s[i - 1] + (s[i] - s[i - 1]) * (kq - k[i - 1]) / (k[i] - k[i - 1])
    return s[-1]


def sticky_strike_put_skew_change_pp(vols: Mapping[int, float], T: float, es_move_pct: float) -> float:
    """Sticky-strike change in put skew (vol pts) for an ES move of es_move_pct %, from the start smile `vols`
    ({call-delta bucket: vol fraction}, needs 50 and 75) at time to expiry T (years). The smile stays fixed in strike;
    the new ATM sits at k = ln(1 + move) and the new 25-delta put at k_p25 + ln(1 + move); T is held at the start."""
    k, s = smile_k_grid(vols, T)
    k_p25 = k_for_abs_delta(0.25, is_put=True, sigma=float(vols[50]), T=T)
    k_shift = math.log1p(es_move_pct / 100.0)
    before = interp_linear_extrap(k_p25, k, s) - interp_linear_extrap(0.0, k, s)
    after = interp_linear_extrap(k_p25 + k_shift, k, s) - interp_linear_extrap(k_shift, k, s)
    return (after - before) * 100.0


def trend_change_pp(times: Sequence[TimeLike], values: Sequence[float], t_start: TimeLike, t_end: TimeLike,
                    min_coverage: float = TREND_MIN_COVERAGE, min_points: int = TREND_MIN_POINTS) -> Optional[float]:
    """OLS slope of `values` on minute over the snapshots in [t_start, t_end] (inclusive), times the window length in
    minutes. None when the window is empty or the snapshots cover < min_coverage of its minutes or < min_points."""
    a, b = minute_of_day(t_start), minute_of_day(t_end)
    if b <= a:
        return None
    pts = {}
    for t, v in zip(times, values):
        m = minute_of_day(t)
        if a <= m <= b and v is not None and v == v and m not in pts:
            pts[m] = float(v)
    n = len(pts)
    if n < min_points or n < min_coverage * (b - a + 1):
        return None
    xs = list(pts)
    mx = sum(xs) / n
    my = sum(pts.values()) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (pts[x] - my) for x in xs)
    return sxy / sxx * (b - a)


def noise_band_pp(length_min: float, t_start: TimeLike, table: Optional[Mapping[int, tuple]] = None) -> Optional[float]:
    """Endpoint-change SD (vol pts) for a window of length_min starting at t_start: linear in length within the start's
    30-min bucket, clamped to the table's shortest and longest lengths. None outside the measured buckets."""
    table = NOISE_BAND_PP if table is None else table
    if table is None:
        return None
    b = (minute_of_day(t_start) - RTH_OPEN_MIN) // BUCKET_MIN
    row = table.get(b)
    if row is None:
        return None
    Ls = NOISE_BAND_LENGTHS
    x = min(max(float(length_min), Ls[0]), Ls[-1])
    for i in range(1, len(Ls)):
        if x <= Ls[i]:
            w = (x - Ls[i - 1]) / (Ls[i] - Ls[i - 1])
            return row[i - 1] + w * (row[i] - row[i - 1])
    return row[-1]


@dataclass(frozen=True)
class StickinessRead:
    actual_pp: Optional[float]          # trend change in put skew, vol pts
    expected_pp: Optional[float]        # sticky-strike change for the same ES move, vol pts
    es_move_pct: float
    ratio: Optional[float]              # actual / expected, None unless reason == "ok"
    reason: str                         # "ok" | "move too small" | "insufficient data"
    band_pp: Optional[float]
    beyond_noise: Optional[bool]        # |actual| > band; None when either is unknown
    late_day: bool = False              # window ends after 11:00 PT (noise band 2-9 vol pts; A1.7)


def stickiness_read(times: Sequence[TimeLike], put_skew_pp: Sequence[float], t_start: TimeLike, t_end: TimeLike,
                    start_vols: Mapping[int, float], es_start: float, es_end: float,
                    band_table: Optional[Mapping[int, tuple]] = None) -> StickinessRead:
    """A14 stickiness over [t_start, t_end]: put skew series (vol pts) by snapshot minute, the start snapshot's smile
    ({call-delta bucket: vol fraction}) and the ES closes at the two endpoints (quote time)."""
    r = es_move_pct(es_start, es_end)
    actual = trend_change_pp(times, put_skew_pp, t_start, t_end)
    try:
        expected = sticky_strike_put_skew_change_pp(start_vols, years_to_close_0dte(t_start), r)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        expected = None
    band = noise_band_pp(minute_of_day(t_end) - minute_of_day(t_start), t_start, band_table)
    beyond = None if (actual is None or band is None) else abs(actual) > band
    late = minute_of_day(t_end) > LATE_DAY_MIN
    if actual is None or expected is None:
        return StickinessRead(actual, expected, r, None, "insufficient data", band, beyond, late)
    if abs(r) < MIN_ES_MOVE_PCT or expected == 0:
        return StickinessRead(actual, expected, r, None, "move too small", band, beyond, late)
    return StickinessRead(actual, expected, r, actual / expected, "ok", band, beyond, late)
