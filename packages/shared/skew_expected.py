"""Fitted expected curve for intraday 0DTE SPX put skew and ATM vol vs the ES move (CR-CJ).

Pure functions, no I/O. Not wired into the Dash chart (CR-CJ K9); the Skew module's sticky-strike
"Expected SS" block (`apps/web/modules/Skew/callbacks.py`) is unchanged.

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
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Sequence, Union

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
