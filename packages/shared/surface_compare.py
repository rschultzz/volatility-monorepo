
# surface_compare.py
from dataclasses import dataclass
from math import exp, log, sqrt, erf
from typing import Optional, Dict

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_ppf(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = sqrt(-2*log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif phigh < p:
        q = sqrt(-2*log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

@dataclass
class TimeSlice:
    S: float
    r: float
    q: float
    T: float
    sigma0: float
    slope: float
    curvature: float = 0.0
    @property
    def F(self) -> float:
        return self.S * exp((self.r - self.q) * self.T)

def sigma_at_k(sigma0: float, slope: float, curvature: float, k: float) -> float:
    return sigma0 + slope * k + 0.5 * curvature * k * k

def predict_atm_iv(prev: TimeSlice, new_forward: float, mode: str = "sticky_strike") -> float:
    if mode not in ("sticky_strike", "sticky_moneyness", "sticky_delta"):
        raise ValueError("mode must be one of 'sticky_strike', 'sticky_moneyness', 'sticky_delta'")
    if mode == "sticky_strike":
        k_shift = log(new_forward / prev.F)
        return sigma_at_k(prev.sigma0, prev.slope, prev.curvature, k_shift)
    else:
        return prev.sigma0

def expected_vs_actual_atm(prev: TimeSlice, new_slice: TimeSlice) -> Dict[str, float]:
    F_new = new_slice.F
    pred_ss  = predict_atm_iv(prev, F_new, mode="sticky_strike")
    pred_sm  = predict_atm_iv(prev, F_new, mode="sticky_moneyness")
    pred_sd  = predict_atm_iv(prev, F_new, mode="sticky_delta")
    actual   = new_slice.sigma0
    return {
        "F_prev": prev.F,
        "F_new": F_new,
        "atm_prev": prev.sigma0,
        "atm_actual": actual,
        "pred_atm_sticky_strike": pred_ss,
        "pred_atm_sticky_moneyness": pred_sm,
        "pred_atm_sticky_delta": pred_sd,
        "residual_sticky_strike": actual - pred_ss,
        "residual_sticky_moneyness": actual - pred_sm,
        "residual_sticky_delta": actual - pred_sd,
    }

def smile_decomposition(prev: TimeSlice, new_slice: TimeSlice) -> Dict[str, float]:
    return {
        "d_level_atm": new_slice.sigma0 - prev.sigma0,
        "d_slope": new_slice.slope - prev.slope,
        "d_convexity": new_slice.curvature - prev.curvature
    }

def fit_smile_quadratic_from_points(
    k_atm: float,
    sigma_atm: float,
    k_put: float,
    sigma_put: float,
    k_call: float,
    sigma_call: float
) -> Dict[str, float]:
    import numpy as np
    M = np.array([
        [1.0, k_atm, 0.5 * k_atm * k_atm],
        [1.0, k_put, 0.5 * k_put * k_put],
        [1.0, k_call, 0.5 * k_call * k_call]
    ], dtype=float)
    y = np.array([sigma_atm, sigma_put, sigma_call], dtype=float)
    a, b, halfc = np.linalg.solve(M, y)
    c = 2.0 * halfc
    return {"sigma0": a, "slope": b, "curvature": c}

def k_for_abs_delta(delta_abs: float, is_put: bool, sigma: float, T: float) -> float:
    if not (0.0 < delta_abs < 0.5):
        raise ValueError("delta_abs must be in (0,0.5)")
    z = norm_ppf(1 - delta_abs) if is_put else norm_ppf(delta_abs)
    return -sigma * sqrt(T) * z + 0.5 * (sigma ** 2) * T
