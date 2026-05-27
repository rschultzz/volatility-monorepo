"""BSM pricing engine — pure compute, no external dependencies beyond scipy.

Unit conventions (locked for downstream consumers):
  - vega:  price change per 1.0 (unit) change in sigma (NOT per 1% vol move)
  - theta: price change per 1 calendar day (raw BSM theta/365.25)
  - rho:   price change per 1.0 (unit) change in r (NOT per 1% rate move)
  - delta: price change per 1.0 move in underlying
  - gamma: delta change per 1.0 move in underlying

Edge-case contract:
  - T <= 0 or sigma <= 0: return intrinsic value (no NaN, no divide-by-zero)
  - S <= 0 or K <= 0: return 0.0
  - Greeks at T <= 0: delta = step function at S vs K (0.5 for ATM call),
    gamma = 0, theta = 0, vega = 0, rho = 0
"""
from __future__ import annotations

import math
import datetime as dt

import numpy as np
from scipy.stats import norm

SECONDS_PER_YEAR: float = 365.25 * 24 * 3600


def _intrinsic(S: float, K: float, flag: str) -> float:
    if flag == "c":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    flag: str,
) -> float:
    """BSM option price.

    flag: 'c' for call, 'p' for put.
    Degenerate cases (T<=0, sigma<=0, S<=0, K<=0) return intrinsic — no NaN.
    """
    if S <= 0.0 or K <= 0.0:
        return 0.0
    if T <= 0.0 or sigma <= 0.0:
        return _intrinsic(S, K, flag)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    if flag == "c":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def _greeks_single(S: float, K: float, T: float, r: float, sigma: float, flag: str) -> dict:
    """Unscaled single-option greeks (before side/qty adjustment)."""
    if T <= 0.0:
        if flag == "c":
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    if sigma <= 0.0:
        if flag == "c":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    nd1 = norm.pdf(d1)
    disc = math.exp(-r * T)

    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * nd1 * sqrt_T  # per 1.0 sigma

    if flag == "c":
        delta = norm.cdf(d1)
        # theta per year (typically negative for long options)
        theta_yr = -(S * nd1 * sigma / (2.0 * sqrt_T)) - r * K * disc * norm.cdf(d2)
        rho = K * T * disc * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta_yr = -(S * nd1 * sigma / (2.0 * sqrt_T)) + r * K * disc * norm.cdf(-d2)
        rho = -K * T * disc * norm.cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_yr / 365.25,  # per calendar day
        "vega": vega,
        "rho": rho,
    }


def _leg_T(leg: dict, evaluation_time: dt.datetime) -> float:
    expiration: dt.datetime = leg["expiration"]
    return max(0.0, (expiration - evaluation_time).total_seconds() / SECONDS_PER_YEAR)


def _leg_sign(leg: dict) -> float:
    return 1.0 if leg.get("side", "long") == "long" else -1.0


def compute_leg_value(leg: dict, S: float, evaluation_time: dt.datetime, market_state: dict) -> float:
    """Mark-to-market value of a single leg at S and evaluation_time.

    leg keys: strike (float), expiration (datetime), flag ('c'/'p'),
              side ('long'/'short'), qty (int, default 1), iv (float annualised).
    Returns signed value accounting for side and qty.
    """
    T = _leg_T(leg, evaluation_time)
    r = market_state.get("risk_free_rate", 0.05)
    price = black_scholes_price(S, leg["strike"], T, r, leg["iv"], leg["flag"])
    return _leg_sign(leg) * leg.get("qty", 1) * price


def compute_position_pl(
    legs: list[dict],
    price_grid: np.ndarray,
    evaluation_time: dt.datetime,
    market_state: dict,
    initial_cost: float,
) -> np.ndarray:
    """P/L across price_grid at evaluation_time.

    initial_cost is the net premium paid (positive = debit) at position entry.
    P/L = current_position_value - initial_cost.
    """
    pl = np.empty(len(price_grid))
    for i, S in enumerate(price_grid):
        value = sum(compute_leg_value(leg, float(S), evaluation_time, market_state) for leg in legs)
        pl[i] = value - initial_cost
    return pl


def compute_position_greeks(
    legs: list[dict],
    S: float,
    evaluation_time: dt.datetime,
    market_state: dict,
) -> dict:
    """Net position greeks at S and evaluation_time.

    Sums each leg's greeks scaled by side and qty.
    See module docstring for unit conventions.
    """
    net = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    r = market_state.get("risk_free_rate", 0.05)
    for leg in legs:
        T = _leg_T(leg, evaluation_time)
        g = _greeks_single(S, leg["strike"], T, r, leg["iv"], leg["flag"])
        scale = _leg_sign(leg) * leg.get("qty", 1)
        for k in net:
            net[k] += scale * g[k]
    return net
