"""Unit tests for packages/shared/pricing/engine.py.

Test structure:
  1. Standard BSM cases (price, put-call parity, spread shape)
  2. Edge cases (T=0, tiny T, deep OTM/ITM, sigma=0, stress boundary)
  3. Greeks sign/magnitude checks

Hand-computed BSM reference for S=K=100, T=30d, r=0.05, sigma=0.20:
  d1 ≈ 0.10031, d2 ≈ 0.04299
  N(d1) ≈ 0.53993, N(d2) ≈ 0.51715
  call ≈ 2.495
  put  ≈ 2.085  (via parity: call - (S - K*exp(-rT)) = 2.495 - 0.410 = 2.085)
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from packages.shared.pricing.engine import (
    SECONDS_PER_YEAR,
    black_scholes_price,
    compute_leg_value,
    compute_position_greeks,
    compute_position_pl,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

S0 = 100.0
K0 = 100.0
T30 = 30.0 / 365.25
R = 0.05
SIG = 0.20

_MARKET = {"risk_free_rate": R}

_EXPIRY = dt.datetime(2025, 3, 1, 20, 0, 0, tzinfo=dt.timezone.utc)
_ENTRY = _EXPIRY - dt.timedelta(seconds=T30 * SECONDS_PER_YEAR)


def _leg(strike, flag, side="long", iv=SIG, qty=1, expiry=_EXPIRY):
    return {"strike": strike, "flag": flag, "side": side, "iv": iv, "qty": qty, "expiration": expiry}


# ── 1. Standard BSM cases ────────────────────────────────────────────────────

class TestStandardCases:
    def test_atm_call_price_matches_reference(self):
        price = black_scholes_price(S0, K0, T30, R, SIG, "c")
        assert abs(price - 2.495) < 0.02, f"ATM call = {price:.4f}, expected ~2.495"

    def test_put_call_parity(self):
        call = black_scholes_price(S0, K0, T30, R, SIG, "c")
        put  = black_scholes_price(S0, K0, T30, R, SIG, "p")
        forward = S0 - K0 * math.exp(-R * T30)
        assert abs((call - put) - forward) < 1e-6, (
            f"PCP violated: call-put={call-put:.6f}, S-Ke^-rT={forward:.6f}"
        )

    def test_short_call_pl_is_negative_long(self):
        legs_long  = [_leg(K0, "c", "long",  expiry=_EXPIRY)]
        legs_short = [_leg(K0, "c", "short", expiry=_EXPIRY)]
        # compute_leg_value at current spot, current time
        val_long  = compute_leg_value(legs_long[0],  S0, _ENTRY, _MARKET)
        val_short = compute_leg_value(legs_short[0], S0, _ENTRY, _MARKET)
        assert abs(val_long + val_short) < 1e-10, (
            f"Long + short should sum to 0, got {val_long + val_short}"
        )

    def test_debit_call_spread_pl_shape_at_expiration(self):
        """Long 4150C / short 4160C debit spread P/L at expiration is piecewise linear."""
        exp = dt.datetime(2025, 6, 1, 20, 0, tzinfo=dt.timezone.utc)
        legs = [
            _leg(4150, "c", "long",  iv=0.18, expiry=exp),
            _leg(4160, "c", "short", iv=0.18, expiry=exp),
        ]
        entry_time = exp - dt.timedelta(days=15)
        mkt = {"risk_free_rate": 0.05}
        # initial_cost = net premium at entry
        initial_cost = sum(compute_leg_value(l, 4150.0, entry_time, mkt) for l in legs)
        assert initial_cost > 0, "Debit spread initial cost must be positive"

        # Evaluate at expiration
        grid = np.arange(4100.0, 4201.0, 1.0)
        pl = compute_position_pl(legs, grid, exp, mkt, initial_cost)

        idx_4100 = 0
        idx_4150 = 50
        idx_4160 = 60

        assert abs(pl[idx_4100] - (-initial_cost)) < 1e-4, "At S=4100 (below both): max loss"
        assert abs(pl[idx_4150] - (-initial_cost)) < 1e-4, "At S=4150 (long strike): max loss"
        assert abs(pl[idx_4160] - (10.0 - initial_cost)) < 1e-4, "At S=4160 (short strike): max profit"

        # Between the strikes: linear interpolation
        for idx in range(idx_4150, idx_4160):
            S = 4150.0 + (idx - idx_4150)
            expected = (S - 4150.0) - initial_cost
            assert abs(pl[idx] - expected) < 1e-3, f"Non-linear at S={S}: got {pl[idx]:.4f} expected {expected:.4f}"

        # Above short strike: capped at max profit
        max_profit = 10.0 - initial_cost
        for idx in range(idx_4160, len(pl)):
            assert abs(pl[idx] - max_profit) < 1e-3, f"Above 4160 should be capped at max profit"

    def test_credit_put_spread_parity(self):
        """Short lower K put / long higher K put — PCP holds on component legs."""
        T = 21.0 / 365.25
        K_low, K_high = 95.0, 100.0
        call_low  = black_scholes_price(S0, K_low,  T, R, SIG, "c")
        put_low   = black_scholes_price(S0, K_low,  T, R, SIG, "p")
        call_high = black_scholes_price(S0, K_high, T, R, SIG, "c")
        put_high  = black_scholes_price(S0, K_high, T, R, SIG, "p")
        fwd_low  = S0 - K_low  * math.exp(-R * T)
        fwd_high = S0 - K_high * math.exp(-R * T)
        assert abs((call_low  - put_low)  - fwd_low)  < 1e-6
        assert abs((call_high - put_high) - fwd_high) < 1e-6


# ── 2. Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_t_zero_call_returns_intrinsic(self):
        assert black_scholes_price(105.0, 100.0, 0.0, R, SIG, "c") == pytest.approx(5.0)
        assert black_scholes_price(95.0,  100.0, 0.0, R, SIG, "c") == pytest.approx(0.0)

    def test_t_zero_put_returns_intrinsic(self):
        assert black_scholes_price(95.0,  100.0, 0.0, R, SIG, "p") == pytest.approx(5.0)
        assert black_scholes_price(105.0, 100.0, 0.0, R, SIG, "p") == pytest.approx(0.0)

    def test_t_zero_no_nan(self):
        # ATM at expiry: should be 0.0, not NaN
        price = black_scholes_price(100.0, 100.0, 0.0, R, SIG, "c")
        assert not math.isnan(price)
        assert price == 0.0

    def test_tiny_t_no_nan(self):
        T_tiny = 1.0 / (365.25 * 24 * 3600)  # 1 second in years
        price = black_scholes_price(S0, K0, T_tiny, R, SIG, "c")
        assert not math.isnan(price)
        assert not math.isinf(price)
        assert price >= 0.0

    def test_tiny_t_converges_to_intrinsic(self):
        # Sub-second to expiry, ITM call: should be close to S-K
        T_tiny = 0.0001
        price = black_scholes_price(110.0, 100.0, T_tiny, R, SIG, "c")
        assert abs(price - 10.0) < 0.01, f"Deep sub-second ITM call should ≈ intrinsic, got {price:.4f}"

    def test_deep_otm_call_is_nonneg_finite_decreasing(self):
        """OTM call price >= 0, finite, and decreasing as K increases (for fixed S)."""
        S = 100.0
        T = T30
        strikes = [110.0, 120.0, 130.0, 140.0]
        prices = [black_scholes_price(S, K, T, R, SIG, "c") for K in strikes]
        for p in prices:
            assert p >= 0.0
            assert not math.isnan(p)
            assert not math.isinf(p)
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], "Call price must decrease with higher strike"

    def test_deep_itm_call_near_intrinsic(self):
        """Very deep ITM call ≈ intrinsic + small time value."""
        price = black_scholes_price(120.0, 100.0, T30, R, SIG, "c")
        intrinsic = 20.0
        assert price >= intrinsic, "Call price must be >= intrinsic"
        assert price < intrinsic + 2.0, f"Deep ITM call excess time value too large: {price - intrinsic:.4f}"

    def test_sigma_zero_returns_intrinsic(self):
        assert black_scholes_price(105.0, 100.0, T30, R, 0.0, "c") == pytest.approx(5.0)
        assert black_scholes_price(95.0,  100.0, T30, R, 0.0, "p") == pytest.approx(5.0)
        assert black_scholes_price(95.0,  100.0, T30, R, 0.0, "c") == pytest.approx(0.0)

    def test_stress_boundary_atm_very_short_t(self):
        """S=K=100, T=0.001yr, sigma=0.20 — must be finite positive, not NaN."""
        price = black_scholes_price(100.0, 100.0, 0.001, R, 0.20, "c")
        assert not math.isnan(price)
        assert not math.isinf(price)
        assert price > 0.0
        assert price < 5.0  # should be small

    def test_nonpositive_S_returns_zero(self):
        assert black_scholes_price(0.0, 100.0, T30, R, SIG, "c") == 0.0
        assert black_scholes_price(-1.0, 100.0, T30, R, SIG, "c") == 0.0

    def test_nonpositive_K_returns_zero(self):
        assert black_scholes_price(100.0, 0.0, T30, R, SIG, "p") == 0.0


# ── 3. Greeks tests ──────────────────────────────────────────────────────────

class TestGreeks:
    def test_long_atm_call_signs(self):
        """Long ATM call: delta ~0.5, gamma > 0, theta < 0, vega > 0, rho > 0."""
        legs = [_leg(K0, "c", "long", expiry=_EXPIRY)]
        g = compute_position_greeks(legs, S0, _ENTRY, _MARKET)
        assert 0.4 < g["delta"] < 0.7, f"ATM call delta expected ~0.5, got {g['delta']:.4f}"
        assert g["gamma"] > 0.0, f"gamma should be positive, got {g['gamma']}"
        assert g["theta"] < 0.0, f"theta should be negative (time decay), got {g['theta']}"
        assert g["vega"] > 0.0, f"vega should be positive, got {g['vega']}"
        assert g["rho"] > 0.0, f"rho should be positive for call, got {g['rho']}"

    def test_short_atm_call_signs_flip(self):
        """Short ATM call: all greeks flip sign vs long."""
        legs_long  = [_leg(K0, "c", "long",  expiry=_EXPIRY)]
        legs_short = [_leg(K0, "c", "short", expiry=_EXPIRY)]
        g_long  = compute_position_greeks(legs_long,  S0, _ENTRY, _MARKET)
        g_short = compute_position_greeks(legs_short, S0, _ENTRY, _MARKET)
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            assert abs(g_long[key] + g_short[key]) < 1e-10, (
                f"{key}: long={g_long[key]:.6f}, short={g_short[key]:.6f} — should sum to 0"
            )

    def test_long_atm_put_signs(self):
        """Long ATM put: delta < 0, gamma > 0, theta < 0, vega > 0, rho < 0."""
        legs = [_leg(K0, "p", "long", expiry=_EXPIRY)]
        g = compute_position_greeks(legs, S0, _ENTRY, _MARKET)
        assert -0.7 < g["delta"] < -0.3, f"ATM put delta expected ~-0.5, got {g['delta']:.4f}"
        assert g["gamma"] > 0.0
        assert g["theta"] < 0.0
        assert g["vega"] > 0.0
        assert g["rho"] < 0.0, f"rho should be negative for put, got {g['rho']}"

    def test_greeks_at_t_zero_no_nan(self):
        """At expiration: no NaN, gamma=0, theta=0, vega=0; delta is step function."""
        legs = [_leg(K0, "c", "long", expiry=_EXPIRY)]
        # Evaluate exactly at expiry → T=0
        g = compute_position_greeks(legs, S0, _EXPIRY, _MARKET)
        for key, val in g.items():
            assert not math.isnan(val), f"{key} is NaN at T=0"
            assert not math.isinf(val), f"{key} is inf at T=0"
        # ATM call at T=0: delta = 0.5 by convention
        assert g["delta"] == pytest.approx(0.5)
        assert g["gamma"] == pytest.approx(0.0)
        assert g["theta"] == pytest.approx(0.0)
        assert g["vega"]  == pytest.approx(0.0)

    def test_greeks_atm_call_magnitudes(self):
        """Sanity-check magnitude ranges for ATM call at T=30d, sigma=0.20."""
        legs = [_leg(K0, "c", "long", expiry=_EXPIRY)]
        g = compute_position_greeks(legs, S0, _ENTRY, _MARKET)
        # Vega per 1.0 sigma for S=100, T=30d: ≈ 11.4
        assert 8.0 < g["vega"] < 15.0, f"Vega out of expected range: {g['vega']:.4f}"
        # Gamma for S=100, T=30d, sigma=0.20: ≈ 0.069
        assert 0.03 < g["gamma"] < 0.15, f"Gamma out of expected range: {g['gamma']:.4f}"
        # Theta per calendar day: ≈ -0.08
        assert -0.15 < g["theta"] < -0.03, f"Theta out of expected range: {g['theta']:.4f}"

    def test_multi_leg_greeks_sum_correctly(self):
        """Debit spread greeks = sum of individual legs."""
        exp = dt.datetime(2025, 6, 1, 20, 0, tzinfo=dt.timezone.utc)
        entry = exp - dt.timedelta(days=30)
        legs = [
            _leg(100.0, "c", "long",  iv=0.20, expiry=exp),
            _leg(105.0, "c", "short", iv=0.20, expiry=exp),
        ]
        mkt = {"risk_free_rate": 0.05}
        g = compute_position_greeks(legs, 100.0, entry, mkt)
        g_long  = compute_position_greeks([legs[0]], 100.0, entry, mkt)
        g_short = compute_position_greeks([legs[1]], 100.0, entry, mkt)
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            expected = g_long[key] + g_short[key]
            assert abs(g[key] - expected) < 1e-10, f"{key} multi-leg sum mismatch"


# ── 4. Mini integration — debit call spread ──────────────────────────────────

class TestDebitCallSpreadIntegration:
    """Construct long 4150C / short 4160C, T=15d, sigma=0.18, S=4150.

    Verify P/L curve at expiration: piecewise linear with expected anchors.
    """

    def setup_method(self):
        self.exp = dt.datetime(2025, 9, 1, 20, 0, tzinfo=dt.timezone.utc)
        self.entry = self.exp - dt.timedelta(days=15)
        self.mkt = {"risk_free_rate": 0.05}
        self.legs = [
            _leg(4150.0, "c", "long",  iv=0.18, expiry=self.exp),
            _leg(4160.0, "c", "short", iv=0.18, expiry=self.exp),
        ]
        self.S_entry = 4150.0
        self.initial_cost = sum(
            compute_leg_value(l, self.S_entry, self.entry, self.mkt) for l in self.legs
        )
        self.grid = np.arange(4100.0, 4201.0, 1.0)
        self.pl = compute_position_pl(self.legs, self.grid, self.exp, self.mkt, self.initial_cost)

    def test_initial_cost_is_positive_debit(self):
        assert self.initial_cost > 0.0, "Debit spread must cost money to enter"
        assert self.initial_cost < 10.0, "Net debit must be less than spread width"

    def test_max_loss_at_s_4100(self):
        idx = int(4100.0 - 4100.0)  # = 0
        assert abs(self.pl[idx] - (-self.initial_cost)) < 1e-3

    def test_max_loss_at_long_strike(self):
        idx = int(4150.0 - 4100.0)  # = 50
        assert abs(self.pl[idx] - (-self.initial_cost)) < 1e-3

    def test_max_profit_at_short_strike(self):
        idx = int(4160.0 - 4100.0)  # = 60
        expected = 10.0 - self.initial_cost
        assert abs(self.pl[idx] - expected) < 1e-3

    def test_linear_between_strikes(self):
        for i in range(50, 61):  # S = 4150..4160
            S = 4100.0 + i
            expected = (S - 4150.0) - self.initial_cost
            assert abs(self.pl[i] - expected) < 1e-3, (
                f"Non-linear at S={S}: got {self.pl[i]:.4f} expected {expected:.4f}"
            )

    def test_capped_above_short_strike(self):
        max_profit = 10.0 - self.initial_cost
        for i in range(60, len(self.pl)):  # S >= 4160
            assert abs(self.pl[i] - max_profit) < 1e-3, (
                f"Not capped at S={4100.0 + i}: got {self.pl[i]:.4f}"
            )
