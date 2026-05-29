"""Unit tests for packages/shared/implied_distribution.py.

Synthetic BSM chains are used throughout — the goal is to verify that the
Breeden-Litzenberger recovery is mathematically correct, not to exercise
live data paths. The true risk-neutral density under BSM is lognormal.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.stats import norm

from packages.shared.implied_distribution import (
    compute_implied_pdf,
    compute_implied_prob_in_range,
)
from packages.shared.pricing.engine import black_scholes_price

# ── Synthetic chain helpers ──────────────────────────────────────────────────

S0 = 100.0
R = 0.05
T30 = 30.0 / 365.25
SIG = 0.20


def _bsm_call(K, sigma=SIG, S=S0, T=T30, r=R):
    return black_scholes_price(S, K, T, r, sigma, "c")


def _chain(strikes, sigma=SIG, S=S0, T=T30, r=R):
    return [{"strike": float(k), "call_price": _bsm_call(k, sigma, S, T, r)} for k in strikes]


def _lognormal_density(K, S=S0, T=T30, r=R, sigma=SIG):
    """True risk-neutral lognormal density at K under BSM."""
    sigma_T = sigma * math.sqrt(T)
    mu = math.log(S) + (r - 0.5 * sigma ** 2) * T
    return norm.pdf((math.log(K) - mu) / sigma_T) / (K * sigma_T)


# ── compute_implied_pdf — dense chain ────────────────────────────────────────

class TestImpliedPdfDense:
    """Dense chain: 2-pt spacing, 31 strikes (70-130). Exercises non-uniform d²."""

    def setup_method(self):
        self.strikes = list(range(70, 132, 2))
        self.chain = _chain(self.strikes)
        self.pdf = compute_implied_pdf(self.chain, S0, R, T30)

    def test_pdf_nonempty(self):
        assert len(self.pdf) > 0

    def test_pdf_nonnegative(self):
        for k, d in self.pdf.items():
            assert d >= 0.0, f"Negative density at K={k}: {d}"

    def test_pdf_integrates_near_one(self):
        """PDF over 70-130 should capture most of the probability mass."""
        ks = sorted(self.pdf.keys())
        ds = [self.pdf[k] for k in ks]
        total = float(np.trapezoid(ds, ks))
        assert 0.85 < total < 1.05, f"PDF integral = {total:.4f} (expected ~0.90-1.0)"

    def test_pdf_matches_lognormal_within_tolerance(self):
        """Recovered density within 10% of true lognormal at interior NTM strikes."""
        for K in range(90, 112, 2):
            if K not in self.pdf:
                continue
            true_d = _lognormal_density(K)
            est_d  = self.pdf[K]
            if true_d < 1e-4:
                continue  # skip deep tail where relative error is meaningless
            rel_err = abs(est_d - true_d) / true_d
            assert rel_err < 0.10, f"K={K}: estimated={est_d:.6f} true={true_d:.6f} rel_err={rel_err:.3f}"

    def test_boundary_strikes_absent(self):
        """Boundary strikes (first and last) are consumed by finite differences."""
        assert float(self.strikes[0]) not in self.pdf
        assert float(self.strikes[-1]) not in self.pdf


# ── compute_implied_pdf — sparse chain ───────────────────────────────────────

class TestImpliedPdfSparse:
    """Sparse chain: 5 strikes at 10-pt spacing. Exercises cubic-spline path."""

    def setup_method(self):
        self.strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        self.chain = _chain(self.strikes)
        self.pdf = compute_implied_pdf(self.chain, S0, R, T30)

    def test_no_nan(self):
        for k, d in self.pdf.items():
            assert not math.isnan(d), f"NaN density at K={k}"

    def test_nonnegative(self):
        for k, d in self.pdf.items():
            assert d >= 0.0, f"Negative density at K={k}: {d}"

    def test_fills_fine_grid(self):
        """Sparse chain with 5 strikes should produce ~99 PDF points on 1-pt grid."""
        assert len(self.pdf) >= 30

    def test_not_all_zero(self):
        assert sum(self.pdf.values()) > 0


# ── compute_implied_pdf — minimal chain ──────────────────────────────────────

class TestImpliedPdfMinimal:
    def test_fewer_than_3_strikes_returns_empty(self):
        chain = _chain([90.0, 100.0])  # only 2 strikes
        pdf = compute_implied_pdf(chain, S0, R, T30)
        assert pdf == {}

    def test_exactly_3_strikes(self):
        # 3 strikes triggers the sparse path (< 8 strikes) → spline fills fine grid
        chain = _chain([90.0, 100.0, 110.0])
        pdf = compute_implied_pdf(chain, S0, R, T30)
        assert len(pdf) >= 1
        for d in pdf.values():
            assert d >= 0.0
            assert not math.isnan(d)


# ── compute_implied_prob_in_range ────────────────────────────────────────────

class TestImpliedProbInRange:
    def setup_method(self):
        # Dense chain for reliable PDF
        strikes = list(range(70, 132, 2))
        chain = _chain(strikes)
        self.pdf = compute_implied_pdf(chain, S0, R, T30)

    def test_range_covers_most_mass(self):
        """ATM ±2-sigma range captures ~90% of probability."""
        sigma_T = SIG * math.sqrt(T30)
        lo = S0 * math.exp(-2 * sigma_T)
        hi = S0 * math.exp(2 * sigma_T)
        p = compute_implied_prob_in_range(self.pdf, lo, hi)
        assert 0.85 < p < 1.05, f"2-sigma range prob = {p:.4f}"

    def test_lower_equals_upper_returns_zero(self):
        p = compute_implied_prob_in_range(self.pdf, 100.0, 100.0)
        assert p == 0.0

    def test_one_sided_lower_none(self):
        """lower=None → integrate from min strike to upper."""
        p_full = compute_implied_prob_in_range(self.pdf, None, None)
        # With both None: lower defaults to min, upper defaults to max → full range
        # That's effectively the full integral
        p_half = compute_implied_prob_in_range(self.pdf, None, 100.0)
        p_other = compute_implied_prob_in_range(self.pdf, 100.0, None)
        # Both halves should sum roughly to the full integral (with some overlap at exactly 100)
        total = compute_implied_prob_in_range(self.pdf, None, None)
        assert p_half > 0
        assert p_other > 0
        # Rough complement check: two halves sum ~= full
        assert abs((p_half + p_other) - total) < 0.05

    def test_out_of_range_emits_warning(self):
        min_k = min(self.pdf.keys())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p = compute_implied_prob_in_range(self.pdf, min_k - 20.0, 110.0)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1
        assert any("nearest-tail" in str(x.message).lower() for x in runtime_warnings)
        assert p > 0.0  # extrapolation produces a value

    def test_empty_pdf_returns_zero(self):
        assert compute_implied_prob_in_range({}, 90.0, 110.0) == 0.0

    def test_narrow_range_small_prob(self):
        # ATM density ~0.07/pt; 2-pt range → expected ~0.14
        p = compute_implied_prob_in_range(self.pdf, 99.0, 101.0)
        assert 0.0 < p < 0.25, f"Narrow 2-pt range prob = {p:.5f}"


# ── sparse chain integration ──────────────────────────────────────────────────

class TestSparseChainIntegration:
    """End-to-end: 5-strike sparse chain → spline → PDF → prob. No NaN."""

    def test_sparse_chain_prob_in_range_no_nan(self):
        chain = _chain([80.0, 90.0, 100.0, 110.0, 120.0])
        pdf = compute_implied_pdf(chain, S0, R, T30)
        p = compute_implied_prob_in_range(pdf, 85.0, 115.0)
        assert not math.isnan(p)
        assert 0.0 <= p <= 1.5   # loose upper bound (sparse noise allowed)

    def test_sparse_chain_pdf_smooth(self):
        """No extreme spikes — max density no more than 5× median."""
        chain = _chain([80.0, 90.0, 100.0, 110.0, 120.0])
        pdf = compute_implied_pdf(chain, S0, R, T30)
        densities = [d for d in pdf.values() if d > 0]
        if not densities:
            return
        median = float(np.median(densities))
        max_d  = max(densities)
        assert max_d < 5.0 * median + 1e-6, f"Spike: max={max_d:.4f}, median={median:.4f}"
