"""Statistical helpers for probability computations (CR-C).

Pure math, no DB I/O.

Public entry points:
    wilson_ci(successes, n, alpha=0.05) → (lower, upper) | (None, None)
"""
from __future__ import annotations

import math
from typing import Optional


def wilson_ci(
    successes: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float]]:
    """Wilson score confidence interval for a proportion.

    Preferred over the normal (Wald) approximation for small-n proportions:
    Wald can produce bounds outside [0, 1] and has poor coverage near
    p = 0 or p = 1. Wilson CI is reliable from n = 1.

    Example — why Wilson matters at small n:
        12 / 20 successes (p̂ = 0.60):
          Wald:   0.60 ± 1.96 × sqrt(0.24/20) → [0.385, 0.815]
          Wilson: [0.387, 0.781]  (tighter; guaranteed in [0, 1])

    Returns (None, None) when n == 0 — no information means no bound
    should be synthesised (CR-021 Lesson 2: no silent fallback).

    Formula (Wilson 1927):
        z     = z_{1-α/2}  (≈ 1.96 for α = 0.05)
        denom = 1 + z²/n
        center = (p̂ + z²/2n) / denom
        margin = z × sqrt(p̂(1-p̂)/n + z²/4n²) / denom
        [lower, upper] = clamp([center − margin, center + margin], 0, 1)

    Reference:
        Wilson, E.B. (1927). "Probable inference, the law of succession,
        and statistical inference." J. Am. Stat. Assoc. 22:209–212.
    """
    if n == 0:
        return (None, None)
    z = _z_score(1.0 - alpha / 2.0)
    z2 = z * z
    p_hat = successes / n
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _z_score(p: float) -> float:
    """Inverse standard normal CDF (Abramowitz & Stegun §26.2.17).

    Accurate to ±4.5e-4 over (0, 1). Sufficient for CI computation.
    Verified: _z_score(0.975) → 1.9600 (alpha=0.05 case).
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    upper = p >= 0.5
    t = math.sqrt(-2.0 * math.log(1.0 - p if upper else p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (
        1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    )
    return z if upper else -z
