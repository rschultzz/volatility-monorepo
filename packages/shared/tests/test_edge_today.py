"""Unit tests for packages/shared/edge_today.py (CR-028 / CR-V).

Covers all 10 smoke tests from the spec plus internal helper correctness.
Uses ONLY synthetic data — no DB, no ORATS.
"""
from __future__ import annotations

import pytest

from packages.shared.edge_today import (
    HORIZON_SESSION_DAYS,
    HORIZONS,
    _market_side,
    _touch_rate_within_sessions,
    _unconditional_close_past_strike,
    compute_todays_edge,
)


# ── Synthetic analogue builder ────────────────────────────────────────────────

def make_analogue(
    *,
    outcome_status: str = "computed",
    reached_touch: bool | None = None,
    days_to_reach: int | None = None,
    session_open_t0: float | None = 5000.0,
    implied_move_1d: float | None = 50.0,
    session_close_t1: float | None = None,
    session_close_t5: float | None = None,
    session_close_t15: float | None = None,
) -> dict:
    return {
        "outcome_status":    outcome_status,
        "reached_touch":     reached_touch,
        "days_to_reach":     days_to_reach,
        "session_open_t0":   session_open_t0,
        "implied_move_1d":   implied_move_1d,
        "session_close_t1":  session_close_t1,
        "session_close_t5":  session_close_t5,
        "session_close_t15": session_close_t15,
    }


# ── Smoke test 1: unconditional close includes non-touchers ───────────────────

class TestUnconditionalCloseIncludesNonTouchers:
    """Smoke test 1 from spec.

    A synthetic set where some analogues never touched but closed past the
    strike, and some touched but closed back — the close rate counts ALL.
    """

    def test_non_touchers_counted_in_close(self):
        TODAY_SPOT = 5000.0
        TODAY_IM = 50.0
        STRIKE_ES = 5030.0  # magnet-above; close >= 5030 is past

        analogues = [
            # Touched AND closed past strike
            make_analogue(reached_touch=True, days_to_reach=1,
                          session_close_t5=5040.0),
            # Never touched but closed past strike (non-toucher!)
            make_analogue(reached_touch=False, days_to_reach=None,
                          session_close_t5=5050.0),
            # Touched but closed BELOW strike (reversion)
            make_analogue(reached_touch=True, days_to_reach=2,
                          session_close_t5=4990.0),
            # Never touched, closed below
            make_analogue(reached_touch=False, days_to_reach=None,
                          session_close_t5=4980.0),
        ]

        rate, ci, n = _unconditional_close_past_strike(
            analogues, "t5", STRIKE_ES, "magnet-above", TODAY_SPOT, TODAY_IM
        )

        # 2 of 4 closed past strike (5040 and 5050, both >= 5030)
        assert n == 4
        assert rate == pytest.approx(0.5, abs=1e-4)
        assert ci is not None


# ── Smoke test 2: unconditional close < post-touch "above" ───────────────────

class TestConditioningFix:
    """Smoke test 2: for a set with touch < 100%, unconditional close-past-strike
    is strictly <= the touch-conditional close rate (guards against regressing
    to the post-touch bars).
    """

    def test_unconditional_lower_than_toucher_only(self):
        TODAY_SPOT = 5000.0
        TODAY_IM = 50.0
        STRIKE_ES = 5020.0

        # 3 of 4 touched; among touchers 3/3 closed past strike
        # But non-toucher closed BELOW → unconditional rate 3/4 < toucher rate 3/3
        analogues = [
            make_analogue(reached_touch=True,  days_to_reach=1, session_close_t5=5030.0),
            make_analogue(reached_touch=True,  days_to_reach=2, session_close_t5=5025.0),
            make_analogue(reached_touch=True,  days_to_reach=1, session_close_t5=5035.0),
            make_analogue(reached_touch=False, days_to_reach=None, session_close_t5=4995.0),
        ]

        unc_rate, _, n_unc = _unconditional_close_past_strike(
            analogues, "t5", STRIKE_ES, "magnet-above", TODAY_SPOT, TODAY_IM
        )
        assert n_unc == 4
        assert unc_rate == pytest.approx(0.75, abs=1e-4)

        # Touch-conditional rate (only touchers)
        touchers = [a for a in analogues if a.get("reached_touch")]
        touch_cond_rate, _, _ = _unconditional_close_past_strike(
            touchers, "t5", STRIKE_ES, "magnet-above", TODAY_SPOT, TODAY_IM
        )
        assert touch_cond_rate == pytest.approx(1.0, abs=1e-4)

        assert unc_rate < touch_cond_rate


# ── Smoke test 3: event matching (touch-vs-touch, close-vs-close) ─────────────

class TestEventMatching:
    """Smoke test 3: touch_edge uses 2×|delta|; close_edge uses |delta|."""

    def _build_analogues(self):
        # 6 of 10 touch within 5 sessions; 4 of 10 close past strike at t5
        return [
            make_analogue(reached_touch=True,  days_to_reach=3, session_close_t5=5030.0),
            make_analogue(reached_touch=True,  days_to_reach=1, session_close_t5=5040.0),
            make_analogue(reached_touch=True,  days_to_reach=5, session_close_t5=5025.0),
            make_analogue(reached_touch=True,  days_to_reach=4, session_close_t5=4980.0),
            make_analogue(reached_touch=True,  days_to_reach=2, session_close_t5=4970.0),
            make_analogue(reached_touch=True,  days_to_reach=5, session_close_t5=5035.0),
            make_analogue(reached_touch=False, days_to_reach=None, session_close_t5=4990.0),
            make_analogue(reached_touch=False, days_to_reach=None, session_close_t5=4985.0),
            make_analogue(reached_touch=False, days_to_reach=None, session_close_t5=4995.0),
            make_analogue(reached_touch=False, days_to_reach=None, session_close_t5=4960.0),
        ]

    def test_touch_edge_uses_2x_delta(self):
        analogues = self._build_analogues()
        delta = 0.20
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=analogues,
            horizons=["t5"],
            magnet_delta_by_horizon={"t5": delta},
            spread_cost_by_horizon={"t5": None},
            spread_width=10.0,
            min_analogues=1,
        )
        row = result["per_horizon"][0]
        # mkt_touch = min(1.0, 2 × 0.20) = 0.40
        assert row["mkt_touch"] == pytest.approx(0.40, abs=1e-4)
        # touch_edge = struct_touch - mkt_touch
        assert row["touch_edge"] == pytest.approx(
            row["struct_touch"] - row["mkt_touch"], abs=1e-4
        )

    def test_close_edge_uses_1x_delta(self):
        analogues = self._build_analogues()
        delta = 0.20
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=analogues,
            horizons=["t5"],
            magnet_delta_by_horizon={"t5": delta},
            spread_cost_by_horizon={"t5": None},
            spread_width=10.0,
            min_analogues=1,
        )
        row = result["per_horizon"][0]
        # mkt_close = |delta| = 0.20
        assert row["mkt_close"] == pytest.approx(0.20, abs=1e-4)
        assert row["close_edge"] == pytest.approx(
            row["struct_close"] - row["mkt_close"], abs=1e-4
        )

    def test_crossing_would_give_different_values(self):
        """Documenting: crossing touch-market and close-market gives 2× error."""
        delta = 0.20
        mkt_close, mkt_touch = _market_side(delta, None, 10.0)
        assert mkt_close == pytest.approx(0.20, abs=1e-4)
        assert mkt_touch == pytest.approx(0.40, abs=1e-4)
        # If close_edge naively used mkt_touch, it would be wrong by 2×
        assert mkt_touch != mkt_close


# ── Smoke test 4: horizon parameterization ────────────────────────────────────

class TestHorizonParameterization:
    """Smoke test 4: calling with 2 horizons then 4 — output list resizes correctly;
    no logic branch on specific horizon names.
    """

    def _base_analogues(self):
        return [
            make_analogue(reached_touch=True,  days_to_reach=1,
                          session_close_t1=5030.0, session_close_t5=5035.0,
                          session_close_t15=5040.0),
            make_analogue(reached_touch=False, days_to_reach=None,
                          session_close_t1=4990.0, session_close_t5=4985.0,
                          session_close_t15=4980.0),
        ]

    def test_two_horizons(self):
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=self._base_analogues(),
            horizons=["t1", "t5"],   # only 2
            magnet_delta_by_horizon={"t1": 0.05, "t5": 0.20},
            spread_cost_by_horizon={"t1": None, "t5": None},
            spread_width=10.0,
            min_analogues=1,
        )
        assert result["per_horizon"] is not None
        assert len(result["per_horizon"]) == 2
        assert [r["horizon"] for r in result["per_horizon"]] == ["t1", "t5"]

    def test_three_horizons(self):
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=self._base_analogues(),
            horizons=["t1", "t5", "t15"],
            magnet_delta_by_horizon={"t1": 0.05, "t5": 0.20, "t15": 0.35},
            spread_cost_by_horizon={"t1": None, "t5": None, "t15": None},
            spread_width=10.0,
            min_analogues=1,
        )
        assert len(result["per_horizon"]) == 3

    def test_unknown_horizon_is_skipped_with_warning(self):
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=self._base_analogues(),
            horizons=["t5", "t99"],   # t99 has no HORIZON_SESSION_DAYS entry
            magnet_delta_by_horizon={"t5": 0.20},
            spread_cost_by_horizon={"t5": None},
            spread_width=10.0,
            min_analogues=1,
        )
        assert len(result["per_horizon"]) == 1  # t99 skipped
        assert any("t99" in w for w in result["warnings"])


# ── Smoke test 5: coordinate space ───────────────────────────────────────────

class TestCoordinateSpace:
    """Smoke test 5: close classified in ES-forward vs strike_es.
    A mis-spaced comparison (SPX vs ES) caught by hand-built case.
    """

    def test_es_forward_space_used_for_close_classification(self):
        # Analogue closes at 5025 ES-forward. strike_es = 5020.
        # Trivial projection (anchor_spot == today_spot, anchor_im == today_im)
        # so projected_close = 5025. For magnet-above: 5025 >= 5020 → past.
        TODAY_SPOT = 5000.0
        TODAY_IM = 50.0
        analogues = [
            make_analogue(session_close_t5=5025.0),  # past strike
            make_analogue(session_close_t5=5010.0),  # NOT past strike
        ]
        rate, _, n = _unconditional_close_past_strike(
            analogues, "t5", 5020.0, "magnet-above", TODAY_SPOT, TODAY_IM
        )
        assert n == 2
        assert rate == pytest.approx(0.5, abs=1e-4)

    def test_spx_strike_would_misclassify(self):
        # If someone accidentally passed a SPX strike (e.g. 5100) as strike_es
        # while analogue closes are ES-forward (e.g. 5025 ES ≈ 5030 SPX),
        # the classification outcome differs. This test documents the difference.
        TODAY_SPOT = 5000.0
        TODAY_IM = 50.0
        analogues = [
            make_analogue(session_close_t5=5025.0),
            make_analogue(session_close_t5=5010.0),
        ]
        # With CORRECT ES strike 5020 → rate 0.5 (5025 passes, 5010 doesn't)
        rate_es, _, _ = _unconditional_close_past_strike(
            analogues, "t5", 5020.0, "magnet-above", TODAY_SPOT, TODAY_IM
        )
        # With WRONG SPX strike 5100 → rate 0.0 (neither 5025 nor 5010 >= 5100)
        rate_spx, _, _ = _unconditional_close_past_strike(
            analogues, "t5", 5100.0, "magnet-above", TODAY_SPOT, TODAY_IM
        )
        assert rate_es != rate_spx


# ── Smoke test 6: low-confidence guard ───────────────────────────────────────

class TestLowConfidenceGuard:
    """Smoke test 6: horizon with < min non-null closes → low_confidence=True."""

    def test_low_confidence_when_below_threshold(self):
        # Only 2 analogues with closes → below min_analogues=8
        analogues = [
            make_analogue(reached_touch=True,  days_to_reach=1, session_close_t5=5030.0),
            make_analogue(reached_touch=False, days_to_reach=None, session_close_t5=4990.0),
        ]
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=analogues,
            horizons=["t5"],
            magnet_delta_by_horizon={"t5": 0.20},
            spread_cost_by_horizon={"t5": None},
            spread_width=10.0,
            min_analogues=8,   # threshold
        )
        row = result["per_horizon"][0]
        assert row["low_confidence"] is True

    def test_not_low_confidence_at_or_above_threshold(self):
        analogues = [make_analogue(reached_touch=True, days_to_reach=1,
                                   session_close_t5=5030.0) for _ in range(8)]
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=analogues,
            horizons=["t5"],
            magnet_delta_by_horizon={"t5": 0.20},
            spread_cost_by_horizon={"t5": None},
            spread_width=10.0,
            min_analogues=8,
        )
        row = result["per_horizon"][0]
        assert row["low_confidence"] is False


# ── Smoke test 7: delta-null fallback + missing-market degradation ─────────────

class TestMarketSideFallback:
    """Smoke test 7: null delta → cost÷width fallback; neither → None + warning."""

    def test_cost_over_width_fallback(self):
        mkt_close, mkt_touch = _market_side(None, 2.0, 10.0)
        assert mkt_close == pytest.approx(0.20, abs=1e-4)
        assert mkt_touch == pytest.approx(0.40, abs=1e-4)

    def test_neither_available_gives_none(self):
        mkt_close, mkt_touch = _market_side(None, None, 10.0)
        assert mkt_close is None
        assert mkt_touch is None

    def test_missing_market_produces_none_edges_and_warning(self):
        analogues = [make_analogue(reached_touch=True, days_to_reach=1,
                                   session_close_t5=5030.0) for _ in range(10)]
        result = compute_todays_edge(
            regime="magnet-above",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=analogues,
            horizons=["t5"],
            magnet_delta_by_horizon={"t5": None},
            spread_cost_by_horizon={"t5": None},
            spread_width=10.0,
            min_analogues=1,
        )
        row = result["per_horizon"][0]
        assert row["mkt_close"] is None
        assert row["mkt_touch"] is None
        assert row["touch_edge"] is None
        assert row["close_edge"] is None
        assert any("t5" in w for w in result["warnings"])


# ── Smoke test 8: no BSM / no implied_distribution imports ───────────────────

class TestNoBsmImport:
    """Smoke test 8: edge_today.py must not import implied_distribution or BSM."""

    def test_no_bsm_imports(self):
        import ast
        import pathlib
        src = pathlib.Path(__file__).parent.parent / "edge_today.py"
        tree = ast.parse(src.read_text())
        forbidden = {"implied_distribution", "black_scholes", "bsm", "pricing.engine"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [getattr(node, "module", "") or ""] + [
                    a.name for a in getattr(node, "names", [])
                ]
                for name in names:
                    for f in forbidden:
                        assert f not in (name or ""), (
                            f"edge_today.py must not import {f!r}"
                        )


# ── Smoke test 9: reuse, not reimplementation ─────────────────────────────────

class TestReuseNotReinvention:
    """Smoke test 9: _project_close and wilson_ci imported from the right places."""

    def test_imports_project_close_from_structural_distribution(self):
        from packages.shared.edge_today import _project_close as imported_fn
        from packages.shared.structural_distribution import _project_close as orig_fn
        assert imported_fn is orig_fn

    def test_imports_wilson_ci_from_stats(self):
        from packages.shared.edge_today import wilson_ci as imported_fn
        from packages.shared.stats import wilson_ci as orig_fn
        assert imported_fn is orig_fn


# ── Smoke test 10: 2× cap ────────────────────────────────────────────────────

class TestTwoxCap:
    """Smoke test 10: mkt_touch = min(1.0, 2×|delta|) — never exceeds 1.0."""

    @pytest.mark.parametrize("delta", [0.5, 0.6, 0.8, 1.0, 1.5])
    def test_mkt_touch_capped_at_1(self, delta):
        mkt_close, mkt_touch = _market_side(delta, None, 10.0)
        assert mkt_touch <= 1.0

    def test_mkt_touch_equals_2x_when_small_delta(self):
        delta = 0.20
        mkt_close, mkt_touch = _market_side(delta, None, 10.0)
        assert mkt_touch == pytest.approx(0.40, abs=1e-4)


# ── Additional: unsupported regime returns None ───────────────────────────────

class TestUnsupportedRegime:
    def test_magnetic_pin_returns_none(self):
        result = compute_todays_edge(
            regime="magnetic-pin",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=[make_analogue(reached_touch=True, days_to_reach=1)],
            horizons=HORIZONS,
            magnet_delta_by_horizon={},
            spread_cost_by_horizon={},
            spread_width=10.0,
        )
        assert result["per_horizon"] is None
        assert "magnetic-pin" in result["warnings"][0]

    def test_bounded_returns_none(self):
        result = compute_todays_edge(
            regime="bounded",
            strike_es=5020.0,
            today_spot=5000.0,
            today_implied_move=50.0,
            analogues=[],
            horizons=HORIZONS,
            magnet_delta_by_horizon={},
            spread_cost_by_horizon={},
            spread_width=10.0,
        )
        assert result["per_horizon"] is None


# ── Additional: magnet-below direction ───────────────────────────────────────

class TestMagnetBelow:
    def test_close_below_strike_counts(self):
        # magnet-below: close <= strike_es is past
        analogues = [
            make_analogue(session_close_t5=4950.0),   # past (below 4970)
            make_analogue(session_close_t5=4980.0),   # NOT past
        ]
        rate, _, n = _unconditional_close_past_strike(
            analogues, "t5", 4970.0, "magnet-below", 5000.0, 50.0
        )
        assert n == 2
        assert rate == pytest.approx(0.5, abs=1e-4)


# ── Additional: touch rate excludes non-computed analogues ───────────────────

class TestTouchRateExcludesNonComputed:
    def test_na_regime_excluded(self):
        analogues = [
            make_analogue(outcome_status="computed", reached_touch=True,  days_to_reach=1),
            make_analogue(outcome_status="na_regime", reached_touch=True,  days_to_reach=1),
            make_analogue(outcome_status="computed", reached_touch=False, days_to_reach=None),
        ]
        rate, _, k = _touch_rate_within_sessions(analogues, 5)
        # only 2 computed: 1 touched
        assert k == 2
        assert rate == pytest.approx(0.5, abs=1e-4)


# ── Additional: HORIZONS constant is a list, HORIZON_SESSION_DAYS has all v1 ─

class TestModuleConstants:
    def test_horizons_is_list(self):
        assert isinstance(HORIZONS, list)

    def test_horizon_session_days_covers_horizons(self):
        for h in HORIZONS:
            assert h in HORIZON_SESSION_DAYS
