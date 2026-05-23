"""
Unit tests for packages/shared/gex_landscape_api.py (CR-008).

build_gex_landscape_response is exercised against a mocked SQLAlchemy
connection — no DB, no network. Stored-landscape fixtures are generated
synthetically via compute_landscape so the expected classifier outputs are
known (the 5/20 magnet-above scenario is reconstructed from a single
dominant 30+ DTE wall at 7520).

Run with:
    python -m unittest packages.shared.tests.test_gex_landscape_api
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import MagicMock

import pandas as pd

from packages.shared.gex_landscape import _landscape_records, compute_landscape
from packages.shared.gex_landscape_api import build_gex_landscape_response

# The real 5/20 row stores its grid centered on this table_spot.
_TABLE_SPOT_5_20 = 7357.62


def _strike_landscape_records(strikes: list, *, center: float,
                              range_pts: float = 300.0) -> list:
    """Build a stored-landscape JSONB records list from synthetic strikes.

    strikes: list of (discounted_level, dte, gex_call, gex_put) tuples.
    """
    df = pd.DataFrame(
        strikes, columns=["discounted_level", "dte", "gex_call", "gex_put"]
    )
    landscape = compute_landscape(
        df, center, range_pts=range_pts, step_pts=1.0, spread_coef=8.0
    )
    return _landscape_records(landscape)


def _landscape_5_20() -> list:
    """5/20-style landscape: one dominant 30+ DTE magnet at 7520, grid
    centered at the real table_spot, range_pts=300 (so 7520 sits interior)."""
    return _strike_landscape_records(
        [(7520.0, 90, 2e12, 0.0)], center=_TABLE_SPOT_5_20
    )


def _stored_row(landscape_records: list, *, table_spot: float = _TABLE_SPOT_5_20,
                walls=None, peaks=None) -> dict:
    """An orats_gex_landscape row as a SQLAlchemy RowMapping would expose it."""
    return {
        "ticker": "SPX",
        "trade_date": dt.date(2026, 5, 20),
        "landscape": landscape_records,
        # walls / peaks_by_bucket are cron diagnostics — the builder must
        # ignore them and recompute. Bogus values here prove that.
        "walls": walls if walls is not None else [],
        "peaks_by_bucket": peaks if peaks is not None else {},
        "spread_coef": 8.0,
        "range_pts": 300.0,
        "step_pts": 1.0,
        "table_spot": table_spot,
        "version": "test-version",
        "computed_at": dt.datetime(2026, 5, 20, 23, 0, 0),
    }


def _mock_conn(row):
    """SQLAlchemy-Connection-shaped mock.

    The builder calls conn.execute(...).mappings().first(); `row` is what
    .first() yields (a dict-like RowMapping, or None for a missing row).
    """
    result = MagicMock()
    result.mappings.return_value.first.return_value = row
    conn = MagicMock()
    conn.execute.return_value = result
    return conn


def _strike_rows(strikes: list, *, stock_price: float = _TABLE_SPOT_5_20) -> list:
    """Raw orats_oi_gamma rows for the high-accuracy path.

    strikes: list of (discounted_level, dte, gex_call, gex_put) tuples.
    """
    return [
        {
            "discounted_level": lvl, "dte": dte,
            "gex_call": gc, "gex_put": gp, "stock_price": stock_price,
        }
        for (lvl, dte, gc, gp) in strikes
    ]


def _params_row(*, spread_coef: float = 8.0, range_pts: float = 300.0,
                step_pts: float = 1.0, version="test-version",
                computed_at=None) -> dict:
    """An orats_gex_landscape params row as _PARAMS_QUERY would expose it."""
    return {
        "spread_coef": spread_coef, "range_pts": range_pts, "step_pts": step_pts,
        "version": version, "computed_at": computed_at,
    }


def _mock_high_conn(strike_rows, params_row):
    """SQLAlchemy-Connection-shaped mock for the accuracy=high path.

    The builder calls conn.execute() twice: first for the orats_oi_gamma
    strikes (.mappings().all()), then for the orats_gex_landscape params
    (.mappings().first()). params_row may be None (defaults fallback).
    """
    strikes_result = MagicMock()
    strikes_result.mappings.return_value.all.return_value = strike_rows
    params_result = MagicMock()
    params_result.mappings.return_value.first.return_value = params_row
    conn = MagicMock()
    conn.execute.side_effect = [strikes_result, params_result]
    return conn


def _high_strikes_5_20() -> list:
    """5/20-style raw strikes: one dominant 30+ DTE magnet at 7520."""
    return _strike_rows([(7520.0, 90, 2e12, 0.0)])


_TOP_LEVEL_KEYS = {
    "ticker", "trade_date", "spot", "iv", "implied_move", "table_spot",
    "spread_coef", "range_pts", "step_pts", "version", "computed_at",
    "landscape", "walls", "peaks_by_bucket", "regime", "per_bucket",
    "bucket_summary", "confluences", "intraday_subtarget", "neg_zones",
    "accuracy", "recomputed_at", "params_source",
}


class TestResponseShape(unittest.TestCase):
    def test_response_contains_all_documented_keys(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload.keys()), _TOP_LEVEL_KEYS)
        self.assertEqual(payload["ticker"], "SPX")
        self.assertEqual(payload["trade_date"], "2026-05-20")
        self.assertEqual(payload["spot"], 7392.0)
        self.assertEqual(payload["implied_move"], 40.0)
        self.assertEqual(payload["range_pts"], 300.0)
        self.assertEqual(payload["computed_at"], "2026-05-20T23:00:00")
        # landscape is the stored field, passed through unchanged.
        self.assertEqual(len(payload["landscape"]), 601)

    def test_per_bucket_has_four_dte_buckets(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, _ = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        self.assertEqual(
            set(payload["per_bucket"].keys()),
            {"0DTE", "1-7 DTE", "8-30 DTE", "30+ DTE"},
        )
        self.assertEqual(
            set(payload["peaks_by_bucket"].keys()),
            {"0DTE", "1-7 DTE", "8-30 DTE", "30+ DTE"},
        )


class TestFiveTwentySnapshot(unittest.TestCase):
    def test_regime_is_magnet_above_toward_7520(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        self.assertEqual(status, 200)
        regime = payload["regime"]
        self.assertEqual(regime["regime"], "magnet-above")
        self.assertEqual(regime["drift_direction"], "up")
        self.assertAlmostEqual(regime["drift_target"], 7520.0, delta=2.0)
        # 128pt / 40pt implied ≈ 3.2 sigma → multi-day structural pull.
        self.assertEqual(regime["target_classification"]["class"], "multi-day")

    def test_30plus_bucket_peak_surfaces_the_magnet(self):
        # The range_pts=300 smoke test: with the wider stored grid the +7520
        # magnet is interior, so the recomputed 30+ DTE peaks include it.
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, _ = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        struct_peaks = payload["peaks_by_bucket"]["30+ DTE"]
        self.assertTrue(struct_peaks, "expected at least one 30+ DTE peak")
        self.assertTrue(
            any(abs(p["price"] - 7520.0) <= 5.0 for p in struct_peaks),
            f"no 30+ DTE peak near 7520 in {[p['price'] for p in struct_peaks]}",
        )

    def test_walls_recomputed_from_landscape_not_stored_arrays(self):
        # Stored walls contain a bogus entry the builder must NOT pass through.
        bogus = [{"price": 99999.0, "gex": 1.0, "prominence": 1.0, "sign": 1}]
        conn = _mock_conn(_stored_row(_landscape_5_20(), walls=bogus))
        payload, _ = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        wall_prices = [w["price"] for w in payload["walls"]]
        self.assertNotIn(99999.0, wall_prices)
        # The real magnet IS recomputed from the landscape field.
        self.assertTrue(
            any(abs(p - 7520.0) <= 3.0 for p in wall_prices),
            f"recomputed walls missing the 7520 magnet: {wall_prices}",
        )


class TestImpliedMoveResolution(unittest.TestCase):
    def test_iv_resolves_to_a_positive_implied_move(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, iv=0.05
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["iv"], 0.05)
        self.assertIsNotNone(payload["implied_move"])
        self.assertGreater(payload["implied_move"], 0.0)

    def test_no_implied_move_omits_zones_and_subtarget(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0
        )
        self.assertEqual(status, 200)
        self.assertIsNone(payload["implied_move"])
        self.assertIsNone(payload["intraday_subtarget"])
        self.assertEqual(payload["neg_zones"], [])
        # Confluence distance classes degrade to "unknown" without a move.
        for c in payload["confluences"]:
            self.assertEqual(c["distance_classification"]["class"], "unknown")

    def test_neg_zones_populate_when_negative_structure_is_proximate(self):
        # A put-heavy strike right next to spot — a significant negative zone.
        records = _strike_landscape_records(
            [(7385.0, 5, 0.0, 9e11)], center=_TABLE_SPOT_5_20
        )
        conn = _mock_conn(_stored_row(records))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["neg_zones"], "expected a proximate negative zone")
        zone = payload["neg_zones"][0]
        self.assertLess(zone["gex"], 0.0)
        self.assertAlmostEqual(zone["price"], 7385.0, delta=5.0)


class TestErrorStatuses(unittest.TestCase):
    def test_404_when_row_missing(self):
        conn = _mock_conn(None)
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_400_on_malformed_date(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "not-a-date", 7392.0, implied_move=40.0
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_400_on_non_numeric_spot(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", "abc", implied_move=40.0
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_400_when_iv_and_implied_move_both_supplied(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, iv=0.1, implied_move=40.0
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_400_on_negative_implied_move(self):
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=-5.0
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)


class TestAccuracyParam(unittest.TestCase):
    """accuracy query-param contract: default, echo, validation (CR-010 AC 1-5)."""

    def test_accuracy_default_is_low(self):
        # No accuracy param → low path, echoed as "low" with null extras.
        conn = _mock_conn(_stored_row(_landscape_5_20()))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["accuracy"], "low")
        self.assertIsNone(payload["recomputed_at"])
        self.assertIsNone(payload["params_source"])
        self.assertEqual(set(payload.keys()), _TOP_LEVEL_KEYS)

    def test_accuracy_low_explicit_matches_default(self):
        # accuracy=low must be identical to the no-param call.
        default_payload, _ = build_gex_landscape_response(
            _mock_conn(_stored_row(_landscape_5_20())),
            "SPX", "2026-05-20", 7392.0, implied_move=40.0,
        )
        explicit_payload, status = build_gex_landscape_response(
            _mock_conn(_stored_row(_landscape_5_20())),
            "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="low",
        )
        self.assertEqual(status, 200)
        self.assertEqual(explicit_payload, default_payload)

    def test_accuracy_invalid_returns_400(self):
        # Unrecognized accuracy values are rejected before any DB work; the
        # error body names the allowed values.
        for bad in ("medium", "", "foo"):
            with self.subTest(accuracy=bad):
                payload, status = build_gex_landscape_response(
                    MagicMock(), "SPX", "2026-05-20", 7392.0,
                    implied_move=40.0, accuracy=bad,
                )
                self.assertEqual(status, 400)
                self.assertIn("error", payload)
                self.assertIn("low", payload["error"])
                self.assertIn("high", payload["error"])

    def test_accuracy_case_insensitive(self):
        # HIGH / High / high all resolve to the same "high" mode.
        for variant in ("HIGH", "High", "high"):
            with self.subTest(accuracy=variant):
                conn = _mock_high_conn(_high_strikes_5_20(), _params_row())
                payload, status = build_gex_landscape_response(
                    conn, "SPX", "2026-05-20", 7392.0,
                    implied_move=40.0, accuracy=variant,
                )
                self.assertEqual(status, 200)
                self.assertEqual(payload["accuracy"], "high")


class TestHighAccuracyPath(unittest.TestCase):
    """accuracy=high recompute path: shape, recentering, correctness (AC 3,6-10)."""

    def test_accuracy_high_basic_shape(self):
        conn = _mock_high_conn(_high_strikes_5_20(), _params_row())
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="high"
        )
        self.assertEqual(status, 200)
        self.assertEqual(set(payload.keys()), _TOP_LEVEL_KEYS)
        self.assertEqual(payload["accuracy"], "high")
        self.assertIsNotNone(payload["recomputed_at"])
        self.assertIsInstance(payload["recomputed_at"], str)
        self.assertIn(payload["params_source"], ("stored", "defaults"))

    def test_accuracy_high_recenters_grid(self):
        # The recomputed grid is centered on the caller's spot, not the stored
        # table_spot, and spans [spot - range_pts, spot + range_pts].
        conn = _mock_high_conn(_high_strikes_5_20(), _params_row(range_pts=300.0))
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="high"
        )
        self.assertEqual(status, 200)
        prices = [r["price"] for r in payload["landscape"]]
        self.assertEqual(prices[0], 7392.0 - 300.0)    # 7092 — spot-centered
        self.assertEqual(prices[-1], 7392.0 + 300.0)   # 7692
        self.assertEqual(len(prices), 601)
        # table_spot (7357.62) ≠ spot — proves the grid did not center on it.
        self.assertNotEqual(prices[0], _TABLE_SPOT_5_20 - 300.0)
        self.assertEqual(payload["table_spot"], _TABLE_SPOT_5_20)

    def test_accuracy_high_5_20_30dte_bucket_peak(self):
        # AC #6 — the 30+ DTE bucket peak at ~7520 surfaces on the high path.
        conn = _mock_high_conn(_high_strikes_5_20(), _params_row())
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="high"
        )
        self.assertEqual(status, 200)
        struct_peaks = payload["peaks_by_bucket"]["30+ DTE"]
        self.assertTrue(struct_peaks, "expected at least one 30+ DTE peak")
        self.assertTrue(
            any(abs(p["price"] - 7520.0) <= 5.0 for p in struct_peaks),
            f"no 30+ DTE peak near 7520 in {[p['price'] for p in struct_peaks]}",
        )

    def test_accuracy_high_5_20_walls_no_regression(self):
        # AC #8 — the +7520 magnet is still present in the recomputed walls.
        conn = _mock_high_conn(_high_strikes_5_20(), _params_row())
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="high"
        )
        self.assertEqual(status, 200)
        wall_prices = [w["price"] for w in payload["walls"]]
        self.assertTrue(
            any(abs(p - 7520.0) <= 3.0 for p in wall_prices),
            f"recomputed walls missing the 7520 magnet: {wall_prices}",
        )

    def test_accuracy_high_missing_strikes_returns_404(self):
        # AC #10 — no orats_oi_gamma strikes → 404 on the high path.
        conn = _mock_high_conn([], None)
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="high"
        )
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_accuracy_high_falls_back_to_default_params(self):
        # AC #9 — strikes present but no orats_gex_landscape row → defaults.
        conn = _mock_high_conn(_high_strikes_5_20(), None)
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2026-05-20", 7392.0, implied_move=40.0, accuracy="high"
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["params_source"], "defaults")
        self.assertEqual(payload["range_pts"], 300.0)
        self.assertEqual(payload["step_pts"], 1.0)
        self.assertEqual(payload["spread_coef"], 8.0)
        self.assertIsNone(payload["version"])


class TestAccuracyLowMissingRow(unittest.TestCase):
    def test_accuracy_low_missing_landscape_returns_404(self):
        # AC #12 — CR-008 behavior unchanged: a missing stored row is a 404 on
        # the low path, both for the default call and explicit accuracy=low.
        for accuracy in (None, "low"):
            with self.subTest(accuracy=accuracy):
                conn = _mock_conn(None)
                kwargs = {} if accuracy is None else {"accuracy": accuracy}
                payload, status = build_gex_landscape_response(
                    conn, "SPX", "2026-05-20", 7392.0,
                    implied_move=40.0, **kwargs,
                )
                self.assertEqual(status, 404)
                self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
