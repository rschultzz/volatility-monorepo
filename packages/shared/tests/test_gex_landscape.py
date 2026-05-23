"""
Unit tests for packages/shared/gex_landscape.py (CR-007).

Two layers:
  - snapshot tests on the pure analytical functions (compute_landscape,
    find_walls, find_peaks_per_bucket, classify_regime) with synthetic input;
  - mocked-DB tests on compute_and_upsert_landscape — no DB, no network.

Run with:
    python -m unittest packages.shared.tests.test_gex_landscape
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from packages.shared.gex_landscape import (
    _UPSERT_SQL,
    classify_confluence_quality,
    classify_regime,
    compute_and_upsert_landscape,
    compute_landscape,
    find_peaks_per_bucket,
    find_walls,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _calibration_set() -> dict:
    """Load the CR-011 confluence-quality calibration fixture."""
    with open(_FIXTURE_DIR / "confluence_calibration.json") as f:
        return json.load(f)


def _confirmed_day(trade_date: str) -> dict:
    """Return the confirmed calibration entry for a trade date."""
    for entry in _calibration_set()["confirmed"]:
        if entry["trade_date"] == trade_date:
            return entry
    raise KeyError(f"no confirmed calibration entry for {trade_date}")

# psycopg's JSONB adapter — the helper wraps each JSONB column in this.
from psycopg.types.json import Jsonb


def _strike_df(rows: list) -> pd.DataFrame:
    """Build a minimal orats_oi_gamma-shaped frame for compute_landscape.

    rows: list of (discounted_level, dte, gex_call, gex_put) tuples.
    """
    return pd.DataFrame(
        rows, columns=["discounted_level", "dte", "gex_call", "gex_put"]
    )


# ────────────────────────────────────────────────────────────────────────
#  compute_landscape
# ────────────────────────────────────────────────────────────────────────

class TestComputeLandscape(unittest.TestCase):
    def test_empty_df_returns_empty_landscape_with_columns(self):
        landscape = compute_landscape(pd.DataFrame(), spot=7400.0)
        self.assertTrue(landscape.empty)
        self.assertEqual(
            list(landscape.columns),
            ["price", "gex_total", "gex_0dte", "gex_near", "gex_med", "gex_struct"],
        )

    def test_grid_has_expected_point_count(self):
        # range_pts=200, step_pts=1 → 401 grid points (spot ± 200 inclusive).
        df = _strike_df([(7400.0, 5, 1e12, 0.0)])
        landscape = compute_landscape(
            df, spot=7400.0, range_pts=200.0, step_pts=1.0, spread_coef=8.0
        )
        self.assertEqual(len(landscape), 401)

    def test_peak_aligns_with_strike_level(self):
        # One positive strike at 7460: the smoothed field should peak there.
        df = _strike_df([(7460.0, 10, 8e11, 0.0)])
        landscape = compute_landscape(
            df, spot=7400.0, range_pts=200.0, step_pts=1.0, spread_coef=8.0
        )
        peak_price = float(landscape.loc[landscape["gex_total"].idxmax(), "price"])
        self.assertAlmostEqual(peak_price, 7460.0, delta=1.0)

    def test_buckets_sum_to_total(self):
        # Strikes spanning all four DTE buckets — the per-bucket columns must
        # reconstruct gex_total at every grid price.
        df = _strike_df([
            (7390.0, 0, 3e11, 1e11),    # 0DTE
            (7410.0, 4, 2e11, 0.0),     # 1-7
            (7440.0, 20, 5e11, 1e11),   # 8-30
            (7470.0, 90, 9e11, 2e11),   # 30+
        ])
        landscape = compute_landscape(
            df, spot=7430.0, range_pts=150.0, step_pts=1.0, spread_coef=8.0
        )
        bucket_sum = (
            landscape["gex_0dte"] + landscape["gex_near"]
            + landscape["gex_med"] + landscape["gex_struct"]
        )
        self.assertTrue(
            np.allclose(bucket_sum.to_numpy(), landscape["gex_total"].to_numpy())
        )


# ────────────────────────────────────────────────────────────────────────
#  find_walls
# ────────────────────────────────────────────────────────────────────────

class TestFindWalls(unittest.TestCase):
    def test_empty_landscape_returns_empty(self):
        walls = find_walls(pd.DataFrame())
        self.assertTrue(walls.empty)

    def test_finds_positive_and_negative_walls_with_sign(self):
        # A positive strike above spot and a put-heavy (negative) strike below
        # produce one positive wall and one negative wall.
        df = _strike_df([
            (7480.0, 15, 9e11, 0.0),    # positive wall
            (7320.0, 15, 0.0, 9e11),    # negative wall (net = -|put|)
        ])
        landscape = compute_landscape(
            df, spot=7400.0, range_pts=200.0, step_pts=1.0, spread_coef=8.0
        )
        walls = find_walls(landscape)
        signs = set(walls["sign"].tolist())
        self.assertIn(1.0, signs)
        self.assertIn(-1.0, signs)
        pos = walls[walls["sign"] > 0].iloc[0]
        neg = walls[walls["sign"] < 0].iloc[0]
        self.assertAlmostEqual(float(pos["price"]), 7480.0, delta=3.0)
        self.assertAlmostEqual(float(neg["price"]), 7320.0, delta=3.0)
        self.assertGreater(float(pos["gex"]), 0.0)
        self.assertLess(float(neg["gex"]), 0.0)


# ────────────────────────────────────────────────────────────────────────
#  find_peaks_per_bucket
# ────────────────────────────────────────────────────────────────────────

class TestFindPeaksPerBucket(unittest.TestCase):
    def test_returns_all_four_bucket_keys(self):
        df = _strike_df([
            (7400.0, 0, 3e11, 0.0),
            (7420.0, 4, 2e11, 0.0),
            (7440.0, 20, 5e11, 0.0),
            (7460.0, 90, 9e11, 0.0),
        ])
        landscape = compute_landscape(
            df, spot=7430.0, range_pts=150.0, step_pts=1.0, spread_coef=8.0
        )
        peaks = find_peaks_per_bucket(landscape)
        self.assertEqual(
            set(peaks.keys()), {"0DTE", "1-7 DTE", "8-30 DTE", "30+ DTE"}
        )
        # Every peak dict carries the four documented fields.
        for bucket_peaks in peaks.values():
            for p in bucket_peaks:
                self.assertEqual(
                    set(p.keys()), {"price", "gex", "prominence", "fwhm"}
                )


# ────────────────────────────────────────────────────────────────────────
#  classify_regime — the 5/20-style magnet-above case
# ────────────────────────────────────────────────────────────────────────

class TestClassifyRegime(unittest.TestCase):
    def test_magnet_above_for_isolated_wall_above_spot(self):
        # One dominant positive wall at 7520, spot 128pt below at 7392, no
        # competing structure in between — the 5/20 regression scenario.
        df = _strike_df([(7520.0, 20, 2e12, 0.0)])
        landscape = compute_landscape(
            df, spot=7392.0, range_pts=200.0, step_pts=1.0, spread_coef=8.0
        )
        regime = classify_regime(landscape, spot=7392.0, implied_move=40.0)
        self.assertEqual(regime["regime"], "magnet-above")
        self.assertAlmostEqual(regime["drift_target"], 7520.0, delta=2.0)
        self.assertEqual(regime["drift_direction"], "up")

    def test_untethered_when_no_walls(self):
        # A flat (all-zero) landscape has no structural walls of either sign.
        flat = pd.DataFrame({
            "price":      np.arange(7300.0, 7500.0, 1.0),
            "gex_total":  0.0,
            "gex_0dte":   0.0,
            "gex_near":   0.0,
            "gex_med":    0.0,
            "gex_struct": 0.0,
        })
        regime = classify_regime(flat, spot=7400.0, implied_move=40.0)
        self.assertEqual(regime["regime"], "untethered")


# ────────────────────────────────────────────────────────────────────────
#  compute_and_upsert_landscape — mocked DB
# ────────────────────────────────────────────────────────────────────────

_Col = namedtuple("_Col", ["name"])

# Column order returned by the helper's orats_oi_gamma query.
_OI_GAMMA_COLS = [
    "discounted_level", "strike", "expir_date", "dte", "stock_price",
    "call_oi", "put_oi", "gamma", "gex_call", "gex_put",
]


def _oi_gamma_row(discounted_level, dte, gex_call, gex_put, stock_price=7395.0):
    """One orats_oi_gamma row in the helper's SELECT column order."""
    return (
        discounted_level, discounted_level, dt.date(2026, 6, 19), dte,
        stock_price, 1000, 1000, 0.01, gex_call, gex_put,
    )


def _mock_conn(fetch_rows, description=None):
    """A psycopg-shaped connection whose cursor yields fetch_rows.

    Returns (conn, cursor) — the cursor's execute call list is the assertion
    surface for the UPSERT.
    """
    cur = MagicMock()
    cur.fetchall.return_value = fetch_rows
    cur.description = description or [_Col(c) for c in _OI_GAMMA_COLS]

    cursor_cm = MagicMock()
    cursor_cm.__enter__.return_value = cur
    cursor_cm.__exit__.return_value = False

    conn = MagicMock()
    conn.cursor.return_value = cursor_cm
    return conn, cur


class TestComputeAndUpsertLandscape(unittest.TestCase):
    def test_upsert_executes_with_expected_payload_shape(self):
        rows = [
            _oi_gamma_row(7350.0, 5, 4e11, 1e11),
            _oi_gamma_row(7400.0, 20, 8e11, 1e11),
            _oi_gamma_row(7480.0, 90, 6e11, 2e11),
        ]
        conn, cur = _mock_conn(rows)

        summary = compute_and_upsert_landscape(
            conn, "SPX", dt.date(2026, 5, 20),
            spread_coef=8.0, range_pts=200.0, step_pts=1.0,
            version="test-version",
        )

        # Two execute() calls: the SELECT, then the UPSERT.
        self.assertEqual(cur.execute.call_count, 2)
        upsert_sql, upsert_params = cur.execute.call_args_list[1].args
        self.assertEqual(upsert_sql, _UPSERT_SQL)
        self.assertEqual(len(upsert_params), 10)

        (ticker, trade_date, landscape_j, walls_j, peaks_j,
         spread_coef, range_pts, step_pts, table_spot, version) = upsert_params

        self.assertEqual(ticker, "SPX")
        self.assertEqual(trade_date, dt.date(2026, 5, 20))
        self.assertIsInstance(landscape_j, Jsonb)
        self.assertIsInstance(walls_j, Jsonb)
        self.assertIsInstance(peaks_j, Jsonb)
        # range_pts=200, step_pts=1 → 401 landscape points.
        self.assertEqual(len(landscape_j.obj), 401)
        self.assertEqual(
            set(peaks_j.obj.keys()), {"0DTE", "1-7 DTE", "8-30 DTE", "30+ DTE"}
        )
        self.assertEqual(spread_coef, 8.0)
        self.assertEqual(range_pts, 200.0)
        self.assertEqual(step_pts, 1.0)
        self.assertEqual(table_spot, 7395.0)   # stock_price of the first row
        self.assertEqual(version, "test-version")

        # The returned summary mirrors what was written.
        self.assertEqual(summary["n_landscape"], 401)
        self.assertEqual(summary["table_spot"], 7395.0)
        self.assertEqual(summary["version"], "test-version")

    def test_landscape_record_shape(self):
        conn, cur = _mock_conn([_oi_gamma_row(7400.0, 20, 8e11, 1e11)])
        compute_and_upsert_landscape(
            conn, "SPX", dt.date(2026, 5, 20), version="v",
        )
        _, upsert_params = cur.execute.call_args_list[1].args
        first_point = upsert_params[2].obj[0]
        self.assertEqual(
            set(first_point.keys()),
            {"price", "gex_total", "gex_0dte", "gex_near", "gex_med", "gex_struct"},
        )
        # Values must be JSON-native floats, not numpy scalars.
        for value in first_point.values():
            self.assertIsInstance(value, float)

    def test_raises_when_no_oi_gamma_rows(self):
        conn, _ = _mock_conn([])
        with self.assertRaises(ValueError):
            compute_and_upsert_landscape(
                conn, "SPX", dt.date(2026, 5, 20), version="v",
            )

    def test_raises_when_stock_price_null(self):
        row = _oi_gamma_row(7400.0, 20, 8e11, 1e11, stock_price=None)
        conn, _ = _mock_conn([row])
        with self.assertRaises(ValueError):
            compute_and_upsert_landscape(
                conn, "SPX", dt.date(2026, 5, 20), version="v",
            )


# ────────────────────────────────────────────────────────────────────────
#  classify_confluence_quality — CR-011 calibration set
# ────────────────────────────────────────────────────────────────────────

class TestClassifyConfluenceQuality(unittest.TestCase):
    """The recalibrated quality classifier against the labeled calibration
    set in fixtures/confluence_calibration.json. Each test loads one confirmed
    day and asserts the classifier reproduces its observed-behavior label from
    the day's top-cluster peak strength (max_gex)."""

    def _assert_day(self, trade_date: str) -> None:
        entry = _confirmed_day(trade_date)
        quality = classify_confluence_quality(entry["top_cluster_max_gex_b"] * 1e9)
        self.assertEqual(
            quality, entry["expected_quality"],
            f"{trade_date}: top-cluster max_gex "
            f"{entry['top_cluster_max_gex_b']}B classified {quality!r}, "
            f"expected {entry['expected_quality']!r}",
        )

    def test_5_06_top_cluster_is_target(self):
        self._assert_day("2026-05-06")

    def test_5_07_top_cluster_is_pin(self):
        self._assert_day("2026-05-07")

    def test_5_18_top_cluster_is_feature(self):
        self._assert_day("2026-05-18")

    def test_5_20_top_cluster_is_feature(self):
        self._assert_day("2026-05-20")


if __name__ == "__main__":
    unittest.main()
