"""Tests for packages/shared/level_placement.py (CR-BT step 1).

Run with:
    python -m pytest packages/shared/tests/test_level_placement.py -v
"""
from __future__ import annotations

import datetime as dt
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.level_placement import (
    CORRECTED_LEVELS_FIRST_SESSION,
    bucket_levels,
    corrected_levels_defined,
    es_snapshot_bar_start_utc,
    place_levels,
)

# 2026-10-23 worked day (Step 0 Q4): ES 15:45 ET close 7686.00, S0 7649.40.
ES = 7686.00


def _rows(*pairs):
    return pd.DataFrame({"strike": [p[0] for p in pairs], "stock_price": [p[1] for p in pairs]})


class TestPlaceLevels(unittest.TestCase):

    def test_matched_expiry_returns_strike(self):
        """F_e == P_target (the expiry that matches the ES contract) → L == K."""
        out = place_levels(_rows((7700.0, ES), (7500.0, ES)), ES)
        np.testing.assert_allclose(out["level_corrected"], [7700.0, 7500.0], rtol=0, atol=1e-9)

    def test_0dte_shifts_by_about_the_basis(self):
        """0DTE F_e ≈ S0 → shift ≈ ES − S0 (+36.6 on the worked day), scaled by K/S0."""
        out = place_levels(_rows((7650.0, 7649.40)), ES)
        shift = float(out["level_corrected"].iloc[0]) - 7650.0
        self.assertAlmostEqual(shift, 7650.0 * (ES / 7649.40 - 1.0), places=9)
        self.assertTrue(35.0 < shift < 38.0)

    def test_expiry_after_es_shifts_down(self):
        """F_e above P_target (expiry after the ES contract) → L below K."""
        out = place_levels(_rows((8000.0, 7840.02)), ES)  # 2027-06-17 row, Step 0 Q4
        level = float(out["level_corrected"].iloc[0])
        self.assertLess(level, 8000.0)
        self.assertAlmostEqual(level, 8000.0 * ES / 7840.02, places=9)  # 7842.84

    def test_input_not_mutated_and_columns_kept(self):
        rows = _rows((7700.0, 7655.30)).assign(expir_date=[dt.date(2026, 10, 30)])
        out = place_levels(rows, ES)
        self.assertNotIn("level_corrected", rows.columns)
        self.assertEqual(list(out.columns), ["strike", "stock_price", "expir_date", "level_corrected"])

    def test_empty(self):
        out = place_levels(pd.DataFrame(columns=["strike", "stock_price"]), ES)
        self.assertTrue(out.empty)
        self.assertIn("level_corrected", out.columns)

    def test_nan_zero_negative_forward_or_strike(self):
        out = place_levels(
            _rows((7700.0, np.nan), (7700.0, 0.0), (7700.0, -1.0), (np.nan, 7650.0), (0.0, 7650.0), (7700.0, 7650.0)),
            ES,
        )
        lv = out["level_corrected"].to_numpy()
        self.assertTrue(np.isnan(lv[:5]).all())
        self.assertTrue(np.isfinite(lv[5]))

    def test_bad_p_target(self):
        for p in (None, math.nan, 0.0, -5.0, "x", math.inf):
            out = place_levels(_rows((7700.0, 7650.0)), p)
            self.assertTrue(out["level_corrected"].isna().all(), p)

    def test_decimal_strings_coerced(self):
        """orats_oi_gamma strike/stock_price are NUMERIC → may arrive as Decimal."""
        from decimal import Decimal
        out = place_levels(_rows((Decimal("7700"), Decimal("7686.00"))), ES)
        self.assertAlmostEqual(float(out["level_corrected"].iloc[0]), 7700.0, places=9)


class TestBucketLevels(unittest.TestCase):

    def test_half_away_from_zero_like_postgres(self):
        np.testing.assert_array_equal(bucket_levels([7680.5, 7681.5, 7680.49, 7680.51], 1.0),
                                      [7681.0, 7682.0, 7680.0, 7681.0])

    def test_bucket_size_and_floor_of_one(self):
        np.testing.assert_array_equal(bucket_levels([7682.4, 7687.6], 5.0), [7680.0, 7690.0])
        np.testing.assert_array_equal(bucket_levels([7680.5], 0.25), [7681.0])  # bucket floored at 1

    def test_nan_passthrough(self):
        self.assertTrue(np.isnan(bucket_levels([np.nan], 1.0)[0]))


class TestSnapshotAndEra(unittest.TestCase):

    def test_snapshot_bar_dst(self):
        # EDT (UTC−4) and EST (UTC−5)
        self.assertEqual(es_snapshot_bar_start_utc(dt.date(2026, 10, 22)), dt.datetime(2026, 10, 22, 19, 45))
        self.assertEqual(es_snapshot_bar_start_utc(dt.date(2026, 1, 15)), dt.datetime(2026, 1, 15, 20, 45))
        # US DST starts 2026-03-08; Europe-style offsets would be wrong on 03-09
        self.assertEqual(es_snapshot_bar_start_utc(dt.date(2026, 3, 9)), dt.datetime(2026, 3, 9, 19, 45))

    def test_era_floor(self):
        self.assertEqual(CORRECTED_LEVELS_FIRST_SESSION, dt.date(2024, 4, 3))
        self.assertFalse(corrected_levels_defined(dt.date(2024, 4, 2)))
        self.assertTrue(corrected_levels_defined(dt.date(2024, 4, 3)))


if __name__ == "__main__":
    unittest.main()
