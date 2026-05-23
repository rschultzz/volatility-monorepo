"""Unit tests for apps/web/modules/Analogues/service.py (CR-013).

Run with:
    python -m unittest apps.web.modules.Analogues.tests.test_service
or, from the packages/ pattern used in the rest of the suite:
    python -m unittest discover -s apps/web/modules/Analogues -p 'test_*.py'
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.Analogues.service import (
    feature_stats,
    rank_analogues,
    similarity_distance,
)
from packages.shared.day_features import FEATURE_NAMES


def _zero_vec() -> dict:
    return {name: 0 for name in FEATURE_NAMES}


class TestSimilarityDistance(unittest.TestCase):

    def test_identical_vectors_distance_zero(self):
        v = _zero_vec()
        v["implied_move_1d"] = 40.0
        stats = feature_stats([v])
        self.assertEqual(similarity_distance(v, v, stats), 0.0)

    def test_null_features_skipped(self):
        """vol-surface placeholders (None on both sides) don't contribute."""
        v1 = _zero_vec()
        v2 = _zero_vec()
        v1["atm_iv_percentile"] = None
        v2["atm_iv_percentile"] = None
        v1["implied_move_1d"] = 40.0
        v2["implied_move_1d"] = 45.0
        stats = feature_stats([v1, v2])
        d = similarity_distance(v1, v2, stats)
        self.assertTrue(math.isfinite(d))
        self.assertGreater(d, 0)

    def test_zero_active_features_returns_inf(self):
        """Two vectors where every shared key is None → inf."""
        v1 = {name: None for name in FEATURE_NAMES}
        v2 = {name: None for name in FEATURE_NAMES}
        d = similarity_distance(v1, v2, feature_stats([]))
        self.assertEqual(d, float("inf"))


class TestRankAnalogues(unittest.TestCase):

    def test_sorted_closest_first(self):
        anchor = _zero_vec()
        anchor["implied_move_1d"] = 40.0
        candidates = [
            ("2026-05-06", {**_zero_vec(), "implied_move_1d": 50.0}),  # far
            ("2026-05-07", {**_zero_vec(), "implied_move_1d": 41.0}),  # closest
            ("2026-05-08", {**_zero_vec(), "implied_move_1d": 45.0}),  # mid
        ]
        ranked = rank_analogues(anchor, candidates, k=3)
        dates = [d for (d, _) in ranked]
        self.assertEqual(dates, ["2026-05-07", "2026-05-08", "2026-05-06"])
        # Distances must be ascending
        distances = [s for (_, s) in ranked]
        self.assertEqual(distances, sorted(distances))

    def test_exclude_date_drops_anchor(self):
        anchor = _zero_vec()
        anchor["implied_move_1d"] = 40.0
        candidates = [
            ("2026-05-22", anchor),                              # anchor itself
            ("2026-05-07", {**_zero_vec(), "implied_move_1d": 50.0}),
        ]
        ranked = rank_analogues(anchor, candidates, k=5,
                                exclude_date="2026-05-22")
        dates = [d for (d, _) in ranked]
        self.assertEqual(dates, ["2026-05-07"])

    def test_k_truncation(self):
        anchor = _zero_vec()
        candidates = [
            (f"2026-05-{i:02d}", {**_zero_vec(), "implied_move_1d": float(i)})
            for i in range(1, 11)
        ]
        ranked = rank_analogues(anchor, candidates, k=3)
        self.assertEqual(len(ranked), 3)

    def test_empty_candidates_returns_empty(self):
        anchor = _zero_vec()
        ranked = rank_analogues(anchor, [], k=5)
        self.assertEqual(ranked, [])


if __name__ == "__main__":
    unittest.main()
