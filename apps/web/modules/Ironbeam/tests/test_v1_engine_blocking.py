"""CR-DU item 4b — the v1-trades overlay's blocked dates (modules/Ironbeam/v1_engine.py:blocked_reason). No DB.

    cd apps/web && PYTHONPATH=../..:. python -m unittest modules.Ironbeam.tests.test_v1_engine_blocking -v
"""
from __future__ import annotations

import datetime as dt
import unittest

from modules.Ironbeam import v1_engine as V

D = dt.date


class TestBlocked(unittest.TestCase):
    def test_holdout_both_ends(self):
        for d in (D(2026, 6, 6), D(2026, 8, 3), D(2026, 10, 26)):
            self.assertEqual(V.blocked_reason(d, set()), "holdout date: not shown")

    def test_train_and_forward_open(self):
        for d in (D(2026, 6, 5), D(2026, 3, 10), D(2026, 10, 27), D(2026, 10, 30)):
            self.assertIsNone(V.blocked_reason(d, set()))

    def test_reserved_before_train(self):
        self.assertEqual(V.blocked_reason(D(2025, 7, 1), set()), "reserved date (R-a / R-b): not shown")
        self.assertEqual(V.blocked_reason(D(2024, 5, 1), set()), "reserved date (R-a / R-b): not shown")

    def test_set2_in_progress(self):
        open_dates = {D(2026, 2, 10), D(2026, 4, 1)}
        self.assertEqual(V.blocked_reason(D(2026, 2, 10), open_dates), "set 2 in progress")
        self.assertIsNone(V.blocked_reason(D(2026, 2, 11), open_dates))
        self.assertIsNone(V.blocked_reason(D(2026, 2, 10), set()))           # set 2 reviewed: no longer blocked

    def test_holdout_wins_over_set2(self):
        self.assertEqual(V.blocked_reason(D(2026, 7, 1), {D(2026, 7, 1)}), "holdout date: not shown")

    def test_warmup(self):
        warm = {D(2026, 1, 5)}
        self.assertEqual(V.blocked_reason(D(2026, 1, 5), set(), warm), "warm-up session: no prior-sessions baseline")

    def test_prior_file_matches_rule(self):
        pri = V.prior_baselines()
        days = sorted(pri["sessions"])
        self.assertEqual(days[0], "2026-02-03")
        self.assertLessEqual(days[-1], "2026-06-05")
        self.assertTrue(all(pri["sessions"][d]["databento"]["sessions"] >= pri["meta"]["warmup"] for d in days))
        self.assertTrue(all(pri["sessions"][d]["databento"]["last"] < d for d in days))     # prior sessions only


if __name__ == "__main__":
    unittest.main()
