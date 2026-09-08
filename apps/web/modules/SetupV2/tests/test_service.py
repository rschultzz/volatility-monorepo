"""Tests for apps/web/modules/SetupV2/service.py (CR-AW decision 8).

Synthetic only — no DB.

Run with:
    python -m pytest apps/web/modules/SetupV2/tests -q
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.SetupV2.service import (
    FACTOR_META,
    FEE_PER_CONTRACT_PER_LEG,
    band_for_sigma,
    expected_pnl,
    fee_points,
    harness_expiry,
    horizon_mix,
    knn_factor_rows,
    match_quality,
    net_debit_by_minute,
    next_reference_run,
    nth_business_day,
    percentile_rank,
    pick_reference_cr_id,
    quantile,
    reference_cell_dict,
    seed_for_date,
    select_reference_rows,
    verdict,
)
from packages.shared.day_features import FEATURE_NAMES
from packages.shared.probability import analogue_fair_value


def _cell(band, *, structure="debit", outcome="close", pattern=None, partition="train",
          threshold=0.05, mean_pnl=1.0, win=0.6, lo=0.4, hi=0.8, n=30, cr_id="CR-AR"):
    return {
        "cr_id": cr_id, "structure_type": structure, "outcome_type": outcome,
        "post_touch_pattern": pattern, "distance_band": band, "partition": partition,
        "threshold": threshold, "mean_pnl": mean_pnl, "win_rate": win,
        "wilson_lo": lo, "wilson_hi": hi, "n_settled": n, "n_dates": n + 1,
        "baseline_mean": mean_pnl + 0.1, "beat_baseline": -0.1,
        "created_at": dt.datetime(2026, 9, 7, 16, 0), "run_id": "r1", "mean_width_actual": 10.0,
    }


class TestVerdict(unittest.TestCase):
    def test_quote_under_max_enters(self):
        v = verdict(4.10, 5.0, listed=True)
        self.assertEqual(v["code"], "enter")
        self.assertIn("enter at the open", v["text"])

    def test_quote_equal_to_max_enters(self):
        self.assertEqual(verdict(5.0, 5.0, listed=True)["code"], "enter")

    def test_quote_above_max_skips(self):
        v = verdict(5.6, 5.0, listed=True)
        self.assertEqual(v["code"], "skip")
        self.assertIn("quote above max", v["text"])

    def test_unlisted_structure(self):
        v = verdict(None, 5.0, listed=False)
        self.assertEqual(v["code"], "no_structure")
        self.assertEqual(v["text"], "no listed structure at this expiry")

    def test_no_quote_or_no_fair_value(self):
        self.assertEqual(verdict(None, 5.0, listed=True)["code"], "no_quote")
        self.assertEqual(verdict(4.1, None, listed=True)["code"], "no_quote")

    def test_never_says_wait(self):
        for q, m, listed in ((4.1, 5.0, True), (5.6, 5.0, True), (None, 5.0, True), (None, None, False)):
            self.assertNotIn("wait", verdict(q, m, listed=listed)["text"].lower())


class TestReferenceSelection(unittest.TestCase):
    def test_latest_ref_wins_over_cr_ar(self):
        ids = [("CR-AR", dt.datetime(2026, 9, 7)), ("REF-2026-09", dt.datetime(2026, 10, 1)),
               ("REF-2026-10", dt.datetime(2026, 11, 1)), ("CR-AH", dt.datetime(2026, 6, 10))]
        self.assertEqual(pick_reference_cr_id(ids), "REF-2026-10")

    def test_falls_back_to_cr_ar_then_none(self):
        self.assertEqual(pick_reference_cr_id([("CR-AR", dt.datetime(2026, 9, 7)), ("CR-AP", dt.datetime(2026, 9, 6))]), "CR-AR")
        self.assertIsNone(pick_reference_cr_id([("CR-AP", dt.datetime(2026, 9, 6))]))
        self.assertIsNone(pick_reference_cr_id([]))

    def test_filters_to_debit_close_pooled(self):
        rows = [
            _cell("near"), _cell("mid"), _cell("far"), _cell("all"),
            _cell("near", structure="credit", mean_pnl=-9),
            _cell("near", outcome="touch", mean_pnl=-9),
            _cell("near", pattern="stepping-stone", mean_pnl=-9),
        ]
        cells = select_reference_rows(rows)
        self.assertEqual(set(cells), {"near", "mid", "far", "all"})
        self.assertEqual(cells["near"]["mean_pnl"], 1.0)

    def test_train_and_lowest_threshold_win(self):
        rows = [_cell("near", partition="holdout", threshold=0.0, mean_pnl=-1),
                _cell("near", threshold=0.10, mean_pnl=2.0),
                _cell("near", threshold=0.05, mean_pnl=1.5)]
        self.assertEqual(select_reference_rows(rows)["near"]["mean_pnl"], 1.5)

    def test_band_thresholds_match_the_harness(self):
        self.assertEqual(band_for_sigma(1.49), "near")
        self.assertEqual(band_for_sigma(1.5), "mid")
        self.assertEqual(band_for_sigma(1.99), "mid")
        self.assertEqual(band_for_sigma(2.0), "far")
        self.assertEqual(band_for_sigma(2.55), "far")
        self.assertIsNone(band_for_sigma(None))

    def test_cell_dict_shape(self):
        d = reference_cell_dict(_cell("near", n=38, mean_pnl=1.9116, win=0.6842, lo=0.5254, hi=0.8092))
        self.assertEqual(d["band"], "near")
        self.assertEqual(d["n"], 38)
        self.assertAlmostEqual(d["mean_pnl"], 1.9116)
        self.assertAlmostEqual(d["wilson_lo"], 0.5254)
        self.assertIsNone(reference_cell_dict(None))


class TestPercentiles(unittest.TestCase):
    def test_percentile_rank_kind_mean(self):
        corpus = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(percentile_rank(corpus, 3), 50.0)     # (2 + 3) / 2 / 5
        self.assertAlmostEqual(percentile_rank(corpus, 0), 0.0)
        self.assertAlmostEqual(percentile_rank(corpus, 6), 100.0)
        self.assertAlmostEqual(percentile_rank(corpus, 3.5), 60.0)
        self.assertIsNone(percentile_rank([], 1.0))

    def test_quantile_interpolates(self):
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 0.25), 1.75)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 0.75), 3.25)
        self.assertAlmostEqual(quantile([7], 0.5), 7)
        self.assertIsNone(quantile([], 0.5))


class TestKnnFactorRows(unittest.TestCase):
    def _corpus(self):
        return [{"cluster_1_signed_distance_sigma": s, "implied_move_1d": 20 + s, "atm_iv_percentile": None}
                for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)]

    def test_every_feature_key_has_a_row_and_metadata(self):
        rows = knn_factor_rows({}, {}, [], [])
        self.assertEqual({r["key"] for r in rows}, set(FEATURE_NAMES))
        self.assertEqual(set(FACTOR_META), set(FEATURE_NAMES))

    def test_today_percentile_band_and_in_band(self):
        today = {"cluster_1_signed_distance_sigma": 1.0, "implied_move_1d": 24.0}
        analog = [{"cluster_1_signed_distance_sigma": s, "implied_move_1d": 30} for s in (0.8, 0.9, 1.1, 1.3, 2.0)]
        rows = {r["key"]: r for r in knn_factor_rows(today, {"cluster_1_signed_distance_sigma": 3.0}, self._corpus(), analog)}
        r = rows["cluster_1_signed_distance_sigma"]
        self.assertEqual(r["weight"], 3.0)
        self.assertTrue(r["populated"])
        self.assertAlmostEqual(r["percentile"], 18.75)    # (1 + 2) / 2 / 8
        self.assertAlmostEqual(r["band_lo"], 0.9)
        self.assertAlmostEqual(r["band_hi"], 1.3)
        self.assertTrue(r["in_band"])
        self.assertEqual(r["corpus_n"], 8)
        self.assertEqual(r["analogue_n"], 5)
        im = rows["implied_move_1d"]
        self.assertEqual(im["weight"], 1.0)               # missing key → config default
        self.assertFalse(im["in_band"])                   # 24 outside [30, 30]

    def test_null_today_is_not_populated_and_not_scored(self):
        rows = {r["key"]: r for r in knn_factor_rows({"implied_move_1d": None}, {}, self._corpus(), self._corpus())}
        r = rows["atm_iv_percentile"]
        self.assertFalse(r["populated"])
        self.assertIsNone(r["percentile"])
        self.assertIsNone(r["in_band"])
        self.assertEqual(r["corpus_n"], 0)

    def test_match_quality_counts_in_band_over_scored(self):
        rows = [{"key": "a", "in_band": True}, {"key": "b", "in_band": False}, {"key": "c", "in_band": None}]
        mq = match_quality(rows)
        self.assertEqual((mq["in_band"], mq["total"]), (1, 2))
        self.assertEqual(mq["outliers"], ["b"])


class TestPnlAndFees(unittest.TestCase):
    def test_fee_points_for_a_vertical(self):
        self.assertAlmostEqual(fee_points(2), 2 * 2 * FEE_PER_CONTRACT_PER_LEG / 100)
        self.assertAlmostEqual(fee_points(2, 0.65), 0.026)

    def test_expected_pnl_subtracts_quote_and_fees(self):
        fv = analogue_fair_value([0.0, -5.0, 5.0, 20.0], 10.0, seed=20260903)   # values 10, 10, 5, 0 → fair 6.25
        p = expected_pnl(fv, 4.10, 0.026)
        self.assertAlmostEqual(p["expected"], 6.25 - 4.10 - 0.026, places=4)
        self.assertLessEqual(p["lo"], p["expected"])
        self.assertGreaterEqual(p["hi"], p["expected"])
        self.assertAlmostEqual(p["fee_pts"], 0.026)

    def test_expected_pnl_none_without_quote(self):
        fv = analogue_fair_value([0.0], 10.0, seed=1)
        self.assertIsNone(expected_pnl(fv, None, 0.026)["expected"])
        self.assertIsNone(expected_pnl({"fair": None}, 4.0, 0.026)["expected"])


class _Bar:
    def __init__(self, hh, mm, bid, ask):
        self.snapshot_pt = dt.datetime(2026, 9, 3, hh, mm, 2)
        self.bid_price = bid
        self.ask_price = ask


class TestNetDebitByMinute(unittest.TestCase):
    def test_valid_minutes_price_and_invalid_minutes_are_kept_as_gaps(self):
        bars = {
            "long":  [_Bar(6, 34, 41.7, 42.4), _Bar(6, 35, 0.0, 0.0), _Bar(6, 36, 41.0, 41.6)],
            "short": [_Bar(6, 34, 37.6, 38.3), _Bar(6, 35, 37.0, 37.6), _Bar(6, 36, 37.0, 37.6), _Bar(6, 37, 1.0, 2.0)],
        }
        rows = net_debit_by_minute(bars, 10.0)
        self.assertEqual([r["minute"] for r in rows], ["06:34", "06:35", "06:36", "06:37"])
        self.assertAlmostEqual(rows[0]["net_debit"], 4.1)
        self.assertTrue(rows[0]["valid"])
        self.assertIsNone(rows[1]["net_debit"])        # long leg invalid (zero quote)
        self.assertAlmostEqual(rows[2]["net_debit"], 4.0)
        self.assertIsNone(rows[3]["net_debit"])        # long leg missing that minute

    def test_price_outside_the_width_range_is_suppressed(self):
        bars = {"long": [_Bar(6, 34, 60.0, 61.0)], "short": [_Bar(6, 34, 37.6, 38.3)]}   # 22.55 > width 10
        rows = net_debit_by_minute(bars, 10.0)
        self.assertIsNone(rows[0]["net_debit"])
        self.assertFalse(rows[0]["valid"])

    def test_empty(self):
        self.assertEqual(net_debit_by_minute({}, 10.0), [])


class TestHarnessExpiry(unittest.TestCase):
    D = dt.date(2026, 9, 3)

    def test_nth_business_day_skips_weekends_and_holidays(self):
        # 2026-09-07 (Labor Day) is skipped: 15 sessions from 09-03 → 09-25
        self.assertEqual(nth_business_day(self.D, 15), dt.date(2026, 9, 25))
        self.assertEqual(nth_business_day(dt.date(2026, 9, 4), 1), dt.date(2026, 9, 8))

    def test_snaps_to_the_nearest_listed_expiry(self):
        listed = [dt.date(2026, 9, 18), dt.date(2026, 9, 24), dt.date(2026, 9, 25), dt.date(2026, 9, 28)]
        e = harness_expiry(self.D, listed)
        self.assertEqual(e["expiry"], dt.date(2026, 9, 25))
        self.assertEqual(e["target"], dt.date(2026, 9, 25))
        self.assertTrue(e["listed"])
        self.assertEqual(e["sessions"], 15)

    def test_nearest_when_target_unlisted_and_ties_go_later(self):
        self.assertEqual(harness_expiry(self.D, [dt.date(2026, 9, 23), dt.date(2026, 9, 28)])["expiry"], dt.date(2026, 9, 23))
        self.assertEqual(harness_expiry(self.D, [dt.date(2026, 9, 24), dt.date(2026, 9, 26)])["expiry"], dt.date(2026, 9, 26))
        self.assertEqual(harness_expiry(self.D, [dt.date(2026, 9, 2), dt.date(2026, 9, 3)])["listed"], False)   # nothing after trade_date

    def test_no_chain_returns_the_target_unlisted(self):
        e = harness_expiry(self.D, [])
        self.assertEqual(e["expiry"], dt.date(2026, 9, 25))
        self.assertFalse(e["listed"])


class TestMisc(unittest.TestCase):
    def test_horizon_mix(self):
        rows = [{"horizon_sessions": 5}, {"horizon_sessions": 20}, {"horizon_sessions": 5}, {"horizon_sessions": None}]
        self.assertEqual(horizon_mix(rows), {"5": 2, "20": 1})

    def test_next_reference_run_is_first_of_next_month(self):
        self.assertEqual(next_reference_run(dt.date(2026, 9, 7)), "2026-10-01")
        self.assertEqual(next_reference_run(dt.date(2026, 12, 31)), "2027-01-01")
        self.assertIsNone(next_reference_run(None))

    def test_seed_is_yyyymmdd(self):
        self.assertEqual(seed_for_date(dt.date(2026, 9, 3)), 20260903)


if __name__ == "__main__":
    unittest.main()
