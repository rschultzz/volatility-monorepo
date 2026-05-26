"""Unit tests for post-touch direction qualification (CR-025 / CR-I Step 4).

Tests cover:
  - dte_to_timeframe() mapping
  - credit_direction_qualifies() and debit_direction_qualifies() helpers
  - apply_direction_qualification() service function — all decision branches

Run with:
    python -m unittest apps.web.modules.TodaySetup.tests.test_post_touch_qualification
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.post_touch_qualification import (
    dte_to_timeframe,
    credit_direction_qualifies,
    debit_direction_qualifies,
)
from apps.web.modules.TodaySetup.service import apply_direction_qualification


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_post_touch(
    pattern_label: str,
    filter_mode: str = "strict",
    above_t15: float = 0.50,
    below_t15: float = 0.30,
    above_t5:  float = 0.50,
    below_t5:  float = 0.30,
    above_t1:  float = 0.50,
    below_t1:  float = 0.30,
) -> dict:
    """Build a minimal post_touch dict with Wilson CIs set to [fraction, fraction+0.1]."""
    def _ci(v):
        return [v, min(v + 0.1, 1.0)]

    return {
        "filter_mode": filter_mode,
        "pattern_label": pattern_label,
        "same_bucket_n": 10,
        "total_touchers": 10,
        "fractions": {
            "t1":  {"below": below_t1,  "at": 0.10, "above": above_t1},
            "t5":  {"below": below_t5,  "at": 0.10, "above": above_t5},
            "t15": {"below": below_t15, "at": 0.10, "above": above_t15},
        },
        "wilson_cis": {
            "t1":  {
                "below": _ci(below_t1),
                "at":    [0.03, 0.28],
                "above": _ci(above_t1),
            },
            "t5":  {
                "below": _ci(below_t5),
                "at":    [0.03, 0.28],
                "above": _ci(above_t5),
            },
            "t15": {
                "below": _ci(below_t15),
                "at":    [0.03, 0.28],
                "above": _ci(above_t15),
            },
        },
    }


def _make_sp(
    pattern_label: str,
    regime_kind: str = "magnet-above",
    filter_mode: str = "strict",
    **pt_kwargs,
) -> dict:
    """Build a minimal structural_probability dict."""
    return {
        "outcome_status": "ok",
        "regime_kind": regime_kind,
        "post_touch": _make_post_touch(pattern_label, filter_mode=filter_mode, **pt_kwargs),
    }


def _make_proposal(template_id: str, dte: int = 15) -> dict:
    return {
        "template_id": template_id,
        "template_kind": "spread",
        "expiry_dte_target": dte,
        "source": {"type": "regime_target", "regime": "magnet-above"},
        "rationale": "test",
        "legs": [],
    }


CREDIT_ID = "directional_spread_to_target"
DEBIT_ID  = "debit_spread_to_target"


# ── dte_to_timeframe ──────────────────────────────────────────────────────────

class TestDteToTimeframe(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(dte_to_timeframe(None))

    def test_0dte_maps_to_t1(self):
        self.assertEqual(dte_to_timeframe(0), "t1")

    def test_3dte_maps_to_t1(self):
        self.assertEqual(dte_to_timeframe(3), "t1")

    def test_4dte_maps_to_t5(self):
        self.assertEqual(dte_to_timeframe(4), "t5")

    def test_9dte_maps_to_t5(self):
        self.assertEqual(dte_to_timeframe(9), "t5")

    def test_10dte_maps_to_t15(self):
        self.assertEqual(dte_to_timeframe(10), "t15")

    def test_15dte_maps_to_t15(self):
        self.assertEqual(dte_to_timeframe(15), "t15")

    def test_45dte_maps_to_t15(self):
        self.assertEqual(dte_to_timeframe(45), "t15")


# ── credit_direction_qualifies ────────────────────────────────────────────────

class TestCreditDirectionQualifies(unittest.TestCase):
    """Spec cases + edge conditions for credit qualification."""

    def test_touch_and_reject_magnet_above_qualifies(self):
        # touch-and-reject + magnet-above + below wilson_lo=0.55 → credit qualifies
        pt = _make_post_touch("touch-and-reject", below_t15=0.55)
        self.assertTrue(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_slow_revert_magnet_below_qualifies(self):
        # slow-revert + magnet-below + above wilson_lo=0.45 → credit qualifies
        pt = _make_post_touch("slow-revert", above_t15=0.45)
        self.assertTrue(credit_direction_qualifies(pt, "magnet-below", 15))

    def test_overshoot_then_revert_magnet_above_qualifies(self):
        # overshoot-then-revert + magnet-above + below wilson_lo=0.42 → credit qualifies
        pt = _make_post_touch("overshoot-then-revert", below_t15=0.42)
        self.assertTrue(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_wrong_pattern_does_not_qualify(self):
        # stepping-stone is a debit pattern — credit should not qualify
        pt = _make_post_touch("stepping-stone", below_t15=0.55)
        self.assertFalse(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_mixed_pattern_does_not_qualify(self):
        pt = _make_post_touch("mixed", below_t15=0.55)
        self.assertFalse(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_below_wilson_floor_does_not_qualify(self):
        # touch-and-reject but wilson_lo=0.35 < 0.40 floor
        pt = _make_post_touch("touch-and-reject", below_t15=0.35)
        self.assertFalse(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_at_wilson_floor_does_not_qualify(self):
        # Exactly 0.40 is NOT strictly greater than — must NOT qualify
        pt = _make_post_touch("touch-and-reject", below_t15=0.40)
        self.assertFalse(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_magnet_below_uses_above_fraction(self):
        # magnet-below: reversion direction is "above"
        # above_t15=0.55 qualifies; below_t15=0.55 is irrelevant
        pt = _make_post_touch("touch-and-reject", above_t15=0.55, below_t15=0.20)
        self.assertTrue(credit_direction_qualifies(pt, "magnet-below", 15))

    def test_magnet_below_wrong_direction_does_not_qualify(self):
        # magnet-below: checks above fraction; below_t15 high is irrelevant
        pt = _make_post_touch("touch-and-reject", above_t15=0.30, below_t15=0.55)
        self.assertFalse(credit_direction_qualifies(pt, "magnet-below", 15))

    def test_dte_selects_correct_timeframe(self):
        # DTE=3 → t1; only t1 fraction qualifies, t15 does not
        pt = _make_post_touch("touch-and-reject", below_t1=0.55, below_t15=0.20)
        self.assertTrue(credit_direction_qualifies(pt, "magnet-above", 3))
        self.assertFalse(credit_direction_qualifies(pt, "magnet-above", 15))

    def test_none_dte_does_not_qualify(self):
        pt = _make_post_touch("touch-and-reject", below_t15=0.55)
        self.assertFalse(credit_direction_qualifies(pt, "magnet-above", None))


# ── debit_direction_qualifies ─────────────────────────────────────────────────

class TestDebitDirectionQualifies(unittest.TestCase):
    """Spec cases + edge conditions for debit qualification."""

    def test_stepping_stone_magnet_above_qualifies(self):
        # stepping-stone + magnet-above + above wilson_lo=0.55 → debit qualifies
        pt = _make_post_touch("stepping-stone", above_t15=0.55)
        self.assertTrue(debit_direction_qualifies(pt, "magnet-above", 15))

    def test_touch_and_pin_magnet_above_qualifies(self):
        # touch-and-pin + magnet-above + above wilson_lo=0.50 → debit qualifies
        pt = _make_post_touch("touch-and-pin", above_t15=0.50)
        self.assertTrue(debit_direction_qualifies(pt, "magnet-above", 15))

    def test_wrong_pattern_does_not_qualify(self):
        # touch-and-reject is a credit pattern — debit should not qualify
        pt = _make_post_touch("touch-and-reject", above_t15=0.55)
        self.assertFalse(debit_direction_qualifies(pt, "magnet-above", 15))

    def test_below_wilson_floor_does_not_qualify(self):
        # stepping-stone but wilson_lo=0.35 < 0.40 floor
        pt = _make_post_touch("stepping-stone", above_t15=0.35)
        self.assertFalse(debit_direction_qualifies(pt, "magnet-above", 15))

    def test_magnet_below_uses_below_fraction(self):
        # magnet-below: continuation direction is "below"
        pt = _make_post_touch("stepping-stone", below_t15=0.55, above_t15=0.20)
        self.assertTrue(debit_direction_qualifies(pt, "magnet-below", 15))

    def test_magnet_below_wrong_direction_does_not_qualify(self):
        pt = _make_post_touch("stepping-stone", below_t15=0.20, above_t15=0.55)
        self.assertFalse(debit_direction_qualifies(pt, "magnet-below", 15))

    def test_dte_t5_selects_t5_timeframe(self):
        # DTE=7 → t5; t5 fraction qualifies, t15 does not
        pt = _make_post_touch("stepping-stone", above_t5=0.55, above_t15=0.20)
        self.assertTrue(debit_direction_qualifies(pt, "magnet-above", 7))
        self.assertFalse(debit_direction_qualifies(pt, "magnet-above", 15))

    def test_none_dte_does_not_qualify(self):
        pt = _make_post_touch("stepping-stone", above_t15=0.55)
        self.assertFalse(debit_direction_qualifies(pt, "magnet-above", None))


# ── apply_direction_qualification ─────────────────────────────────────────────

class TestApplyDirectionQualification(unittest.TestCase):
    """End-to-end tests for the service-layer orchestration function."""

    def _two_magnet_proposals(self, dte: int = 15) -> list[dict]:
        """Standard two-proposal list: one credit + one debit."""
        return [
            _make_proposal(CREDIT_ID, dte),
            _make_proposal(DEBIT_ID, dte),
        ]

    # ── Corpus-insufficient branches ──────────────────────────────────────────

    def test_insufficient_badges_all_passes_through(self):
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", filter_mode="insufficient")
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertIn("low-confidence", p["confidence_badge"])

    def test_zero_dte_corpus_badges_all_passes_through(self):
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", filter_mode="zero_dte_corpus_insufficient")
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertEqual(p["confidence_badge"], "0DTE corpus insufficient")

    # ── Pin regime (unaffected) ───────────────────────────────────────────────

    def test_magnetic_pin_regime_passes_through_unchanged(self):
        props = [_make_proposal("pin_butterfly_tight", 15)]
        sp = {"outcome_status": "ok", "regime_kind": "magnetic-pin", "post_touch": None}
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertNotIn("confidence_badge", result[0])

    # ── Debit-only direction ──────────────────────────────────────────────────

    def test_stepping_stone_high_above_emits_debit_only(self):
        # stepping-stone + magnet-above + above wilson_lo=0.55 → debit-only
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", above_t15=0.55, below_t15=0.20)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["template_id"], DEBIT_ID)
        self.assertEqual(result[0]["confidence_badge"], "debit-to-target supported")

    def test_touch_and_pin_high_above_emits_debit_only(self):
        # touch-and-pin + magnet-above + above wilson_lo=0.50 → debit qualifies
        props = self._two_magnet_proposals()
        sp = _make_sp("touch-and-pin", above_t15=0.50, below_t15=0.20)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["template_id"], DEBIT_ID)
        self.assertEqual(result[0]["confidence_badge"], "debit-to-target supported")

    # ── Credit-only direction ─────────────────────────────────────────────────

    def test_touch_and_reject_high_below_emits_credit_only(self):
        # touch-and-reject + magnet-above + below wilson_lo=0.55 → credit-only
        props = self._two_magnet_proposals()
        sp = _make_sp("touch-and-reject", below_t15=0.55, above_t15=0.20)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["template_id"], CREDIT_ID)
        self.assertEqual(result[0]["confidence_badge"], "credit-fade supported")

    def test_slow_revert_magnet_below_emits_credit_only(self):
        # slow-revert + magnet-below + above wilson_lo=0.45 → credit qualifies
        props = [_make_proposal(CREDIT_ID, 15), _make_proposal(DEBIT_ID, 15)]
        # For magnet-below, override source regime
        for p in props:
            p["source"]["regime"] = "magnet-below"
        sp = _make_sp("slow-revert", regime_kind="magnet-below", above_t15=0.45, below_t15=0.20)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["template_id"], CREDIT_ID)
        self.assertEqual(result[0]["confidence_badge"], "credit-fade supported")

    # ── Mixed / neither qualifies ─────────────────────────────────────────────

    def test_mixed_pattern_emits_both_with_mixed_badge(self):
        # mixed pattern → both emitted with mixed badge
        props = self._two_magnet_proposals()
        sp = _make_sp("mixed", above_t15=0.30, below_t15=0.30)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertEqual(p["confidence_badge"], "mixed pattern — no clear direction")

    def test_stepping_stone_low_wilson_emits_both_with_mixed_badge(self):
        # stepping-stone but wilson_lo=0.35 < 0.40 floor → neither qualifies → both emitted
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", above_t15=0.35, below_t15=0.25)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 2)
        for p in result:
            self.assertEqual(p["confidence_badge"], "mixed pattern — no clear direction")

    # ── Magnet-below symmetry ─────────────────────────────────────────────────

    def test_magnet_below_debit_uses_below_fraction(self):
        # magnet-below + stepping-stone + below wilson_lo=0.55 → debit qualifies
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", regime_kind="magnet-below", below_t15=0.55, above_t15=0.20)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["template_id"], DEBIT_ID)

    def test_magnet_below_credit_uses_above_fraction(self):
        # magnet-below + touch-and-reject + above wilson_lo=0.55 → credit qualifies
        props = self._two_magnet_proposals()
        sp = _make_sp("touch-and-reject", regime_kind="magnet-below", above_t15=0.55, below_t15=0.20)
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["template_id"], CREDIT_ID)

    # ── No post_touch data ────────────────────────────────────────────────────

    def test_no_post_touch_passes_through(self):
        props = self._two_magnet_proposals()
        sp = {"outcome_status": "ok", "regime_kind": "magnet-above", "post_touch": None}
        result = apply_direction_qualification(props, sp)
        self.assertEqual(len(result), 2)

    # ── Non-magnet proposals preserved ───────────────────────────────────────

    def test_pin_proposals_preserved_when_magnet_filtered(self):
        pin_prop = {"template_id": "pin_butterfly_tight", "template_kind": "butterfly",
                    "expiry_dte_target": 15, "source": {"type": "cluster"},
                    "rationale": "pin", "legs": []}
        props = [pin_prop, _make_proposal(CREDIT_ID), _make_proposal(DEBIT_ID)]
        sp = _make_sp("stepping-stone", above_t15=0.55, below_t15=0.20)
        result = apply_direction_qualification(props, sp)
        # pin preserved + 1 debit
        self.assertEqual(len(result), 2)
        template_ids = {p["template_id"] for p in result}
        self.assertIn("pin_butterfly_tight", template_ids)
        self.assertIn(DEBIT_ID, template_ids)


if __name__ == "__main__":
    unittest.main()
