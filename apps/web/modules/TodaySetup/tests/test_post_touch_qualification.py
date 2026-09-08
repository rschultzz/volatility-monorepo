"""Unit tests for post-touch direction qualification (CR-025 / CR-I Step 4).

Tests cover:
  - dte_to_timeframe() mapping
  - credit_direction_qualifies() and debit_direction_qualifies() helpers
  - apply_direction_qualification() service function — CR-AV: advisory only, never filters

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


# ── apply_direction_qualification (CR-AV: advisory only) ──────────────────────

class TestApplyDirectionQualification(unittest.TestCase):
    """CR-AV decisions 2 and 4: the function never filters or promotes; it
    annotates post_touch with advisory_only / advisory and returns the
    proposals unchanged (no confidence_badge)."""

    def _two_magnet_proposals(self, dte: int = 15) -> list[dict]:
        return [_make_proposal(CREDIT_ID, dte), _make_proposal(DEBIT_ID, dte)]

    def _assert_unchanged(self, props, result):
        self.assertEqual(result, props)
        for p in result:
            self.assertNotIn("confidence_badge", p)

    def test_stepping_stone_does_not_filter_or_badge(self):
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", above_t15=0.55, below_t15=0.20)
        self._assert_unchanged(props, apply_direction_qualification(props, sp))

    def test_touch_and_reject_does_not_filter_or_badge(self):
        props = self._two_magnet_proposals()
        sp = _make_sp("touch-and-reject", below_t15=0.55, above_t15=0.20)
        self._assert_unchanged(props, apply_direction_qualification(props, sp))

    def test_stepping_stone_and_mixed_yield_identical_proposal_sets(self):
        # decision 5: label changes the advisory text only
        props = self._two_magnet_proposals()
        r1 = apply_direction_qualification([dict(p) for p in props], _make_sp("stepping-stone", above_t15=0.55))
        r2 = apply_direction_qualification([dict(p) for p in props], _make_sp("mixed", above_t15=0.30, below_t15=0.30))
        self.assertEqual(r1, r2)
        self.assertEqual([p["template_id"] for p in r1], [CREDIT_ID, DEBIT_ID])

    def test_insufficient_and_zero_dte_do_not_badge(self):
        for mode in ("insufficient", "zero_dte_corpus_insufficient"):
            props = self._two_magnet_proposals()
            sp = _make_sp("stepping-stone", filter_mode=mode)
            result = apply_direction_qualification(props, sp)
            self._assert_unchanged(props, result)
            self.assertTrue(sp["post_touch"]["advisory_only"])
            self.assertEqual(sp["post_touch"]["filter_mode"], mode)   # kept for audit

    def test_advisory_block_label_n_timeframe_direction_fraction(self):
        props = self._two_magnet_proposals(dte=5)     # 4–9 → t5
        sp = _make_sp("stepping-stone", above_t5=0.61, below_t5=0.20)
        apply_direction_qualification(props, sp)
        pt = sp["post_touch"]
        self.assertTrue(pt["advisory_only"])
        adv = pt["advisory"]
        self.assertEqual(adv["pattern_label"], "stepping-stone")
        self.assertEqual(adv["n"], 10)
        self.assertEqual(adv["n_pooled"], 10)
        self.assertEqual(adv["timeframe"], "t5")
        self.assertEqual(adv["direction"], "above")
        self.assertAlmostEqual(adv["fraction"], 0.61)
        self.assertAlmostEqual(adv["wilson_lo"], 0.61)
        self.assertAlmostEqual(adv["wilson_hi"], 0.71)

    def test_advisory_direction_follows_regime(self):
        props = self._two_magnet_proposals(dte=15)
        sp = _make_sp("stepping-stone", regime_kind="magnet-below", below_t15=0.58)
        apply_direction_qualification(props, sp)
        adv = sp["post_touch"]["advisory"]
        self.assertEqual((adv["timeframe"], adv["direction"]), ("t15", "below"))
        self.assertAlmostEqual(adv["fraction"], 0.58)

    def test_existing_post_touch_fields_kept(self):
        props = self._two_magnet_proposals()
        sp = _make_sp("mixed")
        before = dict(sp["post_touch"])
        apply_direction_qualification(props, sp)
        for k, v in before.items():
            self.assertEqual(sp["post_touch"][k], v)

    def test_non_magnet_regime_still_annotates_without_direction(self):
        props = [_make_proposal("pin_butterfly_tight", 15)]
        props[0]["source"] = {"type": "cluster"}
        sp = _make_sp("touch-and-pin", regime_kind="magnetic-pin")
        result = apply_direction_qualification(props, sp)
        self.assertEqual(result, props)
        adv = sp["post_touch"]["advisory"]
        self.assertTrue(sp["post_touch"]["advisory_only"])
        self.assertIsNone(adv["direction"]); self.assertIsNone(adv["fraction"])
        self.assertEqual(adv["pattern_label"], "touch-and-pin")

    def test_no_post_touch_passes_through(self):
        props = self._two_magnet_proposals()
        sp = {"outcome_status": "ok", "regime_kind": "magnet-above", "post_touch": None}
        self.assertEqual(apply_direction_qualification(props, sp), props)
        self.assertIsNone(sp["post_touch"])

    def test_n_falls_back_to_total_touchers(self):
        props = self._two_magnet_proposals()
        sp = _make_sp("stepping-stone", filter_mode="pooled-fallback")
        sp["post_touch"]["same_bucket_n"] = None
        sp["post_touch"]["total_touchers"] = 23
        apply_direction_qualification(props, sp)
        self.assertEqual(sp["post_touch"]["advisory"]["n"], 23)

    def test_pin_proposals_preserved_with_magnet(self):
        pin_prop = {"template_id": "pin_butterfly_tight", "template_kind": "butterfly",
                    "expiry_dte_target": 15, "source": {"type": "cluster"},
                    "rationale": "pin", "legs": []}
        props = [pin_prop, _make_proposal(CREDIT_ID), _make_proposal(DEBIT_ID)]
        sp = _make_sp("stepping-stone", above_t15=0.55, below_t15=0.20)
        self.assertEqual(apply_direction_qualification(props, sp), props)


if __name__ == "__main__":
    unittest.main()
