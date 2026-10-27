"""CR-CQ — Dash gex_overlay Min |GEX| (B) default is 40; blank falls back to it, an explicit 0 is kept.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_gex_overlay_default
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from apps.web.modules.Ironbeam.indicators import gex_overlay as go


class GexOverlayDefaultTest(unittest.TestCase):
    def _threshold(self, cfg):
        seen = {}

        def fake_traces(**kwargs):
            seen["threshold_abs"] = kwargs["threshold_abs"]
            return []

        ctx = {"session_date": "2026-10-26", "band_min": 6000.0, "band_max": 7000.0}
        with patch.object(go, "build_gex_overlay_traces", side_effect=fake_traces):
            go.GexOverlay().build(ctx, cfg)
        return seen["threshold_abs"]

    def test_default_config_is_40(self):
        self.assertEqual(go.GexOverlay().default_config(), {"min_abs_b": 40.0})

    def test_default_ignores_compute_env(self):
        with patch.dict("os.environ", {"GEX_ABS_THRESHOLD": "1e10"}):
            self.assertEqual(go.GexOverlay().default_config()["min_abs_b"], 40.0)

    def test_blank_or_invalid_falls_back_to_40(self):
        for cfg in ({}, None, {"min_abs_b": None}, {"min_abs_b": ""}, {"min_abs_b": "  "}, {"min_abs_b": "abc"}):
            with self.subTest(cfg=cfg):
                self.assertEqual(self._threshold(cfg), 40e9)

    def test_explicit_values_kept(self):
        self.assertEqual(self._threshold({"min_abs_b": 0}), 0.0)
        self.assertEqual(self._threshold({"min_abs_b": "0"}), 0.0)
        self.assertEqual(self._threshold({"min_abs_b": 60}), 60e9)
        self.assertEqual(self._threshold({"min_abs_b": -5}), 0.0)


if __name__ == "__main__":
    unittest.main()
