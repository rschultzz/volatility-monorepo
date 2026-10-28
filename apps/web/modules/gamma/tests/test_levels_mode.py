"""CR-DH — Dashboard GEX chart Levels switch (spec items 1–5).

DB reads are replayed from fixtures/cr_dh_gex_levels.pkl.gz (record_levels_fixture.py); the
goldens are origin/main's figure on the same data. No DB needed.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.gamma.tests.test_levels_mode
"""
from __future__ import annotations

import datetime as dt
import json
import unittest

from apps.web.modules.gamma.tests import levels_fixture as lf


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store = lf.load()
        cls.mods = lf.modules()

    def render(self, date_iso, mode):
        with lf.replay(self.store):
            return lf.run_render(self.mods["gamma"], date_iso, mode)

    def golden(self, date_iso):
        return (lf.GOLDEN_DIR / f"cr_dh_origin_main_{date_iso}.json").read_text()


class TestSwitch(unittest.TestCase):

    def test_defaults_and_persistence(self):
        from modules.gamma.components import gex_block, GEX_LEVELS_MODE_ID
        radio = gex_block().children[0].children[1]
        self.assertEqual(radio.id, GEX_LEVELS_MODE_ID)
        self.assertEqual(radio.value, "corrected")
        self.assertEqual([o["label"] for o in radio.options], ["Current", "Corrected"])
        self.assertTrue(radio.persistence)
        self.assertEqual(radio.persistence_type, "local")


class TestCurrentByteIdentical(_Base):

    def test_current_matches_origin_main(self):
        for d in lf.DATES:
            with self.subTest(date=d):
                self.assertEqual(self.render(d, "current").to_json(), self.golden(d))


class TestCorrected(_Base):

    def test_levels_equal_price_chart_table(self):
        d = "2026-10-27"
        fig = self.render(d, "corrected")
        with lf.replay(self.store):
            table, _, reason = self.mods["ib"]._corrected_gex_tables(dt.date.fromisoformat(d))
        self.assertIsNone(reason)
        puts, calls = fig.data
        self.assertEqual(list(puts.y), list(table["level"]))
        self.assertEqual(list(puts.x), list(table["put_gamma"]))
        self.assertEqual(list(calls.x), list(table["call_gamma"]))
        self.assertTrue((table["put_gamma"] <= 0).all())  # puts left (calls are signed sums, as in Current)
        self.assertNotEqual(fig.to_json(), self.golden(d))

    def test_labels(self):
        layout = json.loads(self.render("2026-10-27", "corrected").to_json())["layout"]
        self.assertTrue(layout["title"]["text"].startswith("GEX by Corrected Level — 2026-10-27"))
        self.assertIn("Corrected Level", layout["yaxis"]["title"]["text"])
        self.assertIn("corrected level", layout["xaxis"]["title"]["text"])

    def test_none_mode_is_corrected(self):
        self.assertEqual(self.render("2026-10-27", None).to_json(), self.render("2026-10-27", "corrected").to_json())

    def test_fallback_is_current_with_reason_in_title(self):
        for d, label in (("2024-03-15", "pre-forward era"), ("2025-12-01", "no ES snapshot")):
            with self.subTest(date=d):
                fig = json.loads(self.render(d, "corrected").to_json())
                cur = json.loads(self.golden(d))
                self.assertEqual(
                    fig["layout"]["title"]["text"],
                    f"GEX by Discounted Level (corrected unavailable: {label}) — {d} (SPX)",
                )
                # Same chart as Current apart from the title: never both placements at once.
                self.assertEqual(fig["data"], cur["data"])
                fig["layout"]["title"] = cur["layout"]["title"]
                self.assertEqual(fig["layout"], cur["layout"])


if __name__ == "__main__":
    unittest.main()
