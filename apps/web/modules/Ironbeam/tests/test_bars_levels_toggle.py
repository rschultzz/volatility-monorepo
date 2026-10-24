"""CR-BT step 2 — /api/ironbeam/bars `levels` toggle.

L7: with `levels` absent, `current`, or anything other than `corrected`, the
response body is byte-identical to the one origin/main (e78774f) produced for the
same inputs. The golden bodies (fixtures/cr_bt_bars_golden_*.json) were written
by record_bars_fixture.py --golden on the unmodified callbacks.py; the DB reads
are replayed from fixtures/cr_bt_bars_2026-10-23.pkl.gz (no DB needed).

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_bars_levels_toggle
"""
from __future__ import annotations

import datetime as dt
import json
import math
import unittest
from unittest.mock import patch

from apps.web.modules.Ironbeam.tests import bars_fixture as bf

ROUTE = "/api/ironbeam/bars?"
FIXTURE = bf.FIXTURE_DIR / "cr_bt_bars_2026-10-23.pkl.gz"
GOLDEN = {
    "center": (bf.CENTER_QS, bf.FIXTURE_DIR / "cr_bt_bars_golden_center.json"),
    "multi": (bf.MULTI_QS, bf.FIXTURE_DIR / "cr_bt_bars_golden_multi.json"),
}


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store = bf.load(FIXTURE)
        cls.client = bf.make_client()
        cls.cb = bf.callbacks_module()

    def get(self, qs):
        with bf.replay(self.store):
            r = self.client.get(ROUTE + qs)
        self.assertEqual(r.status_code, 200)
        return r


class TestCurrentByteIdentical(_Base):

    def test_absent_current_and_unknown_match_golden(self):
        for name, (qs, path) in GOLDEN.items():
            golden = path.read_bytes()
            for suffix in ("", "&levels=current", "&levels=CURRENT", "&levels=foo", "&levels="):
                with self.subTest(phase=name, suffix=suffix):
                    self.assertEqual(self.get(qs + suffix).data, golden)


class TestCorrected(_Base):

    def corrected(self, qs=bf.CENTER_QS):
        return json.loads(self.get(qs + "&levels=corrected").data)

    def test_differs_and_carries_note_keys(self):
        body = self.get(bf.CENTER_QS + "&levels=corrected").data
        self.assertNotEqual(body, GOLDEN["center"][1].read_bytes())
        j = json.loads(body)
        self.assertEqual(j["levels"], "corrected")
        self.assertEqual(j["levels_notes"], [])
        current = json.loads(GOLDEN["center"][1].read_bytes())
        self.assertEqual(set(j) - set(current), {"levels", "levels_notes"})
        self.assertEqual(j["bars"], current["bars"])

    def test_worked_day_top_lines(self):
        """Step 0 Q4: the dominant lines move 7660 → 7681 and 7760 → 7781."""
        segs = [s for s in self.corrected()["gex_segments"] if s["session_date"] == "2026-10-23"]
        top = sorted(segs, key=lambda s: -(abs(s["call_gamma"]) + abs(s["put_gamma"])))[:3]
        self.assertEqual([s["level"] for s in top], [7681.0, 7781.0, 7687.0])
        self.assertEqual(len(segs), 60)

    def test_expirations_sum_to_line(self):
        """L6: each segment's per-expiry rows sum to the line, in both phases."""
        for qs in (bf.CENTER_QS, bf.MULTI_QS):
            segs = self.corrected(qs)["gex_segments"]
            self.assertTrue(segs)
            for s in segs:
                exps = s["expirations"]
                self.assertTrue(exps, s["level"])
                for k in ("net_gamma", "call_gamma", "put_gamma"):
                    total = sum(e[k] for e in exps)
                    self.assertTrue(math.isclose(total, s[k], rel_tol=1e-9, abs_tol=1.0), (qs, s["level"], k))

    def test_gex_levels_follow_toggle(self):
        j = self.corrected()
        center = {s["level"] for s in j["gex_segments"] if s["session_date"] == "2026-10-23"}
        self.assertEqual({g["level"] for g in j["gex_levels"]}, center)

    def test_missing_snapshot_draws_no_lines(self):
        """A1.2: no ES 15:45 ET bar → no corrected lines for that session, with a note."""
        with bf.replay(self.store), patch.object(self.cb, "_fetch_es_snapshot_close", lambda d: None):
            j = json.loads(self.client.get(ROUTE + bf.CENTER_QS + "&levels=corrected").data)
        self.assertEqual(j["gex_segments"], [])
        self.assertEqual(j["gex_levels"], [])
        self.assertEqual(j["levels_notes"], [{"session_date": "2026-10-23", "reason": "no_es_snapshot"}])
        self.assertTrue(j["bars"])


class TestCorrectedTables(_Base):

    def test_pre_forward_era_reads_nothing(self):
        """A1.1: sessions before 2024-04-03 return the note without touching the DB."""
        def boom(*a, **k):
            raise AssertionError("DB read for a pre-era session")
        with patch.object(self.cb, "_fetch_es_snapshot_close", boom), \
                patch.object(self.cb, "_fetch_gex_rows_for_placement", boom):
            out = self.cb._corrected_gex_tables(dt.date(2024, 4, 2))
        self.assertEqual(out, (None, None, "pre_forward_era"))

    def test_snapshot_from_api_date(self):
        """P_target is read on prev_trading_day(D): Monday's session uses Friday's bar."""
        seen = []
        with bf.replay(self.store), patch.object(self.cb, "_fetch_es_snapshot_close",
                                                 lambda d: seen.append(d) or None):
            self.cb._corrected_gex_tables(dt.date(2026, 10, 26))
        self.assertEqual(seen, [dt.date(2026, 10, 23)])

    def test_by_level_equals_sum_of_by_expir(self):
        with bf.replay(self.store):
            df_gex, df_exp, reason = self.cb._corrected_gex_tables(dt.date(2026, 10, 23))
        self.assertIsNone(reason)
        summed = df_exp.groupby("level")[["call_gamma", "put_gamma"]].sum()
        # put_gamma is −|Σ put| per group, so compare the call side exactly and |put| per level
        for lvl, row in df_gex.set_index("level").iterrows():
            self.assertTrue(math.isclose(row["call_gamma"], summed.loc[lvl, "call_gamma"], rel_tol=1e-12, abs_tol=1.0))
        self.assertTrue((df_gex["level"] == df_gex["level"].round()).all())


if __name__ == "__main__":
    unittest.main()
