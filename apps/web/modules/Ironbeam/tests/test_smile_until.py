"""CR-CC A2 — `until` cap on /api/ironbeam/smile-data and /skew-data.

Byte identity: with `until` absent, empty or unparseable, the responses equal the goldens
recorded from the pre-A2 callbacks.py (f11b1de). ORATS reads are replayed from
fixtures/cr_cc_smile_2026-10-23.pkl.gz (no DB needed).

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_smile_until
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from apps.web.modules.Ironbeam.tests import smile_fixture as sf
from apps.web.modules.Ironbeam.tests.record_smile_fixture import UNTIL, UNTIL_QUERIES

PT = ZoneInfo("America/Los_Angeles")
NO_CUTOFF_SUFFIXES = ("", "&until=", "&until=garbage")


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = sf.make_client()
        cls.cb = sf.callbacks_module()
        cls.store = sf.load(sf.FIXTURE_DIR / "cr_cc_smile_2026-10-23.pkl.gz")

    def get(self, url):
        with sf.replay(self.store):
            r = self.client.get(url)
        self.assertEqual(r.status_code, 200)
        return r

    def json(self, url):
        return json.loads(self.get(url).data)


class TestNoCutoffByteIdentical(_Base):
    def test_match_golden(self):
        for name, url in sf.QUERIES.items():
            golden = (sf.FIXTURE_DIR / f"cr_cc_smile_golden_{name}.json").read_bytes()
            for suffix in NO_CUTOFF_SUFFIXES:
                with self.subTest(query=name, suffix=suffix):
                    self.assertEqual(self.get(url + suffix).data, golden)


class TestCap(_Base):
    def test_times_after_until_dropped(self):
        smile = self.json(UNTIL_QUERIES["smile_until"])
        self.assertEqual(smile["until"], UNTIL)
        self.assertEqual(smile["times"], ["07:00", "07:30"])
        self.assertEqual([s["time"] for s in smile["snapshots"]], ["07:00", "07:30"])
        self.assertTrue(all(not t["name"].startswith(("08:15", "10:00")) and "08:15" not in t["name"]
                            and "10:00" not in t["name"] for t in smile["traces"]))

    def test_expected_mode_on_capped_snapshots(self):
        # The expected curve / skew deltas at 07:30 depend only on 07:00, so the capped rows
        # equal the uncapped rows for the same snapshots, and nothing later is computed.
        golden = json.loads((sf.FIXTURE_DIR / "cr_cc_smile_golden_skew.json").read_bytes())["skew_data"]
        capped = self.json(UNTIL_QUERIES["skew_until"])["skew_data"]
        self.assertEqual(capped, golden[:2])
        self.assertIsNotNone(capped[1]["d_call"])
        smile_golden = json.loads((sf.FIXTURE_DIR / "cr_cc_smile_golden_smile.json").read_bytes())
        smile = self.json(UNTIL_QUERIES["smile_until"])
        # traces come per snapshot in time order (curve, expected curve, ATM marker)
        first_late = next(i for i, t in enumerate(smile_golden["traces"]) if "08:15" in t["name"])
        keep = smile_golden["traces"][:first_late]
        self.assertEqual(smile["traces"], keep)
        self.assertTrue(any(t["name"] == "Exp (SS) - 07:30" for t in smile["traces"]))

    def test_default_is_latest_snapshot_at_or_before_until(self):
        for key in ("smile_until_default", "skew_until_late"):
            with self.subTest(key=key):
                body = self.json(UNTIL_QUERIES[key])
                self.assertEqual(body["times"], ["08:00"])
        self.assertEqual([s["time"] for s in self.json(UNTIL_QUERIES["smile_until_default"])["snapshots"]], ["08:00"])

    def test_no_snapshot_before_first_of_day(self):
        body = self.json(UNTIL_QUERIES["smile_until_eth"])
        self.assertEqual(body["times"], [])
        self.assertEqual(body["snapshots"], [])

    def test_live_ignored_under_until(self):
        # Even "today" in market hours, the replay never gets a live row.
        def boom(*a, **k):
            raise AssertionError("live data must not be read under until")

        with patch.object(self.cb, "is_market_hours", boom):
            body = self.json(UNTIL_QUERIES["smile_until"] + "&live_data=%7B%7D")
        self.assertFalse(any(s["is_live"] for s in body["snapshots"]))

    def test_cap_helper(self):
        until = dt.datetime(2026, 10, 23, 8, 0, tzinfo=PT)
        with patch.object(self.cb, "_fetch_latest_snapshot_hhmm", return_value="07:59") as latest:
            self.assertEqual(self.cb._cap_times_until("2026-10-23", "2026-10-23", ["08:00", "07:59", "08:01", "x"], until),
                             ["07:59", "08:00"])  # snapshot time <= until is kept
            latest.assert_not_called()
            self.assertEqual(self.cb._cap_times_until("2026-10-23", "2026-10-23", ["08:01"], until), ["07:59"])
            self.assertEqual(self.cb._cap_times_until("bad", "2026-10-23", ["07:00"], until), [])


if __name__ == "__main__":
    unittest.main()
