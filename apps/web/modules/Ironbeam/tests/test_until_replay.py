"""CR-CC step 3 — `until` replay on /api/ironbeam/bars, /live-trades-overlay and /flow (L5, A1.2, A1.3).

Byte identity: with `until` absent, empty or unparseable, the bars responses equal the
CR-BT goldens (fixtures/cr_bt_bars_golden_*.json) and the flow responses equal the
CR-CC flow goldens recorded from the pre-CR-CC callbacks.py (59e2fa8). DB reads are
replayed from fixtures (no DB needed).

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_until_replay
"""
from __future__ import annotations

import datetime as dt
import json
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from apps.web.modules.Ironbeam.tests import bars_fixture as bf
from apps.web.modules.Ironbeam.tests import flow_fixture as ff
from apps.web.modules.Ironbeam.tests.record_until_fixture import UNTIL_ETH, UNTIL_RTH

PT = ZoneInfo("America/Los_Angeles")
BARS = "/api/ironbeam/bars?"
FLOW = "/api/ironbeam/flow?"
NO_CUTOFF_SUFFIXES = ("", "&until=", "&until=garbage", "&until=2026-13-45T99:00")


def epoch(pt_iso: str) -> int:
    return int(dt.datetime.fromisoformat(pt_iso).replace(tzinfo=PT).timestamp())


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = bf.make_client()
        cls.cb = bf.callbacks_module()
        cls.bt_store = bf.load(bf.FIXTURE_DIR / "cr_bt_bars_2026-10-23.pkl.gz")
        cls.until_store = bf.load(bf.FIXTURE_DIR / "cr_cc_bars_until_2026-10-23.pkl.gz")
        cls.flow_store = ff.load(ff.FIXTURE_DIR / "cr_cc_flow_2026-10-23.pkl.gz")

    def bars(self, qs, store=None):
        with bf.replay(store or self.until_store):
            r = self.client.get(BARS + qs)
        self.assertEqual(r.status_code, 200)
        return r

    def flow(self, qs):
        with ff.replay(self.flow_store):
            r = self.client.get(FLOW + qs)
        self.assertEqual(r.status_code, 200)
        return r


class TestNoCutoffByteIdentical(_Base):
    def test_bars_match_cr_bt_golden(self):
        for name, qs in (("center", bf.CENTER_QS), ("multi", bf.MULTI_QS)):
            golden = (bf.FIXTURE_DIR / f"cr_bt_bars_golden_{name}.json").read_bytes()
            for suffix in NO_CUTOFF_SUFFIXES:
                with self.subTest(phase=name, suffix=suffix):
                    self.assertEqual(self.bars(qs + suffix, self.bt_store).data, golden)

    def test_flow_matches_cr_cc_golden(self):
        for name, qs in ff.QUERIES.items():
            golden = (ff.FIXTURE_DIR / f"cr_cc_flow_golden_{name}.json").read_bytes()
            for suffix in NO_CUTOFF_SUFFIXES:
                with self.subTest(view=name, suffix=suffix):
                    self.assertEqual(self.flow(qs + suffix).data, golden)

    def test_parse_until(self):
        p = self.cb._parse_until_pt
        self.assertIsNone(p(None, PT))
        self.assertIsNone(p("  ", PT))
        self.assertIsNone(p("07:37", PT))
        self.assertEqual(p("2026-10-23T07:37:42", PT), dt.datetime(2026, 10, 23, 7, 37, tzinfo=PT))
        self.assertEqual(p("2026-10-23 07:37", PT), dt.datetime(2026, 10, 23, 7, 37, tzinfo=PT))


class TestBarsUntil(_Base):
    def payload(self, qs):
        return json.loads(self.bars(qs).data)

    def test_center_bars_end_before_cutoff(self):
        full = self.payload(bf.CENTER_QS)
        cut = self.payload(bf.CENTER_QS + "&until=" + UNTIL_RTH)
        u = epoch(UNTIL_RTH)
        self.assertEqual(cut["until"], UNTIL_RTH)
        self.assertNotIn("until", full)
        self.assertTrue(cut["bars"])
        self.assertTrue(all(b["time"] < u for b in cut["bars"]))
        self.assertEqual(cut["bars"][-1]["time"], u - 60)  # the signal bar 07:36 is the last shown
        # identical bars up to the cutoff
        self.assertEqual(cut["bars"], [b for b in full["bars"] if b["time"] < u])

    def test_gex_band_from_bars_up_to_cutoff(self):
        pad = self.until_store["constants"]["GEX_LEVEL_PADDING"]

        def band(p):
            return (min(b["low"] for b in p["bars"]) - pad, max(b["high"] for b in p["bars"]) + pad)

        for until in (UNTIL_RTH, UNTIL_ETH):
            with self.subTest(until=until):
                cut = self.payload(bf.CENTER_QS + "&until=" + until)
                lo, hi = band(cut)
                levels = [s["level"] for s in cut["gex_segments"]] + [g["level"] for g in cut["gex_levels"]]
                self.assertTrue(levels)
                self.assertTrue(all(lo <= x <= hi for x in levels), (lo, hi, levels))
        # On 10-23 the session high/low is already set by 07:37 PT, so only the ETH cutoff
        # narrows the band; there the line set differs from the full-session one.
        full = self.payload(bf.CENTER_QS)
        eth = self.payload(bf.CENTER_QS + "&until=" + UNTIL_ETH)
        self.assertNotEqual(band(eth), band(full))
        self.assertNotEqual({s["level"] for s in eth["gex_segments"]}, {s["level"] for s in full["gex_segments"]})

    def test_multi_loads_nothing_after_cutoff(self):
        cut = self.payload(bf.MULTI_QS + "&until=" + UNTIL_RTH)
        u = epoch(UNTIL_RTH)
        self.assertTrue(all(b["time"] < u for b in cut["bars"]))
        self.assertTrue(all(d < "2026-10-23" for d in cut["loaded_dates"]))

    def test_expected_move_suppressed_before_0633(self):
        rth = self.payload(bf.CENTER_QS + "&until=" + UNTIL_RTH)
        eth = self.payload(bf.CENTER_QS + "&until=" + UNTIL_ETH)
        self.assertEqual([e["date"] for e in rth["expected_move_levels"]], ["2026-10-23"])
        self.assertEqual(eth["expected_move_levels"], [])
        self.assertTrue(all(b["time"] < epoch(UNTIL_ETH) for b in eth["bars"]))

    def test_5min_last_bucket_is_partial_not_lookahead(self):
        # _resample_ohlc labels 5-min buckets right/closed-right on bar starts: the bucket
        # labelled 07:40 holds the 1-min bars starting 07:36..07:40. Under a 07:37 cutoff it
        # holds only the 07:36 bar.
        one = self.payload(bf.CENTER_QS + "&until=" + UNTIL_RTH)
        five = self.payload(bf.CENTER_QS.replace("interval=1min", "interval=5min") + "&until=" + UNTIL_RTH)
        last = five["bars"][-1]
        self.assertEqual(last["time"], epoch("2026-10-23T07:40:00"))
        inside = [b for b in one["bars"] if epoch("2026-10-23T07:35:00") < b["time"] <= last["time"]]
        self.assertEqual([b["time"] for b in inside], [epoch("2026-10-23T07:36:00")])
        self.assertEqual((last["open"], last["high"], last["low"], last["close"]),
                         (inside[0]["open"], inside[0]["high"], inside[0]["low"], inside[0]["close"]))

    def test_corrected_with_cutoff(self):
        cut = self.payload(bf.CENTER_QS + "&levels=corrected&until=" + UNTIL_RTH)
        self.assertEqual(cut["levels"], "corrected")
        self.assertEqual(cut["until"], UNTIL_RTH)
        self.assertTrue(all(b["time"] < epoch(UNTIL_RTH) for b in cut["bars"]))


class TestOverlayAndFlowUntil(_Base):
    def test_overlay_empty_without_db(self):
        def boom(*a, **k):
            raise AssertionError("overlay must not read the DB under until")

        with patch.object(self.cb, "_fetch_bars_pt", boom), patch.object(self.cb, "_fetch_trades_utc", boom):
            r = self.client.get("/api/ironbeam/live-trades-overlay?trade_date=2026-10-23&interval=1min&until=" + UNTIL_RTH)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["bars"], [])
        self.assertFalse(body["is_live_day"])
        self.assertEqual(body["until"], UNTIL_RTH)

    def test_flow_ends_before_cutoff(self):
        u = epoch(UNTIL_RTH)
        for name, qs in ff.QUERIES.items():
            with self.subTest(view=name):
                full = json.loads(self.flow(qs).data)
                cut = json.loads(self.flow(qs + "&until=" + UNTIL_RTH).data)
                self.assertEqual(cut["until"], UNTIL_RTH)
                self.assertTrue(cut["flow_points"])
                self.assertTrue(all(p["time"] < u for p in cut["flow_points"]))
                self.assertLess(cut["point_count"], full["point_count"])
                # EMA is causal: points before the cutoff are unchanged
                self.assertEqual(cut["flow_points"], [p for p in full["flow_points"] if p["time"] < u])


if __name__ == "__main__":
    unittest.main()
