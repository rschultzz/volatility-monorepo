"""CR-CD step 2 — `/api/ironbeam/flow?mode=v2`, and Current mode byte-identical (L1, A1, A2).

DB reads are replayed from fixtures (no DB needed):
  - Current: the CR-CC flow fixture and goldens (recorded from the pre-CR-CC module).
  - v2: fixtures/cr_cd_flow_v2.pkl.gz (record_flow_v2_fixture.py), golden for 2026-05-01 FULL.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_flow_v2
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import inspect
import json
import unittest
from zoneinfo import ZoneInfo

from apps.web.modules.Ironbeam.tests import bars_fixture as bf
from apps.web.modules.Ironbeam.tests import flow_fixture as ff
from apps.web.modules.Ironbeam.tests import flow_v2_fixture as vf

PT = ZoneInfo("America/Los_Angeles")
FLOW = "/api/ironbeam/flow?"
# sha256 of `_build_react_flow_payload`'s source at origin/main 578a8d5 (CR-CD base).
CURRENT_BUILDER_SHA = "c564a3ed4da95edc4ae05384b46cfdec998285f0bb2355dff365e1dafdc8860b"
CURRENT_MODE_SUFFIXES = ("", "&mode=", "&mode=current", "&mode=CURRENT", "&mode=bogus", "&mode=v1",
                         "&m_swing=4&m_gap=15", "&effort=1", "&mode=current&effort=1")   # A5: effort is v2-only
V2_NO_EFFORT_SUFFIXES = ("", "&effort=", "&effort=0", "&effort=true", "&effort=on", "&effort=bogus")


def epoch(pt_iso: str) -> int:
    return int(dt.datetime.fromisoformat(pt_iso).replace(tzinfo=PT).timestamp())


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = bf.make_client()
        cls.cb = bf.callbacks_module()
        cls.flow_store = ff.load(ff.FIXTURE_DIR / "cr_cc_flow_2026-10-23.pkl.gz")
        cls.v2_store = vf.load(vf.FIXTURE)

    def v2(self, qs):
        with vf.replay(self.v2_store):
            r = self.client.get(FLOW + qs)
        self.assertEqual(r.status_code, 200, r.data[:300])
        return r


class TestCurrentModeUnchanged(_Base):
    def test_current_matches_golden_for_every_non_v2_mode(self):
        for name, qs in (("full", ff.FULL_QS), ("rth", ff.RTH_QS)):
            golden = (ff.FIXTURE_DIR / f"cr_cc_flow_golden_{name}.json").read_bytes()
            for suffix in CURRENT_MODE_SUFFIXES:
                with self.subTest(view=name, suffix=suffix), ff.replay(self.flow_store):
                    r = self.client.get(FLOW + qs + suffix)
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(r.data, golden)

    def test_current_builder_source_unchanged(self):
        src = inspect.getsource(self.cb)
        node = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_build_react_flow_payload")
        seg = ast.get_source_segment(src, node)
        self.assertEqual(hashlib.sha256(seg.encode()).hexdigest(), CURRENT_BUILDER_SHA)


class TestFlowV2(_Base):
    def test_golden_2026_05_01(self):
        golden = (vf.FIXTURE_DIR / "cr_cd_flow_v2_golden_0501.json").read_bytes()
        self.assertEqual(self.v2(vf.QS_0501).data, golden)

    def test_sources_and_segments_2026_05_01(self):
        p = self.v2(vf.QS_0501).get_json()
        self.assertEqual(p["mode"], "v2")
        self.assertEqual(p["source"], "databento")
        self.assertEqual([(s["name"], s["source"]) for s in p["segments"]], [("eth", "ironbeam"), ("rth", "databento")])
        self.assertEqual(p["symbol"], "XCME:ES.M26")
        rth = [b for b in p["bars"] if b["seg"] == "rth"]
        eth = [b for b in p["bars"] if b["seg"] == "eth"]
        self.assertEqual(len(rth), 390)
        self.assertEqual(rth[0]["time"], epoch("2026-05-01T06:30:00"))
        self.assertEqual(eth[0]["time"], epoch("2026-04-30T15:00:00"))
        self.assertEqual(rth[0]["cvd"], rth[0]["delta"])         # RTH CVD anchored at 06:30
        self.assertEqual(eth[0]["cvd"], eth[0]["delta"])         # ETH CVD anchored at 15:00
        self.assertFalse(any(b["absorb"] for b in eth))         # D10: absorption RTH-only
        self.assertTrue(all(m["time"] >= epoch("2026-05-01T06:30:00") for m in p["markers"]))
        b = p["baseline"]
        self.assertEqual((b["source"], b["n_sessions"], b["sessions"]), ("databento", 20, ["2026-04-02", "2026-04-30"]))
        self.assertAlmostEqual(b["abs_median"], 4.5455, places=3)   # Step 0 Q4: 4.55 / 1.94
        self.assertAlmostEqual(b["abs_p20"], 1.9387, places=3)
        self.assertEqual(len(p["markers"]), 7)                       # A2 smoke figure (defaults)

    def test_after_holdout_cut_serves_ironbeam_and_reads_no_databento(self):
        p = self.v2(vf.QS_1023).get_json()
        self.assertEqual(p["source"], "ironbeam")
        self.assertEqual({s["source"] for s in p["segments"]}, {"ironbeam"})
        self.assertIn("holdout", " ".join(p["notes"]))
        self.assertEqual(p["baseline"]["source"], "ironbeam")
        self.assertTrue(any(g["minutes"] > 100 for g in p["gaps"]))   # the 07:53-09:56 trades-ws hole
        tables = {k[1] for k in self.v2_store["calls"] if "2026, 10, 2" in k[1]}
        self.assertTrue(tables)
        self.assertFalse(any("es_flow_1s_databento" in k for k in tables))

    def test_two_contract_session_uses_the_bars_contract(self):
        p = self.v2(vf.QS_0317).get_json()
        rth = next(s for s in p["segments"] if s["name"] == "rth")
        self.assertEqual((rth["symbol"], rth["symbol_choice"]), ("XCME:ES.H26", "bars"))

    def test_until_truncates_and_markers_are_causal(self):
        full = self.v2(vf.QS_0501).get_json()
        cut = self.v2(vf.QS_0501_UNTIL).get_json()
        u = epoch("2026-05-01T07:39:00")
        self.assertEqual(cut["until"], "2026-05-01T07:39:00")
        self.assertTrue(cut["bars"])
        self.assertTrue(all(b["time"] < u for b in cut["bars"]))
        self.assertEqual(cut["bars"], [b for b in full["bars"] if b["time"] < u])
        self.assertEqual(cut["markers"], [m for m in full["markers"] if m["time"] < u])
        self.assertEqual(cut["baseline"], full["baseline"])

    def test_marker_params(self):
        dflt = self.v2(vf.QS_0501).get_json()
        a1 = self.v2(vf.QS_0501 + "&m_swing=4&m_push=0&m_div=0&m_fade=0&m_gap=15").get_json()
        junk = self.v2(vf.QS_0501 + "&m_swing=abc&m_push=-5&m_gap=nan").get_json()
        self.assertEqual(a1["marker_params"], {"swing": 4.0, "min_prev_push": 0.0, "min_div_cvd": 0.0,
                                               "fade_min_share": 0.0, "gap_min": 15})
        self.assertEqual(len(a1["markers"]), 42)                      # A2 table, setting A, 05-01
        self.assertEqual(junk["marker_params"]["swing"], 8.0)
        self.assertEqual(junk["marker_params"]["min_prev_push"], 0.0)  # clamped to the floor
        self.assertEqual(junk["marker_params"]["gap_min"], 30)
        self.assertEqual(dflt["bars"], a1["bars"])

    def test_without_effort_1_payload_is_the_v2_golden(self):
        golden = (vf.FIXTURE_DIR / "cr_cd_flow_v2_golden_0501.json").read_bytes()
        for suffix in V2_NO_EFFORT_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertEqual(self.v2(vf.QS_0501 + suffix).data, golden)

    def test_bad_requests(self):
        for qs, code in (("mode=v2", 400), ("mode=v2&trade_date=2026-99-99", 400)):
            r = self.client.get(FLOW + qs)
            self.assertEqual(r.status_code, code)
            self.assertEqual(r.get_json()["mode"], "v2")


class TestEffortLegs(_Base):
    """CR-CD A5: `effort=1` adds RTH swing legs (and, CR-DM, the leg in progress) and changes nothing else."""

    def test_effort_adds_only_legs(self):
        base = json.loads((vf.FIXTURE_DIR / "cr_cd_flow_v2_golden_0501.json").read_bytes())
        p = self.v2(vf.QS_0501 + "&effort=1").get_json()
        legs = p.pop("legs")
        live = p.pop("live_leg")
        self.assertEqual(p, base)
        self.assertTrue(legs)
        self.assertEqual(set(legs[0]), {"start_time", "end_time", "confirm_time", "dir", "start_px", "end_px",
                                        "pts", "vol", "delta", "cpp", "dpp", "ratio"})
        self.assertEqual(set(live), {"start_time", "end_time", "last_time", "dir", "start_px", "end_px", "pts", "vol",
                                     "delta", "cpp", "dpp", "ratio", "prev_cpp", "minutes"})
        # CR-DM: the leg in progress starts where the last confirmed leg ends, the other way
        self.assertEqual((live["start_time"], live["start_px"]), (legs[-1]["end_time"], legs[-1]["end_px"]))
        self.assertNotEqual(live["dir"], legs[-1]["dir"])

    def test_legs_2026_05_01(self):
        p = self.v2(vf.QS_0501 + "&effort=1").get_json()
        legs = p["legs"]
        rth0 = epoch("2026-05-01T06:30:00")
        rth = {b["time"]: b for b in p["bars"] if b["seg"] == "rth"}
        for a, b in zip(legs[:-1], legs[1:]):
            self.assertNotEqual(a["dir"], b["dir"])
            self.assertEqual((b["start_time"], b["start_px"]), (a["end_time"], a["end_px"]))
        prev = {}
        for lg in legs:
            self.assertGreaterEqual(lg["start_time"], rth0)
            self.assertLess(lg["end_time"], lg["confirm_time"])
            self.assertGreaterEqual(abs(lg["pts"]), 8.0)
            self.assertEqual(lg["pts"] > 0, lg["dir"] == "up")
            span = [t for t in rth if lg["start_time"] < t <= lg["end_time"]]
            self.assertAlmostEqual(lg["vol"], sum(rth[t]["vol"] for t in span))
            self.assertAlmostEqual(lg["delta"], sum(rth[t]["delta"] for t in span))
            self.assertAlmostEqual(lg["cpp"], lg["vol"] / abs(lg["pts"]))
            self.assertAlmostEqual(lg["dpp"], lg["delta"] / abs(lg["pts"]))
            if lg["dir"] in prev:
                self.assertAlmostEqual(lg["ratio"], lg["cpp"] / prev[lg["dir"]])
            else:
                self.assertIsNone(lg["ratio"])
            prev[lg["dir"]] = lg["cpp"]

    def test_swing_setting_drives_legs(self):
        d8 = self.v2(vf.QS_0501 + "&effort=1").get_json()["legs"]
        d4 = self.v2(vf.QS_0501 + "&effort=1&m_swing=4").get_json()["legs"]
        self.assertGreater(len(d4), len(d8))
        self.assertTrue(all(abs(lg["pts"]) >= 4.0 for lg in d4))

    def test_until_legs_are_causal(self):
        full = self.v2(vf.QS_0501 + "&effort=1").get_json()
        cut = self.v2(vf.QS_0501_UNTIL + "&effort=1").get_json()
        u = epoch("2026-05-01T07:39:00")
        self.assertTrue(cut["legs"])
        self.assertEqual(cut["legs"], [lg for lg in full["legs"] if lg["confirm_time"] < u])
        live = cut.pop("live_leg")   # CR-DM: built from bars before `until` only
        self.assertLess(live["last_time"], u)
        self.assertLessEqual(live["end_time"], live["last_time"])
        cut.pop("legs")
        self.assertEqual(cut, self.v2(vf.QS_0501_UNTIL).get_json())

    def test_live_leg_at_until(self):
        """CR-DM: at the replay time the live leg's numbers are sums of the bars before it, and the leg the
        full session later confirms from the same start has the same numbers if its extreme was already in."""
        full = self.v2(vf.QS_0501 + "&effort=1").get_json()
        cut = self.v2(vf.QS_0501_UNTIL + "&effort=1").get_json()
        u = epoch("2026-05-01T07:39:00")
        rth = {b["time"]: b for b in full["bars"] if b["seg"] == "rth" and b["time"] < u}
        live = cut["live_leg"]
        span = [t for t in rth if live["start_time"] < t <= live["end_time"]]
        self.assertEqual(live["minutes"], len(span))
        self.assertAlmostEqual(live["vol"], sum(rth[t]["vol"] for t in span))
        self.assertAlmostEqual(live["delta"], sum(rth[t]["delta"] for t in span))
        self.assertAlmostEqual(live["cpp"], live["vol"] / abs(live["pts"]))
        prev = next((lg for lg in reversed(cut["legs"]) if lg["dir"] == live["dir"]), None)
        self.assertEqual(live["prev_cpp"], None if prev is None else prev["cpp"])
        later = next(lg for lg in full["legs"] if lg["start_time"] == live["start_time"])
        self.assertGreaterEqual(later["confirm_time"], u)
        if later["end_time"] == live["end_time"]:
            keys = ("dir", "start_px", "end_px", "pts", "vol", "delta", "cpp", "dpp", "ratio")
            self.assertEqual({k: live[k] for k in keys}, {k: later[k] for k in keys})

    def test_ironbeam_session_and_rth_only_session(self):
        for qs in (vf.QS_1023, vf.QS_0317):
            with self.subTest(qs=qs):
                p = self.v2(qs + "&effort=1").get_json()
                self.assertTrue(p["legs"])
                rth = next(s for s in p["segments"] if s["name"] == "rth")
                self.assertTrue(all(lg["start_time"] >= rth["start"] for lg in p["legs"]))


if __name__ == "__main__":
    unittest.main()
