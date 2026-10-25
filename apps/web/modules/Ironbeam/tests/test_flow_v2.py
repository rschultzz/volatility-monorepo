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
                         "&m_swing=4&m_gap=15")


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

    def test_bad_requests(self):
        for qs, code in (("mode=v2", 400), ("mode=v2&trade_date=2026-99-99", 400)):
            r = self.client.get(FLOW + qs)
            self.assertEqual(r.status_code, code)
            self.assertEqual(r.get_json()["mode"], "v2")


if __name__ == "__main__":
    unittest.main()
