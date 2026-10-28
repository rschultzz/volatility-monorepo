"""CR-DC step 2 — `/api/ironbeam/flow/cells` (spec DC12–DC14, A1.2, A1.3).

DB reads are replayed from fixtures/cr_dc_flow_cells.pkl.gz (record_flow_cells_fixture.py); no DB needed.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_flow_cells
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

from apps.web.modules.Ironbeam.tests import flow_cells_fixture as cf

ROUTE = "/api/ironbeam/flow/cells?"
UTC = dt.timezone.utc


def pt_epoch(s: str) -> int:
    from zoneinfo import ZoneInfo
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZoneInfo("America/Los_Angeles")).timestamp())


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = cf.make_client()
        cls.store = cf.load(cf.FIXTURE)
        cls.v3 = cf.v3_module()

    def get(self, qs, overrides=None):
        with cf.replay(self.store, overrides) as calls:
            r = self.client.get(ROUTE + qs)
        self.assertEqual(r.status_code, 200, r.data[:300])
        return r.get_json(), calls, r

    @staticmethod
    def seg(p, name):
        return next(s for s in p["segments"] if s["name"] == name)


class TestSources(_Base):
    def test_golden_2026_05_01(self):
        golden = (cf.FIXTURE_DIR / "cr_dc_flow_cells_golden_0501.json").read_bytes()
        self.assertEqual(self.get(cf.QS_0501)[2].data, golden)

    def test_past_train_date_is_approximate_and_has_no_marks(self):
        p, _, _ = self.get(cf.QS_0501)
        self.assertEqual([(s["name"], s["source"], s["tier"], s["exact"]) for s in p["segments"]],
                         [("eth", "ironbeam", "seconds", False), ("rth", "databento", "seconds", False)])
        rth = self.seg(p, "rth")
        self.assertEqual(rth["badge"], "approximate: 1-s data")
        self.assertIsNone(rth["marks"])                       # A1.2: no marks on approximate data
        self.assertEqual(rth["marks_note"], "Absorbed: exact data only")
        self.assertGreater(len(rth["cells"]["t"]), 1000)

    def test_retained_date_is_exact_with_marks(self):
        p, _, _ = self.get(cf.QS_1027)
        self.assertEqual([(s["name"], s["tier"], s["badge"]) for s in p["segments"]],
                         [("eth", "trades", "exact: trades"), ("rth", "trades", "exact: trades")])
        rth = self.seg(p, "rth")
        self.assertIsInstance(rth["marks"], list)
        self.assertGreater(len(rth["marks"]), 0)
        self.assertIsNone(self.seg(p, "eth")["marks"])        # marks are RTH only (DC9)
        for mk in rth["marks"]:
            self.assertIn(mk["kind"], ("sellers_absorbed", "buyers_absorbed"))
            self.assertGreaterEqual(mk["aggressor"], 3 * mk["median"])
            self.assertEqual(mk["confirm_time"], mk["time"] + 4 * 60)

    def test_holdout_dates_read_no_databento(self):
        p, calls, _ = self.get(cf.QS_1023)
        self.assertEqual([(s["source"], s["tier"]) for s in p["segments"]], [("ironbeam", "seconds")])
        self.assertTrue(any("holdout protection" in n for n in p["notes"]))
        flat = repr(calls)
        self.assertNotIn("es_flow_1s_databento", flat)
        self.assertNotIn("'databento'", flat)

    def test_cell_volume_equals_one_second_flow(self):
        """1-s tier: Σ cells = Σ 1-s buy / sell of the chosen contract (the low/high rule moves volume, never drops it)."""
        p, _, _ = self.get(cf.QS_0501_RTH)
        rth = self.seg(p, "rth")
        rows = next(v for k, v in self.store["calls"].items() if k[0] == "fetch_seconds"
                    and "es_flow_1s_databento" in k[1] and "2026, 5, 1, 20, 0" in k[1])
        self.assertAlmostEqual(sum(rth["cells"]["sell"]), float(rows["sell"].sum()))
        self.assertAlmostEqual(sum(rth["cells"]["buy"]), float(rows["buy"].sum()))


class TestUntil(_Base):
    def test_nothing_after_until(self):
        for qs, until in ((cf.QS_0501_UNTIL, "2026-05-01T07:39:00"), (cf.QS_1027_UNTIL, "2026-10-27T08:15:00")):
            p, calls, _ = self.get(qs)
            u = pt_epoch(until)
            for s in p["segments"]:
                self.assertLessEqual(s["end"], u)
                self.assertTrue(all(t < u for t in s["cells"]["t"]))
                for mk in s["marks"] or []:
                    self.assertLessEqual(mk["confirm_time"], u)
                    self.assertTrue(mk["break_time"] is None or mk["break_time"] < u)
            for name, args, kwargs in calls:   # no read reaches past `until`
                ends = [x for x in list(args) + list(kwargs.values()) if isinstance(x, dt.datetime)]
                if ends:
                    self.assertLessEqual(max(ends).timestamp(), u, name)

    def test_until_cells_are_the_full_session_cut(self):
        full = self.seg(self.get(cf.QS_0501_RTH)[0], "rth")["cells"]
        cut = self.seg(self.get(cf.QS_0501_UNTIL)[0], "rth")["cells"]
        u = pt_epoch("2026-05-01T07:39:00")
        keep = [i for i, t in enumerate(full["t"]) if t < u]
        self.assertEqual(cut, {k: [full[k][i] for i in keep] for k in full})

    def test_until_marks_are_the_full_session_marks_confirmed_by_then(self):
        full = self.seg(self.get(cf.QS_1027)[0], "rth")["marks"]
        cut = self.seg(self.get(cf.QS_1027_UNTIL)[0], "rth")["marks"]
        u = pt_epoch("2026-10-27T08:15:00")
        view = lambda m: (m["kind"], m["time"], m["price"], m["aggressor"], m["opposite"])  # noqa: E731
        self.assertEqual([view(m) for m in cut], [view(m) for m in full if m["confirm_time"] <= u])


class TestPrice1m(_Base):
    """DC17: es_flow_price_1m first when it exists and covers the segment (simulated from the trades)."""

    def _price_rows(self, drop_every=None):
        trades = {k: v for k, v in self.store["calls"].items() if k[0] == "fetch_trade_cells"}

        def fake(symbol, source, a, b):
            frames = [v for k, v in trades.items() if symbol in repr(k)]
            r = pd.concat(frames, ignore_index=True).drop_duplicates(["t", "price", "side"])   # overlapping windows
            r = r[(r["t"] >= a.timestamp()) & (r["t"] < b.timestamp())]
            g = r.assign(buy=r["size"].where(r["side"] == "BUY", 0.0), sell=r["size"].where(r["side"] == "SELL", 0.0))
            g = g.groupby(["t", "price"], as_index=False)[["buy", "sell"]].sum()
            if drop_every:
                keep = sorted(g["t"].unique())[::drop_every]
                g = g[g["t"].isin(keep)]
            return g
        return fake

    def test_price_1m_tier_used_and_equals_trades(self):
        ref = self.seg(self.get(cf.QS_1027)[0], "rth")
        p, calls, _ = self.get(cf.QS_1027, {"_exists_price_1m": lambda: True, "fetch_price_1m": self._price_rows()})
        rth = self.seg(p, "rth")
        self.assertEqual((rth["tier"], rth["badge"], rth["exact"]), ("price_1m", "exact: es_flow_price_1m", True))
        self.assertEqual(rth["cells"], ref["cells"])
        self.assertEqual(rth["marks"], ref["marks"])
        self.assertFalse(any(n == "fetch_trade_cells" for n, _, _ in calls))
        self.assertTrue(all(kw.get("source", a[1] if len(a) > 1 else None) == "ironbeam"
                            for n, a, kw in calls if n == "fetch_price_1m"))     # B4: never Databento after 06-05

    def test_partial_price_1m_falls_through_to_trades(self):
        p, _, _ = self.get(cf.QS_1027, {"_exists_price_1m": lambda: True, "fetch_price_1m": self._price_rows(2)})
        self.assertEqual(self.seg(p, "rth")["tier"], "trades")

    def test_absent_table_is_skipped(self):
        _, calls, _ = self.get(cf.QS_1027)
        self.assertFalse(any(n == "fetch_price_1m" for n, _, _ in calls))


class TestParamsAndCache(_Base):
    def test_absorb_params(self):
        f = self.v3.parse_absorb_params
        self.assertEqual(f({}), self.v3.fh.AbsorbParams())
        self.assertEqual((f({"a_k": "99", "a_m": "0"}).k, f({"a_k": "99", "a_m": "0"}).m), (50.0, 1))
        self.assertEqual((f({"a_k": "x", "a_m": "nan"}).k, f({"a_k": "x"}).m), (3.0, 3))
        p, _, _ = self.get(cf.QS_1027 + "&a_k=6")
        q, _, _ = self.get(cf.QS_1027)
        self.assertLess(len(self.seg(p, "rth")["marks"]), len(self.seg(q, "rth")["marks"]))
        self.assertEqual(p["absorb_params"], {"k": 6.0, "m": 3})

    def test_live_cells_are_read_incrementally(self):
        """DC14: a second poll reads only from the cached cut (now - 2 min), and equals a full read."""
        v3 = self.v3
        t0 = dt.datetime(2026, 10, 27, 13, 30, tzinfo=UTC)
        full = pd.DataFrame({"t": [int(t0.timestamp()) + 60 * i for i in range(60)], "p": 20000, "buy": 1.0, "sell": 2.0})
        reads = []

        def fake(engine, tier, table, symbol, source, a, b):
            reads.append((a, b))
            return full[(full["t"] >= a.timestamp()) & (full["t"] < b.timestamp())].reset_index(drop=True)

        v3._CACHE.clear()
        b = t0 + dt.timedelta(hours=1)
        with patch.object(v3, "_cells_for", fake):
            c1 = v3._segment_cells(None, "trades", "t", "S", "ironbeam", t0, b, False, t0 + dt.timedelta(minutes=20))
            c2 = v3._segment_cells(None, "trades", "t", "S", "ironbeam", t0, b, False, t0 + dt.timedelta(minutes=40))
        v3._CACHE.clear()
        self.assertEqual(reads[1][0], t0 + dt.timedelta(minutes=18))
        self.assertTrue(c1.equals(full[full["t"] < b.timestamp()].reset_index(drop=True)))
        self.assertTrue(c2.equals(full.reset_index(drop=True)))


if __name__ == "__main__":
    unittest.main()
