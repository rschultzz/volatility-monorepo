"""CR-DC step 2 — `/api/ironbeam/flow/cells` (spec DC12–DC14, A1.2, A1.3).

DB reads are replayed from fixtures/cr_dc_flow_cells.pkl.gz (record_flow_cells_fixture.py); no DB needed.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_flow_cells
"""
from __future__ import annotations

import datetime as dt
import json
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
        # CR-DF adds `bars` to each segment; everything else is CR-DC's golden unchanged.
        golden = json.loads((cf.FIXTURE_DIR / "cr_dc_flow_cells_golden_0501.json").read_bytes())
        p = self.get(cf.QS_0501)[0]
        for s in p["segments"]:
            self.assertIn("bars", s)
            del s["bars"]
        self.assertEqual(p, golden)

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


class TestLiveSegment(_Base):
    """CR-DV: a live session (2026-10-27 FULL view replayed with now inside RTH): marks on closed minutes only, provisional
    candidates, the feed-stall flag (> 60 s without a trade) and the amendment-2 splice (stored es_flow_price_1m minutes up
    to its last written minute + trades for the closed minutes after it), with incremental reads. Reads on ranges the
    fixture never recorded are served by slicing its recorded whole-window results."""
    NOW = dt.datetime(2026, 10, 27, 17, 0, 20, tzinfo=UTC)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rec = cls.store["calls"]
        pick = lambda name: [v for k, v in rec.items() if k[0] == name]
        cls.trades = pd.concat(pick("fetch_trade_cells"), ignore_index=True).drop_duplicates()
        cls.flow_rows = {}
        for k, v in rec.items():
            if k[0] == "fetch_flow_minutes":
                cls.flow_rows.setdefault(k[1].split(",")[0], []).append(v)
        cls.flow_rows = {t: pd.concat(v, ignore_index=True).drop_duplicates(subset=["symbol", "t"]) for t, v in cls.flow_rows.items()}
        cls.bar_closes = {}
        for v in pick("fetch_bar_closes"):
            cls.bar_closes.update(v)

    def trade_cells(self, symbol, a, b):
        t = self.trades
        return t[(t["t"] >= a.timestamp()) & (t["t"] < b.timestamp())].reset_index(drop=True)

    def stored_rows(self, until):
        """es_flow_price_1m as the cron would have written it: every minute < `until` (epoch), from the trades."""
        def fetch(symbol, source, a, b):
            t = self.trade_cells(symbol, a, b)
            t = t[t["t"] < until]
            g = t.assign(price=(t["price"] * 4) // 1 / 4,
                         buy=t["size"].where(t["side"] == "BUY", 0.0), sell=t["size"].where(t["side"] == "SELL", 0.0))
            return g.groupby(["t", "price"], as_index=False)[["buy", "sell"]].sum()
        return fetch

    def overrides(self, last, stored_until, reads=None):
        def log(name, fn):
            def w(*a, **k):
                if reads is not None:
                    reads.append((name,) + a)
                return fn(*a, **k)
            return w

        def flow(table, ranges, now_utc=None):
            rows = next(v for k, v in self.flow_rows.items() if repr(table) in k)
            keep = pd.Series(False, index=rows.index)
            for a, b in ranges:
                keep |= (rows["t"] >= a.timestamp()) & (rows["t"] < b.timestamp())
            return rows[keep].reset_index(drop=True)

        return {"_trades_last": log("_trades_last", lambda s: last),
                "fetch_price_1m": log("fetch_price_1m", self.stored_rows(stored_until)),
                "fetch_trade_cells": log("fetch_trade_cells", self.trade_cells),
                "fetch_flow_minutes": log("fetch_flow_minutes", flow),
                "fetch_bar_closes": log("fetch_bar_closes", lambda a, b: {t: c for t, c in self.bar_closes.items()
                                                                         if a.timestamp() <= t < b.timestamp()})}

    def build(self, now, last, stored_until=None, reads=None, keep_cache=False):
        o = self.overrides(last, stored_until if stored_until is not None else int(now.timestamp()) // 60 * 60 - 300, reads)
        with cf.replay(self.store, o):
            if keep_cache:
                self.v3._CACHE.update(self._saved)
            p, status = self.v3._build(None, "2026-10-27", session="FULL", until_pt=None, params=self.v3.fh.AbsorbParams(),
                                       session_date=dt.date(2026, 10, 27), session_note=None, now_utc=now)
            self._saved = dict(self.v3._CACHE)
        self.assertEqual(status, 200)
        return p

    def test_live_marks_closed_minutes_only(self):
        rth = self.seg(self.build(self.NOW, self.NOW - dt.timedelta(seconds=1)), "rth")
        closed = int(self.NOW.timestamp()) // 60 * 60
        self.assertTrue(rth["marks"])
        for mk in rth["marks"]:
            self.assertIn(mk["provisional"], (True, False))
            self.assertLess(mk["time"] + 60, closed + 1)                       # its minute has closed
            if mk["provisional"]:
                self.assertGreater(mk["confirm_time"], closed)
            else:
                self.assertLessEqual(mk["confirm_time"], closed)

    def test_stall_flag_fires_after_60_s(self):
        last, reads = self.NOW - dt.timedelta(seconds=61), []
        rth = self.seg(self.build(self.NOW, last, reads=reads), "rth")
        self.assertEqual(rth["feed_stalled_since"], int(last.timestamp()))
        self.assertEqual([r[1] for r in reads if r[0] == "_trades_last"], [rth["symbol"]])

    def test_no_stall_flag_on_quiet_seconds(self):
        for gap in (1, 30, 60):
            rth = self.seg(self.build(self.NOW, self.NOW - dt.timedelta(seconds=gap)), "rth")
            self.assertNotIn("feed_stalled_since", rth, gap)

    def test_splice_equals_trades_for_closed_minutes_and_drops_the_forming_minute(self):
        p = self.build(self.NOW, self.NOW)
        closed = int(self.NOW.timestamp()) // 60 * 60
        for s in p["segments"]:
            self.assertEqual((s["tier"], s["badge"]), ("splice", "exact: es_flow_price_1m + trades"))
            t = self.trade_cells(s["symbol"], dt.datetime.fromtimestamp(s["start"], UTC),
                                 dt.datetime.fromtimestamp(min(s["end"], closed), UTC))
            want = self.v3.fh.cells_from_trades(t["t"], t["price"], t["size"], t["side"])
            got = pd.DataFrame(s["cells"]).rename(columns={}).sort_values(["t", "p"]).reset_index(drop=True)
            self.assertEqual(len(got), len(want), s["name"])
            self.assertTrue((got[["t", "p"]].to_numpy() == want[["t", "p"]].to_numpy()).all())
            self.assertTrue(((got[["buy", "sell"]] - want[["buy", "sell"]].to_numpy()).abs().to_numpy() < 1e-9).all())
            self.assertLess(max(s["cells"]["t"]), closed)                     # the forming minute is not served

    def test_stored_rows_win_over_trades(self):
        until = int(self.NOW.timestamp()) // 60 * 60 - 600
        base = self.stored_rows(until)
        bumped = lambda *a: base(*a).assign(buy=lambda d: d["buy"] + 1000.0)
        o = self.overrides(self.NOW, until)
        o["fetch_price_1m"] = bumped
        with cf.replay(self.store, o):
            p, _ = self.v3._build(None, "2026-10-27", session="FULL", until_pt=None, params=self.v3.fh.AbsorbParams(),
                                  session_date=dt.date(2026, 10, 27), session_note=None, now_utc=self.NOW)
        c = pd.DataFrame(self.seg(p, "rth")["cells"])
        self.assertTrue((c[c["t"] < until]["buy"] >= 1000).all())             # stored minutes come from the table
        self.assertTrue((c[c["t"] >= until]["buy"] < 1000).all())             # later minutes from the trades

    def test_no_coverage_switch_when_stored_covers_everything(self):
        # stored minutes reach the newest closed minute: still the splice, nothing dropped
        p = self.build(self.NOW, self.NOW, stored_until=int(self.NOW.timestamp()) // 60 * 60)
        self.assertEqual([s["tier"] for s in p["segments"]], ["splice", "splice"])
        self.assertEqual(max(self.seg(p, "rth")["cells"]["t"]), int(self.NOW.timestamp()) // 60 * 60 - 60)

    def test_no_stall_read_on_a_past_session(self):
        p, calls, _ = self.get(cf.QS_1027)
        self.assertNotIn("_trades_last", [c[0] for c in calls])
        self.assertTrue(all("feed_stalled_since" not in s for s in p["segments"]))
        self.assertTrue(all(mk["provisional"] is False for mk in self.seg(p, "rth")["marks"]))

if __name__ == "__main__":
    unittest.main()


class TestBars(_Base):
    """CR-DF: each segment's 1-minute flow bars (the edge readout's closes; the marks' hold / break series)."""

    def check(self, p, until_epoch=None):
        for s in p["segments"]:
            b = s["bars"]
            n = len(b["t"])
            self.assertGreater(n, 0)
            self.assertEqual({len(b[k]) for k in ("t", "h", "l", "c")}, {n})
            self.assertEqual(b["t"], sorted(set(b["t"])))
            self.assertGreaterEqual(b["t"][0], s["start"])
            self.assertLess(b["t"][-1], s["end"])
            if until_epoch is not None:
                self.assertLess(b["t"][-1], until_epoch)
            for h, l, c in zip(b["h"], b["l"], b["c"]):
                self.assertTrue(l - 1e-9 <= c <= h + 1e-9)
            for mk in s["marks"] or []:                     # breaks land on a bar of the same series
                if mk["break_time"] is not None:
                    self.assertIn(mk["break_time"], b["t"])

    def test_bars_full_and_exact(self):
        for qs in (cf.QS_0501, cf.QS_1027):
            self.check(self.get(qs)[0])

    def test_bars_respect_until(self):
        self.check(self.get(cf.QS_0501_UNTIL)[0], pt_epoch("2026-05-01T07:39:00"))
        self.check(self.get(cf.QS_1027_UNTIL)[0], pt_epoch("2026-10-27T08:15:00"))

    def test_rth_bars_match_cell_minutes(self):
        rth = self.seg(self.get(cf.QS_1027)[0], "rth")
        self.assertTrue(set(rth["cells"]["t"]) <= set(rth["bars"]["t"]))
