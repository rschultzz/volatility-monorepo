"""CR-BZ step 3 — fragment-bar repair: plan, logging order, guarded single transaction (A1.3).

No DB: `write` runs against a recording fake connection.

Run with:
    apps/web/.venv/bin/python -m unittest scripts.tests.test_cr_bz_repair_fragment_bars
"""
from __future__ import annotations

import datetime as dt
import unittest
from contextlib import contextmanager

import pandas as pd

from scripts import cr_bz_repair_fragment_bars as rep

T = dt.datetime(2026, 10, 8, 16, 17)


def frag(minute=0, nsym=1, em_present=True, em_close=7842.0, rth=True):
    """One selected row shaped like SELECT_SQL output (10-08 16:17 UTC TAIL fragment)."""
    return {
        "datetime": pd.Timestamp(T + dt.timedelta(minutes=minute)),
        "bar_open": 7847.0, "bar_high": 7850.0, "bar_low": 7842.0, "bar_close": 7842.0, "bar_volume": 8469.0,
        "bar_source": "ironbeam",
        "new_open": 7826.0, "new_high": 7850.0, "new_low": 7825.75, "new_close": 7842.0, "new_volume": 23906.0,
        "symbol": "XCME:ES.Z26", "nsym": nsym, "n_sec": 57,
        "em_present": em_present,
        "em_open": 7847.0 if em_present else None, "em_high": 7850.0 if em_present else None,
        "em_low": 7842.0 if em_present else None, "em_close": em_close if em_present else None,
        "em_volume": 8469.0 if em_present else None, "em_is_rth": rth,
        "rth": rth, "month": "2026-10",
    }


def frame(*rows):
    df = pd.DataFrame(list(rows))
    df["em_equals_bar"] = df["em_present"] & df.apply(
        lambda r: all(r[f"em_{f}"] == r[f"bar_{f}"] for f in rep.OHLCV), axis=1)
    return df


class TestPlan(unittest.TestCase):

    def test_new_values_from_flow_and_both_tables(self):
        bars, em = rep.plan(frame(frag()))
        self.assertEqual(len(bars), 1)
        self.assertEqual(len(em), 1)
        r = bars.iloc[0]
        self.assertEqual((r.old_open, r.new_open), (7847.0, 7826.0))
        self.assertEqual((r.old_low, r.new_low), (7842.0, 7825.75))
        self.assertEqual((r.old_volume, r.new_volume), (8469.0, 23906.0))
        self.assertEqual(em.iloc[0].old_close, 7842.0)

    def test_multi_symbol_minutes_skipped_and_missing_es_minutes_not_planned(self):
        df = frame(frag(0), frag(1, nsym=2), frag(2, em_present=False))
        bars, em = rep.plan(df)
        self.assertEqual(len(bars), 2)  # minute 1 skipped
        self.assertEqual(len(em), 1)    # minute 2 has no es_minutes row
        s = rep.summarize(df, bars, em)
        self.assertEqual(s["skipped_multi_symbol"], 1)
        self.assertEqual(s["es_minutes_missing"], 1)

    def test_es_minutes_logged_with_its_own_old_values(self):
        df = frame(frag(em_close=7841.0))
        bars, em = rep.plan(df)
        self.assertEqual(em.iloc[0].old_close, 7841.0)
        self.assertEqual(rep.summarize(df, bars, em)["es_minutes_differs_from_bar"], 1)


class FakeConn:
    def __init__(self, rowcounts=None):
        self.calls = []
        self.rowcounts = rowcounts or {}
        self.in_tx = False
        self.committed = False

    @contextmanager
    def transaction(self):
        self.in_tx = True
        try:
            yield
            self.committed = True
        finally:
            self.in_tx = False

    def execute(self, sql, params=None):
        assert self.in_tx, "write outside the transaction"
        kind = ("log" if "INSERT INTO" in sql else
                "bars" if "UPDATE ironbeam_es_1m_bars" in sql else
                "em" if "UPDATE es_minutes" in sql else "other")
        self.calls.append((kind, params))
        return type("R", (), {"rowcount": self.rowcounts.get(kind, 1)})()


class TestWrite(unittest.TestCase):

    def test_logs_both_tables_before_any_update(self):
        bars, em = rep.plan(frame(frag(0), frag(1)))
        conn = FakeConn()
        rep.write(conn, "run-1", bars, em)
        kinds = [k for k, _ in conn.calls]
        self.assertEqual(kinds, ["log"] * 4 + ["bars"] * 2 + ["em"] * 2)
        self.assertEqual({p["target_table"] for k, p in conn.calls if k == "log"}, {"ironbeam_es_1m_bars", "es_minutes"})
        self.assertEqual(conn.calls[0][1]["bar_ts"].tzinfo, dt.timezone.utc)
        self.assertTrue(conn.committed)

    def test_guard_miss_aborts_the_transaction(self):
        bars, em = rep.plan(frame(frag()))
        conn = FakeConn(rowcounts={"em": 0})
        with self.assertRaises(RuntimeError):
            rep.write(conn, "run-1", bars, em)
        self.assertFalse(conn.committed)

    def test_updates_are_guarded_on_old_values(self):
        for sql in (rep.UPDATE_BARS, rep.UPDATE_ES_MINUTES):
            for f in ("open", "high", "low", "close"):
                self.assertIn(f"{f} = %(old_{f})s", sql)
            self.assertIn("volume IS NOT DISTINCT FROM %(old_volume)s", sql)


if __name__ == "__main__":
    unittest.main()
