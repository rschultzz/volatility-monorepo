"""CR-BZ step 2 — minute-bar writer: forming-bar filter and revision log (A1.2).

No network and no Postgres: `websocket` is stubbed (not installed in the web venv) and
write_snapshot_to_db runs against in-memory SQLite.

Run with:
    apps/web/.venv/bin/python -m unittest apps.cron.tests.test_ironbeam_minute_snapshot
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
import sys
import types
import unittest
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.modules.setdefault("websocket", types.ModuleType("websocket"))
_PATH = Path(__file__).resolve().parents[1] / "ironbeam_minute_snapshot.py"
_spec = importlib.util.spec_from_file_location("ironbeam_minute_snapshot", _PATH)
snap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(snap)

# SQLite stores to_sql datetimes with microseconds; bind parameters the same way so
# range comparisons behave as they do on Postgres timestamps.
for _t in (dt.datetime, pd.Timestamp):
    sqlite3.register_adapter(_t, lambda d: d.isoformat(" ", timespec="microseconds"))

T = dt.datetime(2026, 10, 23, 18, 0)  # naive UTC bar start


def bar(minute: int, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0) -> dict:
    return {"datetime": T + dt.timedelta(minutes=minute), "open": o, "high": h, "low": l, "close": c, "volume": v}


class TestForming(unittest.TestCase):

    def test_drops_only_bars_not_yet_ended(self):
        df = pd.DataFrame([bar(0), bar(1), bar(2)])
        # bar 1 ends 18:02:00; at 18:02:01 it is within the 2 s grace, at 18:02:02 it is complete
        kept = snap.drop_forming_bars(df, T + dt.timedelta(minutes=2, seconds=1))
        self.assertEqual(list(kept["datetime"]), [T])
        kept = snap.drop_forming_bars(df, T + dt.timedelta(minutes=2, seconds=2))
        self.assertEqual(list(kept["datetime"]), [T, T + dt.timedelta(minutes=1)])

    def test_epoch_parse_is_naive_utc_bar_start(self):
        t_end = dt.datetime(2026, 10, 23, 18, 1, tzinfo=dt.timezone.utc).timestamp() * 1000
        out = snap.parse_time_bars_from_message({"ti": {"t": t_end, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 7}})
        self.assertEqual(out[0]["datetime"], T)
        self.assertIsNone(out[0]["datetime"].tzinfo)


class TestFindRevisions(unittest.TestCase):

    def test_only_differing_stored_bars_are_returned(self):
        stored = pd.DataFrame([bar(0), bar(1), bar(2)])
        snapshot = pd.DataFrame([bar(0), bar(1, h=105.0, v=50.0), bar(3)])  # bar 3 is new, not a revision
        rev = snap.find_revisions(snapshot, stored)
        self.assertEqual(list(rev["datetime"]), [T + dt.timedelta(minutes=1)])
        self.assertEqual(rev.loc[0, "stored_high"], 101.0)
        self.assertEqual(rev.loc[0, "snap_high"], 105.0)
        self.assertEqual(rev.loc[0, "snap_volume"], 50.0)

    def test_null_volume_on_both_sides_is_equal(self):
        stored = pd.DataFrame([bar(0, v=None)])
        snapshot = pd.DataFrame([bar(0, v=None)])
        self.assertTrue(snap.find_revisions(snapshot, stored).empty)


class TestWriteSnapshot(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine("sqlite://")
        with self.engine.begin() as conn:
            pd.DataFrame([bar(0), bar(1)]).to_sql(snap.DB_TABLE_NAME, conn, index=False)
            conn.execute(text(
                f"CREATE TABLE {snap.REVISION_TABLE_NAME} (datetime TIMESTAMP, "
                "stored_open REAL, stored_high REAL, stored_low REAL, stored_close REAL, stored_volume REAL, "
                "snap_open REAL, snap_high REAL, snap_low REAL, snap_close REAL, snap_volume REAL, "
                "seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY (datetime, snap_open, snap_high, snap_low, snap_close, snap_volume))"
            ))

    def rows(self, table):
        with self.engine.connect() as conn:
            return pd.read_sql(text(f"SELECT * FROM {table} ORDER BY datetime"), conn)

    def run_snapshot(self, bars, now):
        snap.write_snapshot_to_db(bars, self.engine, now_utc=now)

    def test_inserts_complete_new_bars_skips_forming_and_never_changes_stored(self):
        now = T + dt.timedelta(minutes=3, seconds=30)  # bar 3 still forming
        self.run_snapshot([bar(0), bar(1, c=100.75, v=12.0), bar(2), bar(3)], now)
        b = self.rows(snap.DB_TABLE_NAME)
        self.assertEqual(len(b), 3)  # bar 2 inserted, bar 3 dropped
        self.assertEqual(b.loc[1, "close"], 100.5)  # stored bar 1 unchanged
        r = self.rows(snap.REVISION_TABLE_NAME)
        self.assertEqual(len(r), 1)
        self.assertEqual((r.loc[0, "stored_close"], r.loc[0, "snap_close"]), (100.5, 100.75))
        self.assertEqual((r.loc[0, "stored_volume"], r.loc[0, "snap_volume"]), (10.0, 12.0))

    def test_same_revision_seen_twice_is_logged_once(self):
        now = T + dt.timedelta(minutes=2, seconds=30)
        for _ in range(2):
            self.run_snapshot([bar(0), bar(1, c=100.75)], now)
        self.assertEqual(len(self.rows(snap.REVISION_TABLE_NAME)), 1)

    def test_missing_revision_table_does_not_block_bars(self):
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {snap.REVISION_TABLE_NAME}"))
        self.run_snapshot([bar(1, c=100.75), bar(2)], T + dt.timedelta(minutes=3, seconds=5))
        self.assertEqual(len(self.rows(snap.DB_TABLE_NAME)), 3)


if __name__ == "__main__":
    unittest.main()
