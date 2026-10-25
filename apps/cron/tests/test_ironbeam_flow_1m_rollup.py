"""CR-CD A4.3 — the 1-minute rollup window in the Ironbeam 1-second aggregate worker (no DB).

Run with:
    apps/web/.venv/bin/python -m unittest apps.cron.tests.test_ironbeam_flow_1m_rollup
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "ironbeam_flow_1s_aggregate.py"
_spec = importlib.util.spec_from_file_location("ironbeam_flow_1s_aggregate", _PATH)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from packages.shared import flow_1m  # noqa: E402

UTC = dt.timezone.utc


def t(h, m, s=0, day=26):
    return dt.datetime(2026, 10, day, h, m, s, tzinfo=UTC)


class TestRollupWindow(unittest.TestCase):
    def test_caught_up_rolls_the_last_three_completed_minutes(self):
        # steady state: stored through the 14:33 minute -> next 1s row at/after 14:31 is 14:31:01;
        # 1s state 14:35:05, now 14:35:07, overlap 10 s -> end 14:34 -> minutes 14:31, 14:32, 14:33
        w = agg.rollup_window(next_1s=t(14, 31, 1), state_ts=t(14, 35, 5), now=t(14, 35, 7), overlap_sec=10)
        self.assertEqual(w, (t(14, 31), t(14, 34)))

    def test_lagging_watermark_re_rolls_from_two_minutes_before_it(self):
        # stored only through 14:31 (a pass was missed): re-roll from 14:29 to 14:34
        w = agg.rollup_window(next_1s=t(14, 29, 1), state_ts=t(14, 35, 5), now=t(14, 35, 7), overlap_sec=10)
        self.assertEqual(w, (t(14, 29), t(14, 34)))

    def test_end_never_passes_the_1s_state(self):
        # the 1-second worker is behind (state 14:20:30) although now is 14:35
        w = agg.rollup_window(next_1s=t(14, 10, 3), state_ts=t(14, 20, 30), now=t(14, 35), overlap_sec=10)
        self.assertEqual(w, (t(14, 10), t(14, 20)))

    def test_catch_up_is_capped_at_six_hours(self):
        w = agg.rollup_window(next_1s=t(1, 0, 4), state_ts=t(14, 35), now=t(14, 35), overlap_sec=10)
        self.assertEqual(w, (t(1, 0), t(7, 0)))

    def test_quiet_stretch_is_jumped(self):
        # stored through Fri 21:00 UTC; the next 1-second row is Sun 22:00 (Globex open) -> start there
        w = agg.rollup_window(next_1s=t(22, 0, 2, day=25), state_ts=t(22, 30, day=25), now=t(22, 30, 1, day=25),
                              overlap_sec=10)
        self.assertEqual(w, (t(22, 0, day=25), t(22, 29, day=25)))

    def test_no_1s_rows(self):
        self.assertIsNone(agg.rollup_window(None, t(14, 35), t(14, 35), 10))

    def test_no_new_data_rolls_only_recent_minutes(self):
        # next row is after the window end (nothing new yet): just the last 3 completed minutes
        w = agg.rollup_window(next_1s=t(14, 40), state_ts=t(14, 35, 15), now=t(14, 35, 16), overlap_sec=10)
        self.assertEqual(w, (t(14, 32), t(14, 35)))


class TestRollupSql(unittest.TestCase):
    def test_worker_upserts_with_the_shared_definition(self):
        sql = flow_1m.insert_sql("ironbeam_es_flow_1s", "symbol = :sym AND ts_utc >= :a AND ts_utc < :b", "update")
        self.assertIn("INSERT INTO ironbeam_es_flow_1m", sql)
        self.assertIn("ON CONFLICT (minute_utc, symbol) DO UPDATE SET volume = EXCLUDED.volume", sql)
        self.assertIn("(array_agg(open ORDER BY ts_utc))[1] AS open", sql)
        self.assertIn("count(*)::integer AS n_seconds", sql)
        self.assertNotIn("raw_symbol", sql)

    def test_backfill_is_insert_only(self):
        sql = flow_1m.insert_sql("es_flow_1s_databento", "ts_utc >= %(a)s AND ts_utc < %(b)s", "nothing")
        self.assertIn("INSERT INTO es_flow_1m_databento", sql)
        self.assertTrue(sql.rstrip().endswith("ON CONFLICT (minute_utc, symbol) DO NOTHING"))
        self.assertIn("max(raw_symbol) AS raw_symbol", sql)
        with self.assertRaises(ValueError):
            flow_1m.insert_sql("es_flow_1s_databento", "true", "replace")


if __name__ == "__main__":
    unittest.main()
