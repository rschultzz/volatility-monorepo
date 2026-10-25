"""CR-CD A4.4 — how Flow v2 splits a window between the 1-minute and 1-second tables (no DB).

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_flow_1m_split
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

from apps.web.modules.Ironbeam.tests import bars_fixture as bf  # noqa: F401  (sys.path, lazy engine)

UTC = dt.timezone.utc


def t(h, m, s=0):
    return dt.datetime(2026, 5, 1, h, m, s, tzinfo=UTC)


class TestSplitRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bf.callbacks_module()
        import modules.Ironbeam.flow_v2 as v2
        cls.v2 = v2

    def test_aligned_window_before_cut_is_all_1m(self):
        self.assertEqual(self.v2.split_ranges([(t(13, 30), t(20, 0))], t(23, 0)),
                         ([(t(13, 30), t(20, 0))], []))

    def test_ragged_edges_go_to_1s(self):
        # a CR-CC `until` mid-minute and a start mid-minute
        r1m, r1s = self.v2.split_ranges([(t(13, 25, 30), t(14, 39, 20))], t(23, 0))
        self.assertEqual(r1m, [(t(13, 26), t(14, 39))])
        self.assertEqual(r1s, [(t(13, 25, 30), t(13, 26)), (t(14, 39), t(14, 39, 20))])

    def test_after_cut_goes_to_1s(self):
        r1m, r1s = self.v2.split_ranges([(t(13, 30), t(20, 0))], t(17, 12))
        self.assertEqual(r1m, [(t(13, 30), t(17, 12))])
        self.assertEqual(r1s, [(t(17, 12), t(20, 0))])

    def test_no_table_or_window_after_cut_is_all_1s(self):
        self.assertEqual(self.v2.split_ranges([(t(13, 30), t(20, 0))], None), ([], [(t(13, 30), t(20, 0))]))
        self.assertEqual(self.v2.split_ranges([(t(18, 0), t(20, 0))], t(17, 0)), ([], [(t(18, 0), t(20, 0))]))
        self.assertEqual(self.v2.split_ranges([(t(13, 30, 5), t(13, 30, 50))], t(23, 0)),
                         ([], [(t(13, 30, 5), t(13, 30, 50))]))

    def test_fetch_concatenates_both_parts_in_order(self):
        v2 = self.v2
        cols = ["symbol", "t", "o", "h", "l", "c", "buy", "sell", "vol", "n"]
        part_1m = pd.DataFrame([["XCME:ES.M26", 120, 1, 2, 0.5, 1.5, 3, 4, 7, 50]], columns=cols)
        part_1s = pd.DataFrame([["XCME:ES.M26", 60, 1, 1, 1, 1, 1, 0, 1, 1],
                                ["XCME:ES.M26", 180, 2, 2, 2, 2, 0, 1, 1, 1]], columns=cols)
        sqls = []

        def fake_read(engine, sql, params):
            sqls.append(sql)
            return part_1m.copy() if "minute_utc" in sql else part_1s.copy()

        with patch.object(v2, "_cutoff_1m", return_value=t(0, 3)), patch.object(v2, "_read", fake_read):
            out = v2.fetch_flow_minutes(None, "ironbeam_es_flow_1s",
                                        [(dt.datetime(2026, 5, 1, 0, 0, 30, tzinfo=UTC), t(0, 3, 30))])
        self.assertEqual(out["t"].tolist(), [60, 120, 180])
        self.assertEqual(list(out.columns), cols)
        self.assertEqual(len(sqls), 2)
        self.assertIn("FROM ironbeam_es_flow_1m", sqls[0])
        self.assertIn("FROM ironbeam_es_flow_1s", sqls[1])


if __name__ == "__main__":
    unittest.main()
