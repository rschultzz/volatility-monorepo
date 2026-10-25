"""CR-CC step 1 — gallery validation and insert (L3, L10, A1.4).

No DB: insert_rows runs against a recording fake connection.

Run with:
    apps/web/.venv/bin/python -m unittest scripts.tests.test_cr_cc_load_setup_candidates
"""
from __future__ import annotations

import copy
import datetime as dt
import unittest

from scripts import load_setup_candidates as lsc

TODAY = dt.date(2026, 10, 25)


def event(**over):
    ev = {
        "id": 1,
        "session_date": "2026-03-10",
        "cell": "put_low",
        "direction": "long",
        "level_price": 6750.25,
        "level_net_bn": -12.5,
        "level_gross_bn": 40.1,
        "entry_ts_pt": "2026-03-10T07:37:00",
        "entry_price": 6752.0,
        "stop_price": 6745.5,
        "extreme_price": 6746.0,
        "test_ts_pt": "2026-03-10T07:21",
        "confirm_ts_pt": "2026-03-10T07:30:00",
        "retest_ts_pt": None,
        "meta": {"overshoot_pts": 4.25, "retest_ts": "x", "reason": "y", "window": 3},
        "outcome": {"exit_ts_pt": "2026-03-10T08:10:00", "exit_price": 6770.0, "pnl_pts": 18.0, "result": "target"},
    }
    ev.update(over)
    return ev


def gallery(*events, **over):
    doc = {"cr_id": "CR-CC", "set_name": "cc:fixture", "level_frame": "es_corrected",
           "generated_at": "2026-10-25T10:00:00", "events": list(events) or [event()]}
    doc.update(over)
    return doc


class TestValidate(unittest.TestCase):
    def test_valid_row_shape(self):
        rows = lsc.validate_gallery(gallery(), TODAY)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(set(r), set(lsc.COLUMNS))
        self.assertEqual(r["id"], "CR-CC:cc:fixture:1")
        self.assertEqual(r["session_date"], dt.date(2026, 3, 10))
        self.assertEqual(r["entry_ts_pt"], dt.datetime(2026, 3, 10, 7, 37))
        self.assertEqual(r["test_ts_pt"], dt.datetime(2026, 3, 10, 7, 21))
        self.assertIsNone(r["retest_ts_pt"])
        self.assertEqual(r["meta"]["level_frame"], "es_corrected")
        self.assertEqual(r["outcome"]["pnl_pts"], 18.0)

    def assertRejected(self, doc, fragment):
        with self.assertRaises(lsc.GalleryError) as ctx:
            lsc.validate_gallery(doc, TODAY)
        self.assertIn(fragment, str(ctx.exception))

    def test_top_level_required(self):
        self.assertRejected(gallery(level_frame="ES"), "level_frame")
        self.assertRejected(gallery(set_name=""), "set_name")
        self.assertRejected(gallery(events=[]), "events")

    def test_session_date_must_be_before_today(self):
        self.assertRejected(gallery(event(session_date="2026-10-25",
                                          entry_ts_pt="2026-10-25T07:00:00",
                                          test_ts_pt=None, confirm_ts_pt=None)), "not before today")

    def test_times_must_be_naive_pt(self):
        self.assertRejected(gallery(event(entry_ts_pt="2026-03-10T14:37:00Z")), "naive PT")
        self.assertRejected(gallery(event(entry_ts_pt="2026-03-10T07:37:00-07:00")), "naive PT")

    def test_times_inside_session_window(self):
        self.assertRejected(gallery(event(entry_ts_pt="2026-03-10T13:30:00")), "outside the session window")
        # ETH the prior evening belongs to the session.
        lsc.validate_gallery(gallery(event(entry_ts_pt="2026-03-09T18:05:00", test_ts_pt=None,
                                           confirm_ts_pt=None)), TODAY)

    def test_outcome_like_meta_keys_rejected(self):
        for key in ("pnl_a", "res_4", "mv_30m", "exit_price", "r10", "f15", "result", "by_exit", "outcome"):
            with self.subTest(key=key):
                self.assertRejected(gallery(event(meta={key: 1})), "outcome-like")

    def test_direction_and_duplicates(self):
        self.assertRejected(gallery(event(direction="up")), "direction")
        self.assertRejected(gallery(event(), event()), "duplicate id")

    def test_all_errors_reported_together(self):
        with self.assertRaises(lsc.GalleryError) as ctx:
            lsc.validate_gallery(gallery(event(direction="x", meta={"pnl": 1})), TODAY)
        msg = str(ctx.exception)
        self.assertIn("direction", msg)
        self.assertIn("outcome-like", msg)


class _Cur:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _FakeConn:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        cand_id = params[0]
        if cand_id in self.existing:
            return _Cur(0)
        self.existing.add(cand_id)
        return _Cur(1)


class TestInsert(unittest.TestCase):
    def test_insert_only_on_conflict_do_nothing(self):
        self.assertIn("ON CONFLICT (id) DO NOTHING", lsc.INSERT_SQL)
        self.assertNotIn("UPDATE", lsc.INSERT_SQL.upper().replace("ON CONFLICT (ID) DO NOTHING", ""))
        rows = lsc.validate_gallery(gallery(event(), event(id=2)), TODAY)
        conn = _FakeConn(existing={"CR-CC:cc:fixture:1"})
        self.assertEqual(lsc.insert_rows(conn, rows), (1, 1))
        params = conn.calls[1][1]
        self.assertEqual(len(params), len(lsc.COLUMNS))
        # meta / outcome are wrapped as JSON
        self.assertEqual(type(params[lsc.COLUMNS.index("meta")]).__name__, "Jsonb")
        self.assertEqual(type(params[lsc.COLUMNS.index("outcome")]).__name__, "Jsonb")

    def test_validate_does_not_mutate_input(self):
        doc = gallery()
        before = copy.deepcopy(doc)
        lsc.validate_gallery(doc, TODAY)
        self.assertEqual(doc, before)


if __name__ == "__main__":
    unittest.main()
