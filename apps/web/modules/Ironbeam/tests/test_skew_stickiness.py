"""Tests for apps/web/modules/Ironbeam/skew_stickiness.py (CR-CK Part 3). No database.

Run with:
    python -m unittest apps.web.modules.Ironbeam.tests.test_skew_stickiness
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[5]
for p in (REPO_ROOT, REPO_ROOT / "apps" / "web"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402
from flask import Flask  # noqa: E402

from packages.shared import skew_expected as se  # noqa: E402
from modules.Ironbeam import skew_stickiness as ss  # noqa: E402


def smile_row(put_skew_pp: float, atm: float = 0.20):
    """vol5..vol95 with vol50 = atm and a convex put wing; vol75 - vol50 = put_skew_pp / 100."""
    T = se.years_to_close_0dte(450)
    row = {}
    for n in ss.VOL_BUCKETS:
        if n == 50:
            k = 0.0
        else:
            p, is_put = se._abs_delta_is_put(n)
            k = se.k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        row[f"vol{n}"] = atm - 0.8 * k + 40.0 * k * k
    row["vol75"] = atm + put_skew_pp / 100.0
    return row


def snaps(minutes, put_fn):
    return pd.DataFrame([{"minute": m, "quote_minute": m, **smile_row(put_fn(m))} for m in minutes])


class TestRows(unittest.TestCase):
    def test_first_time_and_window(self):
        df = snaps(range(450, 471), lambda m: 2.0 - 0.05 * (m - 450))           # put skew -1.0 over 07:30-07:50
        es = {m - 1: 7000.0 + 1.4 * (m - 450) for m in range(445, 472)}        # +28 pts = +0.4 %
        rows = ss.build_skew_stickiness_rows(["07:50", "07:30"], df, es)
        self.assertEqual([r["time"] for r in rows], ["07:30", "07:50"])
        self.assertEqual(rows[0]["reason"], "first time")
        r = rows[1]
        self.assertEqual(r["window"], "07:30-07:50")
        self.assertEqual(r["reason"], "ok")
        self.assertAlmostEqual(r["actual"], -1.0, places=2)
        self.assertAlmostEqual(r["es_move_pct"], 0.4, places=3)
        self.assertLess(r["expected"], 0.0)                                     # rally, convex smile: SS says skew falls
        self.assertAlmostEqual(r["ratio"], round(r["actual"] / r["expected"], 2), delta=0.02)
        self.assertIsNotNone(r["beyond_noise"])

    def test_late_day(self):
        df = snaps(range(650, 671), lambda m: 2.0)                              # 10:50 .. 11:10
        es = {m - 1: 7000.0 for m in range(645, 672)}
        rows = ss.build_skew_stickiness_rows(["10:50", "11:00", "11:10"], df, es)
        self.assertEqual([r["late_day"] for r in rows], [False, False, True])

    def test_move_too_small_and_missing(self):
        df = snaps(range(450, 471), lambda m: 2.0)
        es = {m - 1: 7000.0 for m in range(445, 472)}
        rows = ss.build_skew_stickiness_rows(["07:30", "07:50"], df, es)
        self.assertEqual(rows[1]["reason"], "move too small")
        self.assertIsNone(rows[1]["ratio"])
        rows = ss.build_skew_stickiness_rows(["07:30", "07:50"], df, {})
        self.assertEqual(rows[1]["reason"], "insufficient data")


class TestPayload(unittest.TestCase):
    def test_not_0dte_and_until(self):
        p, st = ss.build_skew_stickiness_payload(MagicMock(), "2026-10-26", "2026-10-27", ["07:30", "07:50"])
        self.assertEqual(st, 200)
        self.assertEqual({r["reason"] for r in p["stickiness"]}, {"0DTE only"})
        p, _ = ss.build_skew_stickiness_payload(MagicMock(), "2026-10-26", "2026-10-27", ["07:30", "07:50"],
                                                until_pt=dt.datetime(2026, 10, 26, 7, 40, tzinfo=ss.PT))
        self.assertEqual([r["time"] for r in p["stickiness"]], ["07:30"])

    def test_missing_args(self):
        self.assertEqual(ss.build_skew_stickiness_payload(MagicMock(), None, "2026-10-26", [])[1], 400)

    def test_single_time_no_db(self):
        conn = MagicMock()
        p, st = ss.build_skew_stickiness_payload(conn, "2026-10-26", "2026-10-26", ["07:30"])
        self.assertEqual(p["stickiness"][0]["reason"], "first time")
        conn.execute.assert_not_called()

    def test_route_registered_once(self):
        app = Flask(__name__)
        eng = MagicMock()
        ss.register_skew_stickiness_route(app, eng, lambda raw, tz: None)
        ss.register_skew_stickiness_route(app, eng, lambda raw, tz: None)
        rules = [r.rule for r in app.url_map.iter_rules() if r.rule == "/api/ironbeam/skew-stickiness"]
        self.assertEqual(len(rules), 1)
        resp = app.test_client().get("/api/ironbeam/skew-stickiness?trade_date=2026-10-26&expiration_date=2026-10-27"
                                     "&times=07:30,07:50")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()["stickiness"]), 2)


if __name__ == "__main__":
    unittest.main()
