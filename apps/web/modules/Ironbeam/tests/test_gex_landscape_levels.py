"""CR-BV step 1 — /api/gex-landscape levels=corrected and the Current-mode byte-identity test (L2/L3).

Current mode (no `levels`, `levels=current`, any other value) must return the exact bytes the
unmodified route returned (golden captured on origin/main 0e943d8 by record_landscape_fixture.py
--golden). Corrected mode is checked on the recorded 2026-10-23 rows and on synthetic rows.

Run with:
    python -m unittest apps.web.modules.Ironbeam.tests.test_gex_landscape_levels
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import landscape_fixture as lf  # noqa: E402

from packages.shared.gex_landscape_api import build_gex_landscape_response  # noqa: E402

CURRENT_VARIANTS = ("", "&levels=current", "&levels=CURRENT", "&levels=foo", "&levels=")


def _local_minima(landscape, col="gex_total"):
    prices = [r["price"] for r in landscape]
    y = [r[col] for r in landscape]
    return sorted(
        ((prices[i], y[i]) for i in range(1, len(y) - 1)
         if y[i] < y[i - 1] and y[i] <= y[i + 1] and y[i] < 0),
        key=lambda t: t[1],
    )


class TestCurrentModeByteIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = lf.make_client(lf.ReplayEngine(lf.load_store()))
        cls.golden = lf.load_golden()

    def test_current_bodies_equal_golden(self):
        with lf.frozen_clock():
            for name, qs in lf.GOLDEN_QS.items():
                for extra in CURRENT_VARIANTS:
                    with self.subTest(request=name, levels=extra or "(absent)"):
                        r = lf.get(self.client, qs + extra)
                        self.assertEqual(r.status_code, 200)
                        self.assertEqual(r.data.decode(), self.golden[name])

    def test_current_mode_adds_no_key(self):
        with lf.frozen_clock():
            body = json.loads(lf.get(self.client, lf.GOLDEN_QS["low_iv"] + "&levels=current").data)
        for key in ("levels", "levels_note", "es_snapshot"):
            self.assertNotIn(key, body)


class TestCorrectedRecorded(unittest.TestCase):
    """2026-10-23 recorded rows: ES snapshot 7686.00 (Step 0 Q4)."""

    @classmethod
    def setUpClass(cls):
        client = lf.make_client(lf.ReplayEngine(lf.load_store()))
        with lf.frozen_clock():
            r = lf.get(client, lf.CORRECTED_QS["corrected_iv"])
        cls.status = r.status_code
        cls.body = json.loads(r.data)

    def test_payload_marks_corrected(self):
        self.assertEqual(self.status, 200)
        self.assertEqual(self.body["levels"], "corrected")
        self.assertIsNone(self.body["levels_note"])
        self.assertEqual(self.body["accuracy"], "high")
        self.assertAlmostEqual(self.body["es_snapshot"], 7686.0)

    def test_prior_spot_is_the_es_snapshot(self):
        # A1.1: table_spot (the classifier's prior_spot) is the ES 15:45 ET close on A.
        self.assertAlmostEqual(self.body["table_spot"], 7686.0)

    def test_grid_is_centred_on_the_request_spot(self):
        prices = [r["price"] for r in self.body["landscape"]]
        self.assertAlmostEqual(prices[0], 7660.75 - self.body["range_pts"])
        self.assertAlmostEqual(prices[-1], 7660.75 + self.body["range_pts"])

    def test_dominant_trough_sits_between_the_corrected_7681_and_7687_lines(self):
        price, _ = _local_minima(self.body["landscape"])[0]
        self.assertGreaterEqual(price, 7681.0)
        self.assertLessEqual(price, 7690.0)


def _conn(es_close, rows):
    """SA-connection mock for the corrected path: ES snapshot, rows, then params."""
    es_res = MagicMock()
    es_res.fetchone.return_value = (es_close,) if es_close is not None else None
    rows_res = MagicMock()
    rows_res.mappings.return_value.all.return_value = rows
    params_res = MagicMock()
    params_res.mappings.return_value.first.return_value = {
        "spread_coef": 8.0, "range_pts": 100.0, "step_pts": 1.0,
        "version": "test", "computed_at": None,
    }
    conn = MagicMock()
    conn.execute.side_effect = [es_res, rows_res, params_res]
    return conn


def _row(strike, expir, dte, fwd, gex_call, gex_put):
    return {"expir_date": expir, "dte": dte, "strike": strike, "stock_price": fwd,
            "gex_call": gex_call, "gex_put": gex_put}


class TestCorrectedSynthetic(unittest.TestCase):
    D = "2026-10-23"

    def test_kernel_centre_is_the_corrected_level(self):
        # One 0DTE strike 7650, F_e 7649.40, ES 7686.00 -> L = 7650 x 7686/7649.4 = 7686.60.
        rows = [_row(7650.0, dt.date(2026, 10, 23), 0, 7649.40, 0.0, 5e11)]
        payload, status = build_gex_landscape_response(
            _conn(7686.0, rows), "SPX", self.D, 7660.0, levels="corrected")
        self.assertEqual(status, 200)
        price, _ = _local_minima(payload["landscape"])[0]
        self.assertAlmostEqual(price, 7687.0, delta=1.0)   # 1-pt grid from 7560

    def test_case_insensitive(self):
        rows = [_row(7650.0, dt.date(2026, 10, 23), 0, 7649.40, 1e11, 0.0)]
        payload, status = build_gex_landscape_response(
            _conn(7686.0, rows), "SPX", self.D, 7660.0, levels=" Corrected ")
        self.assertEqual(status, 200)
        self.assertEqual(payload["levels"], "corrected")

    def test_pre_forward_era_note_and_no_curve(self):
        conn = MagicMock()
        payload, status = build_gex_landscape_response(
            conn, "SPX", "2024-04-02", 5250.0, levels="corrected")
        self.assertEqual(status, 200)
        self.assertEqual(payload["levels_note"], "pre_forward_era")
        self.assertEqual(payload["landscape"], [])
        conn.execute.assert_not_called()

    def test_no_es_snapshot_note(self):
        payload, status = build_gex_landscape_response(
            _conn(None, []), "SPX", self.D, 7660.0, levels="corrected")
        self.assertEqual(status, 200)
        self.assertEqual(payload["levels_note"], "no_es_snapshot")
        self.assertEqual(payload["landscape"], [])

    def test_no_rows_note(self):
        payload, status = build_gex_landscape_response(
            _conn(7686.0, []), "SPX", self.D, 7660.0, levels="corrected")
        self.assertEqual(status, 200)
        self.assertEqual(payload["levels_note"], "no_rows")


if __name__ == "__main__":
    unittest.main()
