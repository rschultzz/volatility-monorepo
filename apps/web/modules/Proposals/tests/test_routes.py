"""Tests for POST /api/proposals/pl-data (CR-G Step 4).

Covers:
  - 400 validation: missing trade_date, invalid timeframe, malformed legs,
    missing regime_block, regime_block without 'regime'
  - 503 on DB connect failure
  - 200 happy path: mocked DB + analogue/edge functions return minimal valid data

Run with:
    python -m pytest apps/web/modules/Proposals/tests/test_routes.py -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask
from apps.web.modules.Proposals.routes import register_proposals_routes, _fetch_smile_row

_MODULE = "apps.web.modules.Proposals.routes"


# ── Fixtures ──────────────────────────────────────────────────────────────────

_VALID_BODY = {
    "trade_date":   "2023-05-01",
    "ticker":       "SPX",
    "timeframe":    "t5",
    "regime_block": {"regime": "magnet-above", "drift_target": 4187.0},
    "legs": [
        {
            "strike":     4225,
            "expiration": "2023-05-05",
            "flag":       "c",
            "side":       "long",
            "qty":        1,
        },
        {
            "strike":     4250,
            "expiration": "2023-05-05",
            "flag":       "c",
            "side":       "short",
            "qty":        1,
        },
    ],
}

# Minimal analogue list (empty is OK for testing the pipeline; edge_zones is mocked)
_MOCK_ANALOGUES: list[dict] = []

_MOCK_FV: dict = {
    "implied_move_1d": 23.59,
    "regime_magnet_above": 1,
    "regime_magnet_below": 0,
}


def _make_app() -> Flask:
    server = Flask(__name__)
    register_proposals_routes(server)
    return server


def _post(app: Flask, body: dict) -> tuple:
    """POST body to /api/proposals/pl-data; returns (status_code, json_dict)."""
    with app.test_client() as client:
        resp = client.post(
            "/api/proposals/pl-data",
            data=json.dumps(body),
            content_type="application/json",
        )
        return resp.status_code, resp.get_json()


# ── _fetch_smile_row unit tests ───────────────────────────────────────────────

class TestFetchSmileRow(unittest.TestCase):
    """_fetch_smile_row must return a 3-tuple: (atmiv, risk_free_rate, yield_rate)."""

    import datetime as _dt

    def _make_conn(self, db_row):
        """Build a minimal mock connection that returns db_row from cursor.fetchone."""
        cursor = MagicMock()
        cursor.__enter__ = lambda s: cursor
        cursor.__exit__  = MagicMock(return_value=False)
        cursor.fetchone.return_value = db_row
        conn = MagicMock()
        conn.cursor.return_value = cursor
        return conn

    def test_returns_three_tuple_on_hit(self):
        import datetime as dt
        conn = self._make_conn((0.18, 0.05, 0.015))
        result = _fetch_smile_row(conn, "SPX", dt.date(2023, 5, 1), dt.date(2023, 5, 19))
        self.assertEqual(len(result), 3)
        atmiv, rfr, yr = result
        self.assertAlmostEqual(atmiv, 0.18)
        self.assertAlmostEqual(rfr, 0.05)
        self.assertAlmostEqual(yr, 0.015)

    def test_returns_none_triple_on_miss(self):
        import datetime as dt
        conn = self._make_conn(None)
        result = _fetch_smile_row(conn, "SPX", dt.date(2023, 5, 1), dt.date(2023, 5, 19))
        self.assertEqual(result, (None, None, None))

    def test_yield_rate_defaults_to_zero_when_db_null(self):
        import datetime as dt
        conn = self._make_conn((0.18, 0.05, None))  # yield_rate is NULL in DB
        _, _, yield_r = _fetch_smile_row(
            conn, "SPX", dt.date(2023, 5, 1), dt.date(2023, 5, 19)
        )
        self.assertEqual(yield_r, 0.0)

    def test_rfr_defaults_to_fallback_when_db_null(self):
        import datetime as dt
        conn = self._make_conn((0.18, None, 0.015))  # risk_free_rate is NULL
        _, rfr, _ = _fetch_smile_row(
            conn, "SPX", dt.date(2023, 5, 1), dt.date(2023, 5, 19)
        )
        self.assertEqual(rfr, 0.05)  # fallback default


# ── 400 Validation tests ──────────────────────────────────────────────────────

class TestValidation(unittest.TestCase):
    """POST /api/proposals/pl-data validation — all paths must return 400."""

    def setUp(self):
        self.app = _make_app()

    def test_400_missing_trade_date(self):
        body = {**_VALID_BODY}
        del body["trade_date"]
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("trade_date" in e for e in data["errors"]))

    def test_400_invalid_trade_date_format(self):
        body = {**_VALID_BODY, "trade_date": "01-05-2023"}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("trade_date" in e for e in data["errors"]))

    def test_400_invalid_timeframe(self):
        body = {**_VALID_BODY, "timeframe": "t3"}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("timeframe" in e for e in data["errors"]))

    def test_400_missing_regime_block(self):
        body = {k: v for k, v in _VALID_BODY.items() if k != "regime_block"}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("regime_block" in e for e in data["errors"]))

    def test_400_regime_block_missing_regime_key(self):
        body = {**_VALID_BODY, "regime_block": {"drift_target": 4187.0}}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("regime_block" in e for e in data["errors"]))

    def test_400_empty_legs(self):
        body = {**_VALID_BODY, "legs": []}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("legs" in e for e in data["errors"]))

    def test_400_legs_not_a_list(self):
        body = {**_VALID_BODY, "legs": "not-a-list"}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])

    def test_400_leg_missing_flag(self):
        bad_legs = [
            {
                "strike":     4225,
                "expiration": "2023-05-05",
                # flag missing
                "side":       "long",
                "qty":        1,
            }
        ]
        body = {**_VALID_BODY, "legs": bad_legs}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertTrue(any("legs[0]" in e for e in data["errors"]))

    def test_400_leg_invalid_side(self):
        bad_legs = [
            {
                "strike":     4225,
                "expiration": "2023-05-05",
                "flag":       "c",
                "side":       "buy",   # invalid — must be 'long'/'short'
                "qty":        1,
            }
        ]
        body = {**_VALID_BODY, "legs": bad_legs}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])

    def test_400_leg_invalid_expiration(self):
        bad_legs = [
            {
                "strike":     4225,
                "expiration": "not-a-date",
                "flag":       "c",
                "side":       "long",
                "qty":        1,
            }
        ]
        body = {**_VALID_BODY, "legs": bad_legs}
        status, data = _post(self.app, body)
        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])


# ── 503 DB failure ────────────────────────────────────────────────────────────

class TestDbFailure(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()

    def test_503_db_connect_failure(self):
        with patch(f"{_MODULE}._conn", side_effect=RuntimeError("connection refused")):
            status, data = _post(self.app, _VALID_BODY)
        self.assertEqual(status, 503)
        self.assertFalse(data["ok"])
        self.assertIn("db connect failed", data["error"])


# ── 200 Happy path ────────────────────────────────────────────────────────────

class TestHappyPath(unittest.TestCase):
    """Happy-path test: mocks all DB queries and downstream analytics functions.

    Validates response shape — keys present and types correct — without requiring
    a live database or real option pricing (those are tested in their own modules).
    """

    def setUp(self):
        self.app = _make_app()

    def _run(self) -> tuple[int, dict]:
        mock_conn = MagicMock()

        # Cursor mock for _fetch_anchor_data — two sequential cursor calls
        fv_cursor    = MagicMock()
        out_cursor   = MagicMock()
        smile_cursor = MagicMock()

        fv_cursor.__enter__ = lambda s: fv_cursor
        fv_cursor.__exit__  = MagicMock(return_value=False)
        fv_cursor.fetchone.return_value = (_MOCK_FV,)

        out_cursor.__enter__ = lambda s: out_cursor
        out_cursor.__exit__  = MagicMock(return_value=False)
        out_cursor.fetchone.return_value = (4184.25,)   # session_open_t0

        smile_cursor.__enter__ = lambda s: smile_cursor
        smile_cursor.__exit__  = MagicMock(return_value=False)
        smile_cursor.fetchone.return_value = (0.15, 0.05, 0.015)  # atmiv, rfr, yield_rate

        # conn.cursor() returns cursors in order: fv, out, smile
        mock_conn.cursor.side_effect = [fv_cursor, out_cursor, smile_cursor]

        # Mock implied_pdf + prob so the test is independent of numpy version:
        # implied_distribution.py uses np.trapezoid (NumPy ≥ 2.0) which isn't
        # available in the local Python 3.9 test environment (NumPy 1.20).
        # Production runs NumPy 2.2.6 from requirements.txt; that layer is
        # tested via packages/shared/tests. Here we just verify the pipeline wires up.
        patches = [
            patch(f"{_MODULE}._conn",                    return_value=mock_conn),
            patch(f"{_MODULE}._rank_analogues_with_outcomes", return_value=_MOCK_ANALOGUES),
            patch(f"{_MODULE}.compute_edge_zones",       return_value=[]),
            patch(f"{_MODULE}.compute_implied_pdf",      return_value={4000.0: 0.01, 4400.0: 0.01}),
            patch(f"{_MODULE}.compute_implied_prob_in_range", return_value=0.37),
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            return _post(self.app, _VALID_BODY)

    def test_200_response_ok(self):
        status, data = self._run()
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])

    def test_200_top_level_keys_present(self):
        _, data = self._run()
        required = {
            "ok", "trade_date", "ticker", "evaluation_time",
            "current_spot", "implied_move", "legs",
            "pl_curve", "iv_curve",
            "trade_thesis", "edge_zones", "greeks", "key_levels", "warnings",
        }
        self.assertTrue(required.issubset(data.keys()), f"Missing keys: {required - data.keys()}")

    def test_200_trade_date_echoed(self):
        _, data = self._run()
        self.assertEqual(data["trade_date"], "2023-05-01")

    def test_200_ticker_echoed(self):
        _, data = self._run()
        self.assertEqual(data["ticker"], "SPX")

    def test_200_evaluation_time_is_utc_iso(self):
        _, data = self._run()
        # Should end in +00:00 (UTC)
        self.assertIn("+00:00", data["evaluation_time"])

    def test_200_pl_curve_has_prices_and_pnl(self):
        _, data = self._run()
        self.assertIn("prices", data["pl_curve"])
        self.assertIn("pnl",    data["pl_curve"])
        self.assertIsInstance(data["pl_curve"]["prices"], list)
        self.assertIsInstance(data["pl_curve"]["pnl"],    list)
        self.assertGreater(len(data["pl_curve"]["prices"]), 0)

    def test_200_legs_have_iv_and_initial_value(self):
        _, data = self._run()
        for leg in data["legs"]:
            self.assertIn("iv",            leg)
            self.assertIn("initial_value", leg)

    def test_200_legs_have_strike_spx(self):
        """Each leg in the response must carry strike_spx as a multiple of 5."""
        _, data = self._run()
        for leg in data["legs"]:
            self.assertIn("strike_spx", leg)
            self.assertIsInstance(leg["strike_spx"], int)
            self.assertEqual(leg["strike_spx"] % 5, 0)

    def test_200_trade_thesis_has_required_fields(self):
        _, data = self._run()
        tt = data["trade_thesis"]
        for field in ("lower", "upper", "regime_kind", "structural_prob",
                      "implied_prob", "edge_ratio"):
            self.assertIn(field, tt)

    def test_200_magnet_above_trade_thesis_lower_is_drift_target(self):
        """magnet-above: lower = drift_target (one-sided upper tail)."""
        _, data = self._run()
        tt = data["trade_thesis"]
        self.assertEqual(tt["regime_kind"], "magnet-above")
        self.assertAlmostEqual(tt["lower"], 4187.0, places=1)
        self.assertIsNone(tt["upper"])

    def test_200_greeks_has_all_keys(self):
        _, data = self._run()
        for k in ("delta", "gamma", "theta", "vega", "rho"):
            self.assertIn(k, data["greeks"])

    def test_200_key_levels_has_required_fields(self):
        _, data = self._run()
        kl = data["key_levels"]
        self.assertIn("max_profit",  kl)
        self.assertIn("max_loss",    kl)
        self.assertIn("breakevens",  kl)


# ── edge case: unknown ticker → 400 from _fetch_anchor_data ───────────────────

class TestUnknownTicker(unittest.TestCase):

    def setUp(self):
        self.app = _make_app()

    def test_400_unknown_ticker_no_feature_vector(self):
        """When _fetch_anchor_data raises ValueError (no data), route returns 400."""
        mock_conn = MagicMock()
        cursor = MagicMock()
        cursor.__enter__ = lambda s: cursor
        cursor.__exit__  = MagicMock(return_value=False)
        cursor.fetchone.return_value = None   # no row → ValueError

        mock_conn.cursor.return_value = cursor

        body = {**_VALID_BODY, "ticker": "UNKNOWN"}
        with patch(f"{_MODULE}._conn", return_value=mock_conn):
            status, data = _post(self.app, body)

        self.assertEqual(status, 400)
        self.assertFalse(data["ok"])
        self.assertIn("feature vector", data["error"])


if __name__ == "__main__":
    unittest.main()
