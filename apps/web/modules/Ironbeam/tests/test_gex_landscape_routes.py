"""Tests for apps/web/modules/Ironbeam/routes.py — /api/gex-landscape
spot-resolution behaviour (CR-016 hotfix 5).

The route is registered via register_gex_landscape_route(server, engine)
where engine is a SQLAlchemy Engine.  Tests pass a mock engine whose
connect() context manager returns a mock SA Connection, bypassing any live DB.

build_gex_landscape_response is patched at the *request* level (not at
registration time) so the mock is active when the route handler executes.

Run with:
    python -m unittest apps.web.modules.Ironbeam.tests.test_gex_landscape_routes
"""
from __future__ import annotations

import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask
from apps.web.modules.Ironbeam.routes import register_gex_landscape_route

_BUILDER_PATH = "apps.web.modules.Ironbeam.routes.build_gex_landscape_response"
_OK_PAYLOAD = {"ok": True, "regime": {"regime": "magnetic-pin"}}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_sa_conn(open_price=None):
    """Mock SQLAlchemy Connection.

    open_price: float | None — returned by the RTH open row query.
    None means fetchone() returns None (no bars for that date).
    """
    row = (open_price,) if open_price is not None else None
    execute_result = MagicMock()
    execute_result.fetchone.return_value = row
    conn = MagicMock()
    conn.execute.return_value = execute_result
    return conn


def _make_engine(sa_conn):
    """Mock SQLAlchemy Engine whose connect() context manager yields sa_conn."""
    @contextmanager
    def _connect():
        yield sa_conn

    engine = MagicMock()
    engine.connect = _connect
    return engine


def _make_app(sa_conn):
    """Create a fresh Flask test app with the gex-landscape route registered.

    The build_gex_landscape_response import is NOT patched here — patches must
    wrap the test_client request so the mock is active when the handler runs.
    """
    server = Flask(__name__)
    # Each call gets a fresh server object so the guard flag is always clear.
    register_gex_landscape_route(server, _make_engine(sa_conn))
    return server


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGexLandscapeSpotResolution(unittest.TestCase):
    """Spot-resolution paths for GET /api/gex-landscape (CR-016 hotfix 5)."""

    def test_200_when_spot_omitted_and_bars_exist(self):
        """When spot= is absent and RTH bars exist, the route resolves the spot
        from the first RTH bar and delegates to build_gex_landscape_response."""
        sa_conn = _make_sa_conn(open_price=7362.825)
        app = _make_app(sa_conn)

        builder = MagicMock(return_value=(_OK_PAYLOAD, 200))
        with patch(_BUILDER_PATH, builder):
            with app.test_client() as client:
                resp = client.get("/api/gex-landscape?date=2026-05-07&ticker=SPX")

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(builder.called, "builder should have been called")
        # positional args: (conn, ticker, date_str, spot, ...)
        spot_arg = builder.call_args[0][3]
        self.assertAlmostEqual(float(spot_arg), 7362.825, places=2)

    def test_400_when_spot_omitted_and_no_bars(self):
        """When spot= is absent and no RTH bars exist, the route returns 400
        and never calls build_gex_landscape_response."""
        sa_conn = _make_sa_conn(open_price=None)
        app = _make_app(sa_conn)

        builder = MagicMock(return_value=(_OK_PAYLOAD, 200))
        with patch(_BUILDER_PATH, builder):
            with app.test_client() as client:
                resp = client.get("/api/gex-landscape?date=2026-01-02&ticker=SPX")

        self.assertEqual(resp.status_code, 400)
        body = json.loads(resp.data)
        self.assertIn("error", body)
        self.assertIn("2026-01-02", body["error"])
        builder.assert_not_called()

    def test_200_when_spot_provided_explicitly(self):
        """When spot= is present the RTH query is skipped; the provided value
        is passed straight through to the builder."""
        # open_price=None so that if execute() is accidentally called it returns
        # None (bars absent), which would trigger a 400 — a fast failure signal.
        sa_conn = _make_sa_conn(open_price=None)
        app = _make_app(sa_conn)

        builder = MagicMock(return_value=(_OK_PAYLOAD, 200))
        with patch(_BUILDER_PATH, builder):
            with app.test_client() as client:
                resp = client.get(
                    "/api/gex-landscape?date=2026-05-07&ticker=SPX&spot=7400"
                )

        self.assertEqual(resp.status_code, 200)
        spot_arg = builder.call_args[0][3]
        self.assertEqual(spot_arg, "7400")
        # conn.execute must NOT have been called (no spot lookup needed)
        sa_conn.execute.assert_not_called()

    def test_resolved_spot_is_a_string(self):
        """The resolved RTH open is converted to str before being passed to
        the builder — matches the existing contract for an explicit spot= param."""
        sa_conn = _make_sa_conn(open_price=7362.825)
        app = _make_app(sa_conn)

        builder = MagicMock(return_value=(_OK_PAYLOAD, 200))
        with patch(_BUILDER_PATH, builder):
            with app.test_client() as client:
                client.get("/api/gex-landscape?date=2026-05-07&ticker=SPX")

        spot_arg = builder.call_args[0][3]
        self.assertIsInstance(spot_arg, str)

    def test_400_when_date_missing(self):
        app = _make_app(_make_sa_conn())
        with app.test_client() as client:
            resp = client.get("/api/gex-landscape?ticker=SPX")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("required", json.loads(resp.data)["error"])

    def test_400_when_ticker_missing(self):
        app = _make_app(_make_sa_conn())
        with app.test_client() as client:
            resp = client.get("/api/gex-landscape?date=2026-05-07")
        self.assertEqual(resp.status_code, 400)

    def test_400_when_date_malformed(self):
        app = _make_app(_make_sa_conn())
        with app.test_client() as client:
            resp = client.get("/api/gex-landscape?date=not-a-date&ticker=SPX")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("YYYY-MM-DD", json.loads(resp.data)["error"])

    def test_double_registration_is_idempotent(self):
        """Calling register_gex_landscape_route twice on the same server
        must not raise (e.g. during Dash hot-reload or repeated setUp)."""
        sa_conn = _make_sa_conn(open_price=7362.0)
        engine = _make_engine(sa_conn)
        server = Flask(__name__ + "_dup")
        register_gex_landscape_route(server, engine)
        register_gex_landscape_route(server, engine)  # must not raise


if __name__ == "__main__":
    unittest.main()
