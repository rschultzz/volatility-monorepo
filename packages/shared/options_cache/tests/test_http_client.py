"""
Unit tests for the HTTP client using mocked responses. No network.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_http_client
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from packages.shared.options_cache import http_client
from packages.shared.options_cache.http_client import (
    OratsPermanentError,
    OratsTransientError,
    get_csv,
    reset_session,
)


class TestHttpClient(unittest.TestCase):
    def setUp(self):
        # Reset the cached session between tests for isolation
        reset_session()
        # Make sure tests don't actually sleep during retry tests
        self._patcher_sleep = patch("packages.shared.options_cache.http_client.time.sleep")
        self._patcher_sleep.start()
        # Set fake API key
        self._old_key = os.environ.get("ORATS_API_KEY")
        os.environ["ORATS_API_KEY"] = "test_token_xyz"

    def tearDown(self):
        self._patcher_sleep.stop()
        reset_session()
        if self._old_key is None:
            os.environ.pop("ORATS_API_KEY", None)
        else:
            os.environ["ORATS_API_KEY"] = self._old_key

    @patch.object(http_client, "_get_session")
    def test_200_returns_body(self, mock_session):
        resp = MagicMock(status_code=200, text="ticker,bidPrice\nSPXW,12.5")
        mock_session.return_value.get.return_value = resp

        body = get_csv("/some/path", {"foo": "bar"})

        self.assertEqual(body, "ticker,bidPrice\nSPXW,12.5")
        # Token should have been added to params
        called_params = mock_session.return_value.get.call_args.kwargs["params"]
        self.assertEqual(called_params["token"], "test_token_xyz")
        self.assertEqual(called_params["foo"], "bar")

    @patch.object(http_client, "_get_session")
    def test_200_with_html_body_raises_permanent(self, mock_session):
        # ORATS sometimes returns 200 with HTML on auth error
        resp = MagicMock(status_code=200, text="<html>error</html>")
        mock_session.return_value.get.return_value = resp

        with self.assertRaises(OratsPermanentError):
            get_csv("/some/path", {})

    @patch.object(http_client, "_get_session")
    def test_401_raises_permanent_no_retry(self, mock_session):
        resp = MagicMock(status_code=401, text="Unauthorized")
        mock_session.return_value.get.return_value = resp

        with self.assertRaises(OratsPermanentError):
            get_csv("/some/path", {}, max_retries=3)

        # Should have been called exactly once (no retries on 4xx)
        self.assertEqual(mock_session.return_value.get.call_count, 1)

    @patch.object(http_client, "_get_session")
    def test_429_retries_then_succeeds(self, mock_session):
        resp_429 = MagicMock(status_code=429, text="rate limited")
        resp_200 = MagicMock(status_code=200, text="ticker\nSPXW")

        mock_session.return_value.get.side_effect = [
            resp_429, resp_429, resp_200,
        ]

        body = get_csv("/some/path", {}, max_retries=3)
        self.assertEqual(body, "ticker\nSPXW")
        self.assertEqual(mock_session.return_value.get.call_count, 3)

    @patch.object(http_client, "_get_session")
    def test_500_retries_then_succeeds(self, mock_session):
        resp_500 = MagicMock(status_code=500, text="server error")
        resp_200 = MagicMock(status_code=200, text="ticker\nSPXW")

        mock_session.return_value.get.side_effect = [resp_500, resp_200]

        body = get_csv("/some/path", {}, max_retries=2)
        self.assertEqual(body, "ticker\nSPXW")

    @patch.object(http_client, "_get_session")
    def test_persistent_500_eventually_raises_transient(self, mock_session):
        resp_500 = MagicMock(status_code=500, text="server error")
        mock_session.return_value.get.return_value = resp_500

        with self.assertRaises(OratsTransientError):
            get_csv("/some/path", {}, max_retries=2)

        # 1 initial + 2 retries = 3 calls
        self.assertEqual(mock_session.return_value.get.call_count, 3)

    @patch.object(http_client, "_get_session")
    def test_timeout_treated_as_transient(self, mock_session):
        mock_session.return_value.get.side_effect = requests.Timeout("slow")

        with self.assertRaises(OratsTransientError):
            get_csv("/some/path", {}, max_retries=1)

        # 1 initial + 1 retry = 2 calls
        self.assertEqual(mock_session.return_value.get.call_count, 2)

    @patch.object(http_client, "_get_session")
    def test_connection_error_treated_as_transient(self, mock_session):
        mock_session.return_value.get.side_effect = requests.ConnectionError(
            "refused"
        )
        with self.assertRaises(OratsTransientError):
            get_csv("/some/path", {}, max_retries=0)

    @patch.object(http_client, "_get_session")
    def test_caller_params_not_mutated(self, mock_session):
        # Adding token should not appear in caller's dict
        resp = MagicMock(status_code=200, text="ticker\nSPXW")
        mock_session.return_value.get.return_value = resp

        params = {"foo": "bar"}
        get_csv("/some/path", params)
        self.assertNotIn("token", params)

    def test_missing_api_key_raises_permanent(self):
        # Clear the env var, and patch the repo-level dotenv loader so it
        # can't accidentally repopulate it from a .env on disk.
        os.environ.pop("ORATS_API_KEY", None)
        with patch(
            "packages.shared.options_cache.repository._load_env_file_if_present"
        ):
            with self.assertRaises(OratsPermanentError):
                get_csv("/some/path", {})


if __name__ == "__main__":
    unittest.main()
