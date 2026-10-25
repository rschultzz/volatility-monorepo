"""CR-BZ step 1 — ES contract resolved from the data, env values as fallbacks (L2, A1.1).

No DB: `_dominant_symbol` (the only new DB read) and the row fetchers are patched.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.Ironbeam.tests.test_symbol_resolution
"""
from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from apps.web.modules.Ironbeam.tests import bars_fixture as bf

cb = bf.callbacks_module()
UTC = ZoneInfo("UTC")
PT = ZoneInfo("America/Los_Angeles")
Z26 = "XCME:ES.Z26"
M26 = "XCME:ES.M26"
T0 = dt.datetime(2026, 10, 23, 18, 0, tzinfo=UTC)
T1 = dt.datetime(2026, 10, 23, 18, 5, tzinfo=UTC)


def _dominant(by_table: dict):
    """A `_dominant_symbol` stand-in answering per table name."""
    return lambda table, start_utc, end_utc: by_table.get(table)


class TestResolveSymbol(unittest.TestCase):

    def test_trades_rows_win_over_env(self):
        with patch.object(cb, "_dominant_symbol", _dominant({cb.DB_TRADES_TABLE: Z26, cb.FLOW_TABLE_NAME: M26})):
            self.assertEqual(cb._resolve_symbol(T0, T1, M26), (Z26, None))

    def test_flow_used_when_trades_empty(self):
        with patch.object(cb, "_dominant_symbol", _dominant({cb.FLOW_TABLE_NAME: Z26})):
            self.assertEqual(cb._resolve_symbol(T0, T1, M26), (Z26, None))

    def test_empty_window_falls_back_to_env_with_note(self):
        with patch.object(cb, "_dominant_symbol", _dominant({})):
            sym, note = cb._resolve_symbol(T0, T1, M26)
        self.assertEqual(sym, M26)
        self.assertIn("env fallback", note)

    def test_empty_window_no_env_returns_empty_with_note(self):
        with patch.object(cb, "_dominant_symbol", _dominant({})):
            sym, note = cb._resolve_symbol(T0, T1, "")
        self.assertEqual(sym, "")
        self.assertIn("no ES rows", note)

    def test_no_contract_literal_default(self):
        src = Path(cb.__file__).read_text()
        self.assertNotIn("XCME:ES.H26", src)


class TestLiveOverlayPayload(unittest.TestCase):
    """Overlay on a live day: base bars end one minute before now; trades fill the rest."""

    def _run(self, dominant: dict, env_symbol: str):
        now_pt = dt.datetime.now(tz=PT)
        session = now_pt.date()
        last_base = pd.Timestamp(now_pt).floor("1min") - pd.Timedelta(minutes=1)
        base = pd.DataFrame({"datetime_pt": [last_base - pd.Timedelta(minutes=1), last_base],
                             "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0]})
        calls = []

        def fetch_trades(start_utc, end_utc, symbol):
            calls.append(symbol)
            if symbol != Z26:
                return pd.DataFrame(columns=["ts_utc", "symbol", "price", "size"])
            ts = pd.Timestamp(now_pt).floor("1min").tz_convert("UTC")
            return pd.DataFrame({"ts_utc": [ts + pd.Timedelta(seconds=1)], "symbol": [Z26],
                                 "price": [7000.25], "size": [3]})

        with patch.object(cb, "TRADES_SYMBOL", env_symbol), \
                patch.object(cb, "_dominant_symbol", _dominant(dominant)), \
                patch.object(cb, "_current_session_trade_date", lambda tz: session), \
                patch.object(cb, "_fetch_bars_pt", lambda *a, **k: base.copy()), \
                patch.object(cb, "_fetch_trades_utc", fetch_trades):
            payload, status = cb._build_react_live_trades_overlay_payload(session.isoformat(), "1min")
        self.assertEqual(status, 200)
        return payload, calls

    def test_z26_rows_tick_whatever_the_env(self):
        for env_symbol in ("", M26, "XCME:ES.U26"):
            with self.subTest(env=env_symbol):
                payload, calls = self._run({cb.DB_TRADES_TABLE: Z26}, env_symbol)
                self.assertEqual(calls, [Z26])
                self.assertEqual(payload["symbol"], Z26)
                self.assertIsNone(payload["symbol_note"])
                self.assertEqual(len(payload["bars"]), 1)
                self.assertEqual(payload["bars"][0]["close"], 7000.25)

    def test_empty_window_returns_empty_with_note(self):
        payload, calls = self._run({}, "")
        self.assertEqual(payload["bars"], [])
        self.assertEqual(payload["symbol"], "")
        self.assertIn("no ES rows", payload["symbol_note"])


class TestFlowPayload(unittest.TestCase):

    def test_flow_echoes_resolved_symbol(self):
        seen = []

        def fetch_flow(start_utc, end_utc, symbol):
            seen.append(symbol)
            return pd.DataFrame({"ts_utc": [start_utc], "symbol": [symbol],
                                 "buy_vol": [5], "sell_vol": [2], "unknown_vol": [0]})

        with patch.object(cb, "FLOW_SYMBOL", M26), \
                patch.object(cb, "_dominant_symbol", _dominant({cb.FLOW_TABLE_NAME: Z26})), \
                patch.object(cb, "_fetch_flow_utc", fetch_flow):
            payload, status = cb._build_react_flow_payload("2026-10-23", session="RTH", resample="1m")
        self.assertEqual(status, 200)
        self.assertEqual(seen, [Z26])
        self.assertEqual(payload["symbol"], Z26)
        self.assertIsNone(payload["symbol_note"])
        self.assertGreater(payload["point_count"], 0)


if __name__ == "__main__":
    unittest.main()
