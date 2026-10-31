"""Unit tests for the CR-DR trade-log importer (no DB).

Run with:
    python -m unittest apps.web.modules.TradeLog.tests.test_import
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.TradeLog import service as s

# 2026-10 layout (synthetic rows shaped like the AMP / TradingView export)
NEW_CSV = b"""Symbol,Side,Type,Quantity,Limit price,Stop price,Active at,Fill quantity,Avg fill price,Commission,Placing time,Status,Status time,Order ID,Duration
F.US.MESZ26,Buy,Stop,1,,7647.75,,0,,,2026-10-30 07:51:06,Cancelled,2026-10-30 08:08:12,3,GTC
F.US.MESZ26,Buy,Limit,1,7642,,,1,7642,,2026-10-30 07:51:06,Filled,2026-10-30 08:08:12,2,GTC
F.US.MESZ26,Sell,Limit,1,7655.75,,,1,7655.75,,2026-10-30 07:49:16,Filled,2026-10-30 07:49:17,1,DAY
F.US.MESZ26,Sell,Limit,1,7774.25,,,0,,,2026-10-28 09:01:35,Rejected,2026-10-28 09:01:35,4,DAY
"""

# April 2026 layout: 'Qty', 'Fill Qty', 'Avg Fill Price', minute-only times
OLD_CSV = b"""Symbol,Side,Type,Qty,Limit Price,Stop Price,Fill Qty,Avg Fill Price,Commission,Placing Time,Status,Status Time,Order ID,Duration,Active At
F.US.MESM26,Buy,Limit,1,7126,,1,7126,,4/21/26 12:10,Filled,4/21/26 12:10,10,DAY,
F.US.MESM26,Sell,Stop,1,,7124,1,7123.75,,4/21/26 12:10,Filled,4/21/26 12:14,11,GTC,
"""


class FakeCursor:
    """es_minutes lookups only: bars keyed by UTC minute."""
    def __init__(self, bars):
        self.bars = bars
        self._rows = []

    def execute(self, sql, params):
        self._rows = [{'ts_utc': m, 'low': self.bars[m][0], 'high': self.bars[m][1]}
                      for m in params[0] if m in self.bars]

    def fetchall(self):
        return self._rows


def utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class ParseTests(unittest.TestCase):
    def test_new_layout_all_statuses_kept_fills_flagged(self):
        orders, warnings = s.parse_tv_csv(NEW_CSV)
        self.assertEqual(len(orders), 4)
        self.assertEqual([o['status'] for o in orders], ['Cancelled', 'Filled', 'Filled', 'Rejected'])
        self.assertEqual([o['is_fill'] for o in orders], [False, True, True, False])
        stop = orders[0]
        self.assertEqual((stop['stop_price'], stop['limit_price'], stop['fill_price']), (7647.75, None, None))
        self.assertEqual(orders[2]['status_naive'], datetime(2026, 10, 30, 7, 49, 17))
        self.assertEqual(warnings, [])

    def test_old_layout_parses(self):
        orders, _ = s.parse_tv_csv(OLD_CSV)
        self.assertEqual([o['fill_qty'] for o in orders], [1, 1])
        self.assertEqual(orders[1]['status_naive'], datetime(2026, 4, 21, 12, 14))
        self.assertEqual(orders[1]['stop_price'], 7124.0)


class TimezoneTests(unittest.TestCase):
    def setUp(self):
        orders, _ = s.parse_tv_csv(NEW_CSV)
        self.fills = [o for o in orders if o['is_fill']]
        # PT 07:49 and 08:08 bars (PDT = UTC-7)
        self.pt_bars = {utc(2026, 10, 30, 14, 49): (7654.0, 7657.0),
                        utc(2026, 10, 30, 15, 8): (7641.75, 7648.0)}

    def test_unique_zone(self):
        r = s.detect_timezone(FakeCursor(self.pt_bars), self.fills)
        self.assertEqual(r['zone'], 'PT')
        self.assertEqual(r['matches'], {'PT': 2, 'CT': 0, 'ET': 0, 'UTC': 0})

    def test_one_tick_tolerance(self):
        bars = dict(self.pt_bars)
        bars[utc(2026, 10, 30, 15, 8)] = (7642.25, 7648.0)   # fill 7642.00 is one tick below the low
        self.assertEqual(s.detect_timezone(FakeCursor(bars), self.fills)['zone'], 'PT')
        bars[utc(2026, 10, 30, 15, 8)] = (7642.50, 7648.0)   # two ticks: refused
        r = s.detect_timezone(FakeCursor(bars), self.fills)
        self.assertIsNone(r['zone'])
        self.assertIn('Time zone not detected', r['error'])
        self.assertIn('Nothing was imported', r['error'])

    def test_ambiguous_refused(self):
        bars = dict(self.pt_bars)
        bars[utc(2026, 10, 30, 11, 49)] = (7654.0, 7657.0)   # ET 07:49 (EDT)
        bars[utc(2026, 10, 30, 12, 8)] = (7641.0, 7643.0)    # ET 08:08 (EDT)
        r = s.detect_timezone(FakeCursor(bars), self.fills)
        self.assertIsNone(r['zone'])
        self.assertIn('ambiguous', r['error'])

    def test_no_bars_refused(self):
        self.assertIsNone(s.detect_timezone(FakeCursor({}), self.fills)['zone'])


class PairingTests(unittest.TestCase):
    def _fills(self, csv_bytes, zone):
        orders, _ = s.parse_tv_csv(csv_bytes)
        return [s._localize(o, zone) for o in orders if o['is_fill']]

    def test_pairs_by_fill_time_not_placing_time(self):
        # April id 7: the stop was *placed* at 12:10 but *filled* 12:14 — the trade is a long.
        trades, warnings = s.fifo_pair_fills(self._fills(OLD_CSV, 'PT'))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual((t['direction'], t['entry_order_id'], t['exit_order_id']), ('long', '10', '11'))
        self.assertAlmostEqual(t['realized_pts'], -2.25)
        self.assertEqual(t['exit_ts_pt'].strftime('%H:%M'), '12:14')
        self.assertEqual(warnings, [])

    def test_short_with_bracket_and_pt_date(self):
        trades, _ = s.fifo_pair_fills(self._fills(NEW_CSV, 'PT'))
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t['direction'], 'short')
        self.assertAlmostEqual(t['realized_pts'], 13.75)
        self.assertAlmostEqual(t['realized_pnl_usd'], 68.75)   # MES x5
        self.assertEqual(t['entry_ts_pt'].strftime('%H:%M:%S'), '07:49:17')
        self.assertEqual(t['trade_date'], '2026-10-30')

    def test_partial_fills_fifo(self):
        csv_bytes = b"""Symbol,Side,Type,Quantity,Fill quantity,Avg fill price,Placing time,Status,Status time,Order ID
F.US.MESZ26,Buy,Limit,2,2,7700,2026-10-30 07:00:00,Filled,2026-10-30 07:00:05,a
F.US.MESZ26,Sell,Limit,2,1,7702,2026-10-30 07:01:00,Cancelled,2026-10-30 07:02:00,b
F.US.MESZ26,Sell,Market,1,1,7699,2026-10-30 07:03:00,Filled,2026-10-30 07:03:00,c
"""
        trades, warnings = s.fifo_pair_fills(self._fills(csv_bytes, 'PT'))
        self.assertEqual([(t['exit_order_id'], t['qty'], t['realized_pts']) for t in trades],
                         [('b', 1, 2.0), ('c', 1, -1.0)])
        self.assertEqual(warnings, [])


class HoldoutTests(unittest.TestCase):
    def test_window(self):
        self.assertFalse(s.is_holdout_date('2026-06-05'))
        self.assertTrue(s.is_holdout_date('2026-06-06'))
        self.assertTrue(s.is_holdout_date('2026-10-26'))
        self.assertFalse(s.is_holdout_date('2026-10-27'))
        self.assertFalse(s.is_holdout_date('2026-04-21'))


if __name__ == '__main__':
    unittest.main()
