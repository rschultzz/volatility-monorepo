"""Unit tests for CR-DR review fields and resting-order lines (no DB).

Run with:
    python -m unittest apps.web.modules.TradeLog.tests.test_review
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.TradeLog import service as s


def pt(hms):
    h, m, sec = (int(x) for x in hms.split(':'))
    return datetime(2026, 10, 30, h, m, sec, tzinfo=s.PT)


# Trade #5 (2026-10-30 07:49 short) and its exit-side orders as in Ryan's export
TRADE = {'symbol': 'F.US.MESZ26', 'direction': 'short', 'entry_order_id': '186344856',
         'entry_ts_pt': pt('07:49:17'), 'exit_ts_pt': pt('08:08:12')}
ORDERS = [
    {'order_id': '186364540', 'symbol': 'F.US.MESZ26', 'side': 'B', 'order_type': 'Take Profit', 'status': 'Cancelled',
     'limit_price': 7642.0, 'stop_price': None, 'placing_time_pt': pt('07:51:03'), 'status_time_pt': pt('07:51:06')},
    {'order_id': '186325255', 'symbol': 'F.US.MESZ26', 'side': 'B', 'order_type': 'Limit', 'status': 'Filled',
     'limit_price': 7642.0, 'stop_price': None, 'placing_time_pt': pt('07:51:06'), 'status_time_pt': pt('08:08:12')},
    {'order_id': '186325256', 'symbol': 'F.US.MESZ26', 'side': 'B', 'order_type': 'Stop', 'status': 'Cancelled',
     'limit_price': None, 'stop_price': 7647.75, 'placing_time_pt': pt('07:51:06'), 'status_time_pt': pt('08:08:12')},
    # entry-side / cancelled-before-entry / after-exit orders are not lines
    {'order_id': '186083505', 'symbol': 'F.US.MESZ26', 'side': 'S', 'order_type': 'Limit', 'status': 'Cancelled',
     'limit_price': 7655.25, 'stop_price': None, 'placing_time_pt': pt('07:46:08'), 'status_time_pt': pt('07:46:30')},
    {'order_id': 'x-after', 'symbol': 'F.US.MESZ26', 'side': 'B', 'order_type': 'Stop', 'status': 'Cancelled',
     'limit_price': None, 'stop_price': 7660.0, 'placing_time_pt': pt('08:10:00'), 'status_time_pt': pt('08:11:00')},
    {'order_id': 'x-rej', 'symbol': 'F.US.MESZ26', 'side': 'B', 'order_type': 'Stop', 'status': 'Rejected',
     'limit_price': None, 'stop_price': 7660.0, 'placing_time_pt': pt('07:50:00'), 'status_time_pt': pt('07:50:00')},
]


class OrderLineTests(unittest.TestCase):
    def test_trade_5_lines(self):
        lines = s.order_lines_for_trade(TRADE, ORDERS)
        got = [(l['kind'], l['price'], l['from_pt'][11:], l['to_pt'][11:]) for l in lines]
        self.assertEqual(got, [('target', 7642.0, '07:51:03', '07:51:06'),
                               ('stop', 7647.75, '07:51:06', '08:08:12'),
                               ('target', 7642.0, '07:51:06', '08:08:12')])
        self.assertEqual(s._stop_at_entry(TRADE, lines), {'stop_at_entry': False, 'stop_after_s': 109})

    def test_stop_placed_before_entry_counts_at_entry(self):
        o = dict(ORDERS[2], placing_time_pt=pt('07:49:00'))
        lines = s.order_lines_for_trade(TRADE, [o])
        self.assertEqual(lines[0]['from_pt'][11:], '07:49:17')
        self.assertEqual(s._stop_at_entry(TRADE, lines), {'stop_at_entry': True, 'stop_after_s': 0})

    def test_no_stop(self):
        self.assertEqual(s._stop_at_entry(TRADE, []), {'stop_at_entry': False, 'stop_after_s': None})


class ReviewFieldTests(unittest.TestCase):
    def test_chips_canonical_order(self):
        out = s._clean_review_fields({'setup_chips': ['other', 'fade top', 'fade top'], 'mistake_chips': []})
        self.assertEqual(out, {'setup_chips': ['fade top', 'other'], 'mistake_chips': []})

    def test_unknown_chip_refused(self):
        with self.assertRaises(s.ReviewFieldError):
            s._clean_review_fields({'mistake_chips': ['revenge']})

    def test_grade_checklist_text_anchor(self):
        out = s._clean_review_fields({'grade': 4, 'checklist_followed': False, 'lesson': '', 'pre_trade_reason': 'A19 top',
                                      'anchor_ts_pt': '2026-10-30T07:40'})
        self.assertEqual(out['grade'], 4)
        self.assertIs(out['checklist_followed'], False)
        self.assertIsNone(out['lesson'])
        self.assertEqual(out['pre_trade_reason'], 'A19 top')
        self.assertEqual(out['anchor_ts_pt'], datetime(2026, 10, 30, 7, 40, tzinfo=s.PT))
        self.assertIsNone(s._clean_review_fields({'anchor_ts_pt': None})['anchor_ts_pt'])
        for bad in ({'grade': 6}, {'grade': True}, {'grade': '3'}, {'checklist_followed': 'yes'}, {'anchor_ts_pt': 'x'}):
            with self.assertRaises(s.ReviewFieldError):
                s._clean_review_fields(bad)

    def test_chip_lists(self):
        self.assertEqual(len(s.SETUP_CHIPS), 8)
        self.assertEqual(len(s.MISTAKE_CHIPS), 11)


if __name__ == '__main__':
    unittest.main()
