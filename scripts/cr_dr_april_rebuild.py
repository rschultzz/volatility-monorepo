"""CR-DR item 1: rebuild the 2026-04-21 trades from their stored fills, paired by fill (Status) time.
Approved by Ryan (chat 2026-10-31). UPDATE / INSERT only, no DELETE. Old rows are archived in the spec (S0.2).
Default is a dry run (rolled back); pass --commit to write."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from apps.web.modules.TradeLog import service

commit = '--commit' in sys.argv
try:
    conn = service.get_conn()
except Exception as e:
    print('connect failed:', type(e).__name__); sys.exit(1)
cur = conn.cursor()
FIELDS = ('direction', 'qty', 'entry_order_id', 'entry_ts_pt', 'entry_price', 'entry_order_type',
          'exit_order_id', 'exit_ts_pt', 'exit_price', 'exit_order_type',
          'realized_pts', 'realized_pnl_usd', 'fees_usd', 'net_pnl_usd')
try:
    cur.execute("SELECT * FROM trade_log_trades WHERE trade_date = '2026-04-21' ORDER BY id")
    old = [dict(r) for r in cur.fetchall()]
    fills = [f for f in service._load_db_fills(cur, ['MES']) if f['status_time_pt'].astimezone(service.PT).date().isoformat() == '2026-04-21']
    new, warnings = service.fifo_pair_fills(fills)
    print(f'{len(fills)} fills -> {len(new)} trades; warnings {warnings}')
    by_ids = {frozenset((o['entry_order_id'], o['exit_order_id'])): o for o in old}
    report = []
    for t in new:
        o = by_ids.get(frozenset((t['entry_order_id'], t['exit_order_id'])))
        if o is None:
            cur.execute("""
                INSERT INTO trade_log_trades
                    (trade_date, symbol, direction, qty,
                     entry_order_id, entry_ts_pt, entry_price, entry_order_type,
                     exit_order_id,  exit_ts_pt,  exit_price,  exit_order_type,
                     realized_pts, realized_pnl_usd, fees_usd, net_pnl_usd,
                     setup_start_ts_pt, setup_target_ts_pt)
                VALUES (%(trade_date)s, %(symbol)s, %(direction)s, %(qty)s,
                     %(entry_order_id)s, %(entry_ts_pt)s, %(entry_price)s, %(entry_order_type)s,
                     %(exit_order_id)s,  %(exit_ts_pt)s,  %(exit_price)s,  %(exit_order_type)s,
                     %(realized_pts)s, %(realized_pnl_usd)s, %(fees_usd)s, %(net_pnl_usd)s,
                     %(setup_start_ts_pt)s, %(setup_target_ts_pt)s)
                RETURNING id""", t)
            report.append(f"NEW id {cur.fetchone()['id']}: {t['direction']} {t['entry_price']} @{t['entry_ts_pt']:%H:%M} -> {t['exit_price']} @{t['exit_ts_pt']:%H:%M} {t['realized_pts']:+.2f}; annotations = placeholders (entry / exit time)")
            continue
        changed = {k: (o[k], t[k]) for k in FIELDS
                   if (float(o[k]) if hasattr(o[k], 'as_tuple') else o[k]) != t[k]}
        if not changed:
            report.append(f"id {o['id']}: unchanged")
            continue
        sets = dict((k, t[k]) for k in changed)
        fills_same_orientation = o['entry_order_id'] == t['entry_order_id']
        if not fills_same_orientation:
            # entry and exit swapped: the old setup times were the old placing-time placeholders
            sets['setup_start_ts_pt'] = t['entry_ts_pt']
            sets['setup_target_ts_pt'] = t['exit_ts_pt']
        cur.execute(f"UPDATE trade_log_trades SET {', '.join(f'{k} = %({k})s' for k in sets)}, updated_at = now() WHERE id = %(id)s",
                    {**sets, 'id': o['id']})
        kept = [k for k in ('setup_start_ts_pt', 'setup_target_ts_pt', 'setup_direction', 'notes', 'context_iv_atm_0dte_pct') if k not in sets]
        report.append(f"id {o['id']}: changed {', '.join(f'{k} {a} -> {b}' for k, (a, b) in changed.items())}"
                      + (f"; setup times reset to the new entry / exit" if not fills_same_orientation else '')
                      + f"; kept {', '.join(kept)} (+ other context columns)")
    # orders table: the April fills are the only April order rows we have
    cur.execute("""
        INSERT INTO trade_log_orders (order_id, symbol, side, order_type, qty, limit_price, stop_price, status,
                                      placing_time_pt, status_time_pt, duration, detected_tz, raw_csv_row)
        SELECT order_id, symbol, side, order_type, qty,
               NULLIF(raw_csv_row->>'Limit Price', '')::numeric, NULLIF(raw_csv_row->>'Stop Price', '')::numeric,
               COALESCE(NULLIF(raw_csv_row->>'Status', ''), 'Filled'),
               placing_time_pt, status_time_pt, duration, 'PT', raw_csv_row
        FROM trade_log_fills WHERE status_time_pt < '2026-05-01'
        ON CONFLICT (order_id) DO NOTHING""")
    report.append(f'trade_log_orders: {cur.rowcount} April order rows inserted from their fills')
    cur.execute("UPDATE trade_log_fills SET detected_tz = 'PT' WHERE status_time_pt < '2026-05-01' AND detected_tz IS NULL")
    report.append(f'trade_log_fills.detected_tz = PT on {cur.rowcount} April fills (Step 0 S0.1: PT 8/8)')
    print('\n'.join(report))
    if commit:
        conn.commit(); print('COMMITTED')
    else:
        conn.rollback(); print('DRY RUN: rolled back')
except Exception as e:
    conn.rollback(); print('failed:', type(e).__name__, str(e).splitlines()[0])
finally:
    conn.close()
