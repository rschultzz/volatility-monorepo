"""CR-DR smoke: hand check of trade #5 (2026-10-30 07:49:17 short 7655.75 -> 08:08:12 7642.00) facts against es_minutes (read-only)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from apps.web.modules.TradeLog import service as s
try:
    conn = s.get_conn(); conn.set_session(readonly=True); cur = conn.cursor()
    cur.execute("""select ts_pt, high, low from es_minutes where ts_pt between '2026-10-30 07:49' and '2026-10-30 08:08' order by ts_pt""")
    rows = cur.fetchall()
    hi = max(rows, key=lambda r: r['high']); lo = min(rows, key=lambda r: r['low'])
    print(f"{len(rows)} bars 07:49..08:08; max high {hi['high']} at {hi['ts_pt']:%H:%M}; min low {lo['low']} at {lo['ts_pt']:%H:%M}")
    print(f"hand: MFE = 7655.75 - {lo['low']} = {7655.75 - lo['low']:.2f}; MAE = {hi['high']} - 7655.75 = {hi['high'] - 7655.75:.2f}")
    after_entry = [r for r in rows if r['ts_pt'].strftime('%H:%M') >= '07:50']
    print(f"excluding the entry minute: max high {max(r['high'] for r in after_entry)}")
    tid = [t for t in s.chart_trades('2026-10-30', '2026-10-30') if t['entry_pt'].endswith('07:49:17')][0]['id']
    r = s.trade_review(tid)
    print({k: r[k] for k in ('id', 'direction', 'qty', 'entry_pt', 'entry_price', 'exit_pt', 'exit_price', 'realized_pts', 'multiplier',
                             'held_s', 'mfe_pts', 'mae_pts', 'stop_at_entry', 'stop_after_s')})
    print('$', r['realized_pts'] * r['qty'] * r['multiplier'], ' lines', [(l['kind'], l['price'], l['from_pt'][11:], l['to_pt'][11:]) for l in r['lines']])
except Exception as e:
    print('failed', type(e).__name__)
