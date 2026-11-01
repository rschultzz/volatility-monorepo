"""CR-DR Step 0: read-only look at trade_log_* schema and the April trades."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import psycopg2, psycopg2.extras
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'].replace('postgresql+psycopg2','postgresql').replace('postgresql+psycopg','postgresql'),
                            cursor_factory=psycopg2.extras.RealDictCursor)
except Exception as e:
    print('connect failed:', type(e).__name__); sys.exit(1)
conn.set_session(readonly=True)
cur = conn.cursor()
for t in ('trade_log_fills', 'trade_log_trades'):
    cur.execute("""select column_name, data_type, is_nullable from information_schema.columns
                   where table_name=%s order by ordinal_position""", (t,))
    print(f'== {t}')
    for r in cur.fetchall(): print('  ', r['column_name'], r['data_type'], r['is_nullable'])
    cur.execute("""select indexname, indexdef from pg_indexes where tablename=%s""", (t,))
    for r in cur.fetchall(): print('   idx', r['indexdef'])
    cur.execute(f"select count(*) n from {t}"); print('   rows', cur.fetchone()['n'])
cur.execute("select tableowner from pg_tables where tablename='trade_log_trades'"); print('owner', cur.fetchone())
cur.execute("select current_user"); print('me', cur.fetchone())
cur.execute("""select id, trade_date, symbol, direction, qty, entry_ts_pt, entry_price, exit_ts_pt, exit_price,
               realized_pts, entry_order_type, exit_order_type from trade_log_trades order by id""")
for r in cur.fetchall(): print(dict(r))
cur.execute("""select order_id, symbol, side, order_type, fill_qty, fill_price, placing_time_pt, status_time_pt, raw_csv_row
               from trade_log_fills order by placing_time_pt""")
for r in cur.fetchall(): print(dict(r))
