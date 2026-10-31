"""CR-DR: apply scripts/cr_dr_migration.sql as the table owner, then print the resulting columns."""
import os, sys
from dotenv import load_dotenv
load_dotenv()
import psycopg2
here = os.path.dirname(os.path.abspath(__file__))
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'].replace('postgresql+psycopg2','postgresql').replace('postgresql+psycopg','postgresql'))
except Exception as e:
    print('connect failed:', type(e).__name__); sys.exit(1)
conn.autocommit = True  # the file carries its own BEGIN / COMMIT
cur = conn.cursor()
try:
    cur.execute(open(os.path.join(here, 'cr_dr_migration.sql')).read())
except Exception as e:
    print('migration failed:', type(e).__name__, str(e).splitlines()[0]); sys.exit(1)
for t in ('trade_log_orders', 'trade_log_fills', 'trade_log_trades'):
    cur.execute("select column_name, data_type, is_nullable from information_schema.columns where table_name=%s order by ordinal_position", (t,))
    print(t, [f'{c}:{d}:{n}' for c, d, n in cur.fetchall()])
cur.execute("select conname from pg_constraint where conname='trade_log_trades_grade_1_5'"); print('grade check', cur.fetchone())
cur.execute("select count(*) from trade_log_trades"); print('trades', cur.fetchone()[0])
cur.execute("select count(*) from trade_log_fills"); print('fills', cur.fetchone()[0])
