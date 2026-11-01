"""CR-DR Step 0: the 2026-10-30 06:22:56 PT stop fill at 7668.00 vs the bars and per-second prints (read-only)."""
import os, sys
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
cur.execute("""select ts_pt, open, high, low, close, volume, source from es_minutes
               where ts_pt between '2026-10-30 06:20' and '2026-10-30 06:25' order by ts_pt""")
for r in cur.fetchall(): print('es_minutes', dict(r))
cur.execute("""select datetime, open, high, low, close, volume, source from ironbeam_es_1m_bars
               where datetime between '2026-10-30 13:20' and '2026-10-30 13:25' order by datetime""")
for r in cur.fetchall(): print('ironbeam_1m', dict(r))
cur.execute("""select column_name from information_schema.columns where table_name='ironbeam_es_flow_1s' order by ordinal_position""")
print('flow_1s cols', [r['column_name'] for r in cur.fetchall()])
cur.execute("""select ts_utc, open, high, low, close, volume, symbol from ironbeam_es_flow_1s
               where ts_utc between '2026-10-30 13:22:45+00' and '2026-10-30 13:23:05+00' order by ts_utc""")
for r in cur.fetchall(): print('flow_1s', dict(r))
