"""CR-DR Step 0: archive every column of the existing trade_log_trades rows (read-only) as markdown."""
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
cur.execute("select * from trade_log_trades order by id")
rows = cur.fetchall()
cols = list(rows[0].keys())
for r in rows:
    print(f"- **id {r['id']}**: " + '; '.join(f"`{c}` = {'NULL' if r[c] is None else r[c]}" for c in cols if c != 'id'))
