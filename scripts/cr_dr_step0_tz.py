"""CR-DR Step 0: which zone makes the stored April fills' raw times match ES 1-min bars (read-only)."""
import os, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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
cur.execute("""select column_name, data_type from information_schema.columns where table_name='ironbeam_es_1m_bars' order by ordinal_position""")
print('ironbeam_es_1m_bars:', [(r['column_name'], r['data_type']) for r in cur.fetchall()])
cur.execute("""select column_name, data_type from information_schema.columns where table_name='es_minutes' order by ordinal_position""")
print('es_minutes:', [(r['column_name'], r['data_type']) for r in cur.fetchall()])

ZONES = {'PT': 'America/Los_Angeles', 'CT': 'America/Chicago', 'ET': 'America/New_York', 'UTC': 'UTC'}
FMTS = ['%m/%d/%y %H:%M', '%m/%d/%y %H:%M:%S', '%Y-%m-%d %H:%M:%S']
def parse(s):
    for f in FMTS:
        try: return datetime.strptime(s.strip(), f)
        except ValueError: pass
    return None

cur.execute("select order_id, side, order_type, fill_price, raw_csv_row from trade_log_fills order by placing_time_pt")
fills = cur.fetchall()
print('\nApril fills: raw Status Time (= fill time) vs es_minutes bar range, per candidate zone')
tally = {z: 0 for z in ZONES}
for f in fills:
    naive = parse(f['raw_csv_row']['Status Time'])
    px = float(f['fill_price'])
    cells = []
    for z, tz in ZONES.items():
        utc = naive.replace(tzinfo=ZoneInfo(tz)).astimezone(ZoneInfo('UTC'))
        cur.execute("select low, high from es_minutes where ts_utc = %s", (utc,))
        b = cur.fetchone()
        ok = b is not None and b['low'] <= px <= b['high']
        tally[z] += ok
        cells.append(f"{z}:{'OK' if ok else 'x'}({b['low'] if b else '-'}-{b['high'] if b else '-'})")
    print(f"  {f['order_id']} {f['side']} {f['order_type']:<9} {px:>9} raw {f['raw_csv_row']['Status Time']:<14}", ' '.join(cells))
print('matches per zone:', tally, 'of', len(fills))
