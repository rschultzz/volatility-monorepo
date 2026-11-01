"""CR-DR Step 0: Ryan's TradingView order-history export, read in place (path argument; never copied).
Per filled row: fill (Status) time read in PT / CT / ET / UTC vs the es_minutes bar high-low; then FIFO by fill time."""
import csv, os, sys
from collections import Counter, deque
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()
import psycopg2, psycopg2.extras

path = sys.argv[1]
rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
print('columns:', list(rows[0].keys()))
print('rows:', len(rows), 'status:', dict(Counter(r['Status'] for r in rows)), 'types:', dict(Counter(r['Type'] for r in rows)))
print('symbols:', dict(Counter(r['Symbol'] for r in rows)), 'durations:', dict(Counter(r['Duration'] for r in rows)))
print('Active at non-empty:', sum(bool(r['Active at']) for r in rows), ' Commission non-empty:', sum(bool(r['Commission']) for r in rows))
print('partial fills (0 < fill < qty):', sum(0 < int(r['Fill quantity'] or 0) < int(r['Quantity']) for r in rows))
print('dates:', dict(Counter(r['Status time'][:10] for r in rows)))

try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'].replace('postgresql+psycopg2','postgresql').replace('postgresql+psycopg','postgresql'),
                            cursor_factory=psycopg2.extras.RealDictCursor)
except Exception as e:
    print('connect failed:', type(e).__name__); sys.exit(1)
conn.set_session(readonly=True)
cur = conn.cursor()
ZONES = {'PT': 'America/Los_Angeles', 'CT': 'America/Chicago', 'ET': 'America/New_York', 'UTC': 'UTC'}
fills = [r for r in rows if r['Status'] == 'Filled']
tally = Counter()
print('\nfilled rows, fill time per zone vs es_minutes [low-high]:')
for r in sorted(fills, key=lambda r: r['Status time']):
    naive = datetime.strptime(r['Status time'], '%Y-%m-%d %H:%M:%S').replace(second=0)
    px = float(r['Avg fill price'])
    cells = []
    for z, tz in ZONES.items():
        utc = naive.replace(tzinfo=ZoneInfo(tz)).astimezone(ZoneInfo('UTC'))
        cur.execute("select low, high, source from es_minutes where ts_utc = %s", (utc,))
        b = cur.fetchone()
        ok = b is not None and b['low'] <= px <= b['high']
        tally[z] += ok
        cells.append(f"{z}:{'OK' if ok else 'x'}[{b['low'] if b else '-'}-{b['high'] if b else '-'}]")
    print(f"  {r['Status time']} {r['Side']:<4} {r['Type']:<11} {r['Fill quantity']} @ {px:<8}", ' '.join(cells))
print('all-fill matches per zone:', dict(tally), 'of', len(fills))

print('\nFIFO by fill (Status) time:')
q, day_pts = deque(), Counter()
for r in sorted(fills, key=lambda r: r['Status time']):
    side, n, px = r['Side'], int(r['Fill quantity']), float(r['Avg fill price'])
    while n and q and q[0][0] != side:
        e = q[0]; m = min(n, e[2])
        pts = (px - e[3]) if e[0] == 'Buy' else (e[3] - px)
        print(f"  {'long ' if e[0]=='Buy' else 'short'} x{m} {e[1]} @{e[3]} -> {r['Status time']} @{px} ({r['Type']})  {pts:+.2f} pts")
        day_pts[r['Status time'][:10]] += pts * m
        e[2] -= m; n -= m
        if not e[2]: q.popleft()
    if n: q.append([side, r['Status time'], n, px])
print('open at end:', list(q), ' pts x qty by day:', dict(day_pts))
