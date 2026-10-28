"""CR-DC — index / existence checks for the cells endpoint. READ-ONLY."""
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv  # noqa: E402
load_dotenv(sys.argv[1] + "/.env")
import time  # noqa: E402
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402
conn = get_backfill_db_conn(); assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
for t in ("ironbeam_es_trades", "ironbeam_es_flow_1s", "es_flow_1s_databento"):
    print(t, conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename=%s", (t,)).fetchall())
print("es_flow_price_1m exists:", conn.execute("SELECT to_regclass('es_flow_price_1m') IS NOT NULL").fetchone())
t0 = time.time(); print(conn.execute("SELECT min(ts_utc) FROM ironbeam_es_trades").fetchone(), round(time.time() - t0, 3), "s")
t0 = time.time()
n = conn.execute("""SELECT count(*) FROM (SELECT floor(extract(epoch FROM ts_utc)/60) m, price, aggressor_side, sum(size)
  FROM ironbeam_es_trades WHERE ts_utc >= '2026-10-27 13:30+00' AND ts_utc < '2026-10-27 20:00+00' AND symbol='XCME:ES.Z26'
  GROUP BY 1,2,3) x""").fetchone()
print("rth cells(side) rows", n, round(time.time() - t0, 3), "s")
