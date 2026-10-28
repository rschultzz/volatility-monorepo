"""CR-DC Step 0 — schemas and retention. READ-ONLY (default_transaction_read_only asserted)."""
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv  # noqa: E402
load_dotenv(sys.argv[1] + "/.env")
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402

conn = get_backfill_db_conn()
assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
conn.execute("SET statement_timeout = '600s'")
assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
for t in ("ironbeam_es_trades", "ironbeam_es_flow_1s", "es_flow_1s_databento"):
    cols = conn.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (t,)).fetchall()
    print(t, cols)
print(conn.execute("SELECT now() AT TIME ZONE 'UTC', min(ts_utc), max(ts_utc), count(*) FROM ironbeam_es_trades").fetchone())
for r in conn.execute("""SELECT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date d, symbol, count(*), sum(size)
                         FROM ironbeam_es_trades GROUP BY 1,2 ORDER BY 1,2""").fetchall():
    print(r)
