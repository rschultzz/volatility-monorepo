"""CR-DC — es_flow_price_1m (CR-DD) shape: schema, sources, span, per-source minute coverage. READ-ONLY.
Mechanical only (counts and spans; no value statistics)."""
import sys
sys.path.insert(0, sys.argv[1])
from dotenv import load_dotenv  # noqa: E402
load_dotenv(sys.argv[1] + "/.env")
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402
conn = get_backfill_db_conn(); assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
print(conn.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='es_flow_price_1m' ORDER BY ordinal_position").fetchall())
print(conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename='es_flow_price_1m'").fetchall())
for r in conn.execute("""SELECT source, min(minute_utc), max(minute_utc), count(*), count(DISTINCT symbol),
                                count(DISTINCT (minute_utc AT TIME ZONE 'America/Los_Angeles')::date)
                         FROM es_flow_price_1m GROUP BY 1""").fetchall():
    print(r)
print(conn.execute("SELECT count(*) FROM es_flow_price_1m WHERE price * 4 <> round(price * 4)").fetchone(), "off-grid rows")
