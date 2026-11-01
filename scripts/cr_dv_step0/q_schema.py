from db import conn
c = conn()
for t in ("es_flow_price_1m", "ironbeam_es_trades", "ironbeam_es_flow_1s", "bt_backfill_runs"):
    print(t, c.execute("select string_agg(column_name||':'||data_type, ', ' order by ordinal_position) from information_schema.columns where table_name=%s", (t,)).fetchone()[0])
    print("  idx", c.execute("select string_agg(indexdef, ' | ') from pg_indexes where tablename=%s", (t,)).fetchone()[0])
print(c.execute("select ts_utc, symbol from ironbeam_es_trades where symbol='XCME:ES.Z26' order by ts_utc asc limit 1").fetchone())
print(c.execute("select max(ts_utc) from ironbeam_es_trades where symbol='XCME:ES.Z26'").fetchone())
print(c.execute("select distinct symbol from es_flow_price_1m where source='ironbeam'").fetchall())
