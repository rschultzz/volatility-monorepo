#!/usr/bin/env python3
import os
import sys
from sqlalchemy import create_engine, text

def main():
    db = os.getenv("DATABASE_URL")
    if not db:
        print("Missing DATABASE_URL")
        sys.exit(1)

    table = os.getenv("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")
    keep_days = int(os.getenv("IRONBEAM_TRADES_RETENTION_DAYS", "2"))

    engine = create_engine(db, pool_pre_ping=True)

    sql = text(f"""
        DELETE FROM {table}
        WHERE ts_utc < (NOW() AT TIME ZONE 'UTC') - (:keep_days || ' days')::interval
    """)

    with engine.begin() as conn:
        res = conn.execute(sql, {"keep_days": keep_days})
        # SQLAlchemy rowcount is supported for DELETE in psycopg
        print(f"✅ Pruned {res.rowcount} rows from {table} older than {keep_days} days")

if __name__ == "__main__":
    main()
