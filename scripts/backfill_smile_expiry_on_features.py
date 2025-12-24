#!/usr/bin/env python
import os
from sqlalchemy import create_engine, text

from pathlib import Path as _P
import sys

REPO_ROOT = _P(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment or .env")

    engine = create_engine(db_url)

    sql = text(
        """
        -- 1) Make sure the column exists
        ALTER TABLE es_minute_features
        ADD COLUMN IF NOT EXISTS smile_expir_primary date;

        -- 2) For each minute, pick the expiry with the smallest DTE
        WITH primary_exp AS (
            SELECT DISTINCT ON (snap_min)
                   snap_min,
                   expir_date::date AS expir_date,
                   dte
            FROM (
                SELECT
                    date_trunc('minute', snapshot_pt) AS snap_min,
                    expir_date,
                    dte
                FROM orats_monies_minute
                WHERE ticker IN ('SPX', 'SPXW')
            ) s
            ORDER BY snap_min, dte ASC
        )
        UPDATE es_minute_features e
        SET smile_expir_primary = p.expir_date
        FROM primary_exp p
        WHERE date_trunc('minute', e.ts_utc) = p.snap_min
          AND e.smile_expir_primary IS NULL;
        """
    )

    with engine.begin() as conn:
        conn.execute(sql)

    print("✅ Backfill complete: smile_expir_primary set from orats_monies_minute.")

if __name__ == "__main__":
    main()
