#!/usr/bin/env python3
"""CR-AQ Step 1 — apply infra/sql/bt_daily_outcomes_session_containment.sql under the owner role.

schema_change class: must use DATABASE_URL (table owner), NOT BACKFILL_DATABASE_URL.
Prints the statement file, the resulting column list, the grants and the view column
count, and the pre-backfill non-null count (expect 0).

Usage:
    python -u scripts/cr_aq_run_migration.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import psycopg
from packages.shared.backfill_safety import _normalize_url

NEW = ("session_high_t0", "session_low_t0", "session_close_t0", "wall_above_price", "wall_below_price",
       "contained_close", "contained_range", "close_pos_in_band", "range_over_im", "close_move_over_im", "breach_side")
SQL_PATH = REPO_ROOT / "infra" / "sql" / "bt_daily_outcomes_session_containment.sql"


def main() -> None:
    raw = os.environ.get("DATABASE_URL", "").strip()
    if not raw:
        sys.exit("ERROR: DATABASE_URL is not set")
    conn = psycopg.connect(_normalize_url(raw))
    conn.autocommit = True   # the SQL file carries its own BEGIN/COMMIT; the view refresh runs after
    with conn.cursor() as cur:
        cur.execute("SELECT current_user"); role = cur.fetchone()[0]
        cur.execute("SELECT tableowner FROM pg_tables WHERE tablename='bt_daily_outcomes'"); owner = cur.fetchone()[0]
    print(f"Connected as {role}; bt_daily_outcomes owner = {owner}")
    if role == "dash_backfill_writer" or role != owner:
        sys.exit("ERROR: must run as the table owner via DATABASE_URL")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name='bt_daily_outcomes' AND column_name = ANY(%s)", (list(NEW),))
        pre = cur.fetchone()[0]
    if pre:
        sys.exit(f"ERROR: {pre} of the new columns already exist — migration already applied?")

    sql = SQL_PATH.read_text()
    print(f"\nApplying {SQL_PATH.relative_to(REPO_ROOT)} ({len(sql)} bytes)")
    print("--- statement ---"); print("\n".join(l for l in sql.splitlines() if l.strip() and not l.startswith("--"))); print("--- end ---")
    with conn.cursor() as cur:
        cur.execute(sql)
    print("applied ✓")

    with conn.cursor() as cur:
        cur.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name='bt_daily_outcomes' ORDER BY ordinal_position")
        cols = cur.fetchall()
    print(f"\n\\d bt_daily_outcomes — {len(cols)} columns:")
    for c in cols:
        print(f"  {c[0]:<36} {c[1]:<28} nullable={c[2]}")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.column_privileges WHERE table_name='bt_daily_outcomes' AND grantee='dash_backfill_writer' AND privilege_type='UPDATE' AND column_name = ANY(%s)", (list(NEW),))
        grants = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM information_schema.columns WHERE table_name='bt_daily_outcomes_active'")
        view_cols = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM bt_daily_outcomes WHERE session_close_t0 IS NOT NULL OR contained_close IS NOT NULL")
        nonnull = cur.fetchone()[0]
    print(f"\nGRANT UPDATE on new columns to dash_backfill_writer: {grants}/{len(NEW)}")
    print(f"bt_daily_outcomes_active columns: {view_cols}")
    print(f"non-null new-column rows before backfill: {nonnull} (expect 0)")
    assert grants == len(NEW) and nonnull == 0
    conn.close()


if __name__ == "__main__":
    main()
