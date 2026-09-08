#!/usr/bin/env python3
"""CR-AR Commit 5 — apply infra/sql/bt_edge_backtest_results_per_point.sql under the owner role.

schema_change class: must use DATABASE_URL (table owner), NOT BACKFILL_DATABASE_URL.
Idempotent (ADD COLUMN IF NOT EXISTS). Prints the statement file, the resulting
column list and the backfill role's table privileges.

Usage:
    python -u scripts/cr_ar_run_migration.py
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

NEW = ("close_pnl_per_point", "baseline_per_point", "beat_per_point", "mean_width_actual")
SQL_PATH = REPO_ROOT / "infra" / "sql" / "bt_edge_backtest_results_per_point.sql"


def main() -> None:
    raw = os.environ.get("DATABASE_URL", "").strip()
    if not raw:
        sys.exit("ERROR: DATABASE_URL is not set")
    conn = psycopg.connect(_normalize_url(raw))
    conn.autocommit = True   # the SQL file carries its own BEGIN/COMMIT
    with conn.cursor() as cur:
        cur.execute("SELECT current_user"); role = cur.fetchone()[0]
        cur.execute("SELECT tableowner FROM pg_tables WHERE tablename='bt_edge_backtest_results'"); row = cur.fetchone()
    owner = row[0] if row else None
    print(f"Connected as {role}; bt_edge_backtest_results owner = {owner}")
    if role == "dash_backfill_writer" or owner is None or role != owner:
        sys.exit("ERROR: must run as the table owner via DATABASE_URL")

    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='bt_edge_backtest_results' AND column_name = ANY(%s)", (list(NEW),))
        pre = sorted(r[0] for r in cur.fetchall())
    print(f"new columns already present before migration: {pre or 'none'}")

    sql = SQL_PATH.read_text()
    print(f"\nApplying {SQL_PATH.relative_to(REPO_ROOT)} ({len(sql)} bytes)")
    print("--- statement ---"); print("\n".join(l for l in sql.splitlines() if l.strip() and not l.startswith("--"))); print("--- end ---")
    with conn.cursor() as cur:
        cur.execute(sql)
    print("applied ✓")

    with conn.cursor() as cur:
        cur.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name='bt_edge_backtest_results' ORDER BY ordinal_position")
        cols = cur.fetchall()
    print(f"\n\\d bt_edge_backtest_results — {len(cols)} columns:")
    for c in cols:
        print(f"  {c[0]:<24} {c[1]:<28} nullable={c[2]}")
    with conn.cursor() as cur:
        cur.execute("SELECT privilege_type FROM information_schema.table_privileges WHERE table_name='bt_edge_backtest_results' AND grantee='dash_backfill_writer' ORDER BY 1")
        privs = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT cr_id, count(*) FROM bt_edge_backtest_results GROUP BY cr_id ORDER BY cr_id")
        counts = cur.fetchall()
    print(f"\ndash_backfill_writer table privileges: {privs}")
    print(f"rows by cr_id: {counts}")
    assert all(c in {r[0] for r in cols} for c in NEW)
    assert "INSERT" in privs and "SELECT" in privs
    conn.close()


if __name__ == "__main__":
    main()
