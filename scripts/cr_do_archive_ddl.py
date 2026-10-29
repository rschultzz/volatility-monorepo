#!/usr/bin/env python3
"""CR-DO D5 — create bt_derived_repair_archive as the owner role and verify its privileges.

Runs infra/sql/bt_derived_repair_archive.sql on DATABASE_URL (the owner, approved by Ryan in chat
2026-10-29), then checks: relacl is exactly INSERT for dash_backfill_writer; no non-superuser role
has SELECT, UPDATE, DELETE or TRUNCATE (INSERT reaches members of dash_backfill_writer). Prints the DDL / grant time (UTC). DB errors print the
error type only. Reads no data.

Usage:
    apps/web/.venv/bin/python scripts/cr_do_archive_ddl.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import psycopg  # noqa: E402

from packages.shared.backfill_safety import _normalize_url  # noqa: E402

TABLE = "bt_derived_repair_archive"
DDL = REPO_ROOT / "infra" / "sql" / f"{TABLE}.sql"


def main() -> int:
    try:
        conn = psycopg.connect(_normalize_url(os.environ["DATABASE_URL"]))
        conn.autocommit = True
        role, owner = conn.execute(
            "SELECT current_user, tableowner FROM pg_tables WHERE tablename = 'bt_daily_outcomes'").fetchone()
        if role != owner:
            print(f"refusing: connected as {role}, bt_daily_outcomes is owned by {owner}")
            return 1
        if conn.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()[0] is not None:
            print(f"{TABLE} already exists; not re-created")
        else:
            conn.execute(DDL.read_text())
        created = conn.execute("SELECT clock_timestamp() AT TIME ZONE 'UTC'").fetchone()[0]
        acl = conn.execute("SELECT relacl::text[] FROM pg_class WHERE oid = %s::regclass", (TABLE,)).fetchone()[0]
        privs = conn.execute(
            "SELECT r.rolname, p.priv FROM pg_roles r, unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE']) p(priv) "
            "WHERE r.rolname !~ '^pg_' AND NOT r.rolsuper AND has_table_privilege(r.oid, %s::regclass, p.priv) "
            "ORDER BY 1, 2", (TABLE,)).fetchall()
        n = conn.execute("SELECT count(*) FROM pg_attribute WHERE attrelid = %s::regclass AND attnum > 0", (TABLE,)).fetchone()[0]
        conn.close()
    except Exception as e:  # never echo the connection string
        print(f"DB error: {type(e).__name__}")
        return 1
    print(f"DDL + grants done as {role} (owner); verified at {created:%Y-%m-%d %H:%M:%S} UTC; {n} columns")
    print(f"relacl: {acl}")
    print(f"effective non-superuser privileges: {privs}")
    # INSERT also reaches roles that are members of dash_backfill_writer; nothing else may
    ok = acl == [f"dash_backfill_writer=a/{role}"] and all(p == "INSERT" for _, p in privs)
    print("privilege check:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
