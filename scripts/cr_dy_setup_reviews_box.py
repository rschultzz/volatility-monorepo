#!/usr/bin/env python3
"""CR-DY goal 7 — setup_reviews gains nullable box_end_ts, box_top, box_bottom (anchor_ts is the start).

Additive only: ADD COLUMN IF NOT EXISTS, no default, existing rows stay NULL; verified below. The run is recorded in
bt_backfill_runs through the backfill role. dash_backfill_writer has no DDL, so the ALTER itself runs as the table owner
(DATABASE_URL), with a short lock timeout. Reads no labels or outcomes. DB errors never echo a connection string.

Usage: apps/web/.venv/bin/python scripts/cr_dy_setup_reviews_box.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, connect_redacted, get_backfill_db_conn, redact_db_url, update_run_smoke,
)

COLUMNS = {"box_end_ts": "timestamptz", "box_top": "numeric", "box_bottom": "numeric"}


def main() -> int:
    try:
        run_conn = get_backfill_db_conn()
        assert_role_or_die(run_conn)
        with backfill_run(run_conn, "CR-DY-7") as run_id:
            owner = connect_redacted(os.environ["DATABASE_URL"], autocommit=True)
            role, tableowner = owner.execute(
                "SELECT current_user, tableowner FROM pg_tables WHERE tablename = 'setup_reviews'").fetchone()
            if role != tableowner:
                raise RuntimeError(f"connected as {role}, setup_reviews is owned by {tableowner}")
            before = owner.execute("SELECT count(*) FROM setup_reviews").fetchone()[0]
            owner.execute("SET lock_timeout = '5s'")
            for col, typ in COLUMNS.items():
                owner.execute(f"ALTER TABLE setup_reviews ADD COLUMN IF NOT EXISTS {col} {typ}")
            cols = dict(owner.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'setup_reviews' AND column_name = ANY(%s)", (list(COLUMNS),)).fetchall())
            after, non_null = owner.execute(
                "SELECT count(*), count(box_end_ts) + count(box_top) + count(box_bottom) FROM setup_reviews").fetchone()
            owner.close()
            smoke = {"rows_before": before, "rows_after": after, "non_null_new_values": non_null, "columns": cols}
            ok = before == after and non_null == 0 and set(cols) == set(COLUMNS) and all(v == "YES" for v in cols.values())
            update_run_smoke(run_conn, run_id, smoke, "additive: 3 nullable columns, existing rows NULL" if ok else "CHECK FAILED")
            print(f"run {run_id}: {smoke} ok={ok}")
            if not ok:
                raise RuntimeError("post-migration check failed")
        run_conn.close()
    except Exception as e:
        print(f"failed: {redact_db_url(e)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
