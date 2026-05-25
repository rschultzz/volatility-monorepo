"""backfill_safety — connection and run-tracking helpers for unattended CR backfills (CR-020).

All unattended scripts must:
    1. Use get_backfill_db_conn() instead of psycopg.connect(DATABASE_URL).
    2. Call verify_safe_role(conn) before any data writes.
    3. Wrap their work in the backfill_run() context manager so every run is tracked.

The dash_backfill_writer role has no DELETE/TRUNCATE/DROP — those operations
fail at the DB layer regardless of code bugs.
"""
from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg.types.json import Jsonb

_SAFE_ROLE = "dash_backfill_writer"


def _normalize_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def get_backfill_db_conn() -> psycopg.Connection:
    """Return a live psycopg connection using BACKFILL_DATABASE_URL (autocommit=True)."""
    raw = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("BACKFILL_DATABASE_URL is not set")
    conn = psycopg.connect(_normalize_url(raw))
    conn.autocommit = True
    return conn


def verify_safe_role(conn: psycopg.Connection) -> None:
    """Raise RuntimeError if current_user is not dash_backfill_writer."""
    row = conn.execute("SELECT current_user").fetchone()
    actual = row[0] if row else "(none)"
    if actual != _SAFE_ROLE:
        raise RuntimeError(
            f"Safety check failed: expected role '{_SAFE_ROLE}', "
            f"got '{actual}'. Use BACKFILL_DATABASE_URL, not DATABASE_URL."
        )


def assert_role_or_die(conn: psycopg.Connection) -> None:
    """Hard-fail at script start if connection is not dash_backfill_writer.

    Convenience wrapper for top-of-script use before any expensive setup.
    """
    try:
        verify_safe_role(conn)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")


@contextmanager
def backfill_run(
    conn: psycopg.Connection,
    cr_id: str,
) -> Generator[str, None, None]:
    """Context manager that tracks a backfill run in bt_backfill_runs.

    On entry: inserts a row with status='running'; yields the run_id (UUID str).
    On clean exit: sets status='completed', completed_at=NOW().
    On exception: sets status='aborted', completed_at=NOW(), notes=full traceback;
                  re-raises.

    Usage::

        with backfill_run(conn, "CR-A") as run_id:
            # ... do work ...
            update_run_smoke(conn, run_id, smoke_results, self_assessment)
    """
    row = conn.execute(
        "INSERT INTO bt_backfill_runs (cr_id) VALUES (%s) RETURNING run_id",
        (cr_id,),
    ).fetchone()
    run_id = str(row[0])
    try:
        yield run_id
    except Exception:
        conn.execute(
            "UPDATE bt_backfill_runs "
            "SET status = 'aborted', completed_at = NOW(), notes = %s "
            "WHERE run_id = %s",
            (traceback.format_exc(), run_id),
        )
        raise
    else:
        conn.execute(
            "UPDATE bt_backfill_runs "
            "SET status = 'completed', completed_at = NOW() "
            "WHERE run_id = %s",
            (run_id,),
        )


def update_run_progress(
    conn: psycopg.Connection,
    run_id: str,
    rows_inserted: int,
) -> None:
    """Update rows_inserted on an in-progress backfill run (progress heartbeat)."""
    conn.execute(
        "UPDATE bt_backfill_runs SET rows_inserted = %s WHERE run_id = %s",
        (rows_inserted, run_id),
    )


def update_run_smoke(
    conn: psycopg.Connection,
    run_id: str,
    smoke_results: dict,
    self_assessment: str,
) -> None:
    """Write smoke test output to the bt_backfill_runs row.

    Call inside backfill_run() after smoke tests pass.
    The context manager sets the final status on exit — don't set it here.
    """
    conn.execute(
        "UPDATE bt_backfill_runs "
        "SET smoke_test_results = %s, self_assessment = %s "
        "WHERE run_id = %s",
        (Jsonb(smoke_results), self_assessment, run_id),
    )
