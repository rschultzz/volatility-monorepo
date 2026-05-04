"""
Repository for the options cache.

Read/write operations against the three cache tables. Wraps SQLAlchemy
text() with bound parameters (matches the pattern in service.py — no ORM,
no model classes mapped to tables, just raw queries with type-safe inputs).

All inserts on data tables are idempotent via ON CONFLICT DO NOTHING.
"""
from __future__ import annotations

import dataclasses
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .models import (
    FetchJob,
    FetchSource,
    FetchedWindow,
    JobStatus,
    OptionMinuteBar,
    TimeRange,
)


# ────────────────────────────────────────────────────────────────────────
#  Engine / connection management
# ────────────────────────────────────────────────────────────────────────

_engine: Optional[Engine] = None


def _load_env_file_if_present() -> None:
    """
    Best-effort load of a .env file from the project root.

    Walks up from this file's location looking for a .env, then loads it
    with python-dotenv if available. No-op if dotenv isn't installed or
    no .env file is found — in those cases we rely on the process env
    being populated externally (e.g. by Render in production, by a
    PyCharm run config, or by the user via `export`).

    Existing environment variables are NOT overridden — env always wins
    over .env, matching standard dotenv behavior.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return  # silently skip if python-dotenv isn't installed

    # Walk up from this file looking for a .env at any ancestor directory.
    # Stops at filesystem root.
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            load_dotenv(candidate, override=False)
            return
        parent = os.path.dirname(here)
        if parent == here:  # reached filesystem root
            return
        here = parent


def get_engine() -> Engine:
    """
    Return the lazily-initialized engine for the options cache.

    Reads DATABASE_URL from environment. If DATABASE_URL is not set,
    attempts to load a .env file from the project root before failing.
    Reuses a single engine across calls (engines are thread-safe and
    pool connections internally).
    """
    global _engine
    if _engine is None:
        url = os.environ.get("DATABASE_URL")
        if not url:
            _load_env_file_if_present()
            url = os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set in the environment. "
                "Set it directly, or place a .env file at the project root "
                "with python-dotenv installed (`pip install python-dotenv`)."
            )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def reset_engine() -> None:
    """For tests: dispose and reset the cached engine."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def _conn() -> Iterator:
    """Context manager yielding a connection with implicit transaction."""
    eng = get_engine()
    with eng.begin() as conn:
        yield conn


# ────────────────────────────────────────────────────────────────────────
#  orats_options_minute — insert
# ────────────────────────────────────────────────────────────────────────

# Column list shared between insert SQL and parameter binding.
# Source of truth: matches CREATE TABLE in 01_schema.sql.
_BAR_COLUMNS = (
    "opra_symbol", "ticker", "expir_date", "expir_date_d", "strike", "option_type",
    "trade_date", "trade_date_d", "quote_date", "snapshot_pt", "snapshot_utc",
    "updated_at", "snap_shot_est_time", "snap_shot_date",
    "stock_price", "spot_price", "dte",
    "bid_price", "ask_price", "bid_size", "ask_size",
    "bid_iv", "mid_iv", "ask_iv",
    "volume", "open_interest",
    "opt_value", "smv_vol", "ext_val", "ext_smv_vol", "residual_rate",
    "delta", "gamma", "theta", "vega", "rho", "phi", "driftless_theta",
    "expiry_tod", "ticker_id", "month_id",
)


def insert_bars(bars: Iterable[OptionMinuteBar]) -> int:
    """
    Bulk-insert bar rows into orats_options_minute.

    Idempotent: existing (opra_symbol, snapshot_pt) rows are silently
    skipped via ON CONFLICT DO NOTHING.

    Returns the number of rows actually inserted (may be < len(bars) if
    some were duplicates).
    """
    rows = [
        {col: getattr(bar, col) for col in _BAR_COLUMNS}
        for bar in bars
    ]
    if not rows:
        return 0

    cols_sql = ", ".join(_BAR_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _BAR_COLUMNS)

    sql = text(
        f"INSERT INTO orats_options_minute ({cols_sql}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (opra_symbol, snapshot_pt) DO NOTHING"
    )

    with _conn() as conn:
        result = conn.execute(sql, rows)
        # rowcount on executemany returns total affected rows
        return result.rowcount or 0


# ────────────────────────────────────────────────────────────────────────
#  orats_options_minute — read
# ────────────────────────────────────────────────────────────────────────

def get_bars_for_contract(
    opra_symbol: str,
    start_pt: datetime,
    end_pt: datetime,
) -> list[OptionMinuteBar]:
    """
    Fetch all cached bars for one contract over [start_pt, end_pt] inclusive.

    Returns rows in chronological order. Empty list if no data cached.
    """
    sql = text("""
        SELECT * FROM orats_options_minute
        WHERE opra_symbol = :sym
          AND snapshot_pt BETWEEN :start AND :end
        ORDER BY snapshot_pt
    """)
    with _conn() as conn:
        result = conn.execute(
            sql, {"sym": opra_symbol, "start": start_pt, "end": end_pt}
        )
        rows = result.mappings().all()

    return [_row_to_bar(r) for r in rows]


def count_bars_for_contract(
    opra_symbol: str,
    start_pt: datetime,
    end_pt: datetime,
) -> int:
    """
    Count cached bars for a contract over [start_pt, end_pt] without
    materializing them. Useful for quick coverage checks.
    """
    sql = text("""
        SELECT COUNT(*) FROM orats_options_minute
        WHERE opra_symbol = :sym
          AND snapshot_pt BETWEEN :start AND :end
    """)
    with _conn() as conn:
        result = conn.execute(
            sql, {"sym": opra_symbol, "start": start_pt, "end": end_pt}
        )
        return int(result.scalar_one())


def _row_to_bar(row: dict) -> OptionMinuteBar:
    """Convert a SQLAlchemy result row mapping into an OptionMinuteBar."""
    return OptionMinuteBar(**{col: row.get(col) for col in _BAR_COLUMNS})


# ────────────────────────────────────────────────────────────────────────
#  orats_options_fetched_windows
# ────────────────────────────────────────────────────────────────────────

def record_fetched_window(
    window: FetchedWindow,
) -> None:
    """
    Insert a fetched-window record. Idempotent on (opra_symbol, window_start_pt).

    Note: this does NOT auto-merge with adjacent windows. Callers wanting
    coalesced ranges should call coalesce_windows() periodically (a
    maintenance task).
    """
    sql = text("""
        INSERT INTO orats_options_fetched_windows (
            opra_symbol, window_start_pt, window_end_pt, row_count, source
        ) VALUES (
            :sym, :start, :end, :n, :src
        )
        ON CONFLICT (opra_symbol, window_start_pt) DO UPDATE SET
            window_end_pt = GREATEST(
                orats_options_fetched_windows.window_end_pt, EXCLUDED.window_end_pt
            ),
            row_count = orats_options_fetched_windows.row_count + EXCLUDED.row_count,
            fetched_at = now()
    """)
    with _conn() as conn:
        conn.execute(sql, {
            "sym": window.opra_symbol,
            "start": window.window_start_pt,
            "end": window.window_end_pt,
            "n": window.row_count,
            "src": window.source,
        })


def get_windows_for_contract(opra_symbol: str) -> list[FetchedWindow]:
    """Return all recorded fetch windows for a contract, ordered by start."""
    sql = text("""
        SELECT opra_symbol, window_start_pt, window_end_pt,
               row_count, source, fetched_at
        FROM orats_options_fetched_windows
        WHERE opra_symbol = :sym
        ORDER BY window_start_pt
    """)
    with _conn() as conn:
        result = conn.execute(sql, {"sym": opra_symbol})
        return [FetchedWindow(**dict(r)) for r in result.mappings().all()]


# ────────────────────────────────────────────────────────────────────────
#  orats_options_fetch_jobs
# ────────────────────────────────────────────────────────────────────────

def create_job(job: FetchJob) -> int:
    """
    Insert a new fetch job. Returns the assigned job_id.
    Mutates the passed job in place to set job_id.
    """
    sql = text("""
        INSERT INTO orats_options_fetch_jobs (
            kind, setup_ref, status, legs_requested, legs_completed
        ) VALUES (
            :kind, :setup_ref, :status, CAST(:legs AS jsonb), :done
        )
        RETURNING job_id, created_at
    """)
    with _conn() as conn:
        result = conn.execute(sql, {
            "kind": job.kind,
            "setup_ref": job.setup_ref,
            "status": job.status,
            "legs": json.dumps(job.legs_requested),
            "done": job.legs_completed,
        })
        row = result.mappings().one()
        job.job_id = row["job_id"]
        job.created_at = row["created_at"]
        return job.job_id


def update_job_status(
    job_id: int,
    status: JobStatus,
    legs_completed: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Update a job's status. Sets started_at on transition to 'running' and
    completed_at on transition to 'completed' or 'failed'.
    """
    started_clause = "started_at = COALESCE(started_at, now())" if status == "running" else "started_at = started_at"
    completed_clause = (
        "completed_at = now()"
        if status in ("completed", "failed")
        else "completed_at = completed_at"
    )

    sql = text(f"""
        UPDATE orats_options_fetch_jobs
        SET status = :status,
            legs_completed = COALESCE(:done, legs_completed),
            error_message = COALESCE(:err, error_message),
            {started_clause},
            {completed_clause}
        WHERE job_id = :id
    """)
    with _conn() as conn:
        conn.execute(sql, {
            "status": status,
            "done": legs_completed,
            "err": error_message,
            "id": job_id,
        })


def get_job(job_id: int) -> Optional[FetchJob]:
    """Fetch a job by ID. Returns None if not found."""
    sql = text("""
        SELECT job_id, kind, setup_ref, status, legs_requested,
               legs_completed, error_message,
               created_at, started_at, completed_at
        FROM orats_options_fetch_jobs
        WHERE job_id = :id
    """)
    with _conn() as conn:
        result = conn.execute(sql, {"id": job_id})
        row = result.mappings().one_or_none()
        if row is None:
            return None
        d = dict(row)
        # legs_requested comes back as already-decoded JSON from psycopg
        return FetchJob(**d)
