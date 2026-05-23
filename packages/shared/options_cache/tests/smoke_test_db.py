"""
End-to-end DB smoke test for the options cache.

Run this AFTER you've executed 01_schema.sql and 02_indexes.sql against
your DB. It verifies that:
    1. The repository can connect using DATABASE_URL
    2. Inserts work (and are idempotent)
    3. Reads return what was written
    4. Fetched-window tracking works
    5. Job lifecycle works

This test creates real rows but uses a sentinel test OPRA symbol
('XXTEST') that won't collide with real ORATS data. After running, it
cleans up after itself.

Run with:
    python -m packages.shared.options_cache.tests.smoke_test_db

NOT a unittest — this is a one-shot script meant for manual verification.
"""
from __future__ import annotations

import sys
from datetime import date, datetime

from packages.shared.options_cache import (
    FetchJob,
    FetchedWindow,
    OptionMinuteBar,
    TimeRange,
)
from packages.shared.options_cache import repository as repo
from packages.shared.options_cache import windows as W
from sqlalchemy import text


TEST_SYMBOL = "XXTEST260117P05800000"  # safe sentinel, no real OPRA root is "XXTEST"


def _cleanup() -> None:
    """Remove any test rows from a previous run."""
    with repo._conn() as conn:
        conn.execute(
            text("DELETE FROM orats_options_minute WHERE opra_symbol LIKE 'XXTEST%'")
        )
        conn.execute(
            text(
                "DELETE FROM orats_options_fetched_windows "
                "WHERE opra_symbol LIKE 'XXTEST%'"
            )
        )
        conn.execute(
            text(
                "DELETE FROM orats_options_fetch_jobs "
                "WHERE setup_ref = 'smoke_test'"
            )
        )


def _make_bar(snapshot_pt: datetime) -> OptionMinuteBar:
    """Build a minimal OptionMinuteBar for testing."""
    return OptionMinuteBar(
        opra_symbol=TEST_SYMBOL,
        ticker="XXTEST",
        expir_date="2026-01-17",
        expir_date_d=date(2026, 1, 17),
        strike=5800.0,
        option_type="P",
        trade_date=snapshot_pt.strftime("%Y-%m-%d"),
        trade_date_d=snapshot_pt.date(),
        quote_date=snapshot_pt.strftime("%Y-%m-%dT%H:%M:%S"),
        snapshot_pt=snapshot_pt,
        snapshot_utc=datetime.fromisoformat(
            snapshot_pt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        ),  # cheating; real ingest converts properly
        bid_price=12.50,
        ask_price=12.75,
        delta=-0.42,
    )


def main() -> int:
    print("─" * 60)
    print("Options cache DB smoke test")
    print("─" * 60)

    print("\n[1/6] Cleaning up any stale test rows from previous runs...")
    _cleanup()
    print("    ✓ clean")

    # ── Test inserts ──
    print("\n[2/6] Inserting 5 bars (1-minute apart)...")
    bars = [
        _make_bar(datetime(2026, 1, 17, 9, 30 + i))
        for i in range(5)
    ]
    n_inserted = repo.insert_bars(bars)
    print(f"    ✓ inserted {n_inserted} rows")
    assert n_inserted == 5, f"expected 5 inserts, got {n_inserted}"

    # ── Test idempotency ──
    print("\n[3/6] Re-inserting the same 5 bars (should be no-ops)...")
    n_dup = repo.insert_bars(bars)
    print(f"    ✓ inserted {n_dup} rows (expected 0)")
    assert n_dup == 0, f"expected 0 duplicates, got {n_dup}"

    # ── Test reads ──
    print("\n[4/6] Reading back...")
    fetched = repo.get_bars_for_contract(
        TEST_SYMBOL,
        datetime(2026, 1, 17, 9, 30),
        datetime(2026, 1, 17, 9, 34),
    )
    print(f"    ✓ read {len(fetched)} bars")
    assert len(fetched) == 5, f"expected 5 bars, got {len(fetched)}"
    assert fetched[0].snapshot_pt < fetched[-1].snapshot_pt, "should be sorted"
    assert fetched[0].bid_price == 12.50
    assert fetched[0].delta == -0.42

    count = repo.count_bars_for_contract(
        TEST_SYMBOL,
        datetime(2026, 1, 17, 9, 30),
        datetime(2026, 1, 17, 9, 34),
    )
    print(f"    ✓ count_bars_for_contract = {count}")
    assert count == 5

    # ── Test fetched-window tracking ──
    print("\n[5/6] Recording a fetched window + gap detection...")
    repo.record_fetched_window(FetchedWindow(
        opra_symbol=TEST_SYMBOL,
        window_start_pt=datetime(2026, 1, 17, 9, 30),
        window_end_pt=datetime(2026, 1, 17, 9, 34),
        row_count=5,
        source="manual",
    ))
    existing = repo.get_windows_for_contract(TEST_SYMBOL)
    print(f"    ✓ recorded; {len(existing)} window(s) tracked")

    # Gap detection: ask for 09:30-09:40 — should report 09:35-09:40 missing
    request = TimeRange(
        start_pt=datetime(2026, 1, 17, 9, 30),
        end_pt=datetime(2026, 1, 17, 9, 40),
    )
    gaps = W.find_gaps(request, existing)
    print(f"    ✓ gaps for 09:30-09:40: {len(gaps)}")
    assert len(gaps) == 1
    assert gaps[0].start_pt == datetime(2026, 1, 17, 9, 35)
    assert gaps[0].end_pt == datetime(2026, 1, 17, 9, 40)

    # Gap detection: ask for what's already covered — should report no gaps
    fully_covered_request = TimeRange(
        start_pt=datetime(2026, 1, 17, 9, 30),
        end_pt=datetime(2026, 1, 17, 9, 34),
    )
    no_gaps = W.find_gaps(fully_covered_request, existing)
    print(f"    ✓ gaps for already-covered range: {len(no_gaps)} (expected 0)")
    assert no_gaps == []

    # ── Test job lifecycle ──
    print("\n[6/6] Job lifecycle...")
    job = FetchJob(
        kind="manual",
        legs_requested=[
            {
                "opra_symbol": TEST_SYMBOL,
                "start_pt": "2026-01-17 09:30",
                "end_pt": "2026-01-17 09:40",
            }
        ],
        setup_ref="smoke_test",
    )
    job_id = repo.create_job(job)
    print(f"    ✓ created job {job_id}, status={job.status}")
    assert job.job_id == job_id
    assert job.created_at is not None

    repo.update_job_status(job_id, "running")
    j = repo.get_job(job_id)
    assert j is not None
    assert j.status == "running"
    assert j.started_at is not None
    print(f"    ✓ transitioned to running (started_at={j.started_at})")

    repo.update_job_status(job_id, "completed", legs_completed=1)
    j = repo.get_job(job_id)
    assert j is not None
    assert j.status == "completed"
    assert j.legs_completed == 1
    assert j.completed_at is not None
    print(f"    ✓ transitioned to completed (completed_at={j.completed_at})")

    print("\n[cleanup] Removing test rows...")
    _cleanup()
    print("    ✓ clean")

    print("\n" + "─" * 60)
    print("All smoke checks passed ✓")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
