"""CR-BD decision 2 — the implied-move and outcomes crons' poll gates (pure parts;
the poll itself is tested in packages/shared/tests/test_snapshot_poll.py)."""
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("BACKFILL_DATABASE_URL", "postgresql://test:test@localhost:1/test")

import scripts.cr_ab_open_implied_move as fill
import scripts.cr_b_backfill_outcomes as outcomes


def test_implied_move_fill_skips_a_holiday_target_immediately():
    assert "not an NYSE trading day" in fill.skip_reason_for(date(2026, 9, 7))      # the row the EOD job mis-stamped
    assert fill.skip_reason_for(date(2026, 9, 8)) is None
    assert "not an NYSE trading day" in fill.skip_reason_for(date(2026, 11, 26))


def test_outcomes_cron_waits_only_on_a_trading_day_nightly_run():
    today = date(2026, 11, 2)                                                      # Monday after the DST change
    assert outcomes.wait_needed(today, None, False, False) is None                 # → poll for today's snapshot
    assert "not an NYSE trading day" in outcomes.wait_needed(date(2026, 11, 26), None, False, False)
    assert outcomes.wait_needed(today, None, True, False) == "--no-wait"
    assert outcomes.wait_needed(today, None, False, True) == "--dry-run"
    assert "historical backfill" in outcomes.wait_needed(today, date(2026, 6, 22), False, False)
    assert outcomes.wait_needed(today, date(2026, 11, 2), False, False) is None     # --to-date today still waits


def test_both_crons_expose_the_poll_flags():
    import argparse
    for mod in (fill, outcomes):
        src = Path(mod.__file__).read_text()
        assert "--max-wait-min" in src and "--poll-seconds" in src and "wait_for_open_snapshot(" in src
        assert "no_snapshot_line(" in src
