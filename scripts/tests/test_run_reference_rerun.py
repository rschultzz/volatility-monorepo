"""CR-AU decision 3 — reference re-run configuration (pure)."""
import sys
from datetime import date
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.run_reference_rerun import build_command, last_month_end, reference_config


def test_last_month_end():
    assert last_month_end(date(2026, 9, 1)) == date(2026, 8, 31)
    assert last_month_end(date(2026, 3, 1)) == date(2026, 2, 28)
    assert last_month_end(date(2026, 1, 15)) == date(2025, 12, 31)


def test_reference_config_and_cr_id():
    cfg = reference_config(date(2026, 10, 1))
    assert cfg["universe_end"] == date(2026, 9, 30) and cfg["split_date"] == date(2026, 6, 5)
    assert cfg["mode"] == "walk-forward" and cfg["cr_id"] == "REF-2026-09" and cfg["dry_run"] is False


def test_command_matches_decision_3():
    cmd = build_command(reference_config(date(2026, 9, 7)), python="py", harness=Path("h.py"))
    assert cmd == ["py", "-u", "h.py", "--universe-end", "2026-08-31", "--split-date", "2026-06-05",
                   "--structural-prob-mode", "walk-forward", "--cr-id", "REF-2026-08"]
    assert "--holdout-read" not in cmd                       # decision 4: the monthly run never reads the holdout


def test_dry_run_is_selection_only_under_a_non_ref_cr_id():
    cfg = reference_config(date(2026, 9, 7), dry_run=True)
    cmd = build_command(cfg, python="py", harness=Path("h.py"))
    assert cfg["cr_id"] == "REFDRY-2026-08" and not cfg["cr_id"].startswith("REF-")
    assert "--selection-only" in cmd and "--no-persist" in cmd
