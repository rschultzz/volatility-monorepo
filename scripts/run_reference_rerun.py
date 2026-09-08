#!/usr/bin/env python3
"""CR-AU decision 3 — monthly reference re-run of the debit/credit edge backtest.

Render cron `reference-rerun`, `0 14 1 * *` (1st of the month,
14:00 UTC). Wraps scripts/cr_ah_step4_analysis.py with the reference configuration:

    --universe-end <last calendar month end>   (e.g. run on 2026-10-01 → 2026-09-30)
    --split-date 2026-06-05
    --structural-prob-mode walk-forward
    --cr-id REF-YYYY-MM                        (YYYY-MM = the universe-end month)

Cells are persisted under that cr_id; the card's stamp reads the latest
`cr_id LIKE 'REF-%'` (falling back to CR-AR until the first run). The holdout
read rule (decision 4) applies inside the harness: with no --holdout-read the
log carries holdout n per band and dates matured only.

--dry-run (G4): the harness runs --selection-only --no-persist under
cr_id REFDRY-YYYY-MM (never matches REF-%), so the universe resolves and the
selection prints without a 30-minute run or any cells.

Output is tee'd to scripts/logs/reference_rerun_<cr_id>_<timestamp>.log.
Exit code = the harness's exit code.

Usage:
    python scripts/run_reference_rerun.py [--dry-run] [--today YYYY-MM-DD] [--split-date 2026-06-05]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "scripts" / "cr_ah_step4_analysis.py"
LOG_DIR = REPO_ROOT / "scripts" / "logs"
DEFAULT_SPLIT_DATE = date(2026, 6, 5)
STRUCTURAL_PROB_MODE = "walk-forward"
CR_ID_PREFIX = "REF-"
DRY_CR_ID_PREFIX = "REFDRY-"


def last_month_end(today: date) -> date:
    """Last day of the calendar month before `today`'s month."""
    return today.replace(day=1) - timedelta(days=1)


def reference_config(today: date, *, split_date: date = DEFAULT_SPLIT_DATE, dry_run: bool = False) -> dict:
    """Pure: the reference run's parameters for a given run date."""
    ue = last_month_end(today)
    prefix = DRY_CR_ID_PREFIX if dry_run else CR_ID_PREFIX
    return {"universe_end": ue, "split_date": split_date, "mode": STRUCTURAL_PROB_MODE,
            "cr_id": f"{prefix}{ue:%Y-%m}", "dry_run": dry_run}


def build_command(cfg: dict, python: str = sys.executable, harness: Path = HARNESS) -> list[str]:
    cmd = [python, "-u", str(harness),
           "--universe-end", cfg["universe_end"].isoformat(),
           "--split-date", cfg["split_date"].isoformat(),
           "--structural-prob-mode", cfg["mode"],
           "--cr-id", cfg["cr_id"]]
    if cfg["dry_run"]:
        cmd += ["--selection-only", "--no-persist"]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CR-AU monthly reference re-run (wraps cr_ah_step4_analysis.py)")
    ap.add_argument("--today", default=None, help="run date YYYY-MM-DD (default: today); universe end = last day of the previous month")
    ap.add_argument("--split-date", type=date.fromisoformat, default=DEFAULT_SPLIT_DATE)
    ap.add_argument("--dry-run", action="store_true", help="G4: --selection-only --no-persist under cr_id REFDRY-YYYY-MM")
    args = ap.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    cfg = reference_config(today, split_date=args.split_date, dry_run=args.dry_run)
    cmd = build_command(cfg)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"reference_rerun_{cfg['cr_id']}_{datetime.now():%Y%m%d_%H%M%S}.log"
    print(f"CR-AU reference re-run  today={today}  universe_end={cfg['universe_end']}  split_date={cfg['split_date']}  "
          f"cr_id={cfg['cr_id']}  dry_run={cfg['dry_run']}\n  cmd: {' '.join(cmd)}\n  log: {log_path}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as fh, subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE,
                                                       stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            fh.write(line); fh.flush()
        proc.wait()
    print(f"reference re-run finished: exit={proc.returncode}  cr_id={cfg['cr_id']}  log={log_path}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
