#!/usr/bin/env python3
"""CR-DO smoke diagnosis (read-only, counts and column names only).

Smoke 1 of run 642e0d88 failed on the v0.6.1 / v0.6.0-openiv holdout groups, including md5_all_post,
which hashes the recompute over the stored-non-NULL ("filled") cells. The bars did not change (no
bar after 2026-10-29 20:59 UTC), so this counts, per version and group, the two cell classes the
smoke's re-derivation cannot tell apart from the stored state alone:
  null_written   stamped row, stored NULL now, post NULL, pre not NULL — a planned cell whose
                 recompute is NULL, now written NULL (or a never-filled cell with a NULL post);
  equal_unwritten  stamped row, CR-BZ column, stored = post — a written cell, or a reproduction-
                 excluded cell that already equalled post (Step 0 'excluded CR-BZ values').
No value, row or date is printed. DB errors print the error type only.

Usage:
    apps/web/.venv/bin/python scripts/cr_do_smoke_diag.py 642e0d88-ae02-4297-b8d5-8d09ebb3b996
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cr_do_step0_dryrun import (  # noqa: E402
    CRBZ_RUN, GROUPS, LOG_DIR, OWNED, analyse, canon, column_types, connect,
)


def main() -> int:
    run_id = sys.argv[1]
    logging.disable(logging.WARNING)
    exp = json.loads((LOG_DIR / "step0_run4.json").read_text())
    try:
        conn = connect()
        conn.read_only = True
        types = column_types(conn)
        crbz_done = conn.execute("SELECT completed_at FROM bt_backfill_runs WHERE run_id = %s", (CRBZ_RUN,)).fetchone()[0]
        for version in OWNED:
            rows = analyse(conn, version, types, crbz_done, run_id=run_id)
            for g in GROUPS:
                rs = [r for r in rows.values() if r["regime"] == g and r["written"]]
                null_written = Counter(
                    c for r in rs for c in OWNED[version]
                    if r["stored"][c] is None and canon(r["post"][c], types[c]) == "NULL"
                    and canon(r["pre"][c], types[c]) != "NULL")
                e = exp["versions"][version][g]
                print(f"{version} {g:7} stamped {len(rs)}; null_written {sum(null_written.values())} "
                      f"{dict(null_written)}; re-derived written {sum(len(r['planned']) for r in rs)} vs Step 0 planned "
                      f"{e['planned_values']}; Step 0 excluded CR-BZ values {e['excluded_crbz_values']}", flush=True)
        conn.close()
    except Exception as ex:  # never echo the connection string or a row
        print(f"error: {type(ex).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
