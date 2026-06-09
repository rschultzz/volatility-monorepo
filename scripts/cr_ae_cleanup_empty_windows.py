"""
CR-AE: One-time cleanup of row_count=0 rows in orats_options_fetched_windows.

DRY-RUN by default — prints count and source breakdown, then exits.
Pass --execute to perform the DELETE.

Runs under DATABASE_URL (the app role; NOT BACKFILL_DATABASE_URL).
The app role can DELETE; dash_backfill_writer cannot.

Blast radius: only row_count=0 rows. orats_options_minute (the actual
market data) is untouched. Gaps self-refill on next live fetch.

No bt_backfill_runs row — this is a one-time maintenance op.
Log the run in the session note instead.

Usage:
    python scripts/cr_ae_cleanup_empty_windows.py             # dry-run
    python scripts/cr_ae_cleanup_empty_windows.py --execute   # delete
"""
from __future__ import annotations

import argparse
import os
import sys


def _get_conn():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        sys.exit("DATABASE_URL is not set")
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url[len("postgresql+psycopg://"):]
    elif url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    try:
        import psycopg
        return psycopg.connect(url)
    except ImportError:
        sys.exit("psycopg not installed — run: pip install psycopg")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Perform the DELETE (default is dry-run)")
    args = parser.parse_args()

    conn = _get_conn()

    # ── 1. Count and breakdown ────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM orats_options_fetched_windows WHERE row_count = 0")
        total = cur.fetchone()[0]

    print(f"row_count=0 rows in orats_options_fetched_windows: {total}")

    if total > 0:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, COUNT(*),
                       MIN(window_start_pt)::date, MAX(window_start_pt)::date,
                       MIN(fetched_at)::date,      MAX(fetched_at)::date
                FROM orats_options_fetched_windows
                WHERE row_count = 0
                GROUP BY source
                ORDER BY source
                """
            )
            rows = cur.fetchall()

        print(f"\n{'source':<30} {'count':>6}  {'win_start':>12}  {'win_end':>12}  {'fetch_start':>12}  {'fetch_end':>12}")
        print("-" * 95)
        for r in rows:
            source, count, ws, we, fs, fe = r
            print(f"{str(source):<30} {count:>6}  {str(ws):>12}  {str(we):>12}  {str(fs):>12}  {str(fe):>12}")

    if not args.execute:
        print("\nDry-run complete — no rows deleted. Pass --execute to delete.")
        conn.close()
        return

    if total == 0:
        print("\n--execute passed but no rows to delete. Nothing to do.")
        conn.close()
        return

    # ── 2. DELETE in a transaction ────────────────────────────────────────────
    print(f"\nDeleting {total} row_count=0 rows …")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DELETE FROM orats_options_fetched_windows WHERE row_count = 0")
            deleted = cur.rowcount

    print(f"Deleted: {deleted} rows")
    print("Done. Log the run in the session note (no bt_backfill_runs row written).")
    conn.close()


if __name__ == "__main__":
    main()
