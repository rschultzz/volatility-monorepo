#!/usr/bin/env python3
"""
backfill_gex_landscape.py
=========================
Backfill the orats_gex_landscape cache (CR-007) for historical trade dates.

Walks the distinct (ticker, trade_date) pairs already present in
orats_oi_gamma and runs compute_and_upsert_landscape for each — the exact
same computation the EOD cron (apps/cron/job_orats_eod.py) performs nightly.
Idempotent: every write is an UPSERT, so re-running is safe. Each date is
committed independently, so a single bad date does not abort the run.

Requires the orats_gex_landscape table to exist — apply
infra/sql/orats_gex_landscape.sql first.

Env: reads DATABASE_URL from a .env file at (or above) the repo root.

Usage:
    python scripts/backfill_gex_landscape.py                    # all dates
    python scripts/backfill_gex_landscape.py --since 2026-01-01 # from a date
    python scripts/backfill_gex_landscape.py --date 2026-05-20  # single date
    python scripts/backfill_gex_landscape.py --version <tag>    # override version
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import psycopg

# Make the repo root importable so `packages.*` resolves. Mirrors the bootstrap
# in scripts/run_gex_fade_backtest.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.gex_landscape import compute_and_upsert_landscape  # noqa: E402

# Mirrors the VERSION constant in apps/cron/job_orats_eod.py so backfilled rows
# carry the same version tag the nightly cron writes. Override with --version
# if the cron's VERSION has since changed.
DEFAULT_VERSION = "eod-shift-hard-upsert-2025-10-31d"

# Landscape compute params — must match the cron's call in job_orats_eod.py.
SPREAD_COEF = 8.0
RANGE_PTS = 200.0
STEP_PTS = 1.0


def load_env() -> None:
    """Load DATABASE_URL from a .env file at (or above) the repo root."""
    for parent in [REPO_ROOT, *REPO_ROOT.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(
                        k.strip(), v.strip().strip('"').strip("'"))
            return


def _normalize_db_url(url: str) -> str:
    """Coerce a DATABASE_URL into a form psycopg.connect accepts.

    Handles the postgres:// alias and strips any SQLAlchemy driver suffix
    (postgresql+psycopg://) — the repo's .env stores the SQLAlchemy form for
    the engine-based tooling, but psycopg.connect wants a plain URL.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def get_conn():
    """Open a psycopg connection from DATABASE_URL."""
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        sys.exit("ERROR: DATABASE_URL is not set (checked .env at the repo root).")
    return psycopg.connect(_normalize_db_url(raw))


def distinct_pairs(conn, *, since, single) -> list:
    """Distinct (ticker, trade_date) pairs in orats_oi_gamma, filtered."""
    sql = "SELECT DISTINCT ticker, trade_date FROM orats_oi_gamma"
    params: list = []
    if single is not None:
        sql += " WHERE trade_date = %s"
        params.append(single)
    elif since is not None:
        sql += " WHERE trade_date >= %s"
        params.append(since)
    sql += " ORDER BY trade_date, ticker"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill orats_gex_landscape.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--date", help="Backfill a single trade_date (YYYY-MM-DD).")
    group.add_argument("--since",
                       help="Backfill all trade_dates >= this (YYYY-MM-DD).")
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    help=f"Version tag stored on each row "
                         f"(default: {DEFAULT_VERSION}).")
    args = ap.parse_args()

    single = dt.date.fromisoformat(args.date) if args.date else None
    since = dt.date.fromisoformat(args.since) if args.since else None

    load_env()
    conn = get_conn()

    try:
        pairs = distinct_pairs(conn, since=since, single=single)
    except Exception as e:
        conn.close()
        sys.exit(f"ERROR: could not read orats_oi_gamma: {e}")

    if not pairs:
        print("No matching (ticker, trade_date) pairs in orats_oi_gamma. "
              "Nothing to do.")
        conn.close()
        return

    print(f"Backfilling orats_gex_landscape — {len(pairs)} "
          f"(ticker, trade_date) pairs")
    print(f"  version={args.version}  spread_coef={SPREAD_COEF}  "
          f"range_pts={RANGE_PTS}  step_pts={STEP_PTS}")

    ok = 0
    failures = []
    for i, (ticker, trade_date) in enumerate(pairs, start=1):
        tag = f"[{i:>4}/{len(pairs)}] {ticker} {trade_date}"
        try:
            summary = compute_and_upsert_landscape(
                conn=conn, ticker=ticker, trade_date=trade_date,
                spread_coef=SPREAD_COEF, range_pts=RANGE_PTS, step_pts=STEP_PTS,
                version=args.version,
            )
            conn.commit()
            ok += 1
            print(f"  {tag}  OK  n_landscape={summary['n_landscape']} "
                  f"n_walls={summary['n_walls']} "
                  f"table_spot={summary['table_spot']:.2f}")
        except Exception as e:
            conn.rollback()
            failures.append((ticker, trade_date, str(e)))
            print(f"  {tag}  FAIL  {e}")

    conn.close()

    print()
    print(f"Done. {ok} upserted, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for ticker, trade_date, err in failures[:20]:
            print(f"  {ticker} {trade_date}: {err[:160]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
