#!/usr/bin/env python3
"""
backfill_daily_features.py
==========================
Backfill the bt_daily_features cache (CR-013) for historical trade dates.

Walks the distinct (ticker, trade_date) pairs already present in
orats_gex_landscape and runs compute_and_upsert_daily_features for each —
the same computation the EOD cron (apps/cron/job_orats_eod.py) performs
nightly after the landscape upsert. Idempotent: every write is an UPSERT,
so re-running is safe. Each date is committed independently, so a single
bad date does not abort the run.

Requires the bt_daily_features table to exist — apply
infra/sql/bt_daily_features.sql first.

Env: reads DATABASE_URL from a .env file at (or above) the repo root.

Usage:
    python scripts/backfill_daily_features.py                    # all dates
    python scripts/backfill_daily_features.py --since 2026-01-01 # from a date
    python scripts/backfill_daily_features.py --date 2026-05-07  # single date
    python scripts/backfill_daily_features.py --version <tag>    # override
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import psycopg

# Make the repo root importable so `packages.*` resolves.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.day_features import (  # noqa: E402
    FEATURE_VERSION,
    compute_and_upsert_daily_features,
)


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
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def get_conn():
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        sys.exit("ERROR: DATABASE_URL is not set (checked .env at the repo root).")
    return psycopg.connect(_normalize_db_url(raw))


def distinct_pairs(conn, *, since, single) -> list:
    """Distinct (ticker, trade_date) pairs in orats_gex_landscape, filtered."""
    sql = "SELECT DISTINCT ticker, trade_date FROM orats_gex_landscape"
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
    ap = argparse.ArgumentParser(description="Backfill bt_daily_features.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--date", help="Backfill a single trade_date (YYYY-MM-DD).")
    group.add_argument("--since",
                       help="Backfill all trade_dates >= this (YYYY-MM-DD).")
    ap.add_argument("--version", default=FEATURE_VERSION,
                    help=f"feature_version tag stored on each row "
                         f"(default: {FEATURE_VERSION}).")
    args = ap.parse_args()

    single = dt.date.fromisoformat(args.date) if args.date else None
    since = dt.date.fromisoformat(args.since) if args.since else None

    load_env()
    conn = get_conn()

    try:
        pairs = distinct_pairs(conn, since=since, single=single)
    except Exception as e:
        conn.close()
        sys.exit(f"ERROR: could not read orats_gex_landscape: {e}")

    if not pairs:
        print("No matching (ticker, trade_date) pairs in orats_gex_landscape. "
              "Nothing to do.")
        conn.close()
        return

    print(f"Backfilling bt_daily_features — {len(pairs)} "
          f"(ticker, trade_date) pairs, version={args.version}")

    ok = 0
    failures = []
    for i, (ticker, trade_date) in enumerate(pairs, start=1):
        tag = f"[{i:>4}/{len(pairs)}] {ticker} {trade_date}"
        try:
            summary = compute_and_upsert_daily_features(
                conn=conn, ticker=ticker, trade_date=trade_date,
                version=args.version,
            )
            conn.commit()
            ok += 1
            print(f"  {tag}  OK  spot={summary['spot']:.2f} "
                  f"implied_move={summary['implied_move']:.2f} "
                  f"n_features={summary['n_features']}")
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
