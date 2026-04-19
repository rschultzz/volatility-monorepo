#!/usr/bin/env python3
"""
ORATS EOD Strikes Backfill for orats_oi_gamma
=============================================
Backfills SPX EOD strike data (OI, gamma, discounted levels) for all trading
days from 2025-01-02 through 2025-10-23 (API dates), stored as
2025-01-03 through 2025-10-24 under the store_trade_date = next_business_day
convention used by job_orats_eod.py.

Also re-pulls 4 legacy dates (2025-10-24, 27, 28, 29) whose dte and
discounted_level were computed by an older version of the ingest code.
These rows get upserted with corrected values.

Logic is byte-identical to job_orats_eod.py: same endpoints, same SPX/SPY
dividend-yield fallback, same rf30 fallback, same discount formula.

Cost: $0 (all calls included in your existing ORATS subscription).

Requirements:
    pip install requests python-dotenv sqlalchemy psycopg pandas pandas-market-calendars

Usage:
    python backfill_orats_oi_gamma.py                # interactive
    python backfill_orats_oi_gamma.py --dry-run      # show plan, don't execute
    python backfill_orats_oi_gamma.py --yes          # skip confirmation
    python backfill_orats_oi_gamma.py --skip-backfill  # only run legacy cleanup
    python backfill_orats_oi_gamma.py --skip-legacy    # only run backfill
"""

import argparse
import datetime as dt
import os
import sys
import time
from math import exp
from pathlib import Path

import pandas_market_calendars as mcal
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# --- Config ---------------------------------------------------------------
TICKER              = "SPX"
DTE_MAX             = 400
CONTRACT_MULTIPLIER = 100.0

STRIKES_URL   = "https://api.orats.io/datav2/hist/strikes"
SUMM_URL      = "https://api.orats.io/datav2/hist/summaries"
MONIM_URL     = "https://api.orats.io/datav2/hist/monies/implied"

STRIKE_FIELDS = ",".join([
    "ticker", "tradeDate", "expirDate", "dte", "strike", "stockPrice",
    "callOpenInterest", "putOpenInterest", "gamma",
])

# Backfill window (API dates, inclusive both ends)
BACKFILL_API_START = dt.date(2025, 1, 2)
BACKFILL_API_END   = dt.date(2025, 10, 23)

# Legacy cleanup API dates (stored as 2025-10-27, 28, 29, 30)
LEGACY_API_DATES = [
    dt.date(2025, 10, 24),  # -> stored 2025-10-27
    dt.date(2025, 10, 27),  # -> stored 2025-10-28
    dt.date(2025, 10, 28),  # -> stored 2025-10-29
    dt.date(2025, 10, 29),  # -> stored 2025-10-30
]

RATE_LIMIT_SEC = 0.5  # sleep between trading days

MATVIEW_NAME = "gex_walls_daily_rounded"


# --- SQL helpers ----------------------------------------------------------
def normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


UPSERT_SQL = text("""
    INSERT INTO orats_oi_gamma (
        ticker, trade_date, expir_date, dte, strike, stock_price,
        call_oi, put_oi, gamma, gex_call, gex_put,
        short_rate, div_yield, discounted_level
    )
    VALUES (
        :ticker, :trade_date, :expir_date, :dte, :strike, :stock_price,
        :call_oi, :put_oi, :gamma, :gex_call, :gex_put,
        :short_rate, :div_yield, :discounted_level
    )
    ON CONFLICT (ticker, trade_date, expir_date, strike)
    DO UPDATE SET
        dte              = EXCLUDED.dte,
        stock_price      = EXCLUDED.stock_price,
        call_oi          = EXCLUDED.call_oi,
        put_oi           = EXCLUDED.put_oi,
        gamma            = EXCLUDED.gamma,
        gex_call         = EXCLUDED.gex_call,
        gex_put          = EXCLUDED.gex_put,
        short_rate       = EXCLUDED.short_rate,
        div_yield        = EXCLUDED.div_yield,
        discounted_level = EXCLUDED.discounted_level
""")

DELETE_TRADE_DATE_SQL = text("""
    DELETE FROM orats_oi_gamma
    WHERE ticker = :ticker AND trade_date = :trade_date
""")


# --- ORATS API helpers ----------------------------------------------------
def _get(session, url, token, params):
    q = dict(params)
    q["token"] = token
    return session.get(url, params=q, timeout=120)


def fetch_monies_map(session, token, ticker, api_date):
    """{expirDate_str: (risk_free_rate, yield_rate)}"""
    r = _get(session, MONIM_URL, token, {
        "ticker": ticker,
        "tradeDate": api_date.isoformat(),
        "fields": "ticker,tradeDate,expirDate,riskFreeRate,yieldRate",
    })
    if r.status_code >= 400:
        return {}
    res = {}
    for row in r.json().get("data", []):
        expd = row.get("expirDate")
        if expd:
            res[expd] = (row.get("riskFreeRate"), row.get("yieldRate"))
    return res


def fetch_rf30(session, token, ticker, api_date):
    r = _get(session, SUMM_URL, token, {
        "ticker": ticker,
        "tradeDate": api_date.isoformat(),
        "fields": "ticker,tradeDate,riskFree30",
    })
    if r.status_code >= 400:
        return None
    d = r.json().get("data", [])
    return d[0].get("riskFree30") if d else None


def fetch_eod_strikes(session, token, ticker, api_date):
    r = _get(session, STRIKES_URL, token, {
        "ticker": ticker,
        "tradeDate": api_date.isoformat(),
        "fields": STRIKE_FIELDS,
    })
    if r.status_code == 401:
        raise RuntimeError("401 from strikes. Check token / entitlement.")
    if r.status_code == 404:
        return []  # no data for this date (e.g., holiday we didn't filter)
    r.raise_for_status()
    return r.json().get("data", [])


# --- Computation helpers --------------------------------------------------
def compute_gex(S, gamma, oi):
    return (gamma or 0.0) * (S or 0.0) ** 2 * (oi or 0) * CONTRACT_MULTIPLIER


def compute_discounted_level(strike, dte, short_rate, div_yield):
    if strike is None or dte is None or short_rate is None or div_yield is None:
        return None
    t = (int(dte) + 1) / 252.0
    return float(strike) * exp((float(short_rate) - float(div_yield)) * t)


def parse_iso_date(s):
    try:
        return dt.date.fromisoformat(str(s)[:10]) if s else None
    except Exception:
        return None


def next_business_day(d: dt.date) -> dt.date:
    nd = d + dt.timedelta(days=1)
    if nd.weekday() == 5: nd += dt.timedelta(days=2)   # Sat -> Mon
    elif nd.weekday() == 6: nd += dt.timedelta(days=1) # Sun -> Mon
    return nd


# --- Core day-processor ---------------------------------------------------
def process_one_day(session, token, api_date: dt.date, store_date: dt.date):
    """Fetch one API date's data, build rows list ready for upsert.
    Returns (rows, log_summary_str) or (None, reason) if skipped."""
    m_spx = fetch_monies_map(session, token, "SPX", api_date)
    m_spy = fetch_monies_map(session, token, "SPY", api_date)
    rf30  = (fetch_rf30(session, token, "SPX", api_date)
             or fetch_rf30(session, token, "SPY", api_date))

    data = fetch_eod_strikes(session, token, TICKER, api_date)
    if DTE_MAX is not None:
        data = [d for d in data if d.get("dte") is None or int(d["dte"]) <= DTE_MAX]

    if not data:
        return None, "no strike rows returned"

    rows = []
    for d in data:
        expd_s = d.get("expirDate")
        expd   = parse_iso_date(expd_s)

        # SPX/SPY fallback for short_rate and div_yield - same as job_orats_eod.py
        sr, dy = (None, None)
        if expd_s and expd_s in m_spx:
            sr, dy = m_spx[expd_s]
        if (dy in (None, 0, 0.0)) and expd_s and expd_s in m_spy:
            sr2, dy2 = m_spy[expd_s]
            if sr is None: sr = sr2
            if dy2 not in (None, 0, 0.0): dy = dy2
        if sr is None: sr = rf30
        if dy is None: dy = 0.0

        S, gamma = d.get("stockPrice"), d.get("gamma")
        coi, poi = d.get("callOpenInterest"), d.get("putOpenInterest")
        gex_call = compute_gex(S, gamma, coi)
        gex_put  = compute_gex(S, gamma, poi)

        eff_dte  = (expd - store_date).days if expd else d.get("dte")
        disc_lvl = compute_discounted_level(d.get("strike"), eff_dte, sr, dy)

        rows.append({
            "ticker":          d.get("ticker"),
            "trade_date":      store_date,
            "expir_date":      expd,
            "dte":             eff_dte,
            "strike":          d.get("strike"),
            "stock_price":     S,
            "call_oi":         coi,
            "put_oi":          poi,
            "gamma":           gamma,
            "gex_call":        gex_call,
            "gex_put":         gex_put,
            "short_rate":      sr,
            "div_yield":       dy,
            "discounted_level": disc_lvl,
        })

    return rows, f"spx_expir={len(m_spx)} spy_expir={len(m_spy)} rf30={rf30} rows={len(rows)}"


def upsert_day(engine, store_date: dt.date, rows, delete_first: bool = False):
    """Insert/update rows for a single stored trade_date.
    If delete_first, clear the day's existing rows first (needed for legacy
    cleanup where strike set may have changed)."""
    with engine.begin() as conn:
        if delete_first:
            conn.execute(DELETE_TRADE_DATE_SQL,
                         {"ticker": TICKER, "trade_date": store_date})
        conn.execute(UPSERT_SQL, rows)


def existing_stored_dates(engine, date_from: dt.date, date_to: dt.date) -> set:
    """Return set of stored trade_dates that already have rows in the range.
    Used for resume-on-restart."""
    q = text("""
        SELECT DISTINCT trade_date FROM orats_oi_gamma
        WHERE ticker = :ticker
          AND trade_date BETWEEN :d_from AND :d_to
    """)
    with engine.begin() as conn:
        rows = conn.execute(q, {
            "ticker": TICKER, "d_from": date_from, "d_to": date_to,
        }).fetchall()
    return {r[0] for r in rows}


def trading_days(api_start: dt.date, api_end: dt.date) -> list:
    """All NYSE trading sessions between api_start and api_end, inclusive."""
    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=api_start, end_date=api_end)
    return [d.date() for d in sched.index]


def refresh_matview(engine, name: str):
    with engine.begin() as conn:
        conn.execute(text(f"REFRESH MATERIALIZED VIEW {name}"))


# --- Main -----------------------------------------------------------------
def run_backfill_phase(session, token, engine, resume: bool) -> dict:
    """Main backfill loop. Returns stats dict."""
    api_days = trading_days(BACKFILL_API_START, BACKFILL_API_END)
    # Compute stored dates for resume lookup
    day_pairs = [(d, next_business_day(d)) for d in api_days]

    already_done = set()
    if resume:
        stored_from = min(s for _, s in day_pairs)
        stored_to   = max(s for _, s in day_pairs)
        already_done = existing_stored_dates(engine, stored_from, stored_to)
        # Exclude legacy cleanup dates so we don't accidentally skip those
        already_done -= {next_business_day(d) for d in LEGACY_API_DATES}

    total = len(day_pairs)
    stats = {"processed": 0, "skipped_resume": 0, "skipped_empty": 0,
             "total_rows": 0, "failures": []}
    t_start = time.time()

    print(f"\n  Total trading days in backfill window: {total}")
    if resume and already_done:
        print(f"  Resume mode: {len(already_done)} stored dates already present, will skip")
    print(f"  Rate limit: {RATE_LIMIT_SEC}s between days")

    for i, (api_date, store_date) in enumerate(day_pairs, start=1):
        progress = f"[{i:>3}/{total}]"

        if store_date in already_done:
            stats["skipped_resume"] += 1
            if i % 25 == 0 or i == total:
                print(f"  {progress} api={api_date} store={store_date}  SKIP (already in DB)")
            continue

        try:
            rows, msg = process_one_day(session, token, api_date, store_date)
            if rows is None:
                stats["skipped_empty"] += 1
                print(f"  {progress} api={api_date} store={store_date}  SKIP ({msg})")
            else:
                # delete_first=True matches job_orats_eod.py behavior
                upsert_day(engine, store_date, rows, delete_first=True)
                stats["processed"]  += 1
                stats["total_rows"] += len(rows)
                elapsed = time.time() - t_start
                eta = (elapsed / i) * (total - i) if i > 0 else 0
                print(f"  {progress} api={api_date} store={store_date}  OK  "
                      f"{msg}  eta={eta:>4.0f}s")
        except Exception as e:
            stats["failures"].append((api_date, store_date, str(e)))
            print(f"  {progress} api={api_date} store={store_date}  FAIL  {e}")

        time.sleep(RATE_LIMIT_SEC)

    return stats


def run_legacy_cleanup(session, token, engine) -> dict:
    """Re-pull the 4 legacy dates to fix old-code dte/discounted_level."""
    stats = {"processed": 0, "failures": []}
    print(f"\n  Legacy cleanup dates: {len(LEGACY_API_DATES)}")
    for api_date in LEGACY_API_DATES:
        store_date = next_business_day(api_date)
        try:
            rows, msg = process_one_day(session, token, api_date, store_date)
            if rows is None:
                print(f"  api={api_date} store={store_date}  SKIP ({msg})")
                continue
            # delete_first=True to clear any stale rows before re-inserting
            upsert_day(engine, store_date, rows, delete_first=True)
            stats["processed"] += 1
            print(f"  api={api_date} store={store_date}  OK  {msg}")
        except Exception as e:
            stats["failures"].append((api_date, store_date, str(e)))
            print(f"  api={api_date} store={store_date}  FAIL  {e}")
        time.sleep(RATE_LIMIT_SEC)
    return stats


def confirm(prompt: str) -> bool:
    r = input(f"\n{prompt} [type 'yes' to proceed]: ").strip().lower()
    return r == "yes"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default=".env", help="Path to .env file")
    p.add_argument("--dry-run", action="store_true", help="Show plan only")
    p.add_argument("--yes", action="store_true", help="Skip confirmation")
    p.add_argument("--skip-backfill", action="store_true",
                   help="Skip main backfill; only run legacy cleanup")
    p.add_argument("--skip-legacy", action="store_true",
                   help="Skip legacy cleanup; only run main backfill")
    p.add_argument("--no-resume", action="store_true",
                   help="Don't skip dates that already have rows in the DB")
    args = p.parse_args()

    env_path = Path(args.env)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    token  = os.getenv("ORATS_TOKEN") or os.getenv("ORATS_API_KEY")
    if not db_url:
        sys.exit("ERROR: DATABASE_URL not found.")
    if not token:
        sys.exit("ERROR: ORATS_TOKEN (or ORATS_API_KEY) not found.")

    engine = create_engine(normalize_db_url(db_url), pool_pre_ping=True)

    # --- Plan summary ---
    api_days = trading_days(BACKFILL_API_START, BACKFILL_API_END)
    est_sec  = len(api_days) * (RATE_LIMIT_SEC + 2.0)  # ~2s per-call budget

    print("=" * 92)
    print("ORATS EOD Strikes Backfill for orats_oi_gamma")
    print("=" * 92)
    print(f"Ticker              : {TICKER}")
    print(f"Token (...{token[-4:]})")
    print(f"Backfill API range  : {BACKFILL_API_START} to {BACKFILL_API_END}")
    print(f"Stored as           : {next_business_day(BACKFILL_API_START)} "
          f"to {next_business_day(BACKFILL_API_END)}")
    print(f"Trading days        : {len(api_days)}")
    print(f"Legacy cleanup dates: {len(LEGACY_API_DATES)}  "
          f"{[d.isoformat() for d in LEGACY_API_DATES]}")
    print(f"Estimated runtime   : ~{est_sec / 60:.1f} min (mostly API wait)")
    print(f"Cost                : $0 (included in ORATS subscription)")
    print()
    print("Phases:")
    print(f"  1. Main backfill       {'[SKIP]' if args.skip_backfill else '[RUN]'}")
    print(f"  2. Legacy cleanup      {'[SKIP]' if args.skip_legacy else '[RUN]'}")
    print(f"  3. REFRESH matview {MATVIEW_NAME}")

    if args.dry_run:
        print("\n[dry-run] No changes made. Exiting.")
        return

    if not args.yes:
        if not confirm("Proceed?"):
            print("Aborted.")
            return

    session = requests.Session()
    backfill_stats = None
    legacy_stats   = None

    try:
        if not args.skip_backfill:
            print("\n--- Phase 1: Main backfill ---")
            backfill_stats = run_backfill_phase(
                session, token, engine, resume=not args.no_resume,
            )

        if not args.skip_legacy:
            print("\n--- Phase 2: Legacy cleanup ---")
            legacy_stats = run_legacy_cleanup(session, token, engine)

        print(f"\n--- Phase 3: Refresh {MATVIEW_NAME} ---")
        t0 = time.time()
        refresh_matview(engine, MATVIEW_NAME)
        print(f"  Done in {time.time() - t0:.1f}s")

    finally:
        session.close()

    # --- Summary ---
    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)

    if backfill_stats is not None:
        bs = backfill_stats
        print(f"  Main backfill:")
        print(f"    processed      : {bs['processed']}")
        print(f"    skipped_resume : {bs['skipped_resume']}")
        print(f"    skipped_empty  : {bs['skipped_empty']}")
        print(f"    total_rows     : {bs['total_rows']:,}")
        if bs["failures"]:
            print(f"    FAILURES       : {len(bs['failures'])}")
            for api, store, err in bs["failures"][:10]:
                print(f"      {api} -> {store}: {err[:120]}")

    if legacy_stats is not None:
        ls = legacy_stats
        print(f"  Legacy cleanup:")
        print(f"    processed : {ls['processed']}")
        if ls["failures"]:
            print(f"    FAILURES  : {len(ls['failures'])}")
            for api, store, err in ls["failures"]:
                print(f"      {api} -> {store}: {err[:120]}")

    # Quick post-run row count
    with engine.begin() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM orats_oi_gamma WHERE ticker = :t"
        ), {"t": TICKER}).scalar()
        earliest = conn.execute(text(
            "SELECT MIN(trade_date) FROM orats_oi_gamma WHERE ticker = :t"
        ), {"t": TICKER}).scalar()
        latest = conn.execute(text(
            "SELECT MAX(trade_date) FROM orats_oi_gamma WHERE ticker = :t"
        ), {"t": TICKER}).scalar()
        mv_count = conn.execute(text(
            f"SELECT COUNT(*) FROM {MATVIEW_NAME}"
        )).scalar()

    print(f"\n  orats_oi_gamma (ticker={TICKER}):")
    print(f"    total rows  : {total:,}")
    print(f"    date range  : {earliest} to {latest}")
    print(f"  {MATVIEW_NAME} rows: {mv_count:,}")
    print()


if __name__ == "__main__":
    main()
