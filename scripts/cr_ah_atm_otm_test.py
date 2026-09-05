"""CR-AH ATM-vs-OTM coverage test — READ-ONLY.

For each of the 13 still-404 dates, hit the ORATS option endpoint for:
  (a) the proposed OTM strike (known 404), and
  (b) an ATM strike on the same trade_date + correct expiry.

If ATM returns bars + OTM 404s → confirms "per-minute OTM coverage gap" (genuine).
If ATM ALSO 404s → whole chain absent → possible request/format issue, stop before re-backfill.

Run as: PYTHONUNBUFFERED=1 python3 -u scripts/cr_ah_atm_otm_test.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ── ENV ───────────────────────────────────────────────────────────────────────

def _find_dotenv() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        c = current / ".env"
        if c.exists():
            return c
        current = current.parent
    return None

_env_path = _find_dotenv()
if _env_path:
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = bak_url

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import psycopg2
from packages.shared.options_cache.http_client import get_csv, OratsPermanentError
from packages.shared.options_cache.opra import format_opra, opra_to_orats_ticker

# ── THE 13 STILL-404 DATES ────────────────────────────────────────────────────
# (trade_date, correct_expiry, proposed_otm_short_strike)

STILL_404 = [
    (date(2024,  5, 22), date(2024,  6, 13), 5405),
    (date(2024,  7,  1), date(2024,  7, 23), 5585),
    (date(2025,  1,  6), date(2025,  1, 29), 6065),
    (date(2025,  2,  4), date(2025,  2, 26), 6205),
    (date(2025,  2, 11), date(2025,  3,  5), 6185),
    (date(2025,  5, 12), date(2025,  6,  3), 5805),
    (date(2025,  6,  4), date(2025,  6, 26), 6055),
    (date(2025,  8, 18), date(2025,  9,  9), 6505),
    (date(2025,  8, 25), date(2025,  9, 16), 6505),
    (date(2026,  1,  6), date(2026,  1, 28), 7005),
    (date(2026,  1,  7), date(2026,  1, 29), 7005),
    (date(2026,  1, 13), date(2026,  2,  4), 7045),
    (date(2026,  2,  3), date(2026,  2, 25), 7055),
]

_PT = ZoneInfo("America/Los_Angeles")
_ET = ZoneInfo("America/New_York")


def format_trade_date_param(d: date) -> str:
    """Format RTH range (06:30–13:00 PT) as ORATS tradeDate param (ET, YYYYMMDDHHMM)."""
    start_pt = datetime(d.year, d.month, d.day, 6, 30, tzinfo=_PT)
    end_pt   = datetime(d.year, d.month, d.day, 13, 0, tzinfo=_PT)
    start_et = start_pt.astimezone(_ET).strftime("%Y%m%d%H%M")
    end_et   = end_pt.astimezone(_ET).strftime("%Y%m%d%H%M")
    return f"{start_et},{end_et}"


def test_orats_option(trade_date: date, expiry_date: date, strike: int) -> tuple[str, int]:
    """
    Hit the ORATS option endpoint for (trade_date, expiry_date, strike).
    Returns (status, row_count): status is 'HTTP 200' or 'HTTP 404' or 'ERROR:...'.
    """
    opra = format_opra("SPX", expiry_date, "C", float(strike))
    ticker = opra_to_orats_ticker(opra)
    trade_date_param = format_trade_date_param(trade_date)

    try:
        csv_text = get_csv(
            "/datav2/hist/live/one-minute/strikes/option",
            {"ticker": ticker, "tradeDate": trade_date_param},
        )
        # CSV has a header row; count data rows
        data_rows = [r for r in csv_text.strip().splitlines() if r and not r.startswith("ticker")]
        return ("HTTP 200", len(data_rows))
    except OratsPermanentError:
        return ("HTTP 404", 0)
    except Exception as e:
        return (f"ERROR: {e}", 0)


def get_spot(conn, trade_date: date, expiry_date: date) -> float | None:
    """Get SPX spot price on trade_date from orats_monies_minute."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT stock_price FROM orats_monies_minute
        WHERE trade_date = %s AND expir_date = %s
        ORDER BY trade_date ASC LIMIT 1
        """,
        (str(trade_date), str(expiry_date)),
    )
    row = cur.fetchone()
    if row:
        return float(row[0])

    # Fallback: any row on that trade_date (any expiry)
    cur.execute(
        "SELECT stock_price FROM orats_monies_minute WHERE trade_date = %s LIMIT 1",
        (str(trade_date),),
    )
    row = cur.fetchone()
    return float(row[0]) if row else None


def main() -> None:
    db_url = bak_url.replace("postgresql+psycopg://", "postgresql://").split("?")[0]
    conn = psycopg2.connect(db_url)

    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print("CR-AH ATM-vs-OTM COVERAGE TEST — 13 STILL-404 DATES", flush=True)
    print(sep, flush=True)

    results = []

    for trade_date, expiry_date, otm_strike in STILL_404:
        spot = get_spot(conn, trade_date, expiry_date)
        if spot is None:
            print(f"\n{trade_date}: NO SPOT — skipping", flush=True)
            results.append({
                "trade_date": trade_date,
                "expiry_date": expiry_date,
                "otm_strike": otm_strike,
                "spot": None,
                "atm_strike": None,
                "otm_status": "NO_SPOT",
                "otm_rows": 0,
                "atm_status": "NO_SPOT",
                "atm_rows": 0,
                "verdict": "NO_SPOT",
            })
            continue

        atm_strike = round(spot / 5) * 5

        print(f"\n{trade_date} expiry={expiry_date}", flush=True)
        print(f"  spot={spot:.0f}  ATM_strike={atm_strike}  OTM_strike={otm_strike}", flush=True)

        print(f"  Testing OTM strike {otm_strike}...", flush=True)
        otm_status, otm_rows = test_orats_option(trade_date, expiry_date, otm_strike)
        print(f"  OTM: {otm_status} ({otm_rows} rows)", flush=True)

        # Only test ATM if it's different from OTM (avoid duplicate call if coincidentally same)
        if atm_strike == otm_strike:
            print(f"  ATM=OTM ({atm_strike}) — skipping duplicate ATM call", flush=True)
            atm_status, atm_rows = otm_status, otm_rows
        else:
            print(f"  Testing ATM strike {atm_strike}...", flush=True)
            atm_status, atm_rows = test_orats_option(trade_date, expiry_date, atm_strike)
            print(f"  ATM: {atm_status} ({atm_rows} rows)", flush=True)

        # Classify
        if "200" in otm_status and "200" in atm_status:
            verdict = "BOTH_200 — OTM unexpectedly available (recheck!)"
        elif "404" in otm_status and "200" in atm_status:
            verdict = "CONFIRMED — ATM available, OTM 404: per-minute OTM-sparsity proven"
        elif "404" in otm_status and "404" in atm_status:
            verdict = "BOTH_404 — whole chain absent for this date"
        elif "200" in otm_status and "404" in atm_status:
            verdict = "OTM_ONLY_200 — unusual (OTM ok but ATM 404?)"
        else:
            verdict = f"MIXED: OTM={otm_status} ATM={atm_status}"

        print(f"  VERDICT: {verdict}", flush=True)

        results.append({
            "trade_date": trade_date,
            "expiry_date": expiry_date,
            "spot": spot,
            "atm_strike": atm_strike,
            "otm_strike": otm_strike,
            "otm_status": otm_status,
            "otm_rows": otm_rows,
            "atm_status": atm_status,
            "atm_rows": atm_rows,
            "verdict": verdict,
        })

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}", flush=True)
    print("SUMMARY TABLE", flush=True)
    print(f"{sep}", flush=True)
    print(f"{'Date':<12}  {'Expiry':<12}  {'Spot':>6}  {'ATM':>5}  {'OTM':>5}  {'ATM result':<12}  {'OTM result':<12}  Verdict", flush=True)
    print("-" * 90, flush=True)

    confirmed_count = 0
    both_404_count = 0

    for r in results:
        atm_str = str(r["atm_strike"]) if r["atm_strike"] else "N/A"
        spot_str = f"{r['spot']:.0f}" if r["spot"] else "N/A"
        atm_res = f"{r['atm_status']} ({r['atm_rows']}r)" if r["atm_rows"] else r["atm_status"]
        otm_res = f"{r['otm_status']} ({r['otm_rows']}r)" if r["otm_rows"] else r["otm_status"]
        print(
            f"{str(r['trade_date']):<12}  {str(r['expiry_date']):<12}  {spot_str:>6}  "
            f"{atm_str:>5}  {r['otm_strike']:>5}  {atm_res:<20}  {otm_res:<12}  {r['verdict'][:50]}",
            flush=True,
        )
        if "CONFIRMED" in r["verdict"]:
            confirmed_count += 1
        elif "BOTH_404" in r["verdict"]:
            both_404_count += 1

    print(f"\n{sep}", flush=True)
    print(f"CONFIRMED (ATM ok, OTM 404):       {confirmed_count}", flush=True)
    print(f"BOTH_404 (whole chain absent):      {both_404_count}", flush=True)
    other = len(results) - confirmed_count - both_404_count
    print(f"Other:                              {other}", flush=True)
    print(f"{sep}", flush=True)

    if both_404_count > 0:
        print(
            "\n⚠️  STOP: Some dates have ATM also 404. "
            "Do NOT proceed to re-backfill until this is understood.",
            flush=True,
        )
    elif confirmed_count == len(results):
        print(
            "\n✓ ALL CONFIRMED: ATM available, OTM 404 for all 13 dates. "
            "'Surface broad, per-minute OTM sparse' proven. Safe to proceed with re-backfill.",
            flush=True,
        )
    else:
        print("\nMixed results — review per-date verdicts above.", flush=True)


if __name__ == "__main__":
    main()
