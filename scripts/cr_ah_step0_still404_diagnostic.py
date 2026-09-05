"""CR-AH Step 0 — Still-404 deep diagnostic.

READ-ONLY. No writes, no re-backfill.

For the 13 D-filter-miss dates that returned HTTP 404 at correct expiry (previous
session), this script:

1. CONTRADICTION CHECK: queries orats_monies_minute for any surface/smile row
   at (trade_date, correct_expiry) — if present, chain exists and 404 is a
   REQUEST bug (wrong strike / wrong parameter), not ORATS absence.

2. STRIKE-GRID TEST: hits ORATS chain endpoint for (trade_date, correct_expiry),
   lists actual strikes near the drift_target. Classifies each date:
     OFF_GRID    — proposed strike not listed but nearby strikes exist
     EXPIRY_NOT_LISTED — ORATS has no chain at all for that expiry
     GENUINE     — monies also missing AND chain confirms no contract

3. For the 17 recoverable dates: confirms proposed strike IS on ORATS's listed
   grid for the correct expiry (not just HTTP 200 on the option endpoint, but
   actually on the 5pt grid).

Usage:
    python -u scripts/cr_ah_step0_still404_diagnostic.py
"""
from __future__ import annotations

import os
import sys
import time as _time
from datetime import date, timedelta
from pathlib import Path

# ── ENV SETUP ─────────────────────────────────────────────────────────────────
def _find_dotenv() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / ".env"
        if candidate.exists():
            return candidate
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

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import urllib.request
import urllib.parse
import json
import psycopg2

# ── CONFIG ────────────────────────────────────────────────────────────────────
ORATS_API_KEY = os.environ.get("ORATS_API_KEY", "")
ORATS_BASE = "https://api.orats.io"
CHAIN_PATH = "/datav2/hist/live/one-minute/strikes/chain"
OPTION_PATH = "/datav2/hist/live/one-minute/strikes/option"

DB_URL = os.environ.get("DATABASE_URL", "").strip()
if not DB_URL:
    sys.exit("ERROR: DATABASE_URL not set")
if not ORATS_API_KEY:
    sys.exit("ERROR: ORATS_API_KEY not set")

# ── DATES ─────────────────────────────────────────────────────────────────────

# 13 filter-miss dates that still 404'd at correct expiry (D-filter-miss, not recoverable)
STILL_404 = [
    # (trade_date, correct_expiry, cell_band/partition)
    (date(2024, 5, 22),  date(2024, 6, 13),  "far/train"),
    (date(2024, 7, 1),   date(2024, 7, 23),  "far/train"),
    (date(2025, 1, 6),   date(2025, 1, 29),  "far/train"),
    (date(2025, 2, 4),   date(2025, 2, 26),  "far/train"),
    (date(2025, 2, 11),  date(2025, 3, 5),   "far/train"),
    (date(2025, 5, 12),  date(2025, 6, 3),   "far/train"),
    (date(2025, 6, 4),   date(2025, 6, 26),  "mid/train"),
    (date(2025, 8, 18),  date(2025, 9, 9),   "mid/holdout"),
    (date(2025, 8, 25),  date(2025, 9, 16),  "mid/holdout"),
    (date(2026, 1, 6),   date(2026, 1, 28),  "far/holdout"),
    (date(2026, 1, 7),   date(2026, 1, 29),  "mid/holdout"),
    (date(2026, 1, 13),  date(2026, 2, 4),   "mid/holdout"),
    (date(2026, 2, 3),   date(2026, 2, 25),  "mid/holdout"),
]

# 17 recoverable dates (correct expiry confirmed HTTP 200 at proposed strike)
RECOVERABLE = [
    (date(2023, 5, 18),  date(2023, 6, 9),   "near/train"),
    (date(2024, 11, 12), date(2024, 12, 4),  "near/train"),
    (date(2025, 6, 11),  date(2025, 7, 3),   "near/train"),
    (date(2026, 1, 29),  date(2026, 2, 20),  "near/holdout"),
    (date(2026, 5, 21),  date(2026, 6, 12),  "near/holdout"),
    (date(2023, 5, 11),  date(2023, 6, 2),   "mid/train"),
    (date(2024, 5, 23),  date(2024, 6, 14),  "mid/train"),
    (date(2025, 8, 19),  date(2025, 9, 10),  "mid/holdout"),
    (date(2026, 2, 12),  date(2026, 3, 6),   "mid/holdout"),
    (date(2026, 5, 19),  date(2026, 6, 10),  "mid/holdout"),
    (date(2023, 5, 25),  date(2023, 6, 16),  "far/train"),
    (date(2024, 6, 12),  date(2024, 7, 5),   "far/train"),
    (date(2024, 6, 24),  date(2024, 7, 16),  "far/train"),
    (date(2024, 8, 15),  date(2024, 9, 6),   "far/train"),
    (date(2025, 8, 21),  date(2025, 9, 12),  "far/holdout"),
    (date(2025, 12, 9),  date(2025, 12, 31), "far/holdout"),
    (date(2025, 12, 17), date(2026, 1, 9),   "far/holdout"),
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def round_to_5pt(price: float) -> int:
    return round(price / 5) * 5

def orats_get(path: str, params: dict) -> dict | None:
    """GET from ORATS API. Returns parsed JSON or None on error."""
    params["token"] = ORATS_API_KEY
    url = ORATS_BASE + path + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise

def get_chain(trade_date: date, expiry: date) -> dict | None:
    """Fetch ORATS chain for (trade_date, expiry). Returns data dict or None."""
    params = {
        "ticker": "SPX",
        "expirDate": expiry.isoformat(),
        "tradeDate": trade_date.isoformat(),
    }
    return orats_get(CHAIN_PATH, params)

def get_option(trade_date: date, expiry: date, strike: int, side: str = "C") -> int | None:
    """Hit ORATS option endpoint. Returns HTTP status (200 or 404)."""
    # Build ORATS ticker: SPX + YYMMDD + strike*1000 (padded to 8 digits)
    exp_str = expiry.strftime("%y%m%d")
    strike_str = str(int(strike * 1000)).zfill(8)
    ticker = f"SPX{exp_str}{strike_str}"
    params = {
        "ticker": ticker,
        "tradeDate": trade_date.isoformat(),
    }
    result = orats_get(OPTION_PATH, params)
    return 200 if result else 404

def get_drift_target(conn, trade_date: date, ticker: str = "SPX") -> float | None:
    """Pull drift_target from bt_daily_features via the landscape payload."""
    # Use the same SQL as the backfill: _LANDSCAPE_ROW_SQL equivalent
    row = conn.execute("""
        SELECT landscape_rows, spot
        FROM bt_daily_features
        WHERE ticker = %s AND trade_date = %s
          AND feature_version = (
              SELECT MAX(feature_version) FROM bt_daily_features
              WHERE ticker = %s AND trade_date = %s
              AND regime_at_classification = 'magnet-above'
          )
          AND regime_at_classification = 'magnet-above'
        LIMIT 1
    """, (ticker, trade_date.isoformat(), ticker, trade_date.isoformat())).fetchone()

    if not row:
        return None

    # The landscape_rows is JSON containing the drift_target in regime block
    # Actually, let's query bt_daily_outcomes which has the signal info
    return None

def get_drift_from_features(conn, trade_date: date) -> float | None:
    """Query the day_features to get drift_target."""
    # Try bt_daily_features payload
    row = conn.execute("""
        SELECT f.payload
        FROM bt_daily_features f
        WHERE f.ticker = 'SPX'
          AND f.trade_date = %s
          AND f.regime_at_classification = 'magnet-above'
        ORDER BY f.feature_version DESC
        LIMIT 1
    """, (trade_date.isoformat(),)).fetchone()

    if row and row[0]:
        try:
            payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            regime = payload.get("regime") or {}
            dt = regime.get("drift_target")
            if dt is not None:
                return float(dt)
        except Exception:
            pass
    return None

def check_monies(conn, trade_date: date, expiry: date) -> bool:
    """Check if orats_monies_minute has any row for this expiry on/near this trade_date."""
    # Check ±5 days around trade_date for this expiry
    from_dt = date(trade_date.year, trade_date.month, trade_date.day) - timedelta(days=5)
    to_dt   = date(trade_date.year, trade_date.month, trade_date.day) + timedelta(days=5)
    row = conn.execute("""
        SELECT COUNT(*) FROM orats_monies_minute
        WHERE ticker = 'SPX'
          AND expir_date_d = %s
          AND trade_date_d BETWEEN %s AND %s
        LIMIT 1
    """, (expiry.isoformat(), from_dt.isoformat(), to_dt.isoformat())).fetchone()
    return row and row[0] > 0

def check_monies_any_on_trade_date(conn, trade_date: date) -> int:
    """Count orats_monies_minute rows for any expiry on this exact trade_date."""
    row = conn.execute("""
        SELECT COUNT(*) FROM orats_monies_minute
        WHERE ticker = 'SPX' AND trade_date_d = %s
    """, (trade_date.isoformat(),)).fetchone()
    return row[0] if row else 0

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True

    print("=" * 80)
    print("CR-AH STILL-404 DEEP DIAGNOSTIC")
    print("=" * 80)

    # ── PART 1: The 13 still-404 dates ───────────────────────────────────────
    print("\n## PART 1: 13 STILL-404 DATES — contradiction check + strike-grid test")
    print("-" * 80)

    results_13 = []

    for trade_date, correct_expiry, cell in STILL_404:
        print(f"\n[{cell}] {trade_date}  correct_expiry={correct_expiry}")

        # 1a. Get drift_target from features
        drift = get_drift_from_features(conn, trade_date)
        if drift is None:
            # Try querying bt_signals or other tables
            print(f"  drift_target: NOT FOUND in bt_daily_features")
            short_strike = None
        else:
            short_strike = round_to_5pt(drift)
            long_strike  = short_strike + 10
            print(f"  drift_target={drift:.1f}  short_strike={short_strike}  long_strike={long_strike}")

        # 1b. Contradiction check: orats_monies_minute for (trade_date, correct_expiry)
        monies_hit = check_monies(conn, trade_date, correct_expiry)
        monies_count = check_monies_any_on_trade_date(conn, trade_date)
        print(f"  orats_monies_minute @ expiry={correct_expiry}: {'FOUND' if monies_hit else 'NOT FOUND'}")
        print(f"  orats_monies_minute @ trade_date={trade_date} (any expiry): {monies_count} rows")

        # 1c. Chain endpoint: get all listed strikes for this expiry on trade_date
        print(f"  Fetching ORATS chain for expiry={correct_expiry}...", end="", flush=True)
        _time.sleep(0.3)  # gentle rate limiting
        chain_data = get_chain(trade_date, correct_expiry)

        chain_strikes = []
        expiry_listed = False
        if chain_data:
            rows = chain_data.get("data", [])
            if rows:
                expiry_listed = True
                chain_strikes = sorted(set(
                    int(float(r.get("strike", 0)))
                    for r in rows
                    if r.get("strike") and r.get("callVolume") is not None
                ))
                print(f" {len(chain_strikes)} strikes found")
            else:
                print(" empty data array")
        else:
            print(" HTTP 404 (no chain)")

        # 1d. Classify: OFF_GRID / EXPIRY_NOT_LISTED / GENUINE
        classification = "?"
        nearest_below = None
        nearest_above = None

        if not expiry_listed:
            # Check if monies has it → contradiction
            if monies_hit:
                classification = "CONTRADICTION"
                print(f"  *** CONTRADICTION: monies has surface but chain 404'd ***")
            else:
                classification = "EXPIRY_NOT_LISTED"
        else:
            # Chain has data — is our strike on the grid?
            if short_strike is not None:
                if short_strike in chain_strikes:
                    classification = "STRIKE_PRESENT_BUT_404"
                    print(f"  *** ANOMALY: strike {short_strike} IS in chain but option endpoint 404'd ***")
                else:
                    classification = "OFF_GRID"
                    below = [s for s in chain_strikes if s <= short_strike]
                    above = [s for s in chain_strikes if s >= short_strike]
                    nearest_below = max(below) if below else None
                    nearest_above = min(above) if above else None
            else:
                classification = "OFF_GRID_UNKNOWN_STRIKE"

        # 1e. Print chain details
        if chain_strikes:
            if short_strike is not None:
                nearby = [s for s in chain_strikes
                         if abs(s - short_strike) <= 100]
                print(f"  Chain strikes near target ({short_strike}±100): {nearby if nearby else chain_strikes[:20]}")
                if classification == "OFF_GRID":
                    print(f"  Nearest listed: below={nearest_below}  above={nearest_above}")
                    gap_b = (short_strike - nearest_below) if nearest_below else None
                    gap_a = (nearest_above - short_strike) if nearest_above else None
                    print(f"  Off-grid gap: below={gap_b}pt  above={gap_a}pt")
            else:
                print(f"  Chain strikes sample: {chain_strikes[:30]}")
        elif expiry_listed:
            print(f"  Chain returned data but no call strikes extracted")

        print(f"  >>> CLASSIFICATION: {classification}")

        results_13.append({
            "trade_date": trade_date,
            "correct_expiry": correct_expiry,
            "cell": cell,
            "drift": drift,
            "short_strike": short_strike,
            "monies_hit": monies_hit,
            "expiry_listed": expiry_listed,
            "chain_strikes": chain_strikes,
            "nearest_below": nearest_below,
            "nearest_above": nearest_above,
            "classification": classification,
        })

        _time.sleep(0.3)

    # ── PART 2: The 17 recoverable dates — validate strike is on grid ─────────
    print("\n\n## PART 2: 17 RECOVERABLE DATES — validate strike on listed grid")
    print("-" * 80)

    recoverable_results = []

    for trade_date, correct_expiry, cell in RECOVERABLE:
        print(f"\n[{cell}] {trade_date}  correct_expiry={correct_expiry}")

        drift = get_drift_from_features(conn, trade_date)
        if drift is None:
            print(f"  drift_target: NOT FOUND")
            recoverable_results.append({
                "trade_date": trade_date, "cell": cell,
                "validation": "NO_DRIFT", "strike": None
            })
            continue

        short_strike = round_to_5pt(drift)
        long_strike  = short_strike + 10
        print(f"  drift_target={drift:.1f}  short={short_strike}  long={long_strike}", end="", flush=True)

        _time.sleep(0.3)
        chain_data = get_chain(trade_date, correct_expiry)
        chain_strikes = []
        if chain_data:
            rows = chain_data.get("data", [])
            chain_strikes = sorted(set(
                int(float(r.get("strike", 0)))
                for r in rows
                if r.get("strike") and r.get("callVolume") is not None
            ))

        short_on_grid = short_strike in chain_strikes
        long_on_grid  = long_strike in chain_strikes

        if short_on_grid and long_on_grid:
            status = "CLEAN"
        elif short_on_grid and not long_on_grid:
            status = "LONG_OFF_GRID"
        elif not short_on_grid and long_on_grid:
            status = "SHORT_OFF_GRID"
        else:
            status = "BOTH_OFF_GRID"

        print(f"  short_on_grid={short_on_grid}  long_on_grid={long_on_grid}  [{status}]")
        if not (short_on_grid and long_on_grid):
            nearby = [s for s in chain_strikes if abs(s - short_strike) <= 100]
            print(f"  Chain strikes near target: {nearby}")

        recoverable_results.append({
            "trade_date": trade_date, "correct_expiry": correct_expiry,
            "cell": cell, "drift": drift, "short_strike": short_strike,
            "long_strike": long_strike, "short_on_grid": short_on_grid,
            "long_on_grid": long_on_grid, "validation": status,
        })
        _time.sleep(0.3)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n### 13 still-404 dates:")
    classifications = {}
    for r in results_13:
        c = r["classification"]
        classifications.setdefault(c, []).append(r)
        print(f"  {r['trade_date']}  [{r['cell']}]  expiry={r['correct_expiry']}  "
              f"strike={r['short_strike']}  monies={r['monies_hit']}  "
              f"expiry_listed={r['expiry_listed']}  → {c}"
              + (f"  (nearest: {r['nearest_below']}↓ {r['nearest_above']}↑)"
                 if c == "OFF_GRID" and r['nearest_below'] else ""))

    print(f"\nClassification breakdown:")
    for cls, items in sorted(classifications.items()):
        print(f"  {cls}: {len(items)}")

    off_grid = classifications.get("OFF_GRID", [])
    if off_grid:
        print(f"\n  OFF_GRID dates (recoverable by snapping strike):")
        for r in off_grid:
            snap = r['nearest_below'] if r['nearest_below'] else r['nearest_above']
            print(f"    {r['trade_date']} [{r['cell']}]: proposed={r['short_strike']} "
                  f"→ snap to {snap} (nearest listed)")

    print(f"\n### 17 recoverable dates (validation):")
    clean = [r for r in recoverable_results if r["validation"] == "CLEAN"]
    not_clean = [r for r in recoverable_results if r["validation"] != "CLEAN"]
    print(f"  CLEAN (both legs on grid): {len(clean)}")
    print(f"  ISSUES: {len(not_clean)}")
    for r in not_clean:
        print(f"    {r['trade_date']} [{r['cell']}] {r['validation']} short={r.get('short_strike')} long={r.get('long_strike')}")

    total_off_grid = len(off_grid)
    total_genuine = len(classifications.get("GENUINE", [])) + \
                    len(classifications.get("EXPIRY_NOT_LISTED", []))
    print(f"\n### Revised recovery estimate:")
    print(f"  Filter-miss recoverable (confirmed 17): 17")
    print(f"  Off-grid (strike-snap recoverable from 13): {total_off_grid}")
    print(f"  Expiry-not-listed / genuinely absent: {total_genuine}")
    print(f"  Total potential recovery: {17 + total_off_grid}")
    print(f"  Remaining D (unrecoverable): {52 - 17 - total_off_grid}")

    conn.close()

if __name__ == "__main__":
    main()
