"""CR-AI Step 0b — Live ORATS coverage probe for shorter-DTE options.

The Step 0a cache-check showed 0% coverage for DTE={5,7,10,21} — expected, since
those options have never been fetched. This script does a live probe fetch of 3 dates
per DTE across the clean set to confirm ORATS has the data before the full backfill.

Uses the backfill safety protocol (BACKFILL_DATABASE_URL, dash_backfill_writer).
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# Route writes to safe role
_bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set")
os.environ["DATABASE_URL"] = _bak_url

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.opra import format_opra

OPRA_ROOT = "SPX"

_NYSE_HOLIDAYS: frozenset[date] = frozenset({
    date(2023, 1, 2),  date(2023, 1, 16), date(2023, 2, 20), date(2023, 4, 7),
    date(2023, 5, 29), date(2023, 6, 19), date(2023, 7, 4),  date(2023, 9, 4),
    date(2023, 11, 23),date(2023, 12, 25),
    date(2024, 1, 1),  date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4),  date(2024, 9, 2),
    date(2024, 11, 28),date(2024, 12, 25),
    date(2025, 1, 1),  date(2025, 1, 9),  date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1),  date(2025, 11, 27),date(2025, 12, 25),
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26),date(2026, 12, 25),
})

def nth_business_day(d: date, n: int) -> date:
    cur, cnt = d, 0
    while cnt < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in _NYSE_HOLIDAYS:
            cnt += 1
    return cur

def round5(p: float) -> int:
    return round(p / 5) * 5

# ── The 110 clean debit dates, probe 3 spread across chronology ───────────────
# Near, mid, far representatives from early/mid/late sample — hardcode 3 known
# clean dates from the CR-AH sample (2023, 2024, 2025) for the probe.
# Using dates whose targets we know from Step 0a output / the original backfill.
PROBE_DATES = [
    # Early (2023), middle (2024), recent (2025) — each with a known drift_target
    {"trade_date": date(2023, 5, 10), "short_strike": 4220.0, "band": "mid"},
    {"trade_date": date(2024, 4, 1),  "short_strike": None,   "band": None},  # will be looked up
    {"trade_date": date(2025, 6, 2),  "short_strike": None,   "band": None},
]
# We'll actually load the first 3 clean dates from the DB and probe those.

CANDIDATE_DTES = [5, 7, 10, 21]
PROBE_N = 3  # dates to probe per DTE

# ── DB ────────────────────────────────────────────────────────────────────────
conn = get_backfill_db_conn()
assert_role_or_die(conn)

# Load first PROBE_N×3 clean dates chronologically from the clean set
# (re-derive a subset cheaply)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import _LANDSCAPE_ROW_SQL, _OPEN_STRADDLE_SQL, _materialize_payload
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.backtest.models import distance_band

TICKER = "SPX"
VERSION = CANONICAL_FEATURE_VERSION
SPLIT_DATE = date(2025, 8, 12)
DTE_BASE = 15

def stride_select(lst, n=50):
    if len(lst) <= n: return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]

# Quick re-load of clean dates
from datetime import time as _time

all_dates = conn.execute(
    "SELECT trade_date FROM bt_daily_features WHERE ticker=%s AND feature_version=%s "
    "AND regime_at_classification='magnet-above' ORDER BY trade_date ASC",
    (TICKER, VERSION),
).fetchall()

all_entries = []
for (td,) in all_dates:
    row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, td)).fetchone()
    if not row or row[0] is None or row[1] is None: continue
    landscape_rows, table_spot = row
    spot = float(table_spot)
    floor_ts = datetime.combine(td, _time(6, 33, 0))
    iv_row = conn.execute(_OPEN_STRADDLE_SQL, (td.isoformat(), TICKER, floor_ts)).fetchone()
    if not iv_row or iv_row[0] is None: continue
    try:
        im = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
    except Exception: continue
    if not im: continue
    payload = _materialize_payload(landscape_rows, spot, im)
    dt_val = (payload.get("regime") or {}).get("drift_target")
    if dt_val is None: continue
    sigma = (float(dt_val) - spot) / im
    all_entries.append({"trade_date": td, "drift_target": float(dt_val), "spot": spot,
                        "sigma": sigma, "band": distance_band(sigma),
                        "partition": "train" if td <= SPLIT_DATE else "holdout"})

by_band: dict[str, list] = {"near": [], "mid": [], "far": []}
for e in all_entries:
    by_band[e["band"]].append(e)
for b in ("near", "mid", "far"):
    by_band[b].sort(key=lambda x: x["trade_date"])

selected = []
for b in ("near", "mid", "far"):
    selected += stride_select(by_band[b])
selected.sort(key=lambda x: x["trade_date"])

# Filter A-bucket at DTE=15
clean_dates = []
for e in selected:
    td = e["trade_date"]
    ss = float(round5(e["drift_target"]))
    ls = ss - 10.0
    expiry = nth_business_day(td, DTE_BASE)
    so = format_opra(OPRA_ROOT, expiry, "C", ss)
    lo = format_opra(OPRA_ROOT, expiry, "C", ls)
    entry_start = datetime(td.year, td.month, td.day, 6, 30)
    entry_end   = datetime(td.year, td.month, td.day, 13, 0)
    row = conn.execute(
        "SELECT SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END)>0, "
        "SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END)>0 "
        "FROM orats_options_minute WHERE opra_symbol=ANY(%s) "
        "AND snapshot_pt>=%s AND snapshot_pt<%s",
        (so, lo, [so, lo], entry_start, entry_end),
    ).fetchone()
    if row and bool(row[0]) and bool(row[1]):
        clean_dates.append({**e, "short_strike": ss, "long_strike": ls})

print(f"\n{'='*65}")
print("CR-AI Step 0b — Live ORATS coverage probe")
print(f"{'='*65}")
print(f"Clean debit dates available: {len(clean_dates)}")
print(f"Probe: {PROBE_N} dates per DTE × {len(CANDIDATE_DTES)} DTEs")

# Pick PROBE_N dates spread across the sample
step = max(1, len(clean_dates) // PROBE_N)
probe_entries = [clean_dates[i * step] for i in range(PROBE_N)]

print(f"\nProbe dates: {[str(e['trade_date']) for e in probe_entries]}")
print(f"Candidate DTEs: {CANDIDATE_DTES}")

# ── Live probe fetch ──────────────────────────────────────────────────────────

results = {}  # (date, dte) → {"short": bool, "long": bool, "short_bars": int, "long_bars": int}

for e in probe_entries:
    td = e["trade_date"]
    ss = e["short_strike"]
    ls = e["long_strike"]
    entry_start = datetime(td.year, td.month, td.day, 6, 30)
    entry_end   = datetime(td.year, td.month, td.day, 13, 0)

    for dte in CANDIDATE_DTES:
        expiry = nth_business_day(td, dte)
        so = format_opra(OPRA_ROOT, expiry, "C", ss)
        lo = format_opra(OPRA_ROOT, expiry, "C", ls)

        print(f"\n  {td} DTE={dte} expiry={expiry}", flush=True)
        print(f"    short={so}", flush=True)
        print(f"    long= {lo}", flush=True)

        short_ok = long_ok = False
        short_bars = long_bars = 0

        try:
            summary = fetch_option_bars(
                [so, lo], entry_start, entry_end,
                source="historical_backfill",
                record_empty_windows=True,
            )
            # Count bars now in cache
            row = conn.execute(
                "SELECT SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END) "
                "FROM orats_options_minute WHERE opra_symbol=ANY(%s) "
                "AND snapshot_pt>=%s AND snapshot_pt<%s",
                (so, lo, [so, lo], entry_start, entry_end),
            ).fetchone()
            short_bars = int(row[0] or 0)
            long_bars  = int(row[1] or 0)
            short_ok = short_bars > 0
            long_ok  = long_bars  > 0
            print(f"    → short_bars={short_bars}  long_bars={long_bars}  "
                  f"fetched={summary.bars_written}  cache_hits={summary.cache_hits}", flush=True)
        except OratsPermanentError as exc:
            print(f"    → ORATS 404/permanent error: {exc}", flush=True)
        except Exception as exc:
            print(f"    → ERROR: {exc}", flush=True)

        results[(td, dte)] = {
            "short_ok": short_ok, "long_ok": long_ok,
            "short_bars": short_bars, "long_bars": long_bars,
        }

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("PROBE SUMMARY")
print(f"{'─'*65}")
print(f"  {'date':<14}  {'DTE':>4}  {'short_bars':>11}  {'long_bars':>10}  {'both_ok':>8}")
for (td, dte), r in sorted(results.items()):
    both = "✓" if r["short_ok"] and r["long_ok"] else "✗"
    print(f"  {str(td):<14}  {dte:>4}  {r['short_bars']:>11}  {r['long_bars']:>10}  {both:>8}")

# By DTE
print(f"\n  DTE coverage summary:")
for dte in CANDIDATE_DTES:
    dte_results = [r for (td, d), r in results.items() if d == dte]
    n_ok = sum(1 for r in dte_results if r["short_ok"] and r["long_ok"])
    print(f"  DTE={dte}: {n_ok}/{len(dte_results)} probe dates have both legs")

conn.close()
print(f"\n{'='*65}")
print("Step 0b probe complete.")
print(f"{'='*65}\n")
