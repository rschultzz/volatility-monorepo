"""CR-AI Step 0 — Diagnosis gate (READ-ONLY).

Answers three gate questions before Stage 2 backfill:
  0a. Confirm the exact CR-AH clean debit date set (A-bucket, DTE=15 coverage confirmed).
  0b. ORATS coverage probe at each candidate DTE {5,7,10,15,21} for those dates.
  0c. Re-proposal logic confirmation (sample date at two DTEs).

No writes. No backfill. Pure probe.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import psycopg
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import _LANDSCAPE_ROW_SQL, _OPEN_STRADDLE_SQL, _materialize_payload
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.backtest.models import distance_band
from packages.shared.options_cache.opra import format_opra

TICKER     = "SPX"
OPRA_ROOT  = "SPX"
VERSION    = CANONICAL_FEATURE_VERSION
SPLIT_DATE = date(2025, 8, 12)
DTE_BASE   = 15   # CR-AH's original DTE (used to define the clean date set)

CANDIDATE_DTES = [5, 7, 10, 15, 21]

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

def stride_select(lst: list, n: int = 50) -> list:
    if len(lst) <= n:
        return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]


# ── DB ────────────────────────────────────────────────────────────────────────
raw_url = os.environ.get("DATABASE_URL", "").strip()
if not raw_url:
    raise RuntimeError("DATABASE_URL not set")
url = raw_url
if url.startswith("postgresql+psycopg://"):
    url = "postgresql://" + url[len("postgresql+psycopg://"):]
if "sslmode=" not in url:
    url += "?sslmode=require"

conn = psycopg.connect(url)

print(f"\n{'='*70}")
print("CR-AI Step 0 — Diagnosis Gate")
print(f"{'='*70}")

# ── 0A: Reproduce CR-AH clean debit date set ─────────────────────────────────
print("\n--- 0A: Reproducing CR-AH clean debit date set ---")

# 1. Load all magnet-above dates with sigma
all_rows = conn.execute(
    "SELECT trade_date FROM bt_daily_features "
    "WHERE ticker=%s AND feature_version=%s AND regime_at_classification='magnet-above' "
    "ORDER BY trade_date ASC",
    (TICKER, VERSION),
).fetchall()
dates = [r[0] for r in all_rows]
print(f"Total magnet-above dates: {len(dates)}")

all_entries = []
for td in dates:
    row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, td)).fetchone()
    if not row or row[0] is None or row[1] is None:
        continue
    landscape_rows, table_spot = row
    spot = float(table_spot)
    floor_ts = datetime.combine(td, time(6, 33, 0))
    iv_row = conn.execute(_OPEN_STRADDLE_SQL, (td.isoformat(), TICKER, floor_ts)).fetchone()
    if not iv_row or iv_row[0] is None:
        continue
    try:
        im = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
    except Exception:
        continue
    if not im:
        continue
    payload = _materialize_payload(landscape_rows, spot, im)
    dt_val = (payload.get("regime") or {}).get("drift_target")
    if dt_val is None:
        continue
    sigma = (float(dt_val) - spot) / im
    all_entries.append({
        "trade_date": td,
        "drift_target": float(dt_val),
        "spot": spot,
        "sigma": sigma,
        "band": distance_band(sigma),
        "partition": "train" if td <= SPLIT_DATE else "holdout",
    })

print(f"Entries with sigma: {len(all_entries)}")

# 2. Stratified selection (50/band)
by_band: dict[str, list] = {"near": [], "mid": [], "far": []}
for e in all_entries:
    by_band[e["band"]].append(e)
for b in ("near", "mid", "far"):
    by_band[b].sort(key=lambda x: x["trade_date"])
selected = []
for b in ("near", "mid", "far"):
    sel = stride_select(by_band[b])
    print(f"  {b}: {len(by_band[b])} total → {len(sel)} stride-selected")
    selected += sel
selected.sort(key=lambda x: x["trade_date"])
print(f"Total stratified: {len(selected)}")

# 3. A-bucket filter at DTE=15 (debit: long=target-10, short=target)
print(f"\nA-bucket coverage filter (debit, DTE={DTE_BASE})...")
clean_dates = []
coverage_miss = {"no_short": 0, "no_long": 0, "no_both": 0}

for e in selected:
    td = e["trade_date"]
    short_strike = float(round5(e["drift_target"]))
    long_strike  = short_strike - 10.0
    expiry = nth_business_day(td, DTE_BASE)

    short_opra = format_opra(OPRA_ROOT, expiry, "C", short_strike)
    long_opra  = format_opra(OPRA_ROOT, expiry, "C", long_strike)

    entry_start = datetime(td.year, td.month, td.day, 6, 30)
    entry_end   = datetime(td.year, td.month, td.day, 13, 0)

    row = conn.execute(
        """
        SELECT
            SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END)>0 AS has_short,
            SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END)>0 AS has_long
        FROM orats_options_minute
        WHERE opra_symbol=ANY(%s) AND snapshot_pt>=%s AND snapshot_pt<%s
        """,
        (short_opra, long_opra, [short_opra, long_opra], entry_start, entry_end),
    ).fetchone()
    has_short = bool(row[0]) if row else False
    has_long  = bool(row[1]) if row else False

    if has_short and has_long:
        clean_dates.append({**e, "short_opra_15": short_opra, "long_opra_15": long_opra, "expiry_15": expiry})
    elif not has_short and not has_long:
        coverage_miss["no_both"] += 1
    elif not has_short:
        coverage_miss["no_short"] += 1
    else:
        coverage_miss["no_long"] += 1

n_train    = sum(1 for d in clean_dates if d["partition"] == "train")
n_holdout  = sum(1 for d in clean_dates if d["partition"] == "holdout")
near_n     = sum(1 for d in clean_dates if d["band"] == "near")
mid_n      = sum(1 for d in clean_dates if d["band"] == "mid")
far_n      = sum(1 for d in clean_dates if d["band"] == "far")

print(f"\n0A RESULT:")
print(f"  Clean debit A-bucket dates (DTE=15): {len(clean_dates)}")
print(f"    train  (≤ {SPLIT_DATE}): {n_train}")
print(f"    holdout (> {SPLIT_DATE}): {n_holdout}")
print(f"    near: {near_n}  mid: {mid_n}  far: {far_n}")
print(f"  Dropped at DTE=15 coverage: no_short={coverage_miss['no_short']}, "
      f"no_long={coverage_miss['no_long']}, no_both={coverage_miss['no_both']}")

# ── 0B: ORATS coverage at each candidate DTE ─────────────────────────────────
print(f"\n--- 0B: Coverage probe at each candidate DTE ---")
print(f"Checking {len(clean_dates)} dates × {len(CANDIDATE_DTES)} DTEs = "
      f"{len(clean_dates)*len(CANDIDATE_DTES)} checks...")

# coverage[dte][date] = True/False
coverage: dict[int, dict[date, bool]] = {dte: {} for dte in CANDIDATE_DTES}

for i, e in enumerate(clean_dates):
    if i % 20 == 0:
        print(f"  {i}/{len(clean_dates)}...", flush=True)
    td = e["trade_date"]
    short_strike = float(round5(e["drift_target"]))
    long_strike  = short_strike - 10.0
    entry_start  = datetime(td.year, td.month, td.day, 6, 30)
    entry_end    = datetime(td.year, td.month, td.day, 13, 0)

    for dte in CANDIDATE_DTES:
        expiry     = nth_business_day(td, dte)
        short_opra = format_opra(OPRA_ROOT, expiry, "C", short_strike)
        long_opra  = format_opra(OPRA_ROOT, expiry, "C", long_strike)

        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END)>0 AS has_short,
                SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END)>0 AS has_long
            FROM orats_options_minute
            WHERE opra_symbol=ANY(%s) AND snapshot_pt>=%s AND snapshot_pt<%s
            """,
            (short_opra, long_opra, [short_opra, long_opra], entry_start, entry_end),
        ).fetchone()
        both = bool(row[0]) and bool(row[1]) if row else False
        coverage[dte][td] = both

print("\n0B RESULT — Coverage matrix (dates with BOTH legs present on entry day):")
print(f"\n  {'DTE':>4}  {'n_covered':>10}  {'n_missing':>10}  {'coverage%':>10}")
print("  " + "─" * 40)
for dte in CANDIDATE_DTES:
    n_cov = sum(1 for v in coverage[dte].values() if v)
    n_mis = len(clean_dates) - n_cov
    pct   = 100 * n_cov / len(clean_dates)
    print(f"  {dte:>4}  {n_cov:>10}  {n_mis:>10}  {pct:>9.1f}%")

# Band × DTE breakdown
print(f"\n  Coverage by band × DTE:")
print(f"  {'Band':<8}", end="")
for dte in CANDIDATE_DTES:
    print(f"  {'DTE='+str(dte):>8}", end="")
print()
print("  " + "─" * (8 + 10*len(CANDIDATE_DTES)))
for band in ("near", "mid", "far"):
    band_dates = [e for e in clean_dates if e["band"] == band]
    print(f"  {band:<8}", end="")
    for dte in CANDIDATE_DTES:
        n_cov = sum(1 for e in band_dates if coverage[dte][e["trade_date"]])
        print(f"  {n_cov:>7}/{len(band_dates)}", end="")
    print()

# Which dates are missing at each shorter DTE but present at DTE=15?
for dte in CANDIDATE_DTES:
    if dte == 15:
        continue
    newly_missing = [
        e["trade_date"]
        for e in clean_dates
        if coverage[15][e["trade_date"]] and not coverage[dte][e["trade_date"]]
    ]
    if newly_missing:
        print(f"\n  DTE={dte}: {len(newly_missing)} dates with DTE=15 coverage but NOT DTE={dte}:")
        for d in newly_missing[:10]:
            print(f"    {d}")
        if len(newly_missing) > 10:
            print(f"    ... (and {len(newly_missing)-10} more)")
    else:
        print(f"\n  DTE={dte}: no dates with DTE=15 coverage that are missing at DTE={dte}")

# ── 0C: Re-proposal confirmation on a sample date ────────────────────────────
print(f"\n--- 0C: Re-proposal logic confirmation ---")
# Pick a mid-band date to demonstrate
sample = next((e for e in clean_dates if e["band"] == "mid"), clean_dates[0])
td = sample["trade_date"]
short_strike = float(round5(sample["drift_target"]))
long_strike  = short_strike - 10.0

print(f"\n  Sample date: {td}  band={sample['band']}  sigma={sample['sigma']:.2f}")
print(f"  drift_target={sample['drift_target']:.1f}  spot={sample['spot']:.1f}")
print(f"  short_strike={short_strike:.0f}  long_strike={long_strike:.0f}  (identical across all DTEs)")
print(f"\n  {'DTE':>4}  {'expiry':>12}  {'short_opra':>30}  {'long_opra':>30}")
for dte in CANDIDATE_DTES:
    expiry = nth_business_day(td, dte)
    so = format_opra(OPRA_ROOT, expiry, "C", short_strike)
    lo = format_opra(OPRA_ROOT, expiry, "C", long_strike)
    print(f"  {dte:>4}  {str(expiry):>12}  {so:>30}  {lo:>30}")

print(f"\n  Conclusion: strikes are IDENTICAL at all DTEs. Only expiry date changes.")
print(f"  Band ({sample['band']}) and drift_target ({sample['drift_target']:.1f}) are DTE-agnostic. ✓")

# ── DTE_TARGET_BY_BUCKET baseline: look up dominant bucket per date ───────────
print(f"\n--- DTE_TARGET_BY_BUCKET baseline: dominant bucket distribution ---")
# bt_daily_outcomes stores dominant_bucket_at_classification
bucket_rows = conn.execute(
    "SELECT trade_date, dominant_bucket_at_classification "
    "FROM bt_daily_outcomes "
    "WHERE ticker=%s AND feature_version=%s AND active=true "
    "ORDER BY trade_date ASC",
    (TICKER, VERSION),
).fetchall()
bucket_map = {r[0]: r[1] for r in bucket_rows}
DTE_BY_BUCKET = {"0DTE": 1, "1-7": 3, "8-30": 15, "30+": 45}

print(f"\n  Dominant bucket distribution across {len(clean_dates)} clean debit dates:")
bucket_counts: dict = {}
for e in clean_dates:
    bkt = bucket_map.get(e["trade_date"], "UNKNOWN")
    bucket_counts[bkt] = bucket_counts.get(bkt, 0) + 1
for bkt, n in sorted(bucket_counts.items(), key=lambda x: -x[1]):
    dte_live = DTE_BY_BUCKET.get(bkt, "?")
    print(f"    {bkt:<12}  n={n:>3}  live_DTE={dte_live}")

# Per-date live DTE
live_dte_counts: dict[str, int] = {}
for e in clean_dates:
    bkt = bucket_map.get(e["trade_date"], "UNKNOWN")
    dte_live = str(DTE_BY_BUCKET.get(bkt, "?"))
    live_dte_counts[dte_live] = live_dte_counts.get(dte_live, 0) + 1
print(f"\n  Live DTE_TARGET_BY_BUCKET distribution (what the live engine would have used):")
for dte_str, n in sorted(live_dte_counts.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
    print(f"    DTE={dte_str}: {n} dates")

conn.close()
print(f"\n{'='*70}")
print("Step 0 complete. Review findings above before proceeding to Step 1.")
print(f"{'='*70}\n")
