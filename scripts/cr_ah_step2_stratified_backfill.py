"""CR-AH Step 2 — Stratified subsampled backfill.

Scope: ~50 dates per distance band (near <1.5σ / mid 1.5–2σ / far >2σ), ~150 total.
Selection: chronological stride within each band — deterministic, reproducible.
Per-date backfill: entry-day RTH, touch-window (if ES touch found), settlement.

Safety: BACKFILL_DATABASE_URL + dash_backfill_writer role enforced before any writes.
        DATABASE_URL is overridden to BACKFILL_DATABASE_URL so the options_cache repo
        (which reads DATABASE_URL) also uses the restricted role.
Idempotent: re-run = cache hits only (fetch_option_bars gap detection).

Usage:
    DRY_RUN = True  (default)  — plans all 3 dates; probe-fetches PROBE_FETCH_COUNT of them.
    DRY_RUN = False             — backfills all ~150 dates via ORATS API.

Touch-resolution outcomes (tagged per trade for Step 4):
    rth_touch               — ES touched drift_target DURING options RTH (06:30–13:00 PT).
                              Exit = option mid at touch minute.  Fetch: ±30-min RTH window.
    gap_touch               — ES touched after-hours AND ES ≥ target at NEXT RTH open (06:30 PT).
                              Exit = option mid at OPEN minute (NOT the uninvestable AH price).
                              Fetch: 06:30–07:30 PT on next-RTH date.
    afterhours_touch_retraced — ES touched after-hours but was BELOW target at next RTH open.
                              No realizable exit → close-path P&L only.
    no_touch                — ES never reached target before expiry → close-path P&L only.

Step 4 can report touch-exit two ways:
    STRICT   = rth_touch only      (actionable during RTH; no overnight action needed)
    RTH+GAP  = rth_touch + gap_touch  (captures gap-open fills; needs open order / fast action)
The delta = incremental edge from capturing overnight gaps.

Design note on strike rounding:
    generate_proposals() uses raw GEX wall prices (continuous floats) as strikes.
    SPX 15-DTE listed strikes are at 5pt spacing. Non-standard strikes return empty
    from ORATS. We round to the nearest 5pt to hit listed strikes while preserving
    the proposal's economic intent (1–3pt rounding error on a 10pt spread is immaterial).

Run as:  PYTHONUNBUFFERED=1 python -u scripts/cr_ah_step2_stratified_backfill.py
"""

from __future__ import annotations

import os
import sys
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ── ENV SETUP (must happen before any imports that read DATABASE_URL) ─────────

def _find_dotenv() -> Path | None:
    """Walk upward from script location until a .env file is found (max 8 levels)."""
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

# Route the options-cache repo (reads DATABASE_URL) to the safe backfill role.
_bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set. Run aborted.")
os.environ["DATABASE_URL"] = _bak_url  # so options_cache.repository also uses safe role

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

from packages.shared.backfill_safety import (
    get_backfill_db_conn,
    assert_role_or_die,
    backfill_run,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import (
    _LANDSCAPE_ROW_SQL,
    _OPEN_STRADDLE_SQL,
    _materialize_payload,
)
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.opra import format_opra
from packages.shared.backtest.models import distance_band

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRY_RUN: bool = False           # flip to False for production runs
REBACKFILL_B_ONLY: bool = True   # True = re-backfill only B-bucket (wrong-expiry) dates

# When DRY_RUN=True, probe-fetch the first N dates (actually calls ORATS + reads back).
# Remaining dry-run dates print plan only. Set to 0 to skip probe entirely.
PROBE_FETCH_COUNT: int = 1

TARGET_PER_BAND: int = 50       # stride-select this many per band
SPLIT_DATE = date(2025, 8, 12)
TICKER = "SPX"
OPRA_ROOT = "SPX"               # ORATS uses 'SPX' root for all SPX expirations (per opra.py)
VERSION = CANONICAL_FEATURE_VERSION
DTE_TARGET: int = 15            # business days to expiry
RTH_TOUCH_WINDOW_MINS: int = 30 # ±N min around RTH touch for option quote window
GAP_TOUCH_WINDOW_HRS: int = 1   # open window for gap-touch next-RTH session (from 06:30 PT)
ES_OPEN_LOOKUP_MINS: int = 5    # look for first ES bar within N min of RTH open

_UTC = ZoneInfo("UTC")
_PT = ZoneInfo("America/Los_Angeles")

# ── NYSE HOLIDAY CALENDAR (2023–2026) ─────────────────────────────────────────
# Hard-coded for the backfill's date range. Includes all standard closures plus
# the Carter state funeral (2025-01-09) and Inauguration Day (2025-01-20, which
# coincides with MLK Day that year).

_NYSE_HOLIDAYS: frozenset[date] = frozenset({
    # 2023
    date(2023, 1, 2),   # New Year's Day (observed)
    date(2023, 1, 16),  # Martin Luther King Jr. Day
    date(2023, 2, 20),  # Presidents' Day
    date(2023, 4, 7),   # Good Friday
    date(2023, 5, 29),  # Memorial Day
    date(2023, 6, 19),  # Juneteenth
    date(2023, 7, 4),   # Independence Day
    date(2023, 9, 4),   # Labor Day
    date(2023, 11, 23), # Thanksgiving
    date(2023, 12, 25), # Christmas
    # 2024
    date(2024, 1, 1),   # New Year's Day
    date(2024, 1, 15),  # Martin Luther King Jr. Day
    date(2024, 2, 19),  # Presidents' Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),   # Independence Day
    date(2024, 9, 2),   # Labor Day
    date(2024, 11, 28), # Thanksgiving
    date(2024, 12, 25), # Christmas
    # 2025
    date(2025, 1, 1),   # New Year's Day
    date(2025, 1, 9),   # Carter state funeral (special NYSE closure)
    date(2025, 1, 20),  # Martin Luther King Jr. Day / Inauguration Day (coincide)
    date(2025, 2, 17),  # Presidents' Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),   # Independence Day
    date(2025, 9, 1),   # Labor Day
    date(2025, 11, 27), # Thanksgiving
    date(2025, 12, 25), # Christmas
    # 2026
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day (observed — Jul 4 falls on Saturday)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
})

# ── B-BUCKET RE-BACKFILL: previously skipped 404 dates (C + D from initial run) ─
# Used in REBACKFILL_B_ONLY mode to exclude the 47 already-404'd dates so only
# the 47 wrong-expiry-but-fetched B-dates are re-processed.
_PREVIOUSLY_SKIPPED_404: frozenset[date] = frozenset({
    # near/train (6)
    date(2023, 5, 18), date(2024, 6, 12), date(2024, 11, 12),
    date(2025, 5, 29), date(2025, 6, 11), date(2025, 7, 16),
    # near/holdout (7)
    date(2025, 9, 23), date(2025, 10, 2), date(2025, 10, 22), date(2025, 12, 3),
    date(2026, 1, 29), date(2026, 4, 30), date(2026, 5, 21),
    # mid/train (4)
    date(2023, 5, 11), date(2024, 5, 23), date(2025, 1, 27), date(2025, 6, 4),
    # mid/holdout (11)
    date(2025, 8, 18), date(2025, 8, 19), date(2025, 8, 25), date(2025, 9, 8),
    date(2026, 1, 7), date(2026, 1, 13), date(2026, 2, 3), date(2026, 2, 12),
    date(2026, 4, 14), date(2026, 5, 4), date(2026, 5, 19),
    # far/train (14)
    date(2023, 5, 25), date(2023, 9, 11), date(2023, 11, 1), date(2024, 5, 6),
    date(2024, 5, 22), date(2024, 6, 24), date(2024, 7, 1), date(2024, 8, 15),
    date(2024, 10, 8), date(2024, 10, 31), date(2025, 1, 6), date(2025, 2, 4),
    date(2025, 2, 11), date(2025, 5, 12),
    # far/holdout (5)
    date(2025, 8, 21), date(2025, 10, 16), date(2025, 12, 9),
    date(2025, 12, 17), date(2026, 1, 6),
})

# ── HELPERS ───────────────────────────────────────────────────────────────────

def nth_business_day(from_date: date, n: int) -> date:
    """Return the nth NYSE trading day after from_date.

    Excludes weekends AND NYSE holidays in _NYSE_HOLIDAYS. Covers 2023–2026.
    """
    current = from_date
    count = 0
    while count < n:
        current += timedelta(days=1)
        if current.weekday() < 5 and current not in _NYSE_HOLIDAYS:
            count += 1
    return current


def _blind_nth_weekday(from_date: date, n: int) -> date:
    """Holiday-blind weekday counter — mirrors the original (buggy) implementation.

    Used only to identify B-bucket dates: those where the holiday-aware result
    differs from the blind result. Not used in the actual backfill calculation.
    """
    current = from_date
    count = 0
    while count < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


def next_weekday(d: date) -> date:
    """Return the next Mon–Fri after d (always moves at least one day forward)."""
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def round_to_5pt(price: float) -> int:
    """Round to nearest 5pt — SPX listed strike spacing for 8-30 DTE options."""
    return round(price / 5) * 5


def utc_naive_to_pt_naive(utc_naive: datetime) -> datetime:
    """Convert naive UTC datetime to naive Pacific Time (DST-aware via ZoneInfo)."""
    return utc_naive.replace(tzinfo=_UTC).astimezone(_PT).replace(tzinfo=None)


def pt_naive_to_utc_naive(pt_naive: datetime) -> datetime:
    """Convert naive Pacific Time datetime to naive UTC (DST-aware via ZoneInfo)."""
    return pt_naive.replace(tzinfo=_PT).astimezone(_UTC).replace(tzinfo=None)


def stride_select(entries: list, target: int = 50) -> list:
    """Uniform chronological stride: select exactly min(len, target) entries."""
    n = len(entries)
    if n <= target:
        return list(entries)
    step = n / target
    return [entries[int(i * step)] for i in range(target)]


# ── TOUCH RESOLUTION ─────────────────────────────────────────────────────────

def _classify_touch(
    touch_pt: datetime,
    drift_target: float,
    conn,
) -> tuple[str, date | None, datetime | None, datetime | None]:
    """Classify an ES touch into one of four resolution outcomes.

    Returns (resolution, next_rth_date, fetch_start_pt, fetch_end_pt):
      rth_touch               → fetch is ±RTH_TOUCH_WINDOW_MINS around touch_pt
      gap_touch               → fetch is GAP_TOUCH_WINDOW_HRS from next-RTH open (06:30 PT)
      afterhours_touch_retraced → (None, None, None) fetch windows
      [no_touch handled at caller level — this fn only called when touch exists]

    All datetime args/returns are naive PT.
    """
    touch_date = touch_pt.date()
    day_open  = datetime(touch_date.year, touch_date.month, touch_date.day, 6, 30)
    day_close = datetime(touch_date.year, touch_date.month, touch_date.day, 13, 0)

    if day_open <= touch_pt <= day_close:
        # RTH touch: standard ±window around touch, clipped to session bounds
        w_start = max(touch_pt - timedelta(minutes=RTH_TOUCH_WINDOW_MINS), day_open)
        w_end   = min(touch_pt + timedelta(minutes=RTH_TOUCH_WINDOW_MINS), day_close)
        return "rth_touch", touch_date, w_start, w_end

    # Outside RTH: determine next RTH session
    if touch_pt < day_open:
        # Pre-market: same day's open is the next RTH open
        next_rth_d  = touch_date
        next_open_pt = day_open
    else:
        # Post-close: next weekday's open
        next_rth_d  = next_weekday(touch_date)
        next_open_pt = datetime(next_rth_d.year, next_rth_d.month, next_rth_d.day, 6, 30)

    # Check ES at next RTH open: first bar within the open minute ±ES_OPEN_LOOKUP_MINS
    next_open_utc = pt_naive_to_utc_naive(next_open_pt)
    es_row = conn.execute(
        """
        SELECT close FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime < %s
        ORDER BY datetime ASC LIMIT 1
        """,
        (next_open_utc, next_open_utc + timedelta(minutes=ES_OPEN_LOOKUP_MINS)),
    ).fetchone()

    if es_row is None:
        # No ES data at next open (holiday / gap in bars) — conservative: treat as retraced
        return "afterhours_touch_retraced", next_rth_d, None, None

    es_at_open = float(es_row[0])
    if es_at_open >= drift_target:
        # Gap held — realizable at open; fetch the first hour of the next RTH session
        gap_end = next_open_pt + timedelta(hours=GAP_TOUCH_WINDOW_HRS)
        return "gap_touch", next_rth_d, next_open_pt, gap_end
    else:
        return "afterhours_touch_retraced", next_rth_d, None, None


# ── SELECTION ─────────────────────────────────────────────────────────────────

def load_signal_dates_with_sigma(conn) -> list[dict]:
    """Load all magnet-above dates and compute drift_target_distance_sigma per date."""
    rows = conn.execute(
        """
        SELECT trade_date FROM bt_daily_features
        WHERE ticker = %s AND feature_version = %s
          AND regime_at_classification = 'magnet-above'
        ORDER BY trade_date ASC
        """,
        (TICKER, VERSION),
    ).fetchall()
    dates = [r[0] for r in rows]
    print(f"Loaded {len(dates)} magnet-above signal dates.", flush=True)

    result = []
    skipped = {"no_landscape": 0, "no_iv": 0, "no_drift_target": 0}

    for i, trade_date in enumerate(dates):
        if i % 50 == 0:
            print(f"  computing sigma {i}/{len(dates)}...", flush=True)

        row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, trade_date)).fetchone()
        if not row or row[0] is None or row[1] is None:
            skipped["no_landscape"] += 1
            continue
        landscape_rows, table_spot = row
        spot = float(table_spot)

        floor_ts = datetime.combine(trade_date, time(6, 33, 0))
        iv_row = conn.execute(
            _OPEN_STRADDLE_SQL, (trade_date.isoformat(), TICKER, floor_ts)
        ).fetchone()

        if iv_row and iv_row[0] is not None:
            try:
                implied_move = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
            except (TypeError, ValueError):
                implied_move = 0.0
        else:
            implied_move = 0.0
            skipped["no_iv"] += 1

        payload = _materialize_payload(landscape_rows, spot, implied_move)
        regime_block = payload.get("regime") or {}
        drift_target = regime_block.get("drift_target")

        if drift_target is None:
            skipped["no_drift_target"] += 1
            continue

        if not implied_move:
            skipped["no_iv"] += 1
            continue

        sigma = (float(drift_target) - spot) / implied_move
        band = distance_band(sigma)
        partition = "train" if trade_date <= SPLIT_DATE else "holdout"
        result.append({
            "trade_date": trade_date,
            "drift_target": float(drift_target),
            "spot": spot,
            "sigma": sigma,
            "band": band,
            "partition": partition,
        })

    print(
        f"  sigma computed for {len(result)}/{len(dates)} dates. "
        f"Skipped: {skipped}",
        flush=True,
    )
    return result


# ── PER-DATE BACKFILL ─────────────────────────────────────────────────────────

def backfill_date(
    entry: dict,
    conn,
    dry_run: bool = True,
    probe_fetch: bool = False,
) -> dict:
    """Backfill one signal date's option quotes.

    dry_run=True, probe_fetch=False  — plan only (no fetches).
    dry_run=True, probe_fetch=True   — plan + real fetches + read-back verification.
    dry_run=False                    — plan + real fetches (normal production path).
    """
    trade_date  = entry["trade_date"]
    drift_target = entry["drift_target"]
    band        = entry["band"]
    partition   = entry["partition"]

    # ── Strikes (round to nearest 5pt listed SPX strike) ──────────────────────
    short_strike = float(round_to_5pt(drift_target))
    long_strike  = short_strike + 10.0

    # ── Expiry resolution: 15th business day from signal date ─────────────────
    expiry_date = nth_business_day(trade_date, DTE_TARGET)

    # ── OPRA symbols (call side — magnet-above vertical is a call spread) ─────
    short_opra = format_opra(OPRA_ROOT, expiry_date, "C", short_strike)
    long_opra  = format_opra(OPRA_ROOT, expiry_date, "C", long_strike)
    opras = [short_opra, long_opra]

    # ── Entry-day window: full RTH 06:30–13:00 PT ─────────────────────────────
    entry_start_pt = datetime(trade_date.year, trade_date.month, trade_date.day, 6, 30)
    entry_end_pt   = datetime(trade_date.year, trade_date.month, trade_date.day, 13, 0)

    # ── ES touch detection: first bar in [signal_date, expiry_date] with close ≥ drift_target
    search_start_utc = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, 0)
    search_end_utc   = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 23, 59, 59)
    touch_row = conn.execute(
        """
        SELECT datetime, close FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime <= %s AND close >= %s
        ORDER BY datetime ASC LIMIT 1
        """,
        (search_start_utc, search_end_utc, drift_target),
    ).fetchone()
    touch_utc: datetime | None = touch_row[0] if touch_row else None

    # ── Touch resolution: classify into 4 outcomes ────────────────────────────
    touch_resolution: str = "no_touch"
    touch_fetch_start_pt: datetime | None = None
    touch_fetch_end_pt: datetime | None = None
    touch_next_rth_date: date | None = None
    es_at_next_open: float | None = None

    if touch_utc is not None:
        touch_pt = utc_naive_to_pt_naive(touch_utc)
        touch_resolution, touch_next_rth_date, touch_fetch_start_pt, touch_fetch_end_pt = (
            _classify_touch(touch_pt, drift_target, conn)
        )
        # For gap/retraced: record the ES open price for transparency
        if touch_resolution in ("gap_touch", "afterhours_touch_retraced"):
            next_open_utc = pt_naive_to_utc_naive(
                datetime(touch_next_rth_date.year, touch_next_rth_date.month,
                         touch_next_rth_date.day, 6, 30)
            )
            es_row = conn.execute(
                "SELECT close FROM ironbeam_es_1m_bars "
                "WHERE datetime >= %s AND datetime < %s ORDER BY datetime ASC LIMIT 1",
                (next_open_utc, next_open_utc + timedelta(minutes=ES_OPEN_LOOKUP_MINS)),
            ).fetchone()
            if es_row:
                es_at_next_open = float(es_row[0])
    else:
        touch_pt = None

    # ── Settlement window: last 10 min of expiry RTH ──────────────────────────
    settle_start_pt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 12, 50)
    settle_end_pt   = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 13, 0)

    # ── Print plan ────────────────────────────────────────────────────────────
    print(
        f"\n  [{partition}/{band}] {trade_date}  target={drift_target:.0f}  "
        f"σ={entry['sigma']:.2f}  expiry={expiry_date}",
        flush=True,
    )
    print(f"    short={short_opra}  long={long_opra}", flush=True)
    print(f"    entry-day:   {entry_start_pt:%H:%M}–{entry_end_pt:%H:%M} PT on {trade_date}", flush=True)

    if touch_utc is None:
        print(f"    ES touch:    NONE  →  touch_resolution=no_touch", flush=True)
    else:
        touch_pt_str = f"{touch_utc:%Y-%m-%d %H:%M UTC} ({touch_pt:%H:%M PT})"
        print(f"    ES touch:    {touch_pt_str}  →  touch_resolution={touch_resolution}", flush=True)

    if touch_resolution == "rth_touch":
        print(
            f"    touch-window:{touch_fetch_start_pt:%Y-%m-%d %H:%M}–{touch_fetch_end_pt:%H:%M} PT",
            flush=True,
        )
    elif touch_resolution == "gap_touch":
        print(
            f"    ES at next open ({touch_next_rth_date} 06:30 PT): {es_at_next_open:.2f} ≥ {drift_target:.0f} → gap held",
            flush=True,
        )
        print(
            f"    gap-open-window:{touch_fetch_start_pt:%Y-%m-%d %H:%M}–{touch_fetch_end_pt:%H:%M} PT",
            flush=True,
        )
    elif touch_resolution == "afterhours_touch_retraced":
        open_str = f"{es_at_next_open:.2f}" if es_at_next_open is not None else "n/a"
        print(
            f"    ES at next open ({touch_next_rth_date} 06:30 PT): {open_str} < {drift_target:.0f} → retraced",
            flush=True,
        )

    print(f"    settlement:  {settle_start_pt:%H:%M}–{settle_end_pt:%H:%M} PT on {expiry_date}", flush=True)

    do_fetch = not dry_run or probe_fetch
    if dry_run and probe_fetch:
        print(f"    [PROBE FETCH — real API calls + read-back]", flush=True)

    if not do_fetch:
        return {
            "trade_date": str(trade_date),
            "band": band,
            "partition": partition,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "expiry_date": str(expiry_date),
            "short_opra": short_opra,
            "long_opra": long_opra,
            "touch_resolution": touch_resolution,
            "touch_utc": str(touch_utc) if touch_utc else None,
            "touch_next_rth_date": str(touch_next_rth_date) if touch_next_rth_date else None,
            "es_at_next_open": es_at_next_open,
            "dry_run": True,
        }

    # ── FETCH ─────────────────────────────────────────────────────────────────
    total_bars = 0
    total_api_calls = 0  # gaps_filled == number of ORATS HTTP range calls
    t_fetch_start = _time.perf_counter()

    try:
        # Entry-day
        print("    FETCH entry-day...", flush=True)
        r_entry = fetch_option_bars(
            opras, entry_start_pt, entry_end_pt,
            source="historical_backfill", record_empty_windows=True,
        )
        total_bars += r_entry.bars_written
        total_api_calls += r_entry.gaps_filled
        print(
            f"      bars_written={r_entry.bars_written}  cache_hits={r_entry.cache_hits}  "
            f"gaps_filled={r_entry.gaps_filled}",
            flush=True,
        )

        # Touch-window (RTH or gap-open, depending on resolution)
        r_touch = None
        if touch_fetch_start_pt is not None:
            window_label = "touch-window" if touch_resolution == "rth_touch" else "gap-open-window"
            print(f"    FETCH {window_label}...", flush=True)
            r_touch = fetch_option_bars(
                opras, touch_fetch_start_pt, touch_fetch_end_pt,
                source="historical_backfill", record_empty_windows=True,
            )
            total_bars += r_touch.bars_written
            total_api_calls += r_touch.gaps_filled
            print(
                f"      bars_written={r_touch.bars_written}  cache_hits={r_touch.cache_hits}  "
                f"gaps_filled={r_touch.gaps_filled}",
                flush=True,
            )

        # Settlement
        print("    FETCH settlement...", flush=True)
        r_settle = fetch_option_bars(
            opras, settle_start_pt, settle_end_pt,
            source="historical_backfill", record_empty_windows=True,
        )
        total_bars += r_settle.bars_written
        total_api_calls += r_settle.gaps_filled
        print(
            f"      bars_written={r_settle.bars_written}  cache_hits={r_settle.cache_hits}  "
            f"gaps_filled={r_settle.gaps_filled}",
            flush=True,
        )

    except OratsPermanentError as exc:
        elapsed = _time.perf_counter() - t_fetch_start
        print(f"    SKIPPED (ORATS 4xx): {exc}", flush=True)
        return {
            "trade_date": str(trade_date),
            "band": band,
            "partition": partition,
            "short_opra": short_opra,
            "long_opra": long_opra,
            "touch_resolution": touch_resolution,
            "bars_written": 0,
            "total_api_calls": total_api_calls,
            "elapsed_s": round(elapsed, 1),
            "skipped_404": True,
            "skip_reason": str(exc),
            "dry_run": dry_run,
            "probe_fetch": probe_fetch,
        }

    elapsed = _time.perf_counter() - t_fetch_start
    print(f"    fetch elapsed: {elapsed:.1f}s  total_api_calls={total_api_calls}", flush=True)

    # Confirm API call structure: 2 OPRA symbols → should see gaps_filled per symbol
    # (ranged call per symbol, NOT 390 per-minute calls)
    print(
        f"    [API structure] entry-day gaps_filled={r_entry.gaps_filled} "
        f"for {len(opras)} symbols → "
        f"{'ranged (OK)' if r_entry.gaps_filled <= len(opras) * 2 else 'WARN: unexpectedly high'}",
        flush=True,
    )

    # ── READ-BACK: confirm bars landed in orats_options_minute ────────────────
    if probe_fetch:
        readback = conn.execute(
            """
            SELECT opra_symbol, COUNT(*) AS bar_count
            FROM orats_options_minute
            WHERE opra_symbol = ANY(%s)
              AND snapshot_pt >= %s
              AND snapshot_pt < %s
            GROUP BY opra_symbol
            ORDER BY opra_symbol
            """,
            (opras, entry_start_pt, entry_end_pt),
        ).fetchall()
        print(f"    READ-BACK (entry-day {trade_date}):", flush=True)
        rb_total = 0
        for sym, cnt in readback:
            print(f"      {sym}: {cnt} bars in orats_options_minute", flush=True)
            rb_total += cnt
        if not readback:
            print(f"      WARNING: 0 rows read back for {opras}", flush=True)
        print(f"      Total: {rb_total} bars read back", flush=True)

    result = {
        "trade_date": str(trade_date),
        "band": band,
        "partition": partition,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "expiry_date": str(expiry_date),
        "short_opra": short_opra,
        "long_opra": long_opra,
        "touch_resolution": touch_resolution,
        "touch_utc": str(touch_utc) if touch_utc else None,
        "touch_next_rth_date": str(touch_next_rth_date) if touch_next_rth_date else None,
        "es_at_next_open": es_at_next_open,
        "bars_written": total_bars,
        "total_api_calls": total_api_calls,
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry_run,
        "probe_fetch": probe_fetch,
    }
    return result


# ── PRE-FLIGHT GRANT CHECK ────────────────────────────────────────────────────

def _preflight_grants(conn) -> None:
    """Exit if the current role is missing any privilege that would cause a mid-run failure.

    Critical: INSERT ... ON CONFLICT DO UPDATE (UPSERT) requires both INSERT AND UPDATE
    privilege, even when no conflict occurs. Without UPDATE on orats_options_fetched_windows,
    the window-tracking INSERT (which makes fetch_option_bars idempotent) will fail after
    each ORATS API call — bars land but window records do not, breaking idempotency.

    Fix: GRANT UPDATE ON orats_options_fetched_windows TO dash_backfill_writer;
    """
    required = [
        ("orats_options_minute",          "INSERT"),
        ("orats_options_fetched_windows", "INSERT"),
        ("orats_options_fetched_windows", "UPDATE"),  # required by ON CONFLICT DO UPDATE
    ]
    missing_sql = []
    for table, priv in required:
        row = conn.execute(
            "SELECT has_table_privilege(%s, %s)", (table, priv)
        ).fetchone()
        if not row or not row[0]:
            missing_sql.append(f"GRANT {priv} ON {table} TO dash_backfill_writer;")

    if missing_sql:
        print("\n" + "!" * 65, flush=True)
        print("GRANT PRE-FLIGHT FAILED — run as superuser before proceeding:", flush=True)
        for sql in missing_sql:
            print(f"  {sql}", flush=True)
        print(
            "\nNOTE: ON CONFLICT DO UPDATE (UPSERT) in record_fetched_window requires\n"
            "UPDATE privilege even on fresh rows where no conflict occurs. Without it,\n"
            "each ORATS API call writes bars to orats_options_minute but fails to record\n"
            "the fetch window, breaking idempotency for reruns.",
            flush=True,
        )
        print("!" * 65, flush=True)
        sys.exit(1)

    print("  [pre-flight] required grants confirmed OK.", flush=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 65
    print(f"\n{sep}", flush=True)
    print("CR-AH Step 2 — Stratified subsampled backfill", flush=True)
    print(
        f"  DRY_RUN={DRY_RUN}  PROBE_FETCH_COUNT={PROBE_FETCH_COUNT}  "
        f"target_per_band={TARGET_PER_BAND}  dte_target={DTE_TARGET}",
        flush=True,
    )
    print(sep, flush=True)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    _preflight_grants(conn)

    # ── Selection ──────────────────────────────────────────────────────────────
    all_entries = load_signal_dates_with_sigma(conn)

    by_band: dict[str, list[dict]] = {"near": [], "mid": [], "far": []}
    for e in all_entries:
        by_band[e["band"]].append(e)

    selected_by_band: dict[str, list[dict]] = {}
    for band in ("near", "mid", "far"):
        sorted_entries = sorted(by_band[band], key=lambda x: x["trade_date"])
        selected_by_band[band] = stride_select(sorted_entries, TARGET_PER_BAND)
        tr = sum(1 for e in selected_by_band[band] if e["partition"] == "train")
        ho = sum(1 for e in selected_by_band[band] if e["partition"] == "holdout")
        print(
            f"\n{band.upper()} ({len(by_band[band])} total → {len(selected_by_band[band])} selected): "
            f"train={tr}  holdout={ho}",
            flush=True,
        )

    all_selected = sorted(
        [e for entries in selected_by_band.values() for e in entries],
        key=lambda x: x["trade_date"],
    )

    # ── REBACKFILL_B_ONLY: filter to B-bucket dates ───────────────────────────
    # B-bucket = selected dates where the holiday-aware expiry differs from the
    # blind expiry (holiday in positions 1-14) AND the date was previously fetched
    # (not in the 47 known-404 list). These dates have wrong-expiry bars in
    # orats_options_minute; re-fetching with the corrected expiry writes new
    # (correct) OPRA symbols without conflicting with the old ones.
    if REBACKFILL_B_ONLY:
        b_bucket = [
            e for e in all_selected
            if _blind_nth_weekday(e["trade_date"], DTE_TARGET)
               != nth_business_day(e["trade_date"], DTE_TARGET)
            and e["trade_date"] not in _PREVIOUSLY_SKIPPED_404
        ]
        print(
            f"\nREBACKFILL_B_ONLY: {len(b_bucket)} B-bucket dates identified "
            f"(expiry changed by holiday-aware fix, previously fetched).",
            flush=True,
        )
        for e in b_bucket:
            old_exp = _blind_nth_weekday(e["trade_date"], DTE_TARGET)
            new_exp = nth_business_day(e["trade_date"], DTE_TARGET)
            print(
                f"  {e['band']}/{e['partition']}  {e['trade_date']}"
                f"  blind={old_exp}  aware={new_exp}",
                flush=True,
            )
        all_selected = b_bucket

    # ── Print full selection ───────────────────────────────────────────────────
    print(f"\n{'─'*65}", flush=True)
    print(f"FULL SELECTION ({len(all_selected)} dates):", flush=True)
    print(f"{'─'*65}", flush=True)
    print(f"  {'band':<5}  {'part':<8}  {'date'}       {'σ':>6}  {'target':>7}", flush=True)
    for e in all_selected:
        print(
            f"  {e['band']:<5}  {e['partition']:<8}  {e['trade_date']}  "
            f"{e['sigma']:>+6.2f}  {e['drift_target']:>7.0f}",
            flush=True,
        )

    # ── Band + holdout n summary ───────────────────────────────────────────────
    print(f"\n{'─'*65}", flush=True)
    print("SELECTION SUMMARY:", flush=True)
    total_tr = total_ho = 0
    for band in ("near", "mid", "far"):
        tr = sum(1 for e in selected_by_band[band] if e["partition"] == "train")
        ho = sum(1 for e in selected_by_band[band] if e["partition"] == "holdout")
        total_tr += tr; total_ho += ho
        note = "  ← THIN HOLDOUT: wide CI at Step 4, read as underpowered" if band == "far" else ""
        print(
            f"  {band:<4}: n={len(selected_by_band[band]):>3}  train={tr:>2}  holdout={ho:>2}{note}",
            flush=True,
        )
    print(f"  TOTAL: n={len(all_selected):>3}  train={total_tr:>2}  holdout={total_ho:>2}", flush=True)
    far_ho = sum(1 for e in selected_by_band["far"] if e["partition"] == "holdout")
    print(
        f"\n  NOTE: far-band holdout = {far_ho} trades. Expected CI is wide for this cell.",
        flush=True,
    )

    # ── Process: dry-run = first 3 (with probe for first N); full = all ────────
    to_process = all_selected[:3] if DRY_RUN else all_selected
    if DRY_RUN:
        mode = "DRY-RUN (first 3 dates)"
    elif REBACKFILL_B_ONLY:
        mode = f"REBACKFILL B-ONLY ({len(to_process)} dates)"
    else:
        mode = f"FULL RUN ({len(to_process)} dates)"
    print(f"\n{sep}", flush=True)
    print(f"PROCESSING: {mode}", flush=True)
    print(sep, flush=True)

    with backfill_run(conn, "CR-AH") as run_id:
        results = []
        probe_results = []  # results from actually-fetched probe dates

        for i, entry in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}]", flush=True)
            do_probe = DRY_RUN and i < PROBE_FETCH_COUNT
            result = backfill_date(entry, conn, dry_run=DRY_RUN, probe_fetch=do_probe)
            results.append(result)
            if do_probe:
                probe_results.append(result)
            update_run_progress(conn, run_id, i + 1)

        # ── Summary stats ─────────────────────────────────────────────────────
        touch_counts = {}
        skipped_404 = [r for r in results if r.get("skipped_404")]
        for r in results:
            if r.get("skipped_404"):
                continue
            res = r.get("touch_resolution", "unknown")
            touch_counts[res] = touch_counts.get(res, 0) + 1

        smoke = {
            "mode": "dry_run" if DRY_RUN else "full_run",
            "probe_fetch_count": len(probe_results),
            "dates_processed": len(results),
            "dates_skipped_404": len(skipped_404),
            "touch_resolution_counts": touch_counts,
            "selection": {
                band: {
                    "n": len(selected_by_band[band]),
                    "train": sum(1 for e in selected_by_band[band] if e["partition"] == "train"),
                    "holdout": sum(1 for e in selected_by_band[band] if e["partition"] == "holdout"),
                }
                for band in ("near", "mid", "far")
            },
        }

        if probe_results:
            total_probe_calls = sum(r.get("total_api_calls", 0) for r in probe_results)
            total_probe_elapsed = sum(r.get("elapsed_s", 0.0) for r in probe_results)
            smoke["probe"] = {
                "dates": len(probe_results),
                "total_api_calls": total_probe_calls,
                "elapsed_s": total_probe_elapsed,
            }

        touches_any = sum(1 for r in results if r.get("touch_resolution") != "no_touch")
        assessment = (
            f"{'DRY-RUN' if DRY_RUN else 'FULL'}: "
            f"{len(results)} dates, touch_resolution={touch_counts}"
        )
        update_run_smoke(conn, run_id, smoke, assessment)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{sep}", flush=True)
    print(f"Done. run_id={run_id}", flush=True)
    fetched_results = [r for r in results if not r.get("skipped_404")]
    print(f"\nDates processed: {len(results)}  fetched: {len(fetched_results)}  skipped_404: {len(skipped_404)}", flush=True)
    if skipped_404:
        print("Skipped dates (ORATS 4xx — no contract data):", flush=True)
        for r in skipped_404:
            print(f"  {r['trade_date']}  {r['band']}/{r['partition']}  {r['short_opra']}", flush=True)
    print(f"\nTouch resolution breakdown ({len(fetched_results)} fetched dates):", flush=True)
    for resolution in ("rth_touch", "gap_touch", "afterhours_touch_retraced", "no_touch"):
        count = touch_counts.get(resolution, 0)
        print(f"  {resolution:<30}: {count}", flush=True)

    if probe_results:
        total_api_calls = sum(r.get("total_api_calls", 0) for r in probe_results)
        total_elapsed   = sum(r.get("elapsed_s", 0.0) for r in probe_results)
        per_date_calls   = total_api_calls / len(probe_results)
        per_date_elapsed = total_elapsed   / len(probe_results)
        print(f"\nPROBE FETCH STATS ({len(probe_results)} date(s) actually fetched):", flush=True)
        print(f"  API calls for probe date(s): {total_api_calls}  ({per_date_calls:.1f}/date)", flush=True)
        print(f"  Elapsed:                     {total_elapsed:.1f}s ({per_date_elapsed:.1f}s/date)", flush=True)
        # Extrapolate to 150 dates
        extrap_calls   = per_date_calls   * len(all_selected)
        extrap_elapsed = per_date_elapsed * len(all_selected)
        print(f"\n  ── EXTRAPOLATION TO {len(all_selected)} dates ──────────────────────────", flush=True)
        print(f"  Expected API calls: ~{extrap_calls:.0f}", flush=True)
        print(f"  Expected wall-clock: ~{extrap_elapsed/60:.0f} min  ({extrap_elapsed:.0f}s)", flush=True)

    if not DRY_RUN:
        total_bars = sum(r.get("bars_written", 0) for r in fetched_results)
        print(f"Total bars written: {total_bars}", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    main()
