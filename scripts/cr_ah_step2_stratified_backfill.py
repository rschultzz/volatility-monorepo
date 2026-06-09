"""CR-AH Step 2 — Stratified subsampled backfill.

Scope: ~50 dates per distance band (near <1.5σ / mid 1.5–2σ / far >2σ), ~150 total.
Selection: chronological stride within each band — deterministic, reproducible.
Per-date backfill: entry-day RTH, touch-window (if ES touch found), settlement.

Safety: BACKFILL_DATABASE_URL + dash_backfill_writer role enforced before any writes.
        DATABASE_URL is overridden to BACKFILL_DATABASE_URL so the options_cache repo
        (which reads DATABASE_URL) also uses the restricted role.
Idempotent: re-run = cache hits only (fetch_option_bars gap detection).

Usage:
    DRY_RUN = True  (default)  — prints selection, backfills first 3 dates (no actual fetch).
    DRY_RUN = False             — backfills all ~150 dates via ORATS API.

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
from packages.shared.options_cache.opra import format_opra
from packages.shared.backtest.models import distance_band

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRY_RUN: bool = True         # flip to False for the full ~150-date run

TARGET_PER_BAND: int = 50    # stride-select this many per band
SPLIT_DATE = date(2025, 8, 12)
TICKER = "SPX"
OPRA_ROOT = "SPX"            # ORATS uses 'SPX' root for all SPX expirations (per opra.py)
VERSION = CANONICAL_FEATURE_VERSION
DTE_TARGET: int = 15         # business days to expiry
TOUCH_WINDOW_MINS: int = 30  # ±30 min around touch minute for option quote window

_UTC = ZoneInfo("UTC")
_PT = ZoneInfo("America/Los_Angeles")

# ── HELPERS ───────────────────────────────────────────────────────────────────

def nth_business_day(from_date: date, n: int) -> date:
    """Return the nth weekday (Mon–Fri) after from_date. Holidays not excluded."""
    current = from_date
    count = 0
    while count < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    return current


def round_to_5pt(price: float) -> int:
    """Round to nearest 5pt — SPX listed strike spacing for 8-30 DTE options."""
    return round(price / 5) * 5


def utc_naive_to_pt_naive(utc_naive: datetime) -> datetime:
    """Convert naive UTC datetime to naive Pacific Time."""
    utc_aware = utc_naive.replace(tzinfo=_UTC)
    return utc_aware.astimezone(_PT).replace(tzinfo=None)


def stride_select(entries: list, target: int = 50) -> list:
    """Uniform chronological stride: select exactly min(len, target) entries."""
    n = len(entries)
    if n <= target:
        return list(entries)
    step = n / target
    return [entries[int(i * step)] for i in range(target)]


# ── SELECTION: load all signal dates and compute drift_target_distance_sigma ──

def load_signal_dates_with_sigma(conn) -> list[dict]:
    """Load all magnet-above dates and compute drift_target_distance_sigma per date.

    Returns a list of dicts:
        trade_date, drift_target, spot, sigma, band, partition
    """
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

def backfill_date(entry: dict, conn, dry_run: bool = True) -> dict:
    """Backfill one signal date's option quotes (entry-day, touch-window, settlement).

    ES touch detection uses ironbeam_es_1m_bars (already complete; no fetch needed).
    Option fetches use fetch_option_bars (idempotent, gap-aware).
    """
    trade_date = entry["trade_date"]
    drift_target = entry["drift_target"]
    band = entry["band"]
    partition = entry["partition"]

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

    # ── Touch-window (option quotes ±30 min around touch, clipped to RTH) ─────
    # If the ES touch falls entirely outside RTH (e.g. overnight / after-hours),
    # ORATS has no option quotes for that moment → skip the touch-window fetch.
    touch_window_start_pt: datetime | None = None
    touch_window_end_pt: datetime | None = None
    if touch_utc is not None:
        touch_pt = utc_naive_to_pt_naive(touch_utc)
        touch_date = touch_pt.date()
        day_open  = datetime(touch_date.year, touch_date.month, touch_date.day, 6, 30)
        day_close = datetime(touch_date.year, touch_date.month, touch_date.day, 13, 0)
        w_start = max(touch_pt - timedelta(minutes=TOUCH_WINDOW_MINS), day_open)
        w_end   = min(touch_pt + timedelta(minutes=TOUCH_WINDOW_MINS), day_close)
        if w_start < w_end:  # valid RTH window
            touch_window_start_pt = w_start
            touch_window_end_pt   = w_end

    # ── Settlement window: last 10 min of expiry RTH ──────────────────────────
    settle_start_pt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 12, 50)
    settle_end_pt   = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 13, 0)

    # ── Print plan ────────────────────────────────────────────────────────────
    if touch_utc is None:
        touch_label = "NONE"
    elif touch_window_start_pt is None:
        touch_label = f"{touch_utc:%Y-%m-%d %H:%M UTC} (after-hours — no option quotes)"
    else:
        touch_label = f"{touch_utc:%Y-%m-%d %H:%M UTC}"
    print(
        f"\n  [{partition}/{band}] {trade_date}  target={drift_target:.0f}  "
        f"σ={entry['sigma']:.2f}  expiry={expiry_date}",
        flush=True,
    )
    print(f"    short={short_opra}  long={long_opra}", flush=True)
    print(f"    entry-day:   {entry_start_pt:%H:%M}–{entry_end_pt:%H:%M} PT on {trade_date}", flush=True)
    print(f"    ES touch:    {touch_label}", flush=True)
    if touch_window_start_pt:
        print(
            f"    touch-window:{touch_window_start_pt:%Y-%m-%d %H:%M}–{touch_window_end_pt:%H:%M} PT",
            flush=True,
        )
    print(f"    settlement:  {settle_start_pt:%H:%M}–{settle_end_pt:%H:%M} PT on {expiry_date}", flush=True)

    if dry_run:
        return {
            "trade_date": str(trade_date),
            "band": band,
            "partition": partition,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "expiry_date": str(expiry_date),
            "short_opra": short_opra,
            "long_opra": long_opra,
            "touch_found": touch_utc is not None,
            "touch_utc": str(touch_utc) if touch_utc else None,
            "dry_run": True,
        }

    # ── FETCH: entry-day ──────────────────────────────────────────────────────
    print("    FETCH entry-day...", flush=True)
    r_entry = fetch_option_bars(
        opras, entry_start_pt, entry_end_pt,
        source="historical_backfill", record_empty_windows=True,
    )
    print(
        f"      bars_written={r_entry.bars_written}  "
        f"cache_hits={r_entry.cache_hits}  gaps_filled={r_entry.gaps_filled}",
        flush=True,
    )

    # ── FETCH: touch-window (only if touch found) ─────────────────────────────
    r_touch = None
    if touch_window_start_pt is not None:
        print("    FETCH touch-window...", flush=True)
        r_touch = fetch_option_bars(
            opras, touch_window_start_pt, touch_window_end_pt,
            source="historical_backfill", record_empty_windows=True,
        )
        print(
            f"      bars_written={r_touch.bars_written}  "
            f"cache_hits={r_touch.cache_hits}  gaps_filled={r_touch.gaps_filled}",
            flush=True,
        )

    # ── FETCH: settlement window ──────────────────────────────────────────────
    print("    FETCH settlement...", flush=True)
    r_settle = fetch_option_bars(
        opras, settle_start_pt, settle_end_pt,
        source="historical_backfill", record_empty_windows=True,
    )
    print(
        f"      bars_written={r_settle.bars_written}  "
        f"cache_hits={r_settle.cache_hits}  gaps_filled={r_settle.gaps_filled}",
        flush=True,
    )

    total_bars = (
        r_entry.bars_written
        + (r_touch.bars_written if r_touch else 0)
        + r_settle.bars_written
    )
    return {
        "trade_date": str(trade_date),
        "band": band,
        "partition": partition,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "expiry_date": str(expiry_date),
        "short_opra": short_opra,
        "long_opra": long_opra,
        "touch_found": touch_utc is not None,
        "touch_utc": str(touch_utc) if touch_utc else None,
        "bars_written": total_bars,
        "dry_run": False,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 65
    print(f"\n{sep}", flush=True)
    print("CR-AH Step 2 — Stratified subsampled backfill", flush=True)
    print(
        f"  DRY_RUN={DRY_RUN}  target_per_band={TARGET_PER_BAND}  dte_target={DTE_TARGET}",
        flush=True,
    )
    print(sep, flush=True)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

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

    # ── Band summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}", flush=True)
    print("SELECTION SUMMARY:", flush=True)
    total_tr = total_ho = 0
    for band in ("near", "mid", "far"):
        tr = sum(1 for e in selected_by_band[band] if e["partition"] == "train")
        ho = sum(1 for e in selected_by_band[band] if e["partition"] == "holdout")
        total_tr += tr; total_ho += ho
        print(f"  {band:<4}: n={len(selected_by_band[band]):>3}  train={tr:>2}  holdout={ho:>2}", flush=True)
    print(f"  TOTAL: n={len(all_selected):>3}  train={total_tr:>2}  holdout={total_ho:>2}", flush=True)

    # ── Process: dry-run = first 3, full run = all 150 ────────────────────────
    to_process = all_selected[:3] if DRY_RUN else all_selected
    mode = "DRY-RUN (first 3 dates)" if DRY_RUN else f"FULL RUN ({len(to_process)} dates)"
    print(f"\n{sep}", flush=True)
    print(f"PROCESSING: {mode}", flush=True)
    print(sep, flush=True)

    with backfill_run(conn, "CR-AH") as run_id:
        results = []
        for i, entry in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}]", flush=True)
            result = backfill_date(entry, conn, dry_run=DRY_RUN)
            results.append(result)
            update_run_progress(conn, run_id, i + 1)

        touches = sum(1 for r in results if r["touch_found"])
        smoke = {
            "mode": "dry_run" if DRY_RUN else "full_run",
            "dates_processed": len(results),
            "touches_found": touches,
            "selection": {
                band: {
                    "n": len(selected_by_band[band]),
                    "train": sum(1 for e in selected_by_band[band] if e["partition"] == "train"),
                    "holdout": sum(1 for e in selected_by_band[band] if e["partition"] == "holdout"),
                }
                for band in ("near", "mid", "far")
            },
        }
        assessment = (
            f"{'DRY-RUN' if DRY_RUN else 'FULL'}: "
            f"{len(results)} dates processed, {touches} with ES touch"
        )
        update_run_smoke(conn, run_id, smoke, assessment)

    print(f"\n{sep}", flush=True)
    print(f"Done. run_id={run_id}", flush=True)
    print(f"ES touch found: {touches}/{len(results)}", flush=True)
    if not DRY_RUN:
        total_bars = sum(r.get("bars_written", 0) for r in results)
        print(f"Total bars written: {total_bars}", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    main()
