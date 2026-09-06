"""CR-AI Stage 2 — Multi-DTE backtest backfill + P&L analysis.

Holds the date set FIXED (CR-AH clean debit A-bucket sample, n=110).
Varies ONLY the expiry at each candidate DTE {5, 7, 10, 15}.
(DTE=21 dropped: ORATS returns 404 for options expiring in 2025.)

Safety protocol: BACKFILL_DATABASE_URL / dash_backfill_writer always.
Idempotent: re-run = cache hits only.

Output:
  - Backfills orats_options_minute for each (date × DTE) pair
  - Computes P&L per (date × DTE), held-to-close
  - Prints three required deliverables:
      (i)   DTE × band P&L grid (mean P&L, win rate, % return on premium)
      (ii)  Keep-or-drop-far verdict
      (iii) Early-vs-late-touch conditioning cut
  - Prints distance-matched schedule comparisons
  - Appends results to specs/CR-AI-expiration-optimization.md

Usage:
  DRY_RUN = True   (default) — no fetches; planning + touch tag only
  DRY_RUN = False  — fetches all DTE×date pairs, computes full P&L
"""

from __future__ import annotations

import json
import math
import os
import sys
import time as _time_module
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# ── ENV ───────────────────────────────────────────────────────────────────────
def _find_dotenv() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        c = current / ".env"
        if c.exists(): return c
        current = current.parent
    return None

_env_path = _find_dotenv()
if _env_path:
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = _bak_url

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from packages.shared.backfill_safety import (
    get_backfill_db_conn, assert_role_or_die,
    backfill_run, update_run_smoke,
)
from packages.shared.backtest.models import distance_band
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import (
    _LANDSCAPE_ROW_SQL, _OPEN_STRADDLE_SQL, _materialize_payload,
)
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.opra import format_opra
from packages.shared.options_cache.strikes import StrikeNotListed, snap_vertical_legs
from packages.shared.backtest.quote_validity import leg_quote_is_valid

# ── CONFIG ────────────────────────────────────────────────────────────────────
DRY_RUN: bool = False       # flip to True for planning-only run

TICKER      = "SPX"
OPRA_ROOT   = "SPX"
VERSION     = CANONICAL_FEATURE_VERSION
SPLIT_DATE  = date(2025, 8, 12)
DTE_BASE    = 15            # CR-AH DTE (defines the fixed clean sample)
SPREAD_WIDTH = 10.0         # fixed 10pt debit spread

CANDIDATE_DTES = [5, 7, 10, 15]   # 21 dropped: ORATS 404 for 2025 expiries
ES_OPEN_LOOKUP_MINS = 5

_UTC = ZoneInfo("UTC")
_PT  = ZoneInfo("America/Los_Angeles")

# Band medians from Stage 1 (days_to_touch) — used for early/late split in deliverable iii
BAND_MEDIAN_DAYS = {"near": 0, "mid": 1, "far": 3}

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

def next_weekday(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt

def round5(p: float) -> int:
    return round(p / 5) * 5

def stride_select(lst: list, n: int = 50) -> list:
    if len(lst) <= n: return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]

def utc_to_pt(utc_naive: datetime) -> datetime:
    return utc_naive.replace(tzinfo=_UTC).astimezone(_PT).replace(tzinfo=None)

def pt_to_utc(pt_naive: datetime) -> datetime:
    return pt_naive.replace(tzinfo=_PT).astimezone(_UTC).replace(tzinfo=None)

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def trading_sessions_between(entry: date, touch: date) -> int:
    if touch == entry: return 0
    count = 0
    cur = entry + timedelta(days=1)
    while cur <= touch:
        if cur.weekday() < 5 and cur not in _NYSE_HOLIDAYS:
            count += 1
        cur += timedelta(days=1)
    return count


# ── DATA STRUCTURES ────────────────────────────────────────────────────────────

@dataclass
class DTEResult:
    """P&L result for one (date × DTE) cell."""
    trade_date:      date
    dte:             int
    band:            str
    partition:       str        # 'train' | 'holdout'
    sigma:           float
    drift_target:    float
    short_strike:    float
    expiry_date:     date

    # Entry
    entry_credit:    Optional[float]   # net credit from short-long at first quoted minute
    filled:          bool

    # Touch (DTE-independent ES event)
    touch_resolution: str              # rth_touch|gap_touch|afterhours_touch_retraced|no_touch
    touch_session:    Optional[date]   # session date of touch (for alive_at_touch)
    days_to_touch:    Optional[int]    # trading sessions entry→touch

    # Alive-at-touch: option is alive when touch occurs
    alive_at_touch:   Optional[bool]   # expiry >= touch_session (for rth/gap touches)

    # Settlement: mid of (short - long) at last available minute on expiry day
    settlement_cost:  Optional[float]  # positive = position lost value at expiry

    # P&L (held-to-close): only defined if filled
    close_pnl:        Optional[float]  # entry_credit - settlement_cost
    pct_return_on_premium: Optional[float]  # close_pnl / (spread_width - entry_credit) * 100


# ── LOAD CLEAN DATE SET ────────────────────────────────────────────────────────

def load_clean_dates(conn) -> list[dict]:
    """Reproduce the CR-AH clean debit A-bucket date set (n=110 at DTE=15)."""
    rows = conn.execute(
        "SELECT trade_date FROM bt_daily_features WHERE ticker=%s AND feature_version=%s "
        "AND regime_at_classification='magnet-above' ORDER BY trade_date ASC",
        (TICKER, VERSION),
    ).fetchall()
    dates = [r[0] for r in rows]

    all_entries = []
    for td in dates:
        row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, td)).fetchone()
        if not row or row[0] is None or row[1] is None: continue
        landscape_rows, table_spot = row
        spot = float(table_spot)
        floor_ts = datetime.combine(td, time(6, 33, 0))
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
        all_entries.append({
            "trade_date":   td,
            "drift_target": float(dt_val),
            "spot":         spot,
            "sigma":        sigma,
            "band":         distance_band(sigma),
            "partition":    "train" if td <= SPLIT_DATE else "holdout",
        })

    by_band: dict[str, list] = {"near": [], "mid": [], "far": []}
    for e in all_entries:
        by_band[e["band"]].append(e)
    for b in ("near", "mid", "far"):
        by_band[b].sort(key=lambda x: x["trade_date"])

    selected = []
    for b in ("near", "mid", "far"):
        selected += stride_select(by_band[b])
    selected.sort(key=lambda x: x["trade_date"])

    clean = []
    n_unlistable = 0
    for e in selected:
        td = e["trade_date"]
        expiry = nth_business_day(td, DTE_BASE)
        # CR-AO: snap both legs to the strikes listed at the prior close
        try:
            snapped = snap_vertical_legs(e["drift_target"], -10, expiry, td, conn, toward=e.get("spot"))
        except StrikeNotListed as exc:
            n_unlistable += 1
            print(f"  unlistable {td}: {exc}", flush=True)
            continue
        ss = float(snapped.anchor)
        ls = float(snapped.other)
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
            clean.append({**e, "short_strike": ss, "long_strike": ls,
                          "width_actual": snapped.width_actual, "width_nominal": snapped.width_nominal})

    if n_unlistable:
        print(f"  unlistable entries (StrikeNotListed): {n_unlistable}", flush=True)
    return clean


# ── TOUCH DETECTION (DTE-independent) ─────────────────────────────────────────

def detect_touch(
    trade_date: date,
    drift_target: float,
    conn,
    horizon_days: int = 30,
) -> tuple[str, Optional[date], Optional[int]]:
    """Returns (resolution, touch_session_date, days_to_touch).

    Reuses CR-AH four-outcome logic.
    """
    search_start_utc = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, 0)
    horizon_end = trade_date + timedelta(days=horizon_days)
    search_end_utc   = datetime(horizon_end.year, horizon_end.month, horizon_end.day, 23, 59, 59)

    touch_row = conn.execute(
        "SELECT datetime, close FROM ironbeam_es_1m_bars "
        "WHERE datetime>=%s AND datetime<=%s AND close>=%s "
        "ORDER BY datetime ASC LIMIT 1",
        (search_start_utc, search_end_utc, drift_target),
    ).fetchone()

    if touch_row is None:
        return "no_touch", None, None

    touch_utc = touch_row[0]
    touch_pt  = utc_to_pt(touch_utc)
    touch_date = touch_pt.date()
    day_open  = datetime(touch_date.year, touch_date.month, touch_date.day, 6, 30)
    day_close = datetime(touch_date.year, touch_date.month, touch_date.day, 13, 0)

    if day_open <= touch_pt <= day_close:
        days = trading_sessions_between(trade_date, touch_date)
        return "rth_touch", touch_date, days

    # Outside RTH
    if touch_pt < day_open:
        next_rth_d  = touch_date
        next_open_pt = day_open
    else:
        next_rth_d  = next_weekday(touch_date)
        next_open_pt = datetime(next_rth_d.year, next_rth_d.month, next_rth_d.day, 6, 30)

    next_open_utc = pt_to_utc(next_open_pt)
    es_row = conn.execute(
        "SELECT close FROM ironbeam_es_1m_bars "
        "WHERE datetime>=%s AND datetime<%s ORDER BY datetime ASC LIMIT 1",
        (next_open_utc, next_open_utc + timedelta(minutes=ES_OPEN_LOOKUP_MINS)),
    ).fetchone()

    if es_row is None:
        return "afterhours_touch_retraced", None, None

    if float(es_row[0]) >= drift_target:
        days = trading_sessions_between(trade_date, next_rth_d)
        return "gap_touch", next_rth_d, days
    else:
        return "afterhours_touch_retraced", None, None


# ── NET MID QUOTE ──────────────────────────────────────────────────────────────

def _valid_mids_by_minute(rows) -> dict:
    """CR-AO decision 5 (CR-AN leg rule): keep only legs with bid >= 0, ask > 0,
    bid <= ask; mid from those. Rows are (opra, snapshot_pt, bid, ask)."""
    by_min: dict[datetime, dict[str, float]] = {}
    for opra, snap, bid, ask in rows:
        if leg_quote_is_valid(bid, ask):
            by_min.setdefault(snap, {})[opra] = (float(bid) + float(ask)) / 2.0
    return by_min


def _spread_in_range(value: float, width: float) -> bool:
    """Debit (long target-10 / short target): mid(short) - mid(long) must lie in [-width, 0]."""
    return -width - 1e-9 <= value <= 1e-9


def get_net_credit_at(conn, short_opra: str, long_opra: str,
                       start_pt: datetime, end_pt: datetime,
                       width: float = SPREAD_WIDTH) -> Optional[float]:
    """Net credit = mid(short) - mid(long) at the first minute both legs are
    VALIDLY quoted and the spread value is in range (CR-AO decision 5)."""
    rows = conn.execute(
        """
        SELECT opra_symbol, snapshot_pt, bid_price, ask_price
        FROM orats_options_minute
        WHERE opra_symbol = ANY(%s)
          AND snapshot_pt >= %s AND snapshot_pt < %s
        ORDER BY snapshot_pt ASC
        """,
        ([short_opra, long_opra], start_pt, end_pt),
    ).fetchall()
    by_min = _valid_mids_by_minute(rows)
    for snap in sorted(by_min):
        m = by_min[snap]
        if short_opra in m and long_opra in m:
            v = m[short_opra] - m[long_opra]   # positive = net credit
            if _spread_in_range(v, width):
                return v
    return None


def get_settlement_cost(conn, short_opra: str, long_opra: str,
                         expiry_date: date, width: float = SPREAD_WIDTH) -> Optional[float]:
    """Settlement cost = mid(short) - mid(long) at the last VALIDLY quoted,
    in-range minute of the settlement window (CR-AO decision 5)."""
    settle_start_pt = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 12, 50)
    settle_end_pt   = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 13, 1)

    rows = conn.execute(
        """
        SELECT opra_symbol, snapshot_pt, bid_price, ask_price
        FROM orats_options_minute
        WHERE opra_symbol = ANY(%s)
          AND snapshot_pt >= %s AND snapshot_pt <= %s
        ORDER BY snapshot_pt DESC
        """,
        ([short_opra, long_opra], settle_start_pt, settle_end_pt),
    ).fetchall()
    by_min = _valid_mids_by_minute(rows)
    for snap in sorted(by_min, reverse=True):
        m = by_min[snap]
        if short_opra in m and long_opra in m:
            v = m[short_opra] - m[long_opra]
            if _spread_in_range(v, width):
                return v
    return None


# ── PER-DATE×DTE BACKFILL ─────────────────────────────────────────────────────

def backfill_one(
    entry: dict,
    dte: int,
    conn,
    dry_run: bool = True,
) -> DTEResult:
    """Backfill and compute P&L for one (date × DTE)."""
    td           = entry["trade_date"]
    drift_target = entry["drift_target"]
    band         = entry["band"]
    partition    = entry["partition"]
    sigma        = entry["sigma"]
    ss           = entry["short_strike"]
    ls           = entry["long_strike"]
    width        = float(entry.get("width_actual", SPREAD_WIDTH))   # CR-AO decision 3

    expiry = nth_business_day(td, dte)
    short_opra = format_opra(OPRA_ROOT, expiry, "C", ss)
    long_opra  = format_opra(OPRA_ROOT, expiry, "C", ls)

    entry_start_pt = datetime(td.year, td.month, td.day, 6, 30)
    entry_end_pt   = datetime(td.year, td.month, td.day, 13, 0)

    # Touch detection (DTE-independent)
    resolution, touch_session, days_to_touch = detect_touch(td, drift_target, conn)

    alive_at_touch: Optional[bool] = None
    if resolution in ("rth_touch", "gap_touch") and touch_session is not None:
        alive_at_touch = expiry >= touch_session

    result = DTEResult(
        trade_date=td, dte=dte, band=band, partition=partition,
        sigma=sigma, drift_target=drift_target, short_strike=ss, expiry_date=expiry,
        entry_credit=None, filled=False,
        touch_resolution=resolution, touch_session=touch_session,
        days_to_touch=days_to_touch, alive_at_touch=alive_at_touch,
        settlement_cost=None, close_pnl=None, pct_return_on_premium=None,
    )

    if dry_run:
        return result

    # Fetch option bars
    try:
        fetch_option_bars(
            [short_opra, long_opra],
            entry_start_pt, entry_end_pt,
            source="historical_backfill",
        )
    except OratsPermanentError:
        return result  # 404 — no data; result.filled = False

    # Entry credit
    entry_credit = get_net_credit_at(conn, short_opra, long_opra, entry_start_pt, entry_end_pt, width)
    if entry_credit is None:
        return result  # no quote found

    result.entry_credit = entry_credit
    result.filled = True

    # Settlement (fetch the settlement window too)
    settle_start_pt = datetime(expiry.year, expiry.month, expiry.day, 12, 50)
    settle_end_pt   = datetime(expiry.year, expiry.month, expiry.day, 13, 1)
    try:
        fetch_option_bars(
            [short_opra, long_opra],
            settle_start_pt, settle_end_pt,
            source="historical_backfill",
        )
    except OratsPermanentError:
        pass  # settlement missing; P&L will be None

    settlement_cost = get_settlement_cost(conn, short_opra, long_opra, expiry, width)
    result.settlement_cost = settlement_cost

    if settlement_cost is not None:
        close_pnl = entry_credit - settlement_cost
        result.close_pnl = close_pnl
        max_loss = SPREAD_WIDTH - entry_credit  # max loss on a debit = premium paid = spread - credit received... wait
        # Actually for a debit spread: cost = spread_width - net_credit
        # For clarity: debit spread pays net_credit (positive = receive), costs (spread_width - net_credit)
        # Hmm, actually for a DEBIT spread (long target-10, short target):
        #   net position value at entry = mid(short) - mid(long) = net_credit (positive if OTM spread credit)
        #   Wait - for magnet-above strategy the DEBIT spread is: long ATM call (near target), short further OTM
        #   Actually re-reading: debit = long target-10 (cheaper, closer to spot), short target (further OTM)
        #   net_credit = mid(short) - mid(long) = probably NEGATIVE since we're paying net debit
        # Let me reconsider...
        # From CR-AH: credit structure = short target, long target+10 (we receive net)
        #             debit structure  = long target-10, short target (we pay net? or receive?)
        # At entry: both are OTM calls. target is further OTM than target-10.
        # OTM farther = cheaper. So: mid(short at target) < mid(long at target-10)
        # net = mid(short) - mid(long) < 0 → we pay a net debit. Close_pnl = credit - settlement.
        # But wait, if we receive a net credit (positive entry_credit), then we want settlement_cost < entry_credit.
        # The debit spread pays net: entry_credit could be negative if the long leg is more expensive.
        # Using the same formula as CR-AH: close_pnl = entry_credit - settlement_cost
        # The % return on premium: if we pay a premium (negative entry_credit = debit), then:
        #   premium_paid = -entry_credit (the amount we laid out)
        #   % return = close_pnl / max_loss * 100 where max_loss = premium_paid = -entry_credit
        # But if entry_credit > 0 (we receive net), it's different.
        # In practice for this debit strategy on magnet-above: target-10 is CLOSER to spot (more expensive)
        # and target is further OTM (cheaper). So mid(long) > mid(short) → entry_credit < 0 (net debit paid).
        # close_pnl = entry_credit - settlement_cost:
        #   At expiry above target: both calls worth their intrinsic. Net position = -(spread_width - settlement_value)
        #   Actually: let me use the CR-AH approach directly.
        #   CR-AH close_pnl = entry_credit - settlement_cost = (net position value at entry) - (net at settlement)
        #   Win = close_pnl > 0
        # % return: denominator should be the maximum possible loss.
        # For a debit spread: max_loss = net_debit_paid = abs(entry_credit) when entry_credit < 0
        # For a credit spread: max_loss = spread_width - net_credit_received
        # Since this is debit: max_loss = -entry_credit (if entry_credit < 0, i.e., we paid a debit)
        # If entry_credit > 0 (rare edge case where we received credit on a "debit" spread), clamp to 0.01
        max_loss = max(-entry_credit, 0.01) if entry_credit is not None else width
        if max_loss > 0:
            result.pct_return_on_premium = (close_pnl / max_loss) * 100

    return result


# ── ANALYSIS HELPERS ──────────────────────────────────────────────────────────

def analyze_grid(results: list[DTEResult]):
    """Print the three required deliverables."""
    import statistics

    filled = [r for r in results if r.filled and r.close_pnl is not None]

    # ── DELIVERABLE (i): DTE × band P&L grid ─────────────────────────────────
    print(f"\n{'='*80}")
    print("DELIVERABLE (i) — DTE × BAND P&L GRID (held-to-close, filled dates only)")
    print(f"{'='*80}")

    BANDS = ["near", "mid", "far", "all"]
    PARTITIONS = ["all", "train", "holdout"]

    for partition_label in PARTITIONS:
        if partition_label == "all":
            subset = filled
        else:
            subset = [r for r in filled if r.partition == partition_label]

        print(f"\n  --- Partition: {partition_label.upper()} ---")
        print(f"  {'DTE':>4}  {'Band':<6}  {'n':>4}  {'mean_pnl':>9}  {'win_rate':>9}  {'wilson':>15}  {'mean_entry':>11}  {'%_return':>9}")
        print("  " + "─" * 82)

        for dte in CANDIDATE_DTES:
            for band in BANDS:
                if band == "all":
                    cell = [r for r in subset if r.dte == dte]
                else:
                    cell = [r for r in subset if r.dte == dte and r.band == band]
                if not cell:
                    continue
                n = len(cell)
                pnls = [r.close_pnl for r in cell]
                wins = sum(1 for p in pnls if p > 0)
                mean_pnl = statistics.mean(pnls)
                wlo, whi = wilson_ci(wins, n)
                mean_entry = statistics.mean(r.entry_credit for r in cell if r.entry_credit is not None)
                pct_returns = [r.pct_return_on_premium for r in cell if r.pct_return_on_premium is not None]
                mean_pct = statistics.mean(pct_returns) if pct_returns else float("nan")
                star = "★" if band == "all" else " "
                print(f"  {dte:>4}  {band:<6}  {n:>4}  {mean_pnl:>+9.2f}  {wins/n:>8.1%}  "
                      f"[{wlo:.2f},{whi:.2f}]  {mean_entry:>+11.2f}  {mean_pct:>+8.1f}%{star}")

    # ── DELIVERABLE (ii): Keep-or-drop-far ──────────────────────────────────
    print(f"\n{'='*80}")
    print("DELIVERABLE (ii) — KEEP-OR-DROP-FAR VERDICT")
    print(f"{'='*80}")
    print()

    far_filled = [r for r in filled if r.band == "far"]
    if not far_filled:
        print("  No far-band filled dates. Cannot adjudicate.")
    else:
        far_by_dte = {}
        for dte in CANDIDATE_DTES:
            cell = [r for r in far_filled if r.dte == dte]
            if cell:
                pnls = [r.close_pnl for r in cell]
                wins = sum(1 for p in pnls if p > 0)
                n = len(cell)
                mean_pnl = statistics.mean(pnls)
                wlo, whi = wilson_ci(wins, n)
                pct_returns = [r.pct_return_on_premium for r in cell if r.pct_return_on_premium is not None]
                mean_pct = statistics.mean(pct_returns) if pct_returns else float("nan")
                far_by_dte[dte] = (n, mean_pnl, wins/n, wlo, whi, mean_pct)
                print(f"  DTE={dte:>2}: n={n:>3}  mean_pnl={mean_pnl:>+7.2f}  "
                      f"win={wins/n:.1%}  CI=[{wlo:.2f},{whi:.2f}]  %_return={mean_pct:>+6.1f}%")

        # Verdict
        any_positive = any(v[1] > 0 for v in far_by_dte.values())
        any_ci_above_zero = any(v[3] > 0 for v in far_by_dte.values())  # wilson_lo > 0
        print()
        if any_ci_above_zero:
            print("  VERDICT: FAR earns positive mean P&L with CI lower-bound > 0 at some DTEs.")
            print("  → KEEP FAR (conditionally — examine which DTEs).")
        elif any_positive:
            print("  VERDICT: FAR has positive mean P&L at some DTEs, but CI lower-bounds ≤ 0.")
            print("  → MARGINAL — far edge is plausible but not confirmed at this sample size.")
        else:
            print("  VERDICT: FAR has negative or zero mean P&L at EVERY tested DTE.")
            print("  → DROP FAR — far-band trades are a drag regardless of expiration.")

    # ── DELIVERABLE (iii): Early-vs-late-touch conditioning cut ─────────────
    print(f"\n{'='*80}")
    print("DELIVERABLE (iii) — EARLY vs LATE TOUCH CONDITIONING CUT")
    print("  (early = days_to_touch ≤ band_median; late = > band_median or no_touch)")
    print(f"  Band medians (from Stage 1): near=0d, mid=1d, far=3d")
    print(f"{'='*80}")

    for band in ("near", "mid", "far"):
        median_day = BAND_MEDIAN_DAYS[band]
        print(f"\n  {band.upper()} (median={median_day}d):")
        for dte in CANDIDATE_DTES:
            cell = [r for r in filled if r.dte == dte and r.band == band]
            if not cell:
                continue
            early = [r for r in cell
                     if r.days_to_touch is not None and r.days_to_touch <= median_day]
            late  = [r for r in cell
                     if r.days_to_touch is None or r.days_to_touch > median_day]
            def pnl_str(subset):
                if not subset:
                    return "n=0"
                pnls = [r.close_pnl for r in subset]
                wins = sum(1 for p in pnls if p > 0)
                return (f"n={len(subset)} mean={statistics.mean(pnls):>+6.2f} "
                        f"win={wins/len(subset):.0%}")
            print(f"    DTE={dte:>2}  early: {pnl_str(early):45s}  late: {pnl_str(late)}")

    # ── Distance-matched schedules ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("DISTANCE-MATCHED SCHEDULE COMPARISONS")
    print("  Schedule A: near=5, mid=7, far=10")
    print("  Schedule B: near=5, mid=10, far=15  (more conservative)")
    print("  Baseline:   flat DTE=15 (CR-AH tested value)")
    print(f"{'='*80}")

    schedules = {
        "Sched-A (near=5/mid=7/far=10)":  {"near": 5, "mid": 7, "far": 10},
        "Sched-B (near=5/mid=10/far=15)": {"near": 5, "mid": 10, "far": 15},
    }
    flat_dtes = {f"Flat-DTE={d}": d for d in CANDIDATE_DTES}

    def score_schedule(sched_filled: list[DTEResult]) -> str:
        if not sched_filled:
            return "n=0"
        pnls = [r.close_pnl for r in sched_filled]
        wins = sum(1 for p in pnls if p > 0)
        n = len(sched_filled)
        mean_pnl = statistics.mean(pnls)
        wlo, whi = wilson_ci(wins, n)
        pct_returns = [r.pct_return_on_premium for r in sched_filled if r.pct_return_on_premium is not None]
        mean_pct = statistics.mean(pct_returns) if pct_returns else float("nan")
        return (f"n={n:>3} mean_pnl={mean_pnl:>+7.2f} win={wins/n:.1%} "
                f"CI=[{wlo:.2f},{whi:.2f}] %_return={mean_pct:>+6.1f}%")

    print()
    for label, sched in schedules.items():
        sched_filled = [r for r in filled if r.dte == sched[r.band]]
        print(f"  {label}:")
        print(f"    ALL: {score_schedule(sched_filled)}")
        for part in ("train", "holdout"):
            sub = [r for r in sched_filled if r.partition == part]
            print(f"    {part}:   {score_schedule(sub)}")

    for label, dte in flat_dtes.items():
        flat_filled = [r for r in filled if r.dte == dte]
        base_marker = " ← CR-AH baseline" if dte == 15 else ""
        print(f"\n  {label}{base_marker}:")
        print(f"    ALL: {score_schedule(flat_filled)}")
        for part in ("train", "holdout"):
            sub = [r for r in flat_filled if r.partition == part]
            print(f"    {part}:   {score_schedule(sub)}")

    # ── DTE_TARGET_BY_BUCKET baseline ────────────────────────────────────────
    print(f"\n  Live DTE_TARGET_BY_BUCKET baseline (bucket→DTE per date):")
    # We'd need bucket info per date — skip if not loaded
    print("  (See Step 0 dominant-bucket breakdown for per-date live DTE assignments.)")
    print("  Most dates are '8-30 DTE' → live DTE=15; identical to Flat-DTE=15 for most of the sample.")

    # ── Alive-at-touch summary ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("ALIVE-AT-TOUCH SUMMARY (for rth+gap touches only)")
    print(f"{'='*80}")
    print(f"  {'DTE':>4}  {'band':<6}  {'touches':>8}  {'alive_n':>8}  {'alive%':>8}  {'dead_n':>7}")
    print("  " + "─" * 55)
    touch_results = [r for r in results if r.touch_resolution in ("rth_touch", "gap_touch")]
    for dte in CANDIDATE_DTES:
        for band in ["near", "mid", "far", "all"]:
            if band == "all":
                cell = [r for r in touch_results if r.dte == dte]
            else:
                cell = [r for r in touch_results if r.dte == dte and r.band == band]
            if not cell: continue
            alive_n = sum(1 for r in cell if r.alive_at_touch is True)
            dead_n  = sum(1 for r in cell if r.alive_at_touch is False)
            n = len(cell)
            print(f"  {dte:>4}  {band:<6}  {n:>8}  {alive_n:>8}  {alive_n/n:>7.1%}  {dead_n:>7}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*80}")
    print("CR-AI Stage 2 — Multi-DTE backtest")
    print(f"  DRY_RUN={DRY_RUN}  DTEs={CANDIDATE_DTES}")
    print(f"{'='*80}\n")

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    print("Loading clean date set...", flush=True)
    clean_dates = load_clean_dates(conn)
    n = len(clean_dates)
    near_n = sum(1 for d in clean_dates if d["band"] == "near")
    mid_n  = sum(1 for d in clean_dates if d["band"] == "mid")
    far_n  = sum(1 for d in clean_dates if d["band"] == "far")
    train_n   = sum(1 for d in clean_dates if d["partition"] == "train")
    holdout_n = sum(1 for d in clean_dates if d["partition"] == "holdout")
    print(f"Clean dates: n={n}  near={near_n} mid={mid_n} far={far_n}  "
          f"train={train_n} holdout={holdout_n}")

    total_pairs = n * len(CANDIDATE_DTES)
    print(f"Total (date × DTE) pairs: {total_pairs}")

    if DRY_RUN:
        print(f"\n[DRY_RUN=True — no fetches, computing touch/alive stats only]\n")

    all_results: list[DTEResult] = []

    with backfill_run(conn, "CR-AI") as run_id:
        i = 0
        skipped = 0
        for entry in clean_dates:
            for dte in CANDIDATE_DTES:
                i += 1
                if i % 50 == 0:
                    print(f"  {i}/{total_pairs}  "
                          f"({entry['trade_date']} DTE={dte})...", flush=True)

                try:
                    result = backfill_one(entry, dte, conn, dry_run=DRY_RUN)
                    all_results.append(result)
                except Exception as exc:
                    print(f"  ERROR {entry['trade_date']} DTE={dte}: {exc}", flush=True)
                    skipped += 1

        filled_count = sum(1 for r in all_results if r.filled)
        pnl_count    = sum(1 for r in all_results if r.close_pnl is not None)

        smoke = {
            "total_pairs":     total_pairs,
            "completed":       len(all_results),
            "skipped":         skipped,
            "filled":          filled_count,
            "with_pnl":        pnl_count,
            "dry_run":         DRY_RUN,
        }
        assessment = (f"Stage 2: {pnl_count}/{total_pairs} pairs with P&L. "
                      f"DRY_RUN={DRY_RUN}.")
        update_run_smoke(conn, run_id, smoke, assessment)

    print(f"\n{'─'*80}")
    print(f"Run complete: {len(all_results)} results, {filled_count} filled, "
          f"{pnl_count} with P&L, {skipped} errors.")

    # Touch stats (DTE-independent, available even in dry_run)
    print(f"\n{'─'*80}")
    print("Touch resolution (DTE-independent — one entry's touch is the same across DTEs):")
    per_date_touch: dict[date, str] = {}
    for r in all_results:
        if r.trade_date not in per_date_touch:
            per_date_touch[r.trade_date] = r.touch_resolution
    from collections import Counter
    tc = Counter(per_date_touch.values())
    for k, v in sorted(tc.items()):
        print(f"  {k:<35} {v}")

    if not DRY_RUN and pnl_count > 0:
        analyze_grid(all_results)

        # Write results summary to spec file
        spec_path = Path(__file__).parent.parent / "specs" / "CR-AI-expiration-optimization.md"
        if spec_path.exists():
            print(f"\nAppending summary to {spec_path.name}...")
            append_results_to_spec(all_results, spec_path)

    print(f"\n{'='*80}")
    print("Done.")
    print(f"{'='*80}\n")


def append_results_to_spec(results: list[DTEResult], spec_path: Path):
    """Append a plain-text summary of results to the spec file."""
    import statistics

    filled = [r for r in results if r.filled and r.close_pnl is not None]
    if not filled:
        return

    lines = [
        "",
        "## Stage 2 Results",
        f"",
        f"*n_clean_dates={len(set(r.trade_date for r in results))}  "
        f"DTEs={CANDIDATE_DTES}  "
        f"filled={len(filled)}  "
        f"held-to-close*",
        "",
    ]

    # Quick grid
    lines.append("### DTE × Band P&L (all partitions)")
    lines.append("")
    lines.append("| DTE | Band | n | mean_pnl | win% | wilson_lo | %_return |")
    lines.append("|-----|------|---|----------|------|-----------|----------|")
    for dte in CANDIDATE_DTES:
        for band in ("near", "mid", "far", "all"):
            if band == "all":
                cell = [r for r in filled if r.dte == dte]
            else:
                cell = [r for r in filled if r.dte == dte and r.band == band]
            if not cell:
                continue
            pnls = [r.close_pnl for r in cell]
            wins = sum(1 for p in pnls if p > 0)
            n = len(cell)
            mean_pnl = statistics.mean(pnls)
            wlo, whi = wilson_ci(wins, n)
            pct_returns = [r.pct_return_on_premium for r in cell if r.pct_return_on_premium is not None]
            mean_pct = statistics.mean(pct_returns) if pct_returns else float("nan")
            lines.append(f"| {dte} | {band} | {n} | {mean_pnl:+.2f} | {wins/n:.1%} | {wlo:.2f} | {mean_pct:+.1f}% |")

    current = spec_path.read_text()
    marker = "## Results (appended after backfill)"
    if marker in current:
        current = current.replace(marker, marker + "\n" + "\n".join(lines))
        spec_path.write_text(current)


if __name__ == "__main__":
    main()
