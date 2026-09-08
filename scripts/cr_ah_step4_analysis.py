"""CR-AH Step 4 — Two-structure × two-axis analysis.

Computes the full per-structure edge backtest result over clean A-bucket dates:
  - DEBIT  (debit_spread_to_target):  long target−10, short target. n=110
  - CREDIT (directional_spread_to_target): short target, long target+10. n=106

For each clean date × structure:
  - Runs entry-crawl from orats_options_minute (full RTH)
  - Detects ES touch via ironbeam_es_1m_bars
  - Computes touch-exit P&L (debit only) and close P&L
  - Computes baseline (fixed open entry, no edge gate)
  - Tags with post-touch pattern from compute_structural_probability()

Threshold sweep on TRAIN only (edge values 0/5/10/15/20 percentage pts).
Holdout read ONCE at the chosen threshold (highest train baseline-beat on close P&L).

Output:
  - Full result tables printed to stdout
  - Result appended to specs/CR-AH-per-structure-edge-backtest.md
  - Aggregate stats written to bt_edge_backtest_results (needs CREATE TABLE first)

Usage: PYTHONUNBUFFERED=1 python3 -u scripts/cr_ah_step4_analysis.py
         [--universe-end YYYY-MM-DD] [--split-date YYYY-MM-DD] [--train-only]
         [--no-persist] [--cr-id CR-AH] [--structural-prob-mode {full,walk-forward}]
         [--seed 20260905]

CR-AL flags (2026-09-05):
  --universe-end          drop signal dates after this date before stratified
                          selection (pin the universe to reproduce June's train set)
  --split-date            train/holdout split: partition = 'train' if trade_date
                          <= split else 'holdout' (default 2026-06-05 per ADR
                          "Holdout Split Moves to 2026-06-05"; June used 2025-08-12).
                          With --universe-end == --split-date there is no holdout;
                          every holdout print path says so and Phase 7 persists
                          train cells only.
  --train-only            drop holdout entries after selection; by-band prints
                          train only; Summary A/B replaced by "holdout not read"
  --no-persist            skip Phase 7 (no bt_edge_backtest_results writes)
  --cr-id                 cr_id for the bt_backfill_runs row (default CR-AH)
  --structural-prob-mode  walk-forward (default): before_date=trade_date;
                          full: allow_lookahead=True (as CR-AH ran in June)
  --seed                  bootstrap seed for the Summary D CI

CR-AU flags (2026-09-07):
  --holdout-read PATH     decision 4: holdout P&L is printed / persisted only when this
                          points at the pre-registration document AND the holdout
                          magnet-above computed count is >= 60. Without it (or below
                          the threshold) holdout rows show n per band and dates
                          matured only: "holdout: n=<k>, unread (threshold 60)".
                          A path that does not exist raises before any DB work.
  --selection-only        stop after Phase 1 (universe, stratified selection, clean
                          filter, unlistable list, holdout gate line); no cells.
                          Used by run_reference_rerun.py --dry-run.
  Fees (decision 5): compute_pnl also returns close_pnl_net = close_pnl −
  VERTICAL_FEE_PTS (2 legs × 2 sides × FEE_PER_CONTRACT_PER_LEG / 100);
  cells persist mean_pnl_net next to the gross mean_pnl.

Decision refs:
  #4  edge = structural_prob - abs(net_credit)/spread_width
  #6  debit: touch_exit + close; credit: close only (touch = breach diagnostic)
  #7  baseline = first-quoted-minute entry, no edge gate
  #8  threshold chosen on train; holdout read once
  #11 selection bias check: far-band surviving vs dropped sigma
  #12 two-axis tagging: distance band + post-touch pattern
  #13 engine hypothesis: does debit/credit profitability split by pattern?
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import sys
import time as _time
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

_admin_url = os.environ.get("DATABASE_URL", "").strip()  # save admin URL before override
_bak_url   = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = _bak_url  # options_cache uses DATABASE_URL; route to safe role

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import psycopg

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_smoke,
)
from packages.shared.backtest.models import QuoteMap, distance_band
from packages.shared.backtest.net_price import net_price_from_real_quotes
from packages.shared.backtest.plugins.debit_vertical import DebitVerticalPlugin
from packages.shared.backtest.plugins.vertical import VerticalPlugin
from packages.shared.backtest.quote_validity import build_quote_map, spread_value_is_valid
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.config import FEE_PER_CONTRACT_PER_LEG, round_trip_fee_pts
from packages.shared.day_features import (
    _LANDSCAPE_ROW_SQL,
    _OPEN_STRADDLE_SQL,
    _materialize_payload,
)
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.options_cache.opra import format_opra
from packages.shared.options_cache.strikes import StructureNotListed, snap_vertical_pair
from packages.shared.probability import compute_structural_probability
from packages.shared.strategy_templates import Leg

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER     = "SPX"
OPRA_ROOT  = "SPX"
VERSION    = CANONICAL_FEATURE_VERSION
DEFAULT_SPLIT_DATE = date(2026, 6, 5)   # CR-AM / ADR 2026-09-05; June's run used 2025-08-12
_NO_HOLDOUT_LINE = "holdout: none (split = universe end)"
DTE_TARGET = 15

# Threshold sweep: edge values in [0, 1] (e.g. 0.10 = 10 percentage points)
THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20]

# CR-AL: structural-probability analogue-pool mode, shown in every section header
_MODE_TAG = "mode=?"

# CR-AU decision 5: commission per two-leg vertical opened and closed (4 contract-sides), in points
VERTICAL_FEE_PTS = round_trip_fee_pts(2)          # 0.026 at $0.65 / contract / leg

# CR-AU decision 4: holdout read rule
HOLDOUT_READ_THRESHOLD = 60


@dataclass(frozen=True)
class HoldoutGate:
    n_holdout_computed: int
    unlocked: bool
    threshold: int = HOLDOUT_READ_THRESHOLD
    preregistration: Optional[str] = None

    @property
    def line(self) -> str:
        if self.unlocked:
            return (f"holdout: n={self.n_holdout_computed}, READ UNLOCKED (threshold {self.threshold}; "
                    f"pre-registration {self.preregistration})")
        return f"holdout: n={self.n_holdout_computed}, unread (threshold {self.threshold})"


def holdout_read_gate(n_holdout_computed: int, holdout_read_path: Optional[str],
                      threshold: int = HOLDOUT_READ_THRESHOLD) -> HoldoutGate:
    """Decision 4. Holdout P&L may be printed only when a pre-registration file is
    named AND n >= threshold. A named file that does not exist raises — a bare
    flag is not consent."""
    if holdout_read_path is not None:
        path = Path(holdout_read_path)
        if not path.is_file():
            raise FileNotFoundError(f"--holdout-read: pre-registration document not found: {holdout_read_path}")
    unlocked = holdout_read_path is not None and n_holdout_computed >= threshold
    return HoldoutGate(n_holdout_computed, unlocked, threshold, holdout_read_path)


_HOLDOUT_COMPUTED_SQL = """
    SELECT count(*)
    FROM bt_daily_outcomes o
    JOIN bt_daily_features f
      ON f.ticker = o.ticker AND f.trade_date = o.trade_date
     AND f.feature_version = o.feature_version AND f.active
    WHERE o.ticker = %s AND o.feature_version = %s AND o.active
      AND o.trade_date > %s
      AND f.regime_at_classification = 'magnet-above'
      AND o.outcome_status = 'computed'
"""


def count_holdout_computed(conn, split_date: date) -> int:
    """Holdout magnet-above signal dates whose outcome is computed (the gate's n)."""
    row = conn.execute(_HOLDOUT_COMPUTED_SQL, (TICKER, VERSION, split_date)).fetchone()
    return int(row[0]) if row else 0

_UTC = ZoneInfo("UTC")
_PT  = ZoneInfo("America/Los_Angeles")

# ── NYSE HOLIDAY CALENDAR ─────────────────────────────────────────────────────

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

# ── HELPERS ───────────────────────────────────────────────────────────────────

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

def utc_to_pt(utc_naive: datetime) -> datetime:
    return utc_naive.replace(tzinfo=_UTC).astimezone(_PT).replace(tzinfo=None)

def pt_to_utc(pt_naive: datetime) -> datetime:
    return pt_naive.replace(tzinfo=_PT).astimezone(_UTC).replace(tzinfo=None)

def stride_select(lst: list, n: int = 50) -> list:
    if len(lst) <= n:
        return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson CI for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def bootstrap_mean_diff_ci(
    a: list[float], b: list[float], n_resamples: int = 1000, seed: int = 20260905,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap 95% CI of mean(a) - mean(b) (CR-AL decision #7)."""
    if not a or not b:
        return (None, None)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    lo = diffs[int(0.025 * (n_resamples - 1))]
    hi = diffs[int(0.975 * (n_resamples - 1))]
    return (lo, hi)


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class MinuteScan:
    snapshot_pt: datetime
    net_credit: Optional[float]   # None if either leg absent
    edge: Optional[float]         # None if net_credit is None

@dataclass
class TradeData:
    """All raw data for one date × structure combination."""
    trade_date: date
    structure: str             # 'debit' | 'credit'
    band: str                  # 'near' | 'mid' | 'far'
    partition: str             # 'train' | 'holdout'
    sigma: float
    drift_target: float
    structural_prob: float
    spread_width: float
    legs: list[Leg]
    short_strike: float
    expiry_date: date

    # Two-axis tagging (decision #12)
    pattern_label: Optional[str]
    reversion_wilson_lo: Optional[float]    # t15/below Wilson lower bound
    continuation_wilson_lo: Optional[float] # t15/above Wilson lower bound

    # Entry-day crawl
    entry_scan: list[MinuteScan]           # all RTH minutes, sorted ascending

    # Baseline: first quoted minute (threshold=0)
    baseline_net_credit: Optional[float]   # None if no quote available

    # Touch
    touch_resolution: str                  # rth_touch|gap_touch|afterhours_touch_retraced|no_touch
    touch_datetime_pt: Optional[datetime]  # actionable exit moment (PT naive); None if not actionable

    # Touch-exit option position value (debit only; at touch_datetime_pt)
    # = plugin.net_price(legs, quote_at_touch) = positive for debit when target reached
    touch_pos_val: Optional[float]

    # Settlement underlying price (ES close at expiry RTH end)
    settlement_price: Optional[float]

    # CR-AO decision 3: legs snapped to the listed grid at the prior close
    width_nominal: float = 10.0
    width_actual: float = 10.0
    other_strike: float = 0.0

    # CR-AN decision 7 observability (entry window) + decision 6 exclusion
    n_minutes_total: int = 0
    n_minutes_valid: int = 0
    baseline_minute_offset: Optional[int] = None
    had_invalid_quote: bool = False
    excluded_reason: Optional[str] = None


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_signal_entries(
    conn,
    universe_end: Optional[date] = None,
    split_date: date = DEFAULT_SPLIT_DATE,
) -> list[dict]:
    """All magnet-above signal dates with sigma/band/partition/drift_target.

    universe_end (CR-AL): drop signal dates after this date before anything
    else, so stratified selection reproduces an earlier run's universe.
    split_date (CR-AM): partition = 'train' if trade_date <= split_date else
    'holdout'.
    """
    rows = conn.execute(
        """
        SELECT trade_date FROM bt_daily_features
        WHERE ticker=%s AND feature_version=%s
          AND regime_at_classification='magnet-above'
        ORDER BY trade_date ASC
        """,
        (TICKER, VERSION),
    ).fetchall()
    dates = [r[0] for r in rows]
    if universe_end is not None:
        n_all = len(dates)
        dates = [d for d in dates if d <= universe_end]
        print(f"  Universe pinned to trade_date <= {universe_end.isoformat()}: "
              f"{len(dates)}/{n_all} magnet-above dates kept.")
    result = []
    skipped = 0

    for trade_date in dates:
        row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, trade_date)).fetchone()
        if not row or row[0] is None or row[1] is None:
            skipped += 1; continue
        landscape_rows, table_spot = row
        spot = float(table_spot)

        floor_ts = datetime.combine(trade_date, time(6, 33, 0))
        iv_row = conn.execute(
            _OPEN_STRADDLE_SQL, (trade_date.isoformat(), TICKER, floor_ts)
        ).fetchone()

        if not iv_row or iv_row[0] is None:
            skipped += 1; continue

        try:
            implied_move = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
        except Exception:
            skipped += 1; continue
        if not implied_move:
            skipped += 1; continue

        payload = _materialize_payload(landscape_rows, spot, implied_move)
        dt = (payload.get("regime") or {}).get("drift_target")
        if dt is None:
            skipped += 1; continue

        sigma = (float(dt) - spot) / implied_move
        result.append({
            "trade_date":    trade_date,
            "drift_target":  float(dt),
            "spot":          spot,
            "implied_move":  implied_move,
            "sigma":         sigma,
            "band":          distance_band(sigma),
            "partition":     "train" if trade_date <= split_date else "holdout",
        })

    print(f"  Loaded {len(result)}/{len(dates)} signal entries (skipped {skipped}).")
    return result


def select_clean_dates(all_entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified selection (50/band), then filter by A-bucket coverage."""
    by_band: dict[str, list] = {"near": [], "mid": [], "far": []}
    for e in all_entries:
        by_band[e["band"]].append(e)

    selected = []
    for b in ("near", "mid", "far"):
        selected += stride_select(sorted(by_band[b], key=lambda x: x["trade_date"]))
    selected.sort(key=lambda x: x["trade_date"])
    print(f"  Stratified selection: {len(selected)} dates")
    return selected


UNLISTABLE: list[dict] = []   # CR-AR decision 3: entries with status 'unlistable' (StructureNotListed), with reason
WIDTH_NOMINAL = 10.0          # the structure's intent; width_actual ∈ [10, 20] after CR-AR's cap


def filter_clean_for_structure(conn, entries: list[dict], structure: str) -> list[dict]:
    """Return entries where BOTH legs for this structure have entry-day bars.

    Credit (structure='credit'): short call at the target + long call above
    Debit  (structure='debit') : short call at the target + long call below

    CR-AR decisions 1–3 (replacing CR-AO's per-leg snap): the vertical is
    snapped as a *pair* from the strikes listed for the expiry at the prior
    close (orats_oi_gamma) — the pair whose anchor (the short leg) is nearest
    the target with width_actual ∈ [10, 20]; ties → width closest to 10, then
    toward spot. Never wider than 20; narrower than 10 only when nothing at
    ≥ 10 exists within the cap (recorded as width_narrower). Entries with no
    listed pair get status 'unlistable' and are collected in UNLISTABLE with
    the reason (excluded before the clean filter, counted in the smoke dict).
    """
    clean = []

    for e in entries:
        trade_date  = e["trade_date"]
        expiry_date  = nth_business_day(trade_date, DTE_TARGET)
        try:
            snapped = snap_vertical_pair(
                e["drift_target"], WIDTH_NOMINAL, structure, expiry_date, trade_date, conn,
                toward=e.get("spot"),
            )
        except StructureNotListed as exc:
            UNLISTABLE.append({"structure": structure, "trade_date": trade_date, "band": e["band"],
                               "status": "unlistable", "reason": str(exc)})
            continue
        short_strike = float(snapped.anchor)
        other_strike = float(snapped.other)

        short_opra = format_opra(OPRA_ROOT, expiry_date, "C", short_strike)
        other_opra = format_opra(OPRA_ROOT, expiry_date, "C", other_strike)

        start_pt = datetime(trade_date.year, trade_date.month, trade_date.day, 6, 30)
        end_pt   = datetime(trade_date.year, trade_date.month, trade_date.day, 13, 0)

        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END) > 0,
                SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END) > 0
            FROM orats_options_minute
            WHERE opra_symbol=ANY(%s) AND snapshot_pt>=%s AND snapshot_pt<%s
            """,
            (short_opra, other_opra, [short_opra, other_opra], start_pt, end_pt),
        ).fetchone()

        has_short = bool(row[0]) if row else False
        has_other = bool(row[1]) if row else False

        if has_short and has_other:
            clean.append({**e, "expiry_date": expiry_date,
                          "short_strike": short_strike, "other_strike": other_strike,
                          "width_actual": snapped.width_actual, "width_nominal": snapped.width_nominal,
                          "width_narrower": snapped.narrower_than_nominal,
                          "listed_prior_close": snapped.prior_close,
                          "short_opra": short_opra, "other_opra": other_opra})

    return clean


# ── ENTRY CRAWL ───────────────────────────────────────────────────────────────

def build_entry_scan(
    conn,
    trade_date: date,
    short_opra: str,
    other_opra: str,
    legs: list[Leg],
    structural_prob: float,
    spread_width: float,
) -> tuple[list[MinuteScan], Optional[float], dict]:
    """Query entry-day RTH quotes and build per-minute scan.

    Returns (entry_scan, baseline_net_credit, observability).
    baseline_net_credit = net_credit at the first VALID minute (no edge gate)
    — CR-AN decisions 2/3: a minute is valid only when both legs' bid/ask
    pass the leg rule and the spread value is in [0, width] in the
    structure's direction; invalid minutes are recorded with net_credit=None
    and never fill or baseline.
    """
    start_pt = datetime(trade_date.year, trade_date.month, trade_date.day, 6, 30)
    end_pt   = datetime(trade_date.year, trade_date.month, trade_date.day, 13, 0)

    rows = conn.execute(
        """
        SELECT snapshot_pt, bid_price, ask_price, opra_symbol
        FROM orats_options_minute
        WHERE opra_symbol=ANY(%s) AND snapshot_pt>=%s AND snapshot_pt<%s
        ORDER BY snapshot_pt ASC
        """,
        ([short_opra, other_opra], start_pt, end_pt),
    ).fetchall()

    # Group raw (strike, type, bid, ask) per minute; the shared validity helper
    # builds the QuoteMap (invalid legs dropped) — CR-AN decision 2.
    raw_by_minute: dict[datetime, list[tuple]] = {}
    for snap_pt, bid, ask, opra in rows:
        strike_key = _opra_to_quote_key(opra)
        if strike_key is None:
            continue
        raw_by_minute.setdefault(snap_pt, []).append((strike_key[0], strike_key[1], bid, ask))

    scan: list[MinuteScan] = []
    baseline_net_credit: Optional[float] = None
    minutes = sorted(raw_by_minute.keys())
    n_valid = 0
    baseline_offset: Optional[int] = None

    for snap_pt in minutes:
        qmap, _n_bad_legs = build_quote_map(raw_by_minute[snap_pt])
        pos_val = net_price_from_real_quotes(legs, qmap)
        if not spread_value_is_valid(pos_val, spread_width, legs):
            scan.append(MinuteScan(snapshot_pt=snap_pt, net_credit=None, edge=None))
            continue
        n_valid += 1
        net_credit = -pos_val
        edge = structural_prob - abs(net_credit) / spread_width
        scan.append(MinuteScan(snapshot_pt=snap_pt, net_credit=net_credit, edge=edge))
        if baseline_net_credit is None:
            baseline_net_credit = net_credit
            baseline_offset = int(round((snap_pt - start_pt).total_seconds() / 60.0))

    obs = {
        "n_minutes_total": len(minutes),
        "n_minutes_valid": n_valid,
        "baseline_minute_offset": baseline_offset,
        "had_invalid_quote": n_valid < len(minutes),
    }
    return scan, baseline_net_credit, obs


def _opra_to_quote_key(opra: str) -> Optional[tuple[float, str]]:
    """Extract (strike, 'C'/'P') from an OPRA symbol.

    OPRA format: {root}{YYMMDD}{C|P}{strike*1000:08d}
    e.g. SPX230522C04205000 → (4205.0, 'C')
    Root is always 'SPX' (3 chars) in this codebase.
    """
    try:
        opt_char   = opra[9]       # position 3+6=9: 'C' or 'P'
        strike_str = opra[10:]     # remaining 8 chars: strike × 1000, zero-padded
        strike_val = int(strike_str) / 1000.0
        return (strike_val, opt_char)
    except (IndexError, ValueError):
        return None


# ── TOUCH DETECTION ───────────────────────────────────────────────────────────

_ES_OPEN_MINS = 5

def detect_touch(
    conn,
    trade_date: date,
    expiry_date: date,
    drift_target: float,
) -> tuple[str, Optional[datetime]]:
    """Find ES touch and classify.

    Returns (touch_resolution, actionable_exit_datetime_pt).
    actionable_exit_datetime_pt is None for afterhours_retraced and no_touch.
    """
    search_start_utc = datetime(trade_date.year, trade_date.month, trade_date.day, 0, 0, 0)
    search_end_utc   = datetime(expiry_date.year, expiry_date.month, expiry_date.day, 23, 59, 59)

    touch_row = conn.execute(
        """
        SELECT datetime, close FROM ironbeam_es_1m_bars
        WHERE datetime>=%s AND datetime<=%s AND close>=%s
        ORDER BY datetime ASC LIMIT 1
        """,
        (search_start_utc, search_end_utc, drift_target),
    ).fetchone()

    if touch_row is None:
        return "no_touch", None

    touch_utc = touch_row[0]
    touch_pt  = utc_to_pt(touch_utc)
    touch_date = touch_pt.date()

    day_open  = datetime(touch_date.year, touch_date.month, touch_date.day, 6, 30)
    day_close = datetime(touch_date.year, touch_date.month, touch_date.day, 13, 0)

    if day_open <= touch_pt <= day_close:
        return "rth_touch", touch_pt

    # Outside RTH — find next RTH open
    if touch_pt < day_open:
        next_rth_date = touch_date
        next_open_pt  = day_open
    else:
        next_rth_date = next_weekday(touch_date)
        next_open_pt  = datetime(next_rth_date.year, next_rth_date.month,
                                 next_rth_date.day, 6, 30)

    next_open_utc = pt_to_utc(next_open_pt)
    es_row = conn.execute(
        """
        SELECT close FROM ironbeam_es_1m_bars
        WHERE datetime>=%s AND datetime<%s ORDER BY datetime ASC LIMIT 1
        """,
        (next_open_utc, next_open_utc + timedelta(minutes=_ES_OPEN_MINS)),
    ).fetchone()

    if es_row is None:
        return "afterhours_touch_retraced", None

    if float(es_row[0]) >= drift_target:
        return "gap_touch", next_open_pt
    else:
        return "afterhours_touch_retraced", None


def get_touch_pos_val(
    conn,
    legs: list[Leg],
    short_opra: str,
    other_opra: str,
    touch_resolution: str,
    touch_datetime_pt: Optional[datetime],
) -> Optional[float]:
    """Query option quotes at/after the actionable touch moment; return position value."""
    if touch_datetime_pt is None or touch_resolution not in ("rth_touch", "gap_touch"):
        return None

    # Query window: touch_datetime_pt to touch_datetime_pt + 90 minutes
    window_end_pt = touch_datetime_pt + timedelta(minutes=90)

    rows = conn.execute(
        """
        SELECT snapshot_pt, bid_price, ask_price, opra_symbol
        FROM orats_options_minute
        WHERE opra_symbol=ANY(%s) AND snapshot_pt>=%s AND snapshot_pt<=%s
        ORDER BY snapshot_pt ASC
        """,
        ([short_opra, other_opra], touch_datetime_pt, window_end_pt),
    ).fetchall()

    # Group raw quotes by minute; CR-AN decision 4: exit only on a valid minute
    raw_by_minute: dict[datetime, list[tuple]] = {}
    for snap_pt, bid, ask, opra in rows:
        key = _opra_to_quote_key(opra)
        if key is None:
            continue
        raw_by_minute.setdefault(snap_pt, []).append((key[0], key[1], bid, ask))

    width = abs(legs[0].strike - legs[1].strike) if len(legs) == 2 else None
    for snap_pt in sorted(raw_by_minute.keys()):
        qmap, _ = build_quote_map(raw_by_minute[snap_pt])
        pos_val = net_price_from_real_quotes(legs, qmap)
        if width is not None and not spread_value_is_valid(pos_val, width, legs):
            continue
        if pos_val is not None:
            return pos_val

    return None


# ── SETTLEMENT ────────────────────────────────────────────────────────────────

def get_settlement_price(conn, expiry_date: date) -> Optional[float]:
    """ES close price at end of expiry RTH (12:50–13:00 PT) as settlement proxy."""
    settle_start_utc = pt_to_utc(
        datetime(expiry_date.year, expiry_date.month, expiry_date.day, 12, 50)
    )
    settle_end_utc = pt_to_utc(
        datetime(expiry_date.year, expiry_date.month, expiry_date.day, 13, 0)
    )
    row = conn.execute(
        """
        SELECT close FROM ironbeam_es_1m_bars
        WHERE datetime>=%s AND datetime<=%s
        ORDER BY datetime DESC LIMIT 1
        """,
        (settle_start_utc, settle_end_utc),
    ).fetchone()
    return float(row[0]) if row else None


# ── STRUCTURAL PROBABILITY ────────────────────────────────────────────────────

def get_structural_prob(
    conn,
    trade_date: date,
    structure: str,
    mode: str = "walk-forward",
) -> tuple[float, Optional[str], Optional[float], Optional[float]]:
    """Compute structural probability + post-touch tags for one date.

    Returns (structural_prob, pattern_label, reversion_wilson_lo, continuation_wilson_lo).
    structural_prob = touch_rate for debit, 1 - touch_rate for credit.

    mode (CR-AL): 'walk-forward' passes before_date=trade_date (analogue pool
    limited to prior dates); 'full' passes allow_lookahead=True (whole corpus,
    as CR-AH ran in June). No mode passes neither — the ADR raise applies.
    """
    row = conn.execute(
        """
        SELECT feature_vector FROM bt_daily_features
        WHERE ticker=%s AND trade_date=%s AND feature_version=%s
        LIMIT 1
        """,
        (TICKER, trade_date, VERSION),
    ).fetchone()
    if row is None:
        return 0.5, None, None, None

    fv = row[0]
    if isinstance(fv, str):
        fv = json.loads(fv)

    if mode == "walk-forward":
        sp_kwargs = {"before_date": trade_date.isoformat()}
    elif mode == "full":
        sp_kwargs = {"allow_lookahead": True}
    else:
        raise ValueError(f"unknown structural-prob mode: {mode!r}")

    result = compute_structural_probability(
        fv, conn, ticker=TICKER,
        exclude_date=trade_date.isoformat(),
        regime_kind="magnet-above",
        **sp_kwargs,
    )

    touch_rate = float(result.get("touch_rate") or 0.5)
    structural_prob = touch_rate if structure == "debit" else (1.0 - touch_rate)

    pattern_label: Optional[str] = None
    rev_lo: Optional[float] = None
    cont_lo: Optional[float] = None

    pt = result.get("post_touch")
    if pt:
        pattern_label = pt.get("pattern_label")
        wci = pt.get("wilson_cis") or {}
        t15 = wci.get("t15") or {}
        below = t15.get("below")
        above = t15.get("above")
        if below and below[0] is not None:
            rev_lo = float(below[0])
        if above and above[0] is not None:
            cont_lo = float(above[0])

    return structural_prob, pattern_label, rev_lo, cont_lo


# ── PER-DATE COLLECTION ───────────────────────────────────────────────────────

def build_legs(
    short_strike: float, structure: str, other_strike: Optional[float] = None,
) -> tuple[list[Leg], float]:
    """Build leg list and spread width for one structure.

    CR-AO decision 3: other_strike is the snapped second leg; the returned
    width is |short − other| (width_actual), not a constant 10. Without
    other_strike the legacy ±10 is used.
    """
    if structure == "credit":
        long_strike = float(other_strike) if other_strike is not None else short_strike + 10.0
        legs = [
            Leg(side="short", type="call", strike=short_strike),
            Leg(side="long",  type="call", strike=long_strike),
        ]
    else:  # debit
        long_strike = float(other_strike) if other_strike is not None else short_strike - 10.0
        legs = [
            Leg(side="long",  type="call", strike=long_strike),
            Leg(side="short", type="call", strike=short_strike),
        ]
    return legs, abs(short_strike - long_strike)  # width_actual


def collect_trade_data(
    conn, entry: dict, structure: str, mode: str = "walk-forward",
) -> Optional[TradeData]:
    """Collect all raw data for one date × structure combination."""
    trade_date  = entry["trade_date"]
    band        = entry["band"]
    partition   = entry["partition"]
    sigma       = entry["sigma"]
    drift_target = entry["drift_target"]
    short_strike = float(entry["short_strike"])
    expiry_date  = entry["expiry_date"]

    legs, spread_width = build_legs(short_strike, structure, entry.get("other_strike"))

    # OPRAs for both legs
    if structure == "credit":
        short_opra = entry["short_opra"]
        other_opra = entry["other_opra"]
    else:
        short_opra = entry["short_opra"]  # target
        other_opra = entry["other_opra"]  # target-10

    # Structural probability + post-touch tags
    structural_prob, pattern_label, rev_lo, cont_lo = get_structural_prob(
        conn, trade_date, structure, mode
    )

    # Entry-day crawl (CR-AN: valid minutes only; observability returned)
    entry_scan, baseline_net_credit, obs = build_entry_scan(
        conn, trade_date, short_opra, other_opra,
        legs, structural_prob, spread_width,
    )
    # CR-AN decision 6: exclude only when the entry window has no valid minute
    excluded_reason = "no_valid_entry_minute" if obs["n_minutes_valid"] == 0 else None

    # Touch detection
    touch_resolution, touch_datetime_pt = detect_touch(
        conn, trade_date, expiry_date, drift_target
    )

    # Touch-exit quotes (debit only; credit records touch but doesn't price exit)
    touch_pos_val = None
    if structure == "debit":
        touch_pos_val = get_touch_pos_val(
            conn, legs, short_opra, other_opra,
            touch_resolution, touch_datetime_pt,
        )

    # Settlement price (underlying)
    settlement_price = get_settlement_price(conn, expiry_date)

    return TradeData(
        trade_date=trade_date,
        structure=structure,
        band=band,
        partition=partition,
        sigma=sigma,
        drift_target=drift_target,
        structural_prob=structural_prob,
        spread_width=spread_width,
        legs=legs,
        short_strike=short_strike,
        expiry_date=expiry_date,
        pattern_label=pattern_label,
        reversion_wilson_lo=rev_lo,
        continuation_wilson_lo=cont_lo,
        entry_scan=entry_scan,
        baseline_net_credit=baseline_net_credit,
        touch_resolution=touch_resolution,
        touch_datetime_pt=touch_datetime_pt,
        touch_pos_val=touch_pos_val,
        settlement_price=settlement_price,
        width_nominal=float(entry.get("width_nominal", 10.0)),
        width_actual=spread_width,
        other_strike=float(entry.get("other_strike", 0.0)),
        n_minutes_total=obs["n_minutes_total"],
        n_minutes_valid=obs["n_minutes_valid"],
        baseline_minute_offset=obs["baseline_minute_offset"],
        had_invalid_quote=obs["had_invalid_quote"],
        excluded_reason=excluded_reason,
    )


# ── P&L COMPUTATION ───────────────────────────────────────────────────────────

# Plugin instances for payoff computation
_CREDIT_PLUGIN = VerticalPlugin()
_DEBIT_PLUGIN  = DebitVerticalPlugin()


def compute_pnl(td: TradeData, threshold: float, fee_pts: float = VERTICAL_FEE_PTS) -> dict:
    """Compute P&L for one trade at a given threshold.

    Returns dict with:
      filled, fill_net_credit, fill_edge,
      touch_exit_pnl (debit only), close_pnl,
      baseline_net_credit, baseline_close_pnl, baseline_touch_exit_pnl,
      close_zone, touch_resolution, settlement_available,
      CR-AU decision 5: close_pnl_net / touch_exit_pnl_net / baseline_close_pnl_net
      (= gross − fee_pts, the round-trip commission of the vertical), fee_pts
    """
    plugin = _DEBIT_PLUGIN if td.structure == "debit" else _CREDIT_PLUGIN

    # Find fill: first entry scan row with edge >= threshold
    fill_net_credit: Optional[float] = None
    fill_edge: Optional[float] = None
    for scan in td.entry_scan:
        if scan.edge is not None and scan.edge >= threshold:
            fill_net_credit = scan.net_credit
            fill_edge = scan.edge
            break

    filled = fill_net_credit is not None

    # Touch-exit P&L (debit only)
    touch_exit_pnl: Optional[float] = None
    if (filled and td.structure == "debit" and td.touch_pos_val is not None
            and td.touch_resolution in ("rth_touch", "gap_touch")):
        touch_exit_pnl = fill_net_credit + td.touch_pos_val

    # Close P&L
    close_pnl: Optional[float] = None
    close_zone: Optional[str] = None
    if filled and td.settlement_price is not None:
        payoff = plugin.payoff(td.legs, td.settlement_price)
        close_pnl = fill_net_credit + payoff
        close_zone = plugin.close_zone(td.legs, td.settlement_price)

    # Baseline
    base_credit = td.baseline_net_credit
    baseline_close_pnl: Optional[float] = None
    baseline_touch_exit_pnl: Optional[float] = None

    if base_credit is not None and td.settlement_price is not None:
        payoff = plugin.payoff(td.legs, td.settlement_price)
        baseline_close_pnl = base_credit + payoff

    if (td.structure == "debit" and base_credit is not None
            and td.touch_pos_val is not None
            and td.touch_resolution in ("rth_touch", "gap_touch")):
        baseline_touch_exit_pnl = base_credit + td.touch_pos_val

    return {
        "filled": filled,
        "fill_net_credit": fill_net_credit,
        "fill_edge": fill_edge,
        "touch_exit_pnl": touch_exit_pnl,
        "close_pnl": close_pnl,
        "close_zone": close_zone,
        "baseline_net_credit": base_credit,
        "baseline_close_pnl": baseline_close_pnl,
        "baseline_touch_exit_pnl": baseline_touch_exit_pnl,
        "settlement_available": td.settlement_price is not None,
        "touch_resolution": td.touch_resolution,
        # CR-AU decision 5: net of commission (gross figures above are unchanged)
        "close_pnl_net": (close_pnl - fee_pts) if close_pnl is not None else None,
        "touch_exit_pnl_net": (touch_exit_pnl - fee_pts) if touch_exit_pnl is not None else None,
        "baseline_close_pnl_net": (baseline_close_pnl - fee_pts) if baseline_close_pnl is not None else None,
        "fee_pts": fee_pts,
    }


# ── AGGREGATE STATS ───────────────────────────────────────────────────────────

@dataclass
class CellStats:
    n: int = 0
    n_filled: int = 0
    n_settlement: int = 0
    pnl_sum: float = 0.0
    pnl_sq_sum: float = 0.0
    n_wins: int = 0          # close_pnl > 0
    baseline_sum: float = 0.0
    baseline_wins: int = 0
    baseline_n: int = 0
    # CR-AR decision 5: width sums for the width-weighted per-point statistics
    pnl_width_sum: float = 0.0        # Σ width_actual over the settled filled trades
    baseline_width_sum: float = 0.0   # Σ width_actual over the baseline trades
    pnl_net_sum: float = 0.0          # CR-AU decision 5: Σ close_pnl_net over the settled filled trades


def aggregate(trades: list[tuple[TradeData, dict]]) -> CellStats:
    """Aggregate P&L stats for a list of (TradeData, compute_pnl result) pairs."""
    s = CellStats()
    s.n = len(trades)
    for td, r in trades:
        if r["filled"]:
            s.n_filled += 1
        if r["close_pnl"] is not None:
            s.n_settlement += 1
            s.pnl_sum += r["close_pnl"]
            s.pnl_sq_sum += r["close_pnl"] ** 2
            s.pnl_width_sum += float(td.width_actual)
            s.pnl_net_sum += r["close_pnl_net"]
            if r["close_pnl"] > 0:
                s.n_wins += 1
        if r["baseline_close_pnl"] is not None:
            s.baseline_sum += r["baseline_close_pnl"]
            s.baseline_wins += (1 if r["baseline_close_pnl"] > 0 else 0)
            s.baseline_n += 1
            s.baseline_width_sum += float(td.width_actual)
    return s


def fmt_stats(s: CellStats, label: str = "") -> dict:
    """Compute derived stats from CellStats."""
    n = s.n_settlement
    if n == 0:
        return {"label": label, "n": 0, "n_filled": s.n_filled,
                "mean_pnl": None, "win_rate": None,
                "wilson_lo": None, "wilson_hi": None,
                "baseline_mean": None, "beat": None,
                "mean_pnl_per_point": None, "baseline_per_point": None,
                "beat_per_point": None, "mean_width": None, "mean_pnl_net": None}

    mean_pnl = s.pnl_sum / n
    win_rate = s.n_wins / n
    wlo, whi = wilson_ci(s.n_wins, n)
    baseline_mean = s.baseline_sum / s.baseline_n if s.baseline_n else None
    beat = (mean_pnl - baseline_mean) if baseline_mean is not None else None
    # CR-AR decision 5: width-weighted per-point statistics —
    # Σ x_i / Σ w_i is the width-weighted mean of x_i / w_i; equals x / 10 when every width is 10.
    mean_pnl_pp = s.pnl_sum / s.pnl_width_sum if s.pnl_width_sum else None
    baseline_pp = s.baseline_sum / s.baseline_width_sum if s.baseline_width_sum else None
    beat_pp = (mean_pnl_pp - baseline_pp) if (mean_pnl_pp is not None and baseline_pp is not None) else None
    return {
        "label": label,
        "n": n,
        "n_filled": s.n_filled,
        "fill_rate": s.n_filled / s.n if s.n else 0,
        "mean_pnl": mean_pnl,
        "win_rate": win_rate,
        "wilson_lo": wlo,
        "wilson_hi": whi,
        "baseline_mean": baseline_mean,
        "beat": beat,
        "mean_pnl_per_point": mean_pnl_pp,
        "baseline_per_point": baseline_pp,
        "beat_per_point": beat_pp,
        "mean_width": s.pnl_width_sum / n,
        "mean_pnl_net": s.pnl_net_sum / n,      # CR-AU decision 5
    }


# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────

def threshold_sweep(
    train_data: list[TradeData],
    structure: str,
) -> tuple[float, list[dict]]:
    """Sweep thresholds on train; return chosen threshold + sweep table."""
    sweep_rows = []
    best_thresh = 0.0
    best_beat   = -float("inf")

    for T in THRESHOLDS:
        pairs = [(td, compute_pnl(td, T)) for td in train_data]
        s = aggregate(pairs)
        row = fmt_stats(s, label=f"T={T:.2f}")
        row["threshold"] = T
        sweep_rows.append(row)

        # Pick threshold with highest beat (close P&L − baseline), min n=5 settled trades
        beat = row.get("beat") or -float("inf")
        if row["n"] >= 5 and beat > best_beat:
            best_beat   = beat
            best_thresh = T

    return best_thresh, sweep_rows


# ── SELECTION BIAS CHECK (decision #11) ──────────────────────────────────────

def selection_bias_check(
    all_selected: list[dict],
    clean_entries: list[dict],
    structure: str,
) -> dict:
    """Far-band sigma comparison: surviving vs dropped trades."""
    clean_dates = {e["trade_date"] for e in clean_entries}
    far_all    = [e for e in all_selected if e["band"] == "far"]
    far_clean  = [e for e in far_all if e["trade_date"] in clean_dates]
    far_drop   = [e for e in far_all if e["trade_date"] not in clean_dates]

    def mean_sigma(lst):
        return (sum(e["sigma"] for e in lst) / len(lst)) if lst else None

    return {
        "structure": structure,
        "far_n_all": len(far_all),
        "far_n_clean": len(far_clean),
        "far_n_dropped": len(far_drop),
        "far_sigma_clean_mean": mean_sigma(far_clean),
        "far_sigma_dropped_mean": mean_sigma(far_drop),
    }


# ── REPORTING ─────────────────────────────────────────────────────────────────

SEP = "=" * 70
SEP2 = "-" * 70

def pf(v, fmt=".2f", none_str="—"):
    return f"{v:{fmt}}" if v is not None else none_str

def fmt_row(r: dict) -> str:
    return (
        f"  {r['label']:<22}  n={r['n']:>3}  filled={r['n_filled']:>3}"
        f"  pnl={pf(r.get('mean_pnl'))}"
        f"  win%={pf(r.get('win_rate'), '.0%')}"
        f"  [{pf(r.get('wilson_lo'), '.0%')}–{pf(r.get('wilson_hi'), '.0%')}]"
        f"  base={pf(r.get('baseline_mean'))}"
        f"  beat={pf(r.get('beat'))}"
    )


def print_sweep(sweep_rows: list[dict], chosen: float, structure: str):
    print(f"\n{structure.upper()} — Threshold sweep (TRAIN only) [{_MODE_TAG}]:")
    print(f"  {'T':>6}  n_settled  fill_n  mean_pnl  win%   beat  beat/pt  chosen?")
    print(f"  {'─'*70}")
    for r in sweep_rows:
        marker = " ← CHOSEN" if abs(r["threshold"] - chosen) < 0.001 else ""
        print(f"  {r['threshold']:.2f}     {r['n']:>3}      {r['n_filled']:>3}"
              f"     {pf(r.get('mean_pnl')):>7}  {pf(r.get('win_rate'), '.0%'):>5}"
              f"  {pf(r.get('beat')):>7}  {pf(r.get('beat_per_point'), '.3f'):>7}{marker}")


def print_by_band(all_data: list[TradeData], threshold: float, structure: str, label: str,
                  train_only: bool = False, holdout_unlocked: bool = False):
    """CR-AU decision 4: holdout rows print n per band and dates matured only unless
    `holdout_unlocked` (the read gate) is True. `net` = mean close P&L − commission."""
    print(f"\n{structure.upper()} — By distance band ({label}, T={threshold:.2f}) [{_MODE_TAG}]:")
    print(f"  {'band':<6}  {'part':<8}  {'n':>3}  {'pnl':>7}  {'net':>7}  {'win%':>5}  "
          f"[lo–hi 95%]     {'base':>7}  {'beat':>7}  {'pnl/pt':>7}  {'beat/pt':>7}  {'width':>5}")
    print(f"  {'─'*105}")
    for part in (("train",) if train_only else ("train", "holdout")):
        if part == "holdout" and not any(td.partition == "holdout" for td in all_data):
            print(f"  {_NO_HOLDOUT_LINE}")
            continue
        for b in ("near", "mid", "far", "all"):
            if b == "all":
                subset = [td for td in all_data if td.partition == part]
            else:
                subset = [td for td in all_data if td.partition == part and td.band == b]
            if not subset:
                continue
            if part == "holdout" and not holdout_unlocked:
                n_matured = sum(1 for td in subset if td.settlement_price is not None)
                print(f"  {b:<6}  {part:<8}  {len(subset):>3}  (unread — n only; dates matured={n_matured})")
                continue
            pairs = [(td, compute_pnl(td, threshold)) for td in subset]
            s = aggregate(pairs)
            r = fmt_stats(s, label=f"{b}/{part}")

            band_note = ""
            if b == "far" and part == "holdout":
                band_note = " ⚠ NOT INTERPRETABLE (n≤5)"
            elif b in ("mid", "far") and part == "holdout":
                band_note = " ⚠ wide CI"

            print(
                f"  {b:<6}  {part:<8}  {r['n']:>3}  "
                f"{pf(r.get('mean_pnl')):>7}  {pf(r.get('mean_pnl_net')):>7}  {pf(r.get('win_rate'), '.0%'):>5}  "
                f"[{pf(r.get('wilson_lo'), '.0%'):>4}–{pf(r.get('wilson_hi'), '.0%'):>4}]  "
                f"{pf(r.get('baseline_mean')):>7}  {pf(r.get('beat')):>7}  "
                f"{pf(r.get('mean_pnl_per_point'), '.3f'):>7}  {pf(r.get('beat_per_point'), '.3f'):>7}  "
                f"{pf(r.get('mean_width'), '.1f'):>5}"
                f"{band_note}"
            )


def print_by_pattern(train_data: list[TradeData], threshold: float, structure: str):
    print(f"\n{structure.upper()} — By post-touch pattern (TRAIN only, T={threshold:.2f}) [{_MODE_TAG}]:")
    patterns = sorted({td.pattern_label for td in train_data if td.pattern_label})
    has_pattern = [td for td in train_data if td.pattern_label]
    null_count  = sum(1 for td in train_data if td.pattern_label is None)

    print(f"  (n with pattern_label={len(has_pattern)}, n without={null_count})")
    if not has_pattern:
        print("  No pattern_label data available.")
        return

    print(f"  {'pattern':<30}  {'n':>3}  {'pnl':>7}  {'win%':>5}  "
          f"[lo–hi 95%]     {'base':>7}  {'beat':>7}  {'beat/pt':>7}")
    print(f"  {'─'*80}")
    for pat in patterns:
        subset = [td for td in train_data if td.pattern_label == pat]
        pairs = [(td, compute_pnl(td, threshold)) for td in subset]
        s = aggregate(pairs)
        r = fmt_stats(s, label=pat)
        print(
            f"  {pat:<30}  {r['n']:>3}  "
            f"{pf(r.get('mean_pnl')):>7}  {pf(r.get('win_rate'), '.0%'):>5}  "
            f"[{pf(r.get('wilson_lo'), '.0%'):>4}–{pf(r.get('wilson_hi'), '.0%'):>4}]  "
            f"{pf(r.get('baseline_mean')):>7}  {pf(r.get('beat')):>7}  {pf(r.get('beat_per_point'), '.3f'):>7}"
        )

    # CR-AL decision #7: labeled vs unlabeled (coverage effect)
    print(f"  {'─'*80}")
    for lab, subset in (("(labeled)", has_pattern),
                        ("(unlabeled)", [td for td in train_data if td.pattern_label is None])):
        pairs = [(td, compute_pnl(td, threshold)) for td in subset]
        r = fmt_stats(aggregate(pairs), label=lab)
        print(
            f"  {lab:<30}  {r['n']:>3}  "
            f"{pf(r.get('mean_pnl')):>7}  {pf(r.get('win_rate'), '.0%'):>5}  "
            f"[{pf(r.get('wilson_lo'), '.0%'):>4}–{pf(r.get('wilson_hi'), '.0%'):>4}]  "
            f"{pf(r.get('baseline_mean')):>7}  {pf(r.get('beat')):>7}  {pf(r.get('beat_per_point'), '.3f'):>7}"
        )


def print_debit_touch_breakdown(train_data: list[TradeData], threshold: float):
    """Debit-specific: touch-exit P&L vs close P&L by resolution and band."""
    print(f"\nDEBIT — Touch resolution breakdown (TRAIN only, T={threshold:.2f}) [{_MODE_TAG}]:")
    print(f"  {'resolution':<30}  n   touch_exit  close_pnl   base_close")
    print(f"  {'─'*65}")
    resolutions = ["rth_touch", "gap_touch", "afterhours_touch_retraced", "no_touch"]
    for res in resolutions:
        subset = [td for td in train_data if td.touch_resolution == res]
        if not subset:
            continue
        pairs = [(td, compute_pnl(td, threshold)) for td in subset]
        touch_pnls = [r["touch_exit_pnl"] for _, r in pairs if r["touch_exit_pnl"] is not None]
        close_pnls = [r["close_pnl"] for _, r in pairs if r["close_pnl"] is not None]
        base_pnls  = [r["baseline_close_pnl"] for _, r in pairs if r["baseline_close_pnl"] is not None]

        t_mean = sum(touch_pnls) / len(touch_pnls) if touch_pnls else None
        c_mean = sum(close_pnls) / len(close_pnls) if close_pnls else None
        b_mean = sum(base_pnls)  / len(base_pnls)  if base_pnls  else None
        print(
            f"  {res:<30}  {len(subset):>2}   "
            f"{pf(t_mean):>9}   {pf(c_mean):>9}   {pf(b_mean):>9}"
        )


def print_selection_bias(bias: dict):
    s = bias["structure"].upper()
    print(f"\n{s} — Selection bias check (far band, decision #11) [{_MODE_TAG}]:")
    print(f"  far/all: n={bias['far_n_all']}  "
          f"clean: n={bias['far_n_clean']} (mean σ={pf(bias['far_sigma_clean_mean'], '.2f')})"
          f"  dropped: n={bias['far_n_dropped']} (mean σ={pf(bias['far_sigma_dropped_mean'], '.2f')})")

    c_s = bias.get("far_sigma_clean_mean") or 0
    d_s = bias.get("far_sigma_dropped_mean") or 0
    if d_s and c_s < d_s * 0.95:
        print("  ⚠ BIAS: clean far trades are significantly CLOSER than dropped far trades.")
        print("    The far-band result may be OPTIMISTICALLY SELECTED (closer to spot → easier trade).")
    else:
        print("  ✓ No significant selection bias detected in far band.")


# ── SUMMARY READS A/B/C/D ────────────────────────────────────────────────────

def build_summary(
    debit_data: list[TradeData],
    credit_data: list[TradeData],
    debit_thresh: float,
    credit_thresh: float,
    train_only: bool = False,
    seed: int = 20260905,
    summary_d_out: Optional[dict] = None,
    holdout_gate: Optional[HoldoutGate] = None,
) -> str:
    """Build four plain-language summary reads.

    train_only (CR-AL): Summary A/B are replaced by a "holdout not read" line.
    summary_d_out: if given, receives per-structure Summary D numbers.
    holdout_gate (CR-AU decision 4): when given and locked, Summary A/B print
    the gate line ("holdout: n=<k>, unread (threshold 60)") and no holdout P&L.
    """
    lines = []
    lines.append(f"[{_MODE_TAG}]")
    holdout_locked = holdout_gate is not None and not holdout_gate.unlocked

    def get_cell(data, band, partition, threshold):
        subset = [td for td in data if td.band == band and td.partition == partition]
        if not subset:
            return None
        pairs = [(td, compute_pnl(td, threshold)) for td in subset]
        s = aggregate(pairs)
        return fmt_stats(s)

    # Summary A: Debit near/holdout
    lines.append("\n── Summary A: Debit near-band holdout ──")
    d_near_ho = None if (train_only or holdout_locked) else get_cell(debit_data, "near", "holdout", debit_thresh)
    if train_only:
        lines.append("  holdout not read (--train-only).")
    elif not any(td.partition == "holdout" for td in debit_data):
        lines.append(f"  {_NO_HOLDOUT_LINE}")
    elif holdout_locked:
        lines.append(f"  {holdout_gate.line}")
    elif d_near_ho and d_near_ho["n"] >= 3:
        beat = d_near_ho.get("beat") or 0
        sign = "+" if beat > 0 else ""
        lines.append(
            f"  Debit near-band holdout (n={d_near_ho['n']}): "
            f"mean close P&L={pf(d_near_ho.get('mean_pnl'))} pts  "
            f"beat={sign}{pf(beat)} vs baseline  "
            f"win%={pf(d_near_ho.get('win_rate'), '.0%')} "
            f"[{pf(d_near_ho.get('wilson_lo'), '.0%')}–{pf(d_near_ho.get('wilson_hi'), '.0%')}]"
        )
        if beat > 0.5:
            lines.append("  READ: Debit near-band shows positive holdout edge over baseline.")
        elif beat < -0.5:
            lines.append("  READ: Debit near-band is BELOW baseline on holdout — edge is negative or absent.")
        else:
            lines.append("  READ: Debit near-band result is directional but inconclusive (wide CI or small beat).")
    else:
        lines.append(f"  Insufficient holdout data (n={d_near_ho['n'] if d_near_ho else 0}). Directional only.")

    # Summary B: Credit near/holdout
    lines.append("\n── Summary B: Credit near-band holdout ──")
    c_near_ho = None if (train_only or holdout_locked) else get_cell(credit_data, "near", "holdout", credit_thresh)
    if train_only:
        lines.append("  holdout not read (--train-only).")
    elif not any(td.partition == "holdout" for td in credit_data):
        lines.append(f"  {_NO_HOLDOUT_LINE}")
    elif holdout_locked:
        lines.append(f"  {holdout_gate.line}")
    elif c_near_ho and c_near_ho["n"] >= 3:
        beat = c_near_ho.get("beat") or 0
        sign = "+" if beat > 0 else ""
        lines.append(
            f"  Credit near-band holdout (n={c_near_ho['n']}): "
            f"mean close P&L={pf(c_near_ho.get('mean_pnl'))} pts  "
            f"beat={sign}{pf(beat)} vs baseline  "
            f"win%={pf(c_near_ho.get('win_rate'), '.0%')} "
            f"[{pf(c_near_ho.get('wilson_lo'), '.0%')}–{pf(c_near_ho.get('wilson_hi'), '.0%')}]"
        )
        if beat > 0.3:
            lines.append("  READ: Credit near-band shows positive holdout edge. Fade has merit near the magnet.")
        elif beat < -0.3:
            lines.append("  READ: Credit near-band is BELOW baseline — fade does not add value near the magnet.")
        else:
            lines.append("  READ: Credit near-band result is neutral / within noise.")
    else:
        lines.append(f"  Insufficient holdout data (n={c_near_ho['n'] if c_near_ho else 0}). Directional only.")

    # Summary C: Crossover — where debit < credit beat
    lines.append("\n── Summary C: Structure crossover by distance (TRAIN, close P&L beat) ──")
    for b in ("near", "mid", "far"):
        d_cell = get_cell(debit_data,  b, "train", debit_thresh)
        c_cell = get_cell(credit_data, b, "train", credit_thresh)
        d_beat = (d_cell.get("beat") or 0) if d_cell else 0
        c_beat = (c_cell.get("beat") or 0) if c_cell else 0
        d_bpp = (d_cell.get("beat_per_point") or 0) if d_cell else 0
        c_bpp = (c_cell.get("beat_per_point") or 0) if c_cell else 0
        winner = "DEBIT" if d_beat >= c_beat else "CREDIT"
        diff = abs(d_beat - c_beat)
        lines.append(
            f"  {b:<5}  debit_beat={pf(d_beat):>7}  credit_beat={pf(c_beat):>7}"
            f"  (per point: {pf(d_bpp, '.3f')} / {pf(c_bpp, '.3f')})"
            f"  → {winner} leads by {pf(diff)}"
        )
    lines.append("  READ: Structure crossover = distance band where debit stops leading and credit starts.")

    # Summary D: Engine hypothesis (decision #13)
    lines.append("\n── Summary D: Engine hypothesis (decision #13) ──")
    from packages.shared.post_touch_qualification import _CREDIT_PATTERNS, _DEBIT_PATTERNS
    debit_train_w_pattern = [td for td in debit_data if td.partition == "train" and td.pattern_label]
    credit_train_w_pattern = [td for td in credit_data if td.partition == "train" and td.pattern_label]

    if not debit_train_w_pattern and not credit_train_w_pattern:
        lines.append("  No post-touch pattern data — engine hypothesis UNTESTABLE with current data.")
        return "\n".join(lines)

    for structure, data, patterns_for, threshold in [
        ("debit",  debit_train_w_pattern,  _DEBIT_PATTERNS,  debit_thresh),
        ("credit", credit_train_w_pattern, _CREDIT_PATTERNS, credit_thresh),
    ]:
        matched   = [td for td in data if td.pattern_label in patterns_for]
        unmatched = [td for td in data if td.pattern_label not in patterns_for]
        pairs_m = [(td, compute_pnl(td, threshold)) for td in matched]
        pairs_u = [(td, compute_pnl(td, threshold)) for td in unmatched]
        sm = aggregate(pairs_m); su = aggregate(pairs_u)
        rm = fmt_stats(sm, "pattern_match"); ru = fmt_stats(su, "pattern_no_match")
        # An empty cell — no labeled trades, or labeled trades with no settled
        # close P&L — cannot be read. Never print "adds value" / "HURTS" on it.
        if not matched or not unmatched or rm["n"] == 0 or ru["n"] == 0:
            lines.append(
                f"  {structure}: engine hypothesis UNTESTABLE "
                f"(n_match={rm['n']}, n_no_match={ru['n']})"
            )
            if summary_d_out is not None:
                summary_d_out[structure] = {
                    "n_match": rm["n"], "n_no_match": ru["n"], "read": "untestable",
                }
            continue
        lines.append(
            f"  {structure.upper()} engine check:"
            f"  pattern_match (n={rm['n']}) pnl={pf(rm.get('mean_pnl'))}"
            f"  vs no_match (n={ru['n']}) pnl={pf(ru.get('mean_pnl'))}"
        )
        m_pnl = rm.get("mean_pnl") or 0
        u_pnl = ru.get("mean_pnl") or 0
        if m_pnl > u_pnl + 0.3:
            lines.append(f"  → Pattern filter adds value for {structure} (pattern_match outperforms).")
        elif m_pnl < u_pnl - 0.3:
            lines.append(f"  → Pattern filter HURTS {structure} (match performs WORSE). Engine rule may be wrong.")
        else:
            lines.append(f"  → No clear separation by pattern for {structure}. Engine hypothesis inconclusive.")

        # CR-AL decision #7/#9: bootstrap CI on mean(match) − mean(no_match), pre-registered read
        pnl_m = [r["close_pnl"] for _, r in pairs_m if r["close_pnl"] is not None]
        pnl_u = [r["close_pnl"] for _, r in pairs_u if r["close_pnl"] is not None]
        diff = (m_pnl - u_pnl) if (pnl_m and pnl_u) else None
        ci_lo, ci_hi = bootstrap_mean_diff_ci(pnl_m, pnl_u, n_resamples=1000, seed=seed)
        excludes_zero = ci_lo is not None and (ci_lo > 0 or ci_hi < 0)
        if diff is None:
            read = "untestable"
        elif diff > 0.3 and excludes_zero:
            read = "SUPPORTED"
        elif diff < -0.3 and excludes_zero:
            read = "HURTS"
        else:
            read = "INCONCLUSIVE"
        lines.append(
            f"  {structure.upper()} decision-9 read [{_MODE_TAG}]: "
            f"match−no_match = {pf(diff, '+.2f')} pts, bootstrap 95% CI "
            f"[{pf(ci_lo, '+.2f')}, {pf(ci_hi, '+.2f')}] (1000 resamples, seed={seed}) → {read}"
        )
        if summary_d_out is not None:
            summary_d_out[structure] = {
                "n_match": rm["n"], "n_no_match": ru["n"],
                "mean_match": m_pnl, "mean_no_match": u_pnl,
                "diff": diff, "ci_lo": ci_lo, "ci_hi": ci_hi, "read": read,
            }

    return "\n".join(lines)


# ── DB PERSISTENCE ────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bt_edge_backtest_results (
    id               SERIAL PRIMARY KEY,
    run_id           UUID   NOT NULL,
    cr_id            TEXT   NOT NULL DEFAULT 'CR-AH',
    structure_type   TEXT   NOT NULL,
    outcome_type     TEXT   NOT NULL,
    distance_band    TEXT   NOT NULL,
    post_touch_pattern TEXT,
    partition        TEXT   NOT NULL,
    threshold        FLOAT  NOT NULL,
    n_dates          INT    NOT NULL,
    n_filled         INT    NOT NULL,
    n_settled        INT    NOT NULL,
    fill_rate        FLOAT,
    mean_pnl         FLOAT,
    win_rate         FLOAT,
    wilson_lo        FLOAT,
    wilson_hi        FLOAT,
    baseline_mean    FLOAT,
    beat_baseline    FLOAT,
    created_at       TIMESTAMP DEFAULT NOW(),
    -- CR-AR decision 5 (infra/sql/bt_edge_backtest_results_per_point.sql)
    close_pnl_per_point FLOAT,
    baseline_per_point  FLOAT,
    beat_per_point      FLOAT,
    mean_width_actual   FLOAT,
    -- CR-AU decision 5: mean close P&L net of commission (mean_pnl stays gross)
    mean_pnl_net        FLOAT
);
"""

# CR-AR: idempotent column add so an un-migrated DB never fails the INSERT
_PER_POINT_COLUMNS_SQL = """
ALTER TABLE bt_edge_backtest_results
  ADD COLUMN IF NOT EXISTS close_pnl_per_point FLOAT,
  ADD COLUMN IF NOT EXISTS baseline_per_point  FLOAT,
  ADD COLUMN IF NOT EXISTS beat_per_point      FLOAT,
  ADD COLUMN IF NOT EXISTS mean_width_actual   FLOAT,
  ADD COLUMN IF NOT EXISTS mean_pnl_net        FLOAT;   -- CR-AU decision 5
"""

_GRANT_SQL = "GRANT SELECT, INSERT ON bt_edge_backtest_results TO dash_backfill_writer;"


def ensure_catalog_table() -> bool:
    """Create bt_edge_backtest_results using admin connection. Returns True on success."""
    if not _admin_url or _admin_url == _bak_url:
        print("  ⚠ DATABASE_URL not set or same as BACKFILL_DATABASE_URL — skipping table creation.")
        print("  Run this SQL as admin before inserting results:")
        print(_CREATE_TABLE_SQL)
        print(_GRANT_SQL)
        return False
    try:
        admin_url = _admin_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(admin_url) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_PER_POINT_COLUMNS_SQL)
            conn.execute(_GRANT_SQL)
            conn.execute("GRANT USAGE, SELECT ON SEQUENCE bt_edge_backtest_results_id_seq TO dash_backfill_writer;")
            conn.commit()
        print("  ✓ bt_edge_backtest_results created/verified.")
        return True
    except Exception as exc:
        print(f"  ⚠ Table creation failed ({exc}). Skipping DB persistence.")
        return False


def persist_cell_stats(
    conn,
    run_id: str,
    cr_id: str,
    structure: str,
    outcome: str,
    band: str,
    pattern: Optional[str],
    partition: str,
    threshold: float,
    data: list[TradeData],
):
    pairs = [(td, compute_pnl(td, threshold)) for td in data]
    s = aggregate(pairs)
    r = fmt_stats(s)
    if r["n"] == 0:
        return

    conn.execute(
        """
        INSERT INTO bt_edge_backtest_results
          (run_id, cr_id, structure_type, outcome_type, distance_band, post_touch_pattern,
           partition, threshold, n_dates, n_filled, n_settled,
           fill_rate, mean_pnl, win_rate, wilson_lo, wilson_hi,
           baseline_mean, beat_baseline,
           close_pnl_per_point, baseline_per_point, beat_per_point, mean_width_actual,
           mean_pnl_net)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            run_id, cr_id, structure, outcome, band, pattern,
            partition, threshold,
            r.get("n", 0), r.get("n_filled", 0), r["n"],
            r.get("fill_rate"), r.get("mean_pnl"), r.get("win_rate"),
            r.get("wilson_lo"), r.get("wilson_hi"),
            r.get("baseline_mean"), r.get("beat"),
            # CR-AR decision 5: width-weighted per-point statistics
            r.get("mean_pnl_per_point"), r.get("baseline_per_point"),
            r.get("beat_per_point"), r.get("mean_width"),
            r.get("mean_pnl_net"),          # CR-AU decision 5
        ),
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="CR-AH Step 4 — two-structure × two-axis analysis")
    p.add_argument("--universe-end", type=date.fromisoformat, default=None, metavar="YYYY-MM-DD",
                   help="drop signal dates after this date before stratified selection")
    p.add_argument("--split-date", type=date.fromisoformat, default=DEFAULT_SPLIT_DATE, metavar="YYYY-MM-DD",
                   help="train/holdout split; partition='train' if trade_date <= split (default 2026-06-05)")
    p.add_argument("--train-only", action="store_true",
                   help="drop holdout entries after selection; never read holdout P&L")
    p.add_argument("--no-persist", action="store_true",
                   help="skip Phase 7 (no writes to bt_edge_backtest_results)")
    p.add_argument("--cr-id", default="CR-AH", help="cr_id for the bt_backfill_runs row")
    p.add_argument("--structural-prob-mode", choices=("full", "walk-forward"), default="walk-forward",
                   help="walk-forward: before_date=trade_date; full: allow_lookahead=True")
    p.add_argument("--seed", type=int, default=20260905, help="bootstrap seed (Summary D)")
    p.add_argument("--holdout-read", default=None, metavar="PATH",
                   help="CR-AU decision 4: path to the pre-registration document; holdout P&L prints only with it and n >= 60")
    p.add_argument("--selection-only", action="store_true",
                   help="CR-AU: stop after Phase 1 (selection, clean filter, unlistable, holdout gate); no cells")
    return p.parse_args(argv)


def main(argv=None):
    global _MODE_TAG
    args = _parse_args(argv)
    mode = args.structural_prob_mode
    _MODE_TAG = f"mode={mode}"
    if args.holdout_read is not None:
        holdout_read_gate(0, args.holdout_read)      # raises before any DB work if the file is missing

    t0 = _time.perf_counter()
    print(f"\n{SEP}")
    print("CR-AH Step 4 — Two-structure × two-axis analysis")
    print(f"  cr_id={args.cr_id}  structural-prob {_MODE_TAG}  train_only={args.train_only}  "
          f"no_persist={args.no_persist}  universe_end={args.universe_end}  "
          f"split_date={args.split_date}  seed={args.seed}  selection_only={args.selection_only}  "
          f"holdout_read={args.holdout_read}  fee_pts={VERTICAL_FEE_PTS:.3f} (${FEE_PER_CONTRACT_PER_LEG}/contract/leg)")
    print(SEP)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    with backfill_run(conn, args.cr_id) as run_id:
        print(f"\nRun ID: {run_id}")

        # ── Phase 1: Load and select dates ────────────────────────────────────
        print(f"\n{SEP2}")
        print("Phase 1: Loading signal dates and selecting clean subset...")

        all_entries = load_signal_entries(conn, universe_end=args.universe_end,
                                          split_date=args.split_date)
        selected    = select_clean_dates(all_entries)

        from collections import Counter as _Counter
        sel_counts = dict(sorted(_Counter(f"{e['band']}/{e['partition']}" for e in selected).items()))
        print(f"  Selection by band/partition: {sel_counts}")
        if args.train_only:
            n_before = len(selected)
            selected = [e for e in selected if e["partition"] == "train"]
            print(f"  --train-only: kept {len(selected)}/{n_before} train entries; "
                  f"holdout not read.")
        elif not any(e["partition"] == "holdout" for e in selected):
            print(f"  {_NO_HOLDOUT_LINE}")

        print("\nFiltering to A-bucket clean dates for each structure...")
        print("  Credit (target + target+10)...")
        credit_clean = filter_clean_for_structure(conn, selected, "credit")
        print(f"    Credit clean: {len(credit_clean)}")

        print("  Debit (target + target-10)...")
        debit_clean  = filter_clean_for_structure(conn, selected, "debit")
        print(f"    Debit clean:  {len(debit_clean)}")

        def band_part_counts(clean):
            from collections import Counter
            return Counter(f"{e['band']}/{e['partition']}" for e in clean)

        print(f"\n  Credit by band/partition: {dict(band_part_counts(credit_clean))}")
        print(f"  Debit  by band/partition: {dict(band_part_counts(debit_clean))}")

        # CR-AR decisions 1–3: pair-snapping report (width cap 2× nominal; unlistable by date)
        print(f"  Unlistable (StructureNotListed, status='unlistable', excluded before the clean filter): {len(UNLISTABLE)}")
        for u in sorted(UNLISTABLE, key=lambda u: (u["trade_date"], u["structure"])):
            print(f"    {u['structure']} {u['trade_date']} {u['band']}: {u['reason']}")
        snap_summary = {}
        for s_, clean_ in (("credit", credit_clean), ("debit", debit_clean)):
            widths = Counter(float(e.get("width_actual", WIDTH_NOMINAL)) for e in clean_)
            snap_summary[s_] = {"n": len(clean_), "width_actual_dist": {str(k): v for k, v in sorted(widths.items())},
                                "n_width_not_nominal": sum(1 for e in clean_ if float(e.get("width_actual", WIDTH_NOMINAL)) != float(e.get("width_nominal", WIDTH_NOMINAL))),
                                "n_width_narrower": sum(1 for e in clean_ if e.get("width_narrower")),
                                "width_actual_max": max((float(e.get("width_actual", WIDTH_NOMINAL)) for e in clean_), default=None),
                                "n_unlistable": sum(1 for u in UNLISTABLE if u["structure"] == s_)}
            print(f"  Snapped pairs [{s_}]: {snap_summary[s_]}")
        _w_max = max((v["width_actual_max"] or 0.0 for v in snap_summary.values()), default=0.0)
        print(f"  max(width_actual) across structures (G4, expect ≤ {2 * WIDTH_NOMINAL:g}): {_w_max:g}")

        # ── CR-AU decision 4: holdout read gate (n = holdout magnet-above computed dates) ──
        gate = holdout_read_gate(count_holdout_computed(conn, args.split_date), args.holdout_read)
        has_holdout_sel = any(e["partition"] == "holdout" for e in selected)
        if has_holdout_sel:
            today_ = date.today()
            ho_band = Counter(e["band"] for e in selected if e["partition"] == "holdout")
            ho_matured = sum(1 for e in selected if e["partition"] == "holdout"
                             and nth_business_day(e["trade_date"], DTE_TARGET) <= today_)
            print(f"  {gate.line}")
            print(f"  holdout n per band (selected): {dict(sorted(ho_band.items()))}; "
                  f"holdout dates matured (expiry ≤ {today_}): {ho_matured}/{sum(ho_band.values())}")
        else:
            print(f"  {_NO_HOLDOUT_LINE}")

        if args.selection_only:
            smoke = {"selection_only": True, "universe_end": args.universe_end.isoformat() if args.universe_end else None,
                     "split_date": args.split_date.isoformat(), "selection_by_band_partition": sel_counts,
                     "credit_clean": len(credit_clean), "debit_clean": len(debit_clean),
                     "snapping": snap_summary, "n_unlistable": len(UNLISTABLE),
                     "holdout_gate": {"n": gate.n_holdout_computed, "unlocked": gate.unlocked, "threshold": gate.threshold},
                     "elapsed_s": round(_time.perf_counter() - t0, 1)}
            update_run_smoke(conn, run_id, smoke,
                             f"selection-only [{_MODE_TAG}]: selected={len(selected)} credit_clean={len(credit_clean)} "
                             f"debit_clean={len(debit_clean)}; {gate.line}; no cells, no P&L")
            print(f"\n  --selection-only: stopping after Phase 1 ({smoke['elapsed_s']}s). No trades collected, no cells, no P&L.")
            return

        # ── Phase 2: Collect per-date data ───────────────────────────────────
        print(f"\n{SEP2}")
        print("Phase 2: Collecting per-date trade data...")

        debit_trades:  list[TradeData] = []
        credit_trades: list[TradeData] = []

        total = len(debit_clean) + len(credit_clean)
        done  = 0

        print(f"  Processing {len(debit_clean)} debit dates...")
        for entry in debit_clean:
            td = collect_trade_data(conn, entry, "debit", mode)
            if td:
                debit_trades.append(td)
            done += 1
            if done % 20 == 0:
                elapsed = _time.perf_counter() - t0
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed", flush=True)

        print(f"  Processing {len(credit_clean)} credit dates...")
        for entry in credit_clean:
            td = collect_trade_data(conn, entry, "credit", mode)
            if td:
                credit_trades.append(td)
            done += 1
            if done % 20 == 0:
                elapsed = _time.perf_counter() - t0
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed", flush=True)

        # ── CR-AN decision 6: trades whose entry window had no valid minute ──
        excluded = [(td.structure, td.trade_date, td.band, td.excluded_reason)
                    for td in debit_trades + credit_trades if td.excluded_reason]
        debit_trades  = [td for td in debit_trades  if not td.excluded_reason]
        credit_trades = [td for td in credit_trades if not td.excluded_reason]
        print(f"\n  Decision-6 exclusions (no valid entry minute): {len(excluded)}")
        for s_, d_, b_, r_ in excluded:
            print(f"    {s_} {d_} {b_}: {r_}")

        # ── CR-AN decision 7: quote-validity observability per structure ──
        def _pct(vals, p):
            if not vals:
                return None
            v = sorted(vals); i = int(round((len(v) - 1) * p))
            return v[i]
        obs_summary: dict = {}
        for s_, data_ in (("debit", debit_trades), ("credit", credit_trades)):
            valid_frac = [td.n_minutes_valid / td.n_minutes_total for td in data_ if td.n_minutes_total]
            offs = [td.baseline_minute_offset for td in data_ if td.baseline_minute_offset is not None]
            obs_summary[s_] = {
                "n": len(data_),
                "had_invalid_quote": sum(1 for td in data_ if td.had_invalid_quote),
                "valid_minute_fraction_median": round(_pct(valid_frac, 0.5), 4) if valid_frac else None,
                "valid_minute_fraction_p05": round(_pct(valid_frac, 0.05), 4) if valid_frac else None,
                "baseline_minute_offset_median": _pct(offs, 0.5),
                "baseline_minute_offset_p95": _pct(offs, 0.95),
                "baseline_minute_offset_max": max(offs) if offs else None,
                "baseline_offset_gt0": sum(1 for o in offs if o > 0),
            }
            print(f"  Quote validity [{s_}]: {obs_summary[s_]}")

        print(f"\n  Collected: debit={len(debit_trades)}, credit={len(credit_trades)}")

        # Coverage summary
        d_settle = sum(1 for td in debit_trades if td.settlement_price is not None)
        c_settle = sum(1 for td in credit_trades if td.settlement_price is not None)
        d_touch  = sum(1 for td in debit_trades if td.touch_resolution in ("rth_touch", "gap_touch"))
        c_touch  = sum(1 for td in credit_trades if td.touch_resolution in ("rth_touch", "gap_touch"))
        print(f"  Settlement available: debit={d_settle}/{len(debit_trades)}, "
              f"credit={c_settle}/{len(credit_trades)}")
        print(f"  Actionable touches: debit={d_touch}/{len(debit_trades)}, "
              f"credit={c_touch}/{len(credit_trades)}")

        # ── CR-AN G2: post-filter, no accepted spread value may be out of range ──
        oor = 0
        for td in debit_trades + credit_trades:
            r0 = compute_pnl(td, 0.0)
            w = td.spread_width
            debit_ = td.structure == "debit"
            for key in ("fill_net_credit", "baseline_net_credit"):
                v = r0.get(key)
                if v is not None and (abs(v) > w + 1e-9 or (v > 1e-9 if debit_ else v < -1e-9)):
                    oor += 1
            for key in ("close_pnl", "baseline_close_pnl", "touch_exit_pnl", "baseline_touch_exit_pnl"):
                v = r0.get(key)
                if v is not None and abs(v) > w + 1e-9:
                    oor += 1
        print(f"  Post-filter out-of-range accepted values (G2, expect 0): {oor}")

        # ── Phase 3: Threshold sweep (train only) ─────────────────────────────
        print(f"\n{SEP2}")
        print("Phase 3: Threshold sweep on TRAIN only...")

        debit_train  = [td for td in debit_trades  if td.partition == "train"]
        credit_train = [td for td in credit_trades if td.partition == "train"]

        debit_thresh,  debit_sweep  = threshold_sweep(debit_train,  "debit")
        credit_thresh, credit_sweep = threshold_sweep(credit_train, "credit")

        print(f"\n  Chosen threshold — DEBIT: {debit_thresh:.2f}  CREDIT: {credit_thresh:.2f}")
        print_sweep(debit_sweep,  debit_thresh,  "DEBIT")
        print_sweep(credit_sweep, credit_thresh, "CREDIT")

        # ── Phase 4: Full results ─────────────────────────────────────────────
        print(f"\n{SEP2}")
        has_holdout = any(td.partition == "holdout" for td in debit_trades + credit_trades)
        if args.train_only:
            print("Phase 4: Full results (TRAIN only — holdout not read)")
        elif not has_holdout:
            print(f"Phase 4: Full results (all train; {_NO_HOLDOUT_LINE})")
        elif not gate.unlocked:
            print(f"Phase 4: Full results (train; {gate.line} — holdout rows show n and dates matured only)")
            for s_, data_ in (("debit", debit_trades), ("credit", credit_trades)):
                ho = [td for td in data_ if td.partition == "holdout"]
                print(f"  {s_}: holdout n={len(ho)}, dates matured={sum(1 for td in ho if td.settlement_price is not None)}")
        else:
            print(f"Phase 4: Full results (HOLDOUT READ at chosen threshold — {gate.line})")

        band_label = "train only" if args.train_only else "all splits"
        print_by_band(debit_trades,  debit_thresh,  "debit",  band_label, train_only=args.train_only, holdout_unlocked=gate.unlocked)
        print_by_band(credit_trades, credit_thresh, "credit", band_label, train_only=args.train_only, holdout_unlocked=gate.unlocked)

        print_by_pattern(debit_train,  debit_thresh,  "debit")
        print_by_pattern(credit_train, credit_thresh, "credit")

        print_debit_touch_breakdown(debit_train, debit_thresh)

        # ── Phase 5: Selection bias ───────────────────────────────────────────
        debit_bias  = selection_bias_check(selected, debit_clean,  "debit")
        credit_bias = selection_bias_check(selected, credit_clean, "credit")
        print_selection_bias(debit_bias)
        print_selection_bias(credit_bias)

        # ── Phase 6: Summary reads A/B/C/D ───────────────────────────────────
        print(f"\n{SEP}")
        print("SUMMARY READS A/B/C/D")
        print(SEP)
        summary_d: dict = {}
        summary = build_summary(
            debit_trades, credit_trades, debit_thresh, credit_thresh,
            train_only=args.train_only, seed=args.seed, summary_d_out=summary_d,
            holdout_gate=gate,
        )
        print(summary)

        # ── Phase 7: DB persistence ───────────────────────────────────────────
        print(f"\n{SEP2}")
        if args.no_persist:
            print("Phase 7: skipped (--no-persist) — bt_edge_backtest_results untouched.")
            table_ok = False
        else:
            print("Phase 7: Persisting aggregate stats to bt_edge_backtest_results...")
            table_ok = ensure_catalog_table()
        if table_ok:
            for structure, data, thresh in [
                ("debit",  debit_trades,  debit_thresh),
                ("credit", credit_trades, credit_thresh),
            ]:
                for part in ("train", "holdout"):
                    if part == "holdout" and not any(td.partition == "holdout" for td in data):
                        print(f"  {structure}: {_NO_HOLDOUT_LINE}")
                        continue
                    if part == "holdout" and not gate.unlocked:      # CR-AU A6: no holdout cells while the read is locked
                        print(f"  {structure}: holdout cells not persisted ({gate.line})")
                        continue
                    for band in ("near", "mid", "far", "all"):
                        if band == "all":
                            subset = [td for td in data if td.partition == part]
                        else:
                            subset = [td for td in data if td.partition == part and td.band == band]
                        if not subset:
                            continue
                        persist_cell_stats(conn, run_id, args.cr_id, structure, "close",
                                          band, None, part, thresh, subset)
            conn.commit()
            print("  ✓ Aggregate stats written.")
        elif not args.no_persist:
            print("  Skipping DB persistence — table not available.")

        # ── Smoke summary ─────────────────────────────────────────────────────
        elapsed = _time.perf_counter() - t0
        smoke = {
            "debit_n": len(debit_trades),
            "credit_n": len(credit_trades),
            "debit_settled": d_settle,
            "credit_settled": c_settle,
            "debit_thresh": debit_thresh,
            "credit_thresh": credit_thresh,
            "elapsed_s": round(elapsed, 1),
            # CR-AL provenance
            "structural_prob_mode": mode,
            "train_only": args.train_only,
            "no_persist": args.no_persist,
            "universe_end": args.universe_end.isoformat() if args.universe_end else None,
            "split_date": args.split_date.isoformat(),
            "seed": args.seed,
            "fee_pts": VERTICAL_FEE_PTS,
            "holdout_gate": {"n": gate.n_holdout_computed, "unlocked": gate.unlocked, "threshold": gate.threshold},
            "selection_by_band_partition": sel_counts,
            "quote_validity": obs_summary,
            "snapping": snap_summary,
            "width_actual_max": _w_max,
            "n_unlistable": len(UNLISTABLE),
            "unlistable": [f"{u['structure']} {u['trade_date']} {u['band']}: {u['reason']}" for u in UNLISTABLE],
            "decision6_excluded": [f"{s_} {d_} {b_}" for s_, d_, b_, _ in excluded],
            "post_filter_out_of_range": oor,
            "debit_labeled_train": sum(1 for td in debit_trades if td.partition == "train" and td.pattern_label),
            "credit_labeled_train": sum(1 for td in credit_trades if td.partition == "train" and td.pattern_label),
            "summary_d": summary_d,
        }
        update_run_smoke(conn, run_id, smoke,
                        f"Step 4 complete [{_MODE_TAG}]: debit={len(debit_trades)}, "
                        f"credit={len(credit_trades)}, T_d={debit_thresh}, "
                        f"T_c={credit_thresh}, {elapsed:.0f}s; "
                        f"summary_d={ {k: v.get('read') for k, v in summary_d.items()} }")

    print(f"\n{SEP}")
    print(f"Step 4 complete in {_time.perf_counter()-t0:.0f}s")
    print(SEP)


if __name__ == "__main__":
    main()
