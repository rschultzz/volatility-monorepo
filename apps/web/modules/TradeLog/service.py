"""
apps/web/modules/TradeLog/service.py

Business logic for the Trade Log feature:
  - TV CSV parsing (CT → PT conversion)
  - FIFO fill pairing into round-trip trades
  - DB insert / CRUD
  - Market-context computation via BacktestsV2 helpers
"""

import csv
import io
import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
import numpy as np
from packages.shared.surface_compare import k_for_abs_delta as _k_for_abs_delta
SURFACE_COMPARE_AVAILABLE = True

import psycopg2
import psycopg2.extras

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ── Timezone constants ────────────────────────────────────────
CT = ZoneInfo('America/Chicago')
PT = ZoneInfo('America/Los_Angeles')

# ── Futures multipliers (pts → USD per contract) ─────────────
MULTIPLIERS = {
    'MES': 5,
    'ES':  50,
    'MNQ': 2,
    'NQ':  20,
}
DEFAULT_MULTIPLIER = 5  # fallback

ET  = ZoneInfo('America/New_York')
_EPS_T                      = 1e-4
_BETA_VOLPTS_PER_1PCT       = 4.5
_BETA_MAX_SHIFT_PP          = 6.0
_THETA_ATM_PP_PER_SQRT_YEAR = -638

# Month codes for futures contract symbol stripping
_MONTH_CODES = frozenset('FGHJKMNQUVXZ')

log = logging.getLogger(__name__)


# ── Skew helpers — use the same shared utility as Skew callbacks ──────────────
try:
    from packages.shared.utils import fetch_skew_data as _fetch_skew_data
    import pandas as _pd
    SKEW_HELPERS_AVAILABLE = True
    log.info('TradeLog: skew helpers loaded OK')
except Exception as _e:
    SKEW_HELPERS_AVAILABLE = False
    _fetch_skew_data = None
    _pd = None
    log.warning('TradeLog: skew helpers not available (%s). Context will return nulls.', _e)

# ── Skew math (mirrors callbacks.py exactly) ──────────────────────────────────
_MIN_SKEW_DENOM_PP = 0.25

def _skews_from_row(row):
    """Return (atm_frac, call_skew_pp, put_skew_pp) from an orats_monies_minute row."""
    import pandas as pd
    atm  = float(pd.to_numeric(row.get('vol50'), errors='coerce'))
    c25  = float(pd.to_numeric(row.get('vol25'), errors='coerce'))
    p25  = float(pd.to_numeric(row.get('vol75'), errors='coerce'))
    return atm, (c25 - atm) * 100.0, (p25 - atm) * 100.0

def _pct_change_frac(curr, base):
    if base in (None, 0) or curr is None: return None
    return (curr - base) / abs(base) * 100.0

def _pct_change_pp(curr_pp, base_pp):
    if curr_pp is None or base_pp is None: return None
    denom = max(abs(base_pp), _MIN_SKEW_DENOM_PP)
    return (curr_pp - base_pp) / denom * 100.0


# ══════════════════════════════════════════════════════════════
# DB connection
# ══════════════════════════════════════════════════════════════

def _clean_db_url(url: str) -> str:
    """
    psycopg2 needs  postgresql://...
    SQLAlchemy stores  postgresql+psycopg://... — strip the driver suffix.
    """
    return (url
            .replace('postgresql+psycopg2', 'postgresql')
            .replace('postgresql+psycopg', 'postgresql'))


def get_conn():
    raw_url = os.environ['DATABASE_URL']
    url = _clean_db_url(raw_url)
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


# ══════════════════════════════════════════════════════════════
# Symbol helpers
# ══════════════════════════════════════════════════════════════

def extract_root(symbol: str) -> str:
    """
    'F.US.MESM26' → 'MES'
    Strips the exchange prefix (F.US.) then removes trailing
    month-letter + digits from the contract code.
    """
    parts = symbol.split('.')
    code = parts[-1] if parts else symbol
    # Walk backwards past digits then past a month-letter
    i = len(code) - 1
    while i >= 0 and code[i].isdigit():
        i -= 1
    if i >= 0 and code[i] in _MONTH_CODES:
        i -= 1
    return code[:i + 1] if i >= 0 else code


def get_multiplier(root: str) -> int:
    return MULTIPLIERS.get(root.upper(), DEFAULT_MULTIPLIER)


# ══════════════════════════════════════════════════════════════
# CSV parsing
# ══════════════════════════════════════════════════════════════

_TV_DATETIME_FORMATS = [
    '%Y-%m-%d %H:%M:%S',
    '%m/%d/%Y %I:%M:%S %p',
    '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%d %H:%M',
    # AMP / short formats with 2-digit year
    '%m/%d/%y %H:%M:%S',
    '%m/%d/%y %I:%M:%S %p',
    '%m/%d/%y %H:%M',
]


def _parse_tv_datetime(value: str):
    """
    Parse a TradingView datetime string (assumed Central Time) and
    return a timezone-aware datetime in Pacific Time.
    Returns None on failure.
    """
    s = (value or '').strip()
    if not s:
        return None
    for fmt in _TV_DATETIME_FORMATS:
        try:
            naive = datetime.strptime(s, fmt)
            return naive.replace(tzinfo=CT).astimezone(PT)
        except ValueError:
            continue
    log.debug('Could not parse TV datetime: %r', value)
    return None


def _normalize_side(raw: str) -> str:
    """'Buy' / 'B' / 'BUY' → 'B',  'Sell' / 'S' → 'S'"""
    s = raw.strip().upper()
    if s in ('B', 'BUY', 'LONG'):
        return 'B'
    if s in ('S', 'SELL', 'SHORT'):
        return 'S'
    return s


def parse_tv_csv(file_bytes: bytes) -> tuple[list[dict], list[str]]:
    """
    Parse a TradingView fills CSV.

    Returns:
        (fills, warnings)
        fills    — list of fill dicts (only filled orders, CT→PT converted)
        warnings — list of human-readable warning strings
    """
    text = file_bytes.decode('utf-8-sig')   # strip BOM if present
    reader = csv.DictReader(io.StringIO(text))

    fills = []
    warnings = []
    skipped_cancelled = 0
    skipped_unparseable = 0

    for raw_row in reader:
        row = {k.strip(): (v or '').strip() for k, v in raw_row.items()}

        status = row.get('Status', '').lower()
        if status == 'cancelled':
            skipped_cancelled += 1
            continue

        fill_qty_raw  = row.get('Fill Qty', '')
        fill_px_raw   = row.get('Avg Fill Price', '')

        if not fill_qty_raw or not fill_px_raw:
            skipped_unparseable += 1
            continue

        try:
            fill_qty   = int(float(fill_qty_raw))
            fill_price = float(fill_px_raw)
        except ValueError:
            skipped_unparseable += 1
            continue

        if fill_qty <= 0:
            skipped_unparseable += 1
            continue

        order_id = row.get('Order ID', '').strip()
        if not order_id:
            warnings.append(f'Row missing Order ID, skipped: {row}')
            continue

        placing_time_pt = _parse_tv_datetime(row.get('Placing Time', ''))
        status_time_pt  = _parse_tv_datetime(row.get('Status Time', ''))

        fills.append({
            'order_id':       order_id,
            'symbol':         row.get('Symbol', ''),
            'side':           _normalize_side(row.get('Side', '')),
            'order_type':     row.get('Type', ''),
            'qty':            int(float(row.get('Qty', 0) or 0)),
            'fill_qty':       fill_qty,
            'fill_price':     fill_price,
            'commission':     float(row.get('Commission', 0) or 0),
            'duration':       row.get('Duration', ''),
            'placing_time_pt': placing_time_pt,
            'status_time_pt':  status_time_pt,
            'raw_csv_row':    row,
        })

    if skipped_cancelled:
        warnings.append(f'{skipped_cancelled} cancelled order(s) skipped.')
    if skipped_unparseable:
        warnings.append(f'{skipped_unparseable} row(s) skipped (missing fill data).')

    # Sort chronologically by placing time (fall back to status time)
    def _sort_key(f):
        return f['placing_time_pt'] or f['status_time_pt'] or datetime.min.replace(tzinfo=PT)

    fills.sort(key=_sort_key)
    return fills, warnings


# ══════════════════════════════════════════════════════════════
# FIFO pairing
# ══════════════════════════════════════════════════════════════

def fifo_pair_fills(fills: list[dict]) -> tuple[list[dict], list[str]]:
    """
    FIFO-pair fills into round-trip trades.
    Handles scaling into a position (buy 1, buy 1, sell 2 → two trades).
    Open positions at the end of the fill list are silently dropped
    with a warning per symbol.

    Returns:
        (trades, warnings)
    """
    # Group by symbol root so MES and ES don't cross-contaminate
    by_root: dict[str, list[dict]] = {}
    for f in fills:
        root = extract_root(f['symbol'])
        by_root.setdefault(root, []).append(f)

    trades: list[dict] = []
    warnings: list[str] = []

    for root, sym_fills in by_root.items():
        multiplier = get_multiplier(root)
        queue: deque[dict] = deque()   # open entry fills with 'remaining' qty

        for fill in sym_fills:
            side      = fill['side']
            remaining = fill['fill_qty']

            if not queue:
                queue.append({**fill, 'remaining': remaining})
                continue

            if side == queue[0]['side']:
                # Same direction — scaling in
                queue.append({**fill, 'remaining': remaining})
                continue

            # Opposite direction — match against queue FIFO
            exit_fill   = fill
            exit_total  = exit_fill['fill_qty']

            while remaining > 0 and queue:
                entry = queue[0]
                matched = min(remaining, entry['remaining'])

                direction = 'long' if entry['side'] == 'B' else 'short'

                if direction == 'long':
                    realized_pts = exit_fill['fill_price'] - entry['fill_price']
                else:
                    realized_pts = entry['fill_price'] - exit_fill['fill_price']

                realized_pnl_usd = realized_pts * matched * multiplier

                # Prorate commissions proportionally to matched qty
                entry_comm = entry['commission'] * (matched / entry['fill_qty'])
                exit_comm  = exit_fill['commission'] * (matched / exit_total)
                fees_usd   = entry_comm + exit_comm
                net_pnl    = realized_pnl_usd - fees_usd

                entry_ts = entry['placing_time_pt'] or entry['status_time_pt']
                exit_ts  = exit_fill['placing_time_pt'] or exit_fill['status_time_pt']
                trade_date = (entry_ts or exit_ts).date().isoformat() if (entry_ts or exit_ts) else None

                trades.append({
                    'trade_date':       trade_date,
                    'symbol':           entry['symbol'],
                    'direction':        direction,
                    'qty':              matched,
                    'entry_order_id':   entry['order_id'],
                    'entry_ts_pt':      entry_ts,
                    'entry_price':      entry['fill_price'],
                    'entry_order_type': entry['order_type'],
                    'exit_order_id':    exit_fill['order_id'],
                    'exit_ts_pt':       exit_ts,
                    'exit_price':       exit_fill['fill_price'],
                    'exit_order_type':  exit_fill['order_type'],
                    'realized_pts':     realized_pts,
                    'realized_pnl_usd': realized_pnl_usd,
                    'fees_usd':         fees_usd,
                    'net_pnl_usd':      net_pnl,
                    # Initial annotation placeholders — user overrides via UI
                    'setup_start_ts_pt':  entry_ts,
                    'setup_target_ts_pt': exit_ts,
                })

                entry['remaining'] -= matched
                remaining -= matched
                if entry['remaining'] <= 0:
                    queue.popleft()

            if remaining > 0:
                # Residual exit qty opens a new position in the opposite direction
                queue.append({**exit_fill, 'remaining': remaining})

        if queue:
            open_qty = sum(e['remaining'] for e in queue)
            warnings.append(
                f'{root}: {open_qty} lot(s) left open at end of fills — unmatched trade(s) dropped.'
            )

    trades.sort(key=lambda t: t['entry_ts_pt'] or datetime.min.replace(tzinfo=PT))
    return trades, warnings


# ══════════════════════════════════════════════════════════════
# Upload orchestration
# ══════════════════════════════════════════════════════════════

def upload_csv(file_bytes: bytes) -> dict:
    """
    Full upload pipeline:
      1. Parse CSV → fills
      2. Insert new fills (dedup by order_id)
      3. Load ALL fills from DB for affected symbols
      4. Re-pair ALL fills → trades
      5. Insert new trades (dedup by entry+exit order_id)
      6. Auto-compute context for newly inserted trades
    Returns a summary dict.
    """
    fills, parse_warnings = parse_tv_csv(file_bytes)
    if not fills:
        return {'ok': False, 'error': 'No filled orders found in CSV.', 'warnings': parse_warnings}

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # ── 1. Insert new fills ───────────────────────────────
                new_fill_count = 0
                for f in fills:
                    cur.execute("""
                        INSERT INTO trade_log_fills
                            (order_id, symbol, side, order_type, qty, fill_qty,
                             fill_price, commission, duration,
                             placing_time_pt, status_time_pt, raw_csv_row)
                        VALUES (%(order_id)s, %(symbol)s, %(side)s, %(order_type)s,
                                %(qty)s, %(fill_qty)s, %(fill_price)s, %(commission)s,
                                %(duration)s, %(placing_time_pt)s, %(status_time_pt)s,
                                %(raw_csv_row)s)
                        ON CONFLICT (order_id) DO NOTHING
                    """, {**f, 'raw_csv_row': json.dumps(f['raw_csv_row'])})
                    new_fill_count += cur.rowcount

                # ── 2. Load ALL fills from DB (for affected symbols) ──
                roots = list({extract_root(f['symbol']) for f in fills})
                # Match by symbol prefix using LIKE patterns
                like_patterns = [f'F.US.{r}%' for r in roots] + roots
                cur.execute("""
                    SELECT order_id, symbol, side, order_type,
                           qty, fill_qty, fill_price, commission, duration,
                           placing_time_pt, status_time_pt
                    FROM trade_log_fills
                    WHERE symbol LIKE ANY(%s)
                       OR symbol = ANY(%s)
                    ORDER BY placing_time_pt ASC NULLS LAST
                """, (like_patterns, roots))
                all_db_fills = []
                for r in cur.fetchall():
                    row = dict(r)
                    # psycopg2 returns NUMERIC columns as Decimal — cast to float
                    # so downstream arithmetic works without type errors.
                    row['commission'] = float(row.get('commission') or 0)
                    row['fill_price'] = float(row.get('fill_price') or 0)
                    row['fill_qty']   = int(row.get('fill_qty') or 0)
                    row['qty']        = int(row.get('qty') or 0)
                    all_db_fills.append(row)

                # ── 3. Re-pair ALL fills ──────────────────────────────
                paired_trades, pair_warnings = fifo_pair_fills(all_db_fills)

                # ── 4. Insert new trades (skip existing by unique constraint) ──
                new_trade_ids = []
                for t in paired_trades:
                    cur.execute("""
                        INSERT INTO trade_log_trades
                            (trade_date, symbol, direction, qty,
                             entry_order_id, entry_ts_pt, entry_price, entry_order_type,
                             exit_order_id,  exit_ts_pt,  exit_price,  exit_order_type,
                             realized_pts, realized_pnl_usd, fees_usd, net_pnl_usd,
                             setup_start_ts_pt, setup_target_ts_pt)
                        VALUES
                            (%(trade_date)s, %(symbol)s, %(direction)s, %(qty)s,
                             %(entry_order_id)s, %(entry_ts_pt)s, %(entry_price)s, %(entry_order_type)s,
                             %(exit_order_id)s,  %(exit_ts_pt)s,  %(exit_price)s,  %(exit_order_type)s,
                             %(realized_pts)s, %(realized_pnl_usd)s, %(fees_usd)s, %(net_pnl_usd)s,
                             %(setup_start_ts_pt)s, %(setup_target_ts_pt)s)
                        ON CONFLICT (entry_order_id, exit_order_id) DO NOTHING
                        RETURNING id
                    """, t)
                    row = cur.fetchone()
                    if row:
                        new_trade_ids.append(row['id'])

        # ── 5. Auto-compute context for newly inserted trades ─────────
        ctx_errors = 0
        for trade_id in new_trade_ids:
            try:
                recompute_context(trade_id)
            except Exception as e:
                log.warning('Context auto-compute failed for trade %s: %s', trade_id, e)
                ctx_errors += 1

        all_warnings = parse_warnings + pair_warnings
        if ctx_errors:
            all_warnings.append(
                f'Market context could not be computed for {ctx_errors} trade(s). '
                'Use "Recompute" after setting annotation times.'
            )

        return {
            'ok': True,
            'new_fills':  new_fill_count,
            'new_trades': len(new_trade_ids),
            'warnings':   all_warnings,
        }
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# CRUD
# ══════════════════════════════════════════════════════════════

def _format_trade(row: dict) -> dict:
    """Convert a DB row to a JSON-serializable dict."""
    def _ts(v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)

    def _dec(v):
        if isinstance(v, Decimal):
            return float(v)
        return v

    return {
        'id':                       row['id'],
        'trade_date':               str(row['trade_date']) if row.get('trade_date') else None,
        'symbol':                   row.get('symbol'),
        'direction':                row.get('direction'),
        'qty':                      row.get('qty'),
        'entry_order_id':           row.get('entry_order_id'),
        'entry_ts_pt':              _ts(row.get('entry_ts_pt')),
        'entry_price':              _dec(row.get('entry_price')),
        'entry_order_type':         row.get('entry_order_type'),
        'exit_order_id':            row.get('exit_order_id'),
        'exit_ts_pt':               _ts(row.get('exit_ts_pt')),
        'exit_price':               _dec(row.get('exit_price')),
        'exit_order_type':          row.get('exit_order_type'),
        'realized_pts':             _dec(row.get('realized_pts')),
        'realized_pnl_usd':         _dec(row.get('realized_pnl_usd')),
        'fees_usd':                 _dec(row.get('fees_usd')),
        'net_pnl_usd':              _dec(row.get('net_pnl_usd')),
        'setup_start_ts_pt':        _ts(row.get('setup_start_ts_pt')),
        'setup_target_ts_pt':       _ts(row.get('setup_target_ts_pt')),
        'setup_direction':          row.get('setup_direction'),
        'notes':                    row.get('notes'),
        'context_iv_atm_0dte_pct':  _dec(row.get('context_iv_atm_0dte_pct')),
        'context_target_spx_price': _dec(row.get('context_target_spx_price')),
        'context_skew_delta_put_pct':  _dec(row.get('context_skew_delta_put_pct')),
        'context_skew_delta_call_pct': _dec(row.get('context_skew_delta_call_pct')),
        'context_skew_delta_atm_iv':   _dec(row.get('context_skew_delta_atm_iv')),
        'context_minutes_to_close':    _dec(row.get('context_minutes_to_close')),
        'context_computed_at':      _ts(row.get('context_computed_at')),
        'created_at':               _ts(row.get('created_at')),
        'updated_at':               _ts(row.get('updated_at')),
    }


def list_trades(date_filter: str | None = None) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if date_filter:
                cur.execute("""
                    SELECT * FROM trade_log_trades
                    WHERE trade_date = %s
                    ORDER BY entry_ts_pt ASC
                """, (date_filter,))
            else:
                cur.execute("""
                    SELECT * FROM trade_log_trades
                    ORDER BY trade_date DESC, entry_ts_pt ASC
                """)
            return [_format_trade(dict(r)) for r in cur.fetchall()]
    finally:
        conn.close()


def get_trade(trade_id: int) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM trade_log_trades WHERE id = %s', (trade_id,))
            row = cur.fetchone()
            return _format_trade(dict(row)) if row else None
    finally:
        conn.close()


def update_trade(trade_id: int, patch: dict) -> dict | None:
    """
    Apply annotation fields from patch to the trade.
    Only setup_start_ts_pt, setup_target_ts_pt, setup_direction, notes
    are writable this way.
    """
    allowed = {'setup_start_ts_pt', 'setup_target_ts_pt', 'setup_direction', 'notes'}
    updates = {k: v for k, v in patch.items() if k in allowed}
    if not updates:
        return get_trade(trade_id)

    # Parse datetime-local strings (YYYY-MM-DDTHH:MM) as PT
    for ts_field in ('setup_start_ts_pt', 'setup_target_ts_pt'):
        if ts_field in updates and updates[ts_field]:
            raw = updates[ts_field]
            try:
                naive = datetime.fromisoformat(raw)
                updates[ts_field] = naive.replace(tzinfo=PT)
            except (ValueError, TypeError):
                updates[ts_field] = None

    set_clauses = ', '.join(f'{k} = %({k})s' for k in updates)
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'UPDATE trade_log_trades SET {set_clauses} WHERE id = %(id)s',
                    {**updates, 'id': trade_id}
                )
        return get_trade(trade_id)
    finally:
        conn.close()


def delete_trade(trade_id: int) -> bool:
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM trade_log_trades WHERE id = %s', (trade_id,))
                return cur.rowcount > 0
    finally:
        conn.close()


def get_aggregate(date_filter: str | None = None) -> dict:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            where = 'WHERE trade_date = %s' if date_filter else ''
            params = (date_filter,) if date_filter else ()
            cur.execute(f"""
                SELECT
                    COUNT(*)                                                   AS total_trades,
                    COUNT(*) FILTER (WHERE net_pnl_usd > 0)                   AS wins,
                    COUNT(*) FILTER (WHERE net_pnl_usd <= 0)                  AS losses,
                    COALESCE(SUM(net_pnl_usd), 0)                             AS total_net_pnl,
                    COALESCE(SUM(realized_pts), 0)                            AS total_realized_pts,
                    AVG(net_pnl_usd) FILTER (WHERE net_pnl_usd > 0)          AS avg_winner,
                    AVG(net_pnl_usd) FILTER (WHERE net_pnl_usd <= 0)         AS avg_loser,
                    COALESCE(SUM(fees_usd), 0)                                AS total_fees
                FROM trade_log_trades
                {where}
            """, params)
            row = dict(cur.fetchone())

        def _f(v):
            return float(v) if v is not None else None

        total  = int(row['total_trades'])
        wins   = int(row['wins'])
        losses = int(row['losses'])
        avg_w  = _f(row['avg_winner'])
        avg_l  = _f(row['avg_loser'])

        rr = None
        if avg_w and avg_l and avg_l != 0:
            rr = round(avg_w / abs(avg_l), 2)

        return {
            'ok':                 True,
            'total_trades':       total,
            'wins':               wins,
            'losses':             losses,
            'win_rate_pct':       round(100 * wins / total, 1) if total else None,
            'total_net_pnl':      _f(row['total_net_pnl']),
            'total_realized_pts': _f(row['total_realized_pts']),
            'avg_winner':         avg_w,
            'avg_loser':          avg_l,
            'risk_reward':        rr,
            'total_fees':         _f(row['total_fees']),
        }
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# Context computation
# ══════════════════════════════════════════════════════════════

def _years_to_exp(snap_shot_date_utc, expir_date_str: str) -> float:
    import datetime as _dt
    if snap_shot_date_utc is None:
        return _EPS_T
    # Handle string — psycopg2 may return snap_shot_date as a string
    if isinstance(snap_shot_date_utc, str):
        snap_shot_date_utc = _dt.datetime.fromisoformat(snap_shot_date_utc.replace('Z', '+00:00'))
    if snap_shot_date_utc.tzinfo is None:
        snap_shot_date_utc = snap_shot_date_utc.replace(tzinfo=timezone.utc)
    ts_et = snap_shot_date_utc.astimezone(ET)
    exp_date  = _dt.date.fromisoformat(expir_date_str)
    exp_dt_et = _dt.datetime.combine(exp_date, _dt.time(16, 0)).replace(tzinfo=ET)
    rem = exp_dt_et - ts_et
    T   = max(0.0, rem.total_seconds() / (365.0 * 24 * 3600))
    return max(T, _EPS_T)


def _available_buckets(row: dict) -> list:
    buckets = []
    for c in row.keys():
        if c.startswith('vol') and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                buckets.append(n)
    puts  = sorted([n for n in buckets if n >= 50], reverse=True)
    calls = sorted([n for n in buckets if n < 50],  reverse=True)
    out, seen = [], set()
    for n in puts + calls:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _abs_delta_is_put(bucket: int):
    if bucket == 50:  return 0.50, False
    if bucket > 50:   return (100 - bucket) / 100.0, True
    return bucket / 100.0, False


def _prev_smile_interp(row: dict, T_prev: float):
    atm_prev = float(row['vol50'])
    buckets  = _available_buckets(row)
    k_list, s_list = [], []
    for n in buckets:
        v = row.get(f'vol{n}')
        if v is None:
            continue
        if n == 50:
            k = 0.0
        else:
            p, is_put = _abs_delta_is_put(n)
            k = _k_for_abs_delta(p, is_put=is_put, sigma=atm_prev, T=T_prev)
        k_list.append(k)
        s_list.append(float(v))
    k_np = np.array(k_list, float)
    s_np = np.array(s_list, float)
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))
    k_np, s_np = k_np[mask], s_np[mask]
    if k_np.size < 3:
        raise ValueError('k-grid degenerate')
    return k_np, s_np


def _interp_linear_extrap(kq: float, k_grid, s_grid) -> float:
    if kq <= k_grid[0]:
        x0, x1, y0, y1 = k_grid[0], k_grid[1], s_grid[0], s_grid[1]
        return float(y0 + (y1 - y0) * (kq - x0) / (x1 - x0))
    if kq >= k_grid[-1]:
        x0, x1, y0, y1 = k_grid[-2], k_grid[-1], s_grid[-2], s_grid[-1]
        return float(y1 + (y1 - y0) * (kq - x1) / (x1 - x0))
    return float(np.interp(kq, k_grid, s_grid))

def recompute_context(trade_id: int) -> dict:
    print(f"DEBUG recompute_context called trade={trade_id}", flush=True)
    trade = get_trade(trade_id)
    if not trade:
        return {'ok': False, 'error': 'Trade not found'}

    print(f"DEBUG SURFACE_COMPARE_AVAILABLE={SURFACE_COMPARE_AVAILABLE}", flush=True)

    ctx = {
        'context_iv_atm_0dte_pct':     None,
        'context_target_spx_price':    None,
        'context_skew_delta_put_pct':  None,
        'context_skew_delta_call_pct': None,
        'context_skew_delta_atm_iv':   None,
        'context_minutes_to_close':    None,
    }

    trade_date = trade.get('trade_date')
    start_raw  = trade.get('setup_start_ts_pt')
    target_raw = trade.get('setup_target_ts_pt')

    print(f"DEBUG trade_date={trade_date} start={start_raw} target={target_raw}", flush=True)


    if trade_date and start_raw and target_raw:
        try:
            start_dt  = datetime.fromisoformat(start_raw).astimezone(PT)
            target_dt = datetime.fromisoformat(target_raw).astimezone(PT)

            start_hhmm  = (start_dt  + timedelta(minutes=1)).strftime('%H:%M')
            target_hhmm = (target_dt + timedelta(minutes=1)).strftime('%H:%M')

            print(f"DEBUG querying start={start_hhmm} target={target_hhmm}", flush=True)

            expir_date = trade_date

            conn = get_conn()
            try:
                with conn.cursor() as cur:
                    row_sql = """
                        SELECT *
                        FROM orats_monies_minute
                        WHERE trade_date = %s
                          AND expir_date = %s
                          AND TO_CHAR(snapshot_pt, 'HH24:MI') = %s
                        ORDER BY snapshot_pt
                        LIMIT 1
                    """
                    cur.execute(row_sql, (trade_date, expir_date, start_hhmm))
                    start_row = cur.fetchone()
                    cur.execute(row_sql, (trade_date, expir_date, target_hhmm))
                    target_row = cur.fetchone()
            finally:
                conn.close()

            print(f"DEBUG start_row={start_row is not None} target_row={target_row is not None}", flush=True)
            print(f"DEBUG start_row expir={start_row.get('expir_date')} snap={start_row.get('snapshot_pt')} vol50={start_row.get('vol50')}", flush=True)
            print(f"DEBUG target_row expir={target_row.get('expir_date')} snap={target_row.get('snapshot_pt')} vol50={target_row.get('vol50')}", flush=True)

            if start_row and target_row:
                start_row  = dict(start_row)
                target_row = dict(target_row)

                atm_s, call_s, put_s = _skews_from_row(start_row)
                atm_t, call_t, put_t = _skews_from_row(target_row)

                stock_s = float(start_row.get('stock_price') or 0) or None
                stock_t = float(target_row.get('stock_price') or 0) or None

                print(f"DEBUG atm_s={atm_s} call_s={call_s} put_s={put_s}", flush=True)
                print(f"DEBUG atm_t={atm_t} call_t={call_t} put_t={put_t}", flush=True)
                print(f"DEBUG stock_s={stock_s} stock_t={stock_t}", flush=True)

                ctx['context_iv_atm_0dte_pct']  = round(atm_s * 100.0, 4)
                ctx['context_target_spx_price'] = stock_t

                if SURFACE_COMPARE_AVAILABLE and stock_s and stock_t:
                    try:
                        T_s = _years_to_exp(start_row.get('snap_shot_date'),  expir_date)
                        T_t = _years_to_exp(target_row.get('snap_shot_date'), expir_date)
                        print(f"DEBUG T_s={T_s} T_t={T_t}", flush=True)

                        k_prev, s_prev = _prev_smile_interp(start_row, T_s)
                        k_shift  = math.log(stock_t / stock_s)
                        ret_frac = (stock_t - stock_s) / stock_s

                        exp_atm_shape  = _interp_linear_extrap(k_shift, k_prev, s_prev)
                        level_shift_pp = max(-_BETA_MAX_SHIFT_PP,
                                            min(_BETA_MAX_SHIFT_PP,
                                                (-ret_frac) * 100.0 * _BETA_VOLPTS_PER_1PCT))
                        droot        = max(0.0, math.sqrt(max(T_s, _EPS_T)) - math.sqrt(max(T_t, _EPS_T)))
                        atm_theta_pp = _THETA_ATM_PP_PER_SQRT_YEAR * droot
                        atm_exp      = exp_atm_shape + (level_shift_pp / 100.0) + (atm_theta_pp / 100.0)

                        k_c25_t = _k_for_abs_delta(0.25, is_put=False, sigma=atm_t, T=T_t)
                        k_p25_t = _k_for_abs_delta(0.25, is_put=True,  sigma=atm_t, T=T_t)

                        exp_c25_shape = _interp_linear_extrap(k_c25_t + k_shift, k_prev, s_prev)
                        exp_p25_shape = _interp_linear_extrap(k_p25_t + k_shift, k_prev, s_prev)

                        shift_frac = atm_exp - exp_atm_shape
                        exp_c25    = exp_c25_shape + shift_frac
                        exp_p25    = exp_p25_shape + shift_frac

                        exp_call_skew_pp = (exp_c25 - atm_exp) * 100.0
                        exp_put_skew_pp  = (exp_p25 - atm_exp) * 100.0

                        print(f"DEBUG exp_call_skew_pp={exp_call_skew_pp} exp_put_skew_pp={exp_put_skew_pp}", flush=True)

                        ctx['context_skew_delta_call_pct'] = _pct_change_pp(call_t, exp_call_skew_pp)
                        ctx['context_skew_delta_put_pct']  = _pct_change_pp(put_t,  exp_put_skew_pp)
                        ctx['context_skew_delta_atm_iv']   = _pct_change_frac(atm_t, atm_exp)

                        print(f"DEBUG final call_delta={ctx['context_skew_delta_call_pct']} put_delta={ctx['context_skew_delta_put_pct']}", flush=True)

                    except Exception as ss_err:
                        print(f"DEBUG SS calc failed: {ss_err}", flush=True)
                        ctx['context_skew_delta_call_pct'] = _pct_change_pp(call_t, call_s)
                        ctx['context_skew_delta_put_pct']  = _pct_change_pp(put_t,  put_s)
                        ctx['context_skew_delta_atm_iv']   = _pct_change_frac(atm_t, atm_s)
                else:
                    print(f"DEBUG falling back to OFF-mode", flush=True)
                    ctx['context_skew_delta_call_pct'] = _pct_change_pp(call_t, call_s)
                    ctx['context_skew_delta_put_pct']  = _pct_change_pp(put_t,  put_s)
                    ctx['context_skew_delta_atm_iv']   = _pct_change_frac(atm_t, atm_s)
            else:
                print(f"DEBUG rows not found for {start_hhmm}→{target_hhmm} on {trade_date}", flush=True)

            start_mins = start_dt.hour * 60 + start_dt.minute
            ctx['context_minutes_to_close'] = max(0, 13 * 60 - start_mins)

        except Exception as e:
            print(f"DEBUG outer exception: {e}", flush=True)

    ctx['context_computed_at'] = datetime.now(timezone.utc)

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trade_log_trades SET
                        context_iv_atm_0dte_pct     = %(context_iv_atm_0dte_pct)s,
                        context_target_spx_price    = %(context_target_spx_price)s,
                        context_skew_delta_put_pct  = %(context_skew_delta_put_pct)s,
                        context_skew_delta_call_pct = %(context_skew_delta_call_pct)s,
                        context_skew_delta_atm_iv   = %(context_skew_delta_atm_iv)s,
                        context_minutes_to_close    = %(context_minutes_to_close)s,
                        context_computed_at         = %(context_computed_at)s
                    WHERE id = %(id)s
                """, {**ctx, 'id': trade_id})
    finally:
        conn.close()

    return {'ok': True, 'trade': get_trade(trade_id), 'helpers_available': True}