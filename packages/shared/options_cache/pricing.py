"""
Build the /api/condor-pricing payload: strikes + leg prices + P&L summary
at two timepoints (entry + eval).

Pure pricing logic, separated from the Flask callback so it can be unit
tested under the existing packages/shared/options_cache/tests/ convention.
The callback in apps/web/modules/Ironbeam/callbacks.py is a thin wrapper
(CORS, param parsing, JSON serialization) that delegates here.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .condor import RTH_CLOSE_PT, condor_strikes_from_smile
from .fetcher import fetch_option_bars
from .http_client import OratsError, OratsPermanentError
from .opra import format_opra
from . import repository as repo
from .strikes import StrikeNotListed, snap_to_candidates, listed_strikes
from packages.shared.backtest.quote_validity import leg_quote_is_valid, spread_value_is_valid
from packages.shared.forward_math import compute_spx_strike
from packages.shared.strategy_templates import Leg

# CR-AO decision 4: when the entry-minute quote is invalid, look back this
# far in the cache for the last valid quote (no extra fetch) and flag it stale.
_STALE_LOOKBACK_MIN = 30

# SPX expirations occur on Mon (0), Wed (2), Fri (4)
_SPX_EXPIRY_WEEKDAYS = frozenset({0, 2, 4})

logger = logging.getLogger(__name__)

_PT = ZoneInfo("America/Los_Angeles")

LEG_ROLES = ("short_put", "long_put", "short_call", "long_call")


def _parse_pt_minute(trade_date: date, pt_label: str) -> Optional[datetime]:
    """Parse an HH:MM PT label against trade_date into a naive PT datetime."""
    try:
        hh, mm = pt_label.strip().split(":", 1)
        return datetime(
            trade_date.year, trade_date.month, trade_date.day,
            int(hh), int(mm),
        )
    except (ValueError, AttributeError):
        return None


def _resolve_eval_minute(trade_date: date, eval_pt: str, now_pt: Optional[datetime] = None) -> tuple[Optional[datetime], bool]:
    """
    Translate eval_pt ('HH:MM' or 'now') into a concrete naive PT minute.

    Returns (minute, is_live). is_live is True when eval_pt == 'now' AND we
    landed on the current-session live bar; False otherwise (specific HH:MM
    requested, or 'now' snapped to a prior close).
    """
    if eval_pt is None:
        return (None, False)
    label = str(eval_pt).strip().lower()
    if label != "now":
        return (_parse_pt_minute(trade_date, label), False)

    cur = now_pt or datetime.now(_PT).replace(tzinfo=None)
    close = datetime(trade_date.year, trade_date.month, trade_date.day,
                     RTH_CLOSE_PT[0], RTH_CLOSE_PT[1])
    # Snap "now" back one minute so we only ever query a fully-completed bar.
    snapped = cur.replace(second=0, microsecond=0) - timedelta(minutes=1)
    if cur.date() != trade_date or snapped >= close:
        # Past close for this session — pin to the final minute (close - 1).
        final = close - timedelta(minutes=1)
        return (final, False)
    return (snapped, True)


def _bar_to_quote(bar) -> dict:
    """Project an OptionMinuteBar into the {bid, ask, mid, valid} response shape.

    CR-AO decision 4 (CR-AN's leg rule): a crossed, negative or one-sided
    quote yields mid=None and valid=False so no downstream sum uses it.
    """
    bid = bar.bid_price
    ask = bar.ask_price
    valid = leg_quote_is_valid(bid, ask)
    mid = round((float(bid) + float(ask)) / 2.0, 4) if valid else None
    return {"bid": bid, "ask": ask, "mid": mid, "valid": valid}


def _net_credit(legs: dict) -> Optional[float]:
    """Sum mid-prices: short credits − long debits. None if any leg missing mid."""
    try:
        sp = legs["short_put"]["mid"]
        lp = legs["long_put"]["mid"]
        sc = legs["short_call"]["mid"]
        lc = legs["long_call"]["mid"]
    except (KeyError, TypeError):
        return None
    if any(x is None for x in (sp, lp, sc, lc)):
        return None
    return round(sp + sc - lp - lc, 4)


def _fetch_minute_quotes(opras: dict, minute_pt: datetime, warnings: list) -> dict:
    """
    Ensure cache coverage for the 4 OPRAs at minute_pt, then return
    {role: {bid, ask, mid}} for each. Missing legs get None entries and
    add a warning.
    """
    legs = {}
    for role in LEG_ROLES:
        opra = opras[role]
        try:
            fetch_option_bars(
                [opra], minute_pt, minute_pt,
                record_empty_windows=False,
                source="live_poll",
            )
        except OratsPermanentError as e:
            logger.info("condor-pricing: permanent error fetching %s @ %s: %s",
                        opra, minute_pt, e)
            warnings.append(
                f"no quote data for {opra} at {minute_pt.strftime('%H:%M')}"
            )
            legs[role] = {"bid": None, "ask": None, "mid": None}
            continue
        except OratsError as e:
            logger.warning("condor-pricing: transient error fetching %s @ %s: %s",
                           opra, minute_pt, e)
            warnings.append(
                f"transient error fetching {opra} at {minute_pt.strftime('%H:%M')}"
            )
            legs[role] = {"bid": None, "ask": None, "mid": None}
            continue

        bars = repo.get_bars_for_contract(opra, minute_pt, minute_pt)
        if not bars:
            warnings.append(
                f"no quote data for {opra} at {minute_pt.strftime('%H:%M')}"
            )
            legs[role] = {"bid": None, "ask": None, "mid": None}
        else:
            legs[role] = _bar_to_quote(bars[0])
    return legs


def build_condor_pricing_payload(
    *,
    trade_date: str,
    expiration_date: str,
    spx: float,
    iv_pct: float,
    minutes_to_expiry: float,
    entry_pt: str,
    eval_pt: str,
    wing_width_pts: float = 10.0,
    strike_increment: float = 5.0,
    now_pt: Optional[datetime] = None,
) -> tuple[dict, int]:
    """
    Construct the /api/condor-pricing response payload.

    Returns (payload, http_status). Status is 200 on success (including
    partial failure with warnings), 400 on bad input, 500 on unexpected
    exceptions.

    now_pt is injectable so tests can pin the 'now' translation.
    """
    warnings: list = []

    try:
        td = date.fromisoformat(str(trade_date))
        exp = date.fromisoformat(str(expiration_date))
    except (ValueError, TypeError) as e:
        return ({"error": f"invalid date: {e}"}, 400)

    strikes = condor_strikes_from_smile(
        spx, iv_pct, minutes_to_expiry,
        wing_width_pts=wing_width_pts,
        strike_increment=strike_increment,
    )
    if strikes is None:
        return ({"error": "invalid smile inputs"}, 400)

    opras = {
        "short_put":  format_opra("SPX", exp, "P", strikes["short_put"]),
        "long_put":   format_opra("SPX", exp, "P", strikes["long_put"]),
        "short_call": format_opra("SPX", exp, "C", strikes["short_call"]),
        "long_call":  format_opra("SPX", exp, "C", strikes["long_call"]),
    }

    entry_min = _parse_pt_minute(td, entry_pt)
    if entry_min is None:
        return ({"error": f"invalid entry_pt: {entry_pt!r}"}, 400)

    eval_min, is_live = _resolve_eval_minute(td, eval_pt, now_pt=now_pt)
    if eval_min is None:
        return ({"error": f"invalid eval_pt: {eval_pt!r}"}, 400)

    entry_legs = _fetch_minute_quotes(opras, entry_min, warnings)
    eval_legs = _fetch_minute_quotes(opras, eval_min, warnings)

    net_credit = _net_credit(entry_legs)
    net_cost_to_close = _net_credit(eval_legs)
    gross_pnl: Optional[float] = None
    per_leg: dict[str, Optional[float]] = {}
    if net_credit is not None and net_cost_to_close is not None:
        gross_pnl = round(net_credit - net_cost_to_close, 4)
    for role in LEG_ROLES:
        entry_mid = entry_legs.get(role, {}).get("mid")
        eval_mid = eval_legs.get(role, {}).get("mid")
        if entry_mid is None or eval_mid is None:
            per_leg[role] = None
            continue
        # Short legs profit when prices fall (cost-to-close < entry credit per leg);
        # long legs profit when prices rise. Treat per-leg P&L from the position's
        # POV: sign convention matches the net_credit formula above.
        sign = 1 if role.startswith("short") else -1
        per_leg[role] = round(sign * (entry_mid - eval_mid), 4)

    payload = {
        "sigma_pts": round(strikes["sigma_pts"], 4),
        "strikes": {
            "short_put":  strikes["short_put"],
            "long_put":   strikes["long_put"],
            "short_call": strikes["short_call"],
            "long_call":  strikes["long_call"],
        },
        "opras": opras,
        "entry": {
            "snapshot_pt": entry_min.strftime("%H:%M"),
            "legs": entry_legs,
            "net_credit": net_credit,
        },
        "eval": {
            "snapshot_pt": eval_min.strftime("%H:%M"),
            "is_live": is_live,
            "legs": eval_legs,
            "net_cost_to_close": net_cost_to_close,
        },
        "pnl": {
            "gross": gross_pnl,
            "per_leg": per_leg,
        },
        "warnings": warnings,
    }
    return (payload, 200)


# ── Proposal leg pricing (CR-T Step 1) ───────────────────────────────────────

def price_proposal_legs(
    legs: list[dict],
    *,
    trade_date: date,
    entry_pt: datetime,
    r: float = 0.05,
    q: float = 0.0,
    live: bool = False,
) -> dict:
    """Price proposal legs at entry using real ORATS mids via the options cache.

    Each leg dict must have:
        flag        — 'c' or 'p'
        strike      — strike in ES discounted-forward space
        expiration  — expiration date (date object)
        qty         — quantity
        side        — 'long' or 'short'

    entry_pt must be a **naive PT datetime** (e.g. datetime(2023,7,28,7,0)).

    Returns:
        {
          legs: [{ flag, side, qty, strike_es, spx_strike, opra,
                   bid, ask, mid, expiration }],
          net_debit: float|None,  # positive = debit; negative = credit; None if any leg missing
          warnings: [str],
        }

    This is a reusable, UI-free unit: pure with respect to (date, entry, legs)
    inputs — no HTTP, no Flask.  A future headless backfill can call this in a
    loop without a separate pricing path.
    """
    warnings_out: list[str] = []
    priced_legs: list[dict] = []

    if not legs:
        return {"legs": [], "net_debit": None, "warnings": []}

    # ── 1. ES→SPX conversion + listed-strike snapping + OPRA construction ──
    # CR-AO decisions 1–3: the 5-point rounding from compute_spx_strike is the
    # *intent*; the tradable strike is the nearest one listed for that expiry
    # at the prior close (orats_oi_gamma). A two-leg same-flag vertical is
    # snapped as a pair (anchor = short leg; other leg strictly on its side)
    # so the width is a real listed width; other structures snap per leg.
    raw_meta: list[dict] = []
    for leg in legs:
        expir_d: date = leg["expiration"]
        dte = (expir_d - trade_date).days
        raw_meta.append({
            "flag":       leg["flag"],
            "side":       leg["side"],
            "qty":        leg.get("qty", 1),
            "strike_es":  leg["strike"],
            "spx_strike_raw": compute_spx_strike(leg["strike"], dte, r, q),
            "expiration": expir_d,
        })
    snapped, width_actual, width_nominal = _snap_leg_strikes(raw_meta, trade_date, warnings_out)

    opra_list: list[str] = []
    leg_meta: list[dict] = []
    for meta in snapped:
        flag_upper = meta["flag"].upper()   # 'C' or 'P'
        if meta["spx_strike"] is None:
            leg_meta.append({**meta, "opra": None})
            continue
        opra = format_opra("SPX", meta["expiration"], flag_upper, meta["spx_strike"])
        opra_list.append(opra)
        leg_meta.append({**meta, "opra": opra})

    # ── 2. Batched fetch (writes to cache; idempotent on cache hit) ────────
    unique_opras = list(dict.fromkeys(opra_list))   # dedup, preserve order
    _fetch_kwargs = (
        {"record_empty_windows": False, "source": "live_poll"} if live else {}
    )
    try:
        fetch_option_bars(unique_opras, entry_pt, entry_pt, **_fetch_kwargs)
    except OratsPermanentError as e:
        warnings_out.append(f"permanent error fetching OPRAs at {entry_pt.strftime('%H:%M')}: {e}")
    except OratsError as e:
        warnings_out.append(f"transient error fetching OPRAs at {entry_pt.strftime('%H:%M')}: {e}")

    # ── 3. Read per-leg mids from cache ────────────────────────────────────
    net_debit: Optional[float] = 0.0
    for meta in leg_meta:
        if meta.get("opra") is None:
            # CR-AO: no listed strike for this expiry — the card shows "no listed strike"
            priced_legs.append({**meta, "bid": None, "ask": None, "mid": None, "delta": None})
            net_debit = None
            continue
        bars = repo.get_bars_for_contract(meta["opra"], entry_pt, entry_pt)
        if not bars:
            warnings_out.append(
                f"no quote data for {meta['opra']} at {entry_pt.strftime('%H:%M')} PT"
            )
            priced_legs.append({**meta, "bid": None, "ask": None, "mid": None, "delta": None})
            net_debit = None
            continue

        bar = bars[0]
        bid = bar.bid_price
        ask = bar.ask_price
        stale_quote = False
        quote_minute = entry_pt
        if not leg_quote_is_valid(bid, ask):
            # CR-AO decision 4: never show an invalid quote; use the last valid
            # one in the cache (no extra fetch) and flag it stale.
            prev = repo.get_bars_for_contract(meta["opra"], entry_pt - timedelta(minutes=_STALE_LOOKBACK_MIN), entry_pt)
            valid_prev = [b for b in (prev or []) if b.snapshot_pt < entry_pt and leg_quote_is_valid(b.bid_price, b.ask_price)]
            if valid_prev:
                bar = max(valid_prev, key=lambda b: b.snapshot_pt)
                bid, ask = bar.bid_price, bar.ask_price
                stale_quote = True
                quote_minute = bar.snapshot_pt
                warnings_out.append(
                    f"invalid quote for {meta['opra']} at {entry_pt.strftime('%H:%M')} PT "
                    f"(bid={bid!r} ask={ask!r}); using last valid at {quote_minute.strftime('%H:%M')}"
                )
            else:
                warnings_out.append(
                    f"invalid quote for {meta['opra']} at {entry_pt.strftime('%H:%M')} PT "
                    f"(bid={bid!r} ask={ask!r}); no valid quote in the last {_STALE_LOOKBACK_MIN} min"
                )
                priced_legs.append({**meta, "bid": bid, "ask": ask, "mid": None, "delta": bar.delta,
                                    "quote_valid": False, "stale_quote": False, "quote_minute": None})
                net_debit = None
                continue
        mid = round((float(bid) + float(ask)) / 2.0, 4)

        priced_legs.append({**meta, "bid": bid, "ask": ask, "mid": mid, "delta": bar.delta,
                            "quote_valid": True, "stale_quote": stale_quote,
                            "quote_minute": quote_minute.strftime("%H:%M")})

        if net_debit is not None and mid is not None:
            sign = 1.0 if meta["side"] == "long" else -1.0
            net_debit += sign * meta["qty"] * mid
        else:
            net_debit = None

    spread_valid: Optional[bool] = None
    if net_debit is not None:
        net_debit = round(net_debit, 4)
        # CR-AO decision 4 (CR-AN range rule): a two-leg vertical's price must
        # lie in [0, width_actual] in the structure's direction; otherwise the
        # card must not show it.
        if width_actual is not None and len(priced_legs) == 2:
            leg_objs = [
                Leg(side=l["side"], type=("call" if l["flag"].lower() == "c" else "put"), strike=float(l["spx_strike"]))
                for l in priced_legs
            ]
            spread_valid = spread_value_is_valid(net_debit, width_actual, leg_objs)
            if not spread_valid:
                warnings_out.append(
                    f"spread price {net_debit:+.4f} outside [0, {width_actual:g}] for this structure; suppressed"
                )
                net_debit = None

    return {
        "legs":      priced_legs,
        "net_debit": net_debit,
        "width_actual":  width_actual,
        "width_nominal": width_nominal,
        "stale_quote": any(l.get("stale_quote") for l in priced_legs),
        "spread_valid": spread_valid,
        "warnings":  warnings_out,
    }


def _snap_leg_strikes(raw_meta: list[dict], trade_date: date, warnings_out: list) -> tuple[list[dict], Optional[float], Optional[float]]:
    """CR-AO: snap each leg's spx_strike_raw to the listed grid at the prior close.

    Returns (legs with spx_strike / listed, width_actual, width_nominal). For a
    two-leg same-flag vertical the pair is snapped together (anchor = short
    leg; the long leg on its own side of the anchor). A leg whose expiry is
    absent from the chain gets spx_strike=None and listed=False plus a warning.
    """
    out = [dict(m, spx_strike=m["spx_strike_raw"], listed=True) for m in raw_meta]
    if not raw_meta:
        return out, None, None
    try:
        with repo._conn() as conn:
            chains: dict[date, list[float]] = {}
            for m in raw_meta:
                if m["expiration"] not in chains:
                    _, chains[m["expiration"]] = listed_strikes(conn, m["expiration"], trade_date)
    except Exception as exc:  # chain unavailable → keep the 5-point rounding, say so
        warnings_out.append(f"listed-strike chain unavailable ({exc}); using 5-point rounding")
        return out, None, None

    is_vertical = (
        len(raw_meta) == 2
        and raw_meta[0]["flag"] == raw_meta[1]["flag"]
        and raw_meta[0]["expiration"] == raw_meta[1]["expiration"]
        and {raw_meta[0]["side"], raw_meta[1]["side"]} == {"short", "long"}
    )
    width_nominal = abs(raw_meta[0]["spx_strike_raw"] - raw_meta[1]["spx_strike_raw"]) if len(raw_meta) == 2 else None
    width_actual: Optional[float] = None

    if is_vertical:
        cands = chains.get(raw_meta[0]["expiration"], [])
        si = 0 if raw_meta[0]["side"] == "short" else 1
        li = 1 - si
        if not cands:
            for o in out:
                o["spx_strike"] = None; o["listed"] = False
            warnings_out.append(f"no listed strikes for expiry {raw_meta[0]['expiration']} at the prior close before {trade_date}")
            return out, None, width_nominal
        anchor = snap_to_candidates(raw_meta[si]["spx_strike_raw"], cands)
        direction = raw_meta[li]["spx_strike_raw"] - raw_meta[si]["spx_strike_raw"]
        side = [c for c in cands if (c > anchor if direction > 0 else c < anchor)]
        if not side:
            out[si]["spx_strike"] = anchor
            out[li]["spx_strike"] = None; out[li]["listed"] = False
            warnings_out.append(f"no listed strike on the required side of {anchor:g} for expiry {raw_meta[0]['expiration']}")
            return out, None, width_nominal
        other = snap_to_candidates(raw_meta[li]["spx_strike_raw"], side, toward=anchor)
        out[si]["spx_strike"] = anchor
        out[li]["spx_strike"] = other
        width_actual = abs(other - anchor)
    else:
        for m, o in zip(raw_meta, out):
            cands = chains.get(m["expiration"], [])
            if not cands:
                o["spx_strike"] = None; o["listed"] = False
                warnings_out.append(f"no listed strikes for expiry {m['expiration']} at the prior close before {trade_date}")
                continue
            o["spx_strike"] = snap_to_candidates(m["spx_strike_raw"], cands)
        if len(out) == 2 and all(o["spx_strike"] is not None for o in out):
            width_actual = abs(out[0]["spx_strike"] - out[1]["spx_strike"])

    for m, o in zip(raw_meta, out):
        if o["spx_strike"] is not None and o["spx_strike"] != m["spx_strike_raw"]:
            warnings_out.append(f"strike {m['spx_strike_raw']:g} not listed for {m['expiration']}; snapped to {o['spx_strike']:g}")
    return out, width_actual, width_nominal


# ── Real implied-distribution strike band (CR-T Step 2) ──────────────────────

def build_real_strike_band(
    spot: float,
    implied_move: float,
    *,
    expiration_date: date,
    entry_pt: datetime,
    spacing: float = 5.0,
    half_sigma: float = 1.5,
    live: bool = False,
) -> list[dict]:
    """Fetch a dense band of real ORATS call mids for Breeden-Litzenberger.

    Builds the strike band [spot - half_sigma*IM, spot + half_sigma*IM] at
    `spacing`-point increments, fetches call OPRAs for each strike in one
    batched call, and returns [{"strike": float, "call_price": float}]
    (the input contract for compute_implied_pdf).

    spot is the SPX cash price — NO ES→SPX conversion applied (band strikes
    are already in SPX cash space).

    entry_pt must be a **naive PT datetime**.

    Missing strikes (404, no bar) are omitted from the output — the caller's
    implied_distribution.py sparse-path triggers automatically when < 8 strikes
    or > 25pt spacing survive.

    Returns the list of {strike, call_price} dicts (may be empty on total miss).
    """
    half = half_sigma * implied_move if implied_move > 0 else 50.0
    lo_raw = spot - half
    hi_raw = spot + half

    # Align to the nearest 5-point SPX grid.  spot is already SPX cash —
    # NO ES→SPX conversion here (that conversion only applies to proposal leg
    # strikes, which live in discounted-forward space).
    lo = round(lo_raw / spacing) * spacing
    hi = round(hi_raw / spacing) * spacing

    # Build a call OPRA for each SPX strike in the band
    unique_strikes: list[int] = []
    seen_k: set = set()
    k = lo
    while k <= hi + 1e-9:
        spx_k = round(k / spacing) * spacing   # idempotent; ensures 5pt grid
        if spx_k not in seen_k:
            seen_k.add(spx_k)
            unique_strikes.append(int(spx_k))
        k += spacing

    if not unique_strikes:
        return []

    call_opras = [format_opra("SPX", expiration_date, "C", s) for s in unique_strikes]

    # Batched fetch — writes to cache, idempotent
    _fetch_kwargs = (
        {"record_empty_windows": False, "source": "live_poll"} if live else {}
    )
    try:
        fetch_option_bars(call_opras, entry_pt, entry_pt, **_fetch_kwargs)
    except (OratsPermanentError, OratsError) as e:
        logger.warning("build_real_strike_band: fetch error for %s: %s", entry_pt, e)
        # Continue: return whatever is in cache from prior fetches

    # Read mids
    chain: list[dict] = []
    for spx_strike, opra in zip(unique_strikes, call_opras):
        bars = repo.get_bars_for_contract(opra, entry_pt, entry_pt)
        if not bars:
            continue
        bar = bars[0]
        if not leg_quote_is_valid(bar.bid_price, bar.ask_price):   # CR-AO decision 4
            continue
        mid = (float(bar.bid_price) + float(bar.ask_price)) / 2.0
        if mid > 0:
            chain.append({"strike": float(spx_strike), "call_price": round(mid, 4)})

    return chain


# ── Per-horizon delta fetch (CR-V Step 2) ────────────────────────────────────


def _add_trading_days(d: date, n: int) -> date:
    """Add n trading days (Mon-Fri, no holiday adjustment) to date d."""
    result = d
    added = 0
    while added < n:
        result = result + timedelta(days=1)
        if result.weekday() < 5:
            added += 1
    return result


def nearest_spx_expiration(trade_date: date, target_session_days: int) -> Optional[date]:
    """Find the nearest SPX expiration ≈ target_session_days trading days ahead.

    SPX weeklies expire Mon/Wed/Fri. Steps forward by target_session_days
    business days from trade_date, then returns the first Mon/Wed/Fri at or
    after that target date. Returns None if none found within 30 calendar days.
    """
    target = _add_trading_days(trade_date, target_session_days)
    for i in range(30):
        candidate = target + timedelta(days=i)
        if candidate.weekday() in _SPX_EXPIRY_WEEKDAYS:
            return candidate
    return None


def fetch_horizon_delta(
    magnet_strike_es: float,
    trade_date: date,
    target_session_days: int,
    entry_pt: datetime,
    r: float = 0.05,
    q: float = 0.0,
    live: bool = False,
) -> Optional[float]:
    """Fetch |delta| at the magnet strike for the horizon-appropriate expiration.

    Finds the nearest SPX expiration ≈ target_session_days trading days from
    trade_date, builds the call OPRA for magnet_strike at that expiry (ES→SPX
    via compute_spx_strike), fetches from the options cache (writing through on
    miss), and returns abs(bar.delta).

    entry_pt must be a **naive PT datetime**.
    Returns None on cache miss, fetch error, or no valid expiration found.
    """
    expir = nearest_spx_expiration(trade_date, target_session_days)
    if expir is None:
        logger.warning(
            "fetch_horizon_delta: no SPX expiration found for trade_date=%s target=%dd",
            trade_date, target_session_days,
        )
        return None

    dte = (expir - trade_date).days
    spx_strike = compute_spx_strike(magnet_strike_es, dte, r, q)
    opra = format_opra("SPX", expir, "C", spx_strike)

    _fetch_kwargs = (
        {"record_empty_windows": False, "source": "live_poll"} if live else {}
    )
    try:
        fetch_option_bars([opra], entry_pt, entry_pt, **_fetch_kwargs)
    except (OratsPermanentError, OratsError) as e:
        logger.warning("fetch_horizon_delta: fetch error %s @ %s: %s", opra, entry_pt, e)

    bars = repo.get_bars_for_contract(opra, entry_pt, entry_pt)
    if not bars:
        return None

    return round(abs(float(bars[0].delta)), 4)
