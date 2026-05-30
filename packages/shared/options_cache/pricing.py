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
from packages.shared.forward_math import compute_spx_strike

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
    """Project an OptionMinuteBar into the {bid, ask, mid} response shape."""
    bid = bar.bid_price
    ask = bar.ask_price
    mid = None
    if bid is not None and ask is not None:
        mid = round((float(bid) + float(ask)) / 2.0, 4)
    return {"bid": bid, "ask": ask, "mid": mid}


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
            fetch_option_bars([opra], minute_pt, minute_pt)
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

    # ── 1. ES→SPX conversion + OPRA construction ──────────────────────────
    opra_list: list[str] = []
    leg_meta: list[dict] = []
    for leg in legs:
        expir_d: date = leg["expiration"]
        dte = (expir_d - trade_date).days
        spx_strike = compute_spx_strike(leg["strike"], dte, r, q)
        flag_upper = leg["flag"].upper()   # 'C' or 'P'
        opra = format_opra("SPX", expir_d, flag_upper, spx_strike)
        opra_list.append(opra)
        leg_meta.append({
            "flag":       leg["flag"],
            "side":       leg["side"],
            "qty":        leg.get("qty", 1),
            "strike_es":  leg["strike"],
            "spx_strike": spx_strike,
            "opra":       opra,
            "expiration": expir_d,
        })

    # ── 2. Batched fetch (writes to cache; idempotent on cache hit) ────────
    unique_opras = list(dict.fromkeys(opra_list))   # dedup, preserve order
    try:
        fetch_option_bars(unique_opras, entry_pt, entry_pt)
    except OratsPermanentError as e:
        warnings_out.append(f"permanent error fetching OPRAs at {entry_pt.strftime('%H:%M')}: {e}")
    except OratsError as e:
        warnings_out.append(f"transient error fetching OPRAs at {entry_pt.strftime('%H:%M')}: {e}")

    # ── 3. Read per-leg mids from cache ────────────────────────────────────
    net_debit: Optional[float] = 0.0
    for meta in leg_meta:
        bars = repo.get_bars_for_contract(meta["opra"], entry_pt, entry_pt)
        if not bars:
            warnings_out.append(
                f"no quote data for {meta['opra']} at {entry_pt.strftime('%H:%M')} PT"
            )
            priced_legs.append({**meta, "bid": None, "ask": None, "mid": None})
            net_debit = None
            continue

        bar = bars[0]
        bid = bar.bid_price
        ask = bar.ask_price
        mid = round((float(bid) + float(ask)) / 2.0, 4) if (bid is not None and ask is not None) else None

        priced_legs.append({**meta, "bid": bid, "ask": ask, "mid": mid})

        if net_debit is not None and mid is not None:
            sign = 1.0 if meta["side"] == "long" else -1.0
            net_debit += sign * meta["qty"] * mid
        else:
            net_debit = None

    if net_debit is not None:
        net_debit = round(net_debit, 4)

    return {
        "legs":      priced_legs,
        "net_debit": net_debit,
        "warnings":  warnings_out,
    }


# ── Real implied-distribution strike band (CR-T Step 2) ──────────────────────

def build_real_strike_band(
    spot: float,
    implied_move: float,
    *,
    expiration_date: date,
    entry_pt: datetime,
    spacing: float = 5.0,
    half_sigma: float = 1.5,
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
    try:
        fetch_option_bars(call_opras, entry_pt, entry_pt)
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
        if bar.bid_price is None or bar.ask_price is None:
            continue
        mid = (float(bar.bid_price) + float(bar.ask_price)) / 2.0
        if mid > 0:
            chain.append({"strike": float(spx_strike), "call_price": round(mid, 4)})

    return chain
