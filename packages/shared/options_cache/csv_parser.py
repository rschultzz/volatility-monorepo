"""
ORATS CSV response parser.

Parses CSV from ORATS' /hist/live/one-minute/strikes/option endpoint.

Per ORATS' docs, this endpoint returns one row per strike, with combined
call+put columns:
    ticker, tradeDate, expirDate, strike, stockPrice, ...
    callBidPrice, callAskPrice, callBidIv, callMidIv, callAskIv,
    callVolume, callOpenInterest, callBidSize, callAskSize,
    callValue, callSmvVol, extCallValue,
    putBidPrice, putAskPrice, putBidIv, putMidIv, putAskIv,
    putVolume, putOpenInterest, putBidSize, putAskSize,
    putValue, putSmvVol, extPutValue,
    smvVol, residualRate, extSmvVol,
    delta, gamma, theta, vega, rho, phi, driftlessTheta, ...

The unprefixed greeks (delta, gamma, theta, vega, rho, phi) are by ORATS'
convention computed for the CALL side. We derive put greeks using
standard relationships:
    put_delta = call_delta - 1
    put_gamma = call_gamma           (identical for European-style)
    put_vega  = call_vega            (identical)
    put_theta = call_theta           (close approximation; exact for
                                      r=0, very close for low rates and
                                      our short-dated use case)
    put_rho   = call_rho - K * T * exp(-r * T)
                — but we don't have r/T cleanly available, so we store
                  None for put_rho and let downstream code derive if needed.
    put_phi   = -call_phi            (sign flip; rough)

For our P&L reconstruction, only delta really matters at the per-bar level.
The other greeks are stored for completeness but aren't used in the
critical path.

Each CSV row is exploded into TWO OptionMinuteBars (one call, one put),
both written to the cache. Idempotent inserts handle the case where the
caller later queries the other side — it's already in the cache.

This is a pure-function module — no network, no DB. Easy to unit test.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .models import ChainFilter, OptionMinuteBar
from .opra import format_opra

_PT = ZoneInfo("America/Los_Angeles")


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case. Matches the cron's helper."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _to_int(v) -> Optional[int]:
    if v is None or pd.isna(v):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _to_float(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (ValueError, TypeError):
        return None


def _to_str(v) -> Optional[str]:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    return s if s else None


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return date.fromisoformat(s[:10])


def parse_orats_csv(
    csv_text: str,
    *,
    expected_ticker: Optional[str] = None,
    expected_strike: Optional[float] = None,
    expected_expir: Optional[date] = None,
    chain_filter: Optional[ChainFilter] = None,
) -> tuple[list[OptionMinuteBar], int, int]:
    """
    Parse an ORATS strikes/option or strikes/chain CSV response into
    OptionMinuteBar objects. Both endpoints return the same chain-shaped
    rows (one row per strike with combined call+put columns); the only
    difference is row count.

    Each kept CSV row produces TWO bars (one call, one put).

    Args:
        csv_text: Raw CSV body from ORATS.
        expected_ticker, expected_strike, expected_expir: Optional sanity
            checks for single-contract responses. If provided, every parsed
            row must match. Mismatches raise ValueError. Don't use these
            with chain endpoint responses (which span many strikes/expirs).
        chain_filter: If provided, rows whose strike/dte fall outside the
            filter are dropped before becoming bars. Used to trim chain
            responses to a useful neighborhood. Pass None to keep everything.

    Returns:
        A 3-tuple: (bars, rows_received, rows_kept).
        - bars: list of OptionMinuteBar (2 per kept row).
        - rows_received: total CSV data rows (pre-filter).
        - rows_kept: rows that passed the filter and produced bars.

    Raises:
        ValueError on malformed CSV, missing required columns, or
        validation failures.
    """
    csv_text = csv_text.strip()
    if not csv_text:
        return [], 0, 0

    if csv_text.lstrip().startswith("<"):
        raise ValueError(
            "Response appears to be HTML/XML, not CSV. Likely an auth error "
            "or invalid endpoint."
        )

    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty:
        return [], 0, 0

    df.columns = [_camel_to_snake(c) for c in df.columns]

    required = {
        "ticker", "trade_date", "expir_date", "strike",
        "snap_shot_date", "quote_date",
        "call_bid_price", "call_ask_price",
        "put_bid_price", "put_ask_price",
        "delta",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {sorted(missing)}. "
            f"Got columns: {sorted(df.columns)}"
        )

    utc_ts = pd.to_datetime(df["snap_shot_date"], errors="coerce", utc=True)
    df["_snapshot_utc"] = utc_ts
    df["_snapshot_pt"] = utc_ts.dt.tz_convert(_PT).dt.tz_localize(None)

    rows_received = len(df)
    bars: list[OptionMinuteBar] = []
    rows_kept = 0

    for _, row in df.iterrows():
        # Apply chain filter if provided. Cheap to evaluate, runs before
        # expensive bar construction.
        if chain_filter is not None:
            strike_v = _to_float(row.get("strike"))
            dte_v = _to_int(row.get("dte"))
            spot_v = _to_float(row.get("stock_price"))
            if strike_v is None or not chain_filter.passes(strike_v, dte_v, spot_v):
                continue

        try:
            call_bar, put_bar = _row_to_bars(row)
        except (ValueError, KeyError):
            continue

        if expected_ticker and call_bar.ticker != expected_ticker:
            raise ValueError(
                f"Row ticker={call_bar.ticker!r} doesn't match "
                f"expected={expected_ticker!r}"
            )
        if expected_strike is not None and call_bar.strike != expected_strike:
            raise ValueError(
                f"Row strike={call_bar.strike} doesn't match "
                f"expected={expected_strike}"
            )
        if expected_expir and call_bar.expir_date_d != expected_expir:
            raise ValueError(
                f"Row expir={call_bar.expir_date_d} doesn't match "
                f"expected={expected_expir}"
            )

        bars.append(call_bar)
        bars.append(put_bar)
        rows_kept += 1

    return bars, rows_received, rows_kept


def _row_to_bars(row) -> tuple[OptionMinuteBar, OptionMinuteBar]:
    """
    Convert one chain-shape row into (call_bar, put_bar).

    Common fields (ticker, expir, strike, timestamps, stockPrice) are
    shared. Side-specific fields (bid/ask/IV/volume/OI) come from the
    appropriately-prefixed columns. Greeks: call uses ORATS values
    as-is; put derives from call.
    """
    # Shared identity fields
    ticker = _to_str(row.get("ticker"))
    expir_str = _to_str(row.get("expir_date"))
    strike = _to_float(row.get("strike"))
    trade_date_str = _to_str(row.get("trade_date"))
    quote_date_str = _to_str(row.get("quote_date"))

    if not all([ticker, expir_str, strike is not None,
                trade_date_str, quote_date_str]):
        raise ValueError("row missing required identity fields")

    expir_date_d = _parse_iso_date(expir_str)
    trade_date_d = _parse_iso_date(trade_date_str)
    if expir_date_d is None or trade_date_d is None:
        raise ValueError("could not parse expir_date or trade_date")

    snapshot_utc = row["_snapshot_utc"]
    snapshot_pt = row["_snapshot_pt"]
    if pd.isna(snapshot_utc) or pd.isna(snapshot_pt):
        raise ValueError("snap_shot_date couldn't be parsed")

    snapshot_utc_py = snapshot_utc.to_pydatetime()
    snapshot_pt_py = snapshot_pt.to_pydatetime()

    # Shared optional fields
    stock_price = _to_float(row.get("stock_price"))
    spot_price = _to_float(row.get("spot_price"))
    dte = _to_int(row.get("dte"))
    smv_vol = _to_float(row.get("smv_vol"))
    residual_rate = _to_float(row.get("residual_rate"))
    ext_smv_vol = _to_float(row.get("ext_smv_vol"))
    expiry_tod = _to_str(row.get("expiry_tod"))
    ticker_id = _to_int(row.get("ticker_id"))
    month_id = _to_int(row.get("month_id"))
    updated_at = _to_str(row.get("updated_at"))
    snap_shot_est_time = _to_int(row.get("snap_shot_est_time"))
    snap_shot_date = _to_str(row.get("snap_shot_date"))

    # Greeks per ORATS convention (call side)
    call_delta = _to_float(row.get("delta"))
    gamma = _to_float(row.get("gamma"))
    vega = _to_float(row.get("vega"))
    theta = _to_float(row.get("theta"))
    rho = _to_float(row.get("rho"))
    phi = _to_float(row.get("phi"))
    driftless_theta = _to_float(row.get("driftless_theta"))

    if call_delta is None:
        raise ValueError("row missing delta (required)")

    # Per-side bid/ask prices
    call_bid = _to_float(row.get("call_bid_price"))
    call_ask = _to_float(row.get("call_ask_price"))
    put_bid = _to_float(row.get("put_bid_price"))
    put_ask = _to_float(row.get("put_ask_price"))

    if any(x is None for x in (call_bid, call_ask, put_bid, put_ask)):
        raise ValueError("row missing required bid/ask prices")

    # Per-side IV
    call_bid_iv = _to_float(row.get("call_bid_iv"))
    call_mid_iv = _to_float(row.get("call_mid_iv"))
    call_ask_iv = _to_float(row.get("call_ask_iv"))
    put_bid_iv = _to_float(row.get("put_bid_iv"))
    put_mid_iv = _to_float(row.get("put_mid_iv"))
    put_ask_iv = _to_float(row.get("put_ask_iv"))

    # Per-side volume / OI / sizes
    call_volume = _to_int(row.get("call_volume"))
    call_open_interest = _to_int(row.get("call_open_interest"))
    call_bid_size = _to_int(row.get("call_bid_size"))
    call_ask_size = _to_int(row.get("call_ask_size"))
    put_volume = _to_int(row.get("put_volume"))
    put_open_interest = _to_int(row.get("put_open_interest"))
    put_bid_size = _to_int(row.get("put_bid_size"))
    put_ask_size = _to_int(row.get("put_ask_size"))

    # Per-side ORATS smoothed values
    call_value = _to_float(row.get("call_value"))
    put_value = _to_float(row.get("put_value"))
    call_smv_vol = _to_float(row.get("call_smv_vol"))
    put_smv_vol = _to_float(row.get("put_smv_vol"))
    ext_call_value = _to_float(row.get("ext_call_value"))
    ext_put_value = _to_float(row.get("ext_put_value"))

    # OPRA symbols for both sides
    call_opra = format_opra(ticker, expir_date_d, "C", strike)
    put_opra = format_opra(ticker, expir_date_d, "P", strike)

    # Common kwargs for both bars
    common = dict(
        ticker=ticker,
        expir_date=expir_str,
        expir_date_d=expir_date_d,
        strike=strike,
        trade_date=trade_date_str,
        trade_date_d=trade_date_d,
        quote_date=quote_date_str,
        snapshot_pt=snapshot_pt_py,
        snapshot_utc=snapshot_utc_py,
        updated_at=updated_at,
        snap_shot_est_time=snap_shot_est_time,
        snap_shot_date=snap_shot_date,
        stock_price=stock_price,
        spot_price=spot_price,
        dte=dte,
        smv_vol=smv_vol,
        residual_rate=residual_rate,
        ext_smv_vol=ext_smv_vol,
        gamma=gamma,
        vega=vega,
        theta=theta,
        driftless_theta=driftless_theta,
        expiry_tod=expiry_tod,
        ticker_id=ticker_id,
        month_id=month_id,
    )

    call_bar = OptionMinuteBar(
        opra_symbol=call_opra,
        option_type="C",
        bid_price=call_bid,
        ask_price=call_ask,
        bid_size=call_bid_size,
        ask_size=call_ask_size,
        bid_iv=call_bid_iv,
        mid_iv=call_mid_iv,
        ask_iv=call_ask_iv,
        volume=call_volume,
        open_interest=call_open_interest,
        opt_value=call_value,
        ext_val=ext_call_value,
        delta=call_delta,
        rho=rho,
        phi=phi,
        **common,
    )

    # Put greeks derived from call greeks
    put_delta = call_delta - 1.0
    # Sign-flip for rho/phi is a rough approximation. Storing None is more
    # honest than storing a wrong-sign value.
    put_rho = None
    put_phi = -phi if phi is not None else None

    put_bar = OptionMinuteBar(
        opra_symbol=put_opra,
        option_type="P",
        bid_price=put_bid,
        ask_price=put_ask,
        bid_size=put_bid_size,
        ask_size=put_ask_size,
        bid_iv=put_bid_iv,
        mid_iv=put_mid_iv,
        ask_iv=put_ask_iv,
        volume=put_volume,
        open_interest=put_open_interest,
        opt_value=put_value,
        ext_val=ext_put_value,
        delta=put_delta,
        rho=put_rho,
        phi=put_phi,
        **common,
    )

    return call_bar, put_bar
