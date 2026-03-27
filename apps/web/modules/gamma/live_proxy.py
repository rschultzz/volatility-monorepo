from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Load .env BEFORE reading any config values.
if load_dotenv is not None:
    for path in (
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ):
        if path.exists():
            load_dotenv(path, override=False)
            break


ORATS_CHAIN_URL = os.getenv(
    "ORATS_STRIKES_CHAIN_URL",
    "https://api.orats.io/datav2/hist/live/one-minute/strikes/chain",
)
ORATS_SUMM_HIST_URL = os.getenv(
    "ORATS_SUMM_HIST_URL",
    "https://api.orats.io/datav2/hist/summaries",
)
ORATS_SUMM_LIVE_URL = os.getenv(
    "ORATS_SUMM_LIVE_URL",
    "https://api.orats.io/datav2/summaries",
)
ORATS_MONIES_HIST_URL = os.getenv(
    "ORATS_MONIES_HIST_URL",
    "https://api.orats.io/datav2/hist/monies/implied",
)
ORATS_MONIES_LIVE_URL = os.getenv(
    "ORATS_MONIES_LIVE_URL",
    "https://api.orats.io/datav2/monies/implied",
)

OPTIONAL_PRICE_COLUMNS = ("spotPrice", "stockPrice")
REQUIRED_COLUMNS = {
    "ticker",
    "expirDate",
    "strike",
    "dte",
    "gamma",
    "callVolume",
    "putVolume",
}
_PT = ZoneInfo("America/Los_Angeles")
_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class LiveProxyConfig:
    ticker: str
    contract_size: float
    level_bucket: float
    proxy_scale: float
    max_dte: float
    min_dte: float
    spot_window_pct: float
    hist_minute_et: str
    price_col_preference: str | None
    expiry_tod_filter: str | None
    request_timeout: int
    current_fallback_minutes: int
    selected_time_fallback_minutes: int


def _cfg() -> LiveProxyConfig:
    return LiveProxyConfig(
        ticker=os.getenv("GEX_TICKER", "SPX").strip().upper(),
        contract_size=float(os.getenv("GEX_LIVE_CONTRACT_SIZE", "100")),
        level_bucket=float(os.getenv("GEX_LIVE_LEVEL_BUCKET", "1")),
        proxy_scale=float(os.getenv("GEX_LIVE_PROXY_SCALE", "1.0")),
        max_dte=float(os.getenv("GEX_LIVE_MAX_DTE", "1")),
        min_dte=float(os.getenv("GEX_LIVE_MIN_DTE", "0")),
        spot_window_pct=float(os.getenv("GEX_LIVE_SPOT_WINDOW_PCT", "0.03")),
        hist_minute_et=os.getenv("GEX_LIVE_HIST_MINUTE_ET", "1559"),
        price_col_preference=(os.getenv("GEX_LIVE_PRICE_COL", "").strip() or None),
        expiry_tod_filter=(os.getenv("GEX_LIVE_EXPIRY_TOD", "").strip().lower() or None),
        request_timeout=int(os.getenv("GEX_LIVE_TIMEOUT", "45")),
        current_fallback_minutes=int(os.getenv("GEX_LIVE_CURRENT_FALLBACK_MINUTES", "3")),
        selected_time_fallback_minutes=int(os.getenv("GEX_LIVE_SELECTED_TIME_FALLBACK_MINUTES", "2")),
    )


def _empty_live_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "level",
            "call_live",
            "put_live",
            "net_live",
            "call_volume_live",
            "put_volume_live",
            "net_volume_live",
            "contracts_touched",
        ]
    ).astype(
        {
            "level": "int64",
            "call_live": "float64",
            "put_live": "float64",
            "net_live": "float64",
            "call_volume_live": "float64",
            "put_volume_live": "float64",
            "net_volume_live": "float64",
            "contracts_touched": "int64",
        }
    )


def _flatten_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                rows.extend(pd.json_normalize(item, sep="_").to_dict(orient="records"))
            else:
                rows.append({"value": item})
        return rows
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.json_normalize(value, sep="_").to_dict(orient="records")
        return pd.json_normalize(payload, sep="_").to_dict(orient="records")
    return [{"value": payload}]


def _response_to_dataframe(resp: requests.Response) -> pd.DataFrame:
    content_type = (resp.headers.get("content-type") or "").lower()
    text_body = resp.text.strip()

    if "json" in content_type or text_body[:1] in "[{":
        try:
            return pd.DataFrame(_flatten_json_records(resp.json()))
        except Exception:
            pass

    if text_body:
        try:
            return pd.read_csv(StringIO(text_body))
        except Exception:
            pass

    raise ValueError("Could not parse ORATS response as JSON or CSV")


def _api_get_json(url: str, params: dict[str, object], timeout: int) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"ORATS HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception as exc:
        raise RuntimeError(f"Could not parse ORATS JSON from {url}: {exc}") from exc


@lru_cache(maxsize=64)
def _fetch_snapshot_df(orats_trade_date: str, ticker: str) -> pd.DataFrame:
    api_key = os.getenv("ORATS_API_KEY")
    cfg = _cfg()
    if not api_key:
        raise RuntimeError("ORATS_API_KEY not found in environment/.env")

    params = {"token": api_key, "ticker": ticker, "tradeDate": orats_trade_date}
    resp = requests.get(ORATS_CHAIN_URL, params=params, timeout=cfg.request_timeout)
    if not resp.ok:
        raise RuntimeError(f"ORATS HTTP {resp.status_code}: {resp.text[:500]}")
    df = _response_to_dataframe(resp)
    if df.empty:
        raise RuntimeError(f"ORATS returned no rows for tradeDate={orats_trade_date}")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise RuntimeError(f"ORATS response missing required columns: {sorted(missing)}")
    return df


@lru_cache(maxsize=128)
def _fetch_monies_map_cached(api_trade_date_iso: str, ticker: str) -> dict[str, tuple[float | None, float | None]]:
    api_key = os.getenv("ORATS_API_KEY")
    cfg = _cfg()
    if not api_key:
        return {}

    out: dict[str, tuple[float | None, float | None]] = {}
    params = {
        "token": api_key,
        "ticker": ticker,
        "tradeDate": api_trade_date_iso,
        "fields": "ticker,tradeDate,expirDate,riskFreeRate,yieldRate",
    }
    for url in (ORATS_MONIES_HIST_URL, ORATS_MONIES_LIVE_URL):
        try:
            payload = _api_get_json(url, params, cfg.request_timeout)
        except Exception:
            continue
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        for row in rows:
            expd = row.get("expirDate")
            if not expd:
                continue
            out[str(expd)] = (row.get("riskFreeRate"), row.get("yieldRate"))
        if out:
            break
    return out


@lru_cache(maxsize=64)
def _fetch_rf30_cached(api_trade_date_iso: str, ticker: str) -> float | None:
    api_key = os.getenv("ORATS_API_KEY")
    cfg = _cfg()
    if not api_key:
        return None

    params = {
        "token": api_key,
        "ticker": ticker,
        "tradeDate": api_trade_date_iso,
        "fields": "ticker,tradeDate,riskFree30",
    }
    for url in (ORATS_SUMM_HIST_URL, ORATS_SUMM_LIVE_URL):
        try:
            payload = _api_get_json(url, params, cfg.request_timeout)
        except Exception:
            continue
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if rows:
            return rows[0].get("riskFree30")
    return None


def _normalize_time_list(selected_times_pt: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if selected_times_pt is None:
        return []
    if isinstance(selected_times_pt, str):
        values = [selected_times_pt]
    else:
        values = list(selected_times_pt)

    out: list[str] = []
    for value in values:
        s = str(value).strip()
        try:
            hh, mm = s.split(":", 1)
            dt.time(int(hh), int(mm))
            out.append(f"{int(hh):02d}:{int(mm):02d}")
        except Exception:
            continue
    return sorted(set(out))


def _selected_pt_to_candidate_trade_dates(trade_date: dt.date, selected_times_pt: str | list[str] | tuple[str, ...] | None) -> tuple[list[str], str | None, str | None]:
    cfg = _cfg()
    normalized = _normalize_time_list(selected_times_pt)
    if not normalized:
        return [], None, None

    chosen_pt = max(normalized)
    try:
        hh, mm = map(int, chosen_pt.split(":"))
        base_pt = dt.datetime.combine(trade_date, dt.time(hh, mm), tzinfo=_PT)
    except Exception:
        return [], None, None

    base_et = base_pt.astimezone(_ET).replace(second=0, microsecond=0)
    candidates = [
        (base_et - dt.timedelta(minutes=back)).strftime("%Y%m%d%H%M")
        for back in range(max(cfg.selected_time_fallback_minutes, 0) + 1)
    ]
    return candidates, chosen_pt, base_et.strftime("%Y%m%d%H%M")


def _current_candidate_trade_dates(trade_date: dt.date) -> list[str]:
    cfg = _cfg()
    now_et = dt.datetime.now(_ET).replace(second=0, microsecond=0)
    today_et = now_et.date()

    if trade_date > today_et:
        return []

    minute = "".join(ch for ch in str(cfg.hist_minute_et) if ch.isdigit())
    if len(minute) != 4:
        minute = "1559"

    if trade_date < today_et:
        return [trade_date.strftime("%Y%m%d") + minute]

    hhmm = int(now_et.strftime("%H%M"))
    if hhmm < 930:
        return []
    if hhmm >= 1600:
        return [trade_date.strftime("%Y%m%d") + minute]

    return [
        (now_et - dt.timedelta(minutes=back)).strftime("%Y%m%d%H%M")
        for back in range(max(cfg.current_fallback_minutes, 0) + 1)
    ]


def _best_price_col(df: pd.DataFrame) -> str:
    cfg = _cfg()
    if cfg.price_col_preference:
        if cfg.price_col_preference not in df.columns:
            raise ValueError(f"Configured GEX_LIVE_PRICE_COL='{cfg.price_col_preference}' not found")
        return cfg.price_col_preference
    for col in OPTIONAL_PRICE_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find an underlier price column. Tried: {', '.join(OPTIONAL_PRICE_COLUMNS)}")


def _parse_iso_date(value: object) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value)[:10]) if value else None
    except Exception:
        return None


def _resolve_rates_for_expiry(
    expir_date_str: str,
    monies_spx: dict[str, tuple[float | None, float | None]],
    monies_spy: dict[str, tuple[float | None, float | None]],
    rf30_spx: float | None,
    rf30_spy: float | None,
) -> tuple[float | None, float | None]:
    sr: float | None = None
    dy: float | None = None

    if expir_date_str in monies_spx:
        sr, dy = monies_spx[expir_date_str]
    if (dy in (None, 0, 0.0)) and expir_date_str in monies_spy:
        sr2, dy2 = monies_spy[expir_date_str]
        if sr is None:
            sr = sr2
        if dy2 not in (None, 0, 0.0):
            dy = dy2
    if sr is None:
        sr = rf30_spx if rf30_spx is not None else rf30_spy
    if dy is None:
        dy = 0.0

    try:
        sr = float(sr) if sr is not None else None
    except Exception:
        sr = None
    try:
        dy = float(dy) if dy is not None else None
    except Exception:
        dy = None
    return sr, dy


def _compute_discounted_level_exact(
    strike: pd.Series,
    eff_dte: pd.Series,
    short_rate: pd.Series,
    div_yield: pd.Series,
) -> pd.Series:
    k = pd.to_numeric(strike, errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(eff_dte, errors="coerce").fillna(0).to_numpy(dtype=float)
    sr = pd.to_numeric(short_rate, errors="coerce").to_numpy(dtype=float)
    dy = pd.to_numeric(div_yield, errors="coerce").to_numpy(dtype=float)

    t = (np.floor(d).astype(int) + 1) / 252.0
    vals = k * np.exp((sr - dy) * t)
    return pd.Series(vals, index=strike.index)


def _bucket_level(level: pd.Series) -> pd.Series:
    cfg = _cfg()
    bucket = cfg.level_bucket if cfg.level_bucket > 0 else 1.0
    vals = np.round(level.to_numpy(dtype=float) / bucket) * bucket
    if abs(bucket - round(bucket)) < 1e-12:
        return pd.Series(vals.round().astype(int), index=level.index)
    return pd.Series(vals, index=level.index)


def fetch_live_proxy_grouped_by_level(
    trade_date: dt.date,
    *,
    ticker: str | None = None,
    selected_times_pt: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cfg = _cfg()
    resolved_ticker = (ticker or cfg.ticker or "SPX").strip().upper()
    selected_candidates, chosen_pt, selected_target_et = _selected_pt_to_candidate_trade_dates(trade_date, selected_times_pt)
    candidate_trade_dates = selected_candidates or _current_candidate_trade_dates(trade_date)
    api_trade_date_iso = trade_date.isoformat()
    meta: dict[str, object] = {
        "ticker": resolved_ticker,
        "requested_trade_date": trade_date.isoformat(),
        "candidate_trade_dates": candidate_trade_dates,
        "api_trade_date": api_trade_date_iso,
        "mode": "selected_time" if selected_candidates else "latest_snapshot",
        "selected_times_pt": _normalize_time_list(selected_times_pt),
        "selected_time_pt": chosen_pt,
        "selected_target_et": selected_target_et,
        "config": {
            "level_bucket": cfg.level_bucket,
            "min_dte": cfg.min_dte,
            "max_dte": cfg.max_dte,
            "spot_window_pct": cfg.spot_window_pct,
            "proxy_scale": cfg.proxy_scale,
        },
    }
    if not candidate_trade_dates:
        meta["error"] = "No intraday snapshot available yet for the selected date/time."
        return _empty_live_df(), meta

    raw: pd.DataFrame | None = None
    last_error: str | None = None
    chosen_trade_date: str | None = None
    for candidate in candidate_trade_dates:
        try:
            maybe = _fetch_snapshot_df(candidate, resolved_ticker).copy()
            if not maybe.empty:
                raw = maybe
                chosen_trade_date = candidate
                break
        except Exception as exc:
            last_error = str(exc)
            continue

    if raw is None:
        meta["error"] = last_error or "No ORATS snapshot returned rows."
        return _empty_live_df(), meta

    meta["orats_trade_date"] = chosen_trade_date
    price_col = _best_price_col(raw)
    meta["price_col"] = price_col

    df = raw.copy()
    for col in ("dte", "strike", "gamma", "callVolume", "putVolume", price_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["ticker"].astype(str).str.upper() == resolved_ticker].copy()
    df = df.dropna(subset=["strike", "gamma", "callVolume", "putVolume", price_col]).copy()
    df["underlier_price"] = df[price_col].astype(float)
    df["gamma_abs"] = df["gamma"].abs()

    if cfg.expiry_tod_filter:
        if "expiryTod" in df.columns:
            df = df[df["expiryTod"].astype(str).str.lower().eq(cfg.expiry_tod_filter)].copy()
        else:
            expir_col = None
            for col in ("expirDate", "expirationDate", "expirdate"):
                if col in df.columns:
                    expir_col = col
                    break
            if expir_col is not None:
                expir_time = df[expir_col].astype(str).str.extract(r"(\d{2}:\d{2}:\d{2})", expand=False)
                df = df[expir_time.fillna("").str.lower().eq(cfg.expiry_tod_filter)].copy()

    expir_col = next((c for c in ("expirDate", "expirationDate", "expirdate") if c in df.columns), None)
    if expir_col is None:
        meta["error"] = "ORATS snapshot missing expirDate column."
        return _empty_live_df(), meta

    df["expir_date_only"] = df[expir_col].map(_parse_iso_date)
    df = df.dropna(subset=["expir_date_only"]).copy()
    df["eff_dte"] = (pd.to_datetime(df["expir_date_only"]).dt.date - trade_date).map(lambda x: x.days)

    if cfg.min_dte > 0:
        df = df[df["eff_dte"] >= float(cfg.min_dte)].copy()
    if cfg.max_dte >= 0:
        df = df[df["eff_dte"] <= float(cfg.max_dte)].copy()

    if cfg.spot_window_pct > 0 and len(df) > 0:
        spot = float(df["underlier_price"].median())
        lo = spot * (1.0 - float(cfg.spot_window_pct))
        hi = spot * (1.0 + float(cfg.spot_window_pct))
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
        meta["spot_filter_lo"] = lo
        meta["spot_filter_hi"] = hi

    if df.empty:
        meta["rows_input"] = int(len(raw))
        meta["rows_used"] = 0
        meta["error"] = "No rows remained after live filters."
        return _empty_live_df(), meta

    monies_spx = _fetch_monies_map_cached(api_trade_date_iso, "SPX")
    monies_spy = _fetch_monies_map_cached(api_trade_date_iso, "SPY")
    rf30_spx = _fetch_rf30_cached(api_trade_date_iso, "SPX")
    rf30_spy = _fetch_rf30_cached(api_trade_date_iso, "SPY")

    rate_rows = df[expir_col].astype(str).map(
        lambda expd: _resolve_rates_for_expiry(expd, monies_spx, monies_spy, rf30_spx, rf30_spy)
    )
    rate_df = pd.DataFrame(rate_rows.tolist(), columns=["short_rate", "div_yield"], index=df.index)
    df["short_rate"] = pd.to_numeric(rate_df["short_rate"], errors="coerce")
    df["div_yield"] = pd.to_numeric(rate_df["div_yield"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["short_rate", "div_yield", "eff_dte"]).copy()
    if df.empty:
        meta["rows_input"] = int(len(raw))
        meta["rows_used"] = 0
        meta["error"] = "No rows had usable rate/dividend inputs after alignment."
        return _empty_live_df(), meta

    df["discounted_level"] = _compute_discounted_level_exact(
        df["strike"],
        df["eff_dte"],
        df["short_rate"],
        df["div_yield"],
    )
    df["level"] = _bucket_level(df["discounted_level"])

    df["call_volume_live"] = df["callVolume"]
    df["put_volume_live"] = -df["putVolume"]
    df["net_volume_live"] = df["call_volume_live"] + df["put_volume_live"]

    df["gamma_unit_proxy"] = df["gamma_abs"] * cfg.contract_size * np.square(df["underlier_price"]) * 0.01 * cfg.proxy_scale
    df["call_live"] = df["callVolume"] * df["gamma_unit_proxy"]
    df["put_live"] = -df["putVolume"] * df["gamma_unit_proxy"]
    df["net_live"] = df["call_live"] + df["put_live"]

    grouped = (
        df.groupby("level", dropna=False, as_index=False)
        .agg(
            call_live=("call_live", "sum"),
            put_live=("put_live", "sum"),
            net_live=("net_live", "sum"),
            call_volume_live=("call_volume_live", "sum"),
            put_volume_live=("put_volume_live", "sum"),
            net_volume_live=("net_volume_live", "sum"),
            contracts_touched=("strike", "count"),
            min_dte_used=("eff_dte", "min"),
            max_dte_used=("eff_dte", "max"),
        )
        .sort_values("level")
        .reset_index(drop=True)
    )

    if grouped.empty:
        meta["error"] = "Live data grouped to zero rows."
        return _empty_live_df(), meta

    if np.issubdtype(grouped["level"].dtype, np.floating) and abs(cfg.level_bucket - round(cfg.level_bucket)) < 1e-12:
        grouped["level"] = grouped["level"].round().astype(int)

    meta["rows_input"] = int(len(raw))
    meta["rows_used"] = int(len(df))
    meta["gross_call_volume"] = float(df["callVolume"].sum())
    meta["gross_put_volume"] = float(df["putVolume"].sum())
    meta["dte_min_used"] = float(df["eff_dte"].min())
    meta["dte_max_used"] = float(df["eff_dte"].max())
    meta["rates_source"] = {
        "spx_monies_expiries": len(monies_spx),
        "spy_monies_expiries": len(monies_spy),
        "rf30_spx": rf30_spx,
        "rf30_spy": rf30_spy,
    }
    meta["discount_preview"] = {
        "strike_min": float(df["strike"].min()),
        "strike_max": float(df["strike"].max()),
        "discounted_min": float(df["discounted_level"].min()),
        "discounted_max": float(df["discounted_level"].max()),
    }
    meta["as_of_label"] = _as_of_label(raw, chosen_trade_date or candidate_trade_dates[0])
    return grouped, meta


def _as_of_label(df: pd.DataFrame, fallback_trade_date: str) -> str:
    for col in ("snapShotEstTime", "snapShotDate", "updatedAt", "quoteDate"):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        value = series.iloc[0]
        if col == "snapShotEstTime":
            try:
                val = int(value)
                hh = val // 100
                mm = val % 100
                return f"as of {hh:02d}:{mm:02d} ET"
            except Exception:
                continue
        return f"as of {value}"
    try:
        d = dt.datetime.strptime(fallback_trade_date, "%Y%m%d%H%M")
        return f"as of {d.strftime('%Y-%m-%d %H:%M')} ET"
    except Exception:
        return f"as of {fallback_trade_date}"
