#!/usr/bin/env python3
"""
gamma_pin_study.py
==================

Standalone hypothesis test. Single day. No DB writes, no Dash.

Question
--------
When ES price arrives at a major SPX GEX level, is realized vol over the next
30 / 60 / 120 minutes systematically different from what SPX ATM IV was pricing
at the moment of arrival?

Hypothesis
----------
    Call-heavy level arrival  -> realized < implied  (sell premium is paid)
    Put-heavy level arrival   -> realized > implied  (buy premium is paid)

What it does
------------
1.  Pull 1-minute ES front-month bars from Databento for the target date (RTH).
2.  Query orats_oi_gamma for per-strike call/put gamma, using discounted_level
    which is ALREADY forward/ES-space thanks to job_orats_eod.py:
        discounted_level = strike * exp((short_rate - div_yield) * (dte+1)/252)
    The ingest also stores rows under the NEXT business day's trade_date,
    so WHERE trade_date='YYYY-MM-DD' gets the prior close's GEX for that day's
    trading session. This is the correct lag for actionable levels.
3.  Cluster nearby strikes into zones; select top-N zones by total gamma.
4.  Walk ES bars (past first 15 min of session), flag arrivals at zones.
    Compare bar price directly to zone center — no basis adjustment needed.
5.  For each arrival: pull ORATS chain AT THAT MINUTE, extract ATM IV
    (next-day expiration, annualized).
6.  Compute realized vol over 30/60/120 min post-arrival.
7.  Compute gap = atm_iv - realized_vol (both annualized).
8.  Print summary, write per-event CSV to outputs/.

Env vars required (reads .env from repo root):
    DATABENTO_API_KEY
    ORATS_API_KEY
    DATABASE_URL

Usage
-----
    python scripts/gamma_pin_study.py
    python scripts/gamma_pin_study.py --date 2026-04-15
    python scripts/gamma_pin_study.py --date 2026-04-17 --dry-run
    python scripts/gamma_pin_study.py --top-n 6 --zone-width 10

Notes
-----
- Level selection uses |call_gamma| + |put_gamma| as the magnitude metric.
  Levels are tagged "call_heavy" or "put_heavy" by which dominates. This is
  observable from the data without resolving dealer long/short convention.
- Zone merging matches production zoneMergeDistancePts pattern: adjacent
  strikes within ZONE_WIDTH are collapsed into one zone at their gamma-
  weighted center (using raw discounted_level, not rounded ints).
- Arrivals are detected starting min_minutes_after_open past RTH open
  to exclude the opening auction from the study.
- ATM IV uses next-day SPX expiration (dte >= 1) to avoid 0DTE weirdness.
- Caches to scripts/.cache/. Re-runs of the same date are free.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import time, timedelta
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATE         = "2026-04-17"

# Level selection from orats_oi_gamma
DEFAULT_TOP_N        = 8          # top-N zones after clustering
DEFAULT_ZONE_WIDTH   = 8.0        # cluster adjacent strikes within N points

# Arrival detection
PROXIMITY_PTS        = 5.0
ARRIVAL_COOLDOWN_MIN = 60

# Measurement horizons (minutes post-arrival)
HORIZONS_MIN         = [30, 60, 120]

# MORNING_CHAIN_HHMM kept for historical cache compatibility (unused now)
MORNING_CHAIN_HHMM   = "1000"

# RTH window in ET
RTH_START_ET         = time(9, 30)
RTH_END_ET           = time(16, 0)

# Annualization
MINUTES_PER_TRADING_YEAR = 252 * 390

# Paths (resolved from script location)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CACHE_DIR  = SCRIPT_DIR / ".cache"
OUTPUT_DIR = REPO_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

def load_env():
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(env_path)
                return env_path
            except ImportError:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)
                return env_path
    return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.parquet"

def cached_df(key: str) -> Optional[pd.DataFrame]:
    p = cache_path(key)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None

def save_cache(key: str, df: pd.DataFrame):
    try:
        df.to_parquet(cache_path(key))
    except Exception as e:
        print(f"[cache] warning: could not save {key}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Databento — ES bars
# ---------------------------------------------------------------------------

def fetch_es_bars(date_str: str, dry_run: bool = False) -> pd.DataFrame:
    import datetime as _dt

    cache_key = f"es_bars_{date_str}"
    cached = cached_df(cache_key)
    if cached is not None:
        return cached
    if dry_run:
        raise RuntimeError(f"No cache for {cache_key} and --dry-run set")

    import databento as db

    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY not set")

    client = db.Historical(api_key)

    # If target date is today, end at "now minus safety margin" — Databento's
    # Historical API trails live by ~15-20 min. Without the margin we get
    # data_end_after_available_end. Also, don't cache partial-day results.
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    target_date = _dt.date.fromisoformat(date_str)
    is_today = (target_date == now_utc.date()) or (target_date == pd.Timestamp.now(tz="America/New_York").date())

    start = f"{date_str}T13:00:00Z"
    if is_today:
        safe_end = now_utc - _dt.timedelta(minutes=25)
        end = safe_end.strftime("%Y-%m-%dT%H:%M:00Z")
    else:
        end = f"{date_str}T21:30:00Z"

    print(f"[databento] fetching ES 1m bars {start} -> {end}")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols="ES.v.0",
        stype_in="continuous",
        schema="ohlcv-1m",
        start=start,
        end=end,
    )
    df = data.to_df()
    if df.empty:
        raise RuntimeError(f"Databento returned no ES bars for {date_str}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = df.reset_index().rename(columns={"ts_event": "ts_utc"})
    if "ts_utc" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "ts_utc"})

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_et"]  = df["ts_utc"].dt.tz_convert("America/New_York")

    keep = ["ts_utc", "ts_et", "open", "high", "low", "close", "volume"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("ts_utc").reset_index(drop=True)

    if not is_today:
        save_cache(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# ORATS chain at a specific minute
# ---------------------------------------------------------------------------

def fetch_orats_chain(date_str: str, hhmm_et: str, dry_run: bool = False) -> pd.DataFrame:
    compact_date = date_str.replace("-", "")
    trade_date_param = f"{compact_date}{hhmm_et}"
    cache_key = f"orats_spx_{trade_date_param}"
    cached = cached_df(cache_key)
    if cached is not None:
        return cached
    if dry_run:
        raise RuntimeError(f"No cache for {cache_key} and --dry-run set")

    import requests
    import datetime as _dt

    token = os.environ.get("ORATS_API_KEY")
    if not token:
        raise RuntimeError("ORATS_API_KEY not set")

    # Endpoint selection:
    #   /historical/... serves data up to YESTERDAY's close
    #   /hist/live/...  serves data up to 1 min ago (requires Live subscription)
    today_et = _dt.datetime.now(_dt.timezone.utc).astimezone().date()  # best effort
    try:
        today_et = pd.Timestamp.now(tz="America/New_York").date()
    except Exception:
        pass
    target_date = _dt.date.fromisoformat(date_str)
    if target_date >= today_et:
        url = "https://api.orats.io/datav2/hist/live/one-minute/strikes/chain"
    else:
        url = "https://api.orats.io/datav2/historical/one-minute/strikes/chain"
    params = {"token": token, "ticker": "SPX", "tradeDate": trade_date_param}

    print(f"[orats] fetching SPX chain {trade_date_param}  ({url.split('/datav2/')[1].split('/')[0]})")
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"ORATS {resp.status_code}: {resp.text[:500]}")

    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"ORATS returned empty chain for {trade_date_param}")

    for col in ("strike", "stockPrice", "gamma", "delta", "callMidIv", "putMidIv",
                "smvVol", "callOpenInterest", "putOpenInterest", "dte"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "expirDate" in df.columns:
        df["expirDate"] = pd.to_datetime(df["expirDate"], errors="coerce")
    if "tradeDate" in df.columns:
        df["tradeDate"] = pd.to_datetime(df["tradeDate"], errors="coerce")

    save_cache(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# orats_oi_gamma — mirrors production callbacks.py query
# ---------------------------------------------------------------------------

def fetch_gex_levels_from_db(date_str: str, ticker: str = "SPX",
                             dry_run: bool = False) -> pd.DataFrame:
    """
    Mirrors apps/web/modules/gamma/callbacks.py query.
    Returns per-integer-strike call_gamma and put_gamma for the day.

    Columns:
        level           int   — rounded discounted_level (SPX strike space)
        call_gamma      float — summed gex_call
        put_gamma       float — summed gex_put  (signed as stored)
        total_abs_gamma float — |call| + |put|  (ranking metric)
    """
    cache_key = f"oi_gamma_{ticker}_{date_str}"
    cached = cached_df(cache_key)
    if cached is not None:
        return cached
    if dry_run:
        raise RuntimeError(f"No cache for {cache_key} and --dry-run set")

    from sqlalchemy import create_engine, text
    from sqlalchemy.engine.url import make_url

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    url = make_url(db_url)
    if url.get_backend_name() == "postgresql" and url.get_driver_name() in (None, "", "psycopg2"):
        url = url.set(drivername="postgresql+psycopg")
    eng = create_engine(url, pool_pre_ping=True)

    sql = text("""
        SELECT
            ROUND(discounted_level)::INT AS level_int,
            AVG(discounted_level)        AS level_precise,
            COALESCE(SUM(gex_call), 0)   AS call_gamma,
            COALESCE(SUM(gex_put),  0)   AS put_gamma
        FROM orats_oi_gamma
        WHERE trade_date = :d
          AND ticker = :tkr
          AND discounted_level IS NOT NULL
        GROUP BY ROUND(discounted_level)::INT
        ORDER BY level_int
    """)
    print(f"[db] querying orats_oi_gamma for {ticker} {date_str}")
    with eng.connect() as con:
        df = pd.read_sql(sql, con, params={"d": date_str, "tkr": ticker})

    if df.empty:
        raise RuntimeError(f"orats_oi_gamma returned no rows for {ticker} {date_str}")

    df["level"]         = df["level_int"].astype(int)
    df["level_precise"] = df["level_precise"].astype(float)
    df["call_gamma"]    = df["call_gamma"].astype(float)
    df["put_gamma"]     = df["put_gamma"].astype(float)
    df["total_abs_gamma"] = df["call_gamma"].abs() + df["put_gamma"].abs()

    save_cache(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# Zone clustering + level selection
# ---------------------------------------------------------------------------

def cluster_and_rank(per_strike: pd.DataFrame, top_n: int,
                     zone_width: float) -> pd.DataFrame:
    """
    Greedy clustering: walk strikes sorted by total_abs_gamma DESC; each
    unclaimed strike becomes a seed, absorbing other unclaimed strikes within
    zone_width. Keep top_n zones by total gamma.

    Zone center uses level_precise (raw discounted_level averaged per bucket)
    for gamma-weighted accuracy, not the rounded integer bucket label.
    """
    if per_strike.empty:
        return pd.DataFrame()

    remaining = per_strike.sort_values("total_abs_gamma", ascending=False).copy().reset_index(drop=True)
    zones = []

    while not remaining.empty and len(zones) < top_n * 3:
        seed = remaining.iloc[0]
        seed_level = int(seed["level"])

        in_zone = (remaining["level"] - seed_level).abs() <= zone_width
        zone_rows = remaining[in_zone].copy()

        # Gamma-weighted center using PRECISE level values (not rounded ints)
        w = zone_rows["total_abs_gamma"].to_numpy()
        l = zone_rows["level_precise"].to_numpy(dtype=float)
        center = float((l * w).sum() / w.sum()) if w.sum() else float(seed["level_precise"])

        call_sum = float(zone_rows["call_gamma"].sum())
        put_sum  = float(zone_rows["put_gamma"].sum())
        total    = abs(call_sum) + abs(put_sum)
        dominant = "call_heavy" if abs(call_sum) >= abs(put_sum) else "put_heavy"

        zones.append({
            "zone_center_strike": center,      # already ES-space, precise
            "zone_low":  int(zone_rows["level"].min()),
            "zone_high": int(zone_rows["level"].max()),
            "call_gamma": call_sum,
            "put_gamma":  put_sum,
            "total_abs_gamma": total,
            "dominant":   dominant,
            "n_strikes":  len(zone_rows),
        })
        remaining = remaining[~in_zone].reset_index(drop=True)

    zdf = pd.DataFrame(zones).sort_values("total_abs_gamma", ascending=False).head(top_n).reset_index(drop=True)
    return zdf


# ---------------------------------------------------------------------------
# (Note: discounted_level in orats_oi_gamma is ALREADY forward/ES-space.
#  The ingest script job_orats_eod.py computes:
#      discounted_level = strike * exp((short_rate - div_yield) * (dte+1)/252)
#  This forward-prices the strike, which is exactly what ES trades at.
#  So no basis adjustment is needed — use discounted_level directly vs ES price.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Arrival detection
# ---------------------------------------------------------------------------

def detect_arrivals(bars: pd.DataFrame, zones: pd.DataFrame,
                    proximity_pts: float, cooldown: timedelta,
                    min_minutes_after_open: int = 15) -> pd.DataFrame:
    """
    Compare ES bar price directly to zone_center_strike (already ES-space
    thanks to discounted_level from orats_oi_gamma).

    An arrival fires when the bar's [low-proximity, high+proximity] range
    contains the level AND we're at least `min_minutes_after_open` past
    the RTH open.
    """
    et = bars["ts_et"]
    rth = bars[(et.dt.time >= RTH_START_ET) & (et.dt.time <= RTH_END_ET)].copy().reset_index(drop=True)
    if rth.empty or zones.empty:
        return pd.DataFrame()

    # Skip first N minutes of the session — opening auction is not an "arrival"
    open_cutoff = time(
        RTH_START_ET.hour + (RTH_START_ET.minute + min_minutes_after_open) // 60,
        (RTH_START_ET.minute + min_minutes_after_open) % 60,
    )
    rth = rth[rth["ts_et"].dt.time >= open_cutoff].reset_index(drop=True)
    if rth.empty:
        return pd.DataFrame()

    last_by_zone: dict[int, pd.Timestamp] = {}
    events = []

    for _, bar in rth.iterrows():
        bar_ts, bar_low, bar_hi, bar_close = bar["ts_utc"], bar["low"], bar["high"], bar["close"]
        for zi, z in zones.iterrows():
            lvl = float(z["zone_center_strike"])   # already ES-space
            if (bar_low - proximity_pts) <= lvl <= (bar_hi + proximity_pts):
                last = last_by_zone.get(int(zi))
                if last is not None and (bar_ts - last) < cooldown:
                    continue
                last_by_zone[int(zi)] = bar_ts
                events.append({
                    "arrival_ts_utc":      bar_ts,
                    "arrival_ts_et":       bar["ts_et"],
                    "zone_idx":            int(zi),
                    "zone_center":         lvl,
                    "zone_low":            int(z["zone_low"]),
                    "zone_high":           int(z["zone_high"]),
                    "dominant":            str(z["dominant"]),
                    "call_gamma":          float(z["call_gamma"]),
                    "put_gamma":           float(z["put_gamma"]),
                    "total_abs_gamma":     float(z["total_abs_gamma"]),
                    "n_strikes":           int(z["n_strikes"]),
                    "bar_close":           float(bar_close),
                    "bar_low":             float(bar_low),
                    "bar_high":            float(bar_hi),
                })
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Vol math
# ---------------------------------------------------------------------------

def realized_vol_window(bars: pd.DataFrame, start_ts: pd.Timestamp,
                        minutes: int) -> Optional[float]:
    end_ts = start_ts + pd.Timedelta(minutes=minutes)
    window = bars[(bars["ts_utc"] >= start_ts) & (bars["ts_utc"] <= end_ts)].copy()
    if len(window) < 5:
        return None
    window = window.sort_values("ts_utc")
    closes = window["close"].astype(float).values
    log_rets = np.diff(np.log(closes))
    if len(log_rets) < 4:
        return None
    sd = np.std(log_rets, ddof=1)
    return float(sd * math.sqrt(MINUTES_PER_TRADING_YEAR))


def atm_iv_from_chain(chain: pd.DataFrame, prefer_next_day: bool = True) -> Optional[dict]:
    if chain.empty:
        return None
    df = chain.dropna(subset=["dte", "strike", "smvVol"]).copy()
    if df.empty:
        return None

    spot = float(chain["stockPrice"].dropna().iloc[0])

    dtes = sorted(df["dte"].unique())
    candidates = [d for d in dtes if d >= (1 if prefer_next_day else 0)]
    if not candidates:
        return None
    use_dte = candidates[0]

    exp_df = df[df["dte"] == use_dte].copy()
    exp_df["strike_diff"] = (exp_df["strike"] - spot).abs()
    atm = exp_df.sort_values("strike_diff").iloc[0]

    return {
        "expir_date": str(pd.to_datetime(atm["expirDate"]).date()) if pd.notna(atm.get("expirDate")) else None,
        "dte":        int(use_dte),
        "atm_strike": float(atm["strike"]),
        "atm_iv":     float(atm["smvVol"]),
        "spot":       spot,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_study(date_str: str, top_n: int, zone_width: float,
              dry_run: bool = False) -> pd.DataFrame:
    print(f"\n=== gamma pin study: {date_str} ===\n")

    bars = fetch_es_bars(date_str, dry_run=dry_run)
    print(f"[bars] {len(bars)} rows, {bars['ts_et'].min()} -> {bars['ts_et'].max()}")

    per_strike = fetch_gex_levels_from_db(date_str, dry_run=dry_run)
    print(f"[db] {len(per_strike)} strikes, range {per_strike['level'].min()}..{per_strike['level'].max()}")

    zones = cluster_and_rank(per_strike, top_n=top_n, zone_width=zone_width)
    if zones.empty:
        print("[zones] no zones found — aborting")
        return pd.DataFrame()

    print(f"[zones] top {len(zones)} by total gamma (values in $B):")
    for _, z in zones.iterrows():
        print(f"    center={z['zone_center_strike']:.1f}  "
              f"({z['zone_low']}-{z['zone_high']}, n={z['n_strikes']})  "
              f"call={z['call_gamma']/1e9:+8.1f}B  "
              f"put={z['put_gamma']/1e9:+8.1f}B  "
              f"total={z['total_abs_gamma']/1e9:7.1f}B  "
              f"[{z['dominant']}]")

    # No basis adjustment: discounted_level is already forward/ES-space.
    arrivals = detect_arrivals(bars, zones, PROXIMITY_PTS,
                               timedelta(minutes=ARRIVAL_COOLDOWN_MIN),
                               min_minutes_after_open=15)
    print(f"\n[arrivals] detected {len(arrivals)} events (first 15 min of session excluded)")
    if arrivals.empty:
        return arrivals

    results = []
    for i, ev in arrivals.iterrows():
        arr_ts = pd.Timestamp(ev["arrival_ts_utc"])
        arr_et = pd.Timestamp(ev["arrival_ts_et"])
        hhmm = arr_et.strftime("%H%M")
        print(f"  [{i+1}/{len(arrivals)}] {arr_et.strftime('%H:%M')} ET  "
              f"zone~{ev['zone_center']:.1f} ({ev['dominant']})  "
              f"ES={ev['bar_close']:.2f}")

        try:
            chain = fetch_orats_chain(date_str, hhmm, dry_run=dry_run)
        except Exception as e:
            print(f"    ! chain fetch failed: {e}")
            continue

        atm = atm_iv_from_chain(chain, prefer_next_day=True)
        if atm is None:
            print("    ! no ATM IV found")
            continue

        row = {
            **ev.to_dict(),
            "spx_spot":    atm["spot"],
            "atm_iv":      atm["atm_iv"],
            "atm_strike":  atm["atm_strike"],
            "iv_dte":      atm["dte"],
            "iv_expir":    atm["expir_date"],
        }
        for h in HORIZONS_MIN:
            rv = realized_vol_window(bars, arr_ts, h)
            gap = (atm["atm_iv"] - rv) if rv is not None else None
            row[f"rv_{h}m"]  = rv
            row[f"iv_{h}m"]  = atm["atm_iv"]
            row[f"gap_{h}m"] = gap
        results.append(row)

    return pd.DataFrame(results)


def summarize(df: pd.DataFrame):
    if df.empty:
        print("\nNo results to summarize.")
        return
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for sign in ("call_heavy", "put_heavy"):
        sub = df[df["dominant"] == sign]
        if sub.empty:
            continue
        print(f"\n{sign.upper()} arrivals: n={len(sub)}")
        print(f"  ATM IV at arrival: mean={sub['atm_iv'].mean():.4f}  "
              f"range={sub['atm_iv'].min():.4f}..{sub['atm_iv'].max():.4f}")
        for h in HORIZONS_MIN:
            rv  = sub[f"rv_{h}m"].dropna()
            gap = sub[f"gap_{h}m"].dropna()
            if rv.empty:
                continue
            hyp_ok = ((sign == "call_heavy" and gap.mean() > 0) or
                      (sign == "put_heavy"  and gap.mean() < 0))
            marker = "[hypothesis MATCH]" if hyp_ok else "[hypothesis MISS]"
            print(f"  {h:>3}min:  RV mean={rv.mean():.4f}   "
                  f"gap mean={gap.mean():+.4f}   "
                  f"med={gap.median():+.4f}   "
                  f"n={len(gap)}   {marker}")
    print("\nKey:")
    print("  call_heavy + gap > 0  =>  IV overpriced vs realized (sell premium pays)")
    print("  put_heavy  + gap < 0  =>  IV underpriced vs realized (buy premium pays)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_DATE, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="top N zones to study")
    parser.add_argument("--zone-width", type=float, default=DEFAULT_ZONE_WIDTH,
                        help="cluster width in SPX points")
    parser.add_argument("--dry-run", action="store_true", help="use cache only")
    args = parser.parse_args()

    env_path = load_env()
    if env_path:
        print(f"[env] loaded {env_path}")
    else:
        print("[env] no .env found — relying on environment")

    for var in ("DATABENTO_API_KEY", "ORATS_API_KEY", "DATABASE_URL"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set", file=sys.stderr)
            sys.exit(1)

    df = run_study(args.date, top_n=args.top_n, zone_width=args.zone_width,
                   dry_run=args.dry_run)
    summarize(df)

    if not df.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"gamma_pin_study_{args.date}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} event rows -> {out_path}")


if __name__ == "__main__":
    main()
