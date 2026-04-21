#!/usr/bin/env python3
"""
trade_420_assessment.py
=======================

Standalone: assess whether short-vol structures entered at 4/20/26 11:09 ET
(ES ~7123 entering a major long-gamma zone) and closed at 12:20 ET (ES pinned
~7137) would have been profitable.

Pulls two ORATS intraday SPX chain snapshots and computes round-trip P&L for:
    - Short same-day ATM straddle
    - Short next-day ATM straddle
    - Same-day iron condor (defined risk, sells premium)
    - Same-day ATM butterfly (buys theta / gamma around a target)

Uses ORATS mid prices (callValue / putValue). Applies a configurable per-leg
spread cost for a rough execution-cost haircut.

Entry / exit times are assumed GIVEN (no signal logic). This is a post-hoc
question: "if I had sold at 08:09 PT, what happened by 09:20 PT".

Reads .env from repo root. Requires ORATS_API_KEY.
Caches to scripts/.cache/.

Usage:
    python scripts/trade_420_assessment.py
    python scripts/trade_420_assessment.py --entry 1109 --exit 1220
    python scripts/trade_420_assessment.py --per-leg-cost 0.25
"""

from __future__ import annotations

import argparse
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CACHE_DIR  = SCRIPT_DIR / ".cache"
OUTPUT_DIR = REPO_ROOT / "outputs"

# ---------- Config defaults ----------
DATE_STR       = "2026-04-20"
ENTRY_HHMM_ET  = "1109"     # 08:09 PT = 11:09 ET
EXIT_HHMM_ET   = "1220"     # 09:20 PT = 12:20 ET
ES_ENTRY       = 7123.0     # ES at entry (user-observed)
ES_EXIT        = 7137.0     # ES at exit (user-observed pin)
PER_LEG_COST   = 0.20       # $ per leg round-trip (tight for SPX ATM)

# ---------------------------------------------------------------------------
# Env + cache (same pattern as gamma_pin_study)
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
        print(f"[cache] warning: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# ORATS chain fetch
# ---------------------------------------------------------------------------

def fetch_orats_chain(date_str: str, hhmm_et: str) -> pd.DataFrame:
    compact_date = date_str.replace("-", "")
    trade_date_param = f"{compact_date}{hhmm_et}"
    cache_key = f"orats_spx_{trade_date_param}"
    cached = cached_df(cache_key)
    if cached is not None:
        return cached

    import requests

    token = os.environ.get("ORATS_API_KEY")
    if not token:
        raise RuntimeError("ORATS_API_KEY not set")

    url = "https://api.orats.io/datav2/hist/live/one-minute/strikes/chain"
    params = {"token": token, "ticker": "SPX", "tradeDate": trade_date_param}

    print(f"[orats] fetching SPX chain {trade_date_param}")
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"ORATS {resp.status_code}: {resp.text[:500]}")

    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        raise RuntimeError(f"ORATS returned empty chain for {trade_date_param}")

    for col in ("strike", "stockPrice", "dte",
                "callBidPrice", "callValue", "callAskPrice",
                "putBidPrice",  "putValue",  "putAskPrice",
                "callMidIv", "putMidIv", "smvVol",
                "delta", "gamma", "theta", "vega"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expirDate"] = pd.to_datetime(df.get("expirDate"), errors="coerce")

    save_cache(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# Structure pricing helpers
# ---------------------------------------------------------------------------

def find_strike_row(chain: pd.DataFrame, dte: int, strike: float) -> Optional[pd.Series]:
    """Nearest strike at target dte."""
    df = chain[chain["dte"] == dte].copy()
    if df.empty:
        return None
    df["diff"] = (df["strike"] - strike).abs()
    return df.sort_values("diff").iloc[0]

def call_price(row: pd.Series) -> float:
    # Use callValue (ORATS SMV mid) as theoretical mid; fall back to (bid+ask)/2
    v = row.get("callValue")
    if v is not None and not pd.isna(v):
        return float(v)
    return float((row.get("callBidPrice", 0) + row.get("callAskPrice", 0)) / 2.0)

def put_price(row: pd.Series) -> float:
    v = row.get("putValue")
    if v is not None and not pd.isna(v):
        return float(v)
    return float((row.get("putBidPrice", 0) + row.get("putAskPrice", 0)) / 2.0)

def get_spx_spot(chain: pd.DataFrame) -> float:
    return float(chain["stockPrice"].dropna().iloc[0])

def round_to_5(x: float) -> float:
    return round(x / 5.0) * 5.0


# ---------------------------------------------------------------------------
# Price the structures at entry and exit
# ---------------------------------------------------------------------------

def straddle_value(chain: pd.DataFrame, dte: int, strike: float) -> tuple[float, dict]:
    """Return (straddle mid price, detail dict)."""
    c = find_strike_row(chain, dte, strike)
    p = find_strike_row(chain, dte, strike)
    if c is None or p is None:
        return float("nan"), {}
    cp = call_price(c)
    pp = put_price(p)
    return cp + pp, {
        "strike": float(c["strike"]),
        "call": cp,
        "put":  pp,
        "call_iv": float(c.get("callMidIv") or np.nan),
        "put_iv":  float(p.get("putMidIv")  or np.nan),
    }

def iron_condor_value(chain: pd.DataFrame, dte: int, center: float,
                      short_wing: float, long_wing: float) -> tuple[float, dict]:
    """
    Short iron condor: sell closer wings, buy farther wings.
        - sell call at center+short_wing
        - buy  call at center+long_wing
        - sell put  at center-short_wing
        - buy  put  at center-long_wing
    Returns NET CREDIT received when selling. Positive = credit in.
    """
    sc = find_strike_row(chain, dte, center + short_wing)
    lc = find_strike_row(chain, dte, center + long_wing)
    sp = find_strike_row(chain, dte, center - short_wing)
    lp = find_strike_row(chain, dte, center - long_wing)
    if any(x is None for x in (sc, lc, sp, lp)):
        return float("nan"), {}

    credit = (call_price(sc) - call_price(lc)) + (put_price(sp) - put_price(lp))
    return credit, {
        "short_call": (float(sc["strike"]), call_price(sc)),
        "long_call":  (float(lc["strike"]), call_price(lc)),
        "short_put":  (float(sp["strike"]), put_price(sp)),
        "long_put":   (float(lp["strike"]), put_price(lp)),
        "net_credit": credit,
    }

def butterfly_value(chain: pd.DataFrame, dte: int, center: float,
                    wing: float, kind: str = "call") -> tuple[float, dict]:
    """
    Long butterfly: buy 1 at (center-wing), sell 2 at center, buy 1 at (center+wing).
    Returns NET DEBIT paid. Positive = debit out.
    Payoff peaks at center.
    """
    if kind == "call":
        lo = find_strike_row(chain, dte, center - wing)
        md = find_strike_row(chain, dte, center)
        hi = find_strike_row(chain, dte, center + wing)
        if any(x is None for x in (lo, md, hi)):
            return float("nan"), {}
        debit = call_price(lo) - 2 * call_price(md) + call_price(hi)
        return debit, {
            "low":  (float(lo["strike"]), call_price(lo)),
            "mid":  (float(md["strike"]), call_price(md)),
            "high": (float(hi["strike"]), call_price(hi)),
            "net_debit": debit,
        }
    else:
        lo = find_strike_row(chain, dte, center - wing)
        md = find_strike_row(chain, dte, center)
        hi = find_strike_row(chain, dte, center + wing)
        if any(x is None for x in (lo, md, hi)):
            return float("nan"), {}
        debit = put_price(lo) - 2 * put_price(md) + put_price(hi)
        return debit, {
            "low":  (float(lo["strike"]), put_price(lo)),
            "mid":  (float(md["strike"]), put_price(md)),
            "high": (float(hi["strike"]), put_price(hi)),
            "net_debit": debit,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(x, w=8, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—".rjust(w)
    return f"{x:+{w}.{d}f}" if abs(x) < 10000 else f"{x:{w}.{d}f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",  default=DATE_STR)
    parser.add_argument("--entry", default=ENTRY_HHMM_ET, help="HHMM ET")
    parser.add_argument("--exit",  default=EXIT_HHMM_ET,  help="HHMM ET")
    parser.add_argument("--es-entry", type=float, default=ES_ENTRY)
    parser.add_argument("--es-exit",  type=float, default=ES_EXIT)
    parser.add_argument("--per-leg-cost", type=float, default=PER_LEG_COST,
                        help="$ haircut per leg round-trip")
    args = parser.parse_args()

    load_env()
    if not os.environ.get("ORATS_API_KEY"):
        print("ERROR: ORATS_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    entry_chain = fetch_orats_chain(args.date, args.entry)
    exit_chain  = fetch_orats_chain(args.date, args.exit)

    spx_entry = get_spx_spot(entry_chain)
    spx_exit  = get_spx_spot(exit_chain)
    basis_entry = spx_entry - args.es_entry
    basis_exit  = spx_exit  - args.es_exit

    # Time-to-expiration at entry (the 0-dte row where dte==0)
    dtes = sorted(entry_chain["dte"].dropna().unique())
    same_day = 0 if 0 in dtes else min(dtes)
    next_day = next((d for d in dtes if d >= 1), None)

    print(f"\n=== 4/20/26 hindsight trade assessment ===")
    print(f"Entry {args.entry} ET: SPX spot = {spx_entry:.2f}  ES = {args.es_entry:.2f}  basis = {basis_entry:+.2f}")
    print(f"Exit  {args.exit} ET:  SPX spot = {spx_exit:.2f}  ES = {args.es_exit:.2f}  basis = {basis_exit:+.2f}")
    print(f"Net SPX move: {spx_exit - spx_entry:+.2f}   Net ES move: {args.es_exit - args.es_entry:+.2f}")
    print(f"Available expirations (dte): {dtes[:8]}{'...' if len(dtes) > 8 else ''}")
    print(f"Same-day dte = {same_day}   Next-day dte = {next_day}\n")

    # For structures, center strikes in SPX space around the observed pin ES
    # price translated back to SPX. Use the basis at exit (where the pin formed).
    target_spx = args.es_exit + basis_exit   # = spx_exit

    # But if the operator wants to center the structure based on what they
    # *expected* at entry, center on SPX equivalent of the ES entry level, then
    # bias toward the pin side. Here we use the simple approach: center on
    # the SPX-space equivalent of the pin target (exit SPX).
    atm_strike = round_to_5(target_spx)

    print(f"Centering structures at SPX strike {atm_strike:.0f} "
          f"(= ES {atm_strike - basis_exit:.0f})\n")

    results = []

    # --- Structure 1: sold same-day ATM straddle
    if same_day in dtes:
        entry_val, entry_det = straddle_value(entry_chain, same_day, atm_strike)
        exit_val,  exit_det  = straddle_value(exit_chain,  same_day, atm_strike)
        gross = entry_val - exit_val
        costs = 2 * args.per_leg_cost     # 2 legs
        net = gross - costs
        results.append({
            "structure": f"Short straddle (same-day, {same_day}DTE)",
            "entry_val": entry_val, "exit_val": exit_val,
            "gross": gross, "costs": costs, "net": net,
            "entry_det": entry_det, "exit_det": exit_det,
        })

    # --- Structure 2: sold next-day ATM straddle
    if next_day is not None:
        entry_val, entry_det = straddle_value(entry_chain, next_day, atm_strike)
        exit_val,  exit_det  = straddle_value(exit_chain,  next_day, atm_strike)
        gross = entry_val - exit_val
        costs = 2 * args.per_leg_cost
        net = gross - costs
        results.append({
            "structure": f"Short straddle (next-day, {next_day}DTE)",
            "entry_val": entry_val, "exit_val": exit_val,
            "gross": gross, "costs": costs, "net": net,
            "entry_det": entry_det, "exit_det": exit_det,
        })

    # --- Structure 3: sold same-day iron condor (10-wide short, 30-wide long)
    if same_day in dtes:
        entry_val, entry_det = iron_condor_value(entry_chain, same_day, atm_strike, 10, 30)
        exit_val,  exit_det  = iron_condor_value(exit_chain,  same_day, atm_strike, 10, 30)
        # Iron condor: we sold for entry_val credit. To close we pay exit_val.
        gross = entry_val - exit_val
        costs = 4 * args.per_leg_cost     # 4 legs
        net = gross - costs
        results.append({
            "structure": f"Short iron condor (same-day, ±10/±30)",
            "entry_val": entry_val, "exit_val": exit_val,
            "gross": gross, "costs": costs, "net": net,
            "entry_det": entry_det, "exit_det": exit_det,
        })

    # --- Structure 4: long same-day call butterfly at atm_strike (bet on pin)
    if same_day in dtes:
        entry_val, entry_det = butterfly_value(entry_chain, same_day, atm_strike, 15, "call")
        exit_val,  exit_det  = butterfly_value(exit_chain,  same_day, atm_strike, 15, "call")
        # Butterfly: we paid entry_val. To close we receive exit_val.
        gross = exit_val - entry_val
        costs = 4 * args.per_leg_cost
        net = gross - costs
        results.append({
            "structure": f"Long call butterfly (same-day, ±15)",
            "entry_val": entry_val, "exit_val": exit_val,
            "gross": gross, "costs": costs, "net": net,
            "entry_det": entry_det, "exit_det": exit_det,
        })

    # --- Print table ---
    print("=" * 78)
    print(f"{'Structure':<42} {'entry':>8} {'exit':>8} {'gross':>8} {'costs':>6} {'net':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['structure']:<42} "
              f"{fmt(r['entry_val'], 8)} "
              f"{fmt(r['exit_val'], 8)} "
              f"{fmt(r['gross'], 8)} "
              f"{r['costs']:>6.2f} "
              f"{fmt(r['net'], 8)}")
    print("=" * 78)
    print()
    print("Sign convention:")
    print("  Short straddle / iron condor: gross = entry_credit - exit_debit (positive = profit)")
    print("  Long butterfly:               gross = exit_value   - entry_debit (positive = profit)")
    print("  All values are per 1-lot SPX options ($100 multiplier applies to $ P&L)")
    print()
    print("Per-lot $ P&L (multiply net by 100):")
    for r in results:
        if not np.isnan(r["net"]):
            print(f"  {r['structure']:<42}  ${r['net'] * 100:+,.2f}")
    print()

    # Detail dump
    print("-" * 78)
    print("Detail:")
    for r in results:
        print(f"\n  {r['structure']}:")
        print(f"    entry: {r['entry_det']}")
        print(f"    exit:  {r['exit_det']}")

    # Write CSV
    rows_out = []
    for r in results:
        rows_out.append({
            "structure": r["structure"],
            "entry_val": r["entry_val"],
            "exit_val":  r["exit_val"],
            "gross":     r["gross"],
            "costs":     r["costs"],
            "net":       r["net"],
            "net_dollars_per_lot": r["net"] * 100 if not np.isnan(r["net"]) else None,
        })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"trade_{args.date}_assessment.csv"
    pd.DataFrame(rows_out).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
