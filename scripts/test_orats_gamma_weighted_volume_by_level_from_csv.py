#!/usr/bin/env python3
from __future__ import annotations

"""
Build a cumulative gamma-weighted volume proxy by discounted level from a single
ORATS strikes-chain CSV snapshot.

Assumption used for sign:
- call volume -> positive positioning contribution
- put volume  -> negative positioning contribution

This is a *proxy*, not true dealer positioning.

First-pass discounted level formula used here:
    discounted_level = strike * exp(-residualRate * T)
where T = dte / 365.

This version uses one cumulative snapshot only, so it effectively applies the
snapshot gamma to the full day-to-date volume. That is a fast first-pass proxy.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "ticker",
    "expirDate",
    "strike",
    "dte",
    "residualRate",
    "gamma",
    "callVolume",
    "putVolume",
}

OPTIONAL_PRICE_COLUMNS = ("stockPrice", "spotPrice")
DEFAULT_CONTRACT_SIZE = 100.0
DEFAULT_BUCKET = 1.0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def _best_price_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested price column '{requested}' not found in CSV")
        return requested
    for col in OPTIONAL_PRICE_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a usable underlier price column. "
        f"Tried: {', '.join(OPTIONAL_PRICE_COLUMNS)}"
    )


def _normalize_types(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    out = df.copy()
    numeric_cols: Iterable[str] = [
        "strike",
        "dte",
        "residualRate",
        "gamma",
        "callVolume",
        "putVolume",
        price_col,
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ticker"] = out["ticker"].astype(str)
    out["expirDate"] = out["expirDate"].astype(str)

    key_cols = ["ticker", "expirDate", "strike"]
    out = out.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")
    return out


def _discounted_level(strike: pd.Series, residual_rate: pd.Series, dte: pd.Series, mode: str) -> pd.Series:
    t = np.maximum(pd.to_numeric(dte, errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0) / 365.0
    r = pd.to_numeric(residual_rate, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    k = pd.to_numeric(strike, errors="coerce").to_numpy(dtype=float)

    if mode == "spot":
        vals = k * np.exp(-r * t)
    elif mode == "forward":
        vals = k * np.exp(r * t)
    elif mode == "none":
        vals = k
    else:
        raise ValueError(f"Unknown discount mode: {mode}")
    return pd.Series(vals, index=strike.index)


def _bucket_level(level: pd.Series, bucket: float) -> pd.Series:
    if bucket <= 0:
        raise ValueError("bucket must be > 0")
    vals = np.round(level.to_numpy(dtype=float) / bucket) * bucket
    if abs(bucket - round(bucket)) < 1e-12:
        return pd.Series(vals.round().astype(int), index=level.index)
    return pd.Series(vals, index=level.index)


def build_gamma_weighted_volume_proxy(
    df: pd.DataFrame,
    *,
    price_col: str,
    discount_mode: str,
    bucket: float,
    contract_size: float,
    min_dte: float | None,
    max_dte: float | None,
    spot_window_pct: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    base = _normalize_types(df, price_col)

    # Basic cleanup for cumulative snapshot usage.
    base["callVolume"] = base["callVolume"].clip(lower=0.0)
    base["putVolume"] = base["putVolume"].clip(lower=0.0)
    base["gamma_abs"] = pd.to_numeric(base["gamma"], errors="coerce").abs().fillna(0.0)
    base["underlier_price"] = pd.to_numeric(base[price_col], errors="coerce").fillna(0.0)

    if min_dte is not None:
        base = base[base["dte"] >= float(min_dte)].copy()
    if max_dte is not None:
        base = base[base["dte"] <= float(max_dte)].copy()

    if spot_window_pct is not None and len(base) > 0:
        spot = float(base["underlier_price"].median())
        lo = spot * (1.0 - float(spot_window_pct))
        hi = spot * (1.0 + float(spot_window_pct))
        base = base[(base["strike"] >= lo) & (base["strike"] <= hi)].copy()

    base["discounted_level"] = _discounted_level(
        base["strike"], base["residualRate"], base["dte"], discount_mode
    )
    base["level"] = _bucket_level(base["discounted_level"], bucket)

    base["gamma_unit_proxy"] = (
        base["gamma_abs"] * contract_size * np.square(base["underlier_price"]) * 0.01
    )

    base["signed_volume"] = base["callVolume"] - base["putVolume"]
    base["call_gamma_proxy"] = base["callVolume"] * base["gamma_unit_proxy"]
    base["put_gamma_proxy"] = -base["putVolume"] * base["gamma_unit_proxy"]
    base["net_gamma_proxy"] = base["call_gamma_proxy"] + base["put_gamma_proxy"]

    grouped = (
        base.groupby("level", dropna=False, as_index=False)
        .agg(
            call_volume=("callVolume", "sum"),
            put_volume=("putVolume", "sum"),
            signed_volume=("signed_volume", "sum"),
            call_gamma_proxy=("call_gamma_proxy", "sum"),
            put_gamma_proxy=("put_gamma_proxy", "sum"),
            net_gamma_proxy=("net_gamma_proxy", "sum"),
            contracts_touched=("strike", "count"),
        )
        .sort_values("level")
        .reset_index(drop=True)
    )

    summary = {
        "rows_input": float(len(df)),
        "rows_used": float(len(base)),
        "gross_call_volume": float(base["callVolume"].sum()),
        "gross_put_volume": float(base["putVolume"].sum()),
        "gross_abs_net_gamma_proxy": float(base["net_gamma_proxy"].abs().sum()),
    }

    detail_cols = [
        "ticker",
        "expirDate",
        "strike",
        "dte",
        "residualRate",
        "underlier_price",
        "discounted_level",
        "level",
        "gamma",
        "gamma_abs",
        "callVolume",
        "putVolume",
        "signed_volume",
        "gamma_unit_proxy",
        "call_gamma_proxy",
        "put_gamma_proxy",
        "net_gamma_proxy",
    ]
    detail = base[detail_cols].copy().sort_values(["level", "expirDate", "strike"])
    return grouped, detail, summary


def _default_output_path(input_csv: Path) -> Path:
    return Path("outputs") / f"{input_csv.stem}_gamma_weighted_volume_by_level.csv"


def _default_detail_output_path(input_csv: Path) -> Path:
    return Path("outputs") / f"{input_csv.stem}_gamma_weighted_volume_detail.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cumulative gamma-weighted volume proxy by discounted level from one ORATS chain CSV snapshot."
    )
    parser.add_argument("--input-csv", required=True, help="ORATS chain snapshot CSV")
    parser.add_argument("--output-csv", default=None, help="Grouped output CSV path")
    parser.add_argument("--detail-csv", default=None, help="Detailed output CSV path")
    parser.add_argument(
        "--price-col",
        default=None,
        help="Underlier price column to use (default: auto-detect stockPrice/spotPrice)",
    )
    parser.add_argument(
        "--discount-mode",
        choices=["spot", "forward", "none"],
        default="spot",
        help="How to convert strike into discounted level (default: spot)",
    )
    parser.add_argument(
        "--level-bucket",
        type=float,
        default=DEFAULT_BUCKET,
        help="Bucket size for discounted levels (default: 1)",
    )
    parser.add_argument(
        "--contract-size",
        type=float,
        default=DEFAULT_CONTRACT_SIZE,
        help="Contract multiplier for gamma proxy units (default: 100)",
    )
    parser.add_argument("--min-dte", type=float, default=None, help="Optional minimum DTE filter")
    parser.add_argument("--max-dte", type=float, default=None, help="Optional maximum DTE filter")
    parser.add_argument(
        "--spot-window-pct",
        type=float,
        default=None,
        help="Optional strike filter around spot, e.g. 0.05 for +/-5%%",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv) if args.output_csv else _default_output_path(input_csv)
    detail_csv = Path(args.detail_csv) if args.detail_csv else _default_detail_output_path(input_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_csv.parent.mkdir(parents=True, exist_ok=True)

    df = _read_csv(input_csv)
    price_col = _best_price_col(df, args.price_col)

    grouped, detail, summary = build_gamma_weighted_volume_proxy(
        df,
        price_col=price_col,
        discount_mode=args.discount_mode,
        bucket=args.level_bucket,
        contract_size=args.contract_size,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        spot_window_pct=args.spot_window_pct,
    )

    grouped.to_csv(output_csv, index=False)
    detail.to_csv(detail_csv, index=False)

    print(f"Read {len(df):,} rows from {input_csv}")
    print(f"Using price column: {price_col}")
    for k, v in summary.items():
        print(f"{k}: {v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{k}: {v:,.4f}")
    print(f"\nWrote grouped output to: {output_csv}")
    print(f"Wrote detail output to:  {detail_csv}")

    if grouped.empty:
        print("\nNo grouped rows after filtering.")
        return

    pos = grouped.sort_values("net_gamma_proxy", ascending=False).head(15)
    neg = grouped.sort_values("net_gamma_proxy", ascending=True).head(15)

    print("\n=== Top 15 positive net_gamma_proxy levels ===")
    print(pos.to_string(index=False))
    print("\n=== Top 15 negative net_gamma_proxy levels ===")
    print(neg.to_string(index=False))


if __name__ == "__main__":
    main()
