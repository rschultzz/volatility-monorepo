#!/usr/bin/env python3
from __future__ import annotations

"""
Compare two ORATS live strikes-chain CSV snapshots and build an intraday
signed gamma-flow proxy by discounted level.

Assumption used for sign:
- call volume increment -> positive positioning contribution
- put volume increment  -> negative positioning contribution

This is a *proxy*, not true dealer positioning.

First-pass discounted level formula used here:
    discounted_level = strike * exp(-residualRate * T)
where T = dte / 365.

If your existing GEX pipeline uses a slightly different discounted-level formula,
we can swap that one line later without changing the rest of the script.
"""

import argparse
import math
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
    # Keep integers as integers when bucket is whole-number-like.
    if abs(bucket - round(bucket)) < 1e-12:
        return pd.Series(vals.round().astype(int), index=level.index)
    return pd.Series(vals, index=level.index)


def build_signed_gamma_proxy(
    prev_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    *,
    price_col: str,
    discount_mode: str,
    bucket: float,
    contract_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    prev = _normalize_types(prev_df, price_col)
    curr = _normalize_types(curr_df, price_col)

    key_cols = ["ticker", "expirDate", "strike"]
    merged = curr.merge(
        prev[key_cols + ["callVolume", "putVolume"]],
        on=key_cols,
        how="left",
        suffixes=("", "_prev"),
    )

    merged["callVolume_prev"] = merged["callVolume_prev"].fillna(0.0)
    merged["putVolume_prev"] = merged["putVolume_prev"].fillna(0.0)

    merged["callVolumeDelta_raw"] = merged["callVolume"] - merged["callVolume_prev"]
    merged["putVolumeDelta_raw"] = merged["putVolume"] - merged["putVolume_prev"]

    neg_call_count = int((merged["callVolumeDelta_raw"] < 0).sum())
    neg_put_count = int((merged["putVolumeDelta_raw"] < 0).sum())

    merged["callVolumeDelta"] = merged["callVolumeDelta_raw"].clip(lower=0.0)
    merged["putVolumeDelta"] = merged["putVolumeDelta_raw"].clip(lower=0.0)

    merged["discounted_level"] = _discounted_level(
        merged["strike"], merged["residualRate"], merged["dte"], discount_mode
    )
    merged["level"] = _bucket_level(merged["discounted_level"], bucket)

    merged["underlier_price"] = pd.to_numeric(merged[price_col], errors="coerce").fillna(0.0)
    merged["gamma_abs"] = pd.to_numeric(merged["gamma"], errors="coerce").abs().fillna(0.0)

    # GEX-like unit proxy per 1 contract.
    merged["gamma_unit_proxy"] = (
        merged["gamma_abs"] * contract_size * np.square(merged["underlier_price"]) * 0.01
    )

    merged["call_gamma_proxy"] = merged["callVolumeDelta"] * merged["gamma_unit_proxy"]
    merged["put_gamma_proxy"] = -merged["putVolumeDelta"] * merged["gamma_unit_proxy"]
    merged["net_gamma_proxy"] = merged["call_gamma_proxy"] + merged["put_gamma_proxy"]

    merged["net_volume_signed"] = merged["callVolumeDelta"] - merged["putVolumeDelta"]

    grouped = (
        merged.groupby("level", dropna=False, as_index=False)
        .agg(
            call_volume=("callVolumeDelta", "sum"),
            put_volume=("putVolumeDelta", "sum"),
            signed_volume=("net_volume_signed", "sum"),
            call_gamma_proxy=("call_gamma_proxy", "sum"),
            put_gamma_proxy=("put_gamma_proxy", "sum"),
            net_gamma_proxy=("net_gamma_proxy", "sum"),
            contracts_touched=("strike", "count"),
        )
        .sort_values("level")
        .reset_index(drop=True)
    )

    summary = {
        "rows_curr": float(len(curr)),
        "rows_prev": float(len(prev)),
        "rows_merged": float(len(merged)),
        "negative_call_deltas_clipped": float(neg_call_count),
        "negative_put_deltas_clipped": float(neg_put_count),
        "gross_call_volume_delta": float(merged["callVolumeDelta"].sum()),
        "gross_put_volume_delta": float(merged["putVolumeDelta"].sum()),
        "gross_abs_net_gamma_proxy": float(merged["net_gamma_proxy"].abs().sum()),
    }

    detail_cols = [
        "ticker",
        "expirDate",
        "strike",
        "level",
        "discounted_level",
        "dte",
        "residualRate",
        "underlier_price",
        "gamma",
        "gamma_abs",
        "callVolume_prev",
        "callVolume",
        "callVolumeDelta_raw",
        "callVolumeDelta",
        "putVolume_prev",
        "putVolume",
        "putVolumeDelta_raw",
        "putVolumeDelta",
        "gamma_unit_proxy",
        "call_gamma_proxy",
        "put_gamma_proxy",
        "net_gamma_proxy",
    ]
    detail = merged[detail_cols].copy().sort_values(["level", "expirDate", "strike"])

    return grouped, detail, summary


def _default_output_path(curr_csv: Path) -> Path:
    stem = curr_csv.stem
    return Path("outputs") / f"{stem}_signed_gamma_proxy_by_level.csv"


def _default_detail_output_path(curr_csv: Path) -> Path:
    stem = curr_csv.stem
    return Path("outputs") / f"{stem}_signed_gamma_proxy_detail.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a signed intraday gamma proxy by discounted level from two ORATS chain CSV snapshots."
    )
    parser.add_argument("--prev-csv", required=True, help="Earlier ORATS chain snapshot CSV")
    parser.add_argument("--curr-csv", required=True, help="Later ORATS chain snapshot CSV")
    parser.add_argument(
        "--discount-mode",
        choices=["spot", "forward", "none"],
        default="spot",
        help="How to convert strike into discounted_level. Default: spot",
    )
    parser.add_argument(
        "--price-col",
        default=None,
        help="Underlier price column to use in gamma scaling. Default: auto-pick stockPrice then spotPrice",
    )
    parser.add_argument(
        "--level-bucket",
        type=float,
        default=DEFAULT_BUCKET,
        help="Bucket size for rounded discounted levels. Default: 1",
    )
    parser.add_argument(
        "--contract-size",
        type=float,
        default=DEFAULT_CONTRACT_SIZE,
        help="Contract multiplier. Default: 100",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Aggregated output CSV path. Default: outputs/<curr_stem>_signed_gamma_proxy_by_level.csv",
    )
    parser.add_argument(
        "--detail-output",
        default=None,
        help="Per-contract detail output CSV path. Default: outputs/<curr_stem>_signed_gamma_proxy_detail.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many largest net levels to print. Default: 15",
    )
    args = parser.parse_args()

    prev_csv = Path(args.prev_csv)
    curr_csv = Path(args.curr_csv)

    prev_df = _read_csv(prev_csv)
    curr_df = _read_csv(curr_csv)

    price_col = _best_price_col(curr_df, args.price_col)

    grouped, detail, summary = build_signed_gamma_proxy(
        prev_df,
        curr_df,
        price_col=price_col,
        discount_mode=args.discount_mode,
        bucket=float(args.level_bucket),
        contract_size=float(args.contract_size),
    )

    output = Path(args.output) if args.output else _default_output_path(curr_csv)
    detail_output = Path(args.detail_output) if args.detail_output else _default_detail_output_path(curr_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail_output.parent.mkdir(parents=True, exist_ok=True)

    grouped.to_csv(output, index=False)
    detail.to_csv(detail_output, index=False)

    print("\n=== Signed gamma proxy build complete ===")
    print(f"prev_csv      : {prev_csv}")
    print(f"curr_csv      : {curr_csv}")
    print(f"price_col     : {price_col}")
    print(f"discount_mode : {args.discount_mode}")
    print(f"level_bucket  : {args.level_bucket}")
    print(f"contract_size : {args.contract_size}")
    print(f"output        : {output}")
    print(f"detail_output : {detail_output}")

    print("\n=== Summary ===")
    for k, v in summary.items():
        if abs(v - round(v)) < 1e-12:
            print(f"{k:30s} {int(round(v))}")
        else:
            print(f"{k:30s} {v:,.2f}")

    if grouped.empty:
        print("\nNo grouped rows were produced.")
        return

    top_n = max(int(args.top_n), 1)
    top_pos = grouped.sort_values("net_gamma_proxy", ascending=False).head(top_n)
    top_neg = grouped.sort_values("net_gamma_proxy", ascending=True).head(top_n)

    print(f"\n=== Top {top_n} positive net_gamma_proxy levels ===")
    print(top_pos.to_string(index=False))

    print(f"\n=== Top {top_n} negative net_gamma_proxy levels ===")
    print(top_neg.to_string(index=False))


if __name__ == "__main__":
    main()
