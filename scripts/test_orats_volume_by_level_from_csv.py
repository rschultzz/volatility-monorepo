#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


PREFERRED_COLUMNS = {
    'ticker': ['ticker'],
    'trade_date': ['tradeDate', 'trade_date'],
    'expir_date': ['expirDate', 'expir_date', 'expirationDate'],
    'dte': ['dte', 'daysToExpiration'],
    'strike': ['strike'],
    'residual_rate': ['residualRate', 'residual_rate', 'rate'],
    'call_volume': ['callVolume', 'call_volume'],
    'put_volume': ['putVolume', 'put_volume'],
}


def pick_col(df: pd.DataFrame, logical_name: str, required: bool = True) -> str | None:
    for cand in PREFERRED_COLUMNS.get(logical_name, []):
        if cand in df.columns:
            return cand
    if required:
        raise KeyError(
            f"Could not find a column for '{logical_name}'. Available columns: {list(df.columns)}"
        )
    return None



def compute_discounted_level(strike: float, residual_rate: float, dte_days: float) -> float:
    t = max(float(dte_days), 0.0) / 365.0
    return float(strike) * math.exp(-float(residual_rate) * t)



def round_to_bucket(values: pd.Series, bucket: float) -> pd.Series:
    if bucket <= 0:
        raise ValueError("bucket must be > 0")
    return (values / bucket).round() * bucket



def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Read one ORATS strikes-chain CSV snapshot and aggregate cumulative call/put volume "
            "by discounted level."
        )
    )
    ap.add_argument('--input-csv', required=True, help='Path to ORATS snapshot CSV')
    ap.add_argument(
        '--output-csv',
        default=None,
        help='Optional output path. Default: outputs/<input_stem>_volume_by_level.csv',
    )
    ap.add_argument(
        '--level-bucket',
        type=float,
        default=1.0,
        help='Bucket size for discounted level rounding. Use 1 for exact integer, 5 for 5-point buckets.',
    )
    ap.add_argument(
        '--top-n',
        type=int,
        default=15,
        help='How many top positive/negative net signed-volume levels to print.',
    )
    ap.add_argument(
        '--keep-zero-rows',
        action='store_true',
        help='Keep rows where both callVolume and putVolume are zero.',
    )
    args = ap.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError('Input CSV is empty.')

    ticker_col = pick_col(df, 'ticker', required=False)
    trade_date_col = pick_col(df, 'trade_date', required=False)
    expir_date_col = pick_col(df, 'expir_date', required=False)
    dte_col = pick_col(df, 'dte')
    strike_col = pick_col(df, 'strike')
    residual_rate_col = pick_col(df, 'residual_rate', required=False)
    call_vol_col = pick_col(df, 'call_volume')
    put_vol_col = pick_col(df, 'put_volume')

    work = df.copy()
    work[strike_col] = pd.to_numeric(work[strike_col], errors='coerce')
    work[dte_col] = pd.to_numeric(work[dte_col], errors='coerce')
    work[call_vol_col] = pd.to_numeric(work[call_vol_col], errors='coerce').fillna(0.0)
    work[put_vol_col] = pd.to_numeric(work[put_vol_col], errors='coerce').fillna(0.0)

    if residual_rate_col is None:
        work['_residual_rate'] = 0.0
        residual_rate_col = '_residual_rate'
    else:
        work[residual_rate_col] = pd.to_numeric(work[residual_rate_col], errors='coerce').fillna(0.0)

    work = work.dropna(subset=[strike_col, dte_col]).copy()
    if not args.keep_zero_rows:
        work = work[(work[call_vol_col] != 0) | (work[put_vol_col] != 0)].copy()

    if work.empty:
        raise ValueError('No usable rows remain after cleaning/filtering.')

    work['discounted_level_raw'] = [
        compute_discounted_level(s, r, d)
        for s, r, d in zip(work[strike_col], work[residual_rate_col], work[dte_col])
    ]
    work['level'] = round_to_bucket(work['discounted_level_raw'], float(args.level_bucket))
    if float(args.level_bucket).is_integer():
        work['level'] = work['level'].astype(int)

    work['signed_volume'] = work[call_vol_col] - work[put_vol_col]

    group_cols = ['level']
    agg = (
        work.groupby(group_cols, dropna=False)
        .agg(
            call_volume=(call_vol_col, 'sum'),
            put_volume=(put_vol_col, 'sum'),
            signed_volume=('signed_volume', 'sum'),
            contracts_touched=(strike_col, 'count'),
        )
        .reset_index()
        .sort_values('level')
    )

    if ticker_col and ticker_col in work.columns and work[ticker_col].nunique() == 1:
        ticker_val = str(work[ticker_col].iloc[0])
    else:
        ticker_val = 'UNKNOWN'

    if trade_date_col and trade_date_col in work.columns and work[trade_date_col].nunique() == 1:
        trade_date_val = str(work[trade_date_col].iloc[0])
    else:
        trade_date_val = 'UNKNOWN'

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path('outputs') / f'{input_path.stem}_volume_by_level.csv'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(output_path, index=False)

    print(f'Read {len(df):,} rows from {input_path}')
    print(f'Usable rows after cleaning/filtering: {len(work):,}')
    print(f'Ticker: {ticker_val}')
    print(f'Trade date: {trade_date_val}')
    print(f'Level bucket: {args.level_bucket}')
    print(f'Wrote grouped output: {output_path}')

    top_n = max(1, int(args.top_n))
    pos = agg.sort_values('signed_volume', ascending=False).head(top_n)
    neg = agg.sort_values('signed_volume', ascending=True).head(top_n)

    print(f'\n=== Top {top_n} positive signed-volume levels ===')
    if pos.empty:
        print('(none)')
    else:
        print(pos.to_string(index=False))

    print(f'\n=== Top {top_n} negative signed-volume levels ===')
    if neg.empty:
        print('(none)')
    else:
        print(neg.to_string(index=False))


if __name__ == '__main__':
    main()
