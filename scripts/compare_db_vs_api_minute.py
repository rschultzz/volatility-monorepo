#!/usr/bin/env python3
from __future__ import annotations
import sys, os, io, argparse
from pathlib import Path
import pandas as pd

# repo root on sys.path
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.shared.options_orats import (
    pt_minute_to_et, fetch_one_minute_monies
)
from packages.shared.ingest.monies_ingest import read_minute_expiry_df_from_db

VOL_ORDER = ["vol90","vol85","vol80","vol75","vol70","vol65","vol50",
             "vol35","vol30","vol25","vol20","vol15","vol10"]

PC2VOL = {  # how we convert P/C buckets -> volXX columns
    "p10":"vol90","p15":"vol85","p20":"vol80","p25":"vol75","p30":"vol70","p35":"vol65",
    "atm":"vol50",
    "c35":"vol35","c30":"vol30","c25":"vol25","c20":"vol20","c15":"vol15","c10":"vol10",
}

def api_row_to_vols(df_api: pd.DataFrame) -> dict[str, float]:
    """Build volXX map from the API minute DF."""
    if df_api is None or df_api.empty:
        return {}

    row = df_api.iloc[0]
    keys = {c.lower(): c for c in df_api.columns}

    def get(*cands):
        for c in cands:
            k = keys.get(c)
            if k is not None:
                v = pd.to_numeric(row[k], errors="coerce")
                if pd.notna(v):
                    return float(v)
        return None

    out = {}

    # Prefer explicit P/C/ATM if present
    for pc, vol in PC2VOL.items():
        v = get(pc)  # e.g. "p10", "atm", "c10"
        if v is not None:
            out[vol] = v

    # Fill any gaps from explicit volXX columns if they exist
    for vol in VOL_ORDER:
        if vol not in out:
            v = get(vol)
            if v is not None:
                out[vol] = v

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=os.getenv("TICKER","SPX"))
    ap.add_argument("--date", required=True, help="Trade date YYYY-MM-DD (PT date)")
    ap.add_argument("--expiry", required=True, help="Expiry YYYY-MM-DD")
    ap.add_argument("--hhmm_pt", required=True, help="Minute HH:MM in PT")
    args = ap.parse_args()

    # ---- DB
    df_db = read_minute_expiry_df_from_db(args.ticker, args.date, args.expiry, args.hhmm_pt)
    db_vols = {k: float(df_db.iloc[0][k]) for k in VOL_ORDER if k in df_db.columns} if not df_db.empty else {}

    # ---- API
    ts_et = pt_minute_to_et(args.date, args.hhmm_pt)
    df_api = fetch_one_minute_monies(ts_et, args.ticker, args.expiry)
    api_vols = api_row_to_vols(df_api)

    # ---- Print side-by-side and diffs
    print(f"\n== {args.ticker} {args.date} {args.expiry} {args.hhmm_pt} PT ==")
    print("{:>6}  {:>8}  {:>8}  {:>8}".format("bucket","db","api","abs_diff"))
    for k in VOL_ORDER:
        dv = db_vols.get(k)
        av = api_vols.get(k)
        if dv is None and av is None:
            continue
        diff = None if (dv is None or av is None) else abs(dv - av)
        print("{:>6}  {:>8}  {:>8}  {:>8}".format(
            k, f"{dv:.5f}" if dv is not None else "—",
               f"{av:.5f}" if av is not None else "—",
               f"{diff:.5f}" if diff is not None else "—",
        ))

    # simple diagnostic
    if db_vols and api_vols:
        missing = [k for k in VOL_ORDER if k in api_vols and k not in db_vols]
        const_db = [k for k in VOL_ORDER if k in db_vols][1:]
        # detect plateaus in DB numbers
        plateaus = []
        prev = None
        for k in VOL_ORDER:
            if k in db_vols:
                if prev is not None and abs(db_vols[k]-prev) < 1e-6:
                    plateaus.append(k)
                prev = db_vols[k]
        if missing:
            print("\n[NOTE] DB missing buckets that API has:", missing)
        if plateaus:
            print("[NOTE] Identical consecutive DB buckets (plateau):", plateaus)

if __name__ == "__main__":
    main()
