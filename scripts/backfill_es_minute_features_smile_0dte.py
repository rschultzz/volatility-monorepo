#!/usr/bin/env python
"""
Backfill primary (front) expiry ATM IV + skew into es_minute_features
from orats_monies_minute.

Logic:
- For each ORATS snapshot (snapshot_pt), for SPX/SPXW:
    * pick the row with the smallest non-negative dte (front expiry)
- Treat snapshot_pt as US/Eastern local time, convert to UTC.
- Join on ts_utc in es_minute_features.
- For those minutes, populate:

    smile_dte_primary                    = dte of front expiry
    iv_atm                               = atmiv
    iv_atm_change_10m                    = iv_atm - iv_atm 10 rows earlier
    call_skew_pp_primary                 = (vol25 - vol50) * 100
    put_skew_pp_primary                  = (vol75 - vol50) * 100
    call_skew_pp_primary_change_10m      = call_skew_pp_primary - value 10 rows earlier
    put_skew_pp_primary_change_10m       = put_skew_pp_primary - value 10 rows earlier

We recompute for ALL ts_utc rows in es_minute_features each time; safe and simple
for your current data size.
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text

# --- load .env for DATABASE_URL ---
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in your environment or .env")

ENGINE = create_engine(DATABASE_URL)

FEATURES_TABLE = os.environ.get("ES_FEATURES_TABLE", "es_minute_features")
MONIES_TABLE = os.environ.get("ORATS_MONIES_TABLE", "orats_monies_minute")


def load_feature_timestamps(engine) -> pd.DataFrame:
    """
    Load ts_utc for ALL feature rows. We'll recompute smile features for
    everything each run (simple + safe at current scale).
    """
    q = text(f"""
        SELECT ts_utc
        FROM {FEATURES_TABLE}
        ORDER BY ts_utc;
    """)
    df = pd.read_sql_query(q, engine)
    print(f"[load_feature_timestamps] Loaded {len(df)} feature timestamps")
    return df


def load_monies_front(engine) -> pd.DataFrame:
    """
    Load ORATS monies snapshots for SPX*, and for each snapshot_pt keep only
    the row with the smallest non-negative dte (front expiry).

    We select the pieces we need:
        atmiv, dte, vol50, vol25, vol75
    and convert snapshot_pt (ET) to UTC.
    """
    q = text(f"""
        SELECT
            (snapshot_pt AT TIME ZONE 'US/Eastern') AS ts_utc,
            atmiv,
            dte,
            vol50,
            vol25,
            vol75
        FROM {MONIES_TABLE}
        WHERE ticker LIKE 'SPX%%'
          AND dte >= 0;
    """)
    df = pd.read_sql_query(q, engine)
    print(f"[load_monies_front] Loaded {len(df)} SPX rows from {MONIES_TABLE}")

    if df.empty:
        return df

    # For each ts_utc, keep the smallest dte (front expiry)
    df = df.sort_values(["ts_utc", "dte"]).reset_index(drop=True)
    df_front = df.drop_duplicates(subset=["ts_utc"], keep="first").reset_index(drop=True)

    print(f"[load_monies_front] Reduced to {len(df_front)} front-expiry records")
    return df_front


def build_feature_updates(df_feat: pd.DataFrame, df_monies: pd.DataFrame) -> pd.DataFrame:
    """
    Merge feature timestamps with front-expiry monies by ts_utc and compute
    ATM IV + skew features and 10-minute changes.
    """
    if df_feat.empty or df_monies.empty:
        return pd.DataFrame(
            columns=[
                "ts_utc",
                "smile_dte_primary",
                "iv_atm",
                "iv_atm_change_10m",
                "call_skew_pp_primary",
                "put_skew_pp_primary",
                "call_skew_pp_primary_change_10m",
                "put_skew_pp_primary_change_10m",
            ]
        )

    # Merge on ts_utc (UTC minute timestamps)
    df = df_feat.merge(df_monies, on="ts_utc", how="left")

    # Keep only rows where we actually have ORATS data
    df = df.dropna(subset=["atmiv"]).copy()

    # Rename / compute core features
    df.rename(columns={"atmiv": "iv_atm"}, inplace=True)
    df["smile_dte_primary"] = df["dte"].astype("int64")

    # Skews in vol points (pp), same as Skew callbacks
    # call = 25Δ call (vol25) vs ATM (vol50)
    # put  = 25Δ put  (vol75) vs ATM (vol50)
    df["call_skew_pp_primary"] = (df["vol25"] - df["vol50"]) * 100.0
    df["put_skew_pp_primary"] = (df["vol75"] - df["vol50"]) * 100.0

    # Sort by time to compute 10-snapshot changes
    df = df.sort_values("ts_utc").reset_index(drop=True)

    df["iv_atm_change_10m"] = df["iv_atm"] - df["iv_atm"].shift(10)
    df["call_skew_pp_primary_change_10m"] = (
        df["call_skew_pp_primary"] - df["call_skew_pp_primary"].shift(10)
    )
    df["put_skew_pp_primary_change_10m"] = (
        df["put_skew_pp_primary"] - df["put_skew_pp_primary"].shift(10)
    )

    df_out = df[
        [
            "ts_utc",
            "smile_dte_primary",
            "iv_atm",
            "iv_atm_change_10m",
            "call_skew_pp_primary",
            "put_skew_pp_primary",
            "call_skew_pp_primary_change_10m",
            "put_skew_pp_primary_change_10m",
        ]
    ].reset_index(drop=True)

    print(f"[build_feature_updates] Prepared {len(df_out)} update rows with smile features")
    return df_out


def apply_updates(engine, df_updates: pd.DataFrame):
    """
    Apply updates to es_minute_features for each ts_utc row.
    """
    if df_updates.empty:
        print("[apply_updates] No updates to apply.")
        return

    q = text(f"""
        UPDATE {FEATURES_TABLE}
        SET
            smile_dte_primary                 = :smile_dte_primary,
            iv_atm                            = :iv_atm,
            iv_atm_change_10m                 = :iv_atm_change_10m,
            call_skew_pp_primary              = :call_skew_pp_primary,
            put_skew_pp_primary               = :put_skew_pp_primary,
            call_skew_pp_primary_change_10m   = :call_skew_pp_primary_change_10m,
            put_skew_pp_primary_change_10m    = :put_skew_pp_primary_change_10m
        WHERE ts_utc = :ts_utc;
    """)

    records = df_updates.to_dict(orient="records")

    with engine.begin() as conn:
        conn.execute(q, records)

    print(f"[apply_updates] Updated {len(records)} rows in {FEATURES_TABLE}")


def main():
    df_feat = load_feature_timestamps(ENGINE)
    if df_feat.empty:
        print("[main] No feature rows found; exiting.")
        return

    df_monies = load_monies_front(ENGINE)
    if df_monies.empty:
        print("[main] No SPX monies rows found; exiting.")
        return

    df_updates = build_feature_updates(df_feat, df_monies)
    apply_updates(ENGINE, df_updates)
    print("[main] Front-expiry smile + skew backfill complete.")


if __name__ == "__main__":
    main()
