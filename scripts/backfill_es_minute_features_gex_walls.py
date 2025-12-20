#!/usr/bin/env python
"""
Backfill GEX wall levels into es_minute_features using orats_oi_gamma.

For each trade_date where es_minute_features needs walls:
- Load per-level net GEX from orats_oi_gamma (using discounted_level).
- For each minute bar (close) on that date:
    * Find nearest GEX level ABOVE price  -> gex_wall_above
    * Find nearest GEX level BELOW price  -> gex_wall_below
    * Store the GEX size at those levels  -> gex_wall_above_gex, gex_wall_below_gex
    * Compute distances in points.

Assumptions:
- Features table: es_minute_features
- GEX table: orats_oi_gamma
- GEX schema:
    ticker           TEXT
    trade_date       DATE
    discounted_level DOUBLE PRECISION
    gex_call         DOUBLE PRECISION
    gex_put          DOUBLE PRECISION
"""

import os
import datetime as dt

import pandas as pd
from sqlalchemy import create_engine, text

# --- Load .env for DATABASE_URL ---
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
GEX_TABLE = os.environ.get("ORATS_GEX_TABLE", "orats_oi_gamma")


def get_dates_needing_walls(engine):
    """
    Find trade_dates where we have features + net_gex but are missing wall info.

    We re-run for any date where either the price level OR the magnitude is NULL,
    so this will also fill in the new *_gex columns for days processed earlier.
    """
    q = text(f"""
        SELECT DISTINCT trade_date
        FROM {FEATURES_TABLE}
        WHERE (gex_wall_above IS NULL OR gex_wall_above_gex IS NULL)
          AND net_gex IS NOT NULL
        ORDER BY trade_date;
    """)
    df = pd.read_sql_query(q, engine)
    dates = [r["trade_date"] for _, r in df.iterrows()]
    print(f"[get_dates_needing_walls] Found {len(dates)} trade_dates needing walls")
    return dates


def load_gamma_levels_for_date(engine, trade_date: dt.date) -> pd.DataFrame | None:
    """
    Load per-level net GEX for a given trade_date.

    Aggregate over all expiries at each discounted_level:
        net_gex_level = SUM(gex_call + gex_put)

    Returns DataFrame with columns:
        level (float), net_gex_level (float)
    """
    q = text(f"""
        SELECT
            discounted_level::DOUBLE PRECISION AS level,
            SUM(gex_call + gex_put)          AS net_gex_level
        FROM {GEX_TABLE}
        WHERE trade_date = :trade_date
          AND ticker = 'SPX'
          AND discounted_level IS NOT NULL
        GROUP BY discounted_level
        ORDER BY level;
    """)

    df = pd.read_sql_query(q, engine, params={"trade_date": trade_date})
    if df.empty:
        print(f"[load_gamma_levels_for_date] {trade_date}: no GEX levels found")
        return None

    print(f"[load_gamma_levels_for_date] {trade_date}: loaded {len(df)} levels")
    return df


def load_bars_for_date(engine, trade_date: dt.date) -> pd.DataFrame:
    """
    Load ts_utc, close for all minutes in es_minute_features for a given trade_date.
    """
    q = text(f"""
        SELECT ts_utc, close
        FROM {FEATURES_TABLE}
        WHERE trade_date = :trade_date
        ORDER BY ts_utc;
    """)
    df = pd.read_sql_query(q, engine, params={"trade_date": trade_date})
    print(f"[load_bars_for_date] {trade_date}: loaded {len(df)} bars")
    return df


def compute_walls_for_date(df_bars: pd.DataFrame, df_levels: pd.DataFrame) -> pd.DataFrame:
    """
    Given bars (ts_utc, close) and gamma levels (level, net_gex_level),
    compute nearest level above/below for each bar, plus GEX at those levels.

    Uses pandas.merge_asof twice:
      - direction='backward' for below (<= close)
      - direction='forward'  for above (>= close)

    Returns DataFrame with:
        ts_utc,
        gex_wall_above,
        gex_wall_above_gex,
        gex_wall_below,
        gex_wall_below_gex,
        dist_to_wall_above_pts,
        dist_to_wall_below_pts
    """
    if df_bars.empty or df_levels.empty:
        return pd.DataFrame(columns=[
            "ts_utc",
            "gex_wall_above",
            "gex_wall_above_gex",
            "gex_wall_below",
            "gex_wall_below_gex",
            "dist_to_wall_above_pts",
            "dist_to_wall_below_pts",
        ])

    # Copy & sort
    bars = df_bars[["ts_utc", "close"]].copy()
    bars_sorted = bars.sort_values("close").reset_index(drop=True)

    levels_sorted = (
        df_levels[["level", "net_gex_level"]]
        .dropna()
        .drop_duplicates(subset=["level"])
        .sort_values("level")
        .reset_index(drop=True)
    )

    # Below: largest level <= close
    tmp = pd.merge_asof(
        bars_sorted,
        levels_sorted.rename(columns={
            "level": "gex_wall_below",
            "net_gex_level": "gex_wall_below_gex",
        }),
        left_on="close",
        right_on="gex_wall_below",
        direction="backward",
    )

    # Above: smallest level >= close
    tmp = pd.merge_asof(
        tmp,
        levels_sorted.rename(columns={
            "level": "gex_wall_above",
            "net_gex_level": "gex_wall_above_gex",
        }),
        left_on="close",
        right_on="gex_wall_above",
        direction="forward",
    )

    # Distances (may be NaN at extremes)
    tmp["dist_to_wall_above_pts"] = tmp["gex_wall_above"] - tmp["close"]
    tmp["dist_to_wall_below_pts"] = tmp["close"] - tmp["gex_wall_below"]

    # Back to ts_utc order
    result = tmp[
        [
            "ts_utc",
            "gex_wall_above",
            "gex_wall_above_gex",
            "gex_wall_below",
            "gex_wall_below_gex",
            "dist_to_wall_above_pts",
            "dist_to_wall_below_pts",
        ]
    ].sort_values("ts_utc").reset_index(drop=True)

    return result


def update_walls_for_date(engine, df_walls: pd.DataFrame):
    """
    Write GEX wall info into es_minute_features for the given date.
    """
    if df_walls.empty:
        print("[update_walls_for_date] No walls to update (empty df)")
        return

    q = text(f"""
        UPDATE {FEATURES_TABLE}
        SET
            gex_wall_above         = :gex_wall_above,
            gex_wall_above_gex     = :gex_wall_above_gex,
            gex_wall_below         = :gex_wall_below,
            gex_wall_below_gex     = :gex_wall_below_gex,
            dist_to_wall_above_pts = :dist_to_wall_above_pts,
            dist_to_wall_below_pts = :dist_to_wall_below_pts
        WHERE ts_utc = :ts_utc;
    """)

    records = df_walls.to_dict(orient="records")

    with engine.begin() as conn:
        conn.execute(q, records)

    print(f"[update_walls_for_date] Updated {len(records)} rows")


def main():
    dates = get_dates_needing_walls(ENGINE)
    if not dates:
        print("[main] No dates need GEX walls; nothing to do.")
        return

    for d in dates:
        print(f"\n[main] Processing {d}...")
        df_levels = load_gamma_levels_for_date(ENGINE, d)
        if df_levels is None:
            print(f"[main] Skipping {d} (no GEX levels)")
            continue

        df_bars = load_bars_for_date(ENGINE, d)
        if df_bars.empty:
            print(f"[main] Skipping {d} (no bars)")
            continue

        df_walls = compute_walls_for_date(df_bars, df_levels)
        update_walls_for_date(ENGINE, df_walls)

    print("\n[main] Done populating GEX walls (with magnitudes).")


if __name__ == "__main__":
    main()
