import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DB_TABLE_NAME = "orats_monies_minute"

def _normalize_db_url(url: str) -> str:
    """
    Ensure SQLAlchemy-friendly driver. Examples:
      postgres://user:pass@host/db  -> postgresql+psycopg://user:pass@host/db
      postgresql://...              -> postgresql+psycopg://...
      postgresql+psycopg://...      -> unchanged
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url

def get_engine(url: Optional[str] = None, *, pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """
    Create a pooled SQLAlchemy Engine using DATABASE_URL or provided url.
    """
    raw = (url or os.getenv("DATABASE_URL", "")).strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set")
    norm = _normalize_db_url(raw)
    engine = create_engine(
        norm,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        future=True,
    )
    return engine

def fetch_data_from_db(trade_date_iso: str, expiration_iso: str, times_pt: List[str]) -> pd.DataFrame:
    """
    Fetches all necessary data for a given trade date, expiration, and time slices
    from the database with a single query.
    """
    if not trade_date_iso or not expiration_iso or not times_pt:
        return pd.DataFrame()

    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in times_pt]
    query = text(f"""
        SELECT *
        FROM "{DB_TABLE_NAME}"
        WHERE
            trade_date = :trade_date AND
            expir_date = :expir_date AND
            snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt;
    """)

    try:
        engine = get_engine()
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "trade_date": trade_date_iso,
                "expir_date": expiration_iso,
            })
        return df
    except Exception as e:
        print(f"Database query failed: {e}")
        return pd.DataFrame()

def fetch_skew_data(trade_date_iso: str, expiration_iso: str, times_pt: List[str]) -> pd.DataFrame:
    if not trade_date_iso or not expiration_iso or not times_pt:
        return pd.DataFrame()
    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in times_pt]
    query = text(f"""
        SELECT * FROM "{DB_TABLE_NAME}"
        WHERE trade_date = :trade_date AND expir_date = :expir_date
          AND snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt;
    """)
    try:
        engine = get_engine()
        with engine.connect() as connection:
            return pd.read_sql(query, connection, params={"trade_date": trade_date_iso, "expir_date": expiration_iso})
    except Exception as e:
        print(f"Skew DB query failed: {e}")
        return pd.DataFrame()

def fetch_term_metrics_data(trade_date_iso: str, times_pt: List[str]) -> pd.DataFrame:
    if not trade_date_iso or not times_pt:
        return pd.DataFrame()

    sorted_times_pt = sorted(times_pt)
    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in sorted_times_pt]

    query = text(f"""
        SELECT snapshot_pt, trade_date, expir_date, vol50
        FROM "{DB_TABLE_NAME}"
        WHERE trade_date = :trade_date AND snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt, expir_date;
    """)

    try:
        engine = get_engine()
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params={"trade_date": trade_date_iso})
            if df.empty:
                return pd.DataFrame()

            df["expir_date"] = pd.to_datetime(df["expir_date"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["dte"] = (df["expir_date"] - df["trade_date"]).dt.days

            def get_closest_iv(group, dte_target):
                if group.empty:
                    return np.nan
                closest_row = group.iloc[(group["dte"] - dte_target).abs().argmin()]
                return closest_row["vol50"]

            metrics_list = []
            for snapshot, group in df.groupby("snapshot_pt"):
                iv_3d = get_closest_iv(group, 3)
                iv_30d = get_closest_iv(group, 30)
                iv_90d = get_closest_iv(group, 90)

                front_back_spread = (iv_3d - iv_30d) if not (pd.isna(iv_3d) or pd.isna(iv_30d)) else np.nan
                front_back_ratio = (iv_3d / iv_30d) if not (pd.isna(iv_3d) or pd.isna(iv_30d) or iv_30d == 0) else np.nan
                slope_30_90 = (iv_30d - iv_90d) if not (pd.isna(iv_30d) or pd.isna(iv_90d)) else np.nan

                metrics_list.append({
                    "snapshot_pt": snapshot,
                    "front_back_spread": front_back_spread,
                    "front_back_ratio": front_back_ratio,
                    "slope_30_90": slope_30_90,
                })

            metrics_df = pd.DataFrame(metrics_list)
            metrics_df = metrics_df.sort_values("snapshot_pt").reset_index(drop=True)
            return metrics_df

    except Exception as e:
        print(f"Term Metrics DB query failed: {e}")
        return pd.DataFrame()

def fetch_term_structure_data(trade_date_iso: str, times_pt: List[str]) -> pd.DataFrame:
    if not trade_date_iso or not times_pt:
        return pd.DataFrame()
    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in times_pt]
    query = text(f"""
        SELECT snapshot_pt, trade_date, expir_date, vol50
        FROM "{DB_TABLE_NAME}"
        WHERE trade_date = :trade_date AND snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt, expir_date;
    """)
    try:
        engine = get_engine()
        with engine.connect() as connection:
            return pd.read_sql(query, connection, params={"trade_date": trade_date_iso})
    except Exception as e:
        print(f"Term Structure DB query failed: {e}")
        return pd.DataFrame()
