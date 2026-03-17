#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median, NormalDist
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text


# =========================
# Config / model constants
# =========================
TABLE_NAME = "orats_monies_minute"
EPS_T = 1e-4
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0
EXPIRY_HOUR_ET = 16
EXPIRY_MINUTE_ET = 0
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
ET_TZ = "America/New_York"


# =========================
# Small utilities
# =========================
def load_dotenv_value(env_path: Path, key: str) -> Optional[str]:
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() != key:
            continue
        v = v.strip()
        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            v = v[1:-1]
        return v
    return None


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / ".env").exists() and (candidate / "packages").exists():
            return candidate
    return start.resolve()


def pct_change(curr: float, base: float) -> float:
    if base == 0:
        return float("nan")
    return (curr - base) / abs(base) * 100.0


def years_to_exp_et(ts_et: pd.Timestamp, expiration_date: dt.date) -> float:
    exp_dt_et = pd.Timestamp(
        dt.datetime.combine(expiration_date, dt.time(EXPIRY_HOUR_ET, EXPIRY_MINUTE_ET)),
        tz=ET_TZ,
    )
    rem_seconds = max(0.0, (exp_dt_et - ts_et).total_seconds())
    T = rem_seconds / (365.0 * 24.0 * 3600.0)
    return max(T, EPS_T)


def normal_ppf(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return NormalDist().inv_cdf(p)


def k_for_abs_delta_local(p_abs: float, is_put: bool, sigma: float, T: float) -> float:
    """
    Forward-delta -> log-moneyness k = ln(K/F).

    Call abs delta  = N(d1)
    Put  abs delta  = N(-d1)
    d1 = (-k + 0.5*sigma^2*T) / (sigma*sqrt(T))
    => k = 0.5*sigma^2*T - d1*sigma*sqrt(T)
    """
    sigma = max(float(sigma), 1e-8)
    T = max(float(T), EPS_T)
    vol_sqrt_t = sigma * math.sqrt(T)
    if is_put:
        d1 = -normal_ppf(p_abs)
    else:
        d1 = normal_ppf(p_abs)
    return 0.5 * sigma * sigma * T - d1 * vol_sqrt_t


try:
    # Prefer your exact project implementation so the calibration matches production.
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = find_repo_root(SCRIPT_DIR)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from packages.shared.surface_compare import k_for_abs_delta as project_k_for_abs_delta  # type: ignore
except Exception:
    project_k_for_abs_delta = None


def k_for_abs_delta(p_abs: float, is_put: bool, sigma: float, T: float) -> float:
    if project_k_for_abs_delta is not None:
        return float(project_k_for_abs_delta(p_abs, is_put=is_put, sigma=sigma, T=T))
    return k_for_abs_delta_local(p_abs, is_put=is_put, sigma=sigma, T=T)


def interp_linear_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(ys[0])
    if x <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    if x >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        return float(y1 + (y1 - y0) * (x - x1) / (x1 - x0))
    return float(np.interp(x, xs, ys))


def available_buckets(columns: Sequence[str]) -> List[int]:
    out: List[int] = []
    for c in columns:
        m = re.fullmatch(r"vol(\d+)", c)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 99:
            out.append(n)
    return sorted(set(out), reverse=True)


def abs_delta_is_put(bucket: int) -> tuple[float, bool]:
    if bucket == 50:
        return 0.50, False
    if bucket > 50:
        return (100 - bucket) / 100.0, True
    return bucket / 100.0, False


def prev_smile_interp(prev_row: pd.Series, vol_cols: Sequence[str], T_prev: float) -> tuple[np.ndarray, np.ndarray]:
    if "vol50" not in prev_row or pd.isna(prev_row["vol50"]):
        raise ValueError("prev row missing vol50")
    atm_prev = float(prev_row["vol50"])
    buckets_prev = [b for b in available_buckets(vol_cols) if not pd.isna(prev_row.get(f"vol{b}"))]
    if len(buckets_prev) < 4:
        raise ValueError("prev row has too few valid vol buckets")

    k_prev: List[float] = []
    s_prev: List[float] = []
    for n in buckets_prev:
        if n == 50:
            k = 0.0
        else:
            p_abs, is_put = abs_delta_is_put(n)
            k = k_for_abs_delta(p_abs, is_put=is_put, sigma=atm_prev, T=T_prev)
        k_prev.append(float(k))
        s_prev.append(float(prev_row[f"vol{n}"]))

    k_np = np.array(k_prev, dtype=float)
    s_np = np.array(s_prev, dtype=float)
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))
    k_np = k_np[mask]
    s_np = s_np[mask]
    if k_np.size < 3:
        raise ValueError("previous k-grid degenerate")
    return k_np, s_np


def expected_atm_no_theta(
    prev_row: pd.Series,
    now_row: pd.Series,
    T_prev: float,
    prev_stock: float,
    now_stock: float,
    vol_cols: Sequence[str],
) -> float:
    """
    Matches the current Pass-0 / theta-off logic in your smile + skew callbacks:
      exp_atm = prev surface at k_shift + leverage shift
    """
    k_prev, s_prev = prev_smile_interp(prev_row, vol_cols, T_prev)
    k_shift = math.log(now_stock / prev_stock) if (prev_stock > 0 and now_stock > 0) else 0.0
    exp_atm_shape = interp_linear_extrap(k_shift, k_prev, s_prev)
    ret_frac = (now_stock - prev_stock) / prev_stock
    level_shift_pp = max(
        -BETA_MAX_SHIFT_PP,
        min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT),
    )
    return float(exp_atm_shape + level_shift_pp / 100.0)


@dataclass
class FitStats:
    name: str
    coefficient: float
    count: int
    mae_before_bp: float
    mae_after_bp: float
    med_before_bp: float
    med_after_bp: float
    mean_before_bp: float
    mean_after_bp: float


# =========================
# DB helpers
# =========================
def discover_columns(engine, table_name: str, schema: str = "public") -> list[str]:
    insp = inspect(engine)
    cols = insp.get_columns(table_name, schema=schema)
    if not cols:
        raise RuntimeError(f"Could not inspect {schema}.{table_name}")
    return [c["name"] for c in cols]


def choose_column(candidates: Sequence[str], available: Sequence[str], label: str) -> str:
    for c in candidates:
        if c in available:
            return c
    raise RuntimeError(f"Could not find a {label} column. Tried: {', '.join(candidates)}")


def fetch_data(
    engine,
    table_name: str,
    schema: str,
    ts_col: str,
    exp_col: str,
    stock_col: str,
    vol_cols: Sequence[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    select_cols = [ts_col, exp_col, stock_col, *vol_cols]
    sql = f"SELECT {', '.join(select_cols)} FROM {schema}.{table_name}"

    where_parts: list[str] = [f"{ts_col} IS NOT NULL", f"{exp_col} IS NOT NULL", f"{stock_col} IS NOT NULL", "vol50 IS NOT NULL"]
    params: dict[str, object] = {}

    if start_date:
        where_parts.append(f"DATE({ts_col} AT TIME ZONE 'America/New_York') >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append(f"DATE({ts_col} AT TIME ZONE 'America/New_York') <= :end_date")
        params["end_date"] = end_date

    sql += " WHERE " + " AND ".join(where_parts)
    sql += f" ORDER BY {exp_col}, {ts_col}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df


# =========================
# Calibration pipeline
# =========================
def prepare_pairs(
    df: pd.DataFrame,
    ts_col: str,
    exp_col: str,
    stock_col: str,
    vol_cols: Sequence[str],
    step_minutes: int,
    max_stock_move_pct: float,
    exclude_open_minutes: int,
    exclude_close_minutes: int,
) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("Query returned no rows.")

    work = df.copy()
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True)
    work["exp_date"] = pd.to_datetime(work[exp_col]).dt.date
    work["ts_et"] = work[ts_col].dt.tz_convert(ET_TZ)
    work["trade_date_et"] = work["ts_et"].dt.date
    work["time_et"] = work["ts_et"].dt.time

    # Keep only same-day expiry rows (0DTE).
    work = work[work["trade_date_et"] == work["exp_date"]].copy()
    if work.empty:
        raise RuntimeError("No 0DTE rows left after filtering trade_date_et == expiration date.")

    open_cut = (dt.datetime.combine(dt.date.today(), MARKET_OPEN) + dt.timedelta(minutes=exclude_open_minutes)).time()
    close_cut = (dt.datetime.combine(dt.date.today(), MARKET_CLOSE) - dt.timedelta(minutes=exclude_close_minutes)).time()
    work = work[(work["time_et"] >= open_cut) & (work["time_et"] <= close_cut)].copy()
    if work.empty:
        raise RuntimeError("No rows left after market-hours open/close exclusion filter.")

    work = work.sort_values(["exp_date", ts_col]).copy()
    work["pair_ts_utc"] = work[ts_col] + pd.to_timedelta(step_minutes, unit="m")

    right_cols = list(dict.fromkeys([ts_col, "exp_date", "trade_date_et", "ts_et", stock_col, *vol_cols]))
    rhs = work[right_cols].copy()
    rename_map = {c: f"now_{c}" for c in rhs.columns if c not in ("exp_date",)}
    rhs = rhs.rename(columns=rename_map)

    pairs = work.merge(
        rhs,
        how="inner",
        left_on=["exp_date", "pair_ts_utc"],
        right_on=["exp_date", f"now_{ts_col}"],
        suffixes=("", ""),
    )
    if pairs.empty:
        raise RuntimeError(
            f"No exact {step_minutes}-minute pairs found. Try a different step size or loosen the data requirement."
        )

    pairs["prev_stock"] = pd.to_numeric(pairs[stock_col], errors="coerce")
    pairs["now_stock"] = pd.to_numeric(pairs[f"now_{stock_col}"], errors="coerce")
    pairs["stock_move_pct"] = (pairs["now_stock"] - pairs["prev_stock"]) / pairs["prev_stock"] * 100.0
    pairs = pairs[np.isfinite(pairs["stock_move_pct"])].copy()
    pairs = pairs[pairs["stock_move_pct"].abs() <= max_stock_move_pct].copy()
    if pairs.empty:
        raise RuntimeError(
            "No pairs left after small-stock-move filter. Increase --max-stock-move-pct or reduce --step-minutes."
        )

    pairs["T_prev"] = pairs["ts_et"].apply(lambda ts: years_to_exp_et(ts, ts.date()))
    pairs["T_now"] = pairs["now_ts_et"].apply(lambda ts: years_to_exp_et(ts, ts.date()))
    pairs["droot"] = np.sqrt(np.maximum(pairs["T_prev"].astype(float), EPS_T)) - np.sqrt(
        np.maximum(pairs["T_now"].astype(float), EPS_T)
    )
    pairs = pairs[pairs["droot"] > 0].copy()
    if pairs.empty:
        raise RuntimeError("No rows left after droot > 0 filter.")

    return pairs.reset_index(drop=True)


def compute_model_residuals(
    pairs: pd.DataFrame,
    vol_cols: Sequence[str],
    ts_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    prev_row_cols = list(dict.fromkeys([*vol_cols, "vol50"]))
    now_row_cols = [f"now_{c}" for c in prev_row_cols]

    for _, row in pairs.iterrows():
        try:
            prev_series = pd.Series({c: row.get(c) for c in prev_row_cols})
            now_series = pd.Series({c.replace("now_", ""): row.get(c) for c in now_row_cols})

            prev_stock = float(row["prev_stock"])
            now_stock = float(row["now_stock"])
            T_prev = float(row["T_prev"])
            T_now = float(row["T_now"])
            actual_atm_now = float(row["now_vol50"])

            atm_exp = expected_atm_no_theta(
                prev_row=prev_series,
                now_row=now_series,
                T_prev=T_prev,
                prev_stock=prev_stock,
                now_stock=now_stock,
                vol_cols=vol_cols,
            )
            y = actual_atm_now - atm_exp
            rows.append(
                {
                    "trade_date": row["trade_date_et"],
                    "prev_ts_utc": row[ts_col],
                    "now_ts_utc": row[f"now_{ts_col}"],
                    "prev_ts_et": row["ts_et"],
                    "now_ts_et": row["now_ts_et"],
                    "exp_date": row["exp_date"],
                    "prev_stock": prev_stock,
                    "now_stock": now_stock,
                    "stock_move_pct": float(row["stock_move_pct"]),
                    "T_prev": T_prev,
                    "T_now": T_now,
                    "droot": float(row["droot"]),
                    "actual_atm_now": actual_atm_now,
                    "expected_atm_no_theta": float(atm_exp),
                    "residual_frac": float(y),
                    "residual_bp": float(y * 10000.0),
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No usable rows left after expected-ATM reconstruction.")
    return out


def ols_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.dot(x, x))
    if denom <= 0:
        raise RuntimeError("OLS denominator is zero.")
    return float(np.dot(x, y) / denom)


def trimmed_ols_no_intercept(x: np.ndarray, y: np.ndarray, trim_frac: float = 0.05) -> float:
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError("trim_frac must be in [0, 0.5).")
    lo = np.quantile(y, trim_frac)
    hi = np.quantile(y, 1.0 - trim_frac)
    mask = (y >= lo) & (y <= hi)
    return ols_no_intercept(x[mask], y[mask])


def median_implied_ratio(x: np.ndarray, y: np.ndarray, min_x: float = 1e-8) -> float:
    mask = np.abs(x) >= min_x
    if not mask.any():
        raise RuntimeError("All droot values are too small for ratio median.")
    return float(np.median(y[mask] / x[mask]))


def metrics(df: pd.DataFrame, coeff: float, name: str) -> FitStats:
    before = df["residual_bp"].astype(float).to_numpy()
    after = (df["residual_frac"].astype(float) - coeff * df["droot"].astype(float)) * 10000.0
    return FitStats(
        name=name,
        coefficient=float(coeff),
        count=int(len(df)),
        mae_before_bp=float(np.mean(np.abs(before))),
        mae_after_bp=float(np.mean(np.abs(after))),
        med_before_bp=float(np.median(before)),
        med_after_bp=float(np.median(after)),
        mean_before_bp=float(np.mean(before)),
        mean_after_bp=float(np.mean(after)),
    )


def pretty_stats(stats: FitStats) -> str:
    return (
        f"[{stats.name}]\n"
        f"  coefficient              : {stats.coefficient:,.4f}\n"
        f"  row count                : {stats.count:,}\n"
        f"  mean residual before (bp): {stats.mean_before_bp:,.2f}\n"
        f"  mean residual after  (bp): {stats.mean_after_bp:,.2f}\n"
        f"  med  residual before (bp): {stats.med_before_bp:,.2f}\n"
        f"  med  residual after  (bp): {stats.med_after_bp:,.2f}\n"
        f"  MAE  residual before (bp): {stats.mae_before_bp:,.2f}\n"
        f"  MAE  residual after  (bp): {stats.mae_after_bp:,.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Pass-1 ATM theta coefficient from orats_monies_minute.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file containing DATABASE_URL (default: .env)")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--table", default=TABLE_NAME)
    parser.add_argument("--start-date", default=None, help="Optional ET trade-date floor, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Optional ET trade-date ceiling, YYYY-MM-DD")
    parser.add_argument("--step-minutes", type=int, default=5, help="Pair t with t + N minutes (default: 5)")
    parser.add_argument("--max-stock-move-pct", type=float, default=0.05, help="Keep only pairs with abs(stock move %% ) <= this value (default: 0.05)")
    parser.add_argument("--exclude-open-minutes", type=int, default=10, help="Drop first N minutes after 9:30 ET (default: 10)")
    parser.add_argument("--exclude-close-minutes", type=int, default=10, help="Drop last N minutes before 16:00 ET (default: 10)")
    parser.add_argument("--trim-frac", type=float, default=0.05, help="Trim fraction for trimmed OLS on residuals (default: 0.05)")
    parser.add_argument("--validation-frac", type=float, default=0.25, help="Fraction of trade dates held out for validation (default: 0.25)")
    parser.add_argument("--save-sample-csv", default=None, help="Optional path to save the calibration rows as CSV")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
    db_url = os.getenv("DATABASE_URL") or load_dotenv_value(env_path, "DATABASE_URL")
    if not db_url:
        raise SystemExit(f"Could not find DATABASE_URL in environment or {env_path}")

    engine = create_engine(db_url)
    cols = discover_columns(engine, args.table, args.schema)

    ts_col = choose_column(["snap_shot_date", "snapshot_date", "ts_utc"], cols, "snapshot timestamp")
    exp_col = choose_column(["expir_date", "expiration", "expir_date_iso", "expiry_date"], cols, "expiration date")
    stock_col = choose_column(["stock_price", "stock", "underlying_price", "spot"], cols, "stock price")
    vol_cols = [c for c in cols if re.fullmatch(r"vol\d+", c)]
    if "vol50" not in vol_cols:
        raise SystemExit(f"{args.schema}.{args.table} does not contain vol50")
    if len(vol_cols) < 4:
        raise SystemExit(f"{args.schema}.{args.table} needs several volNN columns; found only {len(vol_cols)}")

    print("Using columns:")
    print(f"  timestamp : {ts_col}")
    print(f"  expiry    : {exp_col}")
    print(f"  stock     : {stock_col}")
    print(f"  vol cols  : {len(vol_cols)} found")
    print()

    raw = fetch_data(
        engine=engine,
        table_name=args.table,
        schema=args.schema,
        ts_col=ts_col,
        exp_col=exp_col,
        stock_col=stock_col,
        vol_cols=vol_cols,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Fetched {len(raw):,} raw rows from {args.schema}.{args.table}")

    pairs = prepare_pairs(
        df=raw,
        ts_col=ts_col,
        exp_col=exp_col,
        stock_col=stock_col,
        vol_cols=vol_cols,
        step_minutes=args.step_minutes,
        max_stock_move_pct=args.max_stock_move_pct,
        exclude_open_minutes=args.exclude_open_minutes,
        exclude_close_minutes=args.exclude_close_minutes,
    )
    print(f"Built {len(pairs):,} candidate {args.step_minutes}-minute quiet 0DTE pairs")

    cal = compute_model_residuals(pairs, vol_cols=vol_cols, ts_col=ts_col)
    print(f"Kept {len(cal):,} usable calibration rows after model reconstruction")
    print()

    if args.save_sample_csv:
        out_path = Path(args.save_sample_csv)
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cal.to_csv(out_path, index=False)
        print(f"Saved calibration sample rows to: {out_path}")
        print()

    x = cal["droot"].astype(float).to_numpy()
    y = cal["residual_frac"].astype(float).to_numpy()

    unique_days = sorted(pd.to_datetime(cal["trade_date"]).dt.date.unique())
    if len(unique_days) < 4:
        raise SystemExit("Need at least 4 trade dates for a meaningful train/validation split.")

    val_days = max(1, int(round(len(unique_days) * args.validation_frac)))
    train_days = unique_days[:-val_days]
    holdout_days = unique_days[-val_days:]

    train_df = cal[cal["trade_date"].isin(train_days)].copy()
    holdout_df = cal[cal["trade_date"].isin(holdout_days)].copy()
    x_train = train_df["droot"].astype(float).to_numpy()
    y_train = train_df["residual_frac"].astype(float).to_numpy()

    coeff_ols = ols_no_intercept(x_train, y_train)
    coeff_trim = trimmed_ols_no_intercept(x_train, y_train, trim_frac=args.trim_frac)
    coeff_med = median_implied_ratio(x_train, y_train)

    print("Train dates  :", train_days[0], "->", train_days[-1], f"({len(train_days)} days)")
    print("Holdout dates:", holdout_days[0], "->", holdout_days[-1], f"({len(holdout_days)} days)")
    print()

    stats_train_ols = metrics(train_df, coeff_ols, "TRAIN / OLS")
    stats_train_trim = metrics(train_df, coeff_trim, "TRAIN / TRIMMED_OLS")
    stats_train_med = metrics(train_df, coeff_med, "TRAIN / MEDIAN_RATIO")
    stats_hold_ols = metrics(holdout_df, coeff_ols, "HOLDOUT / OLS")
    stats_hold_trim = metrics(holdout_df, coeff_trim, "HOLDOUT / TRIMMED_OLS")
    stats_hold_med = metrics(holdout_df, coeff_med, "HOLDOUT / MEDIAN_RATIO")

    print(pretty_stats(stats_train_ols))
    print()
    print(pretty_stats(stats_hold_ols))
    print("\n" + "-" * 72 + "\n")
    print(pretty_stats(stats_train_trim))
    print()
    print(pretty_stats(stats_hold_trim))
    print("\n" + "-" * 72 + "\n")
    print(pretty_stats(stats_train_med))
    print()
    print(pretty_stats(stats_hold_med))
    print()

    candidates = [
        ("OLS", coeff_ols, stats_hold_ols.mae_after_bp),
        ("TRIMMED_OLS", coeff_trim, stats_hold_trim.mae_after_bp),
        ("MEDIAN_RATIO", coeff_med, stats_hold_med.mae_after_bp),
    ]
    best_name, best_coeff, best_mae = min(candidates, key=lambda t: t[2])

    print("Suggested starting coefficient for Pass 1:")
    print(f"  {best_name}: {best_coeff:,.4f}")
    print(f"  holdout MAE after theta: {best_mae:,.2f} bp")
    print()
    print("Interpretation:")
    print("  * Negative values imply ATM expected IV decays lower as time passes.")
    print("  * Copy the suggested value into THETA_ATM_PP_PER_SQRT_YEAR.")
    print("  * If OLS / trimmed / median are all in the same neighborhood, that is a good sign.")
    print("  * If they are far apart, your sample likely contains more than pure theta (vol crush / regime shifts).")


if __name__ == "__main__":
    main()
