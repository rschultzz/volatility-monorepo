#!/usr/bin/env python3
"""CR-BQ extraction — per-day statistics (READ-ONLY, train only). No decile-level statistic is computed here.

Spec: specs/CR-BQ-gamma-autocorrelation.md (L1–L5, Step 0 definitions, amendment A1).
Inputs: CR-BK parquet (scripts/logs/cr_bk/), es_minutes (is_rth). No ORATS monies field at the open is read.
Output: scripts/output/cr_bq/days.parquet — one row per L1 universe day (641) with
  G (A, primary), G_B (diagnostic), trailing-250 ranks and deciles (A1.2), raw / ADV deciles (secondary), sign,
  rho5, VR, the six 60-min returns, first/last 30-min returns, rROD (A1.3), Gao window, realized skew and RV,
  previous-session rho5 / last-30 (L5), day of week.

Usage: apps/web/.venv/bin/python scripts/cr_bq_extract.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import cr_bq_common as Q

RANK_WINDOW, RANK_MIN = 250, 120     # A1.2 (CR-BK pct_rank)
ADV_N = 20                           # A1.2


def acf1(x: np.ndarray) -> float:
    d = x - x.mean()
    den = np.sum(d * d)
    return float(np.sum(d[:-1] * d[1:]) / den) if den > 0 else np.nan


def day_stats(day: pd.DataFrame) -> dict:
    P = Q.price_grid(day)
    lp = np.log(P)
    r1 = np.diff(lp)
    r5 = lp[5::5] - lp[0:-5:5]
    r30 = lp[30::30] - lp[0:-30:30]
    r60 = lp[60:361:60] - lp[0:301:60]
    assert len(r1) == 390 and len(r5) == 78 and len(r30) == 13 and len(r60) == 6
    s2 = np.sum(r1 ** 2)
    m = set(day.minute)
    return {
        "n_bars": len(day), "rho5": acf1(r5), "VR": float(np.sum(r30 ** 2) / np.sum(r5 ** 2)),
        **{f"r60_{i + 1}": float(v) for i, v in enumerate(r60)},
        "r_first30": float(lp[30] - lp[0]), "r_last30": float(lp[390] - lp[360]),
        "lp360": float(lp[360]), "lp30": float(lp[30]),
        "rv": float(s2), "skew": float(np.sqrt(390) * np.sum(r1 ** 3) / s2 ** 1.5) if s2 > 0 else np.nan,
        "close_1259": float(day.loc[day.minute == 389, "close"].iloc[0]) if 389 in m else np.nan,
        "open_0630": float(day.loc[day.minute == 0, "open"].iloc[0]) if 0 in m else np.nan,
        "close_1245": float(day.loc[day.minute == 375, "close"].iloc[0]) if 375 in m else np.nan,
        "volume": float(day.volume.sum()),
    }


def trailing_rank(dates: list, vals: np.ndarray) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    for i, v in enumerate(vals):
        prior = vals[max(0, i - RANK_WINDOW):i]
        if len(prior) >= RANK_MIN:
            out[i] = ((prior < v).sum() + 0.5 * (prior == v).sum()) / len(prior)
    return out


def decile_of(rank: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(rank), np.nan, np.minimum(np.floor(10 * rank) + 1, 10))


def main() -> None:
    conn = Q.connect()
    summ, prof = Q.load_bk()
    es = Q.load_es(conn)
    counts = Q.bar_counts(es)
    print(f"es_minutes RTH rows {len(es)} on {es.trade_date.nunique()} dates", flush=True)

    ES = pd.DataFrame({d: day_stats(g) for d, g in es.groupby("trade_date")}).T
    ES.index = list(ES.index)
    ok = {d: Q.es_day_ok(d, counts) for d in ES.index}

    # ── G history over every CR-BK-usable session (A1.2) ──────────────────────
    h = summ[summ.usable.astype(bool)].sort_values("d").reset_index(drop=True)
    pgrid = {d: (g.price.to_numpy(), g.gex_total.to_numpy()) for d, g in prof.groupby("d")}

    def g_b(d, s0):
        p = Q.prev_session(d)
        if d not in ES.index or p not in ES.index:
            return np.nan
        o, c = ES.at[d, "open_0630"], ES.at[p, "close_1245"]
        if not (np.isfinite(o) and np.isfinite(c)):
            return np.nan
        x = s0 * o / c
        gp, gv = pgrid[d]
        return float(np.interp(x, gp, gv)) if gp.min() <= x <= gp.max() else np.nan

    h["G"] = h.net_at_s0_linear
    h["G_B"] = [g_b(d, s0) for d, s0 in zip(h.d, h.S0)]
    h["rank"] = trailing_rank(list(h.d), h.G.to_numpy())
    hb = h[h.G_B.notna()].copy()
    hb["rank_B"] = trailing_rank(list(hb.d), hb.G_B.to_numpy())
    h = h.merge(hb[["d", "rank_B"]], on="d", how="left")

    # ── L1 universe ──────────────────────────────────────────────────────────
    u, steps = Q.universe(summ, counts)
    assert len(u) == 641, len(u)
    U = u[["d", "S0"]].merge(h[["d", "G", "G_B", "rank", "rank_B"]], on="d", how="left")
    U["decile"] = decile_of(U["rank"].to_numpy())
    U["decile_B"] = decile_of(U["rank_B"].to_numpy())
    U["neg"] = U.G < 0
    U["neg_B"] = np.where(U.G_B.notna(), U.G_B < 0, np.nan)
    U["decile_raw"] = pd.qcut(U.G, 10, labels=False) + 1

    # ADV: mean RTH contract volume over the 20 prior ES-ok sessions
    ok_days = sorted(d for d, v in ok.items() if v)
    vol = ES.volume
    adv = []
    for d in U.d:
        prior = [x for x in ok_days if x < d][-ADV_N:]
        adv.append(float(np.mean([vol[x] for x in prior])) if len(prior) == ADV_N else np.nan)
    U["adv"] = adv
    U["G_adv"] = U.G / U.adv
    U["decile_adv"] = np.nan
    m = U.G_adv.notna()
    U.loc[m, "decile_adv"] = pd.qcut(U.loc[m, "G_adv"], 10, labels=False) + 1

    # per-day ES statistics
    cols = ["n_bars", "rho5", "VR"] + [f"r60_{i}" for i in range(1, 7)] + ["r_first30", "r_last30", "rv", "skew"]
    for c in cols:
        U[c] = [float(ES.at[d, c]) for d in U.d]
    prev = [Q.prev_session(d) for d in U.d]
    U["prev_d"] = prev
    U["prev_close"] = [float(ES.at[p, "close_1259"]) if p in ES.index else np.nan for p in prev]
    U["r_rod"] = [ES.at[d, "lp360"] - np.log(c) if np.isfinite(c) else np.nan for d, c in zip(U.d, U.prev_close)]
    U["r_gao"] = [ES.at[d, "lp30"] - np.log(c) if np.isfinite(c) else np.nan for d, c in zip(U.d, U.prev_close)]
    U["prev_ok"] = [bool(ok.get(p, False)) for p in prev]
    U["rho5_prev"] = [float(ES.at[p, "rho5"]) if ok.get(p, False) else np.nan for p in prev]
    U["r_last30_prev"] = [float(ES.at[p, "r_last30"]) if ok.get(p, False) else np.nan for p in prev]
    U["rv_tercile"] = pd.qcut(U.rv, 3, labels=False) + 1
    U["dow"] = [d.weekday() for d in U.d]
    U["year"] = [d.year for d in U.d]
    U["source"] = [r[0] for r in (conn.execute(
        "SELECT min(source) FROM es_minutes WHERE is_rth AND trade_date=%s", (d,)).fetchone() for d in U.d)]

    Q.OUT_DIR.mkdir(parents=True, exist_ok=True)
    U.to_parquet(Q.OUT_DIR / "days.parquet")

    # counts and sanity only — no decile-level statistic
    print(f"\nuniverse {len(U)}; trailing-rank decile defined on {int(U.decile.notna().sum())} "
          f"(first ranked day {U.loc[U.decile.notna(), 'd'].min()}; no decile on {int(U.decile.isna().sum())})")
    print(f"B computable {int(U.G_B.notna().sum())}; B decile defined {int(U.decile_B.notna().sum())}")
    print(f"ADV deciles defined {int(U.decile_adv.notna().sum())}; rROD defined {int(U.r_rod.notna().sum())}; "
          f"prev-session ok {int(U.prev_ok.sum())}")
    print(f"G < 0: {int(U.neg.sum())} of {len(U)}")
    print("decile counts:", U.decile.value_counts().sort_index().to_dict())
    big = U.r_rod.abs() > 0.05
    print(f"|rROD| > 5 %: {int(big.sum())} {[str(d) for d in U.loc[big, 'd']]}")
    print(f"saved {Q.OUT_DIR / 'days.parquet'}")


if __name__ == "__main__":
    main()
