#!/usr/bin/env python3
"""CR-BK — corrected gamma profile (test-only, READ-ONLY, train only). Spec: specs/CR-BK-corrected-gamma-profile-test.md.

Full repricing (L2), SPX cash (L3), sticky strike (L4), net = call − |put| (L5), grid S0 ± 300 step 5 (L6),
amended by A1: S0 = stock_price of the nearest live expiry (ORATS 15:46 ET snapshot of the prior session);
IV per strike from the orats_monies_minute call-delta smile at the latest snapshot ≤ 12:46 PT of the prior
session; forward = the row's per-expiry stock_price, shifted proportionally on the grid (F_e(S) = S_e · S / S0);
wings linear in log-strike beyond Δ5/Δ95 (primary) or flat (robustness diagnostic only); holiday-stamped
orats_oi_gamma store dates moved to next_trading_day.

Commands:
  validate  — gate 2 re-validation (A1.3): exposure-weighted median |gamma error| on a fresh 20-date sample.
  shape     — gate 6 shape report (A1.8) + gate 7 counts; saves profiles to scripts/logs/cr_bk/ (git-ignored).
No outcome is computed here.

Usage: apps/web/.venv/bin/python scripts/cr_bk_profile.py validate|shape
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.gex_landscape import find_walls  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day  # noqa: E402

SPLIT = dt.date(2026, 6, 5)
FIRST = dt.date(2023, 5, 2)            # monies start 2023-05-01 → first store date with a prior-session smile
GRID_HALF, GRID_STEP = 300.0, 5.0
SNAP_PT = dt.time(12, 46)
EXPIRY_PT = dt.time(13, 0)
DELTAS = list(range(5, 100, 5))
SIG_FLOOR, SIG_CAP = 0.03, 3.0
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bk"
STEP0_SEED = 20261011                  # Step-0 sample (excluded from the re-validation)
REVAL_SEED = 20261012


def connect():
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    conn.execute("SET statement_timeout = '300s'")
    return conn


def effective_dates(conn) -> dict[dt.date, dt.date]:
    """{effective trading date: orats_oi_gamma store date}; holiday-stamped store dates → next trading day (A1.6)."""
    out = {}
    for (s,) in conn.execute("SELECT DISTINCT trade_date FROM orats_oi_gamma WHERE ticker='SPX' "
                             "AND trade_date BETWEEN %s AND %s ORDER BY 1", (FIRST, SPLIT)).fetchall():
        eff = s if is_trading_day(s) else next_trading_day(s)
        if eff <= SPLIT:
            assert eff not in out, f"two store dates map to {eff}"
            out[eff] = s
    return out


def load_day(conn, eff: dt.date, store: dt.date) -> dict | None:
    """OI rows live on `eff`, the 12:46 PT prior-session smile, S0. None if the smile snapshot is missing."""
    api = prev_trading_day(eff)
    rows = pd.DataFrame(conn.execute(
        """SELECT expir_date, strike::float8, stock_price::float8, gamma, coalesce(call_oi, 0), coalesce(put_oi, 0)
           FROM orats_oi_gamma WHERE ticker='SPX' AND trade_date=%s AND expir_date >= %s
             AND (coalesce(call_oi, 0) > 0 OR coalesce(put_oi, 0) > 0)""", (store, eff)).fetchall(),
        columns=["exp", "K", "S_e", "gamma", "coi", "poi"])
    if rows.empty:
        return None
    snap = conn.execute("SELECT max(snapshot_pt) FROM orats_monies_minute WHERE trade_date=%s AND ticker='SPX' "
                        "AND snapshot_pt <= %s", (api.isoformat(), dt.datetime.combine(api, SNAP_PT))).fetchone()[0]
    if snap is None or snap < dt.datetime.combine(api, dt.time(12, 30)):
        return None
    mon = conn.execute(f"""SELECT expir_date, expiry_tod, {",".join(f"vol{d}" for d in DELTAS)}
                           FROM orats_monies_minute WHERE trade_date=%s AND snapshot_pt=%s AND ticker='SPX'""",
                       (api.isoformat(), snap)).fetchall()
    smiles = {dt.date.fromisoformat(r[0]): (r[1], np.array(r[2:], dtype=float)) for r in mon}
    s0 = float(rows.loc[rows.exp == rows.exp.min(), "S_e"].iloc[0])
    t0 = dt.datetime.combine(api, SNAP_PT)
    rows["T"] = [((dt.datetime.combine(e, EXPIRY_PT) - t0).total_seconds() / 86400) / 365.0 for e in rows.exp]
    return {"eff": eff, "store": store, "api": api, "snap": snap, "rows": rows, "smiles": smiles, "S0": s0}


def attach_iv(day: dict, wing: str) -> pd.DataFrame:
    """Per-row sticky-strike IV from the expiry's delta smile; rows without a usable smile are dropped."""
    parts = []
    for e, g in day["rows"].groupby("exp"):
        sm = day["smiles"].get(e)
        T = g["T"].iloc[0]
        if sm is None or T <= 0:
            continue
        tod, vols = sm
        ok = np.isfinite(vols) & (vols > 0)
        if ok.sum() < 3:
            continue
        F = g["S_e"].iloc[0]
        v = vols[ok]
        d1 = norm.ppf(np.array(DELTAS, dtype=float)[ok] / 100.0)
        lks = np.log(F) - d1 * v * np.sqrt(T) + 0.5 * v * v * T          # log-strike of each delta point
        o = np.argsort(lks)
        lks, v = lks[o], v[o]
        lk = np.log(g["K"].to_numpy())
        sig = np.interp(lk, lks, v)                                       # flat beyond the end points
        if wing == "linear":
            lo, hi = lk < lks[0], lk > lks[-1]
            sig[lo] = v[0] + (lk[lo] - lks[0]) * (v[1] - v[0]) / (lks[1] - lks[0])
            sig[hi] = v[-1] + (lk[hi] - lks[-1]) * (v[-1] - v[-2]) / (lks[-1] - lks[-2])
        parts.append(g.assign(sig=np.clip(sig, SIG_FLOOR, SIG_CAP), am=(tod == "am")))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def unit_gamma(F, K, sig, T):
    """Black gamma per unit underlying at forward F (the per-expiry stock_price convention of ORATS)."""
    sq = sig * np.sqrt(T)
    return norm.pdf((np.log(F / K) + 0.5 * sq * sq) / sq) / (F * sq)


def profile(rows: pd.DataFrame, s0: float) -> pd.DataFrame:
    """Net dealer gamma exposure (call − |put|) at every grid price, full repricing, sticky strike."""
    grid = np.arange(s0 - GRID_HALF, s0 + GRID_HALF + GRID_STEP / 2, GRID_STEP)
    K, Se, sig, T = (rows[c].to_numpy(dtype=float) for c in ("K", "S_e", "sig", "T"))
    net_oi = rows["coi"].to_numpy(dtype=float) - rows["poi"].to_numpy(dtype=float)
    out = np.empty(len(grid))
    for i, S in enumerate(grid):
        F = Se * S / s0
        out[i] = np.sum(unit_gamma(F, K, sig, T) * F * F * 100.0 * net_oi)
    return pd.DataFrame({"price": grid, "gex_total": out})


def shape_of(prof: pd.DataFrame, s0: float) -> dict:
    w = find_walls(prof, min_prominence_pct=0.10)
    pos, neg = w[w.sign > 0], w[w.sign < 0]
    g, p = prof.gex_total.to_numpy(), prof.price.to_numpy()
    flips = [p[i] - g[i] * (p[i + 1] - p[i]) / (g[i + 1] - g[i]) for i in range(len(g) - 1) if g[i] * g[i + 1] < 0]
    return {"n_pos": len(pos), "n_neg": len(neg),
            "top_pos": float(pos.loc[pos.gex.idxmax(), "price"]) if len(pos) else np.nan,
            "pos_prices": [float(x) for x in pos.price], "n_flip": len(flips),
            "flip_nearest": float(min(flips, key=lambda x: abs(x - s0))) if flips else np.nan,
            "net_at_s0": float(np.interp(s0, p, g))}


# ── gate 2 ──────────────────────────────────────────────────────────────────
def cmd_validate() -> None:
    conn = connect()
    effmap = effective_dates(conn)
    store_dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT trade_date FROM orats_oi_gamma WHERE ticker='SPX' AND trade_date BETWEEN %s AND %s ORDER BY 1",
        (FIRST, SPLIT)).fetchall()]
    step0 = sorted(np.random.default_rng(STEP0_SEED).choice(store_dates, 20, replace=False))
    used = {s if is_trading_day(s) else next_trading_day(s) for s in step0}
    pool = sorted(d for d in effmap if d not in used)
    sample = sorted(np.random.default_rng(REVAL_SEED).choice(pool, 20, replace=False))
    print(f"excluded Step-0 dates (effective): {[str(d) for d in sorted(used)]}")
    print(f"fresh sample (seed {REVAL_SEED}): {[str(d) for d in sample]}", flush=True)
    frames = []
    for d in sample:
        day = load_day(conn, d, effmap[d])
        if day is None:
            print(f"  {d}: no 12:46 smile — skipped"); continue
        r = attach_iv(day, "linear")
        r = r[r.gamma > 0]
        g = unit_gamma(r.S_e.to_numpy(), r.K.to_numpy(), r.sig.to_numpy(), r["T"].to_numpy())
        frames.append(pd.DataFrame({"d": d, "pe": np.abs(g / r.gamma.to_numpy() - 1),
                                    "w": r.gamma.to_numpy() * (r.coi + r.poi).to_numpy() * r.S_e.to_numpy() ** 2,
                                    "dte": [(e - d).days for e in r.exp]}))
    x = pd.concat(frames, ignore_index=True)

    def wq(s: pd.DataFrame, qv: float) -> float:
        o = s.sort_values("pe"); c = o.w.cumsum() / o.w.sum()
        return float(o.pe.iloc[min(c.searchsorted(qv), len(o) - 1)])
    wmed = wq(x, .5)
    print(f"\nrows {len(x)} on {x.d.nunique()} dates | exposure-weighted median {100 * wmed:.2f} %  p90 {100 * wq(x, .9):.2f} % "
          f"| row median {100 * x.pe.median():.2f} % (info)")
    for lo, hi in ((-1, 0), (0, 7), (7, 30), (30, 90), (90, 400)):
        s = x[(x.dte > lo) & (x.dte <= hi)]
        print(f"  DTE ({lo},{hi}]: rows {len(s)}, share {100 * s.w.sum() / x.w.sum():.1f} %, weighted median {100 * wq(s, .5):.2f} %")
    print(f"\nGATE 2 (exposure-weighted median ≤ 5 %): {'PASS' if wmed <= 0.05 else 'FAIL'}")


# ── gate 6 / 7 ──────────────────────────────────────────────────────────────
def cmd_shape() -> None:
    from packages.shared.spx_cash import build_series, daily_ohlc, fetch_minutes
    from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION

    conn = connect()
    effmap = effective_dates(conn)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs, profs = [], []
    for i, (d, s) in enumerate(sorted(effmap.items())):
        day = load_day(conn, d, s)
        if day is None:
            recs.append({"d": d, "usable": False}); continue
        rec = {"d": d, "usable": True, "S0": day["S0"], "remapped": d != s, "snap": day["snap"]}
        for wing in ("linear", "flat"):
            r = attach_iv(day, wing)
            pr = profile(r, day["S0"])
            sh = shape_of(pr, day["S0"])
            rec.update({f"{k}_{wing}": v for k, v in sh.items()})
            if wing == "linear":
                tot = (day["rows"].gamma.fillna(0) * (day["rows"].coi + day["rows"].poi) * day["rows"].S_e ** 2).sum()
                rec["exposure_share_without_smile"] = 1 - (r.gamma.fillna(0) * (r.coi + r.poi) * r.S_e ** 2).sum() / tot
                rec["am_rows"] = int(r.am.sum())
            profs.append(pr.assign(d=d, wing=wing))
        recs.append(rec)
        if i % 50 == 0:
            print(f"  {i}/{len(effmap)} {d}", flush=True)
    summ = pd.DataFrame(recs)
    pd.concat(profs, ignore_index=True).to_parquet(OUT_DIR / "profiles.parquet")
    summ.to_parquet(OUT_DIR / "shape_summary.parquet")

    u = summ[summ.usable].copy()
    L = "linear"
    print(f"\nEffective train dates {len(summ)} (remapped holiday store dates {int(u.remapped.sum())}); usable (12:46 smile) {len(u)}")
    print(f"Exposure without a smile (dropped): median {100 * u.exposure_share_without_smile.median():.2f} %, "
          f"max {100 * u.exposure_share_without_smile.max():.2f} %; AM-flagged rows per day median {u.am_rows.median():.0f}")
    print("\n## Gate 6 — shape (linear wings, find_walls prominence 10 %)")
    print(pd.crosstab(u[f"n_pos_{L}"].clip(upper=5), u[f"n_neg_{L}"].clip(upper=5), margins=True).to_string())
    single = ((u[f"n_pos_{L}"] == 1) & (u[f"n_neg_{L}"] == 0)).mean()
    print(f"days with exactly one positive peak and no trough: {100 * single:.1f} %  → stop rule (> 80 %): {'STOP' if single > .8 else 'pass'}")
    print(f"days with ≥ 1 positive peak: {100 * (u[f'n_pos_{L}'] >= 1).mean():.1f} %; ≥ 1 trough: {100 * (u[f'n_neg_{L}'] >= 1).mean():.1f} %")
    print(f"zero-gamma flip inside the grid: {100 * (u[f'n_flip_{L}'] >= 1).mean():.1f} % of days "
          f"(≥ 2 flips: {100 * (u[f'n_flip_{L}'] >= 2).mean():.1f} %); net gamma at S0 < 0 on {100 * (u[f'net_at_s0_{L}'] < 0).mean():.1f} %")
    both = u[u.top_pos_linear.notna() & u.top_pos_flat.notna()]
    moved = (both.top_pos_linear - both.top_pos_flat).abs() > 10
    print(f"\n## Wing robustness (A1.4): top positive peak moves > 10 pts linear vs flat on {100 * moved.mean():.1f} % of "
          f"{len(both)} days (median |shift| {(both.top_pos_linear - both.top_pos_flat).abs().median():.1f} pts)")
    print(f"top positive peak − S0 (linear): median {(u.top_pos_linear - u.S0).median():+.1f}, "
          f"p10 {(u.top_pos_linear - u.S0).quantile(.1):+.1f}, p90 {(u.top_pos_linear - u.S0).quantile(.9):+.1f} pts")

    # gate 7 — counts need the SPX open (not an outcome) and IM > 0
    clean, _ = build_series(fetch_minutes(conn, dt.date(2023, 5, 1), SPLIT))
    spx = daily_ohlc(clean)
    assert max(spx.index) <= SPLIT
    sess = list(spx.index); pos = {d: i for i, d in enumerate(sess)}
    im = {r[0]: r[1] for r in conn.execute(
        "SELECT trade_date, (feature_vector->>'implied_move_1d')::float FROM bt_daily_features WHERE ticker='SPX' "
        "AND active AND feature_version=%s AND trade_date <= %s", (CANONICAL_FEATURE_VERSION, SPLIT)).fetchall()}
    v = u[[d in pos and (im.get(d) or 0) > 0 for d in u.d]].copy()
    v["open"] = [float(spx.loc[d, "open"]) for d in v.d]
    print(f"\n## Gate 7 — usable counts (SPX session + IM > 0): {len(v)}")
    for h in (5, 20):
        w = v[[pos[d] + h - 1 < len(sess) for d in v.d]]
        above = (w.top_pos_linear > w.open).sum(); below = (w.top_pos_linear < w.open).sum()
        print(f"  T1 {h}-session: top peak above open {above}, below open {below} (window inside train: {len(w)})")
    t2 = sum(any(p < o for p in ps) and any(p > o for p in ps) for ps, o in zip(v.pos_prices_linear, v.open))
    print(f"  T2: positive peak on both sides of the open: {t2} of {len(v)}")
    print(f"  T3: dates with ≥ 120 prior usable sessions: {max(len(v) - 120, 0)}")
    print(f"\nsaved: {OUT_DIR / 'profiles.parquet'}, {OUT_DIR / 'shape_summary.parquet'}")


if __name__ == "__main__":
    {"validate": cmd_validate, "shape": cmd_shape}[sys.argv[1]]()
