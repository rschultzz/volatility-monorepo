#!/usr/bin/env python3
"""CR-BL T2 / T3 — vol state with CR-BK's corrected gamma. READ-ONLY.

Pre-registered in specs/CR-BL-vol-state-range-labels.md ("T2 / T3 pre-registration", A2) before any
join of gamma to labels. Modes, in order:

  --control  rebuild CR-BK's T3 frame and rank exactly as cr_bk_placebo.py (df8307c) did and compare
             with CR-BK's committed T3 numbers; mismatch → exit non-zero, nothing else computed.
  --counts   control, then population / cell n per horizon (label non-NULL flags only, no values).
  --run      control, counts, then T2 and T3 once. Writes scripts/logs/cr_bl_t2t3.md.

Gamma X (A2.2, verbatim CR-BK x_at): np.interp(open_d, price, gex_total) on the linear-wing profile of d
from CR-BK's git-ignored profiles.parquet (read in place, never modified); NaN if no profile / off-grid.
CR-BL rank (A2.3): over NYSE sessions, 100 × share of X among the 250 prior sessions (X defined) <= X_d,
>= 120 values, d excluded. Train only throughout; the label train filter is in the SQL WHERE.

Usage: apps/web/.venv/bin/python scripts/cr_bl_t2t3.py --control | --counts | --run
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from cr_bl_t1 import group_diff  # noqa: E402  (T1's month-cluster bootstrap, same seed / resamples)
from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.spx_cash import build_series, daily_ohlc, fetch_minutes  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

SPLIT = dt.date(2026, 6, 5)
VERSION = "cr-bl-v1"
SEED, N_BOOT = 20260920, 10_000
PROFILES = Path("/Users/ryan/code/volatility-monorepo/scripts/logs/cr_bk/profiles.parquet")  # CR-BK output
BK_FEATURE_VERSION = "v0.6.0-openiv"   # CR-BK / CR-BI harness base
BK_WINDOW, BK_MIN = 250, 120
PCT_WINDOW, PCT_MIN = 250, 120
LOW, HIGH = 33.0, 67.0
PASS_DELTA, MIN_CELL = 0.08, 40
OUT = REPO_ROOT / "scripts" / "logs" / "cr_bl_t2t3.md"

# CR-BK committed T3 (corrected) at df8307c, printed precision.
BK_EXPECTED = {
    "n": 617, "first": "2023-10-30", "missing": 4,
    "spear_y1": "-0.554 [-0.629, -0.454]", "spear_y2": "-0.115 [-0.200, -0.025]",
    "top": "257 0.683 1.013", "bot": "182 1.404 1.111",
    "diff_y1": "-0.721 [-1.119, -0.435]", "diff_y2": "-0.098 [-0.202, +0.005]",
}

_LABELS_FLAGS_SQL = """
    SELECT l.trade_date, l.horizon_sessions, (l.abs_close_over_implied IS NOT NULL) AS has_label, v.iv_7d_pct
    FROM bt_range_labels_daily l
    JOIN bt_vol_state_daily v ON v.ticker = l.ticker AND v.trade_date = l.trade_date AND v.version = l.version
    WHERE l.version = %(v)s AND l.ticker = 'SPX' AND l.trade_date <= %(split)s AND l.horizon_end_date <= %(split)s
"""
_LABELS_SQL = """
    SELECT l.trade_date, l.horizon_sessions, l.range_over_implied, l.abs_close_over_implied, v.iv_7d_pct
    FROM bt_range_labels_daily l
    JOIN bt_vol_state_daily v ON v.ticker = l.ticker AND v.trade_date = l.trade_date AND v.version = l.version
    WHERE l.version = %(v)s AND l.ticker = 'SPX' AND l.trade_date <= %(split)s AND l.horizon_end_date <= %(split)s
"""


# ── gamma X (verbatim CR-BK x_at) ────────────────────────────────────────────
def load_grids() -> dict:
    prof = pd.read_parquet(PROFILES)
    prof = prof[prof.wing == "linear"]
    return {dt.date.fromisoformat(str(d)): (g.price.to_numpy(), g.gex_total.to_numpy()) for d, g in prof.groupby("d")}


def x_at(grids, d, o):
    if d not in grids:
        return np.nan
    p, g = grids[d]
    return float(np.interp(o, p, g)) if p.min() <= o <= p.max() else np.nan


# ── control: CR-BK T3 as built ───────────────────────────────────────────────
def bk_pct_rank(x: pd.Series) -> pd.Series:
    vals, out = x.to_numpy(), []
    for i, v in enumerate(vals):
        prior = vals[max(0, i - BK_WINDOW):i]
        out.append(np.nan if len(prior) < BK_MIN else ((prior < v).sum() + 0.5 * (prior == v).sum()) / len(prior))
    return pd.Series(out, index=x.index)


def bk_boot_stat(df: pd.DataFrame, stat) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    months = df.month.unique()
    groups = {m: g for m, g in df.groupby("month")}
    idx = rng.integers(0, len(months), size=(N_BOOT, len(months)))
    est = np.array([stat(pd.concat([groups[months[j]] for j in row], ignore_index=True)) for row in idx])
    est = est[np.isfinite(est)]
    return float(stat(df)), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def control(conn, spx: pd.DataFrame, grids: dict, say) -> None:
    feats = conn.execute(
        """SELECT f.trade_date, f.regime_at_classification, (f.feature_vector->>'implied_move_1d')::float,
                  o.horizon_sessions
           FROM bt_daily_features f LEFT JOIN bt_daily_outcomes o
             ON o.ticker=f.ticker AND o.trade_date=f.trade_date AND o.feature_version=f.feature_version AND o.active
           WHERE f.ticker='SPX' AND f.feature_version=%s AND f.active AND f.trade_date <= %s ORDER BY 1""",
        (BK_FEATURE_VERSION, SPLIT)).fetchall()
    sess = list(spx.index)
    pos = {d: i for i, d in enumerate(sess)}
    O, H, L = spx["open"].to_numpy(), spx["high"].to_numpy(), spx["low"].to_numpy()
    base = pd.DataFrame([{"d": d, "im": im} for d, _r, im, _hz in feats if im and im > 0 and d in pos])
    base["open"] = [O[pos[d]] for d in base.d]
    base["month"] = [f"{d.year}-{d.month:02d}" for d in base.d]
    base["y1"] = [(H[pos[d]] - L[pos[d]]) / o for d, o in zip(base.d, base.open)]
    base["y2"] = [(H[pos[d]] - L[pos[d]]) / im for d, im in zip(base.d, base.im)]
    s = base.copy()
    s["x"] = [x_at(grids, d, o) for d, o in zip(s.d, s.open)]
    missing = int(s.x.isna().sum())
    s = s[s.x.notna()].sort_values("d").reset_index(drop=True)
    s["rank"] = bk_pct_rank(s.x)
    s = s[s["rank"].notna()].copy()

    def spear(col):
        return lambda x: spearmanr(x["rank"], x[col]).statistic

    def tdiff(col):
        return lambda x: x.loc[x["rank"] > 2 / 3, col].mean() - x.loc[x["rank"] < 1 / 3, col].mean()

    r1, r2 = bk_boot_stat(s, spear("y1")), bk_boot_stat(s, spear("y2"))
    top, bot = s[s["rank"] > 2 / 3], s[s["rank"] < 1 / 3]
    d1, d2 = bk_boot_stat(s, tdiff("y1")), bk_boot_stat(s, tdiff("y2"))
    def fmt(v):
        a, lo, hi = v
        return f"{a:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    got = {
        "n": len(s), "first": str(s.d.min()), "missing": missing,
        "spear_y1": fmt(r1), "spear_y2": fmt(r2),
        "top": f"{len(top)} {100 * top.y1.mean():.3f} {top.y2.mean():.3f}",
        "bot": f"{len(bot)} {100 * bot.y1.mean():.3f} {bot.y2.mean():.3f}",
        "diff_y1": fmt(tuple(100 * x for x in d1)), "diff_y2": fmt(d2),
    }
    say("## Control (A2.4) — CR-BK T3 rebuilt from profiles.parquet under CR-BK's own rank rule\n")
    say("| item | CR-BK committed (df8307c) | rebuilt | match |")
    say("|---|---|---|---|")
    ok = True
    for k, exp in BK_EXPECTED.items():
        m = str(exp) == str(got[k])
        ok &= m
        say(f"| {k} | {exp} | {got[k]} | {'✓' if m else '✗'} |")
    say()
    if not ok:
        say("**CONTROL FAILED — stopping (A2.4). No CR-BL gamma rank computed.**")
        flush(say)
        sys.exit(2)
    say("**Control: PASS.**\n")


# ── CR-BL gamma rank (A2.3) ──────────────────────────────────────────────────
def gamma_pct(spx: pd.DataFrame, grids: dict) -> pd.DataFrame:
    sess = [d.date() for d in pd.date_range(dt.date(2023, 5, 1), SPLIT) if is_trading_day(d.date())]
    x = np.array([x_at(grids, d, float(spx.at[d, "open"])) if d in spx.index else np.nan for d in sess])
    pct = np.full(len(x), np.nan)
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        w = x[max(0, i - PCT_WINDOW):i]
        w = w[~np.isnan(w)]
        if len(w) >= PCT_MIN:
            pct[i] = 100.0 * float((w <= v).mean())
    return pd.DataFrame({"trade_date": sess, "gamma_x": x, "gamma_pct": pct})


def cells(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.iv_7d_pct.notna() & df.gamma_pct.notna()].copy()
    df["t2"] = (df.iv_7d_pct <= LOW) & (df.gamma_pct <= LOW)
    df["t3"] = (df.iv_7d_pct >= HIGH) & (df.gamma_pct >= HIGH)
    return df


_LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def flush(_say=None) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_LOG) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--control", action="store_true")
    g.add_argument("--counts", action="store_true")
    g.add_argument("--run", action="store_true")
    args = ap.parse_args()

    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    say(f"# CR-BL T2 / T3 (read-only; {'run' if args.run else 'counts' if args.counts else 'control'}; "
        f"{dt.datetime.now():%Y-%m-%d %H:%M})\n")
    clean, _ = build_series(fetch_minutes(conn, dt.date(2023, 5, 1), SPLIT))
    spx = daily_ohlc(clean)[["open", "high", "low", "close"]]
    assert max(spx.index) <= SPLIT
    grids = load_grids()
    say(f"Profiles: {len(grids)} dates ({min(grids)} → {max(grids)}), linear wing. SPX sessions: {len(spx)}.\n")

    control(conn, spx, grids, say)
    if args.control:
        flush()
        return

    gp = gamma_pct(spx, grids)
    say(f"CR-BL gamma rank (A2.3): X defined on {int(gp.gamma_x.notna().sum())} of {len(gp)} sessions; "
        f"gamma_pct from {gp[gp.gamma_pct.notna()].trade_date.min()} ({int(gp.gamma_pct.notna().sum())} sessions).\n")

    fl = pd.DataFrame(conn.execute(_LABELS_FLAGS_SQL, {"v": VERSION, "split": SPLIT}).fetchall(),
                      columns=["trade_date", "h", "has_label", "iv_7d_pct"])
    fl = cells(fl[fl.has_label].merge(gp, on="trade_date", how="left"))
    say("## Cell counts (label presence only, before any outcome)\n")
    say("| h | population | T2 cell (iv ≤ 33 & gamma ≤ 33) | T3 cell (iv ≥ 67 & gamma ≥ 67) |")
    say("|---|---|---|---|")
    counts = {}
    for h in (1, 5):
        f = fl[fl.h == h]
        counts[h] = (len(f), int(f.t2.sum()), int(f.t3.sum()))
        say(f"| {h} | {counts[h][0]} | {counts[h][1]} | {counts[h][2]} |")
    say()
    t2_h = [h for h in (1, 5) if counts[h][1] >= MIN_CELL]
    t3_ok = counts[1][2] >= MIN_CELL
    say(f"T2 evaluable at: {t2_h or 'none → T2 NOT RUN'}; T3 (h1): {'evaluable' if t3_ok else 'n < 40 → T3 NOT RUN'}.\n")
    if not args.run:
        flush()
        return

    lb = pd.DataFrame(conn.execute(_LABELS_SQL, {"v": VERSION, "split": SPLIT}).fetchall(),
                      columns=["trade_date", "h", "range_oi", "abs_close_oi", "iv_7d_pct"])
    assert lb.trade_date.max() <= SPLIT
    lb = cells(lb[lb.abs_close_oi.notna()].merge(gp, on="trade_date", how="left"))
    lb["month"] = [f"{d.year}-{d.month:02d}" for d in lb.trade_date]
    lb["beyond"] = (lb.abs_close_oi > 1).astype(float)
    lb["inside_half"] = (lb.abs_close_oi < 0.5).astype(float)

    def row(d, cell, y, label):
        a, b = d[cell], ~d[cell]
        p = group_diff(d.reset_index(drop=True), y, a.reset_index(drop=True), b.reset_index(drop=True))
        scale = 100 if y != "range_oi" else 1
        unit = " pts" if scale == 100 else ""
        return p, (f"{label} | {scale * d.loc[a, y].mean():.{1 if scale == 100 else 3}f} | "
                   f"{scale * d.loc[b, y].mean():.{1 if scale == 100 else 3}f} | "
                   f"{scale * p[0]:+.{1 if scale == 100 else 3}f}{unit} [{scale * p[1]:+.{1 if scale == 100 else 3}f}, "
                   f"{scale * p[2]:+.{1 if scale == 100 else 3}f}]")

    verdict = {}
    for test, cell, prim in (("T2", "t2", "beyond"), ("T3", "t3", "inside_half")):
        say(f"## {test} — cell `{cell}` vs all other days\n")
        say("| h | n cell / other | statistic | cell | other | Δ [95 % CI] |")
        say("|---|---|---|---|---|---|")
        for h in (1, 5):
            d = lb[lb.h == h]
            nc, no = int(d[cell].sum()), int((~d[cell]).sum())
            for y, lab in ((prim, f"P({'>1' if prim == 'beyond' else '<0.5'}) **primary**" if (test == "T2" or h == 1)
                                   else "P(<0.5) (h5, reported)"),
                           ("inside_half" if prim == "beyond" else "beyond",
                            "P(<0.5)" if prim == "beyond" else "P(>1)"),
                           ("range_oi", "mean R/I")):
                p, txt = row(d, cell, y, lab)
                say(f"| {h} | {nc} / {no} | {txt} |")
                if y == prim:
                    verdict[(test, h)] = (nc, p)
        say()

    say("## Verdict (locked criteria)\n")
    t2_pass = False
    for h in (1, 5):
        nc, p = verdict[("T2", h)]
        if nc < MIN_CELL:
            say(f"- T2 h{h}: cell n {nc} < 40 → not evaluated")
            continue
        ok = p[0] >= PASS_DELTA and p[1] > 0
        t2_pass |= ok
        say(f"- T2 h{h}: Δ P(>1) {100 * p[0]:+.1f} pts, CI [{100 * p[1]:+.1f}, {100 * p[2]:+.1f}], n {nc} → {'PASS' if ok else 'fail'}")
    nc, p = verdict[("T3", 1)]
    if nc < MIN_CELL:
        t3_pass, t3_txt = False, f"cell n {nc} < 40 → NOT RUN"
    else:
        t3_pass = p[0] >= PASS_DELTA and p[1] > 0
        t3_txt = (f"Δ P(<0.5) {100 * p[0]:+.1f} pts, CI [{100 * p[1]:+.1f}, {100 * p[2]:+.1f}], n {nc} → "
                  f"{'PASS' if t3_pass else 'fail'}")
    say(f"- T3 h1: {t3_txt}")
    say(f"\n**T2 {'PASSES' if t2_pass else 'FAILS' if t2_h else 'NOT RUN'}; T3 "
        f"{'PASSES' if t3_pass else 'FAILS' if t3_ok else 'NOT RUN'}.**")
    flush()
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
