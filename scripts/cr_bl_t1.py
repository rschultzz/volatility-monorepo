#!/usr/bin/env python3
"""CR-BL T1 — vol state alone vs realized-over-implied. READ-ONLY. One run.

Pre-registered in specs/CR-BL-vol-state-range-labels.md ("T1 pre-registration") before this file
was run. Train only: the train filter is in the SQL WHERE (trade_date and horizon_end_date
<= 2026-06-05), so no holdout label row is fetched.

X = iv_7d_pct; cheap = X <= 33, rich = X >= 67. Primary: Δ P(abs_close_over_implied > 1),
cheap − rich, per horizon. CI: month-cluster bootstrap (calendar month of trade_date), 10 000
resamples, seed 20260920, 95 % percentile. Pass: Δ >= 0.08 and CI lower > 0 at h1 or h5.

Usage: apps/web/.venv/bin/python scripts/cr_bl_t1.py
Output: scripts/logs/cr_bl_t1.md
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

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402

SPLIT = dt.date(2026, 6, 5)
VERSION = "cr-bl-v1"
SEED, N_BOOT = 20260920, 10_000
CHEAP, RICH = 33.0, 67.0
PASS_DELTA = 0.08
OUT = REPO_ROOT / "scripts" / "logs" / "cr_bl_t1.md"

_SQL = """
    SELECT l.trade_date, l.horizon_sessions, l.horizon_end_date, l.range_over_implied, l.abs_close_over_implied,
           v.iv_7d_pct, v.vrp_7_pct, v.sessions_in_7d
    FROM bt_range_labels_daily l
    JOIN bt_vol_state_daily v ON v.ticker = l.ticker AND v.trade_date = l.trade_date AND v.version = l.version
    WHERE l.version = %(v)s AND l.ticker = 'SPX'
      AND l.trade_date <= %(split)s AND l.horizon_end_date <= %(split)s
"""


def boot_idx(n_months: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n_months, size=(N_BOOT, n_months))


def group_diff(df: pd.DataFrame, y: str, a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Point Δ = mean(y | a) − mean(y | b) and month-cluster bootstrap 95 % CI."""
    months = sorted(df["month"].unique())
    mi = {m: i for i, m in enumerate(months)}
    k = df["month"].map(mi).to_numpy()
    yv = df[y].astype(float).to_numpy()
    M = len(months)
    sa = np.bincount(k[a.to_numpy()], yv[a.to_numpy()], M); ca = np.bincount(k[a.to_numpy()], None, M)
    sb = np.bincount(k[b.to_numpy()], yv[b.to_numpy()], M); cb = np.bincount(k[b.to_numpy()], None, M)
    idx = boot_idx(M)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)
    est = est[~np.isnan(est)]
    point = yv[a.to_numpy()].mean() - yv[b.to_numpy()].mean()
    return float(point), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def spearman_ci(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, float]:
    months = sorted(df["month"].unique())
    groups = [df[df["month"] == m] for m in months]
    rho = df[x].corr(df[y], method="spearman")
    idx = boot_idx(len(months))
    ests = np.empty(N_BOOT)
    for j in range(N_BOOT):
        s = pd.concat([groups[i] for i in idx[j]])
        ests[j] = s[x].corr(s[y], method="spearman")
    return float(rho), float(np.quantile(ests, .025)), float(np.quantile(ests, .975))


def main() -> None:
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    rows = conn.execute(_SQL, {"v": VERSION, "split": SPLIT}).fetchall()
    df = pd.DataFrame(rows, columns=["trade_date", "h", "end", "range_oi", "abs_close_oi", "iv_7d_pct",
                                     "vrp_7_pct", "sessions_in_7d"])
    assert df["trade_date"].max() <= SPLIT and df["end"].max() <= SPLIT
    df["month"] = [f"{d.year}-{d.month:02d}" for d in df["trade_date"]]
    df["beyond"] = (df["abs_close_oi"] > 1).astype(float)
    df["inside_half"] = (df["abs_close_oi"] < 0.5).astype(float)

    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    say(f"# CR-BL T1 — vol state alone (read-only, one run; {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    say(f"Train rows fetched: {len(df)} (trade_date and horizon end ≤ {SPLIT}); version `{VERSION}`. "
        f"Cheap X ≤ {CHEAP:g}, rich X ≥ {RICH:g}. CI: month-cluster bootstrap, {N_BOOT} resamples, seed {SEED}.\n")

    verdict = {}
    for xname, primary in (("iv_7d_pct", True), ("vrp_7_pct", False)):
        say(f"## {'Primary' if primary else 'Secondary readout (not a criterion)'} — X = `{xname}`\n")
        say("| h | n (cheap / mid / rich) | P(>1) cheap | P(>1) rich | Δ P(>1) [95 % CI] | mean R/I cheap | mean R/I rich "
            "| Δ mean R/I [CI] | P(<0.5) cheap | P(<0.5) rich | Δ P(<0.5) [CI] | Spearman X vs R/I [CI] |")
        say("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in (1, 5):
            d = df[(df.h == h) & df[xname].notna() & df["abs_close_oi"].notna()].reset_index(drop=True)
            a, b = d[xname] <= CHEAP, d[xname] >= RICH
            n_mid = int((~a & ~b).sum())
            p1 = group_diff(d, "beyond", a, b)
            r1 = group_diff(d, "range_oi", a, b)
            q1 = group_diff(d, "inside_half", a, b)
            sp = spearman_ci(d, xname, "range_oi")
            say(f"| {h} | {int(a.sum())} / {n_mid} / {int(b.sum())} | {100 * d.beyond[a].mean():.1f} % | "
                f"{100 * d.beyond[b].mean():.1f} % | {100 * p1[0]:+.1f} pts [{100 * p1[1]:+.1f}, {100 * p1[2]:+.1f}] | "
                f"{d.range_oi[a].mean():.3f} | {d.range_oi[b].mean():.3f} | {r1[0]:+.3f} [{r1[1]:+.3f}, {r1[2]:+.3f}] | "
                f"{100 * d.inside_half[a].mean():.1f} % | {100 * d.inside_half[b].mean():.1f} % | "
                f"{100 * q1[0]:+.1f} pts [{100 * q1[1]:+.1f}, {100 * q1[2]:+.1f}] | "
                f"{sp[0]:+.3f} [{sp[1]:+.3f}, {sp[2]:+.3f}] |")
            if primary:
                verdict[h] = (p1[0] >= PASS_DELTA and p1[1] > 0, p1)
        say()

    say("## Descriptive — holiday-window days (`sessions_in_7d` = 4) by tercile of iv_7d_pct\n")
    for h in (1, 5):
        d = df[(df.h == h) & df.iv_7d_pct.notna() & df.abs_close_oi.notna()]
        t = pd.cut(d.iv_7d_pct, [-1, CHEAP, RICH - 1e-9, 101], labels=["cheap", "mid", "rich"])
        share = (d.sessions_in_7d == 4).groupby(t, observed=False).agg(["sum", "size"])
        say(f"- h{h}: " + ", ".join(f"{k} {int(r['sum'])}/{int(r['size'])}" for k, r in share.iterrows()))
    say()

    say("## Verdict (T1 pass criterion: Δ P(close beyond implied) ≥ 8 pts at either horizon, CI lower > 0)\n")
    for h, (ok, p) in verdict.items():
        say(f"- h{h}: Δ {100 * p[0]:+.1f} pts, CI [{100 * p[1]:+.1f}, {100 * p[2]:+.1f}] → {'PASS' if ok else 'fail'}")
    say(f"\n**T1 {'PASSES' if any(ok for ok, _ in verdict.values()) else 'FAILS'}.**")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out) + "\n")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
