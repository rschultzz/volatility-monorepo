#!/usr/bin/env python3
"""CR-BQ statistics + report — the single pre-registered run (READ-ONLY; reads only days.parquet).

Spec: specs/CR-BQ-gamma-autocorrelation.md (L4–L8, Step 0 definitions, amendment A1).
Input:  scripts/output/cr_bq/days.parquet (scripts/cr_bq_extract.py)
Output: scripts/output/cr_bq/report.md, scripts/output/cr_bq/summary.json

Usage: apps/web/.venv/bin/python scripts/cr_bq_stats.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(os.environ.get("CR_BQ_DIR") or Path(__file__).resolve().parents[1] / "scripts" / "output" / "cr_bq")  # env: synthetic dry run only
SEED, B = 17, 10_000
R60 = [f"r60_{i}" for i in range(1, 7)]


def rng():
    return np.random.default_rng(SEED)


# ── statistics on a set of days ─────────────────────────────────────────────
def r60_features(df: pd.DataFrame) -> np.ndarray:
    """Per-day sums for the pooled lag-1 autocorrelation of 60-min returns (within-day pairs only)."""
    r = df[R60].to_numpy(dtype=float)
    return np.column_stack([r.sum(1), (r * r).sum(1), (r[:, :-1] * r[:, 1:]).sum(1), r[:, :-1].sum(1), r[:, 1:].sum(1)])


def rho60_from(S: np.ndarray, n_days) -> np.ndarray:
    """S: (..., 5) summed features; n_days: matching day counts. Pooled acf around the pooled mean."""
    m = S[..., 0] / (6 * n_days)
    num = S[..., 2] - m * (S[..., 3] + S[..., 4]) + 5 * n_days * m * m
    den = S[..., 1] - 6 * n_days * m * m
    return num / den


def stat_point(df: pd.DataFrame, kind: str) -> float:
    if kind == "rho60":
        return float(rho60_from(r60_features(df).sum(0), len(df)))
    return float(df[kind].mean())


def stat_boot(df: pd.DataFrame, kind: str, g: np.random.Generator) -> np.ndarray:
    n = len(df)
    idx = g.integers(0, n, (B, n))
    if kind == "rho60":
        F = r60_features(df)
        return rho60_from(F[idx].sum(1), n)
    return df[kind].to_numpy(dtype=float)[idx].mean(1)


def ci(a: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def one(df, kind):
    b = stat_boot(df, kind, rng())
    return (stat_point(df, kind), *ci(b), len(df))


def diff(d1, d0, kind):
    """stat(d1) − stat(d0), groups resampled independently (rng seeded once per statistic)."""
    g = rng()
    b1, b0 = stat_boot(d1, kind, g), stat_boot(d0, kind, g)
    return (stat_point(d1, kind) - stat_point(d0, kind), *ci(b1 - b0), len(d1), len(d0))


# ── regressions ─────────────────────────────────────────────────────────────
def ols_hc1(X: np.ndarray, y: np.ndarray):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    V = XtXi @ (X.T * e * e) @ X @ XtXi * n / (n - k)
    return b, V


def slope(x, y):
    """y = a + b x: b, HC1 t, pairs-bootstrap CI, n."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    X = np.column_stack([np.ones_like(x), x])
    b, V = ols_hc1(X, y)
    idx = rng().integers(0, len(x), (B, len(x)))
    xs, ys = x[idx], y[idx]
    xm, ym = xs.mean(1, keepdims=True), ys.mean(1, keepdims=True)
    bs = ((xs - xm) * (ys - ym)).sum(1) / ((xs - xm) ** 2).sum(1)
    return float(b[1]), float(b[1] / np.sqrt(V[1, 1])), *ci(bs), int(len(x))


def interaction(df, xcol, negcol):
    """Pooled y = a + b x + c N + d N·x; HC1 t on d (L6 'flat across gamma sign' = |t_d| < 2)."""
    s = df[np.isfinite(df[xcol]) & np.isfinite(df.r_last30) & df[negcol].notna()]
    N = s[negcol].astype(float).to_numpy()
    x = s[xcol].to_numpy(float)
    X = np.column_stack([np.ones_like(x), x, N, N * x])
    b, V = ols_hc1(X, s.r_last30.to_numpy(float))
    return float(b[3]), float(b[3] / np.sqrt(V[3, 3])), int(len(s))


def design(s: pd.DataFrame, base: list[str]) -> np.ndarray:
    cols = [s[c].to_numpy(float) for c in base]
    for t in (2, 3):
        cols.append((s.rv_tercile == t).to_numpy(float))
    for w in (1, 2, 3, 4):
        cols.append((s.dow == w).to_numpy(float))
    return np.column_stack(cols)


def boot_contrast(X, y, c):
    g = rng()
    n = len(y)
    out = np.empty(B)
    for i in range(B):
        j = g.integers(0, n, n)
        bb = np.linalg.lstsq(X[j], y[j], rcond=None)[0]
        out[i] = c @ bb
    return ci(out)


def cond_rho5(df: pd.DataFrame, deccol: str):
    """L5 for (a): rho5 on decile dummies (no intercept) + rho5_prev + RV tercile + DOW; contrast D1 − D10."""
    s = df[df[deccol].notna() & df.rho5_prev.notna()].copy()
    for k in range(1, 11):
        s[f"D{k}"] = (s[deccol] == k).astype(float)
    X = design(s, [f"D{k}" for k in range(1, 11)] + ["rho5_prev"])
    y = s.rho5.to_numpy(float)
    b, V = ols_hc1(X, y)
    c = np.zeros(X.shape[1]); c[0], c[9] = 1, -1
    est, se = float(c @ b), float(np.sqrt(c @ V @ c))
    return est, est / se, *boot_contrast(X, y, c), len(s), float(b[10]), float(b[10] / np.sqrt(V[10, 10]))


def cond_balt(s: pd.DataFrame, xcol: str):
    """L5 for (d): r_last30 on x + previous session's last-30 + RV tercile + DOW (with intercept)."""
    s = s[np.isfinite(s[xcol]) & s.r_last30_prev.notna()].copy()
    s["one"] = 1.0
    X = design(s, ["one", xcol, "r_last30_prev"])
    y = s.r_last30.to_numpy(float)
    b, V = ols_hc1(X, y)
    c = np.zeros(X.shape[1]); c[1] = 1
    return float(b[1]), float(b[1] / np.sqrt(V[1, 1])), *boot_contrast(X, y, c), len(s), float(b[2]), float(b[2] / np.sqrt(V[2, 2]))


# ── formatting ──────────────────────────────────────────────────────────────
def f3(t):
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def clear(t):
    return t[1] > 0 or t[2] < 0


def main() -> None:
    U = pd.read_parquet(OUT / "days.parquet")
    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    summ: dict = {}
    KINDS = [("rho5", "(a) ρ5"), ("rho60", "(b) ρ60 pooled"), ("VR", "(c) VR 30/5"), ("skew", "(e) skew (secondary)")]

    say(f"# CR-BQ — T-S2a: ES intraday autocorrelation by dealer net-gamma decile (single run, {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    say("Spec `specs/CR-BQ-gamma-autocorrelation.md` (L1–L8, Step 0 definitions, amendment A1). Train only (≤ 2026-06-05), read-only, holdout unread. "
        "G = `net_at_s0_linear` (CR-BK corrected profile at the prior session's 15:46 ET price; call OI +, put OI −). "
        f"Bootstrap {B:,} resamples over days, `default_rng({SEED})` per statistic. Baltussen t = OLS with HC1 SE.\n")
    R = U[U.decile.notna()]
    say(f"Universe {len(U)} days; trailing-250 rank decile (min 120 prior sessions) on **{len(R)}** "
        f"(first {R.d.min()}; **{len(U) - len(R)} days carry no decile** and are excluded from decile statistics). "
        f"G < 0 on {int(U.neg.sum())} of {len(U)} days (sign split uses all {len(U)}).\n")

    # ── per-decile table ────────────────────────────────────────────────────
    say("## 1. Primary — by trailing-rank decile of G (D1 = lowest rank)\n")
    say("| decile | n | median G ($bn/1 %) | share G < 0 | ρ5 [95 % CI] | ρ60 pooled [CI] | VR [CI] | skew [CI] |")
    say("|---|---|---|---|---|---|---|---|")
    for k in range(1, 11):
        s = R[R.decile == k]
        cells = [f3(one(s, kind)) for kind, _ in KINDS]
        say(f"| D{k} | {len(s)} | {s.G.median() / 1e11:+.1f} | {100 * s.neg.mean():.0f} % | " + " | ".join(cells) + " |")
    say()
    d1, d10 = R[R.decile == 1], R[R.decile == 10]
    grad = {kind: diff(d1, d10, kind) for kind, _ in KINDS}
    say("**Gradient D1 − D10** (groups resampled independently):\n")
    say("| statistic | D1 − D10 [95 % CI] | n D1 / D10 | CI clear of 0 |")
    say("|---|---|---|---|")
    for kind, lab in KINDS:
        t = grad[kind]
        say(f"| {lab} | {f3(t)} | {t[3]} / {t[4]} | {'yes' if clear(t) else 'no'} |")
    say()
    summ["gradient"] = {k: list(v) for k, v in grad.items()}

    # per-sd slope for the Barbon–Buraschi prior
    z = (U.G - U.G.mean()) / U.G.std()
    b_sd = slope(z, U.rho5)
    zr = (R["rank"] - R["rank"].mean()) / R["rank"].std()
    b_sdr = slope(zr, R.rho5)
    sp = spearmanr(R["rank"], R.rho5).statistic
    say(f"ρ5 per +1 sd of raw G (all {b_sd[4]} days): **{b_sd[0]:+.4f}** (t {b_sd[1]:+.2f}) [{b_sd[2]:+.4f}, {b_sd[3]:+.4f}]; "
        f"per +1 sd of trailing rank ({b_sdr[4]} days): {b_sdr[0]:+.4f} (t {b_sdr[1]:+.2f}) [{b_sdr[2]:+.4f}, {b_sdr[3]:+.4f}]. "
        f"Spearman(rank, ρ5) {sp:+.3f}. Prior (Barbon–Buraschi Table VI): −0.015 to −0.020 per +1 sd.\n")
    summ["rho5_per_sd_rawG"] = list(b_sd)
    summ["rho5_per_sd_rank"] = list(b_sdr)

    # conditional (L5)
    c5 = cond_rho5(U, "decile")
    say(f"**L5 conditional (a):** ρ5 on decile dummies + previous-session ρ5 + RV tercile + day of week (n {c5[4]}): "
        f"D1 − D10 **{c5[0]:+.4f}** (HC1 t {c5[1]:+.2f}) bootstrap [{c5[2]:+.4f}, {c5[3]:+.4f}]; previous-session ρ5 coefficient {c5[5]:+.3f} (t {c5[6]:+.2f}).\n")
    summ["cond_rho5"] = list(c5)

    # ── by sign ─────────────────────────────────────────────────────────────
    say("## 2. Sign split (raw G, all days)\n")
    neg, pos = U[U.neg], U[~U.neg]
    say("| statistic | G < 0 | G ≥ 0 | (G < 0) − (G ≥ 0) [CI] |")
    say("|---|---|---|---|")
    for kind, lab in KINDS:
        a, b_, dd = one(neg, kind), one(pos, kind), diff(neg, pos, kind)
        say(f"| {lab} | {f3(a)} (n {a[3]}) | {f3(b_)} (n {b_[3]}) | {f3(dd)} |")
    say()

    # ── Baltussen ───────────────────────────────────────────────────────────
    say("## 3. Baltussen regression — last 30 min (12:30–13:00 PT) on the earlier return\n")
    say("Slope (raw log-return units; ×100 not applied), HC1 t, pairs-bootstrap 95 % CI. Primary regressor (A1.3) = rROD: prior-session 12:59 PT close → 12:30 PT.\n")
    say("| regressor | sample | slope | t (HC1) | 95 % CI | n |")
    say("|---|---|---|---|---|---|")
    balt = {}
    for xcol, xl in (("r_rod", "**rROD (primary)**"), ("r_first30", "first 30 min 06:30–07:00 (frozen L4(d))"), ("r_gao", "Gao: prior close → 07:00")):
        for nm, s in (("all", U), ("G < 0", neg), ("G ≥ 0", pos)):
            t = slope(s[xcol], s.r_last30)
            balt[(xcol, nm)] = t
            say(f"| {xl} | {nm} | {t[0]:+.4f} | {t[1]:+.2f} | [{t[2]:+.4f}, {t[3]:+.4f}] | {t[4]} |")
        it = interaction(U, xcol, "neg")
        balt[(xcol, "interaction")] = it
        say(f"| {xl} | (G < 0) × x interaction | {it[0]:+.4f} | {it[1]:+.2f} | — | {it[2]} |")
    say()
    for nm, s in (("G < 0", neg), ("G ≥ 0", pos), ("all", U)):
        c = cond_balt(s, "r_rod")
        say(f"L5 conditional rROD, {nm} (+ previous-session last-30, RV tercile, DOW; n {c[4]}): slope {c[0]:+.4f} (t {c[1]:+.2f}) "
            f"[{c[2]:+.4f}, {c[3]:+.4f}]; previous last-30 coefficient {c[5]:+.3f} (t {c[6]:+.2f}).")
        summ[f"cond_balt_{nm}"] = list(c)
    say()
    summ["baltussen"] = {f"{k[0]}|{k[1]}": list(v) for k, v in balt.items()}

    # ── L6 ──────────────────────────────────────────────────────────────────
    a = grad["rho5"]
    bn, bp, bi = balt[("r_rod", "G < 0")], balt[("r_rod", "G ≥ 0")], balt[("r_rod", "interaction")]
    pass_a = a[0] > 0 and a[1] > 0
    sig = lambda t: t[0] > 0 and t[1] >= 2  # noqa: E731
    pass_d = sig(bn) and not sig(bp)
    any_grad = any(clear(grad[k]) for k in ("rho5", "rho60", "VR"))
    flat = abs(bi[1]) < 2
    verdict = "PASS" if (pass_a or pass_d) else ("KILL S2" if (not any_grad and flat) else "not supported, not killed")
    say("## 4. L6 verdict (as written; A1.3 primary regressor)\n")
    say(f"- (a) D1 − D10 ρ5 = {f3(a)} → {'> 0 with CI clear of zero' if pass_a else 'not > 0 with CI clear of zero'}.")
    say(f"- (d) rROD slope: G < 0 {bn[0]:+.4f} (t {bn[1]:+.2f}); G ≥ 0 {bp[0]:+.4f} (t {bp[1]:+.2f}) → "
        f"{'positive with t ≥ 2 on G < 0 and not on G ≥ 0' if pass_d else 'pass clause not met'}.")
    say(f"- Kill test: gradient CI clear of zero on (a) {clear(grad['rho5'])}, (b) {clear(grad['rho60'])}, (c) {clear(grad['VR'])}; "
        f"interaction t {bi[1]:+.2f} → {'flat' if flat else 'not flat'} across sign.")
    say(f"\n**L6: {verdict}.**\n")
    summ["L6"] = verdict

    # ── L7 placebo ──────────────────────────────────────────────────────────
    say("## 5. L7 placebo — decile and sign labels permuted across days (`default_rng(17)`)\n")
    perm = rng().permutation(len(U))
    P = U.copy()
    P["decile"] = U.decile.to_numpy()[perm]
    P["neg"] = U.neg.to_numpy()[perm]
    PR = P[P.decile.notna()]
    say("| statistic | D1 − D10, shuffled [CI] | CI clear of 0 |")
    say("|---|---|---|")
    pl = {}
    for kind, lab in KINDS:
        t = diff(PR[PR.decile == 1], PR[PR.decile == 10], kind)
        pl[kind] = t
        say(f"| {lab} | {f3(t)} | {'yes' if clear(t) else 'no'} |")
    pn, pp = slope(P[P.neg].r_rod, P[P.neg].r_last30), slope(P[~P.neg].r_rod, P[~P.neg].r_last30)
    pi = interaction(P, "r_rod", "neg")
    say(f"\nShuffled rROD slope: G < 0 {pn[0]:+.4f} (t {pn[1]:+.2f}), G ≥ 0 {pp[0]:+.4f} (t {pp[1]:+.2f}), interaction t {pi[1]:+.2f}.\n")
    summ["placebo"] = {k: list(v) for k, v in pl.items()} | {"balt_neg": list(pn), "balt_pos": list(pp), "interaction": list(pi)}

    # ── secondary and diagnostic decile constructions ───────────────────────
    say("## 6. Secondary / diagnostic decile constructions (reported only, not gated)\n")
    say("| construction | n D1 / D10 | ρ5 D1 − D10 | ρ60 D1 − D10 | VR D1 − D10 | skew D1 − D10 |")
    say("|---|---|---|---|---|---|")
    sec = {}
    for col, lab in (("decile_raw", "raw-dollar deciles, full sample (A1.2)"), ("decile_adv", "G / ES ADV deciles (A1.2)"),
                     ("decile_B", "B: trailing rank of G at ES-implied open (A1.1)")):
        s = U[U[col].notna()]
        ts = [diff(s[s[col] == 1], s[s[col] == 10], kind) for kind, _ in KINDS]
        sec[col] = [list(t) for t in ts]
        say(f"| {lab} | {ts[0][3]} / {ts[0][4]} | " + " | ".join(f3(t) + (" *" if clear(t) else "") for t in ts) + " |")
    say("\n`*` = CI clear of zero.\n")
    sB = U[U.neg_B.notna()]
    nB, pB = sB[sB.neg_B.astype(bool)], sB[~sB.neg_B.astype(bool)]
    tnB, tpB = slope(nB.r_rod, nB.r_last30), slope(pB.r_rod, pB.r_last30)
    sB2 = sB.assign(negB=sB.neg_B.astype(bool))
    iB = interaction(sB2, "r_rod", "negB")
    say(f"B sign split, rROD slope: G_B < 0 {tnB[0]:+.4f} (t {tnB[1]:+.2f}, n {tnB[4]}); G_B ≥ 0 {tpB[0]:+.4f} (t {tpB[1]:+.2f}, n {tpB[4]}); interaction t {iB[1]:+.2f}.")
    gB = diff(nB, pB, "rho5")
    say(f"B sign split, ρ5 (G_B < 0) − (G_B ≥ 0): {f3(gB)}.\n")
    summ["secondary"] = sec
    summ["B_balt"] = {"neg": list(tnB), "pos": list(tpB), "interaction": list(iB)}

    # ── by year (A1.5) ──────────────────────────────────────────────────────
    say("## 7. By year (A1.5 diagnostic; deciles from the full run, not re-cut)\n")
    say("| year | ranked days | n D1 / D10 | ρ5 D1 − D10 | ρ60 D1 − D10 | VR D1 − D10 | rROD slope G < 0 (t, n) | rROD slope G ≥ 0 (t, n) | interaction t |")
    say("|---|---|---|---|---|---|---|---|---|")
    yr = {}
    for y in (2023, 2024, 2025, 2026):
        Y, YR = U[U.year == y], R[R.year == y]
        a1, a10 = YR[YR.decile == 1], YR[YR.decile == 10]
        cells = []
        for kind in ("rho5", "rho60", "VR"):
            cells.append(f3(diff(a1, a10, kind)) if len(a1) >= 3 and len(a10) >= 3 else "n < 3")
        yn, yp = Y[Y.neg], Y[~Y.neg]
        sn, sp_ = slope(yn.r_rod, yn.r_last30), slope(yp.r_rod, yp.r_last30)
        iy = interaction(Y, "r_rod", "neg")
        yr[y] = {"n_rank": len(YR), "n_d1": len(a1), "n_d10": len(a10), "neg": list(sn), "pos": list(sp_), "int": list(iy)}
        lab = "2026 H1" if y == 2026 else str(y)
        say(f"| {lab} | {len(YR)} | {len(a1)} / {len(a10)} | " + " | ".join(cells) +
            f" | {sn[0]:+.3f} ({sn[1]:+.2f}, {sn[4]}) | {sp_[0]:+.3f} ({sp_[1]:+.2f}, {sp_[4]}) | {iy[1]:+.2f} |")
    say()
    summ["by_year"] = yr
    say("Decile composition by year (primary deciles):\n")
    ct = pd.crosstab(R.decile.astype(int), R.year)
    say("| decile | " + " | ".join(str(c) for c in ct.columns) + " |")
    say("|---|" + "---|" * len(ct.columns))
    for i, rr in ct.iterrows():
        say(f"| D{i} | " + " | ".join(str(x) for x in rr) + " |")
    say()

    (OUT / "report.md").write_text("\n".join(out) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"\nsaved {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
