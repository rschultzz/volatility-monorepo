#!/usr/bin/env python3
"""CR-BR statistics + report — the single pre-registered run (READ-ONLY; reads only days.parquet).

Spec: specs/CR-BR-term-structure-slope.md (L4–L8, Step 0, amendments A1, A2).
Input:  scripts/output/cr_br/days.parquet, extract_meta.json (scripts/cr_br_extract.py)
Output: scripts/output/cr_br/report.md, summary.json

Usage: apps/web/.venv/bin/python scripts/cr_br_stats.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(os.environ.get("CR_BR_DIR") or Path(__file__).resolve().parents[1] / "scripts" / "output" / "cr_br")  # env: dry run only
SEED, B, BLOCK = 17, 10_000, 5
SLOPES = [("r", "r_t = log(IV30/IV7) — primary (A2.1)"), ("s", "s_t = IV30 − IV7 — secondary"),
          ("pc2_log", "PC2 of log IV² — Johnson-object column, reported only (A2.3)"),
          ("pc2_lvl", "PC2 of IV² — frozen definition, failed sign check (Step 0), reported only")]


def rng():
    return np.random.default_rng(SEED)


def boot_idx(n: int, h: int, g: np.random.Generator) -> np.ndarray:
    """(B, n) resample indices: i.i.d. for h = 1, moving 5-day blocks over date order for h = 5 (L5)."""
    if h == 1 or n <= BLOCK:
        return g.integers(0, n, (B, n))
    nb = int(np.ceil(n / BLOCK))
    starts = g.integers(0, n - BLOCK + 1, (B, nb))
    return (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(B, -1)[:, :n]


def ci(a):
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def mean_ci(y: np.ndarray, h: int):
    b = y[boot_idx(len(y), h, rng())].mean(1)
    return (float(y.mean()), *ci(b), len(y))


def diff_ci(y1: np.ndarray, y5: np.ndarray, h: int):
    g = rng()
    b = y1[boot_idx(len(y1), h, g)].mean(1) - y5[boot_idx(len(y5), h, g)].mean(1)
    return (float(y1.mean() - y5.mean()), *ci(b), len(y1), len(y5))


def spear_ci(x: np.ndarray, y: np.ndarray, h: int):
    idx = boot_idx(len(x), h, rng())
    rx = np.argsort(np.argsort(x[idx], axis=1), axis=1).astype(float)
    ry = np.argsort(np.argsort(y[idx], axis=1), axis=1).astype(float)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    b = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    return (float(spearmanr(x, y).statistic), *ci(b), len(x))


def f(t, k=3):
    return f"{t[0]:+.{k}f} [{t[1]:+.{k}f}, {t[2]:+.{k}f}]"


def clear(t):
    return t[1] > 0 or t[2] < 0


def sample(D, h, ycol):
    s = D[D[f"in_h{h}"] & D[ycol].notna()].sort_values("t")
    return s


def main() -> None:
    D = pd.read_parquet(OUT / "days.parquet")
    meta = json.loads((OUT / "extract_meta.json").read_text())
    gated = meta["gate_A2_4"]["gated_slope"]
    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    summ: dict = {"gated_slope": gated}

    say(f"# CR-BR — T-S4c: term-structure slope at the close → next-day realized vs implied (single run, {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    say("Spec `specs/CR-BR-term-structure-slope.md` (L1–L8, Step 0, A1, A2). Train only, read-only, holdout unread. "
        "Implied inputs: latest ≤ 12:46 PT monies snapshot on t, CR-BL constant-maturity method. "
        "Outcome y = log(RV / IV7²·h/365), RV = ES 13:00 PT → 13:00 PT (CR-BO), total variance. "
        "Level-neutral y_ln = residual of y on log IV7². Bootstrap 10,000, `default_rng(17)` per statistic; h = 5 moving 5-day blocks.\n")
    say(f"**Sign convention (once):** positive slope = upward (back above front); Q1 = most negative / most inverted. "
        f"Johnson's sign = Q1 − Q5 > 0 (realized exceeds implied more after an inverted curve).\n")
    say(f"A2.4 gate: inverted share by r_t quintile {' / '.join(f'{100 * x:.1f} %' for x in meta['gate_A2_4']['inv_share_by_r_quintile'])} → "
        f"{'PASS; r_t is the gated primary' if gated == 'r' else 'FAIL; run on s_t'}. Samples: x-complete {meta['n']['x_complete']}, "
        f"h = 1 {meta['n']['h1']}, h = 5 {meta['n']['h5']}. Level-neutral fits y = a + b·log IV7²: "
        + "; ".join(f"{k} b = {v[1]:+.3f}" for k, v in meta["level_neutral_coefs"].items()) + ".\n")

    # unconditional
    say("## 0. Unconditional realized vs implied\n")
    for h in (1, 5):
        s = sample(D, h, f"y{h}")
        m = mean_ci(s[f"y{h}"].to_numpy(), h)
        say(f"- h = {h}: mean log(RV / implied) {f(m)} (n {m[3]}); RV / implied = {np.exp(m[0]):.3f}; share of days RV > implied {100 * (s[f'y{h}'] > 0).mean():.1f} %.")
    say()

    # diagnostics A2.3 / A1.3
    say("## 1. What each slope definition is (A1.3, A2.3; 778 x-complete days)\n")
    X = D
    say("| definition | Spearman with log IV7² | Spearman with 1[IV90 < IV7] | Spearman with r_t | inverted share Q1 … Q5 |")
    say("|---|---|---|---|---|")
    for c, lab in SLOPES:
        sh = [X.loc[X[f"q_{c}"] == k, "inv"].mean() for k in range(1, 6)]
        say(f"| {lab} | {spearmanr(X[c], X.log_iv7sq).statistic:+.3f} | {spearmanr(X[c], X.inv).statistic:+.3f} | "
            f"{spearmanr(X[c], X.r).statistic:+.3f} | " + " / ".join(f"{100 * v:.0f} %" for v in sh) + " |")
    agree = (X.q_pc2_log == X.q_r).mean()
    say(f"\nPC2 (log IV²) vs r_t quintile agreement: same quintile on {100 * agree:.1f} % of days; Spearman of quintile labels "
        f"{spearmanr(X.q_pc2_log, X.q_r).statistic:+.3f}. PC2 (log) loadings 7…90 d: "
        + " ".join(f"{v:+.3f}" for v in meta["loadings_log"].values()) + f"; variance share {100 * meta['var_share_log'][1]:.2f} %.\n")
    summ["bridge"] = {"spearman_pc2log_r": float(spearmanr(X.pc2_log, X.r).statistic),
                      "spearman_pc2log_lev": float(spearmanr(X.pc2_log, X.log_iv7sq).statistic), "quintile_agree": float(agree)}

    # main tables
    res = {}
    for c, lab in SLOPES:
        say(f"## 2.{SLOPES.index((c, lab)) + 1} {lab}\n")
        say("| h | outcome | Q1 | Q2 | Q3 | Q4 | Q5 | **Q1 − Q5** [95 % CI] | Spearman(slope, y) [CI] |")
        say("|---|---|---|---|---|---|---|---|---|")
        for h in (1, 5):
            for yc, yl in ((f"y{h}_ln", "level-neutral"), (f"y{h}", "raw")):
                s = sample(D, h, yc)
                qs = [s.loc[s[f"q_{c}"] == k, yc].to_numpy() for k in range(1, 6)]
                cells = [f"{q.mean():+.3f} (n {len(q)})" for q in qs]
                d15 = diff_ci(qs[0], qs[4], h)
                sp = spear_ci(s[c].to_numpy(), s[yc].to_numpy(), h)
                res[(c, h, yl)] = {"q_means": [float(q.mean()) for q in qs], "q_n": [len(q) for q in qs], "q1_q5": list(d15), "spearman": list(sp),
                                   "q_ci": [list(mean_ci(q, h)) for q in qs]}
                mono = all(a > b for a, b in zip(res[(c, h, yl)]["q_means"], res[(c, h, yl)]["q_means"][1:]))
                say(f"| {h} | {yl} | " + " | ".join(cells) + f" | **{f(d15)}**{' *' if clear(d15) else ''} | {f(sp)}{' *' if clear(sp) else ''} |")
                res[(c, h, yl)]["monotone_decreasing"] = mono
        say("\n`*` = CI clear of zero. Per-quintile 95 % CIs in summary.json.\n")
    summ["main"] = {f"{k[0]}|h{k[1]}|{k[2]}": v for k, v in res.items()}

    # L6
    g = res[(gated, 1, "level-neutral")]["q1_q5"], res[(gated, 5, "level-neutral")]["q1_q5"]
    pass_ = any(t[0] > 0 and t[1] > 0 for t in g)
    kill = all(not clear(res[(c, h, "level-neutral")]["q1_q5"]) for c in ("r", "s") for h in (1, 5))
    verdict = "PASS" if pass_ else ("KILL S4's slope signal (not S4)" if kill else "not supported, not killed")
    say("## 3. L6 verdict (as written, A2.5)\n")
    say(f"- Pass clause, {gated}_t level-neutral: h = 1 Q1 − Q5 {f(g[0])}; h = 5 {f(g[1])} → "
        f"{'met' if pass_ else 'not met'} (needs > 0 with CI clear of zero at h = 1 or h = 5).")
    say("- Kill clause, level-neutral, CI spans zero at both horizons on both r_t and s_t: " + "; ".join(
        f"{c} h{h} {'spans' if not clear(res[(c, h, 'level-neutral')]['q1_q5']) else 'clear'}" for c in ("r", "s") for h in (1, 5))
        + f" → {'met' if kill else 'not met'}.")
    say(f"\n**L6: {verdict}.**\n")
    summ["L6"] = verdict

    # L7
    say("## 4. L7 placebos\n")
    say("**(i) quintile labels permuted across days** (`default_rng(17)`, per horizon universe):\n")
    say("| slope | h | outcome | Q1 − Q5 shuffled [CI] |")
    say("|---|---|---|---|")
    pl = {}
    for c in ("r", "s"):
        for h in (1, 5):
            for yc, yl in ((f"y{h}_ln", "level-neutral"), (f"y{h}", "raw")):
                s = sample(D, h, yc).copy()
                s["qp"] = s[f"q_{c}"].to_numpy()[rng().permutation(len(s))]
                t = diff_ci(s.loc[s.qp == 1, yc].to_numpy(), s.loc[s.qp == 5, yc].to_numpy(), h)
                pl[f"{c}|h{h}|{yl}"] = list(t)
                say(f"| {c} | {h} | {yl} | {f(t)}{' *' if clear(t) else ''} |")
    say("\n**(ii) outcome shifted one day earlier** (night t−1 → t, or nights t−5 … t−1 → t, normalised by IV7(t)²; slope from t; contemporaneous, not predictive):\n")
    say("| slope | h | outcome | n | Q1 − Q5 [CI] | Spearman [CI] |")
    say("|---|---|---|---|---|---|")
    for c in ("r", "s"):
        for h in (1, 5):
            for yc, yl in ((f"y{h}_prev_ln", "level-neutral"), (f"y{h}_prev", "raw")):
                s = D[D[yc].notna()].sort_values("t")
                t = diff_ci(s.loc[s[f"q_{c}"] == 1, yc].to_numpy(), s.loc[s[f"q_{c}"] == 5, yc].to_numpy(), h)
                sp = spear_ci(s[c].to_numpy(), s[yc].to_numpy(), h)
                pl[f"shift|{c}|h{h}|{yl}"] = [list(t), list(sp)]
                say(f"| {c} | {h} | {yl} | {len(s)} | {f(t)}{' *' if clear(t) else ''} | {f(sp)}{' *' if clear(sp) else ''} |")
    say()
    summ["placebo"] = pl

    # by year
    say("## 5. By year (diagnostic; full-sample quintiles)\n")
    say("| year | slope | h | outcome | n Q1 / Q5 | Q1 − Q5 [CI] | Spearman |")
    say("|---|---|---|---|---|---|---|")
    yr = {}
    for y in (2023, 2024, 2025, 2026):
        for c in ("r", "s"):
            for h in (1, 5):
                for yc, yl in ((f"y{h}_ln", "level-neutral"), (f"y{h}", "raw")):
                    s = sample(D, h, yc); s = s[s.year == y]
                    a, b = s.loc[s[f"q_{c}"] == 1, yc].to_numpy(), s.loc[s[f"q_{c}"] == 5, yc].to_numpy()
                    t = diff_ci(a, b, h) if len(a) >= 3 and len(b) >= 3 else None
                    spv = spearmanr(s[c], s[yc]).statistic if len(s) > 5 else np.nan
                    yr[f"{y}|{c}|h{h}|{yl}"] = {"n": [len(a), len(b)], "q1_q5": list(t) if t else None, "spearman": float(spv)}
                    lab = "2026 H1" if y == 2026 else str(y)
                    say(f"| {lab} | {c} | {h} | {yl} | {len(a)} / {len(b)} | {f(t) + (' *' if clear(t) else '') if t else 'n < 3'} | {spv:+.3f} |")
    say()
    summ["by_year"] = yr
    (OUT / "report.md").write_text("\n".join(out) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"saved {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
