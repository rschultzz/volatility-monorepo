#!/usr/bin/env python3
"""CR-BO statistics + report — the single pre-registered run (specs/CR-BO-session-premium.md L5–L9).
Reads scripts/output/cr_bo/nights.parquet (scripts/cr_bo_extract.py). No DB access.

Fixed here before the run (readings of L7 / L8 that the spec leaves open):
  * Units. Variance points = total variance × 10⁴ (i.e. %²). Ratio D/RV = ratio of means (mean D / mean RV);
    the median of the per-night ratio is printed alongside.
  * Primary side for L8 = atm (`atmiv`). put / call (L6(d)) are reported, not gated.
  * "Non-event weeknight" = t Mon–Thu with t+1 the next calendar day (cal_gap 1; this drops Fri→Mon and
    holiday-spanning nights) AND no CPI / PPI / NFP / PCE at 08:30 ET on t+1. FOMC on t+1 stays in (L6(c)
    makes it an RTH-leg event, not an event night); it is also reported as its own split.
  * L8 statistic = the MEAN, 95 % percentile bootstrap CI (10,000 resamples, seed 17); medians reported.
    Pass: S1 mean P_on CI lower bound > 0 AND (S1 mean P_rth ≤ 0 OR its CI contains 0).
    Kill: (mean P_on CI contains 0 in every bucket S1, S2, S7) OR (S1 mean P_rth ≥ S1 mean P_on).
    Otherwise "not supported, not killed". "Spans zero" is read literally (lower < 0 < upper).
  * Sign test: two-sided exact binomial on #(P_on > 0) vs n, p0 = 0.5.
  * Muravyev–Ni contrast: per-night paired P_on − P_rth, mean with bootstrap CI, per bucket.
  * Shares (L4): per night v_share = RV_on/(RV_on+RV_rth) and vol_share = √RV_on/(√RV_on+√RV_rth), over every
    distinct night in the universe; mean and median with bootstrap CI of the mean, plus the pooled
    ΣRV_on/Σ(RV_on+RV_rth). Benchmarks: Gatheral 40 % (variance share), Bennett 1/6 (volatility share).
  * L9 placebo: P_on_pl = D_on(t) − RV_on(t+1 → t+2), on nights where the next night passes the ES-side
    exclusions; the main premium is recomputed on the same nights for comparison, with corr(D_on, RV_on)
    vs corr(D_on, RV_on_next).
  * Readability: stock_price(15:59, t) × sign(v)·√|v| in index points, for mean D_on, mean RV_on, mean P_on.

Usage: apps/web/.venv/bin/python -u scripts/cr_bo_stats.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bo"
N_BOOT = 10_000
SEED = 17
VP = 1e4   # total variance → %²
BUCKETS = ("S1", "S2", "S7")


def boot(x: np.ndarray, stat=np.mean) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    s = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def fmt(c, k=VP, d=3):
    v, lo, hi = c
    if np.isnan(v):
        return "—"
    return f"{v*k:+.{d}f} [{lo*k:+.{d}f}, {hi*k:+.{d}f}]"


def spans(c) -> bool:
    return c[1] < 0 < c[2]


def cell(g: pd.DataFrame) -> dict:
    p_on, p_rth = g.P_on.values, g.P_rth.values
    pos = int((p_on > 0).sum())
    return dict(
        n=len(g),
        P_on_mean=boot(p_on), P_on_med=boot(p_on, np.median),
        P_rth_mean=boot(p_rth), P_rth_med=boot(p_rth, np.median),
        contrast=boot(p_on - p_rth),
        sign_pos=pos, sign_p=float(binomtest(pos, len(g), 0.5).pvalue) if len(g) else np.nan,
        D_on=float(g.D_on.mean()), RV_on=float(g.RV_on.mean()), D_rth=float(g.D_rth.mean()), RV_rth=float(g.RV_rth.mean()),
        ratio_on=float(g.D_on.mean() / g.RV_on.mean()), ratio_rth=float(g.D_rth.mean() / g.RV_rth.mean()),
        ratio_on_med=float((g.D_on / g.RV_on).median()), ratio_rth_med=float((g.D_rth / g.RV_rth).median()),
        S=float(g.stock_price_t.median()),
    )


def pts(v: float, S: float) -> str:
    return f"{np.sign(v) * S * np.sqrt(abs(v)):+.2f}"


def table(say, cells: list[tuple[str, dict]], title: str) -> None:
    say(f"\n**{title}** (variance points = total variance × 10⁴; mean [95 % CI])\n")
    say("| cell | n | P_on mean | P_on median | P_on > 0 (sign p) | P_rth mean | P_rth median | P_on − P_rth | D_on/RV_on (median) | D_rth/RV_rth (median) |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for name, c in cells:
        if c["n"] == 0:
            say(f"| {name} | 0 | | | | | | | | |"); continue
        say(f"| {name} | {c['n']} | {fmt(c['P_on_mean'])} | {fmt(c['P_on_med'])} | {c['sign_pos']}/{c['n']} ({c['sign_p']:.3g}) | "
            f"{fmt(c['P_rth_mean'])} | {fmt(c['P_rth_med'])} | {fmt(c['contrast'])} | {c['ratio_on']:.3f} ({c['ratio_on_med']:.3f}) | {c['ratio_rth']:.3f} ({c['ratio_rth_med']:.3f}) |")


def main() -> None:
    L: list[str] = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    df = pd.read_parquet(OUT_DIR / "nights.parquet")
    df["wd_label"] = np.where(df.cal_gap == 1, df.weekday,
                              np.where((df.weekday == "Fri") & (df.cal_gap == 3), "Fri→Mon", "holiday-span"))
    df["nonevent_weeknight"] = (df.cal_gap == 1) & df.weekday.isin(["Mon", "Tue", "Wed", "Thu"]) & ~df.event_0830
    say(f"# CR-BO — T-S1a session premium — single run {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Train only (t+1 ≤ 2026-06-05), read-only, holdout unread. Readings fixed in `scripts/cr_bo_stats.py` header before the run.\n")
    atm = df[df.side == "atm"]
    say(f"Rows {len(df)} (night × bucket × side). Nights per bucket (atm): " + ", ".join(f"{b} {int((atm.bucket == b).sum())}" for b in BUCKETS)
        + f"; distinct nights {atm.t.nunique()}.\n")

    # ------------------------------------------------ L8 headline
    head = {b: cell(atm[(atm.bucket == b) & atm.nonevent_weeknight]) for b in BUCKETS}
    table(say, [(b, head[b]) for b in BUCKETS], "L8 cells — atm, non-event weeknights")
    s1 = head["S1"]
    passed = s1["P_on_mean"][1] > 0 and (s1["P_rth_mean"][0] <= 0 or spans(s1["P_rth_mean"]))
    kill_a = all(spans(head[b]["P_on_mean"]) for b in BUCKETS)
    kill_b = s1["P_rth_mean"][0] >= s1["P_on_mean"][0]
    verdict = "PASS" if passed and not (kill_a or kill_b) else ("KILL S1" if (kill_a or kill_b) else "not supported, not killed")
    if passed and (kill_a or kill_b):
        verdict = "PASS and KILL both met — reported as KILL S1 (kill is the conservative reading)"
    say(f"\n**L8:** S1 P_on mean CI clear above 0: {s1['P_on_mean'][1] > 0}; S1 P_rth ≤ 0 or CI spans 0: "
        f"{s1['P_rth_mean'][0] <= 0 or spans(s1['P_rth_mean'])}; P_on CI spans 0 in every bucket: {kill_a}; "
        f"S1 mean P_rth ≥ mean P_on: {kill_b}. **Verdict: {verdict}.**\n")
    say("Readability (S1, index points, stock_price × signed √variance at the median S "
        f"{s1['S']:.0f}): D_on {pts(s1['D_on'], s1['S'])}, RV_on {pts(s1['RV_on'], s1['S'])}, P_on {pts(s1['P_on_mean'][0], s1['S'])}; "
        f"D_rth {pts(s1['D_rth'], s1['S'])}, RV_rth {pts(s1['RV_rth'], s1['S'])}, P_rth {pts(s1['P_rth_mean'][0], s1['S'])}. "
        "`stock_price` is the ORATS parity-implied price; the 2023 jitter caveat applies (CR-BH).\n")

    # ------------------------------------------------ splits
    for b in BUCKETS:
        g = atm[atm.bucket == b]
        cells = [("all nights", cell(g))]
        for w in ["Mon", "Tue", "Wed", "Thu", "Fri→Mon", "holiday-span"]:
            cells.append((f"t = {w}", cell(g[g.wd_label == w])))
        cells.append(("08:30 event on t+1", cell(g[g.event_0830])))
        cells.append(("no 08:30 event", cell(g[~g.event_0830])))
        for tag in ["CPI", "PPI", "NFP", "PCE"]:
            cells.append((f"  {tag} on t+1", cell(g[g.event_tags.str.contains(tag)])))
        cells.append(("FOMC on t+1", cell(g[g.fomc_t1])))
        cells.append(("non-event weeknight, no FOMC", cell(g[g.nonevent_weeknight & ~g.fomc_t1])))
        table(say, cells, f"Splits — bucket {b}, atm (L6 a–c)")
        if b == "S1":
            say(f"\nS1 calendar days to expiry at 15:59 on t: {g.cal_dte.value_counts().sort_index().to_dict()}.")

    # ------------------------------------------------ L6(d) sides
    cells = []
    for b in BUCKETS:
        for side in ["put", "call", "atm"]:
            g = df[(df.bucket == b) & (df.side == side)]
            cells.append((f"{b} {side}, non-event weeknights", cell(g[g.nonevent_weeknight])))
            cells.append((f"{b} {side}, all nights", cell(g)))
    table(say, cells, "L6(d) Terstegge split — put side `vol75`, call side `vol25` (same τ, same RV)")
    for side in ["put", "call"]:
        s = df[df.side == side]
        say(f"\n{side}: rows with an IV > 2.0 at any of the three snapshots (ORATS wing fits on expiring contracts): "
            f"{int(((s.iv_a > 2) | (s.iv_b > 2) | (s.iv_c > 2)).sum())} of {len(s)}, by bucket "
            f"{s[(s.iv_a > 2) | (s.iv_b > 2) | (s.iv_c > 2)].bucket.value_counts().to_dict()}. Not filtered (no rule pre-registered); read the medians.")

    # ------------------------------------------------ shares
    nights = atm.drop_duplicates("t")[["t", "RV_on", "RV_rth", "cal_gap", "weekday"]].copy()
    nights["vshare"] = nights.RV_on / (nights.RV_on + nights.RV_rth)
    nights["volshare"] = np.sqrt(nights.RV_on) / (np.sqrt(nights.RV_on) + np.sqrt(nights.RV_rth))
    say("\n## Realized overnight share (L4) — every distinct night in the universe\n")
    say("| series | n | mean [95 % CI] | median | pooled Σ | benchmark |")
    say("|---|---|---|---|---|---|")
    vs = boot(nights.vshare.values); vo = boot(nights.volshare.values)
    pooled = nights.RV_on.sum() / (nights.RV_on.sum() + nights.RV_rth.sum())
    say(f"| variance share RV_on/(RV_on+RV_rth) | {len(nights)} | {fmt(vs, 100, 1)} % | {100*nights.vshare.median():.1f} % | {100*pooled:.1f} % | Gatheral ≈ 40 % |")
    say(f"| volatility share √RV_on/(√RV_on+√RV_rth) | {len(nights)} | {fmt(vo, 100, 1)} % | {100*nights.volshare.median():.1f} % | — | Bennett ≈ 1/6 = 16.7 % |")
    wk = nights[nights.cal_gap == 1]; fm = nights[(nights.weekday == "Fri") & (nights.cal_gap == 3)]
    say(f"| variance share, weeknights only | {len(wk)} | {fmt(boot(wk.vshare.values), 100, 1)} % | {100*wk.vshare.median():.1f} % | {100*wk.RV_on.sum()/(wk.RV_on.sum()+wk.RV_rth.sum()):.1f} % | |")
    say(f"| variance share, Fri→Mon only | {len(fm)} | {fmt(boot(fm.vshare.values), 100, 1)} % | {100*fm.vshare.median():.1f} % | {100*fm.RV_on.sum()/(fm.RV_on.sum()+fm.RV_rth.sum()):.1f} % | |")
    imp = atm[atm.bucket == "S1"]
    say(f"\nImplied counterpart (unregistered context): S1 D_on / (D_on + D_rth), ratio of means = {imp.D_on.mean()/(imp.D_on.mean()+imp.D_rth.mean()):.3f} "
        f"(calendar-time share of a weeknight 16:00→09:30 vs 09:30→16:00 = {17.5/24:.3f}).")

    # ------------------------------------------------ L9 placebo
    say("\n## L9 placebo — D_on(t) against RV_on of the following night\n")
    say("| bucket | n | main P_on mean (same nights) | placebo P_on mean | corr(D_on, RV_on) | corr(D_on, RV_on_next) | unconditional mean D_on − mean RV_on |")
    say("|---|---|---|---|---|---|---|")
    pl = {}
    for b in BUCKETS:
        g = atm[(atm.bucket == b) & atm.nonevent_weeknight & atm.RV_on_next.notna()]
        main = boot(g.P_on.values); plc = boot((g.D_on - g.RV_on_next).values)
        unc = g.D_on.mean() - atm[atm.bucket == b].RV_on.mean()
        pl[b] = dict(n=len(g), main=main, placebo=plc, c_main=float(g.D_on.corr(g.RV_on)), c_pl=float(g.D_on.corr(g.RV_on_next)))
        say(f"| {b} (non-event weeknights) | {len(g)} | {fmt(main)} | {fmt(plc)} | {pl[b]['c_main']:+.3f} | {pl[b]['c_pl']:+.3f} | {unc*VP:+.3f} |")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n")
    summ = dict(verdict=verdict, pass_=bool(passed), kill_a=bool(kill_a), kill_b=bool(kill_b),
                headline={b: {k: (list(v) if isinstance(v, tuple) else v) for k, v in head[b].items()} for b in BUCKETS},
                shares=dict(vshare_mean=list(vs), volshare_mean=list(vo), vshare_pooled=float(pooled),
                            vshare_median=float(nights.vshare.median()), volshare_median=float(nights.volshare.median()), n=len(nights)),
                placebo={b: {k: (list(v) if isinstance(v, tuple) else v) for k, v in pl[b].items()} for b in BUCKETS})
    (OUT_DIR / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
