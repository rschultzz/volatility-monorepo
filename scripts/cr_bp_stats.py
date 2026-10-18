#!/usr/bin/env python3
"""CR-BP statistics + report — the single pre-registered run (specs/CR-BP-session-premium-live-open.md, as amended by A1).
Reads scripts/output/cr_bp/nights.parquet (scripts/cr_bp_extract.py). No DB access.

Readings: CR-BO's fixed readings verbatim (scripts/cr_bo_stats.py header: mean is the L8 statistic, non-event weeknight,
kill (a)/(b), sign test, contrast, shares, placebo, readability), applied to the P1 read; plus
  * A1.4 era split (E0 t < 2024-03-01, E1 t ≥ 2024-03-01) beside every pooled statistic. If S1's E0 and E1 mean P_on
    disagree in sign, that is a finding and the decision line must be "modify".
  * A1 reading r1: S2 / S7 have no P1 rows; L8 kill (a) is evaluated over the buckets with n > 0 and the report says so.
  * C6(ii): an L8 cell with n < 40 is named in a note; the verdict is computed exactly as L8 says.
  * P2 (atmiv at 09:50, C6(i) RV windows) on every CR-BO-universe night, and on the P1 nights.
  * P4: straddle-equivalent per bucket beside the CR-AS 0DTE bid–ask lower-bound proxy (Step 0 item 9).
  * A1.1 sensitivity: S1 P1 era means re-labelled at 2024-02-01 and 2024-04-01 (diagnostic; P1 coverage is not re-derived).

Usage: apps/web/.venv/bin/python -u scripts/cr_bp_stats.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.cr_bo_stats import VP, boot, cell, fmt, pts, spans, table

OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bp"
BUCKETS = ("S1", "S2", "S7")
PRIORS = {"S1": 0.05, "S2": 0.0, "S7": 0.14}
ALT_CUTS = (date(2024, 2, 1), date(2024, 4, 1))
N_FLOOR = 40


def as_p2(g: pd.DataFrame) -> pd.DataFrame:
    g = g.dropna(subset=["P_on2"]).copy()
    for a, b in [("P_on", "P_on2"), ("P_rth", "P_rth2"), ("D_on", "D_on2"), ("D_rth", "D_rth2"), ("RV_on", "RV_on2"), ("RV_rth", "RV_rth2")]:
        g[a] = g[b]
    return g


def main() -> None:
    L: list[str] = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    df = pd.read_parquet(OUT_DIR / "nights.parquet")
    df["t"] = pd.to_datetime(df.t).dt.date
    df["wd_label"] = np.where(df.cal_gap == 1, df.weekday, np.where((df.weekday == "Fri") & (df.cal_gap == 3), "Fri→Mon", "holiday-span"))
    df["nonevent_weeknight"] = (df.cal_gap == 1) & df.weekday.isin(["Mon", "Tue", "Wed", "Thu"]) & ~df.event_0830
    atm = df[df.side == "atm"]
    p1 = atm[atm.p1]
    say(f"# CR-BP — T-S1a′ session premium, live-open v_b — single run {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Train only, read-only, holdout unread. Spec as amended by A1 (era-aware P3). Variance points = total variance × 10⁴; mean [95 % CI], 10,000 resamples, seed 17.\n")
    say("P1 v_b source by era × bucket (atm rows): " + str(p1.groupby(["era", "bucket", "vb_source"]).size().to_dict())
        + f". S2 / S7 have no P1 rows (A1 reading r1: only 0DTE is captured). P1-less S1 nights: {int(((atm.bucket == 'S1') & ~atm.p1).sum())} (E0, no field passed).\n")

    # ------------------------------------------------ L8
    head = {}
    rows = []
    for b in BUCKETS:
        g = p1[(p1.bucket == b) & p1.nonevent_weeknight]
        for lab, h in [("pooled", g), ("E0", g[g.era == "E0"]), ("E1", g[g.era == "E1"])]:
            c = cell(h) if len(h) else dict(n=0)
            rows.append((f"{b} {lab}", c))
            if lab == "pooled":
                head[b] = c
            else:
                head[f"{b}_{lab}"] = c
    table(say, rows, "L8 cells — P1 read, atm, non-event weeknights (pooled, then E0 / E1)")
    live = [b for b in BUCKETS if head[b]["n"] > 0]
    s1 = head["S1"]
    passed = s1["n"] > 0 and s1["P_on_mean"][1] > 0 and (s1["P_rth_mean"][0] <= 0 or spans(s1["P_rth_mean"]))
    kill_a = all(spans(head[b]["P_on_mean"]) for b in live)
    kill_b = s1["n"] > 0 and s1["P_rth_mean"][0] >= s1["P_on_mean"][0]
    verdict = "KILL S1" if (kill_a or kill_b) else ("PASS" if passed else "not supported, not killed")
    sign_e0 = np.sign(head["S1_E0"]["P_on_mean"][0]) if head["S1_E0"]["n"] else 0
    sign_e1 = np.sign(head["S1_E1"]["P_on_mean"][0]) if head["S1_E1"]["n"] else 0
    era_disagree = bool(sign_e0 and sign_e1 and sign_e0 != sign_e1)
    small = [name for name, c in rows if 0 < c["n"] < N_FLOOR]
    say(f"\n**L8 (pooled P1 read):** S1 P_on mean CI clear above 0: {s1['P_on_mean'][1] > 0}; S1 P_rth ≤ 0 or CI spans 0: "
        f"{s1['P_rth_mean'][0] <= 0 or spans(s1['P_rth_mean'])}; P_on CI spans 0 in every bucket with n > 0 ({', '.join(live)}): {kill_a}; "
        f"S1 mean P_rth ≥ mean P_on: {kill_b}. **Verdict: {verdict}.**")
    say(f"\n**A1.4 era check:** S1 mean P_on E0 {head['S1_E0']['P_on_mean'][0]*VP:+.3f} (n {head['S1_E0']['n']}) vs E1 {head['S1_E1']['P_on_mean'][0]*VP:+.3f} (n {head['S1_E1']['n']}) — "
        + ("**signs disagree: recorded as a finding; the decision line must be 'modify'.**" if era_disagree else "same sign."))
    say(f"\n**C6(ii):** cells with 0 < n < {N_FLOOR}: {', '.join(small) if small else 'none'}. S2 / S7 have n 0 on the P1 read (r1); kill (a) was evaluated over {', '.join(live)} only.")
    say(f"\n**P6 priors:** S1 ≈ {PRIORS['S1']:+.2f}, S2 ≈ {PRIORS['S2']:+.2f}, S7 ≈ {PRIORS['S7']:+.2f}. Measured S1 pooled {s1['P_on_mean'][0]*VP:+.3f} {fmt(s1['P_on_mean'])}; CR-BO's S1 was +0.260.\n")
    say(f"Readability (S1 pooled, index points at median S {s1['S']:.0f}): D_on {pts(s1['D_on'], s1['S'])}, RV_on {pts(s1['RV_on'], s1['S'])}, P_on {pts(s1['P_on_mean'][0], s1['S'])}; "
        f"D_rth {pts(s1['D_rth'], s1['S'])}, RV_rth {pts(s1['RV_rth'], s1['S'])}, P_rth {pts(s1['P_rth_mean'][0], s1['S'])}.\n")

    # A1.1 sensitivity
    say("**A1.1 sensitivity (diagnostic; P1 coverage not re-derived):** S1 non-event weeknights, mean P_on by era under alternative cuts — "
        + "; ".join(
            f"cut {c}: E0 {boot(p1[(p1.bucket=='S1') & p1.nonevent_weeknight & (p1.t < c)].P_on.values)[0]*VP:+.3f} (n {int(((p1.bucket=='S1') & p1.nonevent_weeknight & (p1.t < c)).sum())}), "
            f"E1 {boot(p1[(p1.bucket=='S1') & p1.nonevent_weeknight & (p1.t >= c)].P_on.values)[0]*VP:+.3f}"
            for c in (date(2024, 3, 1),) + ALT_CUTS) + ".\n")

    # S1 V_px-only nights
    v = p1[(p1.bucket == "S1") & (p1.vb_source == "vpx")]
    table(say, [("S1, v_b = V_px only, all nights", cell(v)), ("  E0", cell(v[v.era == "E0"])), ("  E1", cell(v[v.era == "E1"]))],
          "S1 on the convention-free nights only (v_b = V_px, primary subset)")

    # ------------------------------------------------ splits S1 P1
    g = p1[p1.bucket == "S1"]
    cells = []
    for lab, h in [("all nights", g)] + [(f"t = {w}", g[g.wd_label == w]) for w in ["Mon", "Tue", "Wed", "Thu", "Fri→Mon", "holiday-span"]] + [
            ("08:30 event on t+1", g[g.event_0830]), ("no 08:30 event", g[~g.event_0830]), ("FOMC on t+1", g[g.fomc_t1])]:
        for e, hh in [("pooled", h), ("E0", h[h.era == "E0"]), ("E1", h[h.era == "E1"])]:
            cells.append((f"{lab} — {e}", cell(hh) if len(hh) else dict(n=0)))
    table(say, cells, "Splits — S1, P1 read (L6 a–c, with A1.4 era)")

    # sides
    cells = []
    for side in ("put", "call", "atm"):
        h = df[(df.side == side) & (df.bucket == "S1") & df.p1 & df.nonevent_weeknight]
        for e, hh in [("pooled", h), ("E0", h[h.era == "E0"]), ("E1", h[h.era == "E1"])]:
            cells.append((f"S1 {side} — {e}", cell(hh) if len(hh) else dict(n=0)))
    table(say, cells, "L6(d) Terstegge split on P1 nights (put `vol75`, call `vol25` at all three snapshots, CR-BO definitions; atm = P1)")

    # ------------------------------------------------ P2
    cells = []
    for b in BUCKETS:
        h = as_p2(atm[(atm.bucket == b) & atm.nonevent_weeknight])
        for e, hh in [("pooled", h), ("E0", h[h.era == "E0"]), ("E1", h[h.era == "E1"])]:
            cells.append((f"{b} P2 all nights — {e}", cell(hh) if len(hh) else dict(n=0)))
    h = as_p2(p1[(p1.bucket == "S1") & p1.nonevent_weeknight])
    cells.append(("S1 P2 on the P1 nights — pooled", cell(h)))
    table(say, cells, "P2 cross-check — atmiv at 09:50 ET, RV_on to 09:50 / RV_rth from 09:50 (C6(i)), non-event weeknights")

    # ------------------------------------------------ L9 (P5 interpretation)
    say("\n## L9 placebo — P1 read, S1 non-event weeknights (P5: not a kill criterion)\n")
    say("| cell | n | main P_on mean | placebo P_on mean | corr(D_on, RV_on) | corr(D_on, RV_on_next) |")
    say("|---|---|---|---|---|---|")
    pl = {}
    for e in ("pooled", "E0", "E1"):
        h = p1[(p1.bucket == "S1") & p1.nonevent_weeknight & p1.RV_on_next.notna()]
        h = h if e == "pooled" else h[h.era == e]
        if not len(h):
            continue
        m_, pc = boot(h.P_on.values), boot((h.D_on - h.RV_on_next).values)
        pl[e] = dict(n=len(h), main=list(m_), placebo=list(pc))
        say(f"| S1 {e} | {len(h)} | {fmt(m_)} | {fmt(pc)} | {h.D_on.corr(h.RV_on):+.3f} | {h.D_on.corr(h.RV_on_next):+.3f} |")
    say("\nPer P5: retaining most of the premium under a one-night shift is consistent with the calendar-time mechanism and is not a kill criterion.\n")

    # ------------------------------------------------ shares
    nights = atm.drop_duplicates("t")[["t", "RV_on", "RV_rth", "cal_gap", "era"]].copy()
    nights["vshare"] = nights.RV_on / (nights.RV_on + nights.RV_rth)
    nights["volshare"] = np.sqrt(nights.RV_on) / (np.sqrt(nights.RV_on) + np.sqrt(nights.RV_rth))
    say("## Realized overnight share (L4) — every distinct CR-BO-universe night\n")
    say("| series | era | n | mean [95 % CI] | median | pooled Σ | benchmark |")
    say("|---|---|---|---|---|---|---|")
    sh = {}
    for e in ("pooled", "E0", "E1"):
        nn = nights if e == "pooled" else nights[nights.era == e]
        vs, vo = boot(nn.vshare.values), boot(nn.volshare.values)
        pooled = nn.RV_on.sum() / (nn.RV_on.sum() + nn.RV_rth.sum())
        sh[e] = dict(vshare=list(vs), volshare=list(vo), pooled=float(pooled), n=len(nn))
        say(f"| variance share | {e} | {len(nn)} | {fmt(vs, 100, 1)} % | {100*nn.vshare.median():.1f} % | {100*pooled:.1f} % | Gatheral ≈ 40 % |")
        say(f"| volatility share | {e} | {len(nn)} | {fmt(vo, 100, 1)} % | {100*nn.volshare.median():.1f} % | — | Bennett ≈ 1/6 |")

    # ------------------------------------------------ P4
    s0 = (OUT_DIR / "step0.md").read_text()
    proxy = [l for l in s0.splitlines() if l.startswith("- Per leg") or l.startswith("- Two-leg")]
    say("\n## P4 — straddle-equivalent per bucket vs the bid–ask lower-bound proxy\n")
    say("| bucket | read | n | mean P_on in index points (stock_price × signed √) | mean D_on in points |")
    say("|---|---|---|---|---|")
    for b in BUCKETS:
        if head[b]["n"]:
            c = head[b]; say(f"| {b} | P1 | {c['n']} | {pts(c['P_on_mean'][0], c['S'])} | {pts(c['D_on'], c['S'])} |")
        h = as_p2(atm[(atm.bucket == b) & atm.nonevent_weeknight]); c = cell(h)
        say(f"| {b} | P2 | {c['n']} | {pts(c['P_on_mean'][0], c['S'])} | {pts(c['D_on'], c['S'])} |")
    say("\nCost reference (Step 0 item 9; **lower-bound proxy — the real cost test is Gate B after quote capture**; no 1-DTE train quotes exist):")
    for l in proxy:
        say(l)

    # ------------------------------------------------ A1.8
    a1 = (OUT_DIR / "step0_a1.md").read_text()
    l8 = [l for l in a1.splitlines() if l.startswith("**E1 median")]
    say("\n## A1.8 — live-card implied_move_1d accuracy\n")
    say((l8[0] if l8 else "—") + " (atmiv²τ / V_px at 09:33 ET, S1, τ per CR-BO L3.) On `implied_move_1d`'s own √(1/252) clock see `step0_a1.md`: primary E1 0.717 (√ 0.847), E0 0.696 (√ 0.834).\n")

    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n")
    summ = dict(verdict=verdict, pass_=bool(passed), kill_a=bool(kill_a), kill_b=bool(kill_b), live_buckets=live, era_sign_disagree=era_disagree,
                small_cells=small, headline={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v_.items()} for k, v_ in head.items()},
                placebo=pl, shares=sh)
    (OUT_DIR / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
