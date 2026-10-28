#!/usr/bin/env python3
"""CR-DA — the single pre-registered run: S1 overnight premium on CR-CY straddle prices.
READ-ONLY: SELECTs only, no run row, no API calls. Holdout never read (t+1 ≤ 2026-06-05).

Spec: specs/CR-DA-overnight-premium-prices.md — D1–D10, V1 (B1 per Ryan's R-DA1), readouts 1–10.
Refuses to run unless scripts/logs/cr_da/step0_gates.json passed.

Per final night (D2), S1, total variance (variance points = × 10⁴):
  V_c, V_o = V_px of the ATM straddle at 15:55 t / 09:35 t+1 (cr_da_lib.instruments)
  D_on = V_c − V_o      RV_on  = Σ r² bars 15:55 t … 09:34 t+1       P_on  = D_on − RV_on     (primary, R-DA2)
  D_rth = V_o − 0       RV_rth = Σ r² bars 09:35 … 15:59 t+1          P_rth = D_rth − RV_rth
  readout 6: RV_on_bo bars 16:00 … 09:29, RV_rth_bo bars 09:30 … 15:59 (CR-BO windows)
  readout 7: the ±5 strangle's V_px in place of V_c / V_o
  readout 9: pnl = Σ bids(K0_c, 15:55) − Σ asks(K0_c, 09:35 | model mark); hedged adds h·(ES 09:35 − ES 15:55) − |h|·0.25

Usage: apps/web/.venv/bin/python -u scripts/cr_da_run.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from scripts import cr_da_lib as L
from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from scripts.cr_bo_step0 import load_es
from scripts.cr_bo_stats import boot

VP = 1e4


def f(c, k=VP, d=3):
    v, lo, hi = c
    return "—" if np.isnan(v) else f"{v*k:+.{d}f} [{lo*k:+.{d}f}, {hi*k:+.{d}f}]"


def cell(g: pd.DataFrame, on="P_on", rth="P_rth") -> dict:
    a, b = g[on].values, g[rth].values
    pos = int((a > 0).sum())
    return dict(n=len(g), on=boot(a), on_med=boot(a, np.median), rth=boot(b), rth_med=boot(b, np.median),
                contrast=boot(a - b), pos=pos, sign_p=float(binomtest(pos, len(g), 0.5).pvalue) if len(g) else np.nan)


def verdict(c: dict) -> tuple[str, dict]:
    """V1 (R-DA1, B1) and CR-BO's L8 kill as a readout."""
    p = c["on"][1] > 0 and (c["rth"][0] <= 0 or c["rth"][1] <= 0 <= c["rth"][2])
    r = c["on"][2] < 0 or (c["rth"][0] >= c["on"][0] and c["contrast"][2] <= 0)
    v = "REFUTED" if r else ("PASS" if p else "INCONCLUSIVE")
    l8 = dict(kill_a=bool(c["on"][1] < 0 < c["on"][2]), kill_b=bool(c["rth"][0] >= c["on"][0]))
    l8["l8_as_written"] = "PASS" if p and not (l8["kill_a"] or l8["kill_b"]) else ("KILL S1" if (l8["kill_a"] or l8["kill_b"]) else "not supported, not killed")
    return v, dict(pass_=bool(p), refuted=bool(r), **l8)


def main() -> None:
    gates = json.loads((L.LOG_DIR / "step0_gates.json").read_text())
    if not gates["pass_"]:
        raise SystemExit("Step 0 gates did not pass — STOP")
    out: list[str] = []
    say = lambda s="": (out.append(s), print(s, flush=True))
    conn = get_backfill_db_conn(); assert_role_or_die(conn); conn.autocommit = True
    n = L.load_universe()
    q = L.load_quotes(conn, sorted(n.t1)); Q = L.quote_map(q, dict(zip(n.t1, n.t)))
    es = load_es(conn); es_close = es.set_index("et").close
    df = L.exclusions(L.instruments(n, Q, es_close))
    if int(df.final.sum()) != gates["final_n"]:
        raise SystemExit(f"final n {int(df.final.sum())} ≠ Step 0 {gates['final_n']}")

    # realised variance (outcomes start here)
    nxt = dict(zip(n.t, n.t1))
    dt = lambda d, h, m: datetime.combine(d, time(h, m))
    df["RV_on"] = [L.rv(es, dt(a, 15, 55), dt(b, 9, 35)) for a, b in zip(df.t, df.t1)]
    df["RV_rth"] = [L.rv(es, dt(b, 9, 35), dt(b, 16, 0)) for b in df.t1]
    df["RV_on_bo"] = [L.rv(es, dt(a, 16, 0), dt(b, 9, 30)) for a, b in zip(df.t, df.t1)]
    df["RV_rth_bo"] = [L.rv(es, dt(b, 9, 30), dt(b, 16, 0)) for b in df.t1]
    df["RV_on_next"] = [L.rv(es, dt(b, 15, 55), dt(nxt[b], 9, 35)) if b in nxt else np.nan for b in df.t1]
    df["D_on"] = df.V_c - df.V_o; df["D_rth"] = df.V_o
    df["P_on"] = df.D_on - df.RV_on; df["P_rth"] = df.D_rth - df.RV_rth
    df["P_on_bo"] = df.D_on - df.RV_on_bo; df["P_rth_bo"] = df.D_rth - df.RV_rth_bo
    df["Ds_on"] = df.Vstr_c - df.Vstr_o
    df["Ps_on"] = df.Ds_on - df.RV_on; df["Ps_rth"] = df.Vstr_o - df.RV_rth
    df["P_on_pl"] = df.D_on - df.RV_on_next
    df["pnl_u"] = df.bid_c - df.exit_ask
    df["pnl_h"] = df.pnl_u + df.hedge_h * (df.es_o - df.es_c) - df.hedge_h.abs() * L.ES_TICK_RT
    df["wd_label"] = np.where(df.cal_gap == 1, df.weekday, np.where((df.weekday == "Fri") & (df.cal_gap == 3), "Fri→Mon", "holiday-span"))

    F = df[df.final].copy(); P = F[F.nonevent_weeknight]
    L.OUT_DIR.mkdir(parents=True, exist_ok=True)
    F.to_parquet(L.OUT_DIR / "nights.parquet", index=False)

    say(f"# CR-DA — S1 overnight premium on straddle prices — single run {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say("Train only (t+1 ≤ 2026-06-05, incl. 2025 per B4's Road 2 exception); holdout unread; read-only. "
        "Variance points = total variance × 10⁴. Mean [95 % bootstrap CI, 10 000, seed 17].\n")
    say(f"Final nights {len(F)} (of 621); **D9 population (non-event weeknights) n = {len(P)}**.\n")

    # ---------------------------------------------------- V1
    c = cell(P); v, vd = verdict(c)
    S = float(P.S_c.median())
    pts = lambda x: f"{np.sign(x) * S * np.sqrt(abs(x)):+.2f}"
    say("## V1 — primary verdict (S1, non-event weeknights, matched clock 15:55 → 09:35 → 16:00)\n")
    say("| n | P_on mean | P_on median | P_on > 0 (sign p) | P_rth mean | P_rth median | P_on − P_rth |")
    say("|---|---|---|---|---|---|---|")
    say(f"| {c['n']} | {f(c['on'])} | {f(c['on_med'])} | {c['pos']}/{c['n']} ({c['sign_p']:.3g}) | {f(c['rth'])} | {f(c['rth_med'])} | {f(c['contrast'])} |")
    say(f"\nMeans: D_on {P.D_on.mean()*VP:+.3f}, RV_on {P.RV_on.mean()*VP:+.3f}; D_rth {P.D_rth.mean()*VP:+.3f}, RV_rth {P.RV_rth.mean()*VP:+.3f}. "
        f"Ratios of means D_on/RV_on {P.D_on.mean()/P.RV_on.mean():.3f}, D_rth/RV_rth {P.D_rth.mean()/P.RV_rth.mean():.3f}.")
    say(f"Index points at median S_c {S:.0f} (readout 10): D_on {pts(P.D_on.mean())}, RV_on {pts(P.RV_on.mean())}, P_on {pts(c['on'][0])}; "
        f"D_rth {pts(P.D_rth.mean())}, RV_rth {pts(P.RV_rth.mean())}, P_rth {pts(c['rth'][0])}.\n")
    say(f"**V1: {v}.** (PASS test: P_on CI lower > 0 → {c['on'][1] > 0}; P_rth ≤ 0 or CI ∋ 0 → "
        f"{c['rth'][0] <= 0 or c['rth'][1] <= 0 <= c['rth'][2]}. REFUTED test: P_on CI upper < 0 → {c['on'][2] < 0}; "
        f"P_rth ≥ P_on with contrast CI upper ≤ 0 → {c['rth'][0] >= c['on'][0] and c['contrast'][2] <= 0}.)\n")
    say(f"Readout — CR-BO L8 as written: kill (a) P_on CI spans 0 → {vd['kill_a']}; kill (b) P_rth ≥ P_on → {vd['kill_b']}; "
        f"L8 as written = **{vd['l8_as_written']}** (not the verdict; R-DA1).\n")

    # ---------------------------------------------------- readouts 1–2
    say("## Readout 1 — the 5 largest |P_on| nights (population)\n")
    say("| t → t+1 | V_c | V_o | D_on | RV_on | P_on | P_rth |")
    say("|---|---|---|---|---|---|---|")
    for r in P.reindex(P.P_on.abs().sort_values(ascending=False).index).head(5).itertuples():
        say(f"| {r.t} → {r.t1} | {r.V_c*VP:.3f} | {r.V_o*VP:.3f} | {r.D_on*VP:+.3f} | {r.RV_on*VP:.3f} | {r.P_on*VP:+.3f} | {r.P_rth*VP:+.3f} |")
    lo_on, hi_on = np.percentile(P.P_on, [1, 99]); lo_r, hi_r = np.percentile(P.P_rth, [1, 99])
    w_on, w_rth = boot(P.P_on.clip(lo_on, hi_on).values), boot(P.P_rth.clip(lo_r, hi_r).values)
    say(f"\n## Readout 2 — winsorised at 1 % / 99 % (readout)\n\nP_on {f(w_on)}; P_rth {f(w_rth)}.\n")

    # ---------------------------------------------------- readouts 3–4
    def table(cells, title):
        say(f"## {title}\n")
        say("| cell | n | P_on mean | P_on median | P_on > 0 | P_rth mean | P_rth median | P_on − P_rth |")
        say("|---|---|---|---|---|---|---|---|")
        for name, g in cells:
            if len(g) == 0:
                say(f"| {name} | 0 | | | | | | |"); continue
            k = cell(g)
            say(f"| {name} | {k['n']} | {f(k['on'])} | {f(k['on_med'])} | {k['pos']}/{k['n']} | {f(k['rth'])} | {f(k['rth_med'])} | {f(k['contrast'])} |")
        say("")
    table([(f"{y}", P[P.year == y]) for y in (2023, 2024, 2025, 2026)] + [("all years", P)], "Readout 3 — by year of t (population; 2025 is R-b, used under the Road 2 exception)")
    cells = [("all final nights", F)] + [(f"t = {w}", F[F.wd_label == w]) for w in ["Mon", "Tue", "Wed", "Thu", "Fri→Mon", "holiday-span"]]
    cells += [("08:30 event on t+1", F[F.event_0830]), ("no 08:30 event", F[~F.event_0830])]
    cells += [(f"  {tg} on t+1", F[F.event_tags.str.contains(tg)]) for tg in ["CPI", "PPI", "NFP", "PCE"]]
    cells += [("FOMC on t+1", F[F.fomc_t1]), ("population, no FOMC", P[~P.fomc_t1])]
    table(cells, "Readout 4 — splits (all final nights)")

    # ---------------------------------------------------- readout 5 placebo
    g = P[P.RV_on_next.notna()]
    say("## Readout 5 — placebo (CR-BO L9): D_on(t) − RV_on(t+1 → t+2)\n")
    say("| n | main P_on (same nights) | placebo P_on | corr(D_on, RV_on) | corr(D_on, RV_on_next) |")
    say("|---|---|---|---|---|")
    pl = dict(n=len(g), main=boot(g.P_on.values), placebo=boot(g.P_on_pl.values), c_main=float(g.D_on.corr(g.RV_on)), c_pl=float(g.D_on.corr(g.RV_on_next)))
    say(f"| {pl['n']} | {f(pl['main'])} | {f(pl['placebo'])} | {pl['c_main']:+.3f} | {pl['c_pl']:+.3f} |\n")

    # ---------------------------------------------------- readouts 6–7
    cb = cell(P, "P_on_bo", "P_rth_bo"); vb, _ = verdict(cb)
    Ps = P[np.isfinite(P.Ps_on)]; cs = cell(Ps, "Ps_on", "Ps_rth"); vs, _ = verdict(cs)
    say("## Readouts 6–7 — CR-BO windows; ±5 strangle (V1 arithmetic, labelled readout)\n")
    say("| readout | n | P_on mean | P_on median | P_rth mean | P_rth median | P_on − P_rth | V1 arithmetic |")
    say("|---|---|---|---|---|---|---|---|")
    say(f"| 6 CR-BO windows 16:00 → 09:30 → 16:00 | {cb['n']} | {f(cb['on'])} | {f(cb['on_med'])} | {f(cb['rth'])} | {f(cb['rth_med'])} | {f(cb['contrast'])} | {vb} |")
    say(f"| 7 ±5 strangle (nearest-wing, not C1's ±0.5 IM) | {cs['n']} | {f(cs['on'])} | {f(cs['on_med'])} | {f(cs['rth'])} | {f(cs['rth_med'])} | {f(cs['contrast'])} | {vs} |")
    say(f"\nRV in the 10 RTH minutes the CR-BO windows leave out of RV_on (population means): 15:55–16:00 + 09:30–09:35 = "
        f"{(P.RV_on.mean() - P.RV_on_bo.mean())*VP:+.3f} var pts.\n")

    # ---------------------------------------------------- readout 8 spreads
    say("## Readout 8 — straddle bid–ask (index points, final nights)\n")
    say("| quantity | median | IQR | mean |")
    say("|---|---|---|---|")
    for name, x in [("full spread at 15:55 (ATM straddle)", F.spr_c), ("full spread at 09:35 (ATM straddle)", F.spr_o),
                    ("round-trip half-spread cost (spr_c + spr_o)/2", (F.spr_c + F.spr_o) / 2)]:
        say(f"| {name} | {x.median():.2f} | {x.quantile(.25):.2f}–{x.quantile(.75):.2f} | {x.mean():.2f} |")
    say(f"\nMean ATM straddle mid: 15:55 {F.st_c.mean():.2f}, 09:35 {F.st_o.mean():.2f} pts.\n")

    # ---------------------------------------------------- readout 9 tradeable
    say("## Readout 9 — tradeable (mechanical cousin, not Ryan's trade, B6): sell the K0_c straddle 15:55 at the bids, buy back 09:35 at the asks\n")
    say("Index points per night per straddle. Buyback marked from quotes where K0_c is quoted at the open, else Black at V_o + the open ATM half-spread (model).\n")
    say("| set | n | model-marked | unhedged mean | unhedged median | hedged mean | hedged median | worst 1 % (unhedged) | worst 1 % (hedged) |")
    say("|---|---|---|---|---|---|---|---|---|")
    trade = {}
    for name, g in [("population (D9)", P), ("population, quote-marked only", P[P.exit_mark == "quote"]), ("all final nights", F),
                    ("all final, quote-marked only", F[F.exit_mark == "quote"])] + [(f"population {y}", P[P.year == y]) for y in (2023, 2024, 2025, 2026)]:
        g = g[np.isfinite(g.pnl_u)]
        u, h = boot(g.pnl_u.values), boot(g.pnl_h.values)
        trade[name] = dict(n=len(g), unhedged=u, hedged=h, p1_u=float(np.percentile(g.pnl_u, 1)), p1_h=float(np.percentile(g.pnl_h, 1)),
                           model=int((g.exit_mark == "model").sum()))
        say(f"| {name} | {len(g)} | {trade[name]['model']} | {f(u, 1, 2)} | {g.pnl_u.median():+.2f} | {f(h, 1, 2)} | {g.pnl_h.median():+.2f} | "
            f"{trade[name]['p1_u']:+.2f} | {trade[name]['p1_h']:+.2f} |")
    p1 = np.percentile(P.pnl_u, 1)
    say(f"\nPopulation nights at or below the unhedged 1st percentile ({p1:+.2f}):")
    for r in P[P.pnl_u <= p1].sort_values("pnl_u").itertuples():
        say(f"- {r.t} → {r.t1}: unhedged {r.pnl_u:+.2f}, hedged {r.pnl_h:+.2f}, ES {r.es_c:.2f} → {r.es_o:.2f} ({r.es_o - r.es_c:+.2f}), mark {r.exit_mark}")
    say(f"\nMean entry credit (bids) {P.bid_c.mean():.2f}; mean exit cost (asks / mark) {P.exit_ask.mean():.2f}; mean |h| {P.hedge_h.abs().mean():.3f}.")

    (L.OUT_DIR / "report.md").write_text("\n".join(out) + "\n")
    J = lambda d: {k: (list(x) if isinstance(x, tuple) else x) for k, x in d.items()}
    (L.OUT_DIR / "summary.json").write_text(json.dumps(dict(
        verdict=v, verdict_detail=vd, primary=J(c), winsorised=dict(P_on=list(w_on), P_rth=list(w_rth)),
        crbo_windows=dict(J(cb), v1=vb), strangle=dict(J(cs), v1=vs), placebo=J(pl), trade={k: J(x) for k, x in trade.items()},
        final_n=len(F), population_n=len(P)), indent=2, default=str))
    print(f"\nwrote {L.OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
