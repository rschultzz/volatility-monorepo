#!/usr/bin/env python3
"""CR-BQ Step 0 — diagnosis only (READ-ONLY, train only, no outcome statistic computed).

Spec: specs/CR-BQ-gamma-autocorrelation.md.
  Q1 CR-BK parquet schema, gamma sign convention and units, with one worked day recomputed from orats_oi_gamma
  Q2 What "net dealer gamma at the open" is in the parquet; monies-free candidates and how far apart they are
  Q3 L1 counts after each exclusion
  Q4 Distribution of G (days < 0; by year; decile cut points)
  Q5 ES RTH minute coverage by year
  Q6 The 12:30–13:00 PT window is the last 30 minutes on every included day
No ORATS monies field at the open is read. The worked day (Q1) reads the 12:46 PT prior-session smile, as CR-BK did.

Usage: apps/web/.venv/bin/python scripts/cr_bq_step0.py   → scripts/output/cr_bq/step0.md
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import cr_bq_common as Q
import cr_bk_profile as BK

WORKED_DAY = dt.date(2025, 2, 11)


def main() -> None:
    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    conn = Q.connect()
    summ, prof = Q.load_bk()
    es = Q.load_es(conn)
    counts = Q.bar_counts(es)

    say(f"# CR-BQ — Step 0 diagnosis (read-only; {dt.datetime.now():%Y-%m-%d %H:%M})\n")

    # ── Q1 ──────────────────────────────────────────────────────────────────
    say("## Q1 — CR-BK parquet schema, sign convention, units\n")
    say(f"`scripts/logs/cr_bk/shape_summary.parquet`: {len(summ)} rows × {summ.shape[1]} columns, d {summ.d.min()} → {summ.d.max()}.")
    say(f"`scripts/logs/cr_bk/profiles.parquet` (linear wings): {len(prof)} rows, {prof.d.nunique()} dates, "
        f"{prof.groupby('d').size().median():.0f} grid points per date (S0 ± 300, step 5).\n")
    say("| column | dtype | non-null |")
    say("|---|---|---|")
    for c in summ.columns:
        say(f"| `{c}` | {summ[c].dtype} | {int(summ[c].notna().sum())} |")
    say()
    say("Per `scripts/cr_bk_profile.py` (`profile`, `load_day`, `shape_of`):")
    say("- `gex_total(price)` = Σ_rows unit_gamma(F, K, σ, T) · F² · 100 · (call_oi − put_oi), with F = S_e · price / S0 "
        "(per-expiry forward shifted proportionally), σ sticky-strike from the 12:46 PT prior-session call-delta smile, "
        "T = calendar/365 to 13:00 PT on expiry.")
    say("- **Sign:** call OI counts +, put OI counts − (CR-BK L5 'net = call − |put|'), i.e. dealers assumed long calls and short puts "
        "(the SqueezeMetrics / Barbon–Buraschi eq. 4 index convention). G > 0 = dealers net long gamma (hedging dampens); G < 0 = net short.")
    say("- **Units:** dollars of delta change for a unit (100 %) relative move, Γ·S²·100·OI. ÷ 100 → $ per 1 % move. ÷ 1e9 → $bn.")
    say("- `S0` = the nearest live expiry's `orats_oi_gamma.stock_price` = ORATS' per-expiry 15:46 ET (12:46 PT) snapshot of the **prior** session (field-conventions; CR-BK A1).")
    say("- `net_at_s0_linear` = gex_total interpolated at S0. **No column holds gamma at the open of day d.** CR-BK's T3 X "
        "(`scripts/cr_bk_placebo.py` `x_at`) was computed in the run, not stored: the profile interpolated at the SPX open from "
        "`packages/shared/spx_cash` = first `orats_monies_minute.stock_price` print ≥ 06:33 PT (`spot_price` shifted 15 min on 2023 days before 2023-11-09).\n")

    effmap = BK.effective_dates(conn)
    day = BK.load_day(conn, WORKED_DAY, effmap[WORKED_DAY])
    r = BK.attach_iv(day, "linear")
    pr = BK.profile(r, day["S0"])
    row = summ[summ.d == WORKED_DAY].iloc[0]
    pp = prof[prof.d == WORKED_DAY].sort_values("price")
    ug = BK.unit_gamma(r.S_e.to_numpy(), r.K.to_numpy(), r.sig.to_numpy(), r["T"].to_numpy())
    calls = float(np.sum(ug * r.S_e.to_numpy() ** 2 * 100 * r.coi.to_numpy()))
    puts = float(np.sum(ug * r.S_e.to_numpy() ** 2 * 100 * r.poi.to_numpy()))
    say(f"### Worked day {WORKED_DAY} (store date {effmap[WORKED_DAY]}, smile snapshot {day['snap']})\n")
    say(f"- Rows with OI on live expiries: {len(day['rows'])}; with a smile: {len(r)}; S0 = {day['S0']:.2f}.")
    say(f"- Recomputed now from `orats_oi_gamma`: call side +{calls / 1e11:.3f}e11, put side −{puts / 1e11:.3f}e11 → net at S0 "
        f"{(calls - puts) / 1e11:+.4f}e11 (= {(calls - puts) / 100 / 1e9:+.2f} $bn per 1 % move).")
    say(f"- Parquet `net_at_s0_linear` {row.net_at_s0_linear / 1e11:+.4f}e11; profile grid at S0 {np.interp(day['S0'], pp.price, pp.gex_total) / 1e11:+.4f}e11; "
        f"max |recomputed − stored| over the grid {np.max(np.abs(pr.gex_total.to_numpy() - pp.gex_total.to_numpy())) / 1e11:.2e}e11.")
    say(f"- Zero-gamma flip {row.flip_nearest_linear:.1f} ({row.flip_nearest_linear - day['S0']:+.1f} vs S0); top positive peak {row.top_pos_linear:.1f}.")
    g_lo, g_hi = np.interp(day["S0"] - 25, pp.price, pp.gex_total), np.interp(day["S0"] + 25, pp.price, pp.gex_total)
    say(f"- Profile slope near S0: G(S0 − 25) {g_lo / 1e11:+.3f}e11, G(S0 + 25) {g_hi / 1e11:+.3f}e11 — "
        "a 25-pt overnight move changes G by this much.\n")

    # ── Q2 ──────────────────────────────────────────────────────────────────
    say("## Q2 — 'gamma at the open' without monies fields at the open\n")
    say("L2 asks for CR-BK's net dealer gamma **at the open**; the kickoff forbids ORATS monies fields at the open. "
        "CR-BK's own value at the open needs the SPX open from `orats_monies_minute` (Q1). Two monies-free candidates, both from the stored profile:\n")
    say("- **A** `net_at_s0_linear` — G at S0, the prior session's 15:46 ET price. Stored column; no construction.")
    say("- **B** G interpolated at S0 · ES_open(d) / ES(prev session, 12:46 PT) — the profile read at an ES-implied SPX open "
        "(ES 06:30 bar open; ES close of the 12:45 PT bar on the prior session = the CR-BK snapshot minute). ES only, frame-neutral (a ratio).\n")
    pgrid = {d: (g.price.to_numpy(), g.gex_total.to_numpy()) for d, g in prof.groupby("d")}
    first_open = es[es.minute == 0].set_index("trade_date").open
    snap_close = es[es.minute == (12 * 60 + 45 - (6 * 60 + 30))].set_index("trade_date").close
    u, steps = Q.universe(summ, counts)
    xb, why = [], []
    for d, s0 in zip(u.d, u.S0):
        p = Q.prev_session(d)
        if d not in first_open.index or p not in snap_close.index:
            xb.append(np.nan); why.append("no 06:30 bar on d" if d not in first_open.index else "no 12:45 PT bar on prev session"); continue
        x = s0 * first_open[d] / snap_close[p]
        g, v = pgrid[d]
        xb.append(float(np.interp(x, g, v)) if g.min() <= x <= g.max() else np.nan); why.append("")
    u["G_A"] = u.net_at_s0_linear
    u["G_B"] = xb
    u["x_move"] = [s0 * first_open.get(d, np.nan) / snap_close.get(Q.prev_session(d), np.nan) - s0 for d, s0 in zip(u.d, u.S0)]
    ok = u.G_B.notna()
    say(f"On the L1 universe (Q3, n {len(u)}): B computable on {int(ok.sum())} days; missing on {int((~ok).sum())} "
        f"({', '.join(f'{k}: {v}' for k, v in pd.Series([w for w in why if w]).value_counts().items()) or '—'}).")
    say(f"ES-implied overnight move S0 → open (pts): median {u.x_move.median():+.1f}, |move| median {u.x_move.abs().median():.1f}, "
        f"p90 {u.x_move.abs().quantile(.9):.1f}, max {u.x_move.abs().max():.1f}.\n")
    v = u[ok]
    rho = spearmanr(v.G_A, v.G_B).statistic
    sign_dis = int(((v.G_A < 0) != (v.G_B < 0)).sum())
    da = pd.qcut(v.G_A, 10, labels=False) + 1
    db = pd.qcut(v.G_B, 10, labels=False) + 1
    say("| A vs B (n %d) | value |" % len(v))
    say("|---|---|")
    say(f"| Spearman(G_A, G_B) | {rho:.3f} |")
    say(f"| days with a different sign | {sign_dis} ({100 * sign_dis / len(v):.1f} %) |")
    say(f"| same decile | {100 * (da == db).mean():.1f} % |")
    say(f"| decile differs by ≥ 2 | {100 * ((da - db).abs() >= 2).mean():.1f} % |")
    say(f"| D1 under A also D1 under B | {int(((da == 1) & (db == 1)).sum())} of {int((da == 1).sum())} |")
    say(f"| D10 under A also D10 under B | {int(((da == 10) & (db == 10)).sum())} of {int((da == 10).sum())} |")
    say()

    # ── Q3 ──────────────────────────────────────────────────────────────────
    say("## Q3 — L1 universe, counts after each exclusion\n")
    sess = [d for d in pd.date_range(summ.d.min(), Q.SPLIT).date if Q.is_trading_day(d)]
    missing = sorted(set(sess) - set(summ.d))
    say(f"NYSE sessions {summ.d.min()} → {Q.SPLIT}: {len(sess)}; not in the CR-BK parquet: {len(missing)} {[str(d) for d in missing]}.\n")
    say("| step | n after |")
    say("|---|---|")
    for lab, n in steps:
        say(f"| {lab} | {n} |")
    say()
    bk_excl = summ[~summ.usable.astype(bool)].d.tolist()
    say(f"CR-BK-excluded dates: {[str(d) for d in bk_excl]}.")
    low = summ[[Q.is_trading_day(d) and summ.usable.astype(bool).iloc[i] and d not in Q.BO.EARLY_CLOSES and not Q.BO.roll_week(d)
                and counts.get(d, 0) < Q.MIN_BARS for i, d in enumerate(summ.d)]]
    say(f"Dropped for < {Q.MIN_BARS} bars: {[f'{d} ({counts.get(d, 0)})' for d in low.d]}.\n")

    # ── Q4 ──────────────────────────────────────────────────────────────────
    say("## Q4 — distribution of G on the L1 universe ($bn per 1 % move = value / 1e11)\n")
    for lab, col in (("A (at S0)", "G_A"), ("B (at ES-implied open)", "G_B")):
        g = u[col].dropna()
        say(f"**{lab}:** n {len(g)}, G < 0 on **{int((g < 0).sum())}** days ({100 * (g < 0).mean():.1f} %); "
            f"median {g.median() / 1e11:+.2f}, p10 {g.quantile(.1) / 1e11:+.2f}, p90 {g.quantile(.9) / 1e11:+.2f}.")
        cuts = g.quantile(np.arange(0.1, 1.0, 0.1))
        say("decile cut points: " + ", ".join(f"{c / 1e11:+.2f}" for c in cuts))
        say()
        say("| year | n | G < 0 | median | sd |")
        say("|---|---|---|---|---|")
        for y, gy in g.groupby([d.year for d in u.loc[g.index, "d"]]):
            say(f"| {y} | {len(gy)} | {int((gy < 0).sum())} | {gy.median() / 1e11:+.2f} | {gy.std() / 1e11:.2f} |")
        dec = pd.qcut(g, 10, labels=False) + 1
        yrs = pd.crosstab(dec, [d.year for d in u.loc[g.index, "d"]])
        say("\nDays per decile by year (decile 1 = most negative):\n")
        say("| decile | " + " | ".join(str(c) for c in yrs.columns) + " |")
        say("|---|" + "---|" * len(yrs.columns))
        for i, rr in yrs.iterrows():
            say(f"| D{i} | " + " | ".join(str(x) for x in rr) + " |")
        say()

    # ── Q5 ──────────────────────────────────────────────────────────────────
    say("## Q5 — ES RTH minute coverage by year (es_minutes, is_rth)\n")
    allsess = [d for d in pd.date_range(dt.date(2023, 5, 2), Q.SPLIT).date if Q.is_trading_day(d)]
    cov = pd.DataFrame({"d": allsess, "n": [int(counts.get(d, 0)) for d in allsess]})
    cov["y"] = [d.year for d in cov.d]
    cov["early"] = [d in Q.BO.EARLY_CLOSES for d in cov.d]
    say("| year | NYSE sessions | 390 bars | 370–389 | 1–369 | 0 bars | of which early close | mean bars (non-early) |")
    say("|---|---|---|---|---|---|---|---|")
    for y, c in cov.groupby("y"):
        ne = c[~c.early]
        say(f"| {y} | {len(c)} | {int((c.n == 390).sum())} | {int(c.n.between(370, 389).sum())} | {int(c.n.between(1, 369).sum())} | "
            f"{int((c.n == 0).sum())} | {int(c.early.sum())} | {ne.n.mean():.1f} |")
    zero = cov[(cov.n == 0)].d.tolist()
    short = cov[cov.n.between(1, 369) & ~cov.early].d.tolist()
    say(f"\nSessions with 0 bars: {[str(d) for d in zero]}.")
    say(f"Non-early sessions with 1–369 bars: {[f'{d} ({counts.get(d, 0)})' for d in short]}.\n")

    # ── Q6 ──────────────────────────────────────────────────────────────────
    say("## Q6 — first and last 30 minutes on included days\n")
    e = es[es.trade_date.isin(set(u.d))]
    g = e.groupby("trade_date").minute
    last, first = g.max(), g.min()
    say(f"Included days {len(u)}: last bar starts 12:59 PT on {int((last == 389).sum())}; first bar 06:30 PT on {int((first == 0).sum())}.")
    say(f"Days whose last bar is not 12:59: {[f'{d} ({int(m)})' for d, m in last[last != 389].items()]}.")
    say(f"Days whose first bar is not 06:30: {[f'{d} ({int(m)})' for d, m in first[first != 0].items()]}.")
    in_last = e[e.minute >= 360].groupby("trade_date").size()
    in_first = e[e.minute < 30].groupby("trade_date").size()
    say(f"Bars present in 12:30–12:59: min {int(in_last.min())}, days < 30: {int((in_last < 30).sum())}. "
        f"Bars present in 06:30–06:59: min {int(in_first.min())}, days < 30: {int((in_first < 30).sum())}.")
    has = lambda m: set(e[e.minute == m].trade_date)  # noqa: E731
    say(f"Boundary bars present — 06:59 (first-30 end): {len(has(29))}/{len(u)}; 12:29 (last-30 start): {len(has(359))}/{len(u)}; "
        f"12:59: {len(has(389))}/{len(u)}.")
    say("No included day is an early close (Q3), so 12:30–13:00 PT is the session's last 30 minutes on every included day "
        "(es_minutes is_rth ends at the 12:59 bar on every non-early session).\n")
    say("Sources in the window (es_minutes.source): " + ", ".join(f"{k} {v}" for k, v in
        conn.execute("SELECT source, count(DISTINCT trade_date) FROM es_minutes WHERE is_rth AND trade_date BETWEEN %s AND %s GROUP BY 1",
                     (dt.date(2023, 5, 2), Q.SPLIT)).fetchall()) + " dates.\n")

    Q.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (Q.OUT_DIR / "step0.md").write_text("\n".join(out) + "\n")
    print(f"\nsaved {Q.OUT_DIR / 'step0.md'}")


if __name__ == "__main__":
    main()
