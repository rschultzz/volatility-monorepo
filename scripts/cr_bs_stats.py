#!/usr/bin/env python3
"""CR-BS statistics + report — the single pre-registered run of T-S3a on scripts/output/cr_bs/days.parquet.
No DB access. Spec: specs/CR-BS-morning-surface.md (L4–L8 and the readings fixed at the freeze).

Primary (gated, L6): walk-forward OOS R² of Y on (L, K, C, T, M), expanding window from the first 250 days, refit
daily; Q5 − Q1 of realized Y by predicted-premium quintile (Q5 = lowest predicted Y = richest predicted premium;
expanding-rank breakpoints after 20 predictions), bootstrap 95 % CI (10,000, seed 17, resampled within quintile).
  PASS  OOS R² ≥ 0.05 AND pooled Q5 − Q1 < 0 with CI upper < 0
  KILL  OOS R² < 0.02 AND pooled Q5 − Q1 CI spans zero (lower < 0 < upper)  [A1.1; OOS set is E1-only, A1.2]
  else  "not supported, not killed"
Reported, never gated: in-sample R² (raw, MA22 (a) Y and C, MA22 (b) all), Spearman of each state variable with Y,
vol50 columns, L7 placebos, L5 secondaries, FOMC / 08:30 splits, clock level shift.

Usage: apps/web/.venv/bin/python -u scripts/cr_bs_stats.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("CR_BS_OUT", REPO_ROOT / "scripts" / "output" / "cr_bs"))   # override only for the synthetic dry run
X = ["L", "K", "C", "T", "M"]
X50 = ["L_v50", "K", "C", "T_v50", "M_v50"]
WF_START, RANK_MIN, NB, SEED = 250, 20, 10_000, 17
SESSION_TAU, CAL_TAU = (5.5 / 6.5) / 252, 5.5 / 24 / 365


# ------------------------------------------------------------------ fitting
def ols(y, Xm):
    A = np.column_stack([np.ones(len(y)), Xm])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def insample(D, xs, y="Y"):
    d = D.dropna(subset=xs + [y])
    b = ols(d[y].to_numpy(), d[xs].to_numpy())
    yh = np.column_stack([np.ones(len(d)), d[xs].to_numpy()]) @ b
    return dict(n=len(d), r2=r2(d[y].to_numpy(), yh), coef=dict(zip(["const"] + xs, b)))


def ma(D, cols, w=22):
    E = D.copy()
    for c in cols:
        E[c] = E[c].rolling(w, min_periods=w).mean()
    return E


def walk_forward(D, xs, y="Y"):
    """Expanding OLS from the first WF_START days; returns frame of OOS rows with yhat, bench, rank quintile."""
    d = D.dropna(subset=xs + [y]).reset_index(drop=True)
    Y_, Xm = d[y].to_numpy(), d[xs].to_numpy()
    yh, bench = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for i in range(WF_START, len(d)):
        b = ols(Y_[:i], Xm[:i])
        yh[i] = b[0] + Xm[i] @ b[1:]
        bench[i] = Y_[:i].mean()
    o = d.iloc[WF_START:].copy()
    o["yhat"], o["bench"] = yh[WF_START:], bench[WF_START:]
    # expanding-rank quintile on predictions made so far (incl. today); Q5 = lowest yhat (richest premium)
    q = np.full(len(o), np.nan); p = o.yhat.to_numpy()
    for j in range(len(o)):
        if j + 1 < RANK_MIN:
            continue
        frac = (p[: j + 1] > p[j]).mean()                       # share of predictions above today's
        q[j] = min(int(frac * 5) + 1, 5)                        # highest yhat → Q1, lowest → Q5
    o["q"] = q
    pooled = pd.qcut(-o.yhat, 5, labels=False) + 1             # diagnostic: pooled-OOS breakpoints
    o["q_pooled"] = pooled.to_numpy()
    return o


def oos_r2(o, y="Y"):
    yy = o[y].to_numpy()
    return dict(ct=1 - np.sum((yy - o.yhat) ** 2) / np.sum((yy - o.bench) ** 2),
                full=1 - np.sum((yy - o.yhat) ** 2) / np.sum((yy - yy.mean()) ** 2), n=len(o))


def q5q1(o, y="Y", qcol="q", rng=None):
    a = o[o[qcol] == 5][y].to_numpy(); b = o[o[qcol] == 1][y].to_numpy()
    if len(a) < 2 or len(b) < 2:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n5=len(a), n1=len(b))
    rng = rng or np.random.default_rng(SEED)
    bs = rng.choice(a, (NB, len(a))).mean(1) - rng.choice(b, (NB, len(b))).mean(1)
    return dict(est=a.mean() - b.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n5=len(a), n1=len(b))


def boot_mean(x, rng=None):
    x = np.asarray(x)[np.isfinite(x)]
    rng = rng or np.random.default_rng(SEED)
    bs = rng.choice(x, (NB, len(x))).mean(1)
    return dict(est=x.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(x))


def boot_spear(a, b, rng=None):
    m = np.isfinite(a) & np.isfinite(b); a, b = np.asarray(a)[m], np.asarray(b)[m]
    rng = rng or np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), (2000, len(a)))   # 2,000 for Spearman CIs (rank refits); stated in the report
    bs = [spearmanr(a[i], b[i])[0] for i in idx]
    return dict(est=spearmanr(a, b)[0], lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(a))


def spans(c):
    return c["lo"] < 0 < c["hi"]


def fmt(c, nd=3):
    if c is None or not np.isfinite(c.get("est", np.nan)):
        return "—"
    return f"{c['est']:+.{nd}f} [{c['lo']:+.{nd}f}, {c['hi']:+.{nd}f}]"


# ------------------------------------------------------------------ main
def main() -> None:
    D = pd.read_parquet(OUT_DIR / "days.parquet").sort_values("t").reset_index(drop=True)
    D["t"] = pd.to_datetime(D.t).dt.date
    L: list[str] = []
    say = lambda s="": (L.append(s), print(s, flush=True))
    J: dict = {}
    say(f"# CR-BS — T-S3a report (single run, {datetime.now():%Y-%m-%d %H:%M} PT)\n")
    say(f"Universe n {len(D)} (E0 {int((D.era == 'E0').sum())}, E1 {int((D.era == 'E1').sum())}), {D.t.min()} → {D.t.max()}. "
        "Y = log(RV 10:30→16:00 / atmiv0²·τ_rem), session clock τ_rem = (5.5/6.5)/252. Premium a seller earns = −Y. "
        "`atmiv` primary; `vol50` beside it. No price-based adjudication exists at 10:30 ET (no train quotes at that minute).\n")

    # ---- state description
    say("## State at 10:30 ET (L2)\n")
    say("| var | E0 mean | E1 mean | pooled mean | sd |")
    say("|---|---|---|---|---|")
    for c in X + ["Y", "Y_cal"]:
        say(f"| {c} | {D[D.era == 'E0'][c].mean():+.3f} | {D[D.era == 'E1'][c].mean():+.3f} | {D[c].mean():+.3f} | {D[c].std():.3f} |")
    cp = D[X].corr(); cs = D[X].corr(method="spearman")
    say("\nPairwise correlations (Pearson above the diagonal, Spearman below):\n")
    say("| | " + " | ".join(X) + " |"); say("|---" * (len(X) + 1) + "|")
    for i, a in enumerate(X):
        say(f"| {a} | " + " | ".join("1" if i == j else (f"{cp.loc[a, b]:+.2f}" if j > i else f"{cs.loc[a, b]:+.2f}") for j, b in enumerate(X)) + " |")
    J["corr_pearson"] = cp.round(4).to_dict(); J["corr_spearman"] = cs.round(4).to_dict()
    say(f"\nClock: mean Y (session) {D.Y.mean():+.3f} vs Y_cal (1/365) {D.Y_cal.mean():+.3f}; the difference is the constant "
        f"log(τ_session/τ_cal) = {np.log(SESSION_TAU / CAL_TAU):+.3f} on every day (level shift only). "
        f"Unconditional RV/implied: session {np.exp(D.Y).mean():.3f} (mean of ratio), exp(mean Y) {np.exp(D.Y.mean()):.3f}; calendar exp(mean) {np.exp(D.Y_cal.mean()):.3f}.\n")

    # ---- primary: walk-forward
    say("## Primary — walk-forward (L4, L6)\n")
    o = walk_forward(D, X)
    r = oos_r2(o)
    by = {}
    for e in ("E0", "E1"):
        oe = o[o.era == e]
        by[e] = dict(q=q5q1(oe), r2=oos_r2(oe) if len(oe) > 1 else None, n=len(oe))
    pooled = q5q1(o)
    J["primary"] = dict(oos_r2_ct=r["ct"], oos_r2_full=r["full"], n_oos=r["n"], q5q1=pooled,
                        by_era={e: dict(q5q1=v["q"], n=v["n"], oos_r2_ct=(v["r2"] or {}).get("ct")) for e, v in by.items()})
    say(f"- OOS predictions n {r['n']} (E0 {by['E0']['n']}, E1 {by['E1']['n']}); ranked (quintile assigned) {int(o.q.notna().sum())}.")
    say(f"- **OOS R² (vs expanding historical mean) {r['ct']:+.4f}**; vs full-OOS mean {r['full']:+.4f} (diagnostic).")
    for e in ("E0", "E1"):
        if by[e]["r2"]:
            say(f"  - {e}: OOS R² {by[e]['r2']['ct']:+.4f} (n {by[e]['n']})")
    say("\n| quintile (Q5 = richest predicted premium) | n | mean ŷ | mean realized Y | E0 n / mean Y | E1 n / mean Y |")
    say("|---|---|---|---|---|---|")
    for k in range(1, 6):
        s = o[o.q == k]
        cells = [f"{len(s[s.era == e])} / {s[s.era == e].Y.mean():+.3f}" if len(s[s.era == e]) else "0 / —" for e in ("E0", "E1")]
        say(f"| Q{k} | {len(s)} | {s.yhat.mean():+.3f} | {s.Y.mean():+.3f} | {cells[0]} | {cells[1]} |")
    say(f"\n- **Pooled Q5 − Q1 of realized Y: {fmt(pooled)}** (n {pooled['n5']} / {pooled['n1']}).")
    say(f"- OOS set = E1 only (E0 OOS n {by['E0']['n']} by construction: the 250-day first window covers all of E0; first prediction {o.t.iloc[0]}).")
    qp = q5q1(o, qcol="q_pooled")
    say(f"- Diagnostic, pooled-OOS breakpoints: Q5 − Q1 {fmt(qp)}.")

    # A1.3 — second half of the OOS set (descriptive)
    h = len(o) // 2
    o2 = o.iloc[h:]
    r_h2 = oos_r2(o2)
    r_h1 = oos_r2(o.iloc[:h])
    J["primary"]["oos_r2_first_half"] = r_h1["ct"]; J["primary"]["oos_r2_second_half"] = r_h2["ct"]
    say(f"- A1.3 (descriptive): OOS R² first half {r_h1['ct']:+.4f} (n {r_h1['n']}, {o.t.iloc[0]} → {o.t.iloc[h - 1]}); "
        f"**second half {r_h2['ct']:+.4f}** (n {r_h2['n']}, {o2.t.iloc[0]} → {o2.t.iloc[-1]}).")

    # verdict (A1.1: kill on the pooled OOS set)
    pass_ = r["ct"] >= 0.05 and pooled["est"] < 0 and pooled["hi"] < 0
    kill = r["ct"] < 0.02 and spans(pooled)
    verdict = "PASS" if pass_ else ("KILL S3's gate (and, with CR-BM, S3 as designed)" if kill else "not supported, not killed")
    J["verdict"] = dict(pass_=bool(pass_), kill=bool(kill), verdict=verdict, pooled_spans=bool(spans(pooled)))
    say(f"\n**L6 (as amended by A1.1): {verdict}.** Pass needs OOS R² ≥ 0.05 ({r['ct']:+.4f}) AND pooled Q5 − Q1 < 0 with CI clear ({fmt(pooled)}). "
        f"Kill needs OOS R² < 0.02 AND pooled Q5 − Q1 CI spans zero ({spans(pooled)}). The OOS set is E1-only by construction (n {r['n']}).\n")

    # A1.2 — by-era descriptive, in-sample
    say("### By era — in-sample (no OOS predictions by construction for E0)\n")
    dfit = D.dropna(subset=X + ["Y"]).copy()
    bb = ols(dfit.Y.to_numpy(), dfit[X].to_numpy())
    dfit["yfit"] = bb[0] + dfit[X].to_numpy() @ bb[1:]
    J["by_era_insample"] = {}
    say("| era | n | in-sample R² (own-era OLS) | in-sample Q5 − Q1 (full-sample fit, quintiles within era) | label |")
    say("|---|---|---|---|---|")
    for e in ("E0", "E1"):
        de = dfit[dfit.era == e].copy()
        de["qi"] = pd.qcut(-de.yfit, 5, labels=False) + 1
        qi = q5q1(de, qcol="qi")
        ie = insample(D[D.era == e], X)
        J["by_era_insample"][e] = dict(n=len(de), r2_own=ie["r2"], q5q1=qi)
        say(f"| {e} | {len(de)} | {ie['r2']:.4f} | {fmt(qi)} | in-sample (no OOS predictions by construction) |")
    say("")

    # walk-forward coefficients at the end
    bfin = ols(D.dropna(subset=X + ['Y']).Y.to_numpy(), D.dropna(subset=X + ['Y'])[X].to_numpy())
    say("Full-sample OLS coefficients (= the last walk-forward fit + 1 day): " + ", ".join(f"{k} {v:+.3f}" for k, v in zip(["const"] + X, bfin)) + "\n")

    # ---- in-sample (reported, never gated)
    say("## In-sample R² (reported against Bandi–Fusari–Renò; never gated)\n")
    ins = insample(D, X)
    insA = insample(ma(D, ["Y", "C"]), X)
    insB = insample(ma(D, ["Y"] + X), X)
    J["insample"] = dict(raw=ins, ma22_a=insA, ma22_b=insB)
    say("| model | n | R² |")
    say("|---|---|---|")
    say(f"| raw, Y on (L, K, C, T, M) | {ins['n']} | {ins['r2']:.4f} |")
    say(f"| MA22 (a): Y and C smoothed (BFR fn. 17, literal analogue) | {insA['n']} | {insA['r2']:.4f} |")
    say(f"| MA22 (b): Y and all five smoothed | {insB['n']} | {insB['r2']:.4f} |")
    for c in X:
        say(f"| univariate {c} (raw) | {insample(D, [c])['n']} | {insample(D, [c])['r2']:.4f} |")
    for e in ("E0", "E1"):
        say(f"| raw, era {e} only | {insample(D[D.era == e], X)['n']} | {insample(D[D.era == e], X)['r2']:.4f} |")
    rngm = np.random.default_rng(SEED); Dn = D.copy(); Dn["Y"] = rngm.permutation(D.Y.to_numpy())
    nA, nB = insample(ma(Dn, ["Y", "C"]), X)["r2"], insample(ma(Dn, ["Y"] + X), X)["r2"]
    J["insample"]["ma22_noise_benchmark"] = dict(a=nA, b=nB)
    say(f"| noise benchmark: Y permuted across days (seed 17), MA22 (a) / (b) | — | {nA:.4f} / {nB:.4f} |")
    say("\nOverlapping 22-day moving averages on both sides inflate R² mechanically (serially correlated, effectively ~n/22 "
        "independent observations); the MA columns are the BFR comparison, not evidence. BFR's 21.20 % / 34.81 % are a "
        "different object (vol-drift premium α^Q − α to next-day 10:30; single equation vs GMM joint with the return equation).\n")

    # ---- Spearman of each state variable with Y
    say("## Spearman of each state variable with Y (bootstrap 2,000, seed 17)\n")
    say("| var | pooled | E0 | E1 |"); say("|---|---|---|---|")
    J["spearman"] = {}
    for c in X:
        row = [boot_spear(D[c].to_numpy(), D.Y.to_numpy())] + [boot_spear(D[D.era == e][c].to_numpy(), D[D.era == e].Y.to_numpy()) for e in ("E0", "E1")]
        J["spearman"][c] = row
        say(f"| {c} | " + " | ".join(fmt(x) for x in row) + " |")

    # ---- vol50 beside
    say("\n## `vol50` beside `atmiv` (reported, never gated)\n")
    o50 = walk_forward(D, X50, y="Y_v50")
    r50 = oos_r2(o50, y="Y_v50"); q50 = q5q1(o50, y="Y_v50")
    i50 = insample(D, X50, y="Y_v50")
    J["vol50"] = dict(oos_r2_ct=r50["ct"], q5q1=q50, insample_r2=i50["r2"])
    say(f"- Y_v50 on (L_v50, K, C, T_v50, M_v50): OOS R² {r50['ct']:+.4f} (n {r50['n']}); Q5 − Q1 {fmt(q50)}; in-sample R² {i50['r2']:.4f}.")
    for e in ("E0", "E1"):
        say(f"  - {e} Q5 − Q1 {fmt(q5q1(o50[o50.era == e], y='Y_v50'))}")

    # ---- L7 placebos
    say("\n## L7 placebos\n")
    rng = np.random.default_rng(SEED)
    Dp = D.copy(); perm = rng.permutation(len(D)); Dp[X] = D[X].to_numpy()[perm]
    op = walk_forward(Dp, X); rp = oos_r2(op); qp1 = q5q1(op)
    J["placebo_shuffle"] = dict(oos_r2_ct=rp["ct"], q5q1=qp1, insample_r2=insample(Dp, X)["r2"])
    say(f"- (i) State rows shuffled across days (seed 17): OOS R² {rp['ct']:+.4f}; Q5 − Q1 {fmt(qp1)}; in-sample R² {insample(Dp, X)['r2']:.4f}.")
    Ds = D.copy()
    prev = {t: i for i, t in enumerate(D.t)}
    Ds["Y_prev"] = [D.Y.iloc[prev[p]] if p in prev else np.nan for p in D.prev_session.map(lambda x: pd.Timestamp(x).date())]
    Ds2 = Ds.dropna(subset=["Y_prev"]).copy(); Ds2["Yp"] = Ds2.Y_prev
    os_ = walk_forward(Ds2, X, y="Yp"); rs = oos_r2(os_, y="Yp"); qs = q5q1(os_, y="Yp"); is_ = insample(Ds2, X, y="Yp")
    J["placebo_shift"] = dict(n=len(Ds2), oos_r2_ct=rs["ct"], q5q1=qs, insample_r2=is_["r2"],
                              spearman={c: boot_spear(Ds2[c].to_numpy(), Ds2.Yp.to_numpy()) for c in X})
    say(f"- (ii) Outcome shifted one day (Y_(t−1) on state_t; t−1 = the previous session, both in the universe; n {len(Ds2)}): "
        f"OOS R² {rs['ct']:+.4f}; Q5 − Q1 {fmt(qs)}; in-sample R² {is_['r2']:.4f}.")
    say("  - Spearman with Y_(t−1): " + "; ".join(f"{c} {fmt(v)}" for c, v in J["placebo_shift"]["spearman"].items()))
    say("  - Contemporaneous (past-outcome) fit is not prediction: a state that tracks Y_(t−1) as well as Y_t is persistence of the premium, not a 10:30 signal.")
    say(f"  - Persistence of Y itself: Spearman(Y_t, Y_(t−1)) {fmt(boot_spear(Ds2.Y.to_numpy(), Ds2.Yp.to_numpy()))}.")

    # ---- L5 secondaries
    say("\n## L5 secondaries (not gated)\n")
    say("### (i) Upside vs downside (Almeida–Hizmeri)\n")
    sh_i = (D.IV_up / (D.IV_up + D.IV_dn)); sh_r = D.RV_up / D.RV_rem
    say(f"- Implied up-share IV_up/(IV_up+IV_dn): median {sh_i.median():.3f}; realized up-share RV_up/RV: median {sh_r.median():.3f}, mean {sh_r.mean():.3f}.")
    say(f"- (IV_up + IV_dn)/atmiv0²τ_rem median {((D.IV_up + D.IV_dn) / D.IV_rem).median():.3f} (smile-integrated vs ATM; wings add variance).")
    J["sides"] = {}
    say("\n| side premium (log RV_side / IV_side; seller earns −) | pooled | E0 | E1 |"); say("|---|---|---|---|")
    for c in ("Y_up", "Y_dn"):
        row = [boot_mean(D[c])] + [boot_mean(D[D.era == e][c]) for e in ("E0", "E1")]
        J["sides"][c] = row
        say(f"| {c} | " + " | ".join(fmt(x) for x in row) + " |")
    diff = D.Y_up - D.Y_dn
    row = [boot_mean(diff)] + [boot_mean(diff[D.era == e]) for e in ("E0", "E1")]
    J["sides"]["Y_up_minus_Y_dn"] = row
    say("| Y_up − Y_dn (paired; < 0 = upside richer, Almeida's sign) | " + " | ".join(fmt(x) for x in row) + " |")
    say("\nτ enters IV_up and IV_dn alike (session clock); the up − down difference is clock-robust to first order, the levels are not.\n")

    say("### (ii) Realized skewness of 1-min returns 10:30 → 16:00 by predicted-premium quintile (Vilkov)\n")
    say("| quintile | n | mean rskew |"); say("|---|---|---|")
    for k in range(1, 6):
        s = o[o.q == k]
        say(f"| Q{k} | {len(s)} | {s.rskew.mean():+.3f} |")
    sk = q5q1(o, y="rskew")
    J["rskew"] = dict(q5q1=sk, spearman_Y=boot_spear(D.rskew.to_numpy(), D.Y.to_numpy()))
    say(f"\n- Q5 − Q1 rskew {fmt(sk)}; Spearman(rskew, Y) {fmt(J['rskew']['spearman_Y'])}.\n")

    say("### (iii) Fournier–Jacobs: Y_t vs the t−1 dealer-loss proxy\n")
    Fj = D.dropna(subset=["loss_prev"])
    sp = boot_spear(Fj.loss_prev.to_numpy(), Fj.Y.to_numpy())
    a, b = Fj[Fj.loss_prev > 0].Y.to_numpy(), Fj[Fj.loss_prev <= 0].Y.to_numpy()
    rng = np.random.default_rng(SEED)
    bs = rng.choice(a, (NB, len(a))).mean(1) - rng.choice(b, (NB, len(b))).mean(1)
    dd = dict(est=a.mean() - b.mean(), lo=np.quantile(bs, .025), hi=np.quantile(bs, .975))
    J["fj"] = dict(n=len(Fj), spearman=sp, loss_minus_noloss=dd, n_loss=len(a), n_noloss=len(b))
    say(f"- Proxy: |close_move_over_im| − 0.798 on the previous session (`bt_daily_outcomes`, ES open→close over SPX `implied_move_1d`; "
        f"the 06:33 PT `atmiv` it is built on understates the open straddle per CR-BP). n {len(Fj)}.")
    say(f"- Spearman(loss_(t−1), Y_t) {fmt(sp)}. Mean Y after a loss day − after a no-loss day {fmt(dd)} (n {len(a)} / {len(b)}). "
        "FJ predict a richer premium (lower Y) after dealer losses: sign < 0.\n")

    # ---- flags
    say("## Flags (diagnostic)\n")
    nf = o[~o.fomc]; ne = o[o.ev0830 == ""]
    say(f"- Q5 − Q1 excluding FOMC days {fmt(q5q1(nf))} (n {len(nf)}); excluding 08:30 release days {fmt(q5q1(ne))} (n {len(ne)}).")
    say(f"- Mean Y: FOMC days {D[D.fomc].Y.mean():+.3f} (n {int(D.fomc.sum())}); 08:30 days {D[D.ev0830 != ''].Y.mean():+.3f} (n {int((D.ev0830 != '').sum())}); other {D[(~D.fomc) & (D.ev0830 == '')].Y.mean():+.3f}.")

    # ---- write
    o_cols = o[["t", "yhat", "bench", "q", "q_pooled"]]
    D2 = D.merge(o_cols, on="t", how="left")
    D2.to_parquet(OUT_DIR / "days_scored.parquet", index=False)
    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n")

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    (OUT_DIR / "summary.json").write_text(json.dumps(clean(J), indent=1))
    print(f"\nwrote {OUT_DIR / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
