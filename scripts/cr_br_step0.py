#!/usr/bin/env python3
"""CR-BR Step 0 — diagnosis only (READ-ONLY, train only; no outcome statistic is computed).

Spec: specs/CR-BR-term-structure-slope.md.
  Q1 L1 counts after each exclusion (h = 1 and h = 5)
  Q2 ≤ 12:46 PT atmiv vs vol50 ratio by era (E0 < 2024-03-01, E1 ≥) at 7 and 30 days — the STOP gate
  Q3 PM-expiry bracketing per tenor (≥ 95 % of days)
  Q4 PCA on train: loadings, variance explained, PC2 sign fixing; one worked day
Usage: apps/web/.venv/bin/python scripts/cr_br_step0.py   → scripts/output/cr_br/step0.md
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import cr_br_common as R

WORKED = (dt.date(2025, 4, 8), dt.date(2024, 5, 17))   # an inverted day (tariff week) and a calm day


def main() -> None:
    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731
    conn = R.connect()
    days = R.sessions()
    say(f"# CR-BR — Step 0 diagnosis (read-only; {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    snaps = R.load_snapshots(conn, days)
    print(f"snapshots {len(snaps)}", flush=True)

    # per-day tenors
    rows = []
    for d in days:
        r = {"t": d, "snap": d in snaps}
        if d in snaps:
            sp, an, vn = snaps[d]
            r["snapshot_pt"] = sp
            r["max_node_days"] = an[-1][0] if an else np.nan
            r["min_node_days"] = an[0][0] if an else np.nan
            for T in R.TENORS:
                a = R.cm_iv(an, T)
                r[f"iv{T}"] = a[0] if a else np.nan
                r[f"lo{T}"], r[f"hi{T}"] = (a[1], a[2]) if a else (np.nan, np.nan)
                if T in (7, 30):
                    b = R.cm_iv(vn, T)
                    r[f"v50_{T}"] = b[0] if b else np.nan
        rows.append(r)
    D = pd.DataFrame(rows)
    D["complete"] = D[[f"iv{T}" for T in R.TENORS]].notna().all(axis=1) if "iv7" in D else False

    # ── Q3 bracketing ────────────────────────────────────────────────────────
    S = D[D.snap]
    say("## Q3 — PM-expiry bracketing per tenor (CR-BL method, ≤ 12:46 PT snapshot on t)\n")
    say(f"Train sessions {len(D)} ({D.t.min()} → {D.t.max()}); with a ≤ 12:46 PT snapshot {len(S)}.\n")
    say("| tenor | bracketed days | share of snapshot days | exact node | median bracket width (days) |")
    say("|---|---|---|---|---|")
    for T in R.TENORS:
        b = S[f"iv{T}"].notna()
        exact = (S[f"lo{T}"] == S[f"hi{T}"]).sum()
        width = (S[f"hi{T}"] - S[f"lo{T}"])[b].median()
        say(f"| {T} d | {int(b.sum())} | {100 * b.mean():.1f} % | {int(exact)} | {width:.0f} |")
    say(f"\nAll five bracketed: {int(S.complete.sum())} of {len(S)} ({100 * S.complete.mean():.1f} %). "
        f"Longest PM node: median {S.max_node_days.median():.0f} d, min {S.max_node_days.min():.0f} d. "
        f"Shortest node ≥ 1 d: median {S.min_node_days.median():.0f} d, max {S.min_node_days.max():.0f} d.")
    miss = S[~S.complete]
    if len(miss):
        say(f"Days missing a tenor: {[str(x) for x in miss.t]}.")
    snaps_late = S.snapshot_pt.dt.time.astype(str).value_counts()
    say(f"Snapshot minute distribution (top 3): {snaps_late.head(3).to_dict()}; earliest {S.snapshot_pt.dt.time.min()}.\n")

    # ── Q2 era gate ──────────────────────────────────────────────────────────
    say("## Q2 — atmiv vs vol50 at the ≤ 12:46 PT snapshot, constant maturity, by era (STOP gate)\n")
    say("Ratio = CM IV from `atmiv` nodes / CM IV from `vol50` nodes (same snapshot, same bracketing, CR-BL method). "
        "Gate (CR-BP item 5 / P3(e) band): both era medians inside 0.95–1.05 and |E1 − E0| ≤ 0.05; otherwise the open-time break is present at the close → STOP.\n")
    say("| tenor | era | n | median | IQR | p5–p95 | share outside 0.95–1.05 |")
    say("|---|---|---|---|---|---|---|")
    gate = True
    for T in (7, 30):
        meds = {}
        for era, m in (("E0 (< 2024-03-01)", S.t < R.E1_START), ("E1 (≥ 2024-03-01)", S.t >= R.E1_START)):
            q = (S.loc[m, f"iv{T}"] / S.loc[m, f"v50_{T}"]).dropna()
            meds[era] = q.median()
            say(f"| {T} d | {era} | {len(q)} | {q.median():.4f} | {q.quantile(.25):.4f}–{q.quantile(.75):.4f} | "
                f"{q.quantile(.05):.4f}–{q.quantile(.95):.4f} | {100 * ((q < .95) | (q > 1.05)).mean():.1f} % |")
        m0, m1 = meds.values()
        ok = all(0.95 <= m <= 1.05 for m in (m0, m1)) and abs(m1 - m0) <= 0.05
        gate &= ok
        say(f"| {T} d | E1 − E0 | | {m1 - m0:+.4f} | | | gate {'pass' if ok else 'FAIL'} |")
    say(f"\n**Era gate: {'PASS — no close-time era break; continue' if gate else 'FAIL — STOP'}.**\n")
    # monthly view of the ratio around the break
    S2 = S.assign(ym=[f"{x.year}-{x.month:02d}" for x in S.t], q7=S.iv7 / S.v50_7, q30=S.iv30 / S.v50_30)
    mon = S2.groupby("ym")[["q7", "q30"]].median()
    sel = [k for k in mon.index if "2023-11" <= k <= "2024-06"]
    say("Monthly median ratio around the CR-BP break: " + "; ".join(f"{k} {mon.loc[k, 'q7']:.3f}/{mon.loc[k, 'q30']:.3f}" for k in sel) + " (7 d / 30 d).\n")

    # ── Q1 universe ──────────────────────────────────────────────────────────
    say("## Q1 — L1 universe, counts after each exclusion\n")
    es = R.load_es(conn)
    pm = R.load_switch_px(conn)
    nights, switches = R.es_nights(es, pm)
    say(f"ES bars {len(es)} ({es.et.min()} → {es.et.max()} ET); contract switches detected (CR-BO rule): {len(switches)}.\n")
    sess_pos = {d: i for i, d in enumerate(days)}
    D["night"] = [nights.get(d) for d in D.t]

    def ok1(d):
        x = nights.get(d)
        return x is not None and x["ok"]

    def ok5(d):
        i = sess_pos[d]
        if i + 5 >= len(days) + 1:
            return False
        ws = days[i:i + 5]
        return len(ws) == 5 and all(ok1(w) for w in ws) and days[i + 4] in nights and nights[days[i + 4]]["t1"] <= R.SPLIT

    steps = []
    u = D.copy()
    steps.append(("train sessions t (2023-05-01 → 2026-06-05)", len(u)))
    u = u[u.snap]; steps.append(("− no ≤ 12:46 PT monies snapshot on t", len(u)))
    for T in R.TENORS:
        u = u[u[f"iv{T}"].notna()]; steps.append((f"− {T} d not bracketed by PM expiries", len(u)))
    x_complete = len(u)
    u = u[[d in nights for d in u.t]]; steps.append(("− no next session ≤ 2026-06-05 (t+1 in holdout)", len(u)))
    flags = [("early", "− early close on t or t+1"), ("roll", "− ES roll week on t or t+1 (all years)"),
             ("switch", "− contract switch inside 13:00 PT t → 13:00 PT t+1"), ("anchor", "− no ES anchor bar 15:55–15:59 ET on t")]
    for k, lab in flags:
        u = u[[(not nights[d][k]) if k != "anchor" else nights[d][k] for d in u.t]]; steps.append((lab, len(u)))
    u = u[[nights[d]["on_miss"] <= 0.05 for d in u.t]]; steps.append(("− > 5 % overnight minutes missing", len(u)))
    u = u[[nights[d]["rth_miss"] <= 0.05 for d in u.t]]; steps.append(("− > 5 % RTH minutes missing on t+1 (i.e. < 371 of 390; implies ≥ 370)", len(u)))
    assert all(ok1(d) for d in u.t)
    h1 = u
    h5 = h1[[ok5(d) for d in h1.t]]
    say("| step (h = 1) | n after |")
    say("|---|---|")
    for lab, n in steps:
        say(f"| {lab} | {n} |")
    say(f"| **h = 5:** all five nights t … t+4 pass and t+5 ≤ 2026-06-05 | **{len(h5)}** |")
    say(f"\nx-complete train days (PCA / quintile sample): **{x_complete}**. h = 1 universe **{len(h1)}**; h = 5 universe **{len(h5)}**.")
    say(f"RTH minutes on t+1 in the h = 1 universe: min {min(390 - round(nights[d]['rth_miss'] * 390) for d in h1.t)} of 390.\n")
    say("By year (h = 1 / h = 5): " + "; ".join(
        f"{y} {int((pd.Series([d.year for d in h1.t]) == y).sum())} / {int((pd.Series([d.year for d in h5.t]) == y).sum())}"
        for y in (2023, 2024, 2025, 2026)) + ".\n")

    # ── Q4 PCA ───────────────────────────────────────────────────────────────
    say("## Q4 — PCA of x_t = (IV7², IV14², IV30², IV60², IV90²) on the x-complete train sample\n")
    X = D[D.complete][[f"iv{T}" for T in R.TENORS]].to_numpy() ** 2
    mu = X.mean(0)
    C = np.cov(X - mu, rowvar=False)
    ev, V = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    if V[-1, 1] - V[0, 1] < 0:          # sign: positive PC2 = back (90 d) above front (7 d)
        V[:, 1] *= -1
    if V[:, 0].sum() < 0:
        V[:, 0] *= -1
    say("| tenor | PC1 loading | PC2 loading | PC3 loading |")
    say("|---|---|---|---|")
    for i, T in enumerate(R.TENORS):
        say(f"| {T} d | {V[i, 0]:+.3f} | {V[i, 1]:+.3f} | {V[i, 2]:+.3f} |")
    say(f"\nVariance explained: PC1 {100 * ev[0] / ev.sum():.2f} %, PC2 {100 * ev[1] / ev.sum():.2f} %, PC3 {100 * ev[2] / ev.sum():.2f} %. "
        f"Sign fixed so that the PC2 loading on 90 d minus on 7 d is positive (L3).")
    pc2 = (X - mu) @ V[:, 1]
    s = D[D.complete].iv30 * 100 - D[D.complete].iv7 * 100
    say(f"Spearman(PC2, s_t = IV30 − IV7) on {len(pc2)} days: {pd.Series(pc2).corr(pd.Series(s.to_numpy()), method='spearman'):+.3f}. "
        f"PC2 < 0 on {int((pc2 < 0).sum())} days; s_t < 0 on {int((s < 0).sum())} days.\n")
    for d in WORKED:
        r = D[D.t == d].iloc[0]
        x = np.array([r[f"iv{T}"] for T in R.TENORS]) ** 2
        p = float((x - mu) @ V[:, 1])
        say(f"**Worked day {d}** (snapshot {r.snapshot_pt}): IV 7/14/30/60/90 = " +
            " / ".join(f"{100 * r[f'iv{T}']:.2f}" for T in R.TENORS) +
            f" vol pts; s_t = {100 * (r.iv30 - r.iv7):+.2f}; PC2 = {p:+.5f} → "
            f"{'inverted (front above back), negative PC2 as required' if p < 0 else 'upward-sloping, positive PC2 as required'}"
            f" — consistent: {(p < 0) == (r.iv90 < r.iv7)}. Brackets 7 d [{int(r.lo7)}, {int(r.hi7)}], 90 d [{int(r.lo90)}, {int(r.hi90)}].")
    say()

    # ── Q5 what the slope definitions sort on (no outcome used) ─────────────
    say("## Q5 — what each slope definition sorts on (train, x-complete days; no outcome used)\n")
    from scipy.stats import spearmanr
    Z = D[D.complete].copy()
    iv = Z[[f"iv{T}" for T in R.TENORS]].to_numpy()
    cands = {"(a) PC2 of IV² (L3 as written)": pc2, "(b) s_t = IV30 − IV7 (L3)": (s.to_numpy())}

    def pc2_of(M):
        m0 = M.mean(0); e, W = np.linalg.eigh(np.cov(M - m0, rowvar=False)); W = W[:, np.argsort(e)[::-1]]
        if W[-1, 1] - W[0, 1] < 0:
            W[:, 1] *= -1
        e = np.sort(e)[::-1]
        return (M - m0) @ W[:, 1], W[:, 1], e[1] / e.sum()
    lg, wl, sl = pc2_of(np.log(iv ** 2))
    sh, ws, ss = pc2_of(iv ** 2 / (iv ** 2).mean(1, keepdims=True))
    cands["(c) candidate: PC2 of log IV²"] = lg
    cands["(d) candidate: PC2 of IV² / row-mean IV² (shape only)"] = sh
    say(f"Candidate loadings (7…90 d): log IV² {' '.join(f'{x:+.3f}' for x in wl)} (PC2 share {100 * sl:.2f} %); "
        f"shape-only {' '.join(f'{x:+.3f}' for x in ws)} (PC2 share {100 * ss:.2f} %).\n")
    inv90 = (Z.iv90 < Z.iv7).to_numpy()
    lv = np.log(Z.iv7.to_numpy() ** 2)
    say("| definition | share IV90 < IV7 by quintile Q1 … Q5 | median IV7 Q1 / Q5 | Spearman with log IV7² | Spearman with IV90 − IV7 |")
    say("|---|---|---|---|---|")
    for lab, v in cands.items():
        q = pd.qcut(v, 5, labels=False) + 1
        sh_ = [inv90[q == k].mean() for k in range(1, 6)]
        say(f"| {lab} | " + " / ".join(f"{100 * x:.0f} %" for x in sh_) +
            f" | {100 * np.median(Z.iv7.to_numpy()[q == 1]):.1f} / {100 * np.median(Z.iv7.to_numpy()[q == 5]):.1f} | "
            f"{spearmanr(v, lv).statistic:+.3f} | {spearmanr(v, (Z.iv90 - Z.iv7).to_numpy()).statistic:+.3f} |")
    say("\nL3 requires Q1 = most inverted and positive PC2 = back above front. A definition meets it when the inverted share falls from Q1 to Q5.\n")
    R.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (R.OUT_DIR / "step0.md").write_text("\n".join(out) + "\n")
    print(f"saved {R.OUT_DIR / 'step0.md'}")


if __name__ == "__main__":
    main()
