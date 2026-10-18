#!/usr/bin/env python3
"""CR-BM — T-S3c: retired 0DTE condor legs split by wing. READ-ONLY, single pre-registered run.

Spec: specs/CR-BM-condor-legs-by-wing.md (L1–L7 + adopted amendment A1). SELECTs only; no
bt_backfill_runs row, no table writes. Outputs: scripts/output/cr_bm/legs_by_wing.parquet and
scripts/output/cr_bm/report.md. Holdout (> 2026-06-05) never read.

Inputs, as recorded:
  rows      scripts/logs/cr_as_analysis_full_fee0.65.json (CR-AS run 85b690ec): per date × box the
            strikes, IM, regime, CR-AS credit and CR-AS close (spot_price, last snapshot ≤ 13:00 PT)
  quotes    orats_options_minute 06:30–06:45 PT for the box's four OPRA symbols
  fill      condor_0dte.first_valid_entry(floor 06:33) — the CR-AS fill rule, unchanged (L2)
  settle    main read: spx_cash.settlement_prints (the harness, CR-BI) (A1.1)
            G1a / G1b: the CR-AS close from the row
  fee       config.FEE_PER_CONTRACT_PER_LEG ($0.65) → 2 legs × 2 sides / 100 = 0.026 pts per wing (A1.4)

Decomposition (L3): call_pnl = (mid short call − mid long call) − call intrinsic loss − 0.026;
put_pnl likewise; condor_pnl = call_pnl + put_pnl. IM units = pts / the row's IM.

Operational definitions fixed before the run (no spec change; how L4/L5/L7 are computed):
  - Bootstrap: iid over dates, 10,000 resamples, np.random.default_rng(17) re-seeded for every
    statistic (so each CI is reproducible on its own); 95 % = 2.5 / 97.5 percentiles.
  - Win = wing P&L > 0 after fees; Wilson 95 % interval.
  - L5 is read per structure (±0.5, ±1.0) in points (primary; L4 lists points first), IM shown
    alongside. Per structure: PASS iff call lo > 0 and put lo ≤ 0; KILL iff put lo > 0 and call
    lo ≤ 0 ("the put wing is the one positive"); else "not supported, not killed". Headline: PASS if
    both structures PASS, KILL if both KILL, otherwise "not supported, not killed" with the
    per-structure lines.
  - IV tercile: bt_vol_state_daily cr-bl-v1 iv_7d_pct ≤ 33 low / ≥ 67 high / else mid (CR-BL cuts).
  - Gamma sign: CR-BK linear-profile gex_total interpolated at the SPX-cash open (daily_ohlc open of
    the spx_cash series) — CR-BL's verbatim CR-BK x_at; > 0 positive, ≤ 0 negative.
  - Regime: the CR-AS row label, magnet-above vs other.
  - L7 placebo: put spreads re-paired with call spreads of other dates by one permutation of the
    dates (rng 17) within each structure. A permutation leaves each wing's mean unchanged by
    construction, so L7 reports what pairing *can* move: the call−put difference CI under paired vs
    shuffled bootstrap, the same-day call/put P&L correlation vs shuffled, and the shuffled condor
    mean / CI. Plus the per-wing credit check that the fill rule's joint validity filter
    (credit ∈ (0, min wing]) did not move the entry minute.

Usage: apps/web/.venv/bin/python -u scripts/cr_bm_condor_legs_by_wing.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from packages.shared.backtest.condor_0dte import CondorBox, first_valid_entry
from packages.shared.config import FEE_PER_CONTRACT_PER_LEG, round_trip_fee_pts
from packages.shared.options_cache.opra import format_opra
from packages.shared.spx_cash import build_series, daily_ohlc, fetch_minutes, settlement_prints
from scripts.cr_as_capture_0dte_condor_legs import OPRA_ROOT, UNIVERSE_END, WINDOW_END, WINDOW_START
from scripts.cr_as_condor_analysis import BARS_SQL, ENTRY_FLOOR

CR_AS_JSON = REPO_ROOT / "scripts" / "logs" / "cr_as_analysis_full_fee0.65.json"
BK_PROFILES = REPO_ROOT / "scripts" / "logs" / "cr_bk" / "profiles.parquet"
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_bm"
SEED, B = 17, 10_000
WING_FEE = round_trip_fee_pts(2, FEE_PER_CONTRACT_PER_LEG)      # 0.026
CONDOR_FEE = round_trip_fee_pts(4, FEE_PER_CONTRACT_PER_LEG)    # 0.052
AM_MIXED = ["2023-05-19", "2023-06-16", "2023-07-21", "2023-08-18", "2023-10-20", "2023-11-17", "2024-01-19", "2024-02-16"]
NO_HARNESS_SETTLE = ["2023-05-19", "2023-07-03", "2024-07-03"]
CR_AS_POOLED = {"pm0.5": -0.001, "pm1": +0.002}                 # run 85b690ec, printed to 3 dp
BOXES = ("pm0.5", "pm1")


# ── statistics ───────────────────────────────────────────────────────────────
def boot(x, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    s = stat(x[idx], axis=1)
    return (float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def boot_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    A, Bm = a[idx], b[idx]
    A = A - A.mean(1, keepdims=True); Bm = Bm - Bm.mean(1, keepdims=True)
    r = (A * Bm).sum(1) / np.sqrt((A * A).sum(1) * (Bm * Bm).sum(1))
    return (float(np.corrcoef(a, b)[0, 1]), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5)))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def f3(t, nd=3, pct=False):
    m, lo, hi = t
    if np.isnan(m):
        return "—"
    if pct:
        return f"{100 * m:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def wing_verdict(call_t, put_t):
    if call_t[1] > 0 and put_t[1] <= 0:
        return "PASS"
    if put_t[1] > 0 and call_t[1] <= 0:
        return "KILL"
    return "not supported, not killed"


# ── build ────────────────────────────────────────────────────────────────────
def main() -> None:
    cr = json.load(open(CR_AS_JSON))
    rows = cr["results"]
    assert max(r["date"] for r in rows) <= UNIVERSE_END.isoformat()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    d0, d1 = date.fromisoformat(min(r["date"] for r in rows)), date.fromisoformat(max(r["date"] for r in rows))
    clean, _ = build_series(fetch_minutes(conn, d0, d1))
    settle = {pd.Timestamp(k).date().isoformat(): float(v) for k, v in settlement_prints(clean).items()}
    ohlc = daily_ohlc(clean)
    spx_open = {pd.Timestamp(k).date().isoformat(): float(v) for k, v in ohlc["open"].items()}

    prof = pd.read_parquet(BK_PROFILES)
    prof = prof[prof.wing == "linear"]
    grids = {str(d)[:10]: (g.price.to_numpy(), g.gex_total.to_numpy()) for d, g in prof.groupby("d")}

    def gamma_at(d):
        if d not in grids or d not in spx_open:
            return np.nan
        p, g = grids[d]
        o = spx_open[d]
        return float(np.interp(o, p, g)) if p.min() <= o <= p.max() else np.nan

    ivr = {r[0].isoformat(): (None if r[1] is None else float(r[1])) for r in conn.execute(
        "SELECT trade_date, iv_7d_pct FROM bt_vol_state_daily WHERE ticker='SPX' AND version='cr-bl-v1' AND trade_date <= %s",
        (UNIVERSE_END,)).fetchall()}

    recs = []
    bars_cache = {}
    for r in rows:
        d = r["date"]
        td = date.fromisoformat(d)
        lp, sp, sc, lc = r["strikes"]
        box = CondorBox(td, 0.5 if r["box"] == "pm0.5" else 1.0, sp, lp, sc, lc, 0.0, r["im"])
        if d not in bars_cache:
            boxes_d = [x for x in rows if x["date"] == d]
            opras = sorted({format_opra(OPRA_ROOT, td, t, k) for x in boxes_d
                            for k, t in zip(x["strikes"], ("P", "P", "C", "C"))})
            bars = conn.execute(BARS_SQL, (opras, datetime.combine(td, WINDOW_START), datetime.combine(td, WINDOW_END))).fetchall()
            by_min = defaultdict(list)
            for snap, strike, typ, bid, ask in bars:
                by_min[snap].append((float(strike), typ, bid, ask))
            bars_cache = {d: sorted(by_min.items())}
        e = first_valid_entry(box, bars_cache[d], floor=ENTRY_FLOOR)
        assert e is not None, f"{d} {r['box']}: no entry on re-run"
        m = e["mids"]
        call_cr = m[(sc, "C")] - m[(lc, "C")]
        put_cr = m[(sp, "P")] - m[(lp, "P")]
        assert abs(call_cr + put_cr - r["credit_pts"]) < 1e-9, f"{d} {r['box']}: credit mismatch"
        assert e["snapshot_pt"].strftime("%H:%M") == r["entry_minute"]

        def legs(close):
            pl = max(0.0, sp - close) - max(0.0, lp - close)
            cl = max(0.0, close - sc) - max(0.0, close - lc)
            return call_cr - cl - WING_FEE, put_cr - pl - WING_FEE, cl, pl

        c_as, p_as, cl_as, pl_as = legs(r["spx_close"])
        h = settle.get(d)
        c_h, p_h, cl_h, pl_h = legs(h) if h is not None else (np.nan,) * 4
        g = gamma_at(d)
        iv = ivr.get(d)
        recs.append({
            "date": d, "box": r["box"], "regime": r["regime"], "im": r["im"],
            "long_put": lp, "short_put": sp, "short_call": sc, "long_call": lc,
            "entry_minute": r["entry_minute"], "call_credit": call_cr, "put_credit": put_cr,
            "close_cras": r["spx_close"], "close_harness": h,
            "call_pnl_cras": c_as, "put_pnl_cras": p_as, "condor_pnl_cras": c_as + p_as, "cras_net_pts": r["net_pts"],
            "call_loss_cras": cl_as, "put_loss_cras": pl_as,
            "call_pnl": c_h, "put_pnl": p_h, "condor_pnl": c_h + p_h, "call_loss": cl_h, "put_loss": pl_h,
            "gamma_open": g, "gamma_sign": (np.nan if np.isnan(g) else ("positive" if g > 0 else "negative")),
            "iv_7d_pct": iv, "iv_tercile": (None if iv is None else ("low" if iv <= 33 else ("high" if iv >= 67 else "mid"))),
            "am_mixed": d in AM_MIXED, "no_harness_settle": d in NO_HARNESS_SETTLE,
        })
    df = pd.DataFrame(recs)
    for c in ("call_pnl", "put_pnl", "condor_pnl", "call_pnl_cras", "put_pnl_cras", "condor_pnl_cras"):
        df[c + "_im"] = df[c] / df["im"]
    df["in_universe"] = ~(df.am_mixed | df.no_harness_settle)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "legs_by_wing.parquet", index=False)

    L = []
    say = lambda s="": (L.append(s), print(s))
    say(f"# CR-BM — T-S3c condor legs by wing — single run {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say(f"Rows {len(df)} (CR-AS run 85b690ec: ±0.5 {sum(df.box == 'pm0.5')}, ±1.0 {sum(df.box == 'pm1')}). "
        f"Fee {WING_FEE:.3f} pts per wing, {CONDOR_FEE:.3f} per condor (${FEE_PER_CONTRACT_PER_LEG:.2f}/contract/leg). "
        f"Bootstrap B={B}, seed {SEED}, re-seeded per statistic. Read-only; no table writes.\n")

    # ── G1a ──────────────────────────────────────────────────────────────
    say("## G1a — decomposition reproduces CR-AS (CR-AS settlement, all CR-AS rows)\n")
    dev = (df.call_pnl_cras + df.put_pnl_cras - df.cras_net_pts).abs()
    ok_rows = bool((dev < 0.005).all())
    say(f"- Per row |call_pnl + put_pnl − CR-AS net_pts|: max {dev.max():.2e} pts over {len(df)} rows → **{'PASS' if ok_rows else 'FAIL'}** (to the cent).")
    g1a_ok = ok_rows
    for b in BOXES:
        s = df[df.box == b]
        m = float(s.condor_pnl_cras_im.mean())
        okm = f"{m:+.3f}" == f"{CR_AS_POOLED[b]:+.3f}"
        g1a_ok &= okm
        say(f"- {b} pooled mean net (IM), n {len(s)}: {m:+.5f} → printed {m:+.3f} vs CR-AS {CR_AS_POOLED[b]:+.3f} → **{'PASS' if okm else 'FAIL'}**")
    say(f"- Entry minute and condor credit re-derived from quotes match every CR-AS row (asserted).")
    say(f"\n**G1a: {'PASS' if g1a_ok else 'FAIL'}**\n")
    if not g1a_ok:
        (OUT_DIR / "report.md").write_text("\n".join(L) + "\n")
        sys.exit("G1a failed — stopping before any read")

    # ── universe ─────────────────────────────────────────────────────────
    U = df[df.in_universe].copy()
    excl = sorted(set(AM_MIXED) | set(NO_HARNESS_SETTLE))
    say("## Universe (A1.3)\n")
    say(f"- Dropped, AM/PM mixed third Fridays (8): {', '.join(AM_MIXED)}")
    say(f"- Dropped, no harness settlement (3): {', '.join(NO_HARNESS_SETTLE)}")
    say(f"- 11 entries, {len(excl)} distinct dates (2023-05-19 is in both lists). Of these, {len(set(excl) & set(df.date))} were in the CR-AS priced set.")
    say(f"- **Universe: {U.date.nunique()} dates**; ±0.5 boxes {sum(U.box == 'pm0.5')}, ±1.0 boxes {sum(U.box == 'pm1')}; both structures {U.groupby('date').box.nunique().eq(2).sum()}.")
    assert U.close_harness.notna().all()
    unj = sorted(U[U.gamma_open.isna()].date.unique())
    say(f"- CR-BK gamma unjoined in the universe ({len(unj)}): {', '.join(unj)}")
    say(f"- IV rank present on {U[U.iv_7d_pct.notna()].date.nunique()} universe dates (first {U[U.iv_7d_pct.notna()].date.min()}).\n")

    # ── G1b ──────────────────────────────────────────────────────────────
    say("## G1b — CR-AS pooled condor P&L under both settlements, same 713-date universe (diagnostic)\n")
    say("| box | n | condor net, CR-AS settlement (IM) | condor net, harness settlement (IM) | Δ (harness − CR-AS) | CR-AS CI half-width | changed boxes | ↑ / ↓ | mean change in changed boxes (IM) |")
    say("|---|---|---|---|---|---|---|---|---|")
    g1b = {}
    for b in BOXES:
        s = U[U.box == b]
        a, h = boot(s.condor_pnl_cras_im), boot(s.condor_pnl_im)
        ch = s.condor_pnl_im - s.condor_pnl_cras_im
        chg = ch[ch.abs() * s.im >= 0.005]
        hw = (a[2] - a[1]) / 2
        g1b[b] = {"cras": a, "harness": h, "delta": h[0] - a[0], "halfwidth": hw}
        say(f"| {b} | {len(s)} | {f3(a)} | {f3(h)} | {h[0] - a[0]:+.4f} | {hw:.4f} | {len(chg)} | {int((chg > 0).sum())} / {int((chg < 0).sum())} | {chg.mean():+.4f} |")
    say("")
    say("Changed boxes by wing (wing intrinsic differs by ≥ 0.005 pts), universe only:\n")
    say("| box | wing | changed | harness worse / better | mean change (IM) |")
    say("|---|---|---|---|---|")
    for b in BOXES:
        s = U[U.box == b]
        for w in ("call", "put"):
            ch = (s[f"{w}_pnl"] - s[f"{w}_pnl_cras"]) / s.im
            c = ch[ch.abs() * s.im >= 0.005]
            say(f"| {b} | {w} | {len(c)} | {int((c < 0).sum())} / {int((c > 0).sum())} | {c.mean() if len(c) else 0:+.4f} |")
    say("")
    # CR-AS H4 (magnet-above, ±0.5: put-wing loss − call-wing loss, IM) re-settled
    s = U[(U.box == "pm0.5") & (U.regime == "magnet-above")]
    h4a = boot((s.put_loss_cras - s.call_loss_cras) / s.im)
    h4h = boot((s.put_loss - s.call_loss) / s.im)
    say(f"CR-AS H4 re-settled (magnet-above, ±0.5, put-wing loss − call-wing loss, IM, n {len(s)}): CR-AS settlement {f3(h4a)}; harness settlement {f3(h4h)}.")
    s = U[U.box == "pm0.5"]
    h1a, h1h = boot(s.condor_pnl_cras_im), boot(s.condor_pnl_im)
    moved = {b: abs(g1b[b]["delta"]) > g1b[b]["halfwidth"] for b in BOXES}
    say(f"A1.7 test — does re-settlement move the pooled condor P&L by more than its CI half-width? "
        f"±0.5: {'yes' if moved['pm0.5'] else 'no'}; ±1.0: {'yes' if moved['pm1'] else 'no'}.\n")

    # ── main read ────────────────────────────────────────────────────────
    say("## Main read — harness settlement (A1.1), by wing\n")

    def wing_rows(sub, label):
        out = []
        for b in BOXES:
            s = sub[sub.box == b]
            if len(s) == 0:
                continue
            for w in ("call", "put", "condor"):
                x, xi = s[f"{w}_pnl"], s[f"{w}_pnl_im"]
                wr = wilson(int((x > 0).sum()), len(x))
                out.append(f"| {label} | {b} | {w} | {len(s)} | {f3(boot(x))} | {f3(boot(x, np.median))} | {f3(boot(xi))} | "
                           f"{f3(boot(xi, np.median))} | {f3(wr, pct=True)} |")
        return out

    hdr = ("| split | box | wing | n | mean P&L (pts) | median (pts) | mean (IM) | median (IM) | win rate [Wilson 95 %] |\n"
           "|---|---|---|---|---|---|---|---|---|")
    say("### All dates\n")
    say(hdr)
    for line in wing_rows(U, "all"):
        say(line)
    say("")
    verdicts = {}
    for b in BOXES:
        s = U[U.box == b]
        ct, pt = boot(s.call_pnl), boot(s.put_pnl)
        cti, pti = boot(s.call_pnl_im), boot(s.put_pnl_im)
        verdicts[b] = (wing_verdict(ct, pt), wing_verdict(cti, pti), ct, pt)
        say(f"- L5 {b} (points): call {f3(ct)}, put {f3(pt)} → **{verdicts[b][0]}**; in IM: call {f3(cti)}, put {f3(pti)} → {verdicts[b][1]}")
    vs = [verdicts[b][0] for b in BOXES]
    headline = "PASS" if all(v == "PASS" for v in vs) else ("KILL S3-as-designed" if all(v == "KILL" for v in vs) else "not supported, not killed")
    say(f"\n**L5 verdict: {headline}** (±0.5: {vs[0]}; ±1.0: {vs[1]}).\n")

    say("### By 7-day IV tercile (`iv_7d_pct`, CR-BL cuts)\n")
    say(hdr)
    for t in ("low", "mid", "high"):
        for line in wing_rows(U[U.iv_tercile == t], f"IV {t}"):
            say(line)
    say(f"\nn without an IV rank: {int(U.iv_tercile.isna().sum())} rows.\n")
    say("### By CR-BK net-gamma sign at the open\n")
    say(hdr)
    for t in ("positive", "negative"):
        for line in wing_rows(U[U.gamma_sign == t], f"gamma {t}"):
            say(line)
    say(f"\nn without gamma: {int(U.gamma_sign.isna().sum())} rows.\n")
    say("### By regime as CR-AS labelled it\n")
    say(hdr)
    for line in wing_rows(U[U.regime == "magnet-above"], "magnet-above"):
        say(line)
    for line in wing_rows(U[U.regime != "magnet-above"], "other"):
        say(line)
    say("")

    say("### Call − put per date (harness)\n")
    for b in BOXES:
        s = U[U.box == b]
        say(f"- {b}: mean call − put {f3(boot(s.call_pnl - s.put_pnl))} pts; {f3(boot(s.call_pnl_im - s.put_pnl_im))} IM; "
            f"credit call {s.call_credit.mean():.3f} / put {s.put_credit.mean():.3f} pts; "
            f"intrinsic loss call {s.call_loss.mean():.3f} / put {s.put_loss.mean():.3f} pts; "
            f"max-loss hits (loss = 10) call {int((s.call_loss >= 10 - 1e-9).sum())} / put {int((s.put_loss >= 10 - 1e-9).sum())}; "
            f"any touch of the short strike: call {int((s.call_loss > 0).sum())} / put {int((s.put_loss > 0).sum())}")
    say("")

    # ── L7 ───────────────────────────────────────────────────────────────
    say("## L7 placebo — call and put spreads re-paired across dates\n")
    say("| box | n | call − put, paired bootstrap (pts) | call − put, shuffled pairing (pts) | corr(call, put) same day | corr shuffled | condor mean, same day (pts) | condor mean, shuffled (pts) |")
    say("|---|---|---|---|---|---|---|---|")
    for b in BOXES:
        s = U[U.box == b].reset_index(drop=True)
        perm = np.random.default_rng(SEED).permutation(len(s))
        c, p = s.call_pnl.to_numpy(), s.put_pnl.to_numpy()
        ps = p[perm]
        say(f"| {b} | {len(s)} | {f3(boot(c - p))} | {f3(boot(c - ps))} | {f3(boot_corr(c, p))} | {f3(boot_corr(c, ps))} | "
            f"{f3(boot(c + p))} | {f3(boot(c + ps))} |")
    say("\nA permutation leaves each wing's mean unchanged, so the mean call − put difference is identical under both pairings by construction; "
        "only its CI and the condor's dispersion can move.")
    off = df.groupby("box").apply(lambda s: int((s.entry_minute != "06:33").sum()))
    say(f"Entry minute ≠ 06:33 (where the joint credit filter could have moved the fill): {off.to_dict()} rows.\n")

    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n")
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "headline": headline, "verdicts": {b: verdicts[b][:2] for b in BOXES},
        "g1a": g1a_ok, "g1b": {b: {k: v for k, v in g1b[b].items()} for b in BOXES}, "a17_moved": moved,
        "h4_cras": h4a, "h4_harness": h4h, "universe_dates": int(U.date.nunique())}, default=float, indent=1))
    print(f"\nwritten {OUT_DIR / 'report.md'}, {OUT_DIR / 'legs_by_wing.parquet'}")


if __name__ == "__main__":
    main()
