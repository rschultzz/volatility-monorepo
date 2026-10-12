#!/usr/bin/env python3
"""CR-BK — T1 (touch, above-open) and T3 (volatility) on the corrected gamma profile. READ-ONLY. One run.

Spec: specs/CR-BK-corrected-gamma-profile-test.md (L1–L10, A1, A2). Reuses the matching logic and the month-cluster
bootstrap of scripts/cr_bi_followup_placebo.py (CR-BI follow-up, commit b05954d). Before any CR-BK outcome is computed,
the CR-BI P1/P2 output is regenerated through this file's matching functions and compared with the published log
scripts/logs/cr_bi_followup_placebo.md (all lines but the timestamp); any difference aborts the run (A2).

Inputs: scripts/logs/cr_bk/shape_summary.parquet and profiles.parquet from `scripts/cr_bk_profile.py shape`.
Usage:  apps/web/.venv/bin/python scripts/cr_bk_placebo.py [--control-only]
Output: scripts/logs/cr_bk_placebo.md (+ control diff in scripts/logs/cr_bk/control.md)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import cr_bi_followup_placebo as bi  # noqa: E402  (boot_ci, line, wall_carry, constants)
from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.spx_cash import build_series, daily_ohlc, fetch_minutes  # noqa: E402

SPLIT, VERSION, SEED, N_BOOT = bi.SPLIT, bi.VERSION, bi.SEED, bi.N_BOOT
CR_BI_LOG = REPO_ROOT / "scripts" / "logs" / "cr_bi_followup_placebo.md"
BK_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bk"
OUT = REPO_ROOT / "scripts" / "logs" / "cr_bk_placebo.md"
TREND_PCT, TREND_LB = 0.03, 20
T3_WINDOW, T3_MIN = 250, 120


# ── harness base (verbatim CR-BI setup) ─────────────────────────────────────
def build_context(conn) -> dict:
    feats = conn.execute(
        """SELECT f.trade_date, f.regime_at_classification, (f.feature_vector->>'implied_move_1d')::float,
                  o.horizon_sessions
           FROM bt_daily_features f LEFT JOIN bt_daily_outcomes o
             ON o.ticker=f.ticker AND o.trade_date=f.trade_date AND o.feature_version=f.feature_version AND o.active
           WHERE f.ticker='SPX' AND f.feature_version=%s AND f.active AND f.trade_date <= %s ORDER BY 1""",
        (VERSION, SPLIT)).fetchall()
    walls = {d: (w if isinstance(w, list) else []) for d, w in conn.execute(
        "SELECT trade_date, walls FROM orats_gex_landscape WHERE ticker='SPX' AND trade_date <= %s", (SPLIT,)).fetchall()}
    clean, _ = build_series(fetch_minutes(conn, dt.date(2023, 5, 1), SPLIT))
    spx = daily_ohlc(clean)[["open", "high", "low", "close"]]
    assert max(spx.index) <= SPLIT
    sess = list(spx.index)
    pos = {d: i for i, d in enumerate(sess)}
    base = pd.DataFrame([{"d": d, "regime": r, "im": im, "hz": hz} for d, r, im, hz in feats
                         if im and im > 0 and d in pos])
    O = spx["open"].to_numpy()
    base["open"] = [O[pos[d]] for d in base.d]
    base["month"] = [f"{d.year}-{d.month:02d}" for d in base.d]
    return {"conn": conn, "walls": walls, "spx": spx, "sess": sess, "pos": pos, "base": base,
            "H": spx["high"].to_numpy(), "L": spx["low"].to_numpy(), "O": O, "C": spx["close"].to_numpy()}


def touch_pool(ctx: dict, h: int) -> pd.DataFrame:
    """CR-BI P1 pool: base days whose h-session window ends inside train; u = (max high − open) / IM."""
    pos, sess, H, O = ctx["pos"], ctx["sess"], ctx["H"], ctx["O"]
    pool = ctx["base"][[pos[d] + h - 1 < len(sess) for d in ctx["base"].d]].copy()
    pool["u"] = [(H[pos[d]:pos[d] + h].max() - O[pos[d]]) / im for d, im in zip(pool.d, pool.im)]
    return pool


def touch_eval(sig: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    """CR-BI P1 matching: sig has columns d, open, im, u, month, target. Adds z, actual, placebo (leave-one-out)."""
    sig = sig.copy()
    sig["z"] = (sig.target - sig.open) / sig.im
    sig["actual"] = sig.u >= sig.z
    u_all, d_all = pl.u.to_numpy(), pl.d.to_numpy()
    sig["placebo"] = [float(np.mean(u_all[d_all != d] >= z)) for d, z in zip(sig.d, sig.z)]
    return sig


def band_frame(ctx: dict) -> pd.DataFrame:
    pos, H, L, O, C = ctx["pos"], ctx["H"], ctx["L"], ctx["O"], ctx["C"]
    b2 = ctx["base"].copy()
    b2["up"] = [(H[pos[d]] - O[pos[d]]) / im for d, im in zip(b2.d, b2.im)]
    b2["dn"] = [(O[pos[d]] - L[pos[d]]) / im for d, im in zip(b2.d, b2.im)]
    b2["cl"] = [(C[pos[d]] - O[pos[d]]) / im for d, im in zip(b2.d, b2.im)]
    return b2


def band_eval(b2: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """CR-BI P2 matching: sig has a, b (IM units above / below the open). Adds actual and leave-one-out placebo."""
    sig = sig.copy()
    sig["close_in"] = (sig.cl > -sig.b) & (sig.cl < sig.a)
    sig["range_in"] = (sig.dn < sig.b) & (sig.up < sig.a)
    up, dn, cl, dd = b2.up.to_numpy(), b2.dn.to_numpy(), b2.cl.to_numpy(), b2.d.to_numpy()
    sig["p_close"] = [float(np.mean(((cl > -b) & (cl < a))[dd != d])) for d, a, b in zip(sig.d, sig.a, sig.b)]
    sig["p_range"] = [float(np.mean(((dn < b) & (up < a))[dd != d])) for d, a, b in zip(sig.d, sig.a, sig.b)]
    return sig


def carry_fn(ctx: dict):
    base, walls = ctx["base"], ctx["walls"]
    items = sorted({(d, round(float(w["price"]), 4), int(w.get("sign", 0))) for d in base.d for w in walls.get(d, [])
                    if w.get("sign") in (1, -1) and w.get("price") is not None})
    carry = bi.wall_carry(ctx["conn"], items)

    def to_cash(d, w):
        c = carry.get((d, round(float(w["price"]), 4), int(w.get("sign", 0))))
        return None if c is None else float(w["price"]) - c
    return carry, items, to_cash


# ── control: regenerate the CR-BI log through the functions above ───────────
def control_lines(ctx: dict, carry, items, to_cash) -> list[str]:
    out: list[str] = []
    say = out.append
    base, sess, walls = ctx["base"], ctx["sess"], ctx["walls"]
    say(f"Train dates with IM > 0 and an SPX session: {len(base)} ({base.d.min()} → {base.d.max()}); SPX sessions {len(sess)}, "
        f"last {sess[-1]} (≤ split {SPLIT}). Frame: SPX cash vs strike-axis walls (P − carry).\n")
    say(f"Walls converted: {len(carry)} of {len(items)}; carry median {np.median(list(carry.values())):+.1f} pts, "
        f"p90 {np.quantile(list(carry.values()), .9):+.1f}.\n")
    say("## P1 — touch: magnet-above vs the same IM-scaled distance on all other days\n")
    say("| horizon / pool | n | actual | placebo | difference | 95 % CI (month-cluster bootstrap) |")
    say("| --- | --- | --- | --- | --- | --- |")
    for h in (5, 20):
        pool = touch_pool(ctx, h)
        sig = pool[(pool.regime == "magnet-above") & (pool.hz == h)].copy()
        tgt = []
        for d in sig.d:
            positive = [w for w in walls.get(d, []) if w.get("sign") == 1]
            w = max(positive, key=lambda x: x["gex"]) if positive else None
            assert w is None or float(w["price"]) == pick_drift_target(walls.get(d, []))
            tgt.append(to_cash(d, w) if w else None)
        sig["target"] = tgt
        sig = sig[sig.target.notna()].copy()
        for pool_name, pl in (("all other days", pool), ("secondary: non-magnet-above days", pool[pool.regime != "magnet-above"])):
            s = touch_eval(sig, pl)
            say(bi.line(f"{h}-session · {pool_name} (pool {len(pl)})", s, "actual", "placebo"))
        say(f"| _{h}-session z = (target − open)/IM: median {s.z.median():.2f}, p10 {s.z.quantile(.1):.2f}, "
            f"p90 {s.z.quantile(.9):.2f}; z ≤ 0 on {int((s.z <= 0).sum())} rows_ | | | | | |")
    say("")
    say("## P2 — containment: two-sided wall band vs equal-shape bands on all other days (t0 session)\n")
    b2 = band_frame(ctx)
    A, B = [], []
    for d, o in zip(b2.d, b2.open):
        lv = [x for x in (to_cash(d, w) for w in walls.get(d, []) if w.get("sign") in (1, -1)) if x is not None]
        below, above = [x for x in lv if x < o], [x for x in lv if x > o]
        A.append((min(above) - o) if above else np.nan)
        B.append((o - max(below)) if below else np.nan)
    b2["a"], b2["b"] = np.array(A) / b2.im, np.array(B) / b2.im
    sig = band_eval(b2, b2[b2.a.notna() & b2.b.notna()])
    say(f"Rows with a wall both sides of the open after conversion: {len(sig)} of {len(b2)}; band half-widths in IM units: "
        f"above median {sig.a.median():.2f}, below median {sig.b.median():.2f}.\n")
    for label, a_col, p_col in (("close-inside", "close_in", "p_close"), ("range-inside", "range_in", "p_range")):
        say(f"### {label}\n")
        say("| regime | n | actual | placebo | difference | 95 % CI (month-cluster bootstrap) |")
        say("| --- | --- | --- | --- | --- | --- |")
        say(bi.line("pooled", sig, a_col, p_col))
        for rg, g in sorted(sig.groupby("regime"), key=lambda kv: -len(kv[1])):
            say(bi.line(rg, g, a_col, p_col))
        say("")
    return out


# ── month-cluster bootstrap for arbitrary statistics (same seed / resamples) ─
def boot_stat(df: pd.DataFrame, stat) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    months = df.month.unique()
    groups = {m: g for m, g in df.groupby("month")}
    idx = rng.integers(0, len(months), size=(N_BOOT, len(months)))
    est = np.array([stat(pd.concat([groups[months[j]] for j in row], ignore_index=True)) for row in idx])
    est = est[np.isfinite(est)]
    return float(stat(df)), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def spear(col):
    return lambda x: spearmanr(x["rank"], x[col]).statistic


def tercile_diff(col):
    return lambda x: x.loc[x["rank"] > 2 / 3, col].mean() - x.loc[x["rank"] < 1 / 3, col].mean()


def flip_diff(col):
    return lambda x: x.loc[x.below_flip, col].mean() - x.loc[~x.below_flip, col].mean()


def pct_rank(x: pd.Series) -> pd.Series:
    vals, out = x.to_numpy(), []
    for i, v in enumerate(vals):
        prior = vals[max(0, i - T3_WINDOW):i]
        out.append(np.nan if len(prior) < T3_MIN else ((prior < v).sum() + 0.5 * (prior == v).sum()) / len(prior))
    return pd.Series(out, index=x.index)


def main() -> None:
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731

    ctx = build_context(conn)
    carry, items, to_cash = carry_fn(ctx)

    # ── control (A2): must equal the published CR-BI log, timestamp line excluded ──
    ctrl = control_lines(ctx, carry, items, to_cash)
    published = CR_BI_LOG.read_text().split("\n")[2:]          # timestamp header "# …\\n" = header line + blank
    mine = "\n".join(ctrl).split("\n")
    BK_DIR.mkdir(parents=True, exist_ok=True)
    (BK_DIR / "control.md").write_text("\n".join(mine))
    if mine != published:
        import difflib
        print("\n".join(difflib.unified_diff(published, mine, "published", "regenerated", lineterm="")))
        sys.exit("CONTROL FAILED — regenerated CR-BI output differs from the published log. Stopping (A2).")
    if "--control-only" in sys.argv:
        print(f"CONTROL PASS ({len(published)} lines identical); --control-only: no CR-BK outcome computed.")
        return
    say(f"# CR-BK — T1 / T3 (read-only, one run; {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    say(f"**Control: PASS** — CR-BI P1/P2 output regenerated through this file's matching functions is identical to "
        f"`scripts/logs/cr_bi_followup_placebo.md` ({len(published)} lines, timestamp line excluded).\n")

    shape = pd.read_parquet(BK_DIR / "shape_summary.parquet")
    shape = shape[shape.usable].set_index("d")
    base, pos, C = ctx["base"], ctx["pos"], ctx["C"]

    # ── T1 ──────────────────────────────────────────────────────────────────
    say("## T1 — touch at the corrected peak-gamma level (above-open only)\n")
    say("Level = highest positive peak of the corrected net profile (linear wings). Placebo = the same IM-scaled distance "
        "on all other train days (CR-BI P1 matching). Pass: difference ≥ +8 pts at either horizon and CI lower bound > 0.\n")
    say("| horizon · level source · subset | n | actual | placebo | difference | 95 % CI (month-cluster bootstrap) |")
    say("| --- | --- | --- | --- | --- | --- |")

    def trend_up(d):
        i = pos[d]
        return None if i < TREND_LB + 1 else (C[i - 1] / C[i - 1 - TREND_LB] - 1) > TREND_PCT

    t1 = {}
    for h in (5, 20):
        pool = touch_pool(ctx, h)
        new = pool[[d in shape.index and pd.notna(shape.at[d, "top_pos_linear"]) for d in pool.d]].copy()
        new["target"] = [float(shape.at[d, "top_pos_linear"]) for d in new.d]
        new = touch_eval(new[new.target > new.open], pool)
        new["trend"] = [trend_up(d) for d in new.d]
        old = pool.copy()
        tg = []
        for d in old.d:
            positive = [w for w in ctx["walls"].get(d, []) if w.get("sign") == 1]
            tg.append(to_cash(d, max(positive, key=lambda x: x["gex"])) if positive else None)
        old["target"] = tg
        old = old[old.target.notna()]
        old = touch_eval(old[old.target > old.open], pool)
        t1[h] = new
        say(bi.line(f"**{h}-session · corrected · all above-open (pool {len(pool)})**", new, "actual", "placebo"))
        say(bi.line(f"{h}-session · corrected · secondary: trend-up (20-session return > 3 %)", new[new.trend == True], "actual", "placebo"))  # noqa: E712
        say(bi.line(f"{h}-session · corrected · secondary: other days", new[new.trend == False], "actual", "placebo"))  # noqa: E712
        say(bi.line(f"{h}-session · old landscape (strike-axis walls) · all above-open", old, "actual", "placebo"))
        say(f"| _{h}-session corrected z: median {new.z.median():.2f}, p10 {new.z.quantile(.1):.2f}, p90 {new.z.quantile(.9):.2f}; "
            f"level − open median {(new.target - new.open).median():+.1f} pts; trend-up n {int((new.trend == True).sum())}, "  # noqa: E712
            f"no 21-session history {int(new.trend.isna().sum())}. Old z median {old.z.median():.2f}_ | | | | | |")
    say("")

    def t1_pass(s):
        lo, _ = bi.boot_ci(s, "actual", "placebo")
        diff = s.actual.astype(float).mean() - s.placebo.mean()
        return diff >= 0.08 and lo > 0, diff, lo
    p1 = {h: t1_pass(t1[h]) for h in (5, 20)}
    T1_PASS = any(v[0] for v in p1.values())

    # ── T3 ──────────────────────────────────────────────────────────────────
    say("## T3 — volatility vs corrected net gamma at the open\n")
    prof = pd.read_parquet(BK_DIR / "profiles.parquet")
    prof = prof[prof.wing == "linear"]
    pgrid = {d: (g.price.to_numpy(), g.gex_total.to_numpy()) for d, g in prof.groupby("d")}
    old_land = {d: (np.array([p["price"] for p in ls]), np.array([p["gex_total"] for p in ls])) for d, ls in conn.execute(
        "SELECT trade_date, landscape FROM orats_gex_landscape WHERE ticker='SPX' AND trade_date <= %s", (SPLIT,)).fetchall()
        if isinstance(ls, list) and ls}
    H, L = ctx["H"], ctx["L"]

    def x_at(grids, d, o):
        if d not in grids:
            return np.nan
        p, g = grids[d]
        return float(np.interp(o, p, g)) if p.min() <= o <= p.max() else np.nan

    t3 = base.copy()
    t3["y1"] = [(H[pos[d]] - L[pos[d]]) / o for d, o in zip(t3.d, t3.open)]
    t3["y2"] = [(H[pos[d]] - L[pos[d]]) / im for d, im in zip(t3.d, t3.im)]
    res = {}
    for name, grids in (("corrected", pgrid), ("old landscape (as built)", old_land)):
        s = t3.copy()
        s["x"] = [x_at(grids, d, o) for d, o in zip(s.d, s.open)]
        n_outside = int(s.x.isna().sum())
        s = s[s.x.notna()].sort_values("d").reset_index(drop=True)
        s["rank"] = pct_rank(s.x)
        s = s[s["rank"].notna()].copy()
        res[name] = s
        say(f"### {name} — n {len(s)} (X missing on {n_outside} base days; first ranked day {s.d.min()})\n")
        say("| statistic | Y1 = range / open | 95 % CI | Y2 = range / IM | 95 % CI |")
        say("| --- | --- | --- | --- | --- |")
        r1, r2 = boot_stat(s, spear("y1")), boot_stat(s, spear("y2"))
        say(f"| Spearman(rank X, Y) | {r1[0]:+.3f} | [{r1[1]:+.3f}, {r1[2]:+.3f}] | {r2[0]:+.3f} | [{r2[1]:+.3f}, {r2[2]:+.3f}] |")
        top, bot = s[s["rank"] > 2 / 3], s[s["rank"] < 1 / 3]
        say(f"| top-tercile mean (n {len(top)}) | {100 * top.y1.mean():.3f} % | | {top.y2.mean():.3f} | |")
        say(f"| bottom-tercile mean (n {len(bot)}) | {100 * bot.y1.mean():.3f} % | | {bot.y2.mean():.3f} | |")
        d1, d2 = boot_stat(s, tercile_diff("y1")), boot_stat(s, tercile_diff("y2"))
        say(f"| top − bottom | {100 * d1[0]:+.3f} pp | [{100 * d1[1]:+.3f}, {100 * d1[2]:+.3f}] | {d2[0]:+.3f} | [{d2[1]:+.3f}, {d2[2]:+.3f}] |")
        if name == "corrected":
            res["r1"] = r1
            fl = s[[d in shape.index and pd.notna(shape.at[d, "flip_nearest_linear"]) for d in s.d]].copy()
            fl["below_flip"] = [o < float(shape.at[d, "flip_nearest_linear"]) for d, o in zip(fl.d, fl.open)]
            b, a = fl[fl.below_flip], fl[~fl.below_flip]
            say(f"| open below zero-gamma flip (n {len(b)}) | {100 * b.y1.mean():.3f} % | | {b.y2.mean():.3f} | |")
            say(f"| open above zero-gamma flip (n {len(a)}) | {100 * a.y1.mean():.3f} % | | {a.y2.mean():.3f} | |")
            f1, f2 = boot_stat(fl, flip_diff("y1")), boot_stat(fl, flip_diff("y2"))
            say(f"| below − above | {100 * f1[0]:+.3f} pp | [{100 * f1[1]:+.3f}, {100 * f1[2]:+.3f}] | {f2[0]:+.3f} | [{f2[1]:+.3f}, {f2[2]:+.3f}] |")
        say("")
    r1 = res["r1"]
    T3_PASS = r1[0] <= -0.10 and r1[2] < 0

    # ── verdict ─────────────────────────────────────────────────────────────
    say("## Pass / fail (locked criteria)\n")
    for h in (5, 20):
        ok, diff, lo = p1[h]
        say(f"- T1 {h}-session: difference {100 * diff:+.1f} pts, CI lower {100 * lo:+.1f} → {'PASS' if ok else 'fail'}")
    say(f"- T1: **{'PASS' if T1_PASS else 'FAIL'}** (either horizon)")
    say("- T2: dropped (A2 — 0 usable days)")
    say(f"- T3 raw (Y1): Spearman {r1[0]:+.3f}, CI [{r1[1]:+.3f}, {r1[2]:+.3f}] → **{'PASS' if T3_PASS else 'FAIL'}**")
    say("- T3 priced (Y2): reported, not pass/fail")
    if T1_PASS:
        verdict = "T1 passes → levels carry information; follow-up CR builds the parallel pipeline and confirms once on the holdout."
    elif T3_PASS:
        verdict = "Only T3 raw passes → gamma matters for volatility, not for levels; redirect toward vol-conditioned structures, no wall targets."
    else:
        verdict = "Nothing passes → wall-as-level is retired; record in State of Knowledge and stop work on the landscape."
    say(f"\n**Verdict (spec rules):** {verdict}")
    OUT.write_text("\n".join(out))
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
