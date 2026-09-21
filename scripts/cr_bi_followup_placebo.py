#!/usr/bin/env python3
"""Follow-up CR Step 0 — placebo tests P1 (touch) and P2 (containment). READ-ONLY. One look each.

Pre-registered in specs/CR-BI-spx-cash-canonical-outcomes.md ("Pre-registration — follow-up CR
Step 0, placebo tests") before this file was run. Frame: SPX cash (CR-BH hybrid daily OHLC) vs
wall levels put back on the strike axis (P − kernel-weighted carry). Train only: signal date and
the whole evaluation window ≤ 2026-06-05. CI: cluster bootstrap over calendar months of the
signal date, 10 000 resamples, seed 20260920, 95 % percentile.

Usage: apps/web/.venv/bin/python scripts/cr_bi_followup_placebo.py
Output: scripts/logs/cr_bi_followup_placebo.md
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.spx_cash import build_series, daily_ohlc, fetch_minutes  # noqa: E402

SPLIT = dt.date(2026, 6, 5)
VERSION = "v0.6.0-openiv"
SEED, N_BOOT = 20260920, 10_000
OUT = REPO_ROOT / "scripts" / "logs" / "cr_bi_followup_placebo.md"


def wall_carry(conn, items: list[tuple[dt.date, float, int]]) -> dict:
    """{(date, price, sign): kernel-weighted mean(discounted_level − strike)} over same-sign net-GEX rows."""
    out = {}
    for i in range(0, len(items), 80):
        vals = ",".join(f"('{d}'::date,{p}::float8,{s})" for d, p, s in items[i:i + 80])
        rows = conn.execute(f"""
            WITH t(trade_date, price, sgn) AS (VALUES {vals}),
            k AS (SELECT t.trade_date, t.price, t.sgn, g.discounted_level - g.strike AS carry,
                         abs(g.gex_call - abs(g.gex_put)) * exp(-power(t.price - g.discounted_level, 2)
                            / (2 * 64 * greatest(coalesce(g.dte, 30), 0.5))) AS w
                  FROM orats_oi_gamma g JOIN t USING (trade_date)
                  WHERE g.ticker = 'SPX' AND g.discounted_level IS NOT NULL AND g.expir_date >= g.trade_date
                    AND sign(g.gex_call - abs(g.gex_put)) = t.sgn
                    AND abs(t.price - g.discounted_level) <= 6 * 8 * sqrt(greatest(coalesce(g.dte, 30), 0.5)))
            SELECT trade_date, price, sgn, sum(w * carry) / nullif(sum(w), 0) FROM k GROUP BY 1, 2, 3""").fetchall()
        for d, p, s, c in rows:
            if c is not None:
                out[(d, round(float(p), 4), int(s))] = float(c)
    return out


def boot_ci(df: pd.DataFrame, col_a: str, col_p: str) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    g = df.assign(diff=df[col_a].astype(float) - df[col_p]).groupby("month")["diff"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    est = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(est, .025)), float(np.quantile(est, .975))


def line(name: str, df: pd.DataFrame, a: str, p: str) -> str:
    if df.empty:
        return f"| {name} | 0 | | | | |"
    lo, hi = boot_ci(df, a, p)
    act, pla = df[a].astype(float).mean(), df[p].mean()
    flag = " ⚑ n<20" if len(df) < 20 else ""
    return (f"| {name}{flag} | {len(df)} | {100 * act:.1f} % | {100 * pla:.1f} % | {100 * (act - pla):+.1f} pts | "
            f"[{100 * lo:+.1f}, {100 * hi:+.1f}] |")


def main() -> None:
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    out: list[str] = []
    say = lambda s="": (print(s, flush=True), out.append(s))  # noqa: E731

    feats = conn.execute(
        """SELECT f.trade_date, f.regime_at_classification, (f.feature_vector->>'implied_move_1d')::float,
                  o.horizon_sessions
           FROM bt_daily_features f LEFT JOIN bt_daily_outcomes o
             ON o.ticker=f.ticker AND o.trade_date=f.trade_date AND o.feature_version=f.feature_version AND o.active
           WHERE f.ticker='SPX' AND f.feature_version=%s AND f.active AND f.trade_date <= %s ORDER BY 1""",
        (VERSION, SPLIT)).fetchall()
    walls = {d: (w if isinstance(w, list) else []) for d, w in conn.execute(
        "SELECT trade_date, walls FROM orats_gex_landscape WHERE ticker='SPX' AND trade_date <= %s", (SPLIT,)).fetchall()}

    # SPX cash daily frame — nothing after the split is fetched
    clean, _ = build_series(fetch_minutes(conn, dt.date(2023, 5, 1), SPLIT))
    spx = daily_ohlc(clean)[["open", "high", "low", "close"]]
    assert max(spx.index) <= SPLIT
    sess = list(spx.index)
    pos = {d: i for i, d in enumerate(sess)}
    H, O = spx["high"].to_numpy(), spx["open"].to_numpy()

    base = pd.DataFrame([{"d": d, "regime": r, "im": im, "hz": hz} for d, r, im, hz in feats
                         if im and im > 0 and d in pos])
    base["open"] = [O[pos[d]] for d in base.d]
    base["month"] = [f"{d.year}-{d.month:02d}" for d in base.d]
    say(f"# Follow-up Step 0 — placebo tests (read-only, one look; {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    say(f"Train dates with IM > 0 and an SPX session: {len(base)} ({base.d.min()} → {base.d.max()}); SPX sessions {len(sess)}, "
        f"last {sess[-1]} (≤ split {SPLIT}). Frame: SPX cash vs strike-axis walls (P − carry).\n")

    # carry for every wall on every base date
    items = sorted({(d, round(float(w["price"]), 4), int(w.get("sign", 0))) for d in base.d for w in walls.get(d, [])
                    if w.get("sign") in (1, -1) and w.get("price") is not None})
    carry = wall_carry(conn, items)
    say(f"Walls converted: {len(carry)} of {len(items)}; carry median {np.median(list(carry.values())):+.1f} pts, "
        f"p90 {np.quantile(list(carry.values()), .9):+.1f}.\n")

    def to_cash(d, w):
        c = carry.get((d, round(float(w["price"]), 4), int(w.get("sign", 0))))
        return None if c is None else float(w["price"]) - c

    # ── P1 touch ─────────────────────────────────────────────────────────────
    say("## P1 — touch: magnet-above vs the same IM-scaled distance on all other days\n")
    say("| horizon / pool | n | actual | placebo | difference | 95 % CI (month-cluster bootstrap) |")
    say("| --- | --- | --- | --- | --- | --- |")
    for h in (5, 20):
        pool = base[[pos[d] + h - 1 < len(sess) for d in base.d]].copy()      # window ends ≤ split by construction
        pool["u"] = [(H[pos[d]:pos[d] + h].max() - O[pos[d]]) / im for d, im in zip(pool.d, pool.im)]
        sig = pool[(pool.regime == "magnet-above") & (pool.hz == h)].copy()
        tgt = []
        for d in sig.d:
            positive = [w for w in walls.get(d, []) if w.get("sign") == 1]
            w = max(positive, key=lambda x: x["gex"]) if positive else None
            assert w is None or float(w["price"]) == pick_drift_target(walls.get(d, []))
            tgt.append(to_cash(d, w) if w else None)
        sig["target_a"] = tgt
        sig = sig[sig.target_a.notna()].copy()
        sig["z"] = (sig.target_a - sig.open) / sig.im
        sig["actual"] = sig.u >= sig.z
        for pool_name, pl in (("all other days", pool), ("secondary: non-magnet-above days", pool[pool.regime != "magnet-above"])):
            u_all, d_all = pl.u.to_numpy(), pl.d.to_numpy()
            sig["placebo"] = [float(np.mean(u_all[d_all != d] >= z)) for d, z in zip(sig.d, sig.z)]
            say(line(f"{h}-session · {pool_name} (pool {len(pl)})", sig, "actual", "placebo"))
        say(f"| _{h}-session z = (target − open)/IM: median {sig.z.median():.2f}, p10 {sig.z.quantile(.1):.2f}, "
            f"p90 {sig.z.quantile(.9):.2f}; z ≤ 0 on {int((sig.z <= 0).sum())} rows_ | | | | | |")
    say("")

    # ── P2 containment ───────────────────────────────────────────────────────
    say("## P2 — containment: two-sided wall band vs equal-shape bands on all other days (t0 session)\n")
    L, C = spx["low"].to_numpy(), spx["close"].to_numpy()
    b2 = base.copy()
    b2["up"] = [(H[pos[d]] - O[pos[d]]) / im for d, im in zip(b2.d, b2.im)]
    b2["dn"] = [(O[pos[d]] - L[pos[d]]) / im for d, im in zip(b2.d, b2.im)]
    b2["cl"] = [(C[pos[d]] - O[pos[d]]) / im for d, im in zip(b2.d, b2.im)]
    A, B = [], []
    for d, o in zip(b2.d, b2.open):
        lv = [x for x in (to_cash(d, w) for w in walls.get(d, []) if w.get("sign") in (1, -1)) if x is not None]
        below, above = [x for x in lv if x < o], [x for x in lv if x > o]
        A.append((min(above) - o) if above else np.nan)
        B.append((o - max(below)) if below else np.nan)
    b2["a"], b2["b"] = np.array(A) / b2.im, np.array(B) / b2.im
    sig = b2[b2.a.notna() & b2.b.notna()].copy()
    sig["close_in"] = (sig.cl > -sig.b) & (sig.cl < sig.a)
    sig["range_in"] = (sig.dn < sig.b) & (sig.up < sig.a)
    up, dn, cl, dd = b2.up.to_numpy(), b2.dn.to_numpy(), b2.cl.to_numpy(), b2.d.to_numpy()
    sig["p_close"] = [float(np.mean(((cl > -b) & (cl < a))[dd != d])) for d, a, b in zip(sig.d, sig.a, sig.b)]
    sig["p_range"] = [float(np.mean(((dn < b) & (up < a))[dd != d])) for d, a, b in zip(sig.d, sig.a, sig.b)]
    say(f"Rows with a wall both sides of the open after conversion: {len(sig)} of {len(b2)}; band half-widths in IM units: "
        f"above median {sig.a.median():.2f}, below median {sig.b.median():.2f}.\n")
    for label, a_col, p_col in (("close-inside", "close_in", "p_close"), ("range-inside", "range_in", "p_range")):
        say(f"### {label}\n")
        say("| regime | n | actual | placebo | difference | 95 % CI (month-cluster bootstrap) |")
        say("| --- | --- | --- | --- | --- | --- |")
        say(line("pooled", sig, a_col, p_col))
        for rg, g in sorted(sig.groupby("regime"), key=lambda kv: -len(kv[1])):
            say(line(rg, g, a_col, p_col))
        say("")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
