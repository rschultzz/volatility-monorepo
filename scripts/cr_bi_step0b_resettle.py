#!/usr/bin/env python3
"""CR-BI Step 0b — read-only re-settlement of the CR-AR debit reference on SPX cash.

NO DB WRITES: the session is forced read-only (default_transaction_read_only = on), the harness
main() is never called (so no bt_backfill_runs row, no bt_edge_backtest_results cells), and only
the harness's pure / SELECT-only functions are imported. Train only: universe_end = split_date =
2026-06-05; no signal date after it is loaded.

  A. identical fills, identical dates; settlement price = SPX cash (CR-BH hybrid series,
     scripts/cr_bh_spx_cash), last print in 12:50–13:00 PT on expiry day. Old (ES) vs new.
  B. flag trades whose expiry is a third Friday and whose captured legs are AM-settled.
  C. quote-based settlement where settlement-window option quotes exist (third column).
  D. frame B − frame A at the drift-target wall, per date: kernel-weighted mean of
     (discounted_level − strike) over the positive-GEX rows that build the wall.

Usage: apps/web/.venv/bin/python scripts/cr_bi_step0b_resettle.py [--use-minutes-cache]
Output: scripts/logs/cr_bi_step0b_resettle.md (+ _trades.csv), both git-ignored.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.cr_ah_step4_analysis as H  # noqa: E402
from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from scripts.cr_bh_spx_cash import build_series, fetch_minutes, is_lagged, normalize_minutes  # noqa: E402

UNIVERSE_END = SPLIT = dt.date(2026, 6, 5)
THRESHOLD = 0.05                     # the CR-AR reference cell
MODE = "walk-forward"
SETTLE_START, SETTLE_END = dt.time(12, 50), dt.time(13, 0)
MINUTES_CACHE = REPO_ROOT / "scripts" / ".cache" / "cr_bh_spot_stock_minutes.pkl"
OUT = REPO_ROOT / "scripts" / "logs" / "cr_bi_step0b_resettle.md"
PUBLISHED = {"all": (103, 1.7129, 0.5825, 1.6157, 0.0972), "near": (38, 1.9116, 0.6842, 1.9452, -0.0337),
             "mid": (34, 1.2349, 0.5588, 0.9143, 0.3206), "far": (31, 1.9937, 0.4839, 1.9598, 0.0339)}


def third_friday(d: dt.date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def pnl_with(td, fill, settle):
    """compute_pnl's close path with a substituted settlement underlying."""
    if fill is None or settle is None:
        return None
    return fill + H._DEBIT_PLUGIN.payoff(td.legs, settle)


def quote_settlement(conn, td):
    """Position value from the last valid settlement-window option-quote minute on expiry day."""
    opras = [H.format_opra(H.OPRA_ROOT, td.expiry_date, "C", s) for s in (td.short_strike, td.other_strike)]
    lo, hi = dt.datetime.combine(td.expiry_date, SETTLE_START), dt.datetime.combine(td.expiry_date, SETTLE_END)
    rows = conn.execute(
        "SELECT snapshot_pt, bid_price, ask_price, opra_symbol, expiry_tod FROM orats_options_minute "
        "WHERE opra_symbol = ANY(%s) AND snapshot_pt >= %s AND snapshot_pt <= %s ORDER BY snapshot_pt DESC",
        (opras, lo, hi)).fetchall()
    by_min: dict = {}
    for snap, bid, ask, opra, _tod in rows:
        key = H._opra_to_quote_key(opra)
        if key:
            by_min.setdefault(snap, []).append((key[0], key[1], bid, ask))
    width = abs(td.legs[0].strike - td.legs[1].strike)
    for snap in sorted(by_min, reverse=True):
        qmap, _ = H.build_quote_map(by_min[snap])
        val = H.net_price_from_real_quotes(td.legs, qmap)
        if val is not None and H.spread_value_is_valid(val, width, td.legs):
            return float(val), snap
    return None, None


def entry_day_tod(conn, td):
    opras = [H.format_opra(H.OPRA_ROOT, td.expiry_date, "C", s) for s in (td.short_strike, td.other_strike)]
    lo = dt.datetime.combine(td.trade_date, dt.time(0))
    rows = conn.execute(
        "SELECT expiry_tod, count(*) FROM orats_options_minute WHERE opra_symbol = ANY(%s) "
        "AND snapshot_pt >= %s AND snapshot_pt < %s GROUP BY 1", (opras, lo, lo + dt.timedelta(days=1))).fetchall()
    return {r[0]: r[1] for r in rows}


def cell(rows: pd.DataFrame, col: str) -> dict:
    x = rows[rows[col].notna()]
    b = rows[rows[col + "_base"].notna()]
    n = len(x)
    if not n:
        return {"n": 0}
    wins = int((x[col] > 0).sum())
    lo, hi = H.wilson_ci(wins, n)
    base = b[col + "_base"].mean() if len(b) else np.nan
    return {"n": n, "mean": x[col].mean(), "win": wins / n, "lo": lo, "hi": hi, "base_n": len(b),
            "base": base, "beat": x[col].mean() - base,
            "pp": x[col].sum() / x["width"].sum(), "beat_pp": x[col].sum() / x["width"].sum() - b[col + "_base"].sum() / b["width"].sum()}


def fmt(c: dict) -> str:
    if not c.get("n"):
        return "n=0"
    return (f"n={c['n']} mean {c['mean']:+.3f} win {100 * c['win']:.1f}% "
            f"[{100 * c['lo']:.1f}, {100 * c['hi']:.1f}] base {c['base']:+.3f} (n={c['base_n']}) beat {c['beat']:+.3f} "
            f"| per-pt {c['pp']:+.4f} beat/pt {c['beat_pp']:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-minutes-cache", action="store_true")
    args = ap.parse_args()

    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
    out: list[str] = []

    def say(s=""):
        print(s, flush=True)
        out.append(s)

    # ── the CR-AR debit trades, rebuilt through the harness's own functions ──
    entries = H.load_signal_entries(conn, universe_end=UNIVERSE_END, split_date=SPLIT)
    selected = H.select_clean_dates(entries)
    selected = [e for e in selected if e["partition"] == "train" and e["trade_date"] <= UNIVERSE_END]
    clean = H.filter_clean_for_structure(conn, selected, "debit")
    assert all(e["trade_date"] <= UNIVERSE_END for e in clean)
    trades = []
    for i, e in enumerate(clean):
        td = H.collect_trade_data(conn, e, "debit", MODE)
        if td and not td.excluded_reason:
            trades.append(td)
        if (i + 1) % 25 == 0:
            print(f"  collected {i + 1}/{len(clean)}", flush=True)
    say(f"# CR-BI Step 0b — CR-AR debit reference re-settled on SPX cash (read-only, {dt.datetime.now():%Y-%m-%d %H:%M})\n")
    say(f"Universe ≤ {UNIVERSE_END}, train only, mode {MODE}, T = {THRESHOLD}. selected {len(selected)} → debit clean {len(clean)} "
        f"→ trades {len(trades)}. No DB writes (session read-only; harness main() not called).\n")

    # ── SPX cash settlement series ───────────────────────────────────────────
    exp_lo, exp_hi = min(t.expiry_date for t in trades), max(t.expiry_date for t in trades)
    raw = (normalize_minutes(pd.read_pickle(MINUTES_CACHE)) if args.use_minutes_cache and MINUTES_CACHE.exists()
           else fetch_minutes(conn, exp_lo - dt.timedelta(days=7), exp_hi + dt.timedelta(days=7)))
    cleanm, _ = build_series(raw)
    w = cleanm[(cleanm["minute"].dt.time >= SETTLE_START) & (cleanm["minute"].dt.time <= SETTLE_END)]
    spx_settle = w.sort_values("minute").groupby("session_date").agg(px=("px", "last"), at=("minute", "last"))

    rows = []
    for td in trades:
        r = H.compute_pnl(td, THRESHOLD)
        fill, base = r["fill_net_credit"], r["baseline_net_credit"]
        sx = spx_settle["px"].get(td.expiry_date)
        sx = float(sx) if sx is not None and not pd.isna(sx) else None
        qv, qat = quote_settlement(conn, td)
        tod = entry_day_tod(conn, td) if third_friday(td.expiry_date) else {}
        rows.append({
            "trade_date": td.trade_date, "expiry": td.expiry_date, "band": td.band, "target": round(td.drift_target, 2),
            "long": min(td.short_strike, td.other_strike), "short": max(td.short_strike, td.other_strike),
            "width": float(td.width_actual), "filled": r["filled"], "fill": fill,
            "es_settle": td.settlement_price, "spx_settle": sx,
            "basis_expiry": (td.settlement_price - sx) if (td.settlement_price is not None and sx is not None) else None,
            "es": r["close_pnl"], "es_base": r["baseline_close_pnl"],
            "spx": pnl_with(td, fill, sx), "spx_base": pnl_with(td, base, sx),
            "quote": (fill + qv) if (fill is not None and qv is not None) else None,
            "quote_base": (base + qv) if (base is not None and qv is not None) else None,
            "quote_at": qat, "third_friday": third_friday(td.expiry_date), "entry_tod": tod,
            "lagged_expiry": is_lagged(td.expiry_date),
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT.with_name("cr_bi_step0b_resettle_trades.csv"), index=False)

    # ── A ────────────────────────────────────────────────────────────────────
    say("## A. Old (ES settlement) vs new (SPX-cash settlement) — gross pts, identical fills\n")
    say("Replica check vs the stored CR-AR cells (n, mean, win, baseline, beat):")
    for band in ("all", "near", "mid", "far"):
        s = df if band == "all" else df[df.band == band]
        c = cell(s, "es")
        p = PUBLISHED[band]
        say(f"- {band}: replica n={c['n']} mean {c['mean']:+.4f} win {c['win']:.4f} base {c['base']:+.4f} beat {c['beat']:+.4f} "
            f"| stored n={p[0]} mean {p[1]:+.4f} win {p[2]:.4f} base {p[3]:+.4f} beat {p[4]:+.4f}")
    say("")
    say(f"SPX settlement missing for {int(df.spx_settle.isna().sum())} trades; ES missing for {int(df.es_settle.isna().sum())}. "
        f"Expiries on lagged-series days: {int(df.lagged_expiry.sum())}.\n")
    both = df[df.es.notna() & df.spx.notna()]
    say("| band | settlement | cell |")
    say("| --- | --- | --- |")
    for band in ("all", "near", "mid", "far"):
        s = df if band == "all" else df[df.band == band]
        sb = both if band == "all" else both[both.band == band]
        say(f"| {band} | ES (old) | {fmt(cell(s, 'es'))} |")
        say(f"| {band} | SPX cash (new) | {fmt(cell(s, 'spx'))} |")
        say(f"| {band} | ES, same trades as SPX | {fmt(cell(sb, 'es'))} |")
    say("")
    b = df.basis_expiry.dropna()
    say(f"Basis on expiry day (ES settle − SPX settle), {len(b)} trades: mean {b.mean():+.1f}, median {b.median():+.1f}, "
        f"p10 {b.quantile(.1):+.1f}, p90 {b.quantile(.9):+.1f}, min {b.min():+.1f}, max {b.max():+.1f}.\n")
    ch = both[(both.es - both.spx).abs() > 1e-9].copy()
    ch["d"] = ch.spx - ch.es
    say(f"### Trades whose close P&L changes: {len(ch)} of {len(both)} (Σ change {ch.d.sum():+.2f} pts; "
        f"win→loss {int(((ch.es > 0) & (ch.spx <= 0)).sum())}, loss→win {int(((ch.es <= 0) & (ch.spx > 0)).sum())})\n")
    say("| trade_date | expiry | band | long/short | fill | ES settle | SPX settle | basis | P&L ES → SPX | Δ |")
    say("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in ch.sort_values("trade_date").itertuples():
        say(f"| {r.trade_date} | {r.expiry} | {r.band} | {r.long:g}/{r.short:g} | {r.fill:+.2f} | {r.es_settle:.2f} | "
            f"{r.spx_settle:.2f} | {r.basis_expiry:+.1f} | {r.es:+.2f} → {r.spx:+.2f} | {r.d:+.2f} |")
    say("")

    # ── B ────────────────────────────────────────────────────────────────────
    tf = df[df.third_friday]
    say(f"## B. Third-Friday expiries: {len(tf)} of {len(df)} trades ({int(tf.filled.sum())} filled)\n")
    say("`OPRA_ROOT = 'SPX'` for every leg; `orats_options_minute.expiry_tod` on the entry day shows which contract the quotes are:\n")
    say("| trade_date | expiry | band | entry-day quote rows by expiry_tod | flag |")
    say("| --- | --- | --- | --- | --- |")
    n_am = 0
    for r in tf.sort_values("trade_date").itertuples():
        tod = r.entry_tod or {}
        flag = "AM-settled" if set(tod) == {"am"} else ("AM+PM mixed under one symbol" if "am" in tod else "PM")
        n_am += "am" in tod
        say(f"| {r.trade_date} | {r.expiry} | {r.band} | {tod} | {flag} |")
    say(f"\nTrades with AM-settled quotes involved: **{n_am}**. For those a 13:00 PT print (ES or SPX) is the wrong settlement "
        f"(SET = opening prices that morning). Counted, not fixed.\n")

    # ── C ────────────────────────────────────────────────────────────────────
    q = df[df.quote.notna()]
    say(f"## C. Quote-based settlement (last valid spread mid, 12:50–13:00 PT on expiry) — available for {len(q)} of "
        f"{int(df.filled.sum())} filled trades\n")
    if len(q):
        three = df[df.quote.notna() & df.es.notna() & df.spx.notna()]
        say("| band | n | ES mean / win | SPX mean / win | quote mean / win | mean \\|SPX − quote\\| | mean \\|ES − quote\\| |")
        say("| --- | --- | --- | --- | --- | --- | --- |")
        for band in ("all", "near", "mid", "far"):
            s = three if band == "all" else three[three.band == band]
            if len(s):
                say(f"| {band} | {len(s)} | {s.es.mean():+.3f} / {100 * (s.es > 0).mean():.1f}% | {s.spx.mean():+.3f} / "
                    f"{100 * (s.spx > 0).mean():.1f}% | {s.quote.mean():+.3f} / {100 * (s.quote > 0).mean():.1f}% | "
                    f"{(s.spx - s.quote).abs().mean():.3f} | {(s.es - s.quote).abs().mean():.3f} |")
    say("")

    # ── D ────────────────────────────────────────────────────────────────────
    say("## D. Frame B − frame A at the drift-target wall\n")
    walls = conn.execute("SELECT trade_date, walls, table_spot FROM orats_gex_landscape WHERE ticker='SPX' "
                         "AND trade_date >= '2023-05-01' AND trade_date <= %s ORDER BY 1", (dt.date(2026, 9, 18),)).fetchall()
    tgt = [(d, float(pick_drift_target(wl if isinstance(wl, list) else []))) for d, wl, _ in walls
           if pick_drift_target(wl if isinstance(wl, list) else []) is not None]
    res = []
    for i in range(0, len(tgt), 60):
        chunk = tgt[i:i + 60]
        vals = ",".join(f"('{d}'::date,{t})" for d, t in chunk)
        res += conn.execute(f"""
            WITH t(trade_date, target) AS (VALUES {vals}),
            k AS (SELECT g.trade_date, g.discounted_level - g.strike AS carry, g.dte,
                         (g.gex_call - abs(g.gex_put)) * exp(-power(t.target - g.discounted_level, 2)
                            / (2 * 64 * greatest(coalesce(g.dte, 30), 0.5))) AS w
                  FROM orats_oi_gamma g JOIN t USING (trade_date)
                  WHERE g.ticker = 'SPX' AND g.discounted_level IS NOT NULL AND g.expir_date >= g.trade_date
                    AND abs(t.target - g.discounted_level) <= 6 * 8 * sqrt(greatest(coalesce(g.dte, 30), 0.5)))
            SELECT trade_date, sum(w * carry) / sum(w), sum(w * dte) / sum(w), count(*)
            FROM k WHERE w > 0 GROUP BY 1""").fetchall()
    d = pd.DataFrame(res, columns=["trade_date", "b_minus_a", "w_dte", "n"]).astype({"b_minus_a": float, "w_dte": float})
    d.to_csv(OUT.with_name("cr_bi_step0b_frame_b_minus_a.csv"), index=False)
    s = d.b_minus_a
    say(f"Per date, kernel-weighted mean of (`discounted_level` − `strike`) over the positive-net-GEX rows that build the wall at "
        f"`drift_target` (same Gaussian kernel as `gex_landscape`, σ = 8·√dte): **{len(d)} dates — median {s.median():+.2f} pts, "
        f"mean {s.mean():+.2f}, p75 {s.quantile(.75):+.2f}, p90 {s.quantile(.9):+.2f}, p99 {s.quantile(.99):+.2f}, max {s.max():+.2f}; "
        f"weighted DTE median {d.w_dte.median():.1f}, p90 {d.w_dte.quantile(.9):.1f}.** Dates > 5 pts: {(s > 5).sum()}; > 10: {(s > 10).sum()}.\n")
    tr = d[d.trade_date.isin(set(df.trade_date))].b_minus_a
    say(f"On the {len(tr)} CR-AR trade dates: median {tr.median():+.2f}, p90 {tr.quantile(.9):+.2f}, max {tr.max():+.2f}.\n")
    d["year"] = [x.year for x in d.trade_date]
    say("| year | dates | median | p90 | max |")
    say("| --- | --- | --- | --- | --- |")
    for y, g in d.groupby("year"):
        say(f"| {y} | {len(g)} | {g.b_minus_a.median():+.2f} | {g.b_minus_a.quantile(.9):+.2f} | {g.b_minus_a.max():+.2f} |")

    OUT.write_text("\n".join(out))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
