#!/usr/bin/env python3
"""CR-AS Step 2 — first priced read of the 0DTE condor on the train universe. READ-ONLY.

Per date × box (decisions 4, 5, 8, all as amended in Step 0):
  entry   = first minute ≥ 06:33 PT (≤ 06:45) where all four legs pass the
            CR-AN leg rule and the condor credit ∈ (0, min wing width]
            (condor_0dte.first_valid_entry); none → `no_valid_entry`
  close   = SPX spot_price at the last orats_monies_minute snapshot ≤ 13:00 PT
  sanity  = |basis@close − basis@06:33| ≤ 50 and basis@close ∈ [−50, +110] (G4)
  payoff  = intrinsic of the four legs at the close; gross = credit − loss,
            net = gross − 0.104 pts (4 legs × 2 sides × $1.30)
Tables per regime × box; H1–H6 with bootstrap 95 % CIs (seed 20260906);
the P-side breach-cost table from bt_daily_outcomes for the SAME dates.

Writes nothing except the bt_backfill_runs row (cr_id CR-AS-analysis) and the
markdown report at --out (default scripts/logs/cr_as_analysis_<ts>.md).

Usage:
    PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_as_condor_analysis.py [--out PATH] [--cr-id CR-AS-analysis]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time
from pathlib import Path

# ── ENV (before any import that reads DATABASE_URL) ──────────────────────────
def _find_dotenv() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        c = current / ".env"
        if c.exists():
            return c
        current = current.parent
    return None

_env_path = _find_dotenv()
if _env_path:
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

_bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = _bak_url

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np

from packages.shared.backfill_safety import assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke
from packages.shared.backtest.condor_0dte import (
    CONDOR_ROUND_TRIP_FEE_PTS, CONDOR_ROUND_TRIP_FEE_USD, WING_WIDTH, build_boxes, first_valid_entry, settle_condor,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.options_cache.opra import format_opra
from scripts.cr_as_capture_0dte_condor_legs import (
    BASIS_MAX, BASIS_MIN, OPEN_SNAPSHOT_CEIL, OPEN_SNAPSHOT_FLOOR, OPRA_ROOT, SPX_OPEN_SQL, TICKER,
    UNIVERSE_END, UNIVERSE_SQL, WINDOW_END, WINDOW_START,
)

ENTRY_FLOOR = time(6, 33)
CLOSE_CEIL = time(13, 0)
DRIFT_MAX = 50.0
CLOSE_BASIS_MIN, CLOSE_BASIS_MAX = -50.0, 110.0
SEED = 20260906
B = 10_000
MIN_N_REGIME = 40
REGIMES = ("magnetic-pin", "magnet-above", "amplification", "untethered", "bounded")
BOXES = ("pm0.5", "pm1")

SPX_CLOSE_SQL = """
SELECT spot_price, snapshot_pt FROM orats_monies_minute
WHERE ticker = %s AND trade_date = %s AND snapshot_pt <= %s AND spot_price IS NOT NULL
ORDER BY snapshot_pt DESC, dte ASC LIMIT 1
"""
BARS_SQL = """
SELECT snapshot_pt, strike, option_type, bid_price, ask_price
FROM orats_options_minute
WHERE opra_symbol = ANY(%s) AND snapshot_pt BETWEEN %s AND %s
ORDER BY snapshot_pt
"""


# ── statistics ───────────────────────────────────────────────────────────────

def boot_mean(x, rng, b=B):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(axis=1)
    return (float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def boot_diff(x, y, rng, b=B):
    """mean(x) − mean(y), independent bootstrap."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return (np.nan, np.nan, np.nan)
    mx = x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    my = y[rng.integers(0, len(y), size=(b, len(y)))].mean(axis=1)
    d = mx - my
    return (float(x.mean() - y.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def boot_ratio(num, den, rng, b=B):
    """mean(num) / mean(den) on paired rows."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0 or den.mean() == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(num), size=(b, len(num)))
    r = num[idx].mean(axis=1) / np.where(den[idx].mean(axis=1) == 0, np.nan, den[idx].mean(axis=1))
    return (float(num.mean() / den.mean()), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5)))


def fmt(t, nd=3):
    m, lo, hi = t
    if np.isnan(m):
        return "—"
    return f"{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def verdict(t, direction, threshold=0.0):
    """'supported' iff the CI excludes `threshold` in `direction` ('>' or '<');
    else 'directional (sign agrees)' / 'directional (sign disagrees)'."""
    m, lo, hi = t
    if np.isnan(m):
        return "n/a"
    if direction == ">":
        if lo > threshold:
            return "**supported**"
        return "directional (sign agrees)" if m > threshold else "directional (sign disagrees)"
    if hi < threshold:
        return "**supported**"
    return "directional (sign agrees)" if m < threshold else "directional (sign disagrees)"


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CR-AS priced read (read-only)")
    ap.add_argument("--cr-id", default="CR-AS-analysis")
    ap.add_argument("--out", default=None, help="markdown report path (default scripts/logs/cr_as_analysis_<ts>.md)")
    ap.add_argument("--universe-end", default=UNIVERSE_END.isoformat())
    args = ap.parse_args(argv)
    universe_end = date.fromisoformat(args.universe_end)
    if universe_end > UNIVERSE_END:
        sys.exit("ERROR: --universe-end past the holdout split (G5)")
    out_path = Path(args.out) if args.out else Path(repo_root) / "scripts" / "logs" / f"cr_as_analysis_{datetime.now():%Y%m%d_%H%M%S}.md"
    rng = np.random.default_rng(SEED)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    rows = conn.execute(UNIVERSE_SQL, (TICKER, CANONICAL_FEATURE_VERSION, universe_end)).fetchall()
    universe = {r[0]: {"regime": r[1], "es_open": float(r[2]), "im": float(r[3])} for r in rows}
    assert all(td <= UNIVERSE_END for td in universe), "G5: date past the split in the universe"
    pside = {r[0]: {"es_close": float(r[1]), "close_move_over_im": float(r[2]), "breach_side": r[3]} for r in conn.execute(
        """SELECT trade_date, session_close_t0, close_move_over_im, breach_side FROM bt_daily_outcomes
           WHERE ticker=%s AND feature_version=%s AND active AND trade_date <= %s AND close_move_over_im IS NOT NULL""",
        (TICKER, CANONICAL_FEATURE_VERSION, universe_end)).fetchall()}

    with backfill_run(conn, args.cr_id) as run_id:
        print(f"CR-AS analysis  run_id={run_id}  universe={len(universe)}  fee={CONDOR_ROUND_TRIP_FEE_PTS:.3f} pts (${CONDOR_ROUND_TRIP_FEE_USD:.2f})")
        results = []          # dict per date × box with a valid entry
        status = Counter()    # date-level
        box_status = Counter()
        sanity_violations = []
        per_date = {}
        for td in sorted(universe):
            u = universe[td]
            so = conn.execute(SPX_OPEN_SQL, (TICKER, td.isoformat(), datetime.combine(td, OPEN_SNAPSHOT_FLOOR))).fetchone()
            if not so or so[1].time() > OPEN_SNAPSHOT_CEIL:
                status["no_spx_open_snapshot"] += 1
                continue
            spx_open = float(so[0])
            basis_open = u["es_open"] - spx_open
            if not (BASIS_MIN <= basis_open <= BASIS_MAX):
                status["bad_spx_open"] += 1
                continue
            try:
                boxes = build_boxes(td, spx_open, u["im"])
            except ValueError:
                status["degenerate"] += 1
                continue
            opras = sorted({format_opra(OPRA_ROOT, td, t, k) for b in boxes for k, t in b.contracts()})
            bars = conn.execute(BARS_SQL, (opras, datetime.combine(td, WINDOW_START), datetime.combine(td, WINDOW_END))).fetchall()
            if not bars:
                status["not_captured"] += 1
                continue
            by_min = defaultdict(list)
            for snap, strike, typ, bid, ask in bars:
                by_min[snap].append((float(strike), typ, bid, ask))
            minutes = sorted(by_min.items())
            cl = conn.execute(SPX_CLOSE_SQL, (TICKER, td.isoformat(), datetime.combine(td, CLOSE_CEIL))).fetchone()
            if not cl:
                status["no_spx_close"] += 1
                continue
            spx_close = float(cl[0])
            es_close = pside[td]["es_close"]
            basis_close = es_close - spx_close
            drift = abs(basis_close - basis_open)
            if drift > DRIFT_MAX or not (CLOSE_BASIS_MIN <= basis_close <= CLOSE_BASIS_MAX):
                sanity_violations.append((td.isoformat(), round(basis_open, 1), round(basis_close, 1), round(drift, 1)))
                status["settlement_sanity_violation"] += 1
                continue
            status["priced_date"] += 1
            per_date[td] = {"regime": u["regime"], "spx_open": spx_open, "spx_close": spx_close, "im": u["im"],
                            "basis_open": basis_open, "basis_close": basis_close, "close_snapshot": cl[1].isoformat()}
            for b in boxes:
                # a box is capturable iff each leg has ≥ 1 bar in the window
                have = {(float(s), t) for _, s, t, _, _ in bars}
                if any(c not in have for c in b.contracts()):
                    box_status[f"{b.label}:unlistable"] += 1
                    continue
                e = first_valid_entry(b, minutes, floor=ENTRY_FLOOR)
                if e is None:
                    box_status[f"{b.label}:no_valid_entry"] += 1
                    continue
                r = settle_condor(b, e["credit"], spx_close)
                box_status[f"{b.label}:priced"] += 1
                results.append({
                    "date": td.isoformat(), "regime": u["regime"], "box": b.label, "im": u["im"],
                    "strikes": b.strikes, "entry_minute": e["snapshot_pt"].strftime("%H:%M"),
                    "entry_offset_min": e["n_minutes_invalid"], "credit_pts": r.credit, "credit_im": r.credit_im,
                    "spx_close": spx_close, "put_loss_pts": r.put_loss, "call_loss_pts": r.call_loss,
                    "loss_im": r.loss_im, "put_loss_im": r.put_loss / u["im"], "call_loss_im": r.call_loss / u["im"],
                    "gross_pts": r.gross_pnl_pts, "net_pts": r.net_pnl_pts, "gross_im": r.gross_pnl_im, "net_im": r.net_pnl_im,
                    "max_loss_pts": r.max_loss_pts, "net_per_max_loss": r.net_pnl_pts / r.max_loss_pts if r.max_loss_pts > 0 else np.nan,
                    "breach": r.breach, "win": r.net_pnl_pts > 0,
                })
        assert all(date.fromisoformat(r["date"]) <= UNIVERSE_END for r in results), "G5"

        # ── tables ──────────────────────────────────────────────────────────
        L = []
        P = L.append
        P(f"### Analysis run `{run_id}` — {datetime.now():%Y-%m-%d %H:%M} PT\n")
        P(f"Universe {len(universe)} dates ≤ {universe_end}. Date status: {dict(status)}. Box status: {dict(box_status)}.")
        P(f"Fee: {CONDOR_ROUND_TRIP_FEE_PTS:.3f} pts per condor round trip (${CONDOR_ROUND_TRIP_FEE_USD:.2f}). Bootstrap B={B}, seed {SEED}. "
          f"Per-regime reads only at n ≥ {MIN_N_REGIME}.\n")
        n_boxes_attempted = sum(v for k, v in box_status.items() if not k.endswith("unlistable"))
        n_nve = sum(v for k, v in box_status.items() if k.endswith("no_valid_entry"))
        P(f"G3 `no_valid_entry` share of attempted boxes: **{n_nve} / {n_boxes_attempted} = {n_nve / n_boxes_attempted if n_boxes_attempted else 0:.1%}** (expected < 15 %).  ")
        P(f"G4 settlement sanity violations: **{len(sanity_violations)}** {sanity_violations[:10]}  ")
        P(f"G5 max date in results: **{max((r['date'] for r in results), default='—')}**\n")

        def sub(box=None, regime=None):
            return [r for r in results if (box is None or r["box"] == box) and (regime is None or r["regime"] == regime)]

        P("#### Per regime × box (net P&L in IM units at settlement; mean [95 % bootstrap CI])\n")
        P("| box | regime | n | credit (IM) | realized loss (IM) | gross P&L (IM) | net P&L (IM) | net / max loss | win % | breach below / above | put-wing loss (IM) | call-wing loss (IM) | entry offset p50/p95 (min) |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for box in BOXES:
            for regime in ("pooled",) + REGIMES:
                s = sub(box, None if regime == "pooled" else regime)
                if not s:
                    continue
                n = len(s)
                offs = [r["entry_offset_min"] for r in s]
                P(f"| {box} | {regime} | {n} | {np.mean([r['credit_im'] for r in s]):.3f} | {np.mean([r['loss_im'] for r in s]):.3f} | "
                  f"{np.mean([r['gross_im'] for r in s]):+.3f} | {fmt(boot_mean([r['net_im'] for r in s], rng))} | "
                  f"{np.nanmean([r['net_per_max_loss'] for r in s]):+.3f} | {100 * np.mean([r['win'] for r in s]):.0f} | "
                  f"{sum(r['breach'] == 'below' for r in s)} / {sum(r['breach'] == 'above' for r in s)} | "
                  f"{np.mean([r['put_loss_im'] for r in s]):.3f} | {np.mean([r['call_loss_im'] for r in s]):.3f} | "
                  f"{np.percentile(offs, 50):.0f} / {np.percentile(offs, 95):.0f} |")
        P("")

        # P-side table on the SAME dates (priced ±0.5 sample)
        P("#### P-side (ES, `bt_daily_outcomes`) vs priced (SPX quotes) on the same dates — ±0.5 IM box\n")
        P("| regime | n | P-side breach cost beyond ±0.5 IM (uncapped, IM) | beyond ±1.0 IM | wing-capped ±0.5 (IM) | priced realized loss ±0.5 (IM) | priced credit ±0.5 (IM) |")
        P("|---|---|---|---|---|---|---|")
        for regime in ("pooled",) + REGIMES:
            s = sub("pm0.5", None if regime == "pooled" else regime)
            if not s:
                continue
            dts = [date.fromisoformat(r["date"]) for r in s]
            cm = np.array([abs(pside[d]["close_move_over_im"]) for d in dts])
            ims = np.array([universe[d]["im"] for d in dts])
            bc5 = np.maximum(0, cm - 0.5)
            bc10 = np.maximum(0, cm - 1.0)
            capped = np.minimum(bc5, WING_WIDTH / ims)
            P(f"| {regime} | {len(s)} | {bc5.mean():.3f} | {bc10.mean():.3f} | {capped.mean():.3f} | "
              f"{np.mean([r['loss_im'] for r in s]):.3f} | {np.mean([r['credit_im'] for r in s]):.3f} |")
        P("")

        # ── hypotheses ─────────────────────────────────────────────────────
        P("#### Hypotheses (pre-registered; read rule: supported only if the CI excludes the threshold in the stated direction)\n")
        H = []
        p5 = sub("pm0.5")
        p1 = sub("pm1")
        # H1
        t = boot_mean([r["net_im"] for r in p5], rng)
        H.append(("H1", f"±0.5 IM condor, pooled mean **net** P&L (IM) > 0 (n={len(p5)})", fmt(t), verdict(t, ">")))
        tg = boot_mean([r["gross_im"] for r in p5], rng)
        H.append(("H1 (gross)", f"same, gross (n={len(p5)})", fmt(tg), verdict(tg, ">")))
        # H2
        pin, mab = sub("pm0.5", "magnetic-pin"), sub("pm0.5", "magnet-above")
        t = boot_diff([r["net_im"] for r in pin], [r["net_im"] for r in mab], rng)
        note = "" if min(len(pin), len(mab)) >= MIN_N_REGIME else f" — n < {MIN_N_REGIME}, no claim"
        H.append(("H2", f"magnetic-pin − magnet-above mean net P&L (±0.5) ≥ 0.03 IM (n={len(pin)} / {len(mab)})", fmt(t),
                  (verdict(t, ">", 0.03) if not note else "n/a") + note))
        # H3
        t = boot_ratio([r["credit_im"] for r in p5], [r["loss_im"] for r in p5], rng)
        H.append(("H3", f"credit / realized breach cost ≥ 1.5, pooled ±0.5 (n={len(p5)})", fmt(t, 2), verdict(t, ">", 1.5)))
        # H4
        for regime in ("magnet-above", "untethered"):
            s = sub("pm0.5", regime)
            t = boot_diff([r["put_loss_im"] for r in s], [r["call_loss_im"] for r in s], rng)
            note = "" if len(s) >= MIN_N_REGIME else f" — n < {MIN_N_REGIME}, no claim"
            H.append((f"H4 {regime}", f"put-wing loss − call-wing loss > 0 (±0.5, n={len(s)})", fmt(t),
                      (verdict(t, ">") if not note else "n/a") + note))
        # H5: paired by date, net per unit max loss
        d5 = {r["date"]: r["net_per_max_loss"] for r in p5}
        d1 = {r["date"]: r["net_per_max_loss"] for r in p1}
        common = sorted(set(d5) & set(d1))
        diff = np.array([d1[d] - d5[d] for d in common])
        t = boot_mean(diff, rng)
        H.append(("H5", f"±1.0 minus ±0.5 mean net P&L per unit max loss < 0 (paired dates n={len(common)})", fmt(t), verdict(t, "<")))
        t5, t1 = boot_mean([d5[d] for d in common], rng), boot_mean([d1[d] for d in common], rng)
        H.append(("H5 (levels)", f"net / max loss: ±0.5 {fmt(t5)} · ±1.0 {fmt(t1)}", "", ""))
        # H6: Ryan's rule
        rule = [r for r in p5 if r["regime"] in ("magnetic-pin", "bounded")]
        allr = p5
        t = boot_diff([r["net_im"] for r in rule], [r["net_im"] for r in allr], rng)
        H.append(("H6", f"rule (pin + bounded) − all days, mean net P&L per trade (±0.5) > 0 (n={len(rule)} / {len(allr)})", fmt(t), verdict(t, ">")))
        tot_rule, tot_all = sum(r["net_im"] for r in rule), sum(r["net_im"] for r in allr)
        H.append(("H6 opportunity cost", f"total net P&L (IM): rule {tot_rule:+.2f} over {len(rule)} trades vs all-days {tot_all:+.2f} over {len(allr)} "
                  f"({len(rule) / len(allr) if allr else 0:.0%} of days); rule captures {tot_rule / tot_all if tot_all else float('nan'):+.0%} of the all-days total", "", ""))
        for regime in ("amplification", "magnet-above"):
            s = sub("pm0.5", regime)
            t = boot_diff([r["net_im"] for r in s], [r["net_im"] for r in allr], rng)
            note = "" if len(s) >= MIN_N_REGIME else f" — n < {MIN_N_REGIME}, no claim"
            H.append((f"H6 secondary {regime}", f"{regime} − pooled mean net P&L (±0.5) < 0, i.e. individually worse (n={len(s)})", fmt(t),
                      (verdict(t, "<") if not note else "n/a") + note))
        P("| # | statement | estimate [95 % CI] | read |")
        P("|---|---|---|---|")
        for h in H:
            P(f"| {h[0]} | {h[1]} | {h[2]} | {h[3]} |")
        P("")

        # sample rows
        P("#### 10 sample rows (±0.5 box)\n")
        P("| date | regime | strikes | entry | credit (pts) | SPX close | put loss | call loss | net (pts) | net (IM) | breach |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        step = max(1, len(p5) // 10)
        for r in p5[::step][:10]:
            P(f"| {r['date']} | {r['regime']} | {'/'.join(f'{k:g}' for k in r['strikes'])} | {r['entry_minute']} | {r['credit_pts']:.2f} | {r['spx_close']:.2f} | "
              f"{r['put_loss_pts']:.2f} | {r['call_loss_pts']:.2f} | {r['net_pts']:+.2f} | {r['net_im']:+.3f} | {r['breach'] or '—'} |")
        P("")
        report = "\n".join(L)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        (out_path.with_suffix(".json")).write_text(json.dumps({"results": results, "status": dict(status), "box_status": dict(box_status),
                                                                "sanity_violations": sanity_violations, "hypotheses": H}, default=str, indent=1))
        print(report)

        sample = Counter(r["regime"] for r in p5)
        smoke = {"universe": len(universe), "date_status": dict(status), "box_status": dict(box_status),
                 "n_results": len(results), "sample_pm05_by_regime": dict(sample),
                 "g3_no_valid_entry_share": round(n_nve / n_boxes_attempted, 4) if n_boxes_attempted else None,
                 "g4_sanity_violations": len(sanity_violations), "g5_max_date": max((r["date"] for r in results), default=None),
                 "hypotheses": [{"id": h[0], "estimate": h[2], "read": h[3]} for h in H], "report": str(out_path), "fee_pts": CONDOR_ROUND_TRIP_FEE_PTS}
        update_run_smoke(conn, run_id, smoke, f"priced {len(p5)} ±0.5 and {len(p1)} ±1.0 boxes over {status['priced_date']} dates; "
                                              f"G3 {smoke['g3_no_valid_entry_share']}, G4 {len(sanity_violations)} violations; report {out_path.name}")
        print(f"\nreport: {out_path}\nrun_id: {run_id}")


if __name__ == "__main__":
    main()
