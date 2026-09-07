#!/usr/bin/env python3
"""CR-AS Step 1 — capture 0DTE SPX condor legs (06:30–06:45 PT) into orats_options_minute. NO READ.

For every date in the train universe (canonical dates ≤ --universe-end with
close_move_over_im IS NOT NULL and implied_move_1d > 0):
  1. open   = SPX spot_price at the 06:33 PT orats_monies_minute snapshot (the
              CR-037 implied-move pin snapshot; Step 0 amendment of decision 2 —
              the ES session_open_t0 carries a +26 pt median basis to SPX)
  2. boxes  = shorts at round5(open ± 0.5·IM) and round5(open ± 1.0·IM),
              10-wide wings (packages.shared.backtest.condor_0dte)
  3. legs   = the distinct contracts across both boxes (≤ 8), same-day expiry,
              root SPX (ORATS keys SPXW under SPX; on 3rd Fridays the option
              endpoint returns the PM contract, expiry_tod='pm' — Step 0)
  4. fetch  = fetch_option_bars, one contract per call, window 06:30–06:45 PT.
              Listing truth is the ORATS response (decision 3 as amended in
              Step 0): a leg that 404s or returns no bars makes its box
              `unlistable`. The prior-close chain (orats_oi_gamma) under-lists
              next-day expiries and is recorded as `chain_missing` only.
Order (decision 6): all `bounded`; then magnetic-pin / magnet-above /
amplification / untethered interleaved round-robin over stride-selected dates
until 40 each; then round-robin over the rest. Checkpoint per date
(scripts/logs/cr_as_capture_checkpoint.json), resumable, --max-hours budget
measured from the first start recorded in the checkpoint.

Logs dates / legs / bars / 404s only. Never computes credit, payoff or P&L.

Usage:
    PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_as_capture_0dte_condor_legs.py \
        [--max-hours 8] [--dry-run] [--dates 2023-06-13,...] [--cr-id CR-AS-capture] \
        [--checkpoint scripts/logs/cr_as_capture_checkpoint.json] [--skip-third-friday]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
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
os.environ["DATABASE_URL"] = _bak_url   # options_cache repo reads DATABASE_URL

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from packages.shared.backfill_safety import (
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)
from packages.shared.backtest.condor_0dte import (
    FIRST_REGIME, REGIME_ORDER, TARGET_PER_REGIME, build_boxes, dedupe_contracts,
    is_third_friday, sample_order, unlisted_legs,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.opra import format_opra
from packages.shared.options_cache.strikes import listed_strikes

TICKER = "SPX"
OPRA_ROOT = "SPX"
UNIVERSE_END = date(2026, 6, 5)
WINDOW_START = time(6, 30)
WINDOW_END = time(6, 45)
OPEN_SNAPSHOT_FLOOR = time(6, 33)
OPEN_SNAPSHOT_CEIL = time(6, 40)
BASIS_MIN, BASIS_MAX = -40.0, 100.0        # ES − SPX at the open (Step 0: median +26, p01 −11, p99 +84)
DEFAULT_CHECKPOINT = Path(repo_root) / "scripts" / "logs" / "cr_as_capture_checkpoint.json"

UNIVERSE_SQL = """
SELECT o.trade_date, o.regime_kind_at_classification, o.session_open_t0,
       (f.feature_vector->>'implied_move_1d')::float AS im
FROM bt_daily_outcomes o
JOIN bt_daily_features f
  ON f.ticker = o.ticker AND f.trade_date = o.trade_date
 AND f.feature_version = o.feature_version AND f.active
WHERE o.ticker = %s AND o.feature_version = %s AND o.active
  AND o.trade_date <= %s
  AND o.close_move_over_im IS NOT NULL
  AND (f.feature_vector->>'implied_move_1d')::float > 0
ORDER BY o.trade_date
"""
BARS_IN_WINDOW_SQL = """
SELECT count(*) FROM orats_options_minute
WHERE opra_symbol = %s AND snapshot_pt BETWEEN %s AND %s
"""
SPX_OPEN_SQL = """
SELECT spot_price, snapshot_pt FROM orats_monies_minute
WHERE ticker = %s AND trade_date = %s AND snapshot_pt >= %s AND spot_price IS NOT NULL
ORDER BY snapshot_pt ASC, dte ASC LIMIT 1
"""


# ── plan (pure, tested via condor_0dte helpers) ──────────────────────────────

def plan_date(td: date, regime: str, es_open: float, im: float, spx_open, spx_snap, listed: list[float],
              skip_third_friday: bool = False) -> dict:
    """Plan (boxes, contracts, skip reason) for one date. Pure."""
    plan = {"trade_date": td.isoformat(), "regime": regime, "es_open": es_open, "im": im,
            "spx_open": spx_open, "spx_open_snapshot": spx_snap.isoformat() if spx_snap else None,
            "basis_open": None, "skip": None, "boxes": {}, "contracts": []}
    if is_third_friday(td) and skip_third_friday:
        plan["skip"] = "third_friday_skipped"
        return plan
    if spx_open is None or spx_snap is None or spx_snap.time() > OPEN_SNAPSHOT_CEIL:
        plan["skip"] = "no_spx_open_snapshot"
        return plan
    basis = float(es_open) - float(spx_open)
    plan["basis_open"] = round(basis, 2)
    if not (BASIS_MIN <= basis <= BASIS_MAX):
        plan["skip"] = "bad_spx_open"
        return plan
    try:
        boxes = build_boxes(td, float(spx_open), float(im))
    except ValueError as exc:
        plan["skip"] = f"degenerate: {exc}"
        return plan
    for b in boxes:
        # informational only: the prior-close chain under-lists next-day expiries (Step 0)
        plan["boxes"][b.label] = {"strikes": list(b.strikes), "chain_missing": unlisted_legs(b, listed) or None,
                                  "unlistable": None, "captured": None}
    plan["contracts"] = [[k, t] for k, t in dedupe_contracts(boxes)]
    return plan


# ── checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"started_at": None, "order": None, "done": {}, "budget_exhausted": False}


def save_checkpoint(path: Path, ck: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ck, indent=1, default=str))
    tmp.replace(path)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CR-AS 0DTE condor leg capture (no read)")
    ap.add_argument("--max-hours", type=float, default=8.0, help="wall-clock budget from the first start in the checkpoint")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and sample order; no fetch, no run row, no checkpoint")
    ap.add_argument("--dates", default=None, help="explicit comma-separated ISO dates (overrides the sample order; still universe-checked)")
    ap.add_argument("--cr-id", default="CR-AS-capture")
    ap.add_argument("--universe-end", default=UNIVERSE_END.isoformat())
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--skip-third-friday", action="store_true",
                    help="skip monthly (3rd-Friday) dates (Step 0 showed ORATS serves the PM SPXW there; default: include)")
    ap.add_argument("--max-dates", type=int, default=None, help="stop after this many dates this invocation (timing runs)")
    args = ap.parse_args(argv)

    universe_end = date.fromisoformat(args.universe_end)
    if universe_end > UNIVERSE_END:
        sys.exit(f"ERROR: --universe-end {universe_end} is past the holdout split {UNIVERSE_END} (G5).")
    ck_path = Path(args.checkpoint)
    today = date.today()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    rows = conn.execute(UNIVERSE_SQL, (TICKER, CANONICAL_FEATURE_VERSION, universe_end)).fetchall()
    universe = {r[0]: {"regime": r[1], "es_open": float(r[2]), "im": float(r[3])} for r in rows}
    by_regime: dict[str, list[date]] = defaultdict(list)
    for td, u in universe.items():
        by_regime[u["regime"]].append(td)
    print(f"CR-AS capture  cr_id={args.cr_id}  dry_run={args.dry_run}  universe={len(universe)} dates ≤ {universe_end}  "
          f"by regime={dict(Counter({r: len(v) for r, v in by_regime.items()}))}  today={today}")
    if any(td > UNIVERSE_END for td in universe):
        sys.exit("ERROR: universe contains a date past the holdout split (G5).")

    if args.dates:
        order = [date.fromisoformat(x.strip()) for x in args.dates.split(",")]
        bad = [d for d in order if d not in universe]
        if bad:
            sys.exit(f"ERROR: --dates not in the train universe: {bad}")
    else:
        order = sample_order(by_regime)
    assert len(set(order)) == len(order)

    ck = load_checkpoint(ck_path) if not args.dry_run else {"started_at": None, "order": None, "done": {}, "budget_exhausted": False}
    if not args.dry_run and not args.dates:
        if ck["order"] is None:
            ck["order"] = [d.isoformat() for d in order]
        else:
            order = [date.fromisoformat(x) for x in ck["order"]]      # deterministic across restarts
    if ck["started_at"] is None:
        ck["started_at"] = datetime.now().isoformat(timespec="seconds")
    started = datetime.fromisoformat(ck["started_at"])
    deadline = started + timedelta(hours=args.max_hours)
    print(f"budget: started {started}  deadline {deadline}  ({args.max_hours} h)  already done={len(ck['done'])}")

    todo = [d for d in order if d.isoformat() not in ck["done"]]
    if args.max_dates:
        todo = todo[: args.max_dates]
    print(f"sample order: {len(order)} dates; todo this invocation: {len(todo)}"
          f"  first 12: {[d.isoformat() for d in order[:12]]}")

    if args.dry_run:
        # plan a handful to show geometry; no fetch
        for td in todo[:10]:
            u = universe[td]
            so = conn.execute(SPX_OPEN_SQL, (TICKER, td.isoformat(), datetime.combine(td, OPEN_SNAPSHOT_FLOOR))).fetchone()
            _, listed = listed_strikes(conn, td, td, TICKER)
            p = plan_date(td, u["regime"], u["es_open"], u["im"], so[0] if so else None, so[1] if so else None,
                          listed, args.skip_third_friday)
            box_txt = ", ".join(
                f"{k}: {v['strikes']}" + (f" chain_missing={v['chain_missing']}" if v["chain_missing"] else "")
                for k, v in p["boxes"].items())
            skip_txt = f"  SKIP {p['skip']}" if p["skip"] else ""
            print(f"  {td} {u['regime']:13s} ES {u['es_open']:.2f} SPX {p['spx_open']} basis {p['basis_open']} IM {u['im']:.1f} "
                  f"boxes {{{box_txt}}} contracts={len(p['contracts'])}{skip_txt}")
        n_regime_order = Counter(universe[d]["regime"] for d in order[:20 + 4 * TARGET_PER_REGIME])
        print(f"dry-run: first {20 + 4 * TARGET_PER_REGIME} of the order by regime: {dict(n_regime_order)}; no fetches, no run row.")
        return

    counters = Counter()
    detail_404: list[str] = []
    detail_exc: list[str] = []
    with backfill_run(conn, args.cr_id) as run_id:
        print(f"\nRun ID: {run_id}\n")
        t_run = _time.perf_counter()
        for i, td in enumerate(todo, 1):
            if datetime.now() >= deadline:
                ck["budget_exhausted"] = True
                print(f"BUDGET EXHAUSTED at {datetime.now():%H:%M:%S} — stopping cleanly before {td}")
                break
            u = universe[td]
            t0 = _time.perf_counter()
            so = conn.execute(SPX_OPEN_SQL, (TICKER, td.isoformat(), datetime.combine(td, OPEN_SNAPSHOT_FLOOR))).fetchone()
            _, listed = listed_strikes(conn, td, td, TICKER)
            p = plan_date(td, u["regime"], u["es_open"], u["im"], so[0] if so else None, so[1] if so else None,
                          listed, args.skip_third_friday)
            rec = {"regime": u["regime"], "skip": p["skip"], "boxes": p["boxes"], "spx_open": p["spx_open"],
                   "basis_open": p["basis_open"], "legs": {}, "elapsed_s": None, "run_id": run_id}
            counters["dates_seen"] += 1
            if p["skip"]:
                counters[f"skip:{p['skip'].split(':')[0]}"] += 1
                print(f"  {td} {u['regime']:13s} SKIP {p['skip']}")
            else:
                w0, w1 = datetime.combine(td, WINDOW_START), datetime.combine(td, WINDOW_END)
                for k, t in p["contracts"]:
                    opra = format_opra(OPRA_ROOT, td, t, k)
                    counters["legs_planned"] += 1
                    try:
                        r = fetch_option_bars([opra], w0, w1, source="historical_backfill", record_empty_windows=True)
                    except OratsPermanentError as exc:
                        counters["legs_404"] += 1
                        detail_404.append(f"{td} {opra}: {exc}")
                        rec["legs"][opra] = {"status": "404", "err": str(exc)[:120]}
                        continue
                    except Exception as exc:                            # noqa: BLE001 — logged, counted, continue
                        counters["legs_exception"] += 1
                        detail_exc.append(f"{td} {opra}: {type(exc).__name__}: {exc}")
                        rec["legs"][opra] = {"status": "exception", "err": f"{type(exc).__name__}: {exc}"[:200]}
                        continue
                    counters["legs_fetched"] += 1
                    counters["bars_written"] += r.bars_written
                    counters["cache_hits"] += r.cache_hits
                    counters["gaps_filled"] += r.gaps_filled
                    n_in_window = conn.execute(BARS_IN_WINDOW_SQL, (opra, w0, w1)).fetchone()[0]
                    if n_in_window == 0:
                        counters["legs_empty"] += 1
                        rec["legs"][opra] = {"status": "empty", "bars": 0, "cache_hit": r.cache_hits}
                    else:
                        rec["legs"][opra] = {"status": "ok", "bars": n_in_window, "written": r.bars_written, "cache_hit": r.cache_hits}
                ok_legs = {o for o, v in rec["legs"].items() if v["status"] == "ok"}
                for label, bx in p["boxes"].items():
                    leg_opras = [format_opra(OPRA_ROOT, td, t, k) for k, t in zip(bx["strikes"], ("P", "P", "C", "C"))]
                    bad = [o for o in leg_opras if o not in ok_legs]
                    bx["unlistable"] = bad or None
                    bx["captured"] = not bad
                    if not bad:
                        counters[f"boxes_captured:{label}"] += 1
                    else:
                        counters[f"boxes_unlistable:{label}"] += 1
                if any(bx.get("captured") for bx in p["boxes"].values()):
                    counters["dates_captured"] += 1
                    counters[f"sample:{u['regime']}"] += 1
                rec["elapsed_s"] = round(_time.perf_counter() - t0, 2)
                print(f"  {td} {u['regime']:13s} SPX {p['spx_open']:.2f} basis {p['basis_open']:+.1f} IM {u['im']:.1f} "
                      f"legs={len(p['contracts'])} ok={len(ok_legs)} bars={sum(v.get('bars', 0) for v in rec['legs'].values())} "
                      f"boxes={'/'.join('ok' if bx['captured'] else 'UNLISTABLE' for bx in p['boxes'].values())} "
                      f"{rec['elapsed_s']}s")
            ck["done"][td.isoformat()] = rec
            save_checkpoint(ck_path, ck)
            if i % 10 == 0:
                update_run_progress(conn, run_id, counters["bars_written"])

        elapsed = round(_time.perf_counter() - t_run, 1)
        # sample achieved across ALL checkpointed dates (this + earlier invocations)
        achieved = Counter()
        for d, rec in ck["done"].items():
            if any(bx.get("captured") for bx in rec["boxes"].values()):
                achieved[rec["regime"]] += 1
        skips_all = Counter(rec["skip"].split(":")[0] for rec in ck["done"].values() if rec["skip"])
        legs_planned = counters["legs_planned"]
        rate_404 = (counters["legs_404"] / legs_planned) if legs_planned else 0.0
        smoke = dict(counters)
        smoke.update({
            "elapsed_s": elapsed, "dates_done_total": len(ck["done"]), "budget_exhausted": ck["budget_exhausted"],
            "sample_achieved_total": dict(achieved), "skips_total": dict(skips_all),
            "rate_404": round(rate_404, 4), "orats_404_detail": detail_404[:50], "exception_detail": detail_exc[:50],
            "universe": len(universe), "universe_end": universe_end.isoformat(), "max_hours": args.max_hours,
            "checkpoint": str(ck_path),
        })
        summary = (f"captured {counters['dates_captured']} dates this run ({len(ck['done'])} checkpointed total); "
                   f"legs planned={legs_planned} fetched={counters['legs_fetched']} 404={counters['legs_404']} "
                   f"({rate_404:.1%}) exceptions={counters['legs_exception']}; bars_written={counters['bars_written']} "
                   f"cache_hits={counters['cache_hits']}; sample achieved={dict(achieved)}; "
                   f"budget_exhausted={ck['budget_exhausted']}; {elapsed}s; no P&L computed")
        update_run_smoke(conn, run_id, smoke, summary)
        print(f"\nSUMMARY: {summary}")
        for d in detail_404[:20]:
            print("  404:", d)
        for d in detail_exc[:20]:
            print("  exception:", d)
        save_checkpoint(ck_path, ck)


if __name__ == "__main__":
    main()
