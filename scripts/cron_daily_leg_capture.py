#!/usr/bin/env python3
"""CR-AU decision 1 — daily leg capture for the live out-of-sample stream. NO READ.

Render cron `daily-leg-capture`, `50 13 * * 1-5` (06:50 PDT; Step 0 amendment
A1: after the 13:35 implied-move fill and the 13:40 outcomes insert, and after
the 06:30–06:45 PT capture window has closed — fetching a window before it
closes would record the missing minutes as fetched-and-empty).

For the most recent canonical `bt_daily_features` row (trade_date ≤ today, or
--date):
  * every day, regardless of regime — the symmetric ±0.5 IM 0DTE condor box on
    the SPX 06:33 PT spot (CR-AS convention: shorts at round5(spot ± 0.5·IM),
    10-wide wings, same-day expiry), 4 legs;
  * if the regime is magnet-above — the debit pair the harness would trade:
    short call at the drift target, long call ~10 below, pair-snapped to the
    strikes listed for the 15-business-day expiry at the prior close
    (`snap_spread_to_listed`, CR-AR), 2 legs;
and fetches the 06:30–06:45 PT window for each leg into orats_options_minute
via fetch_option_bars (one contract per call). Legs whose window is already
covered in orats_options_fetched_windows are skipped (dedupe). Listing truth
is the ORATS response: a leg that 404s or returns no bars marks its structure
`unlistable` in the run record.

Logs legs / bars / 404s / unlistable only. Never computes a price or P&L.

Env: BACKFILL_DATABASE_URL (role dash_backfill_writer) and ORATS_API_KEY.
DATABASE_URL is overridden in-process to the backfill URL (options_cache reads it).

Usage:
    python scripts/cron_daily_leg_capture.py                 # today's row
    python scripts/cron_daily_leg_capture.py --date 2026-09-04
    python scripts/cron_daily_leg_capture.py --date 2026-09-04 --dry-run   # leg list only, no fetch, no run row

Exit: 0 on success or nothing to do (no feature row / no open snapshot);
      1 on a fetch exception other than a 404, or when the window is not closed yet.
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pure helpers only at import time (tests import this module without any env).
from packages.shared.backtest.condor_0dte import CondorBox, build_box
from packages.shared.options_cache.models import FetchedWindow, TimeRange
from packages.shared.options_cache.opra import format_opra
from packages.shared.options_cache.strikes import StructureNotListed, snap_spread_to_listed
from packages.shared.options_cache.windows import find_gaps

TICKER = "SPX"
OPRA_ROOT = "SPX"
CR_ID = "DAILY-CAPTURE"
WINDOW_START = time(6, 30)
WINDOW_END = time(6, 45)
OPEN_SNAPSHOT_FLOOR = time(6, 33)
OPEN_SNAPSHOT_CEIL = time(6, 40)
CONDOR_HALF_WIDTH_IM = 0.5
DEBIT_WIDTH_NOMINAL = 10.0
MIN_LAG_MIN = 5                      # A1: the window must have closed this long ago before a fetch
BASIS_MIN, BASIS_MAX = -40.0, 100.0  # ES − SPX at the open (CR-AS Step 0), checked when ES open is known
_PT = ZoneInfo("America/Los_Angeles")


# ── plan (pure) ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Leg:
    opra: str
    strike: float
    option_type: str          # 'C' | 'P'
    expiry: date
    structure: str            # 'debit' | 'condor'
    role: str                 # 'short' | 'long'


@dataclass
class Plan:
    trade_date: date
    regime: Optional[str]
    implied_move: Optional[float]
    im_source: Optional[str]
    spx_open: Optional[float]
    spx_open_snapshot: Optional[datetime]
    es_open: Optional[float]
    basis_open: Optional[float]
    skip: Optional[str] = None                    # whole-day skip reason
    debit: Optional[dict] = None                  # {'short','long','width_actual','expiry','prior_close'} or {'unlistable': reason}
    condor: Optional[dict] = None                 # {'strikes': (lp, sp, sc, lc), 'expiry'} or {'unlistable': reason}
    legs: list[Leg] = field(default_factory=list)

    @property
    def window(self) -> tuple[datetime, datetime]:
        return capture_window(self.trade_date)


def capture_window(td: date) -> tuple[datetime, datetime]:
    """The entry window every leg is fetched for: 06:30–06:45 PT, naive PT."""
    return datetime.combine(td, WINDOW_START), datetime.combine(td, WINDOW_END)


def window_is_closed(now_pt: datetime, td: date, min_lag_min: int = MIN_LAG_MIN) -> bool:
    """A1: only fetch once the window end is at least `min_lag_min` in the past."""
    _, w1 = capture_window(td)
    return now_pt >= w1 + timedelta(minutes=min_lag_min)


def plan_condor(td: date, spx_open: float, implied_move: float) -> dict:
    """±0.5 IM box on the SPX 06:33 spot, same-day expiry (CR-AS decision 2 as amended)."""
    try:
        box: CondorBox = build_box(td, float(spx_open), float(implied_move), CONDOR_HALF_WIDTH_IM)
    except ValueError as exc:
        return {"unlistable": f"degenerate: {exc}", "expiry": td}
    return {"strikes": box.strikes, "expiry": td, "box": box}


def plan_debit(regime: Optional[str], target: Optional[float], expiry: date, chain, spot: Optional[float]) -> Optional[dict]:
    """The harness's debit pair for a magnet-above day; None on any other regime."""
    if regime != "magnet-above":
        return None
    if target is None:
        return {"unlistable": "no drift_target for the date", "expiry": expiry}
    try:
        s = snap_spread_to_listed(float(target), DEBIT_WIDTH_NOMINAL, "debit", None, chain, toward=spot)
    except StructureNotListed as exc:
        return {"unlistable": str(exc), "expiry": expiry}
    return {"short": float(s.anchor), "long": float(s.other), "width_actual": float(s.width_actual),
            "narrower_than_nominal": s.narrower_than_nominal, "expiry": expiry}


def legs_for(plan_debit_: Optional[dict], plan_condor_: Optional[dict]) -> list[Leg]:
    legs: list[Leg] = []
    if plan_debit_ and "unlistable" not in plan_debit_:
        ex = plan_debit_["expiry"]
        legs.append(Leg(format_opra(OPRA_ROOT, ex, "C", plan_debit_["short"]), plan_debit_["short"], "C", ex, "debit", "short"))
        legs.append(Leg(format_opra(OPRA_ROOT, ex, "C", plan_debit_["long"]), plan_debit_["long"], "C", ex, "debit", "long"))
    if plan_condor_ and "unlistable" not in plan_condor_:
        ex = plan_condor_["expiry"]
        lp, sp, sc, lc = plan_condor_["strikes"]
        for k, t, role in ((lp, "P", "long"), (sp, "P", "short"), (sc, "C", "short"), (lc, "C", "long")):
            legs.append(Leg(format_opra(OPRA_ROOT, ex, t, k), float(k), t, ex, "condor", role))
    return legs


def build_plan(td: date, *, regime: Optional[str], implied_move: Optional[float], im_source: Optional[str],
               spx_open: Optional[float], spx_open_snapshot: Optional[datetime], es_open: Optional[float],
               target: Optional[float], spot: Optional[float], debit_expiry: date, chain) -> Plan:
    """Pure: everything the day needs, from already-loaded inputs."""
    p = Plan(td, regime, implied_move, im_source, spx_open, spx_open_snapshot, es_open, None)
    if spx_open is None or spx_open_snapshot is None or spx_open_snapshot.time() > OPEN_SNAPSHOT_CEIL:
        p.skip = "no_spx_open_snapshot"
        return p
    if es_open is not None:
        p.basis_open = round(float(es_open) - float(spx_open), 2)
        if not (BASIS_MIN <= p.basis_open <= BASIS_MAX):
            p.skip = "bad_spx_open"
            return p
    if not implied_move or implied_move <= 0:
        p.skip = "no_implied_move"
        return p
    p.condor = plan_condor(td, spx_open, implied_move)
    p.debit = plan_debit(regime, target, debit_expiry, chain, spot)
    p.legs = legs_for(p.debit, p.condor)
    return p


def dedupe_legs(legs: list[Leg], window: tuple[datetime, datetime],
                existing: dict[str, list[FetchedWindow]]) -> tuple[list[Leg], list[Leg]]:
    """(to_fetch, covered): a leg is covered when orats_options_fetched_windows
    already spans the whole window (find_gaps → no gap)."""
    req = TimeRange(start_pt=window[0], end_pt=window[1])
    to_fetch, covered = [], []
    for leg in legs:
        if find_gaps(req, existing.get(leg.opra, [])):
            to_fetch.append(leg)
        else:
            covered.append(leg)
    return to_fetch, covered


def format_plan(p: Plan, covered: Optional[set[str]] = None) -> str:
    w0, w1 = p.window
    lines = [f"{p.trade_date}  regime={p.regime}  IM={p.implied_move if p.implied_move is None else round(p.implied_move, 2)} ({p.im_source})"
             f"  SPX06:33={p.spx_open}@{p.spx_open_snapshot.time() if p.spx_open_snapshot else None}"
             f"  ES_open={p.es_open}  basis={p.basis_open}  window={w0:%H:%M}–{w1:%H:%M} PT"]
    if p.skip:
        lines.append(f"  SKIP {p.skip} — no legs")
        return "\n".join(lines)
    if p.debit is None:
        lines.append("  debit: none (regime is not magnet-above)")
    elif "unlistable" in p.debit:
        lines.append(f"  debit: UNLISTABLE ({p.debit['unlistable']}) expiry {p.debit['expiry']}")
    else:
        lines.append(f"  debit: long {p.debit['long']:g} / short {p.debit['short']:g} C  expiry {p.debit['expiry']}"
                     f"  width {p.debit['width_actual']:g}{' (narrower than nominal)' if p.debit.get('narrower_than_nominal') else ''}")
    if p.condor and "unlistable" in p.condor:
        lines.append(f"  condor: UNLISTABLE ({p.condor['unlistable']})")
    elif p.condor:
        lp, sp, sc, lc = p.condor["strikes"]
        lines.append(f"  condor ±{CONDOR_HALF_WIDTH_IM:g} IM: {lp:g}P / {sp:g}P / {sc:g}C / {lc:g}C  expiry {p.condor['expiry']} (0DTE)")
    for leg in p.legs:
        tag = "  [cached]" if covered and leg.opra in covered else ""
        lines.append(f"    {leg.structure:6s} {leg.role:5s} {leg.opra}{tag}")
    return "\n".join(lines)


# ── env / DB (only from main) ────────────────────────────────────────────────

def _bootstrap_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    bak = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
    if not bak:
        sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
    os.environ["DATABASE_URL"] = bak          # options_cache repository reads DATABASE_URL
    if not os.environ.get("ORATS_API_KEY", "").strip():
        sys.exit("ERROR: ORATS_API_KEY not set (needed by fetch_option_bars).")


_FEATURE_ROW_SQL = """
SELECT trade_date, regime_at_classification, (feature_vector->>'implied_move_1d')::float
FROM bt_daily_features
WHERE ticker = %s AND feature_version = %s AND active AND trade_date = %s
LIMIT 1
"""
_LATEST_FEATURE_DATE_SQL = """
SELECT max(trade_date) FROM bt_daily_features
WHERE ticker = %s AND feature_version = %s AND active AND trade_date <= %s
"""
_SPX_OPEN_SQL = """
SELECT spot_price, snapshot_pt FROM orats_monies_minute
WHERE ticker = %s AND trade_date = %s AND snapshot_pt >= %s AND spot_price IS NOT NULL
ORDER BY snapshot_pt ASC, dte ASC LIMIT 1
"""
_ES_OPEN_SQL = """
SELECT session_open_t0 FROM bt_daily_outcomes
WHERE ticker = %s AND feature_version = %s AND active AND trade_date = %s LIMIT 1
"""
_TABLE_SPOT_SQL = "SELECT table_spot FROM orats_gex_landscape WHERE ticker = %s AND trade_date = %s"


def load_inputs(conn, td: date) -> dict:
    """Everything build_plan needs, from the DB (backfill role, read-only here)."""
    from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
    from packages.shared.day_features import _OPEN_STRADDLE_SQL
    from packages.shared.gex_landscape import compute_implied_move
    from packages.shared.options_cache.strikes import listed_strikes
    from scripts.cr_ah_step4_analysis import DTE_TARGET, nth_business_day
    from scripts.cr_am_holdout_leg_capture import _payload_target

    row = conn.execute(_FEATURE_ROW_SQL, (TICKER, CANONICAL_FEATURE_VERSION, td)).fetchone()
    regime = row[1] if row else None
    im = float(row[2]) if row and row[2] is not None else None
    im_source = "feature_vector.implied_move_1d" if im else None
    so = conn.execute(_SPX_OPEN_SQL, (TICKER, td.isoformat(), datetime.combine(td, OPEN_SNAPSHOT_FLOOR))).fetchone()   # trade_date is text in orats_monies_minute
    spx_open = float(so[0]) if so else None
    spx_snap = so[1] if so else None
    spot_row = conn.execute(_TABLE_SPOT_SQL, (TICKER, td)).fetchone()
    spot = float(spot_row[0]) if spot_row and spot_row[0] is not None else None
    if im is None and spot is not None:
        # A7 fallback: the CR-AB pin (first 06:33+ snapshot, smallest dte > 0)
        iv_row = conn.execute(_OPEN_STRADDLE_SQL, (td.isoformat(), TICKER, datetime.combine(td, OPEN_SNAPSHOT_FLOOR))).fetchone()
        if iv_row and iv_row[0] is not None:
            im_calc = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
            if im_calc:
                im, im_source = float(im_calc), "open_straddle_0633 (feature row IM was NULL)"
    es_row = conn.execute(_ES_OPEN_SQL, (TICKER, CANONICAL_FEATURE_VERSION, td)).fetchone()
    es_open = float(es_row[0]) if es_row and es_row[0] is not None else None
    expiry = nth_business_day(td, DTE_TARGET)
    target = _payload_target(conn, td) if regime == "magnet-above" else None
    _, chain = listed_strikes(conn, expiry, td, TICKER) if regime == "magnet-above" else (None, [])
    return {"feature_row": bool(row), "regime": regime, "implied_move": im, "im_source": im_source,
            "spx_open": spx_open, "spx_open_snapshot": spx_snap, "es_open": es_open,
            "target": target, "spot": spot, "debit_expiry": expiry, "chain": chain}


def resolve_date(conn, explicit: Optional[str], today: date) -> Optional[date]:
    from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
    if explicit:
        return date.fromisoformat(explicit)
    row = conn.execute(_LATEST_FEATURE_DATE_SQL, (TICKER, CANONICAL_FEATURE_VERSION, today)).fetchone()
    return row[0] if row and row[0] else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CR-AU daily leg capture (no read)")
    ap.add_argument("--date", default=None, help="trade_date YYYY-MM-DD (default: most recent canonical feature row ≤ today)")
    ap.add_argument("--dry-run", action="store_true", help="print the leg list; no fetch, no run row")
    ap.add_argument("--cr-id", default=CR_ID)
    ap.add_argument("--min-lag-min", type=int, default=MIN_LAG_MIN)
    args = ap.parse_args(argv)

    _bootstrap_env()
    from packages.shared.backfill_safety import assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke
    from packages.shared.options_cache import repository as repo

    now_pt = datetime.now(_PT).replace(tzinfo=None)
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    td = resolve_date(conn, args.date, now_pt.date())
    print(f"CR-AU daily leg capture  cr_id={args.cr_id}  dry_run={args.dry_run}  now_pt={now_pt:%Y-%m-%d %H:%M}  trade_date={td}")
    if td is None:
        print("no canonical feature row ≤ today — nothing to do")
        return 0
    inp = load_inputs(conn, td)
    if not inp["feature_row"]:
        print(f"no canonical feature row for {td} — nothing to do")
        return 0
    plan = build_plan(td, **{k: v for k, v in inp.items() if k != "feature_row"})

    existing = {leg.opra: repo.get_windows_for_contract(leg.opra) for leg in plan.legs}
    to_fetch, covered = dedupe_legs(plan.legs, plan.window, existing)
    print(format_plan(plan, covered={l.opra for l in covered}))
    print(f"legs planned={len(plan.legs)}  to_fetch={len(to_fetch)}  already_covered={len(covered)}")

    if args.dry_run:
        print("dry-run: no fetches, no run row.")
        return 0
    if plan.skip:
        print(f"skip {plan.skip}: nothing to fetch, no run row.")
        return 0
    if not window_is_closed(now_pt, td, args.min_lag_min):
        print(f"ERROR: window {plan.window[1]:%H:%M} PT + {args.min_lag_min} min lag has not passed (now {now_pt:%H:%M} PT) — "
              f"fetching now would record unfilled minutes as fetched. Nothing recorded.")
        return 1

    from packages.shared.options_cache.fetcher import fetch_option_bars
    from packages.shared.options_cache.http_client import OratsPermanentError

    w0, w1 = plan.window
    counters = {"legs_planned": len(plan.legs), "legs_covered": len(covered), "legs_fetched": 0, "legs_404": 0,
                "legs_empty": 0, "legs_exception": 0, "bars_written": 0, "cache_hits": 0, "gaps_filled": 0}
    leg_status: dict[str, dict] = {l.opra: {"status": "covered", "structure": l.structure} for l in covered}
    detail_404: list[str] = []
    detail_exc: list[str] = []
    with backfill_run(conn, args.cr_id) as run_id:
        print(f"\nRun ID: {run_id}")
        t0 = _time.perf_counter()
        for leg in to_fetch:
            try:
                r = fetch_option_bars([leg.opra], w0, w1, source="historical_backfill", record_empty_windows=True)
            except OratsPermanentError as exc:
                counters["legs_404"] += 1
                detail_404.append(f"{leg.structure} {leg.opra}: {exc}")
                leg_status[leg.opra] = {"status": "404", "structure": leg.structure, "err": str(exc)[:160]}
                print(f"  {leg.structure:6s} {leg.opra}: ORATS 4xx — {exc}")
                continue
            except Exception as exc:                      # noqa: BLE001 — counted, reported, exit 1
                counters["legs_exception"] += 1
                detail_exc.append(f"{leg.structure} {leg.opra}: {type(exc).__name__}: {exc}")
                leg_status[leg.opra] = {"status": "exception", "structure": leg.structure, "err": f"{type(exc).__name__}: {exc}"[:200]}
                print(f"  {leg.structure:6s} {leg.opra}: EXCEPTION {type(exc).__name__}: {exc}")
                continue
            counters["legs_fetched"] += 1
            counters["bars_written"] += r.bars_written
            counters["cache_hits"] += r.cache_hits
            counters["gaps_filled"] += r.gaps_filled
            n = conn.execute("SELECT count(*) FROM orats_options_minute WHERE opra_symbol = %s AND snapshot_pt BETWEEN %s AND %s",
                             (leg.opra, w0, w1)).fetchone()[0]
            if n == 0:
                counters["legs_empty"] += 1
            leg_status[leg.opra] = {"status": "ok" if n else "empty", "structure": leg.structure, "bars": int(n), "written": r.bars_written}
            print(f"  {leg.structure:6s} {leg.opra}: bars_in_window={n} written={r.bars_written} cache_hits={r.cache_hits}")

        def _structure_state(name: str, plan_part: Optional[dict]) -> Optional[str]:
            if plan_part is None:
                return None
            if "unlistable" in plan_part:
                return f"unlistable: {plan_part['unlistable']}"
            bad = [o for o, s in leg_status.items() if s["structure"] == name and s["status"] in ("404", "empty", "exception")]
            return f"unlistable: {bad}" if bad else "captured"

        states = {"debit": _structure_state("debit", plan.debit), "condor": _structure_state("condor", plan.condor)}
        smoke = dict(counters)
        smoke.update({"trade_date": td.isoformat(), "regime": plan.regime, "implied_move": plan.implied_move,
                      "im_source": plan.im_source, "spx_open": plan.spx_open, "basis_open": plan.basis_open,
                      "window": [w0.isoformat(), w1.isoformat()], "debit": {k: (v.isoformat() if isinstance(v, date) else v) for k, v in (plan.debit or {}).items()},
                      "condor": {"strikes": list(plan.condor["strikes"]), "expiry": td.isoformat()} if plan.condor and "strikes" in plan.condor else plan.condor,
                      "structures": states, "legs": leg_status, "orats_404_detail": detail_404, "exception_detail": detail_exc,
                      "elapsed_s": round(_time.perf_counter() - t0, 1)})
        summary = (f"{td} {plan.regime}: legs planned={counters['legs_planned']} covered={counters['legs_covered']} "
                   f"fetched={counters['legs_fetched']} 404={counters['legs_404']} empty={counters['legs_empty']} "
                   f"exceptions={counters['legs_exception']} bars_written={counters['bars_written']}; "
                   f"debit={states['debit']} condor={states['condor']}; no P&L computed")
        update_run_smoke(conn, run_id, smoke, summary)
        print(f"\nSUMMARY: {summary}")
        for d in detail_404:
            print("  404:", d)
        for d in detail_exc:
            print("  exception:", d)
    conn.close()
    return 1 if counters["legs_exception"] else 0


if __name__ == "__main__":
    sys.exit(main())
