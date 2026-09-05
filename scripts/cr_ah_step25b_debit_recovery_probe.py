"""CR-AH Step 2.5b — Debit-recoverable check on excluded dates. READ-ONLY.

Among the 44 excluded (non-A-bucket) dates from the 150-date stratified selection,
identifies dates where ONLY the credit long leg (target+10) 404'd — meaning the
date might still be usable for the DEBIT structure (target + target-10).

Classification per excluded date:
  no_long  (target present, target+10 absent) — probe ORATS for target-10.
    → debit_recoverable: target-10 returns rows     → usable for debit only
    → both_unusable:     target-10 also 404s        → unusable for both
  no_both  (both target and target+10 absent) — target absent → unusable for both.
    (no probe — target is the shared leg; without it neither structure can price)

READ-ONLY: uses get_csv() directly. NO writes to orats_options_minute.
Run as: PYTHONUNBUFFERED=1 python3 -u scripts/cr_ah_step25b_debit_recovery_probe.py
"""

from __future__ import annotations

import os
import sys
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ── ENV ───────────────────────────────────────────────────────────────────────

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

bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = bak_url

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────

from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.backtest.models import distance_band
from packages.shared.day_features import _LANDSCAPE_ROW_SQL, _OPEN_STRADDLE_SQL, _materialize_payload
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.options_cache.http_client import get_csv, OratsPermanentError
from packages.shared.options_cache.opra import format_opra, opra_to_orats_ticker

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER      = "SPX"
OPRA_ROOT   = "SPX"
DTE_TARGET  = 15
SPLIT_DATE  = date(2025, 8, 12)
_PT = ZoneInfo("America/Los_Angeles")
_ET = ZoneInfo("America/New_York")

_NYSE_HOLIDAYS: frozenset[date] = frozenset({
    date(2023,1,2),  date(2023,1,16), date(2023,2,20), date(2023,4,7),
    date(2023,5,29), date(2023,6,19), date(2023,7,4),  date(2023,9,4),
    date(2023,11,23),date(2023,12,25),
    date(2024,1,1),  date(2024,1,15), date(2024,2,19), date(2024,3,29),
    date(2024,5,27), date(2024,6,19), date(2024,7,4),  date(2024,9,2),
    date(2024,11,28),date(2024,12,25),
    date(2025,1,1),  date(2025,1,9),  date(2025,1,20), date(2025,2,17),
    date(2025,4,18), date(2025,5,26), date(2025,6,19), date(2025,7,4),
    date(2025,9,1),  date(2025,11,27),date(2025,12,25),
    date(2026,1,1),  date(2026,1,19), date(2026,2,16), date(2026,4,3),
    date(2026,5,25), date(2026,6,19), date(2026,7,3),  date(2026,9,7),
    date(2026,11,26),date(2026,12,25),
})

# ── HELPERS ───────────────────────────────────────────────────────────────────

def nth_biz(d: date, n: int) -> date:
    cur, cnt = d, 0
    while cnt < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in _NYSE_HOLIDAYS:
            cnt += 1
    return cur

def round5(p: float) -> int:
    return round(p / 5) * 5

def stride(lst: list, n: int = 50) -> list:
    if len(lst) <= n:
        return list(lst)
    step = len(lst) / n
    return [lst[int(i * step)] for i in range(n)]

def format_trade_date_param(d: date) -> str:
    """Full RTH window (06:30–13:00 PT) as ORATS tradeDate param (ET, YYYYMMDDHHMM)."""
    start_et = datetime(d.year, d.month, d.day, 6, 30, tzinfo=_PT).astimezone(_ET).strftime("%Y%m%d%H%M")
    end_et   = datetime(d.year, d.month, d.day, 13, 0, tzinfo=_PT).astimezone(_ET).strftime("%Y%m%d%H%M")
    return f"{start_et},{end_et}"

def probe_orats(trade_date: date, expiry_date: date, strike: float) -> tuple[str, int]:
    """Direct ORATS probe — READ-ONLY. Returns (status, row_count)."""
    opra   = format_opra(OPRA_ROOT, expiry_date, "C", strike)
    ticker = opra_to_orats_ticker(opra)
    param  = format_trade_date_param(trade_date)
    try:
        csv_text  = get_csv(
            "/datav2/hist/live/one-minute/strikes/option",
            {"ticker": ticker, "tradeDate": param},
        )
        data_rows = [r for r in csv_text.strip().splitlines() if r and not r.startswith("ticker")]
        return ("200", len(data_rows))
    except OratsPermanentError:
        return ("404", 0)
    except Exception as exc:
        return (f"ERR:{exc}", 0)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 65
    print(f"\n{sep}", flush=True)
    print("CR-AH Step 2.5b — Debit-recoverable check (READ-ONLY)", flush=True)
    print(sep, flush=True)

    conn = get_backfill_db_conn()

    # ── Rebuild the 150-date stratified selection (same logic as backfill script) ──
    rows = conn.execute(
        "SELECT trade_date FROM bt_daily_features "
        "WHERE ticker=%s AND feature_version=%s AND regime_at_classification='magnet-above' "
        "ORDER BY trade_date",
        (TICKER, CANONICAL_FEATURE_VERSION),
    ).fetchall()
    dates = [r[0] for r in rows]

    entries = []
    for td in dates:
        row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, td)).fetchone()
        if not row or row[0] is None or row[1] is None:
            continue
        landscape_rows, spot = row
        spot = float(spot)
        floor_ts = datetime.combine(td, time(6, 33, 0))
        iv_row = conn.execute(_OPEN_STRADDLE_SQL, (td.isoformat(), TICKER, floor_ts)).fetchone()
        if not iv_row or iv_row[0] is None:
            continue
        try:
            im = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
        except Exception:
            continue
        if not im:
            continue
        payload = _materialize_payload(landscape_rows, spot, im)
        dt = (payload.get("regime") or {}).get("drift_target")
        if dt is None:
            continue
        sigma = (float(dt) - spot) / im
        entries.append({
            "trade_date":   td,
            "drift_target": float(dt),
            "band":         distance_band(sigma),
            "partition":    "train" if td <= SPLIT_DATE else "holdout",
        })

    by_band: dict[str, list] = {"near": [], "mid": [], "far": []}
    for e in entries:
        by_band[e["band"]].append(e)
    selected = []
    for b in ("near", "mid", "far"):
        selected += stride(sorted(by_band[b], key=lambda x: x["trade_date"]))
    selected.sort(key=lambda x: x["trade_date"])
    print(f"Stratified selection: {len(selected)} dates", flush=True)

    # ── Classify each selected date by DB coverage ────────────────────────────
    excluded: list[dict] = []
    for e in selected:
        td = e["trade_date"]; dt = e["drift_target"]
        ss = float(round5(dt))
        exp = nth_biz(td, DTE_TARGET)
        start_pt = datetime(td.year, td.month, td.day, 6, 30)
        end_pt   = datetime(td.year, td.month, td.day, 13, 0)

        credit_long_opra = format_opra(OPRA_ROOT, exp, "C", ss + 10)
        short_opra       = format_opra(OPRA_ROOT, exp, "C", ss)

        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END) > 0,
                SUM(CASE WHEN opra_symbol=%s THEN 1 ELSE 0 END) > 0
            FROM orats_options_minute
            WHERE opra_symbol = ANY(%s) AND snapshot_pt >= %s AND snapshot_pt < %s
            """,
            (short_opra, credit_long_opra, [short_opra, credit_long_opra], start_pt, end_pt),
        ).fetchone()
        has_short, has_credit_long = (bool(r) for r in row)

        if not (has_short and has_credit_long):
            excluded.append({
                **e,
                "expiry_date":       exp,
                "short_strike":      ss,
                "credit_long_strike": ss + 10,
                "debit_long_strike":  ss - 10,
                "short_opra":        short_opra,
                "credit_long_opra":  credit_long_opra,
                "has_short":         has_short,
                "has_credit_long":   has_credit_long,
                # DB sub-class
                "db_class": "no_long" if (has_short and not has_credit_long) else "no_both",
            })

    no_long  = [e for e in excluded if e["db_class"] == "no_long"]
    no_both  = [e for e in excluded if e["db_class"] == "no_both"]
    print(f"\nExcluded (non-A-bucket): {len(excluded)} total  "
          f"(no_long={len(no_long)}, no_both={len(no_both)})", flush=True)

    # ── no_both: target absent → both structures unusable, skip probe ─────────
    print(f"\n{'─'*65}", flush=True)
    print(f"no_both ({len(no_both)} dates) — target absent from DB; unusable for both structures.", flush=True)
    for e in sorted(no_both, key=lambda x: (x["band"], x["partition"], x["trade_date"])):
        print(f"  {e['band']:<4} {e['partition']:<8} {e['trade_date']}  "
              f"target={e['short_strike']:.0f}", flush=True)

    # ── no_long: probe ORATS for target-10 ───────────────────────────────────
    print(f"\n{'─'*65}", flush=True)
    print(f"no_long ({len(no_long)} dates) — target present, target+10 absent.", flush=True)
    print("Probing ORATS for target-10 (debit long leg)...\n", flush=True)

    probe_results: list[dict] = []
    for i, e in enumerate(sorted(no_long, key=lambda x: x["trade_date"])):
        td   = e["trade_date"]
        exp  = e["expiry_date"]
        dl_s = e["debit_long_strike"]

        print(f"  [{i+1}/{len(no_long)}] {e['band']}/{e['partition']}  {td}  "
              f"target={e['short_strike']:.0f}  expiry={exp}  "
              f"debit_long_strike={dl_s:.0f}", flush=True)

        t0 = _time.perf_counter()
        status, rows_returned = probe_orats(td, exp, dl_s)
        elapsed = _time.perf_counter() - t0

        debit_recoverable = (status == "200" and rows_returned > 0)
        label = "DEBIT-RECOVERABLE" if debit_recoverable else "BOTH-UNUSABLE"
        print(f"    ORATS target-10: {status}  rows={rows_returned}  "
              f"elapsed={elapsed:.1f}s  → {label}", flush=True)

        probe_results.append({
            **e,
            "debit_long_status": status,
            "debit_long_rows":   rows_returned,
            "debit_recoverable": debit_recoverable,
        })
        _time.sleep(0.5)  # gentle rate limit

    # ── Summary ───────────────────────────────────────────────────────────────
    debit_recoverable = [r for r in probe_results if r["debit_recoverable"]]
    debit_both_unusable = [r for r in probe_results if not r["debit_recoverable"]]

    print(f"\n{sep}", flush=True)
    print("SUMMARY", flush=True)
    print(sep, flush=True)
    print(f"\nExcluded dates breakdown:", flush=True)
    print(f"  no_both  (target absent):          {len(no_both):>3}  → both-unusable (no probe)", flush=True)
    print(f"  no_long  probed for target-10:     {len(probe_results):>3}", flush=True)
    print(f"    → DEBIT-RECOVERABLE:             {len(debit_recoverable):>3}", flush=True)
    print(f"    → both-unusable (target-10 also 404'd): {len(debit_both_unusable):>3}", flush=True)

    print(f"\n{'─'*65}", flush=True)
    print("DEBIT-RECOVERABLE by band × split:", flush=True)
    print(f"  {'band':<5} {'split':<8} {'n':>3}  dates", flush=True)
    totals: dict[str, int] = {}
    for band in ("near", "mid", "far"):
        for part in ("train", "holdout"):
            sub = [r for r in debit_recoverable if r["band"] == band and r["partition"] == part]
            if sub:
                dates_str = ", ".join(str(r["trade_date"]) for r in sub)
                print(f"  {band:<5} {part:<8} {len(sub):>3}  {dates_str}", flush=True)
            totals[f"{band}/{part}"] = len(sub)

    # Current far/holdout A-bucket count
    a_far_ho = 5  # from Step 2.5 coverage table
    rec_far_ho = totals.get("far/holdout", 0)
    print(f"\n{'─'*65}", flush=True)
    print(f"FAR/HOLDOUT impact:  current n={a_far_ho}  recoverable={rec_far_ho}  "
          f"potential n={a_far_ho + rec_far_ho}", flush=True)
    print(f"FAR/TRAIN   impact:  current n={totals.get('far/train', 0) + 25}  "   # 25 = current A far/train
          f"recoverable={totals.get('far/train', 0)}  "
          f"potential n={25 + totals.get('far/train', 0)}", flush=True)

    print(f"\n{'─'*65}", flush=True)
    if not debit_recoverable:
        print("VERDICT: 0 debit-recoverable dates. n=106 locks for both structures.", flush=True)
    else:
        print(f"VERDICT: {len(debit_recoverable)} debit-recoverable date(s). "
              "Recommend small target-10 backfill for these before Step 4.", flush=True)
        for r in debit_recoverable:
            print(f"  {r['trade_date']}  {r['band']}/{r['partition']}  "
                  f"target-10={r['debit_long_strike']:.0f}  "
                  f"ORATS rows={r['debit_long_rows']}", flush=True)


if __name__ == "__main__":
    main()
