#!/usr/bin/env python3
"""CR-A Step 2 — determinism check for landscape feature extraction.

Re-runs the feature extraction pipeline for a given trade_date from the
stored orats_gex_landscape + orats_monies_minute inputs, then compares
field-by-field against the stored bt_daily_features row
(feature_version='v0.5.0').

Verifies that compute_and_upsert_daily_features is deterministic: running it
again on the same stored inputs should produce bit-identical or float-
drift-only differences.  Any MISMATCH outside tolerance is a stop condition
for CR-A — the backfill would diverge from the original v0.5.0 corpus.

Usage:
    python scripts/cr_a_determinism_check.py --date 2026-05-01
    python scripts/cr_a_determinism_check.py --date 2026-05-01 --ticker SPX

Exit codes:  0 = all within tolerance;  1 = one or more MISMATCHes detected.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.day_features import (
    FEATURE_NAMES,
    VOL_SURFACE_PLACEHOLDERS,
    _IMPLIED_MOVE_SQL,
    _LANDSCAPE_ROW_SQL,
    _materialize_payload,
    compute_feature_config_hash,
    extract_features,
)
from packages.shared.gex_landscape import compute_implied_move

# ── Tolerance ──────────────────────────────────────────────────────────────────

# Integer-valued features: stored as float in JSONB but semantically integers.
# Exact equality required; any difference is a stop condition.
_INTEGER_FEATURES = frozenset({
    "is_pin_day", "is_magnet_day", "is_bounded_day", "is_untethered_day",
    "is_amplification_day", "magnet_direction_signed",
    "cluster_1_quality_ordinal", "cluster_2_quality_ordinal",
    "cluster_3_quality_ordinal",
    "n_clusters_total", "n_pin", "n_target", "n_feature",
    "n_clusters_above_spot", "n_clusters_below_spot", "n_neg_zones",
})

FLOAT_DRIFT_TOL = 1e-9   # ≤ this: DRIFT (float accumulation noise, not a concern)
FLOAT_STOP_TOL  = 1e-5   # > this: MISMATCH — stop condition

# ── DB helpers ─────────────────────────────────────────────────────────────────

def _load_stored(conn, ticker: str, trade_date: dt.date) -> tuple[dict, str | None]:
    """Return (feature_vector_dict, regime_at_classification) from bt_daily_features."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT feature_vector, regime_at_classification
            FROM bt_daily_features
            WHERE ticker = %s AND trade_date = %s AND feature_version = 'v0.5.0'
            """,
            (ticker, trade_date),
        )
        row = cur.fetchone()
    if not row:
        raise ValueError(
            f"No v0.5.0 row in bt_daily_features for ({ticker!r}, {trade_date})"
        )
    return row[0], row[1]


def _recompute(
    conn, ticker: str, trade_date: dt.date
) -> tuple[dict, str | None, float, float]:
    """Re-run _materialize_payload + extract_features from stored inputs.

    Reads orats_gex_landscape and orats_monies_minute — the same sources used
    by compute_and_upsert_daily_features — but skips the UPSERT.

    Returns (features, regime_at_classification, spot, implied_move).
    """
    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_ROW_SQL, (ticker, trade_date))
        row = cur.fetchone()
    if not row:
        raise ValueError(
            f"No orats_gex_landscape row for ({ticker!r}, {trade_date})"
        )
    landscape_rows, table_spot = row
    spot = float(table_spot)

    with conn.cursor() as cur:
        cur.execute(_IMPLIED_MOVE_SQL, (trade_date.isoformat(), ticker))
        iv_row = cur.fetchone()
    if iv_row and iv_row[0] is not None:
        try:
            implied_move = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
        except (TypeError, ValueError):
            implied_move = 0.0
    else:
        implied_move = 0.0

    payload  = _materialize_payload(landscape_rows, spot, implied_move)
    features = extract_features(payload, spot, implied_move)
    regime   = (payload.get("regime") or {}).get("regime") or None

    return features, regime, spot, implied_move


# ── Field-by-field comparison ─────────────────────────────────────────────────

def _compare(
    stored_fv: dict,
    fresh_fv:  dict,
    stored_regime: str | None,
    fresh_regime:  str | None,
) -> list[dict]:
    """Return one result dict per checked field, covering:
       - regime_at_classification (categorical column)
       - every key in FEATURE_NAMES (inside feature_vector JSONB)
    """
    results: list[dict] = []

    # ── regime_at_classification — categorical, exact required ─────────────────
    regime_ok = (stored_regime == fresh_regime)
    results.append({
        "field":  "regime_at_classification",
        "stored": stored_regime,
        "fresh":  fresh_regime,
        "status": "EXACT" if regime_ok else "MISMATCH",
        "delta":  None,
    })

    # ── feature_vector JSONB keys (all 35 entries in FEATURE_NAMES) ───────────
    for name in FEATURE_NAMES:
        s = stored_fv.get(name)
        f = fresh_fv.get(name)

        # NULL vol-surface placeholders — both must be None
        if name in VOL_SURFACE_PLACEHOLDERS:
            status = "NULL_MATCH" if (s is None and f is None) else "NULL_MISMATCH"
            results.append({
                "field":  name,
                "stored": s,
                "fresh":  f,
                "status": status,
                "delta":  None,
            })
            continue

        # Integer-valued: exact equality required
        if name in _INTEGER_FEATURES:
            ok = (s == f)
            results.append({
                "field":  name,
                "stored": s,
                "fresh":  f,
                "status": "EXACT" if ok else "MISMATCH",
                "delta":  None,
            })
            continue

        # Continuous float — tolerance-graded comparison
        try:
            sf, ff = float(s), float(f)
        except (TypeError, ValueError):
            results.append({
                "field":  name,
                "stored": s,
                "fresh":  f,
                "status": "MISMATCH",
                "delta":  None,
            })
            continue

        delta = abs(sf - ff)
        if delta == 0.0:
            status = "EXACT"
        elif delta <= FLOAT_DRIFT_TOL:
            status = "DRIFT"
        elif delta <= FLOAT_STOP_TOL:
            status = "WARN"
        else:
            status = "MISMATCH"

        results.append({
            "field":  name,
            "stored": sf,
            "fresh":  ff,
            "status": status,
            "delta":  delta,
        })

    return results


# ── Reporting ──────────────────────────────────────────────────────────────────

_STATUS_ORDER = {"MISMATCH": 0, "NULL_MISMATCH": 1, "WARN": 2,
                 "DRIFT": 3, "EXACT": 4, "NULL_MATCH": 5}


def _print_results(results: list[dict]) -> dict[str, list[str]]:
    """Print per-field table sorted by severity, return dict of status → field list."""
    by_status: dict[str, list[str]] = {}
    for r in results:
        by_status.setdefault(r["status"], []).append(r["field"])

    sorted_results = sorted(results, key=lambda r: _STATUS_ORDER.get(r["status"], 9))

    col_w = max(len(r["field"]) for r in results) + 2
    print(f"\n{'FIELD':<{col_w}} {'STATUS':<14} {'DELTA':<14} STORED → FRESH")
    print("─" * 95)

    for r in sorted_results:
        delta_s = f"{r['delta']:.3e}" if r["delta"] is not None else ""

        # For non-exact results, show stored and fresh values
        if r["status"] in ("MISMATCH", "NULL_MISMATCH", "WARN"):
            vals = f"{r['stored']!r} → {r['fresh']!r}"
        else:
            vals = ""

        print(f"{r['field']:<{col_w}} {r['status']:<14} {delta_s:<14} {vals}")

    return by_status


def _print_summary(by_status: dict[str, list[str]]) -> None:
    print("\nSummary:")
    total = sum(len(v) for v in by_status.values())
    for status in ["MISMATCH", "NULL_MISMATCH", "WARN", "DRIFT", "EXACT", "NULL_MATCH"]:
        fields = by_status.get(status, [])
        if fields:
            label = ", ".join(fields) if len(fields) <= 4 else f"{len(fields)} fields"
            print(f"  {status:<14} {len(fields):>3}  ({label})")
    print(f"  {'TOTAL':<14} {total:>3}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date",   required=True, help="Trade date YYYY-MM-DD")
    ap.add_argument("--ticker", default="SPX",  help="Ticker (default: SPX)")
    args = ap.parse_args()

    trade_date = dt.date.fromisoformat(args.date)
    ticker     = args.ticker

    # Load .env
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    conn = get_backfill_db_conn()

    print(f"=== CR-A Determinism Check ===")
    print(f"Ticker: {ticker}  Date: {trade_date}")
    print(f"Tolerance: DRIFT ≤ {FLOAT_DRIFT_TOL:.0e}  |  "
          f"WARN ≤ {FLOAT_STOP_TOL:.0e}  |  MISMATCH > {FLOAT_STOP_TOL:.0e}")

    try:
        stored_fv, stored_regime = _load_stored(conn, ticker, trade_date)
        fresh_fv, fresh_regime, spot, implied_move = _recompute(conn, ticker, trade_date)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        conn.close()
        sys.exit(1)

    config_hash = compute_feature_config_hash("v0.5.0")
    print(f"\nContext:  spot={spot:.2f}  implied_move={implied_move:.4f}  "
          f"config_hash={config_hash}")

    results   = _compare(stored_fv, fresh_fv, stored_regime, fresh_regime)
    by_status = _print_results(results)
    _print_summary(by_status)

    conn.close()

    # ── Exit decision ──────────────────────────────────────────────────────────
    mismatches = by_status.get("MISMATCH", []) + by_status.get("NULL_MISMATCH", [])
    warns      = by_status.get("WARN", [])

    if mismatches:
        print(f"\n*** STOP CONDITION: {len(mismatches)} MISMATCH(ES) — "
              f"do not proceed to Step 3. ***")
        sys.exit(1)

    if warns:
        print(f"\nWARN: {len(warns)} field(s) above DRIFT threshold but within STOP threshold.")
        print("Review the WARN entries above before proceeding.")
    else:
        print("\nAll fields within tolerance — determinism confirmed.")

    sys.exit(0)


if __name__ == "__main__":
    main()
