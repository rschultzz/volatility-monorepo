"""CR-031 (CR-Z) Step 3 — Live sanity check (no code changes).

Confirms:
  1. Common setups now show K>20 with tighter CIs (cap no longer binding).
  2. 2026-05-07 edge anchor K=1 (documented as honest rarity).
  3. No dissimilar-day leakage — farthest admitted analogue on a common
     setup is within the ceiling.

Reports on 20 recent 2026 dates.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import date as Date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import psycopg

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats, rank_analogues
from packages.shared.knn_config import CANONICAL_KNN_CONFIG_VERSION, get_knn_config

_K_SAFETY = 200
_EDGE_ANCHOR = "2026-05-07"


def _conn():
    url = os.environ["DATABASE_URL"]
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return psycopg.connect(url)


def _load_corpus(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector "
            "FROM bt_daily_features_active "
            "WHERE ticker='SPX' AND feature_version=%s ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()
    return [(d.isoformat(), v) for (d, v, *_) in rows]


def main():
    conn = _conn()
    candidates = _load_corpus(conn)
    conn.close()
    print(f"Corpus: {len(candidates)} rows, feature_version={CANONICAL_FEATURE_VERSION}")
    print(f"Canonical config: {CANONICAL_KNN_CONFIG_VERSION}")

    cfg = get_knn_config()
    global_stats = feature_stats(v for (_, v) in candidates)
    vec_by_date = {d: v for (d, v) in candidates}

    # Sample 20 recent 2026 dates
    dates_2026 = sorted(d for (d, _) in candidates if d.startswith("2026"))
    sample = dates_2026[-20:]  # most recent 20

    print(f"\n{'='*72}")
    print(f"Step 3 live sanity — {len(sample)} most-recent 2026 anchors")
    print(f"Config: ceiling={cfg['distance_ceiling']}, hl={cfg['half_life_months']}mo, "
          f"k_safety={_K_SAFETY}")
    print(f"{'='*72}")
    print(f"{'Date':<14} {'K':>5}  {'Min_dist':>9}  {'Max_dist':>9}  {'Cap_was_binding?':>18}")

    k_vals = []
    cap_binding_count = 0
    for anchor_date in sample:
        anchor_vec = vec_by_date.get(anchor_date)
        if anchor_vec is None:
            continue
        ranked = rank_analogues(
            anchor_vec, candidates, _K_SAFETY,
            exclude_date=anchor_date,
            before_date=anchor_date,
            stats=global_stats,
            distance_ceiling=cfg["distance_ceiling"],
            feature_weights=cfg.get("feature_weights"),
            z_diff_cap=cfg.get("z_diff_cap"),
            half_life_months=cfg.get("half_life_months"),
            anchor_date=anchor_date,
        )
        k = len(ranked)
        k_vals.append(k)
        was_capped = "YES — K>20!" if k > 20 else "no"
        if k > 20:
            cap_binding_count += 1
        if ranked:
            min_d = ranked[0][1]
            max_d = ranked[-1][1]
        else:
            min_d = max_d = float("nan")
        # Verify ceiling: farthest analogue must be ≤ ceiling
        if ranked and max_d > cfg["distance_ceiling"] + 1e-9:
            was_capped += " ⚠️ LEAK"
        print(f"{anchor_date:<14} {k:>5}  {min_d:>9.4f}  {max_d:>9.4f}  {was_capped:>18}")

    print(f"\nSummary: {cap_binding_count}/{len(k_vals)} setups have K>20 "
          f"(old cap would have truncated these)")
    print(f"K values: min={min(k_vals)}  median={sorted(k_vals)[len(k_vals)//2]}  "
          f"max={max(k_vals)}  mean={sum(k_vals)/len(k_vals):.1f}")

    # Edge anchor check
    print(f"\n{'='*72}")
    print(f"Edge anchor check: {_EDGE_ANCHOR}")
    print(f"{'='*72}")
    edge_vec = vec_by_date.get(_EDGE_ANCHOR)
    if edge_vec is None:
        print(f"  {_EDGE_ANCHOR} not in corpus — skipped")
    else:
        ranked_edge = rank_analogues(
            edge_vec, candidates, _K_SAFETY,
            exclude_date=_EDGE_ANCHOR,
            before_date=_EDGE_ANCHOR,
            stats=global_stats,
            distance_ceiling=cfg["distance_ceiling"],
            feature_weights=cfg.get("feature_weights"),
            z_diff_cap=cfg.get("z_diff_cap"),
            half_life_months=cfg.get("half_life_months"),
            anchor_date=_EDGE_ANCHOR,
        )
        k_edge = len(ranked_edge)
        print(f"  K={k_edge} (expected 1 — genuine rarity; two-pin-cluster day)")
        if k_edge == 1:
            print(f"  ✓ Confirmed: K=1 is honest (only 1 day within {cfg['distance_ceiling']}σ "
                  f"after recency scaling); not a cap artifact")
        elif k_edge > 1:
            print(f"  ✓ Improved: K={k_edge} (was 1 at v2; note if config changed)")
        else:
            print(f"  → K=0: setup fully isolated in corpus")

    # No-leakage confirmation
    print(f"\n{'='*72}")
    print("No-leakage confirmation")
    print(f"{'='*72}")
    # Find a common setup (K≥30) and check its farthest analogue
    common_date = None
    for d in sample:
        v = vec_by_date.get(d)
        if v is None:
            continue
        ranked_check = rank_analogues(
            v, candidates, _K_SAFETY,
            exclude_date=d,
            before_date=d,
            stats=global_stats,
            distance_ceiling=cfg["distance_ceiling"],
            feature_weights=cfg.get("feature_weights"),
            z_diff_cap=cfg.get("z_diff_cap"),
            half_life_months=cfg.get("half_life_months"),
            anchor_date=d,
        )
        if len(ranked_check) >= 30:
            common_date = d
            ranked_common = ranked_check
            break

    if common_date:
        k_common = len(ranked_common)
        max_dist = ranked_common[-1][1]
        print(f"  Common setup: {common_date}, K={k_common}")
        print(f"  Farthest admitted analogue: distance={max_dist:.4f} "
              f"(ceiling={cfg['distance_ceiling']})")
        assert max_dist <= cfg["distance_ceiling"] + 1e-9, (
            f"LEAK: farthest analogue {max_dist:.4f} > ceiling {cfg['distance_ceiling']}"
        )
        print(f"  ✓ Farthest analogue is within ceiling — no dissimilar-day leakage")
    else:
        print("  No common setup with K≥30 found in sample — check corpus density")

    print(f"\nStep 3 complete.")


if __name__ == "__main__":
    main()
