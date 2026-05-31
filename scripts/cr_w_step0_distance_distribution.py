"""CR-029 Step 0 — empirical analogue distance distribution.

Queries the full corpus, runs rank_analogues from every stored feature row
(excluding itself), and collects the distribution of top-K distances.

Outputs:
  - Overall percentile table for min/mean/max distance per anchor
  - Per-anchor stats for the nearest neighbor and the K=20th neighbor
  - Suggested ceiling candidates at p90 / p95 / p99 of kth-neighbor distance
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import psycopg

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats, rank_analogues


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


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = p / 100.0 * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def main():
    K = 20  # match current _K_MAX

    conn = _conn()
    print(f"Connected. Loading corpus (feature_version={CANONICAL_FEATURE_VERSION})…")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector FROM bt_daily_features_active"
            " WHERE ticker = 'SPX' AND feature_version = %s"
            " ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()

    print(f"Corpus: {len(rows)} rows")
    if not rows:
        print("No corpus rows — aborting.")
        return

    candidates = [(d.isoformat(), v) for (d, v) in rows]
    stats = feature_stats(v for (_, v) in candidates)

    # For each anchor day, compute the top-K distances
    nearest_distances: list[float] = []        # distance to rank-1 neighbour
    kth_distances: list[float] = []            # distance to rank-K neighbour (if K exist)
    all_distances: list[float] = []            # every returned distance (all K)

    print("Running rank_analogues for every anchor…")
    for i, (anchor_date, anchor_vec) in enumerate(candidates):
        ranked = rank_analogues(
            anchor_vec, candidates, K,
            exclude_date=anchor_date, stats=stats,
        )
        if not ranked:
            continue
        dists = [d for (_, d) in ranked if math.isfinite(d)]
        if not dists:
            continue
        nearest_distances.append(dists[0])
        kth_distances.append(dists[-1])   # the worst of the K
        all_distances.extend(dists)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(candidates)} anchors processed…")

    print(f"\n=== Distance distribution ({len(nearest_distances)} anchors) ===\n")

    for label, vals in [
        ("Nearest-neighbour (rank-1) distance", nearest_distances),
        (f"K={K}th-neighbour (worst of K) distance", kth_distances),
        ("All returned distances (every K)", all_distances),
    ]:
        sv = sorted(vals)
        print(f"{label}:")
        print(f"  n={len(sv)}, min={sv[0]:.3f}, p25={percentile(sv,25):.3f},"
              f" median={percentile(sv,50):.3f}, p75={percentile(sv,75):.3f},"
              f" p90={percentile(sv,90):.3f}, p95={percentile(sv,95):.3f},"
              f" p99={percentile(sv,99):.3f}, max={sv[-1]:.3f}")
        print()

    sv_kth = sorted(kth_distances)
    print("=== Ceiling candidates (based on kth-neighbor distance) ===")
    for p in [50, 75, 90, 95, 99]:
        v = percentile(sv_kth, p)
        print(f"  p{p:02d} = {v:.3f}σ")

    print("\n=== Ceiling candidates (based on all returned distances) ===")
    sv_all = sorted(all_distances)
    for p in [75, 90, 95, 99]:
        v = percentile(sv_all, p)
        print(f"  p{p:02d} = {v:.3f}σ")

    # Show the distribution of anchors whose NEAREST neighbour is above a threshold
    for thresh in [2.0, 3.0, 4.0, 5.0]:
        count = sum(1 for d in nearest_distances if d > thresh)
        pct = 100.0 * count / len(nearest_distances) if nearest_distances else 0
        print(f"  Anchors with nearest-neighbour > {thresh}σ: {count}/{len(nearest_distances)} ({pct:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
