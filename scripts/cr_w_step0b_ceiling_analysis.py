"""CR-029 Step 0-B — ceiling sensitivity analysis.

For candidate ceilings, compute:
  - % of anchors that get 0 analogues within ceiling
  - median / p25 / p75 adaptive-K
  - distance distribution of the specific problem cases (signed_distance_sigma ≤ 1σ)
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
    K = 20

    conn = _conn()
    print(f"Loading corpus…")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector FROM bt_daily_features_active"
            " WHERE ticker = 'SPX' AND feature_version = %s"
            " ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()

    print(f"Corpus: {len(rows)} rows")
    candidates = [(d.isoformat(), v) for (d, v) in rows]
    stats = feature_stats(v for (_, v) in candidates)

    # Compute all-pairs top-K distances once
    all_ranked: dict[str, list[float]] = {}
    print("Computing all-pairs top-K…")
    for i, (anchor_date, anchor_vec) in enumerate(candidates):
        ranked = rank_analogues(
            anchor_vec, candidates, K,
            exclude_date=anchor_date, stats=stats,
        )
        all_ranked[anchor_date] = [d for (_, d) in ranked if math.isfinite(d)]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(candidates)}…")

    print("\n=== Ceiling sensitivity (adaptive-K) ===\n")
    for ceil in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        adaptive_ks = []
        zero_count = 0
        for dists in all_ranked.values():
            within = [d for d in dists if d <= ceil]
            adaptive_ks.append(len(within))
            if not within:
                zero_count += 1
        sv = sorted(adaptive_ks)
        zero_pct = 100.0 * zero_count / len(adaptive_ks) if adaptive_ks else 0
        print(
            f"  ceiling={ceil:.1f}σ | 0-analogue anchors: {zero_count:3d}/{len(adaptive_ks)}"
            f" ({zero_pct:4.1f}%) | adaptive-K: p25={percentile(sv,25):.0f}"
            f" median={percentile(sv,50):.0f} p75={percentile(sv,75):.0f}"
            f" p90={percentile(sv,90):.0f}"
        )

    # Focus on close-magnet setups (signed_distance_sigma ≤ 1σ absolute)
    print("\n=== Close-magnet anchors (|signed_distance_sigma| ≤ 1) ===\n")
    close_magnet_dates = []
    for d, v in candidates:
        sds = v.get("signed_distance_sigma")
        if sds is not None:
            try:
                if abs(float(sds)) <= 1.0:
                    close_magnet_dates.append(d)
            except (TypeError, ValueError):
                pass

    print(f"  Close-magnet anchors: {len(close_magnet_dates)}")
    if close_magnet_dates:
        nn_dists = []
        kth_dists = []
        for d in close_magnet_dates:
            dists = all_ranked.get(d, [])
            if dists:
                nn_dists.append(dists[0])
                kth_dists.append(dists[-1])

        sv_nn = sorted(nn_dists)
        sv_kth = sorted(kth_dists)
        print(f"  NN distance: min={sv_nn[0]:.3f} median={percentile(sv_nn,50):.3f}"
              f" p75={percentile(sv_nn,75):.3f} p90={percentile(sv_nn,90):.3f}"
              f" max={sv_nn[-1]:.3f}")
        print(f"  K=20th dist: min={sv_kth[0]:.3f} median={percentile(sv_kth,50):.3f}"
              f" p75={percentile(sv_kth,75):.3f} p90={percentile(sv_kth,90):.3f}"
              f" max={sv_kth[-1]:.3f}")

        print("\n  Close-magnet ceiling sensitivity:")
        for ceil in [2.0, 2.5, 3.0, 3.5, 4.0]:
            adaptive_ks = []
            zero_count = 0
            for d in close_magnet_dates:
                dists = all_ranked.get(d, [])
                within = [x for x in dists if x <= ceil]
                adaptive_ks.append(len(within))
                if not within:
                    zero_count += 1
            sv = sorted(adaptive_ks)
            zero_pct = 100.0 * zero_count / len(adaptive_ks) if adaptive_ks else 0
            print(
                f"    ceiling={ceil:.1f}σ | 0-analogue: {zero_count}/{len(adaptive_ks)}"
                f" ({zero_pct:.1f}%) | median-K={percentile(sv,50):.0f}"
                f" p25={percentile(sv,25):.0f} p75={percentile(sv,75):.0f}"
            )

    # Show the specific 2026-05-13 case if present
    target = "2026-05-13"
    if target in all_ranked:
        dists = all_ranked[target]
        print(f"\n=== Specific case {target} (0.54σ anchor) ===")
        print(f"  All distances returned: {[round(d,2) for d in dists]}")
        for ceil in [2.0, 2.5, 3.0, 3.5, 4.0]:
            within = [d for d in dists if d <= ceil]
            print(f"  ceiling={ceil:.1f}σ → {len(within)} analogues within ceiling"
                  + (f" (max={max(within):.2f})" if within else " (0 — CI degrades)"))
    else:
        print(f"\n  Note: {target} not in corpus (no feature vector stored).")

    print("\nDone.")


if __name__ == "__main__":
    main()
