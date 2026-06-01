"""CR-030 Step 3 — config selection sanity pass.

Re-verify the two named problem cases under v2:
  1. 2026-05-07 (magnetic-pin, two-pin-cluster) — was K=0 under v1
  2. 2026-05-13 (magnet-above, the CR-029 problem case) — was K=5 under v1

Also confirm:
  - cluster_2_quality_ordinal is no longer the dominant contributor for 2026-05-07
  - Zero-analogue rate under v2 (corpus-wide)
  - corpus-wide K distribution
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import psycopg

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats, rank_analogues
from packages.shared.knn_config import (
    CANONICAL_KNN_CONFIG_VERSION,
    get_knn_config,
)


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


def main():
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector, regime_at_classification "
            "FROM bt_daily_features_active "
            "WHERE ticker='SPX' AND feature_version=%s ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()

    candidates = [(d.isoformat(), v) for (d, v, _) in rows]
    regime_by_date = {d.isoformat(): (r or "unknown") for (d, _, r) in rows}
    stats = feature_stats(v for (_, v) in candidates)
    vec_by_date = {d: v for (d, v) in candidates}

    cfg_v1 = get_knn_config("v1")
    cfg_v2 = get_knn_config(CANONICAL_KNN_CONFIG_VERSION)
    K = 20

    print(f"Corpus: {len(candidates)} rows")
    print(f"Canonical config: {CANONICAL_KNN_CONFIG_VERSION!r}  ceiling={cfg_v2['distance_ceiling']}σ")

    # ── Case 1: 2026-05-07 ─────────────────────────────────────────────────
    anchor = "2026-05-07"
    print(f"\n{'='*66}")
    print(f"Case 1: {anchor}  ({regime_by_date.get(anchor)})")
    print(f"{'='*66}")
    av = vec_by_date[anchor]

    r_v1 = rank_analogues(av, candidates, K, exclude_date=anchor, stats=stats,
                          distance_ceiling=cfg_v1["distance_ceiling"])
    r_v2 = rank_analogues(av, candidates, K, exclude_date=anchor, stats=stats,
                          distance_ceiling=cfg_v2["distance_ceiling"],
                          feature_weights=cfg_v2.get("feature_weights"),
                          z_diff_cap=cfg_v2.get("z_diff_cap"),
                          half_life_months=cfg_v2.get("half_life_months"),
                          anchor_date=anchor)

    print(f"  v1 (4σ): K={len(r_v1)}", end="")
    if r_v1: print(f", nearest={r_v1[0][1]:.3f}σ, regimes={Counter(regime_by_date.get(d) for d,_ in r_v1)}")
    else: print()

    print(f"  v2 (5σ): K={len(r_v2)}", end="")
    if r_v2:
        print(f", nearest={r_v2[0][1]:.3f}σ, regimes={Counter(regime_by_date.get(d) for d,_ in r_v2)}")
        print(f"  Top-5 analogues:")
        for date, dist in r_v2[:5]:
            print(f"    {date}  {dist:.3f}σ  {regime_by_date.get(date)}")
    else:
        print()

    # Verify cluster_2_quality_ordinal no longer dominates
    if r_v2:
        from packages.shared.day_features import EPSILON, FEATURE_NAMES
        best_cand = vec_by_date[r_v2[0][0]]
        contribs = {}
        total_w_sq = 0.0
        for name in FEATURE_NAMES:
            q = av.get(name)
            c = best_cand.get(name)
            if q is None or c is None: continue
            try: qf, cf = float(q), float(c)
            except: continue
            s = stats.get(name, {"mean": 0.0, "std": 0.0})
            std = max(s["std"], EPSILON)
            diff = (qf - s["mean"]) / std - (cf - s["mean"]) / std
            cap = cfg_v2.get("z_diff_cap")
            if cap and abs(diff) > cap: diff = math.copysign(cap, diff)
            w = cfg_v2.get("feature_weights", {}).get(name, 1.0)
            wsq = w * diff * diff
            contribs[name] = wsq
            total_w_sq += wsq
        top3 = sorted(contribs.items(), key=lambda x: -x[1])[:3]
        c2q_pct = 100 * contribs.get("cluster_2_quality_ordinal", 0) / total_w_sq if total_w_sq > 0 else 0
        print(f"\n  Top-3 distance contributors for nearest ({r_v2[0][0]}):")
        for nm, wsq in top3:
            print(f"    {nm:<42} {100*wsq/total_w_sq:.1f}%")
        print(f"  cluster_2_quality_ordinal contribution: {c2q_pct:.1f}%  (was ~60% under v0, <20% target)")
        if c2q_pct < 20.0:
            print(f"  ✓ Blowup regression fixed")
        else:
            print(f"  ✗ Still dominated — review weights/cap")

    # ── Case 2: 2026-05-13 ─────────────────────────────────────────────────
    anchor2 = "2026-05-13"
    print(f"\n{'='*66}")
    print(f"Case 2: {anchor2}  ({regime_by_date.get(anchor2)})")
    print(f"{'='*66}")
    av2 = vec_by_date.get(anchor2)
    if av2:
        r2_v1 = rank_analogues(av2, candidates, K, exclude_date=anchor2, stats=stats,
                               distance_ceiling=cfg_v1["distance_ceiling"])
        r2_v2 = rank_analogues(av2, candidates, K, exclude_date=anchor2, stats=stats,
                               distance_ceiling=cfg_v2["distance_ceiling"],
                               feature_weights=cfg_v2.get("feature_weights"),
                               z_diff_cap=cfg_v2.get("z_diff_cap"),
                               half_life_months=cfg_v2.get("half_life_months"),
                               anchor_date=anchor2)
        print(f"  v1: K={len(r2_v1)}", end="")
        if r2_v1: print(f", dists=[{r2_v1[0][1]:.2f}..{r2_v1[-1][1]:.2f}]σ, regimes={Counter(regime_by_date.get(d) for d,_ in r2_v1)}")
        else: print()
        print(f"  v2: K={len(r2_v2)}", end="")
        if r2_v2: print(f", dists=[{r2_v2[0][1]:.2f}..{r2_v2[-1][1]:.2f}]σ, regimes={Counter(regime_by_date.get(d) for d,_ in r2_v2)}")
        else: print()
        if len(r2_v2) >= len(r2_v1): print(f"  ✓ No regression — K maintained or improved")
        else: print(f"  ✗ Regression — fewer analogues under v2")

    # ── Corpus-wide zero-analogue rate ─────────────────────────────────────
    print(f"\n{'='*66}")
    print("Corpus-wide K distribution under v2")
    print(f"{'='*66}")
    k_counts = []
    zeros = 0
    for d, v in candidates:
        r = rank_analogues(v, candidates, K, exclude_date=d, stats=stats,
                           distance_ceiling=cfg_v2["distance_ceiling"],
                           feature_weights=cfg_v2.get("feature_weights"),
                           z_diff_cap=cfg_v2.get("z_diff_cap"),
                           half_life_months=cfg_v2.get("half_life_months"),
                           anchor_date=d)
        k_counts.append(len(r))
        if len(r) == 0: zeros += 1

    k_sorted = sorted(k_counts)
    n = len(k_sorted)
    print(f"  Total anchors:       {n}")
    print(f"  Zero-analogue:       {zeros} ({100*zeros/n:.1f}%)  [v1 was 7.3% = 54 anchors]")
    print(f"  Median K:            {k_sorted[n//2]}")
    print(f"  K=20 (full):         {k_counts.count(20)} ({100*k_counts.count(20)/n:.1f}%)")
    print(f"  K≥10:                {sum(1 for k in k_counts if k>=10)} ({100*sum(1 for k in k_counts if k>=10)/n:.1f}%)")

    print(f"\n{'='*66}")
    print(f"Step 3 sanity complete. v2 selected as canonical ({CANONICAL_KNN_CONFIG_VERSION!r}).")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
