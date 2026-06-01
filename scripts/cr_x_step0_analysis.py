"""CR-030 (CR-X) Step 0 empirical analysis.

Applies the proposed v2 config (expert-prior weights + z_diff_cap=3.0 +
half_life=18mo) over the full corpus and answers:

  1. Does the z_diff cap + down-weighting fix the cluster_2_quality_ordinal
     blowup for 2026-05-07? (validate regression fix)
  2. What is the new distance distribution under v2?
  3. What ceiling should v2 use to get zero-analogue rate < 5%?
  4. Does 2026-05-07 find analogues under the new ceiling?
  5. Top-3 per-feature breakdowns under v2 for 2026-05-07.

NOTE: recency is NOT applied here because this is a static corpus distribution
check. Recency is applied in the live rank_analogues path using the request
date as anchor_date. For corpus statistics, we want the structural distance
distribution, not the recency-adjusted one.
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
from packages.shared.day_features import EPSILON, FEATURE_NAMES


# ── Proposed v2 expert-prior weights ──────────────────────────────────────
# Rationale groups:
#   Regime flags (2.5): regime match is the strongest structural signal
#   magnet_direction_signed (2.0): directional nuance within regime
#   cluster_1_signed_distance_sigma (3.0): THE proximity-to-wall signal
#   cluster_2/3_signed_distance_sigma (2.0/1.5): secondary wall proximity
#   implied_move_1d (2.0): vol context
#   cluster_1_quality_ordinal (1.5): TOP cluster quality matters
#   cluster_1_max_gex (1.0): absolute strength (varies widely, moderate weight)
#   cluster_2/3_max_gex (0.5/0.25): secondary magnitude less important
#   cluster_2/3_quality_ordinal (0.25): DOWN-WEIGHTED — blowup source + redundant
#   n_pin/n_target/n_feature (0.25): DOWN-WEIGHTED — correlated with quality ordinals
#   cluster aggregates (0.5): weakly informative counts
#   top_cluster_fraction (0.75): ratio somewhat informative
#   bucket dominances (0.75 each): somewhat informative
#   neg zone features (0.3–0.5): mostly noise / trivia
PROPOSED_WEIGHTS: dict[str, float] = {
    # Regime / structure
    "is_pin_day":                        2.5,
    "is_magnet_day":                     2.5,
    "is_bounded_day":                    2.5,
    "is_untethered_day":                 2.5,
    "is_amplification_day":              2.5,
    "magnet_direction_signed":           2.0,
    # Top 3 cluster slots
    "cluster_1_max_gex":                 1.0,
    "cluster_1_quality_ordinal":         1.5,
    "cluster_1_signed_distance_sigma":   3.0,
    "cluster_2_max_gex":                 0.5,
    "cluster_2_quality_ordinal":         0.25,  # DOWN-WEIGHTED: blowup source
    "cluster_2_signed_distance_sigma":   2.0,
    "cluster_3_max_gex":                 0.25,
    "cluster_3_quality_ordinal":         0.25,  # DOWN-WEIGHTED
    "cluster_3_signed_distance_sigma":   1.5,
    # Cluster aggregates
    "n_clusters_total":                  0.5,
    "n_pin":                             0.25,  # DOWN-WEIGHTED: correlated
    "n_target":                          0.25,  # DOWN-WEIGHTED: correlated
    "n_feature":                         0.25,  # DOWN-WEIGHTED: correlated
    "n_clusters_above_spot":             0.5,
    "n_clusters_below_spot":             0.5,
    "top_cluster_fraction_of_total_max_gex": 0.75,
    # Per-bucket dominance
    "dominance_0DTE":                    0.75,
    "dominance_1_7":                     0.75,
    "dominance_8_30":                    0.75,
    "dominance_30plus":                  0.75,
    # Neg zones
    "n_neg_zones":                       0.3,
    "nearest_neg_signed_distance_sigma": 0.5,
    "total_neg_max_gex":                 0.3,
    # Vol regime
    "implied_move_1d":                   2.0,
    # Vol surface placeholders (NULL in v0.5 — weights don't apply but include)
    "atm_iv_percentile":                 1.0,
    "skew_percentile":                   1.0,
    "smile_convexity":                   1.0,
    "term_structure_slope":              1.0,
    "vol_risk_premium":                  1.0,
}

Z_DIFF_CAP = 3.0  # per-feature z_diff cap (σ)
K = 20


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


def _feature_stats(vectors):
    stats = {}
    vecs = list(vectors)
    for name in FEATURE_NAMES:
        values = []
        for v in vecs:
            x = v.get(name)
            if x is None:
                continue
            try:
                values.append(float(x))
            except (TypeError, ValueError):
                continue
        if not values:
            stats[name] = {"mean": 0.0, "std": 0.0}
        elif len(values) == 1:
            stats[name] = {"mean": values[0], "std": 0.0}
        else:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            stats[name] = {"mean": mean, "std": math.sqrt(var)}
    return stats


def weighted_distance(query_vec: dict, cand_vec: dict, stats: dict,
                      weights: dict, z_diff_cap: float) -> float:
    """Weighted z_diff-capped Euclidean distance."""
    n_total = len(FEATURE_NAMES)
    sum_sq = 0.0
    n_active = 0
    for name in FEATURE_NAMES:
        q = query_vec.get(name)
        c = cand_vec.get(name)
        if q is None or c is None:
            continue
        try:
            qf, cf = float(q), float(c)
        except (TypeError, ValueError):
            continue
        s = stats.get(name, {"mean": 0.0, "std": 0.0})
        std = max(s["std"], EPSILON)
        z_q = (qf - s["mean"]) / std
        z_c = (cf - s["mean"]) / std
        diff = z_q - z_c
        if z_diff_cap is not None and abs(diff) > z_diff_cap:
            diff = math.copysign(z_diff_cap, diff)
        w = weights.get(name, 1.0)
        sum_sq += w * diff * diff
        n_active += 1
    if n_active == 0:
        return float("inf")
    rescale = math.sqrt(n_total / n_active)
    return math.sqrt(sum_sq) * rescale


def weighted_breakdown(query_vec: dict, cand_vec: dict, stats: dict,
                       weights: dict, z_diff_cap: float, top_n: int = 12) -> list:
    """Per-feature contribution breakdown for weighted distance."""
    contributions = []
    n_total = len(FEATURE_NAMES)
    items = []
    for name in FEATURE_NAMES:
        q = query_vec.get(name)
        c = cand_vec.get(name)
        if q is None or c is None:
            continue
        try:
            qf, cf = float(q), float(c)
        except (TypeError, ValueError):
            continue
        s = stats.get(name, {"mean": 0.0, "std": 0.0})
        std = max(s["std"], EPSILON)
        z_q = (qf - s["mean"]) / std
        z_c = (cf - s["mean"]) / std
        diff = z_q - z_c
        capped = False
        if z_diff_cap is not None and abs(diff) > z_diff_cap:
            diff = math.copysign(z_diff_cap, diff)
            capped = True
        w = weights.get(name, 1.0)
        items.append({
            "feature": name,
            "w_sq": w * diff * diff,
            "z_diff_raw": round(z_q - z_c, 3),
            "z_diff_eff": round(diff, 3),
            "capped": capped,
            "weight": w,
            "q_val": round(qf, 3),
            "c_val": round(cf, 3),
        })

    total_w_sq = sum(x["w_sq"] for x in items)
    items.sort(key=lambda x: x["w_sq"], reverse=True)
    for x in items:
        x["pct"] = round(100.0 * x["w_sq"] / total_w_sq, 1) if total_w_sq > 0 else 0.0
    return items[:top_n]


def rank_weighted(anchor_vec: dict, candidates: list, k: int,
                  exclude_date: str, stats: dict,
                  weights: dict, z_diff_cap: float,
                  ceiling: float = math.inf) -> list:
    scored = [
        (d, weighted_distance(anchor_vec, v, stats, weights, z_diff_cap))
        for (d, v) in candidates
        if d != exclude_date
    ]
    scored.sort(key=lambda x: x[1])
    if math.isfinite(ceiling):
        scored = [(d, dist) for (d, dist) in scored if dist <= ceiling]
    return scored[:k]


def percentile(sorted_list: list, p: float) -> float:
    n = len(sorted_list)
    if n == 0:
        return float("nan")
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_list[lo] * (1 - frac) + sorted_list[hi] * frac


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
    regime_lookup = {d.isoformat(): (r or "unknown") for (d, _, r) in rows}
    stats = _feature_stats(v for (_, v) in candidates)
    print(f"Corpus: {len(candidates)} rows")
    print(f"z_diff_cap: {Z_DIFF_CAP}σ")

    # ── 1. Validate 2026-05-07 z_diff cap fix ─────────────────────────────
    ANCHOR = "2026-05-07"
    anchor_vec = next((v for (d, v) in candidates if d == ANCHOR), None)
    if anchor_vec is None:
        print(f"\nERROR: {ANCHOR} not in corpus")
        return

    print(f"\n{'='*70}")
    print(f"Section 1: z_diff cap validation for {ANCHOR}")
    print(f"{'='*70}")
    # Top 3 nearest under v2 (no ceiling yet — just check distances)
    v2_all = rank_weighted(anchor_vec, candidates, K, ANCHOR, stats,
                           PROPOSED_WEIGHTS, Z_DIFF_CAP)
    print(f"\nNearest 10 under v2 weights+cap (no ceiling):")
    print(f"  {'Rank':<5} {'Date':<13} {'Dist v2':>9}  {'Regime'}")
    for i, (d, dist) in enumerate(v2_all[:10], 1):
        print(f"  {i:<5} {d:<13}  {dist:>7.3f}σ  {regime_lookup.get(d, '?')}")

    print(f"\nPer-feature breakdown (top 3 nearest under v2):")
    vec_lookup = {d: v for (d, v) in candidates}
    for (date, dist) in v2_all[:3]:
        cand_vec = vec_lookup[date]
        regime = regime_lookup.get(date, "?")
        bd = weighted_breakdown(anchor_vec, cand_vec, stats,
                                PROPOSED_WEIGHTS, Z_DIFF_CAP)
        print(f"\n  Neighbour: {date}  dist_v2={dist:.3f}σ  regime={regime}")
        print(f"  {'Feature':<42} {'%':>6}  {'w':>5}  {'z_diff_raw':>10}  {'capped':>6}  raw_q→raw_c")
        for r in bd:
            if r["pct"] < 0.5:
                break
            cap_mark = "CAP" if r["capped"] else ""
            print(f"  {r['feature']:<42} {r['pct']:>5.1f}%  {r['weight']:>5.2f}"
                  f"  {r['z_diff_raw']:>10.3f}  {cap_mark:>6}  {r['q_val']}→{r['c_val']}")

    # Confirm cluster_2_quality_ordinal no longer dominates
    nearest_d, nearest_dist = v2_all[0]
    bd_nearest = weighted_breakdown(anchor_vec, vec_lookup[nearest_d], stats,
                                    PROPOSED_WEIGHTS, Z_DIFF_CAP)
    c2q_pct = next((r["pct"] for r in bd_nearest if r["feature"] == "cluster_2_quality_ordinal"), 0.0)
    print(f"\n  cluster_2_quality_ordinal contribution to nearest-day distance: {c2q_pct:.1f}%")
    if c2q_pct < 20.0:
        print(f"  ✓ Blowup fixed: was ~60%, now {c2q_pct:.1f}% (well under 20%)")
    else:
        print(f"  ✗ Still dominated at {c2q_pct:.1f}% — consider lowering z_diff_cap or weight further")

    # ── 2. Full corpus nearest-neighbour distribution under v2 ──────────────
    print(f"\n{'='*70}")
    print(f"Section 2: Full corpus NN distance distribution under v2")
    print(f"{'='*70}")
    nn_dists = []  # nearest-neighbour distance per anchor
    k20_dists = []  # 20th-neighbour distance per anchor (worst of K)
    all_dists = []  # all returned distances (for the full distribution)
    zero_anchors_at_ceilings = {}  # ceiling → zero-analogue count

    for i, (anchor_date, anchor_v) in enumerate(candidates):
        ranked = rank_weighted(anchor_v, candidates, K, anchor_date, stats,
                               PROPOSED_WEIGHTS, Z_DIFF_CAP)
        if not ranked:
            nn_dists.append(float("inf"))
            k20_dists.append(float("inf"))
            continue
        nn_dists.append(ranked[0][1])
        k20_dists.append(ranked[-1][1])
        for (_, d) in ranked:
            all_dists.append(d)

    nn_dists_finite = sorted(d for d in nn_dists if math.isfinite(d))
    k20_finite = sorted(d for d in k20_dists if math.isfinite(d))
    all_sorted = sorted(d for d in all_dists if math.isfinite(d))

    print(f"\nNearest-neighbour distance (1st of K=20):")
    print(f"  min={nn_dists_finite[0]:.3f}σ  median={percentile(nn_dists_finite,50):.3f}σ"
          f"  p90={percentile(nn_dists_finite,90):.3f}σ  p99={percentile(nn_dists_finite,99):.3f}σ"
          f"  max={nn_dists_finite[-1]:.3f}σ")

    print(f"\n20th-neighbour distance (worst of K=20):")
    print(f"  min={k20_finite[0]:.3f}σ  median={percentile(k20_finite,50):.3f}σ"
          f"  p90={percentile(k20_finite,90):.3f}σ  p99={percentile(k20_finite,99):.3f}σ"
          f"  max={k20_finite[-1]:.3f}σ")

    print(f"\nAll returned distances (N={len(all_sorted)}):")
    print(f"  median={percentile(all_sorted,50):.3f}σ  p90={percentile(all_sorted,90):.3f}σ"
          f"  p99={percentile(all_sorted,99):.3f}σ")

    # Zero-analogue rate at various ceiling candidates
    print(f"\nZero-analogue rate at candidate ceilings (v2 weights+cap):")
    total = len(candidates)
    inf_anchors = sum(1 for d in nn_dists if not math.isfinite(d))
    print(f"  (anchors with no corpus match at all: {inf_anchors})")
    print(f"  {'Ceiling':>10}  {'Zero anchors':>13}  {'Zero rate':>10}  {'Median K':>9}")
    for ceil_v in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        zeros = sum(1 for d in nn_dists if d > ceil_v)
        # median K at this ceiling (excluding anchors with 0)
        k_vals = []
        for anchor_date, anchor_v in candidates:
            ranked = rank_weighted(anchor_v, candidates, K, anchor_date, stats,
                                   PROPOSED_WEIGHTS, Z_DIFF_CAP, ceiling=ceil_v)
            k_vals.append(len(ranked))
        k_median = sorted(k_vals)[len(k_vals) // 2]
        print(f"  {ceil_v:>10.1f}σ  {zeros:>10}  {100*zeros/total:>9.1f}%  {k_median:>9}")

    # ── 3. 2026-05-07 analogue check at various ceilings ───────────────────
    print(f"\n{'='*70}")
    print(f"Section 3: {ANCHOR} analogue count at candidate ceilings (v2)")
    print(f"{'='*70}")
    for ceil_v in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        res = rank_weighted(anchor_vec, candidates, K, ANCHOR, stats,
                            PROPOSED_WEIGHTS, Z_DIFF_CAP, ceiling=ceil_v)
        if res:
            nearest = res[0][1]
            pin_count = sum(1 for (d, _) in res if regime_lookup.get(d, "?") == "magnetic-pin")
            print(f"  ceiling={ceil_v:>5.1f}σ → K={len(res)}, nearest={nearest:.3f}σ, "
                  f"pin analogues={pin_count}")
        else:
            print(f"  ceiling={ceil_v:>5.1f}σ → K=0 (still zero)")

    # ── 4. Comparison: v0 vs v1 vs v2 for 2026-05-07 ──────────────────────
    print(f"\n{'='*70}")
    print(f"Section 4: Config comparison for {ANCHOR}")
    print(f"{'='*70}")

    from packages.shared.knn import feature_stats as knn_feature_stats, rank_analogues
    from packages.shared.knn_config import get_knn_config

    knn_stats = knn_feature_stats(v for (_, v) in candidates)

    for ver, cfg_extra in [("v0", {}), ("v1", {})]:
        cfg = get_knn_config(ver)
        res = rank_analogues(anchor_vec, candidates, K,
                             exclude_date=ANCHOR, stats=knn_stats,
                             distance_ceiling=cfg["distance_ceiling"])
        print(f"  {ver}: K={len(res)}, nearest={res[0][1]:.3f}σ" if res else f"  {ver}: K=0")

    # v2 at a few candidate ceilings
    for ceil_v in [5.0, 7.0, 10.0]:
        res = rank_weighted(anchor_vec, candidates, K, ANCHOR, stats,
                            PROPOSED_WEIGHTS, Z_DIFF_CAP, ceiling=ceil_v)
        print(f"  v2 (ceil={ceil_v}σ): K={len(res)}", end="")
        if res:
            print(f", nearest={res[0][1]:.3f}σ, regimes={Counter(regime_lookup.get(d,'?') for (d,_) in res)}")
        else:
            print()

    print(f"\n{'='*70}")
    print("Step 0 analysis complete. Use Section 2 to pick v2 ceiling.")
    print("Target: zero-analogue rate < 5%, median K >= 15.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
