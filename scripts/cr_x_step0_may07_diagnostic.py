"""CR-X Step 0 diagnostic — 2026-05-07 pin-day zero-analogue investigation.

Ranks the 2026-05-07 magnetic-pin anchor under knn_config_version=v0 (infinite
ceiling) to answer:
  1. Are the 20 nearest days relevant pin/near-pin days sitting just over 4.0σ
     (e.g. 4.1–4.9σ), meaning the ceiling is clipping real precedent?
  2. Or are the nearest genuinely far (6σ+) — honest zero?
  3. Per-feature distance breakdown for the 2–3 nearest days.
  4. Confirm v1 (4.0σ ceiling) returns 0 — ruling out before_date misfire.

Read-only. No writes to the database.
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
from packages.shared.day_features import EPSILON, FEATURE_NAMES
from packages.shared.knn import feature_stats, rank_analogues, similarity_distance
from packages.shared.knn_config import get_knn_config

ANCHOR_DATE = "2026-05-07"
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


def feature_distance_breakdown(query_vec: dict, candidate_vec: dict,
                               stats: dict, top_n: int = 10) -> list[dict]:
    """Return per-feature σ-distance contributions, sorted by contribution desc."""
    n_total = len(FEATURE_NAMES)
    active_contribs = []
    for name in FEATURE_NAMES:
        q = query_vec.get(name)
        c = candidate_vec.get(name)
        if q is None or c is None:
            continue
        try:
            qf, cf = float(q), float(c)
        except (TypeError, ValueError):
            continue
        fstats = stats.get(name, {"mean": 0.0, "std": 0.0})
        std = max(fstats["std"], EPSILON)
        z_q = (qf - fstats["mean"]) / std
        z_c = (cf - fstats["mean"]) / std
        sq = (z_q - z_c) ** 2
        active_contribs.append({
            "feature": name,
            "sq_contribution": sq,
            "q_value": qf,
            "c_value": cf,
            "z_q": round(z_q, 3),
            "z_c": round(z_c, 3),
            "z_diff": round(z_q - z_c, 3),
        })

    n_active = len(active_contribs)
    rescale = math.sqrt(n_total / n_active) if n_active > 0 else 1.0
    total_sq = sum(x["sq_contribution"] for x in active_contribs)

    for x in active_contribs:
        # Contribution to total distance (as fraction of sum_sq before rescale)
        x["pct_of_distance_sq"] = round(
            100.0 * x["sq_contribution"] / total_sq, 1
        ) if total_sq > 0 else 0.0
        x["sq_contribution"] = round(x["sq_contribution"], 4)

    active_contribs.sort(key=lambda x: x["sq_contribution"], reverse=True)
    return active_contribs[:top_n]


def main():
    conn = _conn()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector, regime_at_classification "
            "FROM bt_daily_features_active "
            "WHERE ticker = 'SPX' AND feature_version = %s "
            "ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()

    # Build (date_iso, vector) pairs; also keep regime lookup
    candidates = []
    regime_lookup: dict[str, str] = {}
    for (d, v, regime) in rows:
        iso = d.isoformat()
        candidates.append((iso, v))
        regime_lookup[iso] = regime or "unknown"

    stats = feature_stats(v for (_, v) in candidates)
    total = len(candidates)
    print(f"\nCorpus: {total} rows, feature_version={CANONICAL_FEATURE_VERSION}")

    # Confirm anchor exists and get its vector
    anchor_vec = next((v for (d, v) in candidates if d == ANCHOR_DATE), None)
    if anchor_vec is None:
        print(f"\nERROR: {ANCHOR_DATE} not found in corpus.")
        return

    anchor_regime = regime_lookup.get(ANCHOR_DATE, "unknown")
    print(f"Anchor: {ANCHOR_DATE}  regime={anchor_regime}")

    # ── v0: infinite ceiling ────────────────────────────────────────────────
    cfg_v0 = get_knn_config("v0")
    v0_results = rank_analogues(
        anchor_vec, candidates, K,
        exclude_date=ANCHOR_DATE,
        stats=stats,
        distance_ceiling=cfg_v0["distance_ceiling"],
    )

    # ── v1: 4.0σ ceiling ───────────────────────────────────────────────────
    cfg_v1 = get_knn_config("v1")
    v1_results = rank_analogues(
        anchor_vec, candidates, K,
        exclude_date=ANCHOR_DATE,
        stats=stats,
        distance_ceiling=cfg_v1["distance_ceiling"],
    )

    # ── Table: 20 nearest under v0 ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"20 nearest days under v0 (no ceiling) — anchor={ANCHOR_DATE}")
    print(f"{'─'*70}")
    CEILING = cfg_v1["distance_ceiling"]
    print(f"{'Rank':<5} {'Date':<13} {'Distance σ':<12} {'Over 4σ?':<10} {'Regime'}")
    print(f"{'─'*70}")
    for rank, (date, dist) in enumerate(v0_results, 1):
        over = "*** OVER ***" if dist > CEILING else ""
        regime = regime_lookup.get(date, "unknown")
        print(f"  {rank:<3}  {date:<13}  {dist:>8.3f}σ  {over:<12}  {regime}")

    print(f"\nv0 returned {len(v0_results)} analogues (ceiling=inf)")
    print(f"v1 returned {len(v1_results)} analogues (ceiling={CEILING}σ)")

    if v0_results:
        dists_v0 = [d for (_, d) in v0_results]
        below = [(date, d) for (date, d) in v0_results if d <= CEILING]
        above = [(date, d) for (date, d) in v0_results if d > CEILING]
        print(f"\nSummary:")
        print(f"  Min distance:      {min(dists_v0):.3f}σ")
        print(f"  Max distance:      {max(dists_v0):.3f}σ")
        print(f"  Within 4.0σ:       {len(below)} (these survive ceiling)")
        print(f"  4.0σ–5.0σ:         {sum(1 for (_, d) in above if d <= 5.0)}")
        print(f"  5.0σ–6.0σ:         {sum(1 for (_, d) in above if 4.0 < d <= 5.0)} (wait, recheck)")
        # Recompute
        near_clip = [(date, d) for (date, d) in above if d <= 5.0]
        mid_clip  = [(date, d) for (date, d) in above if 5.0 < d <= 6.0]
        far_clip  = [(date, d) for (date, d) in above if d > 6.0]
        print(f"\n  Days clipped by ceiling breakdown:")
        print(f"    4.01σ–5.00σ (near-clip): {len(near_clip)} — {[d for (_, d) in near_clip]}")
        print(f"    5.01σ–6.00σ (mid-clip):  {len(mid_clip)} — {[d for (_, d) in mid_clip]}")
        print(f"    >6.00σ (genuinely far):  {len(far_clip)} — {[d for (_, d) in far_clip]}")

        # Regime breakdown of all 20 neighbours
        print(f"\n  Regime breakdown of 20 nearest:")
        from collections import Counter
        ctr = Counter(regime_lookup.get(d, "unknown") for (d, _) in v0_results)
        for regime, n in ctr.most_common():
            marker = " ← pin" if regime == "magnetic-pin" else ""
            print(f"    {regime:<25} {n:>3}{marker}")

    # ── Confirm v1 is purely ceiling-driven (not before_date misfire) ───────
    print(f"\n{'─'*70}")
    print("Ceiling misfire check")
    print(f"{'─'*70}")
    if len(v0_results) > 0 and len(v1_results) == 0:
        print(f"  v0={len(v0_results)} analogue(s), v1=0 — zero is PURELY the 4.0σ ceiling.")
        print(f"  Candidate pool is full (not empty); before_date not set → no misfire.")
        print(f"  Min v0 distance: {min(d for (_, d) in v0_results):.3f}σ "
              f"(nearest day is already >{CEILING}σ)")
    elif len(v1_results) > 0:
        print(f"  v1 returned {len(v1_results)} — anchor is NOT a zero-analogue case under v1.")
    else:
        print(f"  Both v0 and v1 returned 0 — candidate pool issue.")

    # ── Per-feature breakdown for nearest 3 days ───────────────────────────
    print(f"\n{'─'*70}")
    print("Per-feature distance breakdown — top 3 nearest (v0)")
    print(f"{'─'*70}")
    top3 = v0_results[:3]
    vec_lookup = {d: v for (d, v) in candidates}
    for (date, dist) in top3:
        cand_vec = vec_lookup[date]
        regime = regime_lookup.get(date, "unknown")
        print(f"\n  Neighbour: {date}  distance={dist:.3f}σ  regime={regime}")
        print(f"  {'Feature':<42} {'%dist²':>7}  {'z_q':>7}  {'z_c':>7}  {'z_diff':>7}  raw_q→raw_c")
        breakdown = feature_distance_breakdown(anchor_vec, cand_vec, stats, top_n=15)
        for row in breakdown:
            if row["sq_contribution"] < 0.001:
                break
            print(f"  {row['feature']:<42} {row['pct_of_distance_sq']:>6.1f}%"
                  f"  {row['z_q']:>7.3f}  {row['z_c']:>7.3f}  {row['z_diff']:>7.3f}"
                  f"  {row['q_value']:.3f}→{row['c_value']:.3f}")

    print(f"\n{'═'*70}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
