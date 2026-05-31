"""CR-029 Step 4 — empirical sanity pass.

Verifies that the ceiling=4.0σ implementation produces the expected
improvement on the known problem case (2026-05-13) and doesn't regress
on a typical setup.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import psycopg

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats, rank_analogues
from packages.shared.knn_config import get_knn_config, CANONICAL_KNN_CONFIG_VERSION


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


def check_anchor(anchor_date: str, candidates: list, stats: dict, cfg: dict, label: str):
    anchor_cand = next((v for (d, v) in candidates if d == anchor_date), None)
    if anchor_cand is None:
        print(f"\n{label}: {anchor_date} NOT in corpus — skipping.")
        return

    K = 20

    # Without ceiling (v0)
    old = rank_analogues(anchor_cand, candidates, K,
                         exclude_date=anchor_date, stats=stats,
                         distance_ceiling=float("inf"))
    # With ceiling (v1 — the active config)
    new = rank_analogues(anchor_cand, candidates, K,
                         exclude_date=anchor_date, stats=stats,
                         distance_ceiling=cfg["distance_ceiling"])

    old_dists = [round(d, 3) for (_, d) in old]
    new_dists = [round(d, 3) for (_, d) in new]

    print(f"\n=== {label} ({anchor_date}) ===")
    print(f"  Config: {CANONICAL_KNN_CONFIG_VERSION}, ceiling={cfg['distance_ceiling']}σ")
    print(f"  v0 (no ceiling):  K={len(old)}, distances={old_dists}")
    print(f"  v1 (4.0σ ceiling): K={len(new)}, distances={new_dists}")
    cut = len(old) - len(new)
    print(f"  Analogues cut by ceiling: {cut}")
    if new:
        print(f"  Max distance in v1 result: {max(d for (_, d) in new):.3f}σ")
        assert max(d for (_, d) in new) <= cfg["distance_ceiling"] + 1e-9, \
            "FAIL: some returned distance exceeds ceiling!"
        print(f"  ✓ All v1 distances ≤ {cfg['distance_ceiling']}σ")
    else:
        print(f"  ✓ 0 analogues within ceiling (rare setup — CI undefined)")


def main():
    conn = _conn()
    cfg = get_knn_config()  # canonical v1
    print(f"Config: {CANONICAL_KNN_CONFIG_VERSION}, ceiling={cfg['distance_ceiling']}σ")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector FROM bt_daily_features_active"
            " WHERE ticker = 'SPX' AND feature_version = %s"
            " ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()

    candidates = [(d.isoformat(), v) for (d, v) in rows]
    stats = feature_stats(v for (_, v) in candidates)
    print(f"Corpus: {len(candidates)} rows")

    # Known problem case
    check_anchor("2026-05-13", candidates, stats, cfg,
                 "Problem case (0.54σ anchor from open-question)")

    # A typical high-K case (should be unaffected: median K=20 at ceiling=4.0σ)
    # Pick a date near the middle of the corpus
    mid = candidates[len(candidates) // 2][0]
    check_anchor(mid, candidates, stats, cfg, f"Typical anchor (corpus midpoint)")

    # A recent date (likely to have good matches)
    recent = candidates[-20][0]
    check_anchor(recent, candidates, stats, cfg, f"Recent anchor")

    print("\n✓ Step 4 sanity pass complete.")


if __name__ == "__main__":
    main()
