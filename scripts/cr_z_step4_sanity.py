"""CR-031 (CR-Z) Step 4 — Probability path live sanity check.

Confirms the probability path (compute_structural_probability /
_rank_analogues_with_outcomes) now uses the ceiling as the real gate:
  1. A common setup returns K>20 with a tighter Wilson CI.
  2. The post-touch distribution denominators are larger (more touchers).
  3. A rare setup still returns its honest small K.

Compares k=20 (old capped path) vs k=200 (new safety-bound path) side-by-side.
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
from packages.shared.knn_config import get_knn_config
from packages.shared.probability import (
    _rank_analogues_with_outcomes,
    _aggregate_outcomes,
    aggregate_post_touch_distribution,
    _derive_anchor_bucket,
)
from packages.shared.stats import wilson_ci


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


def _load_corpus_features(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT trade_date, feature_vector, regime_at_classification "
            "FROM bt_daily_features_active "
            "WHERE ticker='SPX' AND feature_version=%s ORDER BY trade_date",
            (CANONICAL_FEATURE_VERSION,),
        )
        rows = cur.fetchall()
    return {d.isoformat(): (v, r) for (d, v, r) in rows}


def _run_prob_path(conn, anchor_date, anchor_vec, k, label):
    """Run probability path and report K, CI, post-touch denominators."""
    rows = _rank_analogues_with_outcomes(
        anchor_vec, conn, k, CANONICAL_FEATURE_VERSION,
        ticker="SPX",
        exclude_date=anchor_date,
    )
    result = _aggregate_outcomes(rows)
    k_actual = len(rows)
    k_outcomes = result.get("k_with_outcomes", 0)
    touch_rate = result.get("touch_rate")
    ci_lo = result.get("touch_ci_lower")
    ci_hi = result.get("touch_ci_upper")
    ci_width = round(ci_hi - ci_lo, 4) if ci_lo is not None and ci_hi is not None else None

    anchor_bucket = _derive_anchor_bucket(anchor_vec)
    post_touch = None
    if anchor_bucket and rows:
        pt = aggregate_post_touch_distribution(rows, anchor_bucket)
        post_touch = {
            "filter_mode": pt["filter_mode"],
            "same_bucket_n": pt["same_bucket_n"],
            "total_touchers": pt["total_touchers"],
            "denom_t1": pt["denominator_t1"],
            "pattern_label": pt["pattern_label"],
        }

    print(f"  [{label}] k_requested={k} → K={k_actual} (K_outcomes={k_outcomes})")
    if touch_rate is not None:
        print(f"    touch_rate={touch_rate:.3f}  CI=[{ci_lo:.3f},{ci_hi:.3f}]  CI_width={ci_width:.4f}")
    else:
        print(f"    touch_rate=None (no computed outcomes)")
    if post_touch:
        print(f"    post_touch: {post_touch}")
    return k_actual


def main():
    conn = _conn()
    corpus = _load_corpus_features(conn)
    print(f"Corpus: {len(corpus)} rows, feature_version={CANONICAL_FEATURE_VERSION}")

    # Pick a common setup: 2026-04-29 had K=80 in Step 3
    # and a rare one: 2026-05-07 (K=1)
    test_cases = [
        ("2026-04-29", "common (Step 3: K=80)"),
        ("2026-05-04", "common (Step 3: K=85)"),
        ("2026-05-07", "rare — two-pin-cluster (Step 3: K=1)"),
        ("2026-04-28", "rare (Step 3: K=1)"),
    ]

    print(f"\n{'='*72}")
    print("Probability path sanity: k=20 (old) vs k=200 (new)")
    print(f"{'='*72}")

    for anchor_date, desc in test_cases:
        if anchor_date not in corpus:
            print(f"\n{anchor_date} ({desc}): NOT IN CORPUS — skipped")
            continue
        anchor_vec, regime = corpus[anchor_date]
        print(f"\n{anchor_date} ({desc}) — regime={regime}")

        k_old = _run_prob_path(conn, anchor_date, anchor_vec, k=20,  label="k=20 old")
        k_new = _run_prob_path(conn, anchor_date, anchor_vec, k=200, label="k=200 new")

        if k_new > k_old:
            print(f"  ✓ K grew: {k_old} → {k_new} (cap was binding)")
        elif k_new == k_old:
            print(f"  ✓ K unchanged: {k_old} (ceiling is the real gate, not cap)")

    conn.close()
    print(f"\n{'='*72}")
    print("Step 4 sanity complete.")
    print("Verify: common setups show larger K and tighter CI at k=200 vs k=20.")
    print("Rare setups should show same small K at both limits.")


if __name__ == "__main__":
    main()
