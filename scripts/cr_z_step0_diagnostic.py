"""CR-031 (CR-Z) Step 0 — Lock analysis.

Three measurements:
  1. Within-ceiling count distribution (cap=9999, so ceiling is the sole gate).
     Answers: how many analogues does a common setup get? Sets the safety bound.
  2. Half-life coherence sweep: 18 / 24 / 36 months, everything else v2.
     Picks the half-life by coherence held + live K at corpus edge.
  3. Zero-analogue rate + count re-check for each half-life (ceiling re-confirm).

No commits — this is the lock document.
"""
from __future__ import annotations

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import psycopg

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats as compute_feature_stats, rank_analogues
from packages.shared.knn_coherence import compare_configs, run_coherence_check
from packages.shared.knn_config import KNN_CONFIGS, get_knn_config


_UNCAPPED_K = 9999           # effectively no k-cap; ceiling is the only gate
_V2_CEILING = 5.0            # v2 distance_ceiling
_EDGE_ANCHOR = "2026-05-07"  # CR-030 known K-starved date


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
    """(date_iso, feature_vec) candidates + regime lookup."""
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
    return candidates, regime_by_date


def _load_outcomes(conn, dates):
    from datetime import date as Date
    date_objs = [Date.fromisoformat(d) for d in dates]
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH session AS (
                SELECT
                    (datetime AT TIME ZONE 'UTC'
                               AT TIME ZONE 'America/Los_Angeles')::date AS trade_date,
                    open, high, low, close,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            (datetime AT TIME ZONE 'UTC'
                                       AT TIME ZONE 'America/Los_Angeles')::date
                        ORDER BY datetime ASC
                    ) AS rn_asc,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            (datetime AT TIME ZONE 'UTC'
                                       AT TIME ZONE 'America/Los_Angeles')::date
                        ORDER BY datetime DESC
                    ) AS rn_desc
                FROM ironbeam_es_1m_bars
                WHERE (datetime AT TIME ZONE 'UTC'
                                AT TIME ZONE 'America/Los_Angeles')::date = ANY(%s)
            )
            SELECT
                trade_date,
                MAX(CASE WHEN rn_asc  = 1 THEN open  END),
                MAX(CASE WHEN rn_desc = 1 THEN close END),
                MAX(high), MIN(low)
            FROM session GROUP BY trade_date ORDER BY trade_date
            """,
            (date_objs,),
        )
        rows = cur.fetchall()
    outcomes = {}
    for (td, fo, lc, dh, dl) in rows:
        d = td.isoformat()
        out = {}
        if fo is not None and lc is not None:
            out["eod_return_pts"] = float(lc) - float(fo)
        if dh is not None and dl is not None:
            out["intraday_range_pts"] = float(dh) - float(dl)
        if out:
            outcomes[d] = out
    return outcomes


def _within_ceiling_counts(candidates, cfg, label):
    """For each anchor, count analogues inside the ceiling with k=_UNCAPPED_K."""
    global_stats = compute_feature_stats(v for (_, v) in candidates)
    counts = []
    for anchor_date, anchor_vec in candidates:
        ranked = rank_analogues(
            anchor_vec, candidates, _UNCAPPED_K,
            exclude_date=anchor_date,
            before_date=anchor_date,      # as-of-history guard
            stats=global_stats,
            distance_ceiling=cfg.get("distance_ceiling", math.inf),
            feature_weights=cfg.get("feature_weights"),
            z_diff_cap=cfg.get("z_diff_cap"),
            half_life_months=cfg.get("half_life_months"),
            anchor_date=anchor_date,
        )
        counts.append(len(ranked))
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    zero_rate = sum(1 for c in counts if c == 0) / n if n else 0
    capped_at_20 = sum(1 for c in counts if c >= 20) / n if n else 0
    pct = lambda p: sorted_counts[int(p * n)] if n else 0
    print(f"\n{label}:")
    print(f"  n_anchors={n}  zero_rate={zero_rate:.1%}  pct_would_hit_K20_cap={capped_at_20:.1%}")
    print(f"  count distribution: "
          f"min={min(sorted_counts)}  p10={pct(.10)}  p25={pct(.25)}  "
          f"median={pct(.50)}  p75={pct(.75)}  p90={pct(.90)}  p95={pct(.95)}  "
          f"max={max(sorted_counts)}  mean={statistics.mean(counts):.1f}")
    return counts


def _edge_k(candidates, cfg, label):
    """Report K for the known K-starved edge anchor (2026-05-07)."""
    global_stats = compute_feature_stats(v for (_, v) in candidates)
    vec_by_date = {d: v for (d, v) in candidates}
    anchor_vec = vec_by_date.get(_EDGE_ANCHOR)
    if anchor_vec is None:
        print(f"  [{label}] Edge anchor {_EDGE_ANCHOR} not in corpus")
        return None
    ranked = rank_analogues(
        anchor_vec, candidates, _UNCAPPED_K,
        exclude_date=_EDGE_ANCHOR,
        before_date=_EDGE_ANCHOR,
        stats=global_stats,
        distance_ceiling=cfg.get("distance_ceiling", math.inf),
        feature_weights=cfg.get("feature_weights"),
        z_diff_cap=cfg.get("z_diff_cap"),
        half_life_months=cfg.get("half_life_months"),
        anchor_date=_EDGE_ANCHOR,
    )
    k = len(ranked)
    print(f"  [{label}] {_EDGE_ANCHOR} edge anchor: K={k}")
    return k


def main():
    conn = _conn()
    candidates, regime_by_date = _load_corpus(conn)
    dates = [d for (d, _) in candidates]
    print(f"Corpus: {len(candidates)} rows, feature_version={CANONICAL_FEATURE_VERSION}")

    outcomes = _load_outcomes(conn, dates)
    print(f"Outcome coverage: {len(outcomes)}/{len(dates)} dates have ES bar data")
    conn.close()

    # ── SECTION 1: Within-ceiling count distribution ─────────────────────────
    print(f"\n{'='*72}")
    print("Section 1: Within-ceiling count distribution (uncapped, k=9999)")
    print("(Before_date guard active — as-of-history counts)")
    print(f"{'='*72}")

    cfg_v2 = get_knn_config("v2")  # half_life=18, ceiling=5.0
    counts_v2 = _within_ceiling_counts(
        candidates, cfg_v2, "v2 (hl=18, ceiling=5.0)"
    )

    # Derive safety bound from p95/max
    sorted_counts = sorted(counts_v2)
    p99 = sorted_counts[int(0.99 * len(sorted_counts))]
    print(f"\n  → p99 within-ceiling count: {p99}  (safety bound proposal: 200)")
    print(f"  → 200 leaves room above all observed counts unless the corpus")
    print(f"    grows to an extreme density; effectively never binds.")

    # ── SECTION 2: Half-life coherence sweep ─────────────────────────────────
    print(f"\n{'='*72}")
    print("Section 2: Half-life coherence sweep (18 / 24 / 36 months)")
    print("(LOO, before_date guard active)")
    print(f"{'='*72}")

    # Build v3-candidate configs: only half_life differs
    cfg_base = dict(cfg_v2)  # copy
    hl_configs = []
    for hl in [18, 24, 36]:
        c = dict(cfg_base, half_life_months=float(hl))
        hl_configs.append((f"hl={hl}mo (ceiling=5.0)", c))

    results = compare_configs(
        candidates, outcomes, regime_by_date,
        hl_configs, min_k=5, k=_UNCAPPED_K,
    )
    print(f"\n{'Label':<35} {'Coherence':>10} {'K=[min..med..max]':<22} n_anchors")
    for label, res in results.items():
        if res.coherence is None:
            print(f"  {label:<35} {'—':>10}  no scored anchors")
        else:
            ks = res.k_stats
            k_str = f"[{ks.get('min',0):.0f}..{ks.get('median',0):.0f}..{ks.get('max',0):.0f}]"
            print(f"  {label:<35} {res.coherence:>10.4f}  {k_str:<22} {res.n_anchors_scored}/{res.n_anchors_total}")

    # A/B vs v2 baseline (hl=18 is v2, so Δ will be 0 for that one)
    baseline_res = results.get("hl=18mo (ceiling=5.0)")
    if baseline_res and baseline_res.coherence:
        print(f"\n  A/B vs hl=18 baseline ({baseline_res.coherence:.4f}):")
        for label, res in results.items():
            if res.coherence and label != "hl=18mo (ceiling=5.0)":
                delta = res.coherence - baseline_res.coherence
                pct_chg = 100 * delta / baseline_res.coherence
                print(f"    {label}: Δ={delta:+.4f} ({pct_chg:+.1f}%)")

    # ── SECTION 3: Edge K per half-life ─────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Section 3: Edge anchor K check ({_EDGE_ANCHOR})")
    print(f"{'='*72}")
    for label, cfg in hl_configs:
        _edge_k(candidates, cfg, label)

    # ── SECTION 4: Within-ceiling counts at 24 and 36 months ─────────────────
    print(f"\n{'='*72}")
    print("Section 4: Count distribution at other half-lives (ceiling re-confirm)")
    print(f"{'='*72}")
    for label, cfg in hl_configs[1:]:  # skip hl=18 (already done above)
        _within_ceiling_counts(candidates, cfg, label)

    # ── SUMMARY / LOCK DECISIONS ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Step 0 lock summary")
    print(f"{'='*72}")
    print("  Safety bound: _K_SAFETY = 200")
    print("  Half-life: see Section 2 coherence table — pick highest while")
    print("    edge anchor K >= 3")
    print("  Ceiling re-pick: needed only if zero-analogue rate jumps above 5%")
    print("    after half-life change (see Section 4)")
    print("\nDone.")


if __name__ == "__main__":
    main()
