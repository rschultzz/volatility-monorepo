"""CR-030 (CR-X) Step 2 — Leave-one-out coherence check.

Measures whether the v2 weighted config (expert priors + z_diff_cap + recency)
finds analogues whose actual outcomes are more similar to the anchor's outcome
than the v1 (ceiling-only) baseline.

Outcome dimensions: EOD return (pts) + intraday range (pts) from ES bars.
Metric: resemblance = max(0, 1 - |anchor_outcome - mean(neighbour_outcomes)| /
                                   std(neighbour_outcomes))
Aggregated coherence = mean(resemblance) over all anchors with K ≥ 5.

As-of-history safety: for every anchor, only candidates dated BEFORE the anchor
are passed to rank_analogues (before_date=anchor_date). A smoke test with
before_date=None demonstrates the guard makes a difference (lookahead inflates
the score).
"""
from __future__ import annotations

import os
import sys
from datetime import date as Date
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import psycopg

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn_coherence import compare_configs, run_coherence_check
from packages.shared.knn_config import get_knn_config

_UTC = ZoneInfo("UTC")
_PT  = ZoneInfo("America/Los_Angeles")


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


def _load_corpus(conn) -> tuple[list, dict]:
    """Load (date_iso, feature_vec) candidates + regime lookup."""
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


def _load_outcomes_batch(conn, dates: list[str]) -> dict[str, dict]:
    """Batch-load EOD outcomes from ironbeam_es_1m_bars for a list of dates.

    Returns {date_iso: {eod_return_pts, intraday_range_pts}} for dates that
    have bar data. Dates with no bars are absent from the dict.

    ES bars stored as UTC-naive datetimes; convert to PT for date matching.
    """
    if not dates:
        return {}

    # Build list of python date objects
    date_objs = [Date.fromisoformat(d) for d in dates]

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH session AS (
                SELECT
                    (datetime AT TIME ZONE 'UTC'
                               AT TIME ZONE 'America/Los_Angeles')::date AS trade_date,
                    datetime,
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
                MAX(CASE WHEN rn_asc  = 1 THEN open  END) AS first_open,
                MAX(CASE WHEN rn_desc = 1 THEN close END) AS last_close,
                MAX(high)                                  AS day_high,
                MIN(low)                                   AS day_low
            FROM session
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            (date_objs,),
        )
        rows = cur.fetchall()

    outcomes: dict[str, dict] = {}
    for (trade_date, first_open, last_close, day_high, day_low) in rows:
        d = trade_date.isoformat()
        out: dict = {}
        if first_open is not None and last_close is not None:
            out["eod_return_pts"] = float(last_close) - float(first_open)
        if day_high is not None and day_low is not None:
            out["intraday_range_pts"] = float(day_high) - float(day_low)
        if out:
            outcomes[d] = out
    return outcomes


def main():
    conn = _conn()
    candidates, regime_by_date = _load_corpus(conn)
    dates = [d for (d, _) in candidates]
    print(f"Corpus: {len(candidates)} rows, feature_version={CANONICAL_FEATURE_VERSION}")

    outcomes = _load_outcomes_batch(conn, dates)
    print(f"Outcome coverage: {len(outcomes)}/{len(dates)} dates have ES bar data")
    conn.close()

    configs = [
        ("v1 (baseline — ceiling only)",  get_knn_config("v1")),
        ("v2 (weighted+cap+recency)",     get_knn_config("v2")),
    ]

    print(f"\n{'='*72}")
    print("Leave-one-out coherence check (before_date guard active)")
    print(f"{'='*72}")
    results = compare_configs(candidates, outcomes, regime_by_date, configs, min_k=5)
    for label, res in results.items():
        print(f"\n{res.summary()}")
        if res.regime_scores:
            print(f"  Regime breakdown:")
            for regime, score in sorted(res.regime_scores.items(),
                                        key=lambda x: -x[1]):
                regime_n = sum(
                    1 for d in candidates
                    if regime_by_date.get(d[0]) == regime
                    and d[0] in outcomes
                )
                print(f"    {regime:<25} {score:.4f}  (n={regime_n})")

    # ── A/B verdict ────────────────────────────────────────────────────────
    r_v1 = results.get("v1 (baseline — ceiling only)")
    r_v2 = results.get("v2 (weighted+cap+recency)")
    print(f"\n{'='*72}")
    print("A/B verdict")
    print(f"{'='*72}")
    if r_v1 and r_v2 and r_v1.coherence and r_v2.coherence:
        delta = r_v2.coherence - r_v1.coherence
        pct = 100 * delta / r_v1.coherence if r_v1.coherence else 0
        print(f"  v1 coherence: {r_v1.coherence:.4f}")
        print(f"  v2 coherence: {r_v2.coherence:.4f}")
        print(f"  Δ = {delta:+.4f} ({pct:+.1f}%)")
        if delta > 0:
            print(f"  ✓ v2 BEATS v1 — weighted config improves coherence")
        elif delta < 0:
            print(f"  ✗ v2 WORSE than v1 — re-examine weights")
        else:
            print(f"  = No change in coherence")

    # ── SMOKE TEST: lookahead check ────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Smoke test: lookahead guard (before_date=None vs before_date=anchor)")
    print(f"{'='*72}")
    # Run v2 without guard on a small subset (first 50 anchors with outcomes)
    subset_dates = [d for d in dates if d in outcomes][:50]
    subset_cands = [(d, v) for (d, v) in candidates if d in set(subset_dates)]

    res_guarded   = run_coherence_check(
        candidates, outcomes, regime_by_date, get_knn_config("v2"),
        "v2-guarded-subset", min_k=5
    )
    # Lookahead version: omit before_date (deliberately leaking)
    res_lookahead = run_coherence_check(
        candidates, outcomes, regime_by_date, get_knn_config("v2"),
        "v2-lookahead-subset", min_k=5, lookahead=True
    )
    print(f"  Guarded   (before_date=anchor): coherence={res_guarded.coherence}")
    print(f"  Lookahead (before_date=None):   coherence={res_lookahead.coherence}")
    if (res_guarded.coherence is not None
            and res_lookahead.coherence is not None
            and res_lookahead.coherence > res_guarded.coherence):
        print(f"  ✓ LOOKAHEAD INFLATES SCORE as expected — guard is load-bearing")
    elif res_lookahead.coherence == res_guarded.coherence:
        print(f"  ! Scores identical — lookahead may not matter here "
              f"(corpus very recent / outcomes sparse)")
    else:
        print(f"  ? Unexpected: lookahead ≤ guarded — investigate")

    print(f"\n{'='*72}")
    print("Coherence check complete.")


if __name__ == "__main__":
    main()
