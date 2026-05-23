# CR-013 — Day Analogue Comparison (Backtesting v0.5)

### Problem

The new GEX landscape system (CR-007 through CR-011) produces rich per-day structural information — regime classification, per-bucket dominance, confluence clusters with quality tags, neg zones, drift targets. We have a small but growing labeled corpus in `bt_signals` (thumbs-up/down per signal, with outcome data). What's missing is the connector between "today's landscape" and "what happened on historically similar days."

The current BacktestsV2 module is a scan engine for one specific pattern (directional zone-to-target). It doesn't expose the landscape's structural features, doesn't support a "find days like today" query, and doesn't have a comparison-mode UI. Building this into v2 would mean modifying its 25-parameter monolithic scanner — high cost and conflates two different use cases (backtest one pattern vs. compare day to day).

The right move is a new module that runs the four-stage backtesting framework's pipeline in observation mode: feature extraction (signal layer), KNN matching against the labeled corpus (degenerate attribution layer, no trade or execution stages used). This is v0.5: the smallest deliverable that exercises the framework architecture and gives the trader real value immediately.

### Goal

Ship a Day Analogue Comparison feature that:

1. Extracts a structured feature vector from each day's landscape + vol context (the *signal* layer of the framework).
2. Persists per-day feature vectors in a new `bt_daily_features` table, populated by the existing EOD cron + a backfill script.
3. Serves an endpoint that, given an anchor date, returns the K nearest historical days from the labeled corpus by σ-normalized weighted Euclidean similarity over the feature vector.
4. Surfaces those analogues in a right-docked panel in the main dashboard. Click a row → non-destructive modal showing that day's mini price chart, mini landscape, outcome metrics, and label. Today's view stays anchored on the main page throughout.

### Feature vector — locked schema for v0.5

30 active features + 5 deferred vol-surface slots. All numeric (binary indicators are 0/1 numeric for distance computation).

**Regime / structure (6 features)**
- `is_pin_day`, `is_magnet_day`, `is_bounded_day`, `is_untethered_day`, `is_amplification_day` — binary indicators derived from `landscape.regime`.
- `magnet_direction_signed` — `+1` if magnet-above, `-1` if magnet-below, `0` if not a magnet day.

Mapping (from `classify_regime` output → indicators):

| `regime` value   | `is_pin` | `is_magnet` | `is_bounded` | `is_untethered` | `is_amplification` | `magnet_direction_signed` |
|------------------|---------:|------------:|-------------:|----------------:|-------------------:|--------------------------:|
| `magnetic-pin`   |        1 |           0 |            0 |               0 |                  0 |                         0 |
| `magnet-above`   |        0 |           1 |            0 |               0 |                  0 |                        +1 |
| `magnet-below`   |        0 |           1 |            0 |               0 |                  0 |                        −1 |
| `bounded`        |        0 |           0 |            1 |               0 |                  0 |                         0 |
| `amplification`  |        0 |           0 |            0 |               0 |                  1 |                         0 |
| `untethered`     |        0 |           0 |            0 |               1 |                  0 |                         0 |
| `broken-magnet`  |        0 |           0 |            0 |               0 |                  0 |                         0 |

`broken-magnet` maps to all-zero across the indicators in v0.5 (lossy but unambiguous). Adding a `is_broken_magnet_day` slot is queued as a Phase-2 schema follow-up — see Step 0 notes.

**Top 3 cluster slots (9 features)**
- For each slot i ∈ {1, 2, 3}, ordered by `max_gex` descending:
  - `cluster_{i}_max_gex` — raw $B, the cluster's `max_gex` value.
  - `cluster_{i}_quality_ordinal` — `pin = 2`, `target = 1`, `feature = 0`.
  - `cluster_{i}_signed_distance_sigma` — `(cluster.center_price - spot) / implied_move_1d`. Positive = above spot, negative = below spot. **σ-normalized at extraction time.**
- When fewer than 3 clusters exist on a day, missing slots are populated as: `max_gex = 0`, `quality_ordinal = 0`, `signed_distance_sigma = 0`. NULL semantics would complicate the distance metric; explicit zeros are simpler and the meaning is clear ("no cluster here, treat as zero distance / zero magnitude / feature-tier").

**Cluster aggregates (7 features)**
- `n_clusters_total` — count of all confluences in the landscape.
- `n_pin`, `n_target`, `n_feature` — count by quality tier.
- `n_clusters_above_spot`, `n_clusters_below_spot` — geographic distribution.
- `top_cluster_fraction_of_total_max_gex` — `cluster_1_max_gex / sum(all max_gex)`. Proxy for "concentrated in one cluster" (high value, near 1) vs "spread across many" (lower value).

**Per-bucket dominance (4 features)**
- `dominance_0DTE`, `dominance_1_7`, `dominance_8_30`, `dominance_30plus` — raw % values summing to 100. From `landscape.bucket_summary` or recomputed from `per_bucket` dominance percentages.

**Neg zones (3 features)**
- `n_neg_zones` — count of negative zones in the landscape's neg_zones array.
- `nearest_neg_signed_distance_sigma` — `(nearest_neg.price - spot) / implied_move_1d`. `0` if no neg zones.
- `total_neg_max_gex` — sum of absolute neg gex across all neg zones, in $B.

**Vol regime (1 active feature)**
- `implied_move_1d` — raw points. The 1σ daily implied move computed from ATM IV. Captures vol regime independently of the distance scaling. For v0.5 this is our only active vol-context feature; vol surface slots below are NULL.

**Vol surface placeholders (5 features, all NULL/0 in v0.5)**
- `atm_iv_percentile`, `skew_percentile`, `smile_convexity`, `term_structure_slope`, `vol_risk_premium`.
- Populated NULL at extraction time. Similarity function detects NULL and skips them (see Similarity section).

**Total: 30 active + 5 deferred = 35 schema slots.**

### Similarity function

Weighted Euclidean distance over the feature vector, with:

1. **σ-normalization at extraction** for distance features (cluster signed_distance_sigma × 3, nearest_neg_signed_distance_sigma). These are already dimensionless when they hit the similarity function.
2. **Inverse-variance scaling per feature, computed across the labeled corpus**: each feature is z-scored using corpus mean/stddev, with a small epsilon floor on stddev to prevent division-by-zero on near-constant features.
3. **NULL-aware**: features whose value is NULL on either the query day or the candidate day are skipped in the distance sum. After summation, the distance is rescaled by `sqrt(n_total_features / n_active_features)` so distances stay comparable across queries with different active-feature counts. For v0.5 this matters only for the 5 vol-surface slots, which are NULL everywhere; effectively every query in v0.5 uses 30-of-35 dimensions.

```python
def similarity_distance(query_vec, candidate_vec, feature_stats):
    """Weighted Euclidean distance, NULL-aware."""
    sum_sq = 0.0
    n_active = 0
    for feature in FEATURE_NAMES:
        q = query_vec[feature]
        c = candidate_vec[feature]
        if q is None or c is None:
            continue
        std = max(feature_stats[feature]["std"], EPSILON)
        z_q = (q - feature_stats[feature]["mean"]) / std
        z_c = (c - feature_stats[feature]["mean"]) / std
        sum_sq += (z_q - z_c) ** 2
        n_active += 1
    if n_active == 0:
        return float("inf")
    rescale = (len(FEATURE_NAMES) / n_active) ** 0.5
    return (sum_sq ** 0.5) * rescale
```

The `feature_stats` table (corpus mean/std per feature) is recomputed at endpoint-call time from the full `bt_daily_features` table. Cheap given small corpus; can be cached later if it becomes a hot path.

### KNN

- **K = 5** for v0.5. Easily changed via endpoint param when corpus grows.
- **Sorted closest-first** (smallest similarity_distance).
- Anchor day is excluded from its own neighbor list.
- Only candidate days with at least one labeled signal in `bt_signals` are returned. Unlabeled days are in `bt_daily_features` (cron populates all) but aren't surfaced as analogues. This keeps the comparison grounded in days where Ryan has a personal read on what happened.

### Data model

New table:

```sql
CREATE TABLE bt_daily_features (
  trade_date          DATE        PRIMARY KEY,
  ticker              TEXT        NOT NULL DEFAULT 'SPX',
  feature_vector      JSONB       NOT NULL,
  feature_version     TEXT        NOT NULL,
  feature_config_hash TEXT        NOT NULL,
  computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX bt_daily_features_version_idx ON bt_daily_features (feature_version);
```

`feature_version` is a constant like `"v0.5.0"` bumped whenever the schema or extraction logic changes. `feature_config_hash` is an md5 of the extraction config (which features, how scaled, etc.) — distinguishes implementation changes within a version.

Rows are upserted; recomputing the feature vector for a day (e.g., after the underlying landscape changes via CR-010's high-accuracy path or a future correction) replaces the existing row.

### Architecture

**Shared module: `packages/shared/day_features.py`**

```python
FEATURE_VERSION = "v0.5.0"

FEATURE_NAMES = [
    # Regime / structure (6)
    "is_pin_day", "is_magnet_day", "is_bounded_day", "is_untethered_day",
    "is_amplification_day", "magnet_direction_signed",
    # Top 3 cluster slots (9)
    "cluster_1_max_gex", "cluster_1_quality_ordinal", "cluster_1_signed_distance_sigma",
    "cluster_2_max_gex", "cluster_2_quality_ordinal", "cluster_2_signed_distance_sigma",
    "cluster_3_max_gex", "cluster_3_quality_ordinal", "cluster_3_signed_distance_sigma",
    # Cluster aggregates (7)
    "n_clusters_total", "n_pin", "n_target", "n_feature",
    "n_clusters_above_spot", "n_clusters_below_spot",
    "top_cluster_fraction_of_total_max_gex",
    # Per-bucket dominance (4)
    "dominance_0DTE", "dominance_1_7", "dominance_8_30", "dominance_30plus",
    # Neg zones (3)
    "n_neg_zones", "nearest_neg_signed_distance_sigma", "total_neg_max_gex",
    # Vol regime (1)
    "implied_move_1d",
    # Vol surface placeholders (5, NULL in v0.5)
    "atm_iv_percentile", "skew_percentile", "smile_convexity",
    "term_structure_slope", "vol_risk_premium",
]

def extract_features(
    landscape_payload: dict,    # full /api/gex-landscape response shape
    spot: float,
    implied_move: float,
) -> dict:
    """Pure function. Returns dict keyed by FEATURE_NAMES."""

def compute_and_upsert_daily_features(
    conn, ticker: str, trade_date: date, *, version: str = FEATURE_VERSION
) -> dict:
    """Cron + backfill helper. Loads the stored landscape row, materializes
    the spot-dependent classifier chain (regime / per_bucket / confluences /
    neg_zones / walls), then runs extract_features and upserts into
    bt_daily_features."""
```

`extract_features` is pure (no DB calls). All DB interaction happens in `compute_and_upsert_daily_features`. Mirrors the pattern from CR-007's `compute_and_upsert_landscape`.

**Important interface note (from Step 0).** The stored `orats_gex_landscape.landscape` JSONB carries only the per-strike grid (601 rows × `{price, gex_0dte, gex_med, gex_near, gex_struct, gex_total}`). `regime`, `bucket_summary`, `confluences`, `neg_zones` are *not* stored — they're materialized at API serve time by `build_gex_landscape_response()` in `gex_landscape_api.py` via `classify_regime`, `classify_per_bucket`, `summarize_per_bucket`, `analyze_confluence`, `find_walls`, `find_proximate_negative_zones`. `compute_and_upsert_daily_features` runs the same classifier chain on the stored DataFrame to produce a payload-shaped dict, which it passes to `extract_features`. No new compute paths; reuse of existing.

**Cron addition: `apps/cron/job_orats_eod.py`**

After the `compute_and_upsert_landscape` block, add:

```python
from packages.shared.day_features import compute_and_upsert_daily_features

compute_and_upsert_daily_features(
    conn=conn, ticker=ticker, trade_date=store_trade_date,
    version="v0.5.0",
)
```

Same transaction as the landscape upsert.

**Backfill script: `scripts/backfill_daily_features.py`**

Walks distinct `(ticker, trade_date)` from `orats_gex_landscape`, calls `compute_and_upsert_daily_features` for each. Idempotent. CLI mirrors `backfill_gex_landscape.py`:

```
python scripts/backfill_daily_features.py                    # all dates
python scripts/backfill_daily_features.py --since 2026-01-01
python scripts/backfill_daily_features.py --date 2026-05-20
```

**Backend module: `apps/web/modules/Analogues/`**

Files:

- `service.py` — pure functions for similarity computation and KNN ranking. No Flask imports.
- `routes.py` — Flask endpoint wiring, follows the BacktestsV2 pattern.
- `__init__.py` — empty.

**Endpoint: `GET /api/analogues`**

Request params:

| Param             | Required | Default       | Notes                                          |
|-------------------|----------|---------------|------------------------------------------------|
| `date`            | yes      | —             | Anchor date, `YYYY-MM-DD`                      |
| `spot`            | yes      | —             | Spot price for σ-normalization                 |
| `implied_move`    | yes      | —             | 1σ daily implied move in points                |
| `k`               | no       | 5             | Number of analogues to return (capped at 20)   |
| `ticker`          | no       | `SPX`         | Future-proofing for multi-ticker corpus        |
| `feature_version` | no       | latest        | Pins to a specific feature schema version      |

Response shape:

```jsonc
{
  "ok": true,
  "anchor": {
    "trade_date": "2026-05-22",
    "feature_vector": {...},          // 35 keys per FEATURE_NAMES
    "implied_move": 40.5
  },
  "analogues": [
    {
      "trade_date": "2026-05-07",
      "similarity_score": 0.34,        // distance, smaller = more similar
      "feature_vector": {...},
      "regime": "pinned",
      "labeled_signals": [
        {
          "signal_id": "...",
          "label": 1,
          "label_note": "...",
          "outcome": "win",
          "realized_pts": 12.3
        }
      ],
      "outcomes": {
        "eod_return_pts": +8.5,
        "intraday_range_pts": 24,
        "mfe_above_open_pts": 12,
        "mfe_below_open_pts": -8,
        "session_start": "06:30",
        "session_end": "13:00"
      },
      "landscape_summary": {
        "regime": "pinned",
        "dominant_bucket": "1-7 DTE",
        "top_cluster": {
          "center_price": 7353,
          "quality": "pin",
          "max_gex": 712e9
        }
      }
    }
  ]
}
```

The endpoint:

1. Loads or computes the anchor day's feature vector. If `(date, feature_version)` is already in `bt_daily_features`, use it; otherwise extract from the landscape table on the fly (anchor day is often today, may not be cached yet by cron).
2. Loads all candidate feature vectors from `bt_daily_features` for the requested `feature_version`, joined to `bt_signals` to filter to labeled days only.
3. Computes corpus mean/std per feature (across the candidate set).
4. Computes `similarity_distance(anchor_vec, candidate_vec, feature_stats)` for each candidate.
5. Sorts ascending, takes top K (excluding anchor itself).
6. For each top-K, queries `bt_signals` for labeled signals + computes `outcomes` from `ironbeam_es_1m_bars` (EOD return, range, MFE).
7. Queries the landscape table for `landscape_summary` (regime + dominant_bucket + top_cluster).
8. Assembles and returns the response.

**Empty-state behavior.** When zero candidate days are labeled, the endpoint returns `{ok: true, anchor: {...}, analogues: []}` with HTTP 200. No crash, no error. This is the day-0 state of the labeled corpus.

**Frontend: extensions to `react_price_preview/`**

The analogues panel lives in the main dashboard alongside the price chart and landscape panel — *not* a separate Vite app. This is part of the daily-monitoring UI, not a separate "study" surface.

New components:

- `react_price_preview/src/components/AnaloguesPanel.jsx`
  - Right-docked, like `GexLandscapePanel`. Independent from it positionally.
  - Header: title + K selector (5 / 10 / 20) + close button.
  - List body: K rows. Each row: date, similarity score, regime badge, label thumb (+1/-1/0 icon), EOD return (+/- pts), intraday range.
  - Clicking a row → opens `AnalogueDetailModal` for that day.
  - Loading state with skeleton rows.
  - Empty state when no labeled days are similar enough (e.g., distance > some threshold; surface as "no close analogues yet").

- `react_price_preview/src/components/AnalogueDetailModal.jsx`
  - Non-destructive modal. Centered overlay. Closes on backdrop click, Escape, or X button.
  - Body: that day's mini price chart (using `lightweight-charts` for consistency); mini landscape (reuse `GexLandscapePanel`'s renderer at small scale, or static image); outcomes block (EOD return, range, MFE, session stats); label block (thumb + label_note).
  - Footer: "Open this date" link → navigates main app to that historical date (escape hatch; explicit action only).

Wiring:

- `react_price_preview/src/components/PriceChart.jsx`: append an `ANALOGUES` pill to the `pills` array next to LANDSCAPE.
- `react_price_preview/src/App.jsx`: state for `analoguesOpen`, debounce-fetch on date/spot/IV change when panel is open, pass data to `AnaloguesPanel`.

### Acceptance criteria

1. **Feature extraction is deterministic.** `extract_features(landscape, spot, implied_move)` returns the same 35-key dict for the same inputs across runs.
2. **σ-normalization is correct.** Unit test: a cluster at +50pt with implied_move=30 produces `signed_distance_sigma = +1.667`; a cluster at -40pt with implied_move=80 produces `signed_distance_sigma = -0.5`.
3. **All four labeled days from CR-011 have populated rows in `bt_daily_features`** after backfill: 5/6, 5/7, 5/18, 5/20. The vectors are inspectable via SQL.
4. **Endpoint returns analogues** for `GET /api/analogues?date=2026-05-22&spot=7400&implied_move=40&k=5`, sorted closest-first by `similarity_score`. With the v0.5 labeled corpus empty, the endpoint returns `analogues: []` gracefully; AC #4 becomes operative once labels are added.
5. **The anchor day is excluded** from its own analogue list — querying for 5/7 with 5/7 in the corpus returns the 5 next-closest, not 5/7 itself.
6. **Unlabeled days are excluded** from results. Querying for any anchor returns only days that have at least one labeled signal in `bt_signals`.
7. **Frontend pill toggle reveals the right-docked panel** showing K rows (or empty-state message).
8. **Clicking a row opens the modal** with mini chart, mini landscape, outcomes, and label info populated.
9. **Closing the modal returns to the main view** with today's data unchanged (no navigation occurred).
10. **180+ backend tests pass.** Adds tests for `extract_features` per-feature correctness, σ-normalization, similarity_distance NULL handling, and KNN sort order.

### Verification plan

After implementation:

1. **Unit tests** for `extract_features` against the four CR-011 calibration days. Snapshot the expected feature vectors in `packages/shared/tests/fixtures/day_features_calibration.json`. Pin once and lock — these become the regression baseline.
2. **σ-normalization sanity** as described in AC #2.
3. **Backfill production DB** via `scripts/backfill_daily_features.py`. Verify `SELECT COUNT(*) FROM bt_daily_features` matches the row count in `orats_gex_landscape`.
4. **Endpoint smoke test** for each of 5/6, 5/7, 5/18, 5/20 as anchor. With the corpus empty, all queries return `[]`; document this state in the wrap-up. Once labels exist, manually inspect the K=5 analogues for each — do they make trading sense?
5. **Frontend smoke** — toggle ANALOGUES pill on, see the empty-state or list, click a row (if any), see the modal, close it, today's chart is unchanged.
6. **No regression on existing functionality** — verify CR-011's confluence quality classification still produces correct tags, BacktestsV2 still serves its endpoints, GEX landscape panel still renders.

### Out of scope

- **Vol surface implementation.** Slots are NULL/0 in v0.5; populated in a follow-up CR after Ryan reconciles the skew framing.
- **Calendar features** (day-of-week, OPEX proximity, FOMC week). Follow-up CR.
- **Trade construction templates.** v1 of the framework, not v0.5.
- **Options-priced execution.** v3 of the framework.
- **Updates to BacktestsV2.** Lives alongside as a separate scan engine for its specific use case.
- **Multi-ticker support.** Schema includes `ticker` for future-proofing but the corpus is SPX-only.
- **Distance threshold for "no close analogues."** Show all K regardless of distance; the trader can judge whether the closest is actually close. A threshold heuristic can be added in a follow-up.
- **K > 20.** Endpoint validates and caps.
- **Feature weight tuning beyond inverse-variance.** All features start with equal weight after z-scoring. If the KNN behaves badly, weight tuning becomes a follow-up CR — likely as a separate config concern with a versioned weights file.
- **`is_broken_magnet_day` indicator.** v0.5 collapses `broken-magnet` to all-zeros across the five regime indicators. Adding the sixth indicator is deferred until a calibration day with `broken-magnet` regime exists.

### Data integrity dependencies

The CR depends on the following data being available and correct:

- **`orats_gex_landscape`** — populated for the date range being analyzed. CR-007 onward.
- **`bt_signals`** — has at least some labeled rows; the KNN's value scales with how many labeled days exist. If empty, the endpoint should return a graceful empty-state response, not crash.
- **`orats_monies_minute`** — needed for `implied_move_1d` per day. Use ATM IV at last snapshot of the session for the smallest `dte > 0`, then `compute_implied_move(spot, iv, dte=1.0)`. Step 0 confirmed full coverage across all current landscape dates.
- **`ironbeam_es_1m_bars`** — needed for the outcome metrics (EOD return, intraday range, MFE). Standard data, should be present.

### Step 0 pre-impl notes

Pre-impl validation against production DB (run on the kickoff branch, pre-spec-commit) surfaced the following:

1. **`bt_signals` is empty of labels** — 10 unlabeled rows, 0 with `label IS NOT NULL`. CR ships with the documented empty-state behavior; KNN becomes useful once Ryan labels signals through the existing thumbs-up/down UI.
2. **`orats_monies_minute` coverage is complete** for the current 6 landscape dates (21k+ rows/day, 6:30→13:00 PT). No fallback needed.
3. **σ-normalization math validates against 5/7.** spot=7362.825, top cluster center=7353.389, ATM IV(dte=1) @ 13:00 PT=0.0939 → implied_move=43.55pt → signed_distance_sigma = -0.2167. Direction (below spot) and magnitude (~0.2σ) both sane. Top cluster quality = `pin`, regime = `magnetic-pin` — matches CR-011 calibration.
4. **Landscape JSONB schema is uniform** across all 6 current rows (post-CR-010). No mixed-schema rows to skip.
5. **`orats_gex_landscape` carries only the per-strike grid**, not the derived `regime/confluences/neg_zones/bucket_summary`. Implementation runs the same classifier chain (`classify_regime`, `classify_per_bucket`, `summarize_per_bucket`, `analyze_confluence`, `find_walls`, `find_proximate_negative_zones`) on the stored DataFrame to materialize a payload-shaped dict before extraction. Documented above in the Architecture section.
