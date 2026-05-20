# CR-007 — GEX landscape data pipeline

Phase 1 of the GEX landscape build is split into two CRs to keep each within typical CR scope:

- **CR-007 (this spec)** — data pipeline: shared-module refactor + new table + cron block + backfill. After CR-007 lands, the landscape is computed and stored nightly; data is queryable via SQL but not yet on the chart.
- **CR-008 (future)** — delivery layer: endpoint + frontend panel. Drafted after CR-007 lands, against the data the cron is now persisting.

Architecture reference: `[[gex-landscape]]` component spec, `[[2026-05-20 - GEX Landscape Spot-Agnostic Storage]]` ADR.

## Problem

Phase 0 of the GEX landscape work (script at `scripts/explore_gex_landscape.py`) validated the math and visualization as a CLI prototype: Gaussian-smoothed continuous GEX field, regime classification, per-DTE-bucket dominance, quality-graded confluence detection, IV-aware target classification, conditional-arrival framing for negative zones. The prototype produces PNGs and structured stdout from `(trade_date, --spot, --implied-move)` inputs.

For production use, the math needs to (a) live in a shared module callable from the cron and a future endpoint, not just from the script, and (b) be persisted nightly into a cached table so the downstream endpoint can read the spot-agnostic field cheaply. Per `[[2026-05-20 - GEX Landscape Spot-Agnostic Storage]]`, spot-dependent interpretations (regime, dominance, confluences, neg zones) are deferred to request time and not stored.

This CR moves the math + adds persistence. It does NOT add the endpoint or any frontend (CR-008).

## Proposed Solution

Four pieces, in order:

**1. Refactor `scripts/explore_gex_landscape.py` → `packages/shared/gex_landscape.py`.** Extract the pure analytical functions into a new shared module. The script keeps argparse, the DB query (`fetch_strike_data`), matplotlib plotting, and `run_one_date` orchestration. All analytical math imports from the shared module.

Functions moved to the shared module:
- `compute_landscape`, `find_walls`, `score_containment_zones`
- `find_peaks_per_bucket`, `find_confluence_clusters`, `score_confluence`, `classify_confluence_quality`, `analyze_confluence`, `_compute_fwhm`
- `classify_regime`, `_annotate_distance_class`
- `classify_per_bucket`, `_bucket_landscape_view`, `summarize_per_bucket`
- `find_intraday_subtarget`, `find_proximate_negative_zones`
- `classify_distance`, `compute_implied_move`
- Constants: `DTE_BUCKETS`, `DISTANCE_THRESHOLDS`

Sanity check: post-refactor, running `python scripts/explore_gex_landscape.py --date 2026-05-20 --spot 7392 --implied-move 40` produces byte-identical stdout to pre-refactor.

**2. Create `orats_gex_landscape` table.** Schema per the component spec:

```sql
CREATE TABLE orats_gex_landscape (
  ticker          TEXT        NOT NULL,
  trade_date      DATE        NOT NULL,
  landscape       JSONB       NOT NULL,
  walls           JSONB       NOT NULL,
  peaks_by_bucket JSONB       NOT NULL,
  spread_coef     NUMERIC     NOT NULL,
  range_pts       NUMERIC     NOT NULL,
  step_pts        NUMERIC     NOT NULL,
  table_spot      NUMERIC,
  version         TEXT        NOT NULL,
  computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ticker, trade_date)
);

CREATE INDEX orats_gex_landscape_date_idx ON orats_gex_landscape (trade_date DESC);
```

Add a new `compute_and_upsert_landscape(conn, ticker, trade_date, *, spread_coef, range_pts, step_pts, version)` helper in `packages/shared/gex_landscape.py`. The helper re-queries the just-upserted `orats_oi_gamma` rows, reads `stock_price` from the first row (used only as `table_spot` metadata + as the landscape grid center), runs `compute_landscape` + `find_walls` + `find_peaks_per_bucket`, serializes to JSONB-compatible dicts, and UPSERTs.

**3. Add cron block to `apps/cron/job_orats_eod.py`.** After the existing `orats_oi_gamma` upsert completes, call `compute_and_upsert_landscape` on the same DB connection — single transaction with the `orats_oi_gamma` upsert so both halves succeed or roll back together.

Defaults: `spread_coef=8.0`, `range_pts=200.0`, `step_pts=1.0`. These match the Phase 0 prototype's defaults and the parameters used to validate against 5/6, 5/7, 5/18, 5/20.

**4. Backfill script `scripts/backfill_gex_landscape.py`.** Walks distinct `(ticker, trade_date)` from `orats_oi_gamma`, calls `compute_and_upsert_landscape` for each. Idempotent (UPSERT). CLI:

```
python scripts/backfill_gex_landscape.py                    # all dates
python scripts/backfill_gex_landscape.py --since 2026-01-01
python scripts/backfill_gex_landscape.py --date 2026-05-20  # single
```

## Affected Files

- `scripts/explore_gex_landscape.py` — analytical functions removed, replaced with imports from shared module. Script keeps CLI, DB query, plotting, orchestration.
- `packages/shared/gex_landscape.py` — **new**. Houses the analytical functions and the `compute_and_upsert_landscape` helper.
- `packages/shared/__init__.py` — may need an export depending on existing conventions; check `packages/shared/options_cache/__init__.py` for the pattern.
- `apps/cron/job_orats_eod.py` — add one `compute_and_upsert_landscape` call after the `orats_oi_gamma` upsert block. No changes to the existing fetch/upsert logic.
- `scripts/backfill_gex_landscape.py` — **new**.
- DB migration — wherever migrations live in this repo. Check `infra/` and the existing migration pattern for `orats_oi_gamma`, `orats_monies_minute`, `bt_signals` tables. **Read first; do not assume a migration tool exists.**
- `packages/shared/tests/test_gex_landscape.py` (or wherever this codebase's Python tests live for `packages/shared/` — see CR-004's lesson: `packages/shared/options_cache/tests/` is the pattern, NOT a top-level `tests/`). Unit tests for the refactored shared functions (snapshot tests on known inputs) and for `compute_and_upsert_landscape` (mocked DB).

## Acceptance Criteria

**Refactor (criterion 1 — byte-identical):**
- Running `python scripts/explore_gex_landscape.py --date 2026-05-20 --spot 7392 --implied-move 40` produces stdout that diffs to zero against pre-refactor stdout (captured before changes begin).
- Same script command produces identical PNGs (or at minimum, the regime, per-DTE breakdown, confluence list with quality tags, intraday actionable section, and high-vol zones section all match line-for-line).

**Storage (criterion 2 — table populated):**
- `orats_gex_landscape` table exists with the schema above.
- After the cron runs once, one row is present for `(SPX, today)`.
- The `landscape` JSONB array has ~400 rows (range_pts=200, step_pts=1, so ~401 points).
- The `walls` JSONB array contains both positive and negative entries with `sign` set correctly.
- The `peaks_by_bucket` JSONB object has keys `"0DTE"`, `"1-7 DTE"`, `"8-30 DTE"`, `"30+ DTE"`, each holding a list of peak dicts with `price`, `gex`, `prominence`, `fwhm`.
- `table_spot` matches `stock_price` from the corresponding `orats_oi_gamma` row.
- `version` matches the cron's `VERSION` constant.

**Cron (criterion 3 — clean integration):**
- A test run of `job_orats_eod.py` on a date that already has `orats_oi_gamma` data successfully writes (or re-UPSERTs) the corresponding `orats_gex_landscape` row.
- If the new block raises, the `orats_oi_gamma` upsert rolls back (same transaction).

**Backfill (criterion 4 — historical coverage):**
- `python scripts/backfill_gex_landscape.py --date 2026-05-20` produces a single row matching what the cron would have produced.
- Running for the four Phase 0 validation dates (`2026-05-06`, `2026-05-07`, `2026-05-18`, `2026-05-20`) produces four rows.
- Spot-check: reading any of those four rows back and re-classifying with the originally-tested spot (`--spot 7400` for 5/7, `--spot 7326` for 5/6, etc.) produces the same regime / per_bucket output the Phase 0 script printed for that day.

## Verification

The repo has no automated test framework outside `packages/shared/`'s unittest setup (per CR-004's lesson). Verification is layered:

**Automated (Claude Code runs):**
- Existing test suite passes (whatever the current baseline count is — verify by running `python -m unittest discover packages/shared/` or whatever the canonical invocation is in this repo).
- New unit tests for `gex_landscape.py` functions pass. At minimum: snapshot test on `compute_landscape` for a small synthetic input, snapshot test on `find_walls` for a known landscape, snapshot test on `classify_regime` for the 5/20 case (`spot=7392, implied_move=40` → regime `magnet-above` with `drift_target=7520`).
- New tests for `compute_and_upsert_landscape` with a mocked DB connection. Verify the helper executes the expected UPSERT with the expected payload shape.

**Manual (Claude Code writes this checklist into the PR description):**
- Capture pre-refactor script stdout: `python scripts/explore_gex_landscape.py --date 2026-05-20 --spot 7392 --implied-move 40 > /tmp/pre.txt` (do this BEFORE any edits — it's the regression baseline).
- After implementation: `python scripts/explore_gex_landscape.py --date 2026-05-20 --spot 7392 --implied-move 40 > /tmp/post.txt`. Diff: `diff /tmp/pre.txt /tmp/post.txt` should produce no output.
- Confirm `orats_gex_landscape` table exists: `\d orats_gex_landscape` in psql shows the schema above.
- Run the cron locally (or trigger on Render) for today's date. Confirm a row appears: `SELECT ticker, trade_date, jsonb_array_length(landscape) AS n_landscape, jsonb_array_length(walls) AS n_walls, version, computed_at FROM orats_gex_landscape WHERE trade_date = CURRENT_DATE;`
- Run backfill against the four Phase 0 dates. Confirm four rows present.

## Out of Scope

- Endpoint (`/api/gex-landscape`) — CR-008.
- Frontend panel, pill, or any React changes — CR-008.
- Intraday landscape recomputation — Phase 1 is EOD cron only.
- Live IV pulling from `orats_monies_minute` for the endpoint — endpoint design, CR-008.
- Removing the script's plotting code — script stays as a CLI tool indefinitely.
- Tuning the `score_confluence` or `classify_confluence_quality` thresholds — current values are defaults from Phase 0; tuning is a future, separable concern.
- Confluence line rendering on the price chart, neg-zone bands, intraday subtarget annotations — Phase 1.5 / CR-009+.

## Handoff Prompt for Claude Code

> Read `specs/CR-007-gex-landscape-data-pipeline.md` and implement it on the current branch (`feat/CR-007-gex-landscape-data-pipeline`). Follow the spec closely.
>
> **Before writing any code, do pre-implementation review (the contradiction-stop rule):**
> 1. Grep new symbol names (`compute_and_upsert_landscape`, `gex_landscape` module path) against the existing codebase to confirm they're net-new.
> 2. Read `packages/shared/options_cache/tests/` for the test convention (unittest, where tests live). Do NOT assume a top-level `tests/` directory exists.
> 3. Read the existing migration pattern for `orats_oi_gamma` (and `orats_monies_minute` if helpful) to determine how DB schema changes land in this repo. Read `infra/` and `render.yaml`. Do NOT invent a migration tool.
> 4. Read `apps/cron/job_orats_eod.py` end-to-end to find the right insertion point for the new compute-and-upsert call, and to confirm the DB connection pattern (`db.get_conn`, psycopg) — match it exactly.
> 5. Read `scripts/explore_gex_landscape.py` end-to-end so the refactor preserves every entry point the script uses.
> 6. Capture pre-refactor regression baseline: `python scripts/explore_gex_landscape.py --date 2026-05-20 --spot 7392 --implied-move 40 > /tmp/pre.txt` BEFORE making any edits.
>
> If pre-implementation review surfaces contradictions between the spec and reality, **stop and amend the spec in its own commit before proceeding**. Each amendment is a separate commit so the audit trail shows what changed and why.
>
> Run all verification commands listed in the Verification section. When finished, write the manual verification checklist into the PR description.

