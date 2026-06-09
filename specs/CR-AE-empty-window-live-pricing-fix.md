CR-AE — Empty-Window Recording Fix for Live Option-Leg Pricing

## Problem

`fetch_option_bars` records a `row_count=0` window for every OPRA it fetches —
including when ORATS returns an empty response. `find_gaps` treats any window as
"fully cached" regardless of row_count. On the live pricing path, an empty response
almost always means ORATS ingestion lag (transient), not genuine absence — but the
first empty response poisons the window permanently: once recorded, `find_gaps` will
never refetch that minute, even after ORATS ingests the real data.

This behavior is deliberate and correct for the historical backtest path (genuinely
illiquid OPRAs). The bug is that the same primitive (`fetch_option_bars`) serves both
live and historical callers with no way to distinguish them. The fix makes empty-window
recording a per-caller switch keyed on access pattern (live vs. historical).

The live condor panel's "⚠ 4 legs missing" symptom is one surface of this bug, but
the primitive has four callers in pricing.py — all four must be classified and wired
correctly.

## Proposed Solution

Three coordinated changes:

**Step 1 — `record_empty_windows` flag on `fetch_option_bars`.**
Add `record_empty_windows: bool = True` to `fetch_option_bars` in `fetcher.py`.
Default `True` preserves backtest behavior exactly. When `False` and a gap fetch
returns zero bars for the requested OPRA, skip the `repo.record_fetched_window(...)` write
for that OPRA entirely — the minute stays a gap and is re-fetched next call.
Non-empty path is untouched (real-data windows always recorded). The counterpart-OPRA
window logic is also untouched for non-empty responses. Do NOT change `find_gaps`.

**Step 2 — Wire the live callers (per Step-0 classification).**
Set `record_empty_windows=False, source="live_poll"` on callers confirmed live.
`live_poll` is the existing `FetchSource` enum value; no new enum value needed.
No change to the historical backtest path or the `orchestrator.py` caller.

**Step 3 — One-time poison cleanup script (manual, guarded).**
`scripts/cr_ae_cleanup_empty_windows.py`:
- Dry-run by default: print `SELECT COUNT(*)` and source breakdown of `row_count=0` rows,
  then exit. Require `--execute` flag to run the DELETE.
- DELETE: `DELETE FROM orats_options_fetched_windows WHERE row_count = 0`
- Wrap DELETE in a transaction; print rows-deleted; log a one-line summary.
- Runs under `DATABASE_URL` (app role can DELETE; not `BACKFILL_DATABASE_URL`).
- No `bt_backfill_runs` row — this is a maintenance op, not a backfill-protocol run.
- Ryan runs manually after Steps 1–2 merge.

**Step 4 — Tests.**
- Unit: `fetch_option_bars` with `record_empty_windows=False` + empty mock response
  writes NO window; with `True` (default) it writes the `row_count=0` window
  (regression-guards backtest behavior).
- Unit: live condor caller passes the flag through; transient-empty-then-populated
  sequence results in a successful second fetch (proves self-heal).
- Smoke (pre-merge): condor panel on a recent live minute populates legs; known
  illiquid historical OPRA still records empty window (backtest unaffected).

## Affected Files

- `packages/shared/options_cache/fetcher.py` — add `record_empty_windows: bool = True`
  to `fetch_option_bars`; skip the window write on empty when `False`
- `packages/shared/options_cache/pricing.py` — wire live callers per Step-0 decision
  (`_fetch_minute_quotes` hard-coded; proposal/edge callers per (i) or (ii))
- `scripts/cr_ae_cleanup_empty_windows.py` — new standalone maintenance script
- `packages/shared/options_cache/tests/test_fetcher.py` — four new unit test cases

## Acceptance Criteria

1. `fetch_option_bars(..., record_empty_windows=False)` with a mocked empty ORATS
   response writes zero rows to `orats_options_fetched_windows`.
2. `fetch_option_bars(..., record_empty_windows=True)` (default) with a mocked empty
   response still writes the `row_count=0` window (backtest behavior unchanged).
3. The condor `_fetch_minute_quotes` passes `record_empty_windows=False, source="live_poll"`.
4. Proposal/edge callers are wired per the Step-0 (i)/(ii) decision.
5. The cleanup script dry-runs without `--execute`, prints a count + source breakdown,
   and exits without deleting. With `--execute`, it deletes and prints rows-deleted.
6. `orchestrator.py`'s `fetch_option_bars` call is untouched (historical path unaffected).

## Verification

- `pytest packages/shared/options_cache/tests/test_fetcher.py` — all existing + new tests pass.
- Smoke: condor panel on a current-session minute populates legs after cleanup + fix.
- Spot-check: one live-path window in `orats_options_fetched_windows` post-fix shows
  `source = 'live_poll'` (confirms the tag is reaching the table).

## Out of Scope

- Empty-window TTL (rejected in ADR — more machinery than the flag).
- `find_gaps` consulting `row_count` (rejected — regresses backtest).
- Coalescing / window maintenance (`coalesce_windows` is a separate concern).
- Schema change to add an `active` flag to `orats_options_fetched_windows`.
- Any change to CR-038, CR-AC, or anything env/contract-related.
- Visible layout/UI changes (cache-correctness fix only).

## Step-0 Caller Classification Gate

Before implementation, classify all `fetch_option_bars` callers and commit findings.

Confirmed callers as of main `642ca21`:

**In `packages/shared/options_cache/pricing.py`:**
1. `_fetch_minute_quotes` (line 106) — condor pricing — **LIVE** (confirmed)
2. `price_proposal_legs` (line 300) — classify: LIVE or HISTORICAL?
3. `build_real_strike_band` (line 398) — classify: LIVE or HISTORICAL?
4. `fetch_horizon_delta` (line 479) — classify: LIVE or HISTORICAL?

**In `packages/shared/options_cache/orchestrator.py` (line 205):**
5. `fetch_for_rows` orchestrator — **HISTORICAL** (backtest backfill only; never
   invoked for live/current-session minutes)

**No other callers outside `packages/shared/options_cache/` in the whole repo** (grep
confirms `apps/web` routes call `price_proposal_legs` / `build_real_strike_band` /
`fetch_horizon_delta` which wrap `fetch_option_bars` — they do not call
`fetch_option_bars` directly).

Step-0 design decision (lock before implementing):
- **(i) Static per-caller:** only condor path gets `record_empty_windows=False`; the
  three proposal/edge paths stay default.
- **(ii) Date-aware:** proposal/edge callers also pass `record_empty_windows=False`
  when `trade_date == today`, threading the live flag through.

**Poison-row provenance query (Step 0.3):**
```sql
SELECT source, COUNT(*), MIN(window_start_pt), MAX(window_start_pt), MIN(fetched_at), MAX(fetched_at)
FROM orats_options_fetched_windows
WHERE row_count = 0
GROUP BY source;
```

---

## Step-0 Diagnosis Findings (committed 2026-06-07)

### Caller classification

Grepped `fetch_option_bars(` across the entire repo. All confirmed callers as of
`642ca21`:

| Caller | File | Line | Classification |
|--------|------|------|----------------|
| `_fetch_minute_quotes` | `pricing.py` | 106 | **LIVE** — condor panel, polls every 10s during RTH, prices the entry minute of the current session. Passes no source → tagged `historical_backfill` in DB. |
| `price_proposal_legs` | `pricing.py` | 300 | **CAN BE LIVE** — see below |
| `build_real_strike_band` | `pricing.py` | 398 | **CAN BE LIVE** — see below |
| `fetch_horizon_delta` | `pricing.py` | 479 | **CAN BE LIVE** — see below |
| `fetch_for_rows` orchestrator | `orchestrator.py` | 205 | **HISTORICAL** — backtest backfill only; always uses historical dates |

No callers exist outside `packages/shared/options_cache/` in `apps/web` or elsewhere.
The web routes call `price_proposal_legs`, `build_real_strike_band`, and
`fetch_horizon_delta` as wrappers; none call `fetch_option_bars` directly.

### Proposal/edge caller trace (the three "CAN BE LIVE" callers)

All three are called exclusively from `apps/web/modules/Proposals/routes.py`
(`POST /api/proposals/pl-data`), at lines 382, 459, and 559/564 respectively.
They share a single `entry_pt_naive` computed as:

```python
entry_time = build_entry_time(trade_date)  # 07:00 PT (10:00 ET) on trade_date
entry_pt_naive = entry_time.astimezone(_PT_TZ).replace(tzinfo=None)
```

The route accepts any `trade_date` in the POST body — including today. The only
gate is `_fetch_anchor_data`, which requires rows in `bt_daily_features_active`
and `bt_daily_outcomes_active` for `trade_date`. These tables are populated by
the morning daily-setup scripts, making today valid during RTH.

**Conclusion:** if `trade_date == today` in a proposals request (which is the normal
use case for Today's Edge surface), all three functions call `fetch_option_bars`
with `entry_pt_naive = today at 07:00 PT`. That minute can race ORATS ingestion
lag if the request is made early in the session (within the first few minutes
after open). The window-poisoning mechanism is identical to the condor case, just
triggered at a fixed minute rather than the rolling current bar.

The Today's Setup route (`GET /api/setup/proposals`) does NOT call any of the three
functions — it uses a separate service. The liveness risk is exclusively through
`POST /api/proposals/pl-data`.

### Step-0 design decision

Based on the trace above: **the three proposal/edge callers CAN price the current
session.** This triggers the (ii) recommendation.

**Recommendation: option (ii) DATE-AWARE.**

Implementation shape:
- `price_proposal_legs` already receives `trade_date: date`. Compare to
  `date.today()` in PT at fetch time; pass `record_empty_windows=False,
  source="live_poll"` to `fetch_option_bars` when live.
- `build_real_strike_band` receives `entry_pt: datetime` (naive PT). Infer liveness
  from `entry_pt.date() == date.today()` in PT.
- `fetch_horizon_delta` receives `trade_date: date`. Same check as
  `price_proposal_legs`.

All three add a single internal live-check; no signature changes needed (the check
can be done inside the function using `date.today()`).

Alternatively: the Proposals route passes an explicit `live: bool` kwarg to the three
functions — cleaner for testability but requires signature changes.

Both sub-options implement (ii). Decision between them deferred to Ryan's sign-off.

### Provenance query result (Step 0.3)

Queried `orats_options_fetched_windows` on 2026-06-07:

```
total rows: 17,138  |  min row_count: 1  |  max row_count: 372
zero-count rows: 0
```

**No `row_count=0` rows exist in the table at this time.** The provenance query
returned no rows. This means:
- The live path has not yet poisoned the table (likely the condor panel was not
  exercised against a minute where ORATS returned empty in this DB).
- The cleanup script (Step 3) will be a no-op against current data (dry-run will
  report 0 rows to delete).
- Steps 1 and 2 are still required to prevent future poisoning once the live path
  races ORATS ingestion lag.

This does NOT change the implementation plan. The cleanup script is written anyway
so Ryan can run it safely after future sessions.

### Contradiction-stop gate

**STOP — awaiting Ryan's sign-off on the (i)/(ii) decision before any implementation
code is written.**

Summary of evidence:
- Condor path: definitively LIVE.
- Proposal/edge three callers: CAN BE LIVE (today is a valid trade_date in the
  proposals route).
- Orchestrator: definitively HISTORICAL.
- Zero existing poison rows (cleanup is a no-op today; fix prevents future poison).
- Recommendation: **(ii) DATE-AWARE** — all three proposal/edge functions check
  `trade_date == date.today()` and pass the live flags when true.
  Sub-option: pass via internal check (no signature change) or via explicit `live: bool`
  kwarg (cleaner for tests but needs signature change at 3 sites + Proposals route).

Awaiting decision before Step 1.
