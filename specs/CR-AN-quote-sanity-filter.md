# CR-AN — Quote-sanity filter; clean-quote walk-forward reference; ORATS 404 classification

> Authority: vault session note `Dash/sessions/2026-09-05 - CR-AN — Quote-Sanity Filter in the Edge Harness.md`
> Branch: `feat/CR-AN-quote-sanity-filter` (off `origin/main` 1e3bd76, the CR-AM merge)
> Scope: minute-level quote validity in the edge harness's quote → spread-price path + tests; one re-run persisted as `cr_id='CR-AN'`; ORATS 404 classification (no fix).
> Mode: unattended; halts at the first STOP gate that misses.

## Problem

`[Certain]` CR-AM found that roughly a quarter of trades in both structures carry a spread price outside `[0, width]` at some minute, mostly at the first-minute baseline. A 10-point vertical cannot be worth less than 0 or more than 10; those minutes are bad quotes (crossed or stale legs, one-sided books at the open). The harness accepts them, so:

- every "beat vs baseline" in June and in CR-AM compares a gated fill to a baseline that may be a bad quote;
- the `bt_edge_backtest_results` rows persisted as `cr_id='CR-AM'` are contaminated the same way;
- CR-AM's clean-subset sensitivity table (in the spec) is the only uncontaminated read, and it was produced by *excluding trades*, which is the wrong fix — a bad first minute shouldn't erase a trade that had good quotes a minute later.

Two of CR-AM's four apparent reversals (credit far-band, debit gap-touch) were this contamination.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Where the filter lives | In the harness's quote → spread-price step, so every caller inherits it. Step 0 locates the function (in `scripts/cr_ah_step4_analysis.py` and/or `packages/shared/backtest/`). If the logic is duplicated across CR-AH and CR-AI scripts, CR-AN fixes the shared path and lists the duplicates as a follow-on; it does not fix two copies. |
| 2 | Minute-level rule | A minute is **valid** for a spread iff, for both legs, `bid ≥ 0`, `ask > 0`, `bid ≤ ask`, and the resulting spread price (mid-based, as the harness computes it today) is in `[0, width]` inclusive, with `width` from the structure. Invalid minutes are skipped, not clamped. |
| 3 | Baseline | First **valid** minute in the entry window, not the first minute. |
| 4 | Gated fill | Threshold crawl considers only valid minutes. |
| 5 | Settlement | Settlement price from the settlement window's valid minutes only; if none are valid, the trade is `unsettled` (existing status), not excluded. |
| 6 | Trade-level exclusion | Only if the entry window has **zero** valid minutes. Count and list these; expect few. |
| 7 | Observability | The harness records per trade: `n_minutes_total`, `n_minutes_valid`, `baseline_minute_offset` (minutes from window open to the first valid minute), and `had_invalid_quote` (bool). The run's smoke dict reports the distribution. |
| 8 | Re-persist | Re-run CR-AM's exact configuration with the filter (`--universe-end 2026-06-05 --split-date 2026-06-05 --structural-prob-mode walk-forward`) and persist with `cr_id='CR-AN'`. CR-AM and CR-AH rows remain as labeled comparison columns. Readers should prefer the latest `cr_id`; note for CR-Y. |
| 9 | Restatement | Same decision-7 rule as CR-AM (holds / weakens / reverses vs June, ≥ 50% magnitude) applied to CR-AN vs **CR-AM as-run** and vs **CR-AM clean-subset**. The clean-subset comparison is the one that should be near-identical; a large gap means decision 2 is doing something the trade-exclusion didn't, and the spec must explain it. |
| 10 | ORATS 404 diagnostic | Step 0b: pick one of CR-AM's 8 entry-day 404 dates and one fetched date; call the ORATS endpoint directly for each (same params the fetcher uses) and record status, and for the 404 also try the same date with a nearby strike and with `dte` widened. Classify: (i) date not served at all → retention/rolloff; (ii) that strike/expiry not served → strike-selection issue; (iii) transient. **No fix in this CR** — classification only, feeds [[debit-edge-needs-powered-holdout]] and CR-AG's reprioritisation. |
| 11 | Not in scope | Changing spread pricing from mid to bid/ask-aware fills; the entry-convention mismatch between CR-AH and CR-AI (open question); CR-AI re-run (only if the shared path is what CR-AI uses **and** Step 0 shows its quotes are contaminated at a similar rate — then list as follow-on, don't run). |

## Expected values (gates)

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — `main` contains the CR-AM merge | yes | STOP |
| G0.2 — filter location identified; single shared code path or explicit list of duplicates | yes | STOP if the spread-price computation can't be located |
| G0.3 — pre-filter contamination count reproduced from CR-AM's log/spec (≈ 25% trades flagged per structure) | within ±5 pp | note |
| G1 — unit tests for decision 2 (valid / crossed / negative / over-width / one-sided) | pass; 420+ & 105 suites still pass | STOP |
| G2 — post-filter: trades with any accepted spread price ∉ `[0, width]` | **0** | STOP |
| G3 — trades excluded under decision 6 | ≤ 5% of clean sample per structure | STOP if > 10% |
| G4 — selection / clean counts | 50/50/50; debit 110 / credit 106 before decision-6 exclusion | STOP |
| G5 — baseline_minute_offset distribution | median 0, p95 ≤ 5 min | note; if p95 > 15 the open is systematically bad and decision 3 needs a window cap — flag, continue |
| G6 — restatement vs CR-AM clean-subset | all cells holds | any reverses → note with explanation; not a halt |
| G7 — no holdout date, no post-2026-06-05 P&L in logs | yes | STOP and redact |

