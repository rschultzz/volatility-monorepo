# CR-DO — Recompute derived rows after CR-BZ's fragment repair (train full, holdout blind)

Branch `feat/CR-DO-crbz-derived-recompute` (worktree `../volatility-monorepo-cr-do`) off `origin/main` `a2c2216` (CR-DL merge).
Kickoff: chat 2026-10-29 (Ryan). Data CR (`data_safety_class: write_backfill`).
Source: vault open-question `crbz-fragment-repair-derived-recompute` (CR-DL F2).

## Problem

CR-BZ (run `88139854-3605-4d94-ab03-7bc34af5da5c`, 2026-10-25) repaired 783 partial ("fragment") minutes in `ironbeam_es_1m_bars` and `es_minutes` (2026-01-02 → 2026-10-22; many at 06:30 / 12:59 PT). No derived table was recomputed afterwards. CR-DL's dry run measured the effect only on rows ≤ 2026-03-20: `v0.6.1-nyse-sessions` 43 rows, `v0.6.0-openiv` 29, `v0.5.0-rebuilt` 26.

## Kickoff (frozen)

Read first: CLAUDE.md (Unattended backfill protocol); vault `components/field-conventions.md`; `research/standing-rulings.md` (B4 incl. the 2026-10-29 blind-maintenance clarification); vault open-question `crbz-fragment-repair-derived-recompute` (F2); `specs/CR-DL-march-2026-derived-recompute.md` (same method: archive first, compare-and-update, only changed columns, one transaction). Data CR: conformance table required.

NEVER print connection strings: catch DB errors and print the error type only.

**Scope:** correct every derived value that depends on a repaired bar, in every live derived table (`bt_daily_outcomes` in all ES-based feature versions, `vol_risk_premium`, and anything else Step 0 finds), using the code that wrote each row (as CR-DL did, including exact copies from git history for retired code paths).

**Two regimes (B4):**
- **TRAIN** (trade dates ≤ 2026-06-05, horizons ending ≤ 06-05): full CR-DL treatment. Per-row before / after tables in the spec, values visible.
- **HOLDOUT** (any row whose trade date or horizon touches 2026-06-06 → 2026-10-26): **BLIND.** Scripts may read and write these rows, but nothing shown to Ryan or in any output, log, spec, commit or session note may contain an outcome value, row, diff, example or summary statistic from them. Allowed output only: row counts per table / version, the names of columns changed, counts of changed values per column, and checksums (e.g. md5 of the sorted recomputed column values) with a second independent run reproducing the same checksums. Logs for holdout steps are written in that form from the start; never write values and redact later.
- **Forward log** (≥ 2026-10-27) and **R-b** (2025): confirm in Step 0 (counts only) that no repaired bar touches them. If any does, STOP and report.

**Step 0** (commit before any write; stop and report):
- a. Every derived table and column that reads Ironbeam bars, and which code wrote each feature version's rows. Confirm each path recomputes stored rows unchanged on the bars as they were before CR-BZ (train: match counts; holdout: checksum equality only).
- b. Train: per table / version, the rows and columns the repair changes, with before / after values, and confirm no outcome label changes; list any that do.
- c. Holdout: counts only — rows per table / version that would change, columns changed, and the number of labels (status / touch / reached) that would change, as a COUNT, no values.
- d. Interaction with CR-DL / CR-DJ: rows already rewritten for March 2026 must come out identical or differ only in CR-BZ columns. Show that for train; checksum only for any holdout overlap.
- e. Write plan under the unattended backfill protocol, any grants needed (don't apply without Ryan), and the blind smoke design for holdout rows.

Ryan decides after Step 0, especially if any label count (b or c) is non-zero.

**Wrap:** session note (holdout parts in counts / checksums only), PR (don't merge), logs copied (holdout logs are already count-only), no deploy.

## Standing rulings conformance

| Ruling | Status | Note |
|---|---|---|
| A1–A22 (the setup) | n/a | Data repair of stored derived rows; no setup, level, entry, exit or event filter is defined or tested. |
| B1, B2, B3 | n/a | No verdict, parameter sweep or setup optimisation. |
| B4 (data split) | honoured, under the 2026-10-29 blind-maintenance clarification | Train rows: full treatment. Holdout rows (trade date or any bar read in 2026-06-06 → 10-26) are read and written by scripts only; every output for them is counts, column names and checksums, from the first run. Forward log and R-b: counts-only confirmation that no repaired bar touches them; no writes there. No research result is read or re-run. |
| B5 (Databento is the reference) | n/a | No test. The recompute reproduces each stored row's own Ironbeam-bar computation on the repaired bars (CR-BZ rebuilt the fragments from Ironbeam's own 1-s flow). |
| B6, B7, B8 | n/a | No exits, placebos or labels. `setup_candidates` / `setup_reviews` are not modified. |
| C1, C2 | n/a | No UI or display change. |
| Revoked R1–R7 | none used | R6 (`discounted_level`) is the stored `bt_daily_outcomes` target frame; the recompute reproduces the existing computation unchanged and introduces it into no new test. |
