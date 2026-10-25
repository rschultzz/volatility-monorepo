---
type: cr
cr_id: CR-CA
title: Databento ES Trades Backfill for Flow History — 2024-04-03 → 2025-12-31 aggressor flow at 1 s, new table, validated against Ironbeam 2026
status: created
run_mode: interactive
size: medium
estimated_days: 1-2
data_safety_class: writes_reviewed
dependencies: []
soft_dependencies: [CR-BW, CR-BT]
depended_on_by: []
branch_name: feat/CR-CA-databento-es-flow-backfill
last_reviewed: 2026-10-25
related_decisions: []
related_open_questions:
  - "[[databento-2025-backfill-calendar-roll-dying-contract]]"
  - "[[ironbeam-1m-bars-miss-trades-in-fast-minutes]]"
related_sessions: []
tags: [dash, cr, created, databento, order-flow, aggressor, backfill, data-pipeline]
---

# CR-CA — Databento ES Trades Backfill for Flow History

## Why

The order-flow tests (CR-BW, CR-BX, CR-BY) ran on 88 sessions, because `ironbeam_es_flow_1s` starts 2025-12-26. Their "levels vs placebo" questions came back **inconclusive, not refuted**: CR-BW T2 +3.8 [−9.7, +17.6]; put-level readout +20 [−1.9, +43.6]. Ryan (2026-10-25): extend the history so wide CIs stop deciding outcomes.

- **Databento** GLBX.MDP3 carries ES trades with the aggressor side back to 2010. Pay per GB; about 2 GB per month for ES trades.
- `DATABENTO_API_KEY` is already in `.env`; `scripts/backfill_databento_es.py` uses it.
- **Corrected levels exist only from 2024-04-03** (CR-BT A1.1), so that is the start date.
- **This window has never been read with flow.** It becomes the **replication set R**: frozen specs run on it unchanged, no re-tuning.

**DB capacity** (chat, 2026-10-25): the database is 22 GB on a 30 GB disk, with autoscaling off. `ironbeam_es_flow_1s` is 2.4 GB for about 10 months, all hours. A full-hours backfill of 21 months would be about 5 GB, which is too close to the limit. Hence an RTH-window table.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Pull Databento ES trades for 2024-04-03 → 2025-12-31, aggregate to 1 second, write to a **new** table. No change to `ironbeam_es_flow_1s`, `ironbeam_es_1m_bars`, `es_minutes` or anything else. |
| L2 | Source | GLBX.MDP3, schema `trades`. Symbols: **outright quarterly contracts** (`stype_in=raw_symbol`, e.g. ESM4, ESU4, …, ESZ5). Never `ES.c.0` (that is the dying-contract problem in the 2025 bars). |
| L3 | Front contract per session | The front contract is the nearest quarterly until the **Monday of the week before its expiry week** (the CR-BM/BO roll-week start); from that Monday on, the next contract. Every row records `raw_symbol`. **Roll weeks are pulled for both contracts** and stored with each row's own symbol, so readers choose; CR tests already exclude roll weeks. |
| L4 | Window kept | 06:15–13:00 PT on NYSE sessions (`packages/shared/sessions.py`), DST-safe. That covers RTH plus the 300 s pre-windows the tests need. Expected size ≈ 1.5–2 GB; Step 0 measures it. |
| L5 | Target table | New `es_flow_1s_databento`, with the same columns as `ironbeam_es_flow_1s` (ts_utc, symbol, trade_date, volume, notional, trade_count, buy_vol, sell_vol, unknown_vol, buy_count, sell_count, unknown_count, open, high, low, close, first_trade_ms, last_trade_ms) plus `source text = 'databento'`. PK (ts_utc, symbol). Same aggregation rules as `ironbeam_flow_1s_aggregate.py`. |
| L6 | Aggressor mapping | Databento `side`: `B` = buy aggressor, `A` = sell aggressor, `N` = unknown. **Verified by measurement before any write** (Step 0 Q2); if the measured sign contradicts this mapping, stop. |
| L7 | Raw files | Downloaded DBN files are kept on Ryan's Mac under a git-ignored path (`data/databento/raw/`), not in the DB and not in git. |
| L8 | Cost guard | `metadata.get_cost` for the full pull before downloading. **Proceed without asking if the total is ≤ $150.** Above that, stop and report. Also report the MBP-1 (top-of-book) cost for the same window, as information only; no MBP-1 pull in this CR. |
| L9 | Write safety | Backfill protocol (CLAUDE.md): `get_backfill_db_conn`, `assert_role_or_die`, `backfill_run(conn, "CR-CA")`, `update_run_smoke`. INSERT only, `ON CONFLICT DO NOTHING`. Table DDL in its own commit; if the backfill role cannot CREATE, Code stops and gives Ryan the DDL to run. |
| L10 | Data policy | The new window is **replication set R**. Any test that reads R must be pre-registered and frozen first (specs written before R is read). R is not tuning data. |

## Step 0 — diagnosis gate (answer all before any write; contradiction-stop)

1. **Cost and size:**
   - `get_cost` for trades per contract per month across the window, plus the total;
   - MBP-1 cost for the same window (information only);
   - estimated DB size of the RTH 1 s table, measured from one month's aggregate;
   - current DB free space.
   - **Stop if the projected DB size after the load exceeds 27 GB.**
2. **Overlap validation** on Ironbeam's window: pull Databento trades for 2026-01-12 → 2026-01-23 (ESH6; outside roll weeks; cost included above). Aggregate to 1 s and compare with `ironbeam_es_flow_1s`:
   - per-minute volume ratio (median, p5–p95);
   - per-minute buy − sell correlation;
   - per-minute OHLC differences.
   - **Sign test:** corr(minute delta, minute Δprice) must be positive (Ironbeam gives +0.49).
   - **Stop if** the volume ratio median is outside 0.95–1.05, the delta correlation is below 0.9, or the sign test fails.
3. **Roll calendar:** list each contract's active sessions under L3, the roll weeks, and the sessions on which two contracts are stored.
4. **Bars cross-check:** for 10 random 2025 sessions, compare the Databento-built 1-minute OHLC (front contract) with the existing `ironbeam_es_1m_bars` (`ES.c.0`, source `databento`). Report differences outside roll weeks, and the dying-contract days inside them. Information for the open question; no fix here.
5. **Schema/DDL:** the proposed DDL, and whether `dash_backfill_writer` can create it.

## Implementation order (one commit each)

1. `scripts/cr_ca_databento_flow.py`: cost check, download to the L7 path, aggregation identical to `ironbeam_flow_1s_aggregate.py`, `--dry-run`, `--month`.
2. DDL commit (or Ryan runs it).
3. Overlap validation output (Step 0 Q2) committed under `scripts/logs/` (git-ignored) with a summary in the spec.
4. Backfill run, month by month, with a smoke per month: row counts, symbols, the sign test per month.
5. Field-conventions row for `es_flow_1s_databento`; session note.

## Not in scope

- MBP-1 / order book pulls (cost reported only).
- Any re-run of CR-BU/BW/BX/BY on R. Those are separate pre-registered replication CRs.
- Fixing the 2025 `ironbeam_es_1m_bars` dying-contract rows.

## Kickoff prompt for Claude Code

```
Start CR-CA in its own git worktree (other CR worktrees may be active; do not switch the main checkout's branch). First action: git fetch && git worktree add <scratch>/crca-wt -b feat/CR-CA-databento-es-flow-backfill origin/main (confirm CR-CA is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CA — Databento ES Trades Backfill for Flow History.md", components/field-conventions.md, scripts/backfill_databento_es.py (DATABENTO_API_KEY in .env), apps/cron/ironbeam_flow_1s_aggregate.py.
Commit 1: spec freeze to specs/CR-CA-databento-es-flow-backfill.md, verbatim.
Commit 2: Step 0 findings — all five questions. The only Databento download allowed in Step 0 is the Q2 overlap window (2026-01-12 .. 2026-01-23, ESH6) after get_cost shows the full-window total <= $150; otherwise stop. No DB writes. Stop and report.
```

## Amendment A1 (Ryan, 2026-10-25, before any Databento request)

Restart note: a cloud run of this CR made unpushed commits (93d9f4d, be55a33) that are lost; this restart ignores them, except that Step 0 Q4 reuses its fixed sample seed (20261025).

- **A1.1 `symbol`** = Ironbeam format (e.g. `XCME:ES.H26`), so rows line up with `ironbeam_es_flow_1s`; plus a **`raw_symbol`** column holding the Databento/CME symbol (e.g. `ESH6`). (Amends L3/L5.)
- **A1.2 `trade_date`** = the UTC date of `ts_utc` (the Ironbeam convention measured in field-conventions). (Amends L5.)
- **A1.3 Bucketing** on Databento `ts_event` (matching-engine time), floored to the second. (Amends L5.)
- **A1.4 Window** [06:15:00, 13:00:00) America/Los_Angeles, end-exclusive. (Amends L4.)
- **A1.5 DDL:** Ryan approves running the spec's CREATE TABLE (with the A1.1 column), plus `GRANT SELECT, INSERT ON es_flow_1s_databento TO dash_backfill_writer`, once, with `DATABASE_URL`, **after Step 0 passes — not before**. (Amends L9.)
- **A1.6 Step 0 order:** `get_cost` first (trades for the full window, plus MBP-1 for information); the Q2 overlap pull (2026-01-12 .. 2026-01-23, ESH6) only if the full-window trades total ≤ $150. Q1 size is measured from that real pulled window. No DB writes in Step 0.
