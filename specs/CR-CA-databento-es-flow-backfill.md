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

## Step 0 findings (2026-10-25) — **STOP at the L8 cost gate**

Script: `scripts/cr_ca_step0.py` (phases `cost`, `pull`, `overlap`, `size`, `roll`, `bars`, `ddl`); outputs under `scripts/logs/cr_ca/` (git-ignored). No Databento download was made and nothing was written to the DB. The only Databento calls were `metadata.get_cost` / `get_billable_size` (free).

### Q1 — cost and size

**Trades** (GLBX.MDP3, `stype_in=raw_symbol`, one `get_cost` per session-contract window [06:15, 13:00) PT; 439 NYSE sessions, 507 session-contracts under L3):

| | USD | billable |
|---|---|---|
| R, front contract only (439) | 188.41 | 7.23 GB |
| R, dying contract in roll weeks (68) | 19.44 | 0.74 GB |
| **R total** | **207.85** | 7.97 GB |
| Q2 overlap (ESH6, 2026-01-12 .. 01-23, 10 sessions) | 4.39 | |
| **Full-window total** | **212.24** | |

Rate ≈ $26.08 per billable GB. Monthly per-contract figures: `scripts/logs/cr_ca/q1_cost.txt`, `cost_trades.csv`. Range $3.3–15.3 per contract-month; heaviest 2025-04 ($15.26, ESM5).

**L8 gate: $212.24 > $150 → STOP.** Per A1.6 the Q2 overlap pull was not run. Even front-only plus Q2 ($192.80) exceeds the cap.

**MBP-1** (information only): R $364.95 (217.7 GB billable, $1.68/GB), Q2 $9.29, total $374.24.

**DB free space:** database 23.15 GB on the 30 GB disk (spec; autoscaling off) → ≈ 6.85 GB free, 3.85 GB headroom to the 27 GB stop. `ironbeam_es_flow_1s`: 10,281,770 rows, 2.543 GB total (1.686 GB heap), 247 B/row with its PK and three secondary indexes (two of them duplicate `trade_date` indexes).

**Table size — not measured from a real month** (needs the Q2 pull). Provisional bound from Ironbeam's own density in the Q2 window: 22,176 rows per session in [06:15, 13:00) PT (9 NYSE sessions; 2026-01-19 MLK excluded). At 261 B/row (Ironbeam's 247 + two text columns): front-only 2.54 GB, upper bound (all 507 session-contracts at the front rate) 2.94 GB → DB 25.7–26.1 GB after the load, under the 27 GB stop by ≈ 0.9–1.3 GB. Provisional; Q1c stays open until a real Databento window is aggregated.

### Q2 — overlap validation

**Not run** (L8 stop). Script phases `pull` → `overlap` → `size` are written and gated on `cost_summary.json` ≤ $150.

### Q3 — roll calendar (L3)

| contract | `symbol` (A1.1) | front | front n | stored through | stored n |
|---|---|---|---|---|---|
| ESM4 | XCME:ES.M24 | 2024-04-03 .. 06-07 | 47 | 06-21 | 56 |
| ESU4 | XCME:ES.U24 | 2024-06-10 .. 09-06 | 62 | 09-20 | 72 |
| ESZ4 | XCME:ES.Z24 | 2024-09-09 .. 12-06 | 64 | 12-20 | 74 |
| ESH5 | XCME:ES.H25 | 2024-12-09 .. 2025-03-07 | 60 | 03-21 | 70 |
| ESM5 | XCME:ES.M25 | 2025-03-10 .. 06-06 | 63 | 06-20 | 72 |
| ESU5 | XCME:ES.U25 | 2025-06-09 .. 09-05 | 62 | 09-19 | 72 |
| ESZ5 | XCME:ES.Z25 | 2025-09-08 .. 12-05 | 64 | 12-19 | 74 |
| ESH6 | XCME:ES.H26 | 2025-12-08 .. 12-31 | 17 | 12-31 | 17 |

Two contracts stored on 68 of 439 sessions: the roll Monday (expiry Friday − 11 days) through expiry Friday — 2024-06-10..21 (9, Juneteenth), 09-09..20, 12-09..20, 2025-03-10..21, 06-09..20 (9), 09-08..19, 12-08..19 (10 each). No session without a contract. Note: Ironbeam's own 2026 data rolled H26 → M26 on 2026-03-23 (after the 03-20 expiry), not on the L3 Monday; the A1.1 `symbol` matches Ironbeam's format, not its roll timing.

### Q4 — bars cross-check

Sample (seed 20261025, `random.Random(seed).sample` of the 2025 NYSE sessions, 10): 2025-03-18 (roll wk), 03-27, 05-23, 06-10 (roll wk), 06-17 (roll wk), 08-05, 09-12 (roll wk), 09-29, 10-20, 11-25. **Not run:** it needs a Databento trades download for those sessions ($3.75), which the frozen kickoff does not allow in Step 0 beyond the Q2 window. The script phase (`bars --allow-q4-pull`) is ready. Caveat: the sample was regenerated with the seed; whether it matches the lost cloud run's draw method is not verifiable.

### Q5 — schema / DDL

`dash_backfill_writer`: no CREATE on schema `public` or the database → the DDL runs under A1.5 (`DATABASE_URL`, after Step 0 passes). `es_flow_1s_databento` does not exist yet. Proposed DDL (also in the script):

```sql
CREATE TABLE es_flow_1s_databento (
  ts_utc timestamptz NOT NULL,
  symbol text NOT NULL,              -- XCME:ES.H26 (A1.1)
  raw_symbol text NOT NULL,          -- ESH6 (A1.1)
  trade_date date NULL,              -- UTC date of ts_utc (A1.2)
  volume integer NOT NULL DEFAULT 0, notional double precision NOT NULL DEFAULT 0,
  trade_count integer NOT NULL DEFAULT 0,
  buy_vol integer NOT NULL DEFAULT 0, sell_vol integer NOT NULL DEFAULT 0, unknown_vol integer NOT NULL DEFAULT 0,
  buy_count integer NOT NULL DEFAULT 0, sell_count integer NOT NULL DEFAULT 0, unknown_count integer NOT NULL DEFAULT 0,
  open double precision, high double precision, low double precision, close double precision,
  first_trade_ms bigint, last_trade_ms bigint,
  source text NOT NULL DEFAULT 'databento',
  PRIMARY KEY (ts_utc, symbol)
);
CREATE INDEX idx_es_flow_1s_databento_trade_date ON es_flow_1s_databento (trade_date);
GRANT SELECT, INSERT ON es_flow_1s_databento TO dash_backfill_writer;
```

One secondary index (trade_date) instead of Ironbeam's three; the (ts_utc, symbol) PK serves time-range reads.

### Decision needed (Ryan)

The cap blocks everything downstream of Q1. Options: raise the L8 cap to ≥ $213 (or ≥ $193 for front-only + Q2, which would amend L3's both-contracts rule); or shorten R. Q2 ($4.39) and Q4 ($3.75) could also be authorized on their own to validate before committing the full spend.

## Amendment A2 (Ryan, 2026-10-25, after the Step 0 L8 stop)

- **A2.1** L8 cost cap raised to **$215 total for trades** (Step 0 measured $212.24). MBP-1 stays information only, not pulled.
- **A2.2 Step 1:** authorize the Q2 overlap ($4.39) and Q4 bars-check ($3.75) downloads only; complete Q2 (volume ratio, delta correlation, OHLC diffs, sign test with B = buy / A = sell) and Q4; measure Q1 table size from the real Q2 window; commit as the Step 0 completion. If any Q2 stop rule fires, stop — no further downloads.
- **A2.3 Step 2**, only if Q2 passes **and** the projected DB size after load is ≤ 27 GB: DDL approved (run once with `DATABASE_URL`, then `GRANT SELECT, INSERT` to `dash_backfill_writer`); full trades pull and load (total spend ≤ $215), month by month under the backfill protocol, with a per-month smoke (rows, symbols, sign test). Stop and report after the load; do not push.
- **A2.4** Never print connection strings: the CR-CA scripts mask `DATABASE_URL` / `BACKFILL_DATABASE_URL` in all log and exception output (redaction helper).

## Step 0 completion under A2 (2026-10-25) — **STOP at the Q2 volume-ratio rule**

Downloads made: the Q2 window only (ESH6, 9 NYSE sessions 2026-01-12 .. 01-23 excluding the 01-19 MLK holiday, [06:15, 13:00) PT; 9 DBN files, 59.7 MB, under `data/databento/raw/`; priced at $4.39). **Q4 was not downloaded**: A2.2 stops all further downloads once a Q2 stop rule fires. No DB writes; no DDL. Connection strings are now masked in all script output (A2.4, `redact()` / `install_redaction()` in `scripts/cr_ca_step0.py`).

### Q2 — overlap validation (Databento ESH6 vs `ironbeam_es_flow_1s` `XCME:ES.H26`)

Raw: 3,486,666 trades, all `action = T`, `flags = 0`; side B 1,751,762 / A 1,734,904 / N 0. Aggregated to 1 s on `ts_event` (A1.3): 199,597 rows, 10,709,233 contracts. Ironbeam over the same 9 sessions: 9,947,511 contracts. Ironbeam `trade_date` equals the UTC date of `ts_utc` on 100 % of rows (A1.2 confirmed). Minutes: 3,645 in both (9 × 405). The 225 Ironbeam-only minutes are the 2026-01-19 MLK Globex session (06:15–10:00 PT), which is not an NYSE session.

| measure | value | rule | result |
|---|---|---|---|
| per-minute volume ratio Ironbeam / Databento | median **0.943**; p5 0.860, p25 0.920, p75 0.961, p95 0.979; pooled 0.921 (per session 0.904–0.933) | median in 0.95–1.05 | **FAIL** |
| per-minute delta (buy − sell) correlation | **0.908** (buy 0.9965, sell 0.9976; sign agreement 0.913; slope Ironbeam on Databento 0.71) | ≥ 0.9 | pass |
| per-minute OHLC Databento − Ironbeam | median 0.00 on all four; equal to the tick on 99.95 / 99.09 / 99.18 / 99.37 % of minutes (O/H/L/C); max abs 0.25 / 0.75 / 0.25 / 0.25; none > 1 pt | report | — |
| sign test corr(minute delta, close − open), B = buy / A = sell | Databento **+0.650**; Ironbeam +0.546 | > 0 | pass |

**L6 aggressor mapping confirmed** (B = buy, A = sell): positive sign test, and buy/sell volumes correlate 0.997 with Ironbeam's BUY/SELL.

**Where the volume gap comes from** (diagnostic on the downloaded window; `scripts/logs/cr_ca/q2_diag.txt`):
- Per second, Databento has more volume than Ironbeam in 36.8 % of seconds and less in only 0.2 %. Missing from Ironbeam: 844,712 contracts; extra in Ironbeam: 2,071.
- Prints, not size: trade count ratio 0.932; mean print size 3.07 (Databento) vs 3.04 (Ironbeam). Ironbeam drops whole prints.
- More are dropped when the market is busier: median ratio 0.960 → 0.951 → 0.945 → 0.938 → 0.921 by Databento minute-volume quintile. By PT hour the ratio is 0.91–0.93, lowest at 06:xx and 11–12:xx.
- Even across sides: 7.83 % of buy volume and 7.90 % of sell volume missing. But per second, the missing delta correlates **0.64** with Databento's delta, which is why the delta slope is 0.71. On minutes where Ironbeam has ≥ 95 % of Databento's volume (1,445 of 3,645), delta correlation is **0.990**.
- Consistent with the existing measurement that Ironbeam's feed misses trades in fast minutes (field-conventions, `ironbeam_es_1m_bars` row; open question `ironbeam-1m-bars-miss-trades-in-fast-minutes`). The gap is all one-sided (Ironbeam short), so this reads as **Ironbeam under-capturing, not Databento over-counting**. That is an inference: neither feed was checked against an independent volume print (e.g. CME cleared volume via Databento `statistics`, which would be a download).

The rule fired as written, so per A2.2 this is a stop. Implication for any later use: the 2026 Ironbeam flow and the Databento R table would not be on the same volume scale (Ironbeam ≈ 0.92× and short more in busy seconds). Delta magnitudes are ~0.7× in Ironbeam. Any test that pools the two, or carries thresholds from 2026 over to R, must account for this.

### Q1c — size from the real Q2 window

22,177 1 s rows per session (of 24,300 possible), matching Ironbeam's density (22,176). Projection at 261 B/row (Ironbeam's 247 B/row with PK + three secondary indexes, + two text columns; the proposed DDL has one secondary index, so this is conservative):
- Front rows scaled by billable trade bytes: 7.66 M rows, 2.00 GB.
- Flat upper bound (all 507 session-contracts at the Q2 rate): 11.24 M rows, 2.94 GB.

DB 23.15 GB → **25.16–26.09 GB after the load: Q1 gate PASS** (≤ 27 GB).

### Q4 — not run (A2.2 stop; sample unchanged above).

### Status

Step 2 (DDL, full pull, load) **not started**: A2.3 requires Q2 to pass. Spend so far: $4.39 of $215. Decision needed: whether the 0.95–1.05 volume rule is the right test against an incomplete reference feed. Options: (a) amend the rule to an independent reference, e.g. Databento `statistics` cleared volume for the same days (a small extra download), with Databento ≈ 1.0 of cleared volume as the gate; (b) accept Q2 as is on the evidence above; (c) stop CR-CA.

## Amendment A3 (Ryan, 2026-10-25, after the Q2 stop under A2)

- **A3.1 Reference feed:** Databento is the reference. The Q2 volume-ratio rule (median 0.95–1.05) is replaced. Q2 passes if: Databento ≥ Ironbeam volume on ≥ 95 % of minutes; OHLC matches to the tick on ≥ 99 % (read per field: O, H, L, C each); sign test > 0; and delta correlation ≥ 0.9 on minutes where Ironbeam is ≥ 95 % complete.
- **A3.2** The Ironbeam undercount (≈ 0.94× volume, ≈ 0.7× delta, worse in busy minutes) is recorded as a finding, with a field-conventions row for `ironbeam_es_flow_1s` and an open question.
- **A3.3 Scope extended:** window **2024-04-03 → 2026-10-23**, same table `es_flow_1s_databento`. Loading 2026 sessions after 2026-06-05 (holdout) is allowed; **reading the holdout is not** (smokes are mechanical row/symbol/sign checks only).
- **A3.4 Cost cap $330 total for trades** (supersedes A2.1); `get_cost` for the extension first; stop if total > $330.
- **A3.5 DB size stop raised to 32 GB** (disk is now 35 GB; supersedes the 27 GB rule in Q1).
- **A3.6** Then: run Q4, apply the DDL with `DATABASE_URL` (A1.5), full backfill month by month under the protocol with the per-month smoke. Stop and report after the load; do not push.

## Step 0 under A3 (2026-10-25) — **PASS; Step 2 proceeds**

### Q2 under A3.1 — PASS (same downloaded window; `scripts/logs/cr_ca/q2_a3.txt`)

| rule | value | result |
|---|---|---|
| Databento ≥ Ironbeam volume on ≥ 95 % of minutes | **100.0 %** of 3,645 | pass |
| OHLC equal to the tick on ≥ 99 % (per field) | O 99.95, H 99.09, L 99.18, C 99.37 % | pass |
| sign test > 0 (B = buy, A = sell) | +0.650 | pass |
| delta corr ≥ 0.9 where Ironbeam ≥ 95 % complete | **0.990** (n 1,445) | pass |

Reading note: all four fields equal in the same minute on 97.6 % of minutes. The rule is read per field, matching the per-field figures A3 was written from. All 33 high and 30 low mismatches have Databento's range wider (Ironbeam missed the extreme print). Ironbeam undercount (A3.2): field-conventions row "`ironbeam_es_flow_1s` vs Databento trades"; open question `ironbeam-es-flow-1s-undercounts-trades`.

### Q1 under A3 — cost PASS

`get_cost` for the extension (2026-01-01 .. 2026-10-23, 232 session-contracts): **$98.10** trades (3.76 GB billable); MBP-1 $200.50 (information). Whole window 2024-04-03 .. 2026-10-23: 739 session-contracts, **trades $305.95** (11.73 GB) ≤ $330 → PASS. This total includes the Q2 and Q4 sessions; their DBN files are reused by the load, so actual spend stays at the priced total. MBP-1 whole window $565.46, not pulled.

Roll calendar extension (L3): ESH6 front 2025-12-08 .. 2026-03-06, ESM6 from 2026-03-09, ESU6 from 2026-06-08, ESZ6 from 2026-09-08 (Labor Day 09-07); two contracts on each roll span through expiry Friday. Full table in the `roll` phase output (`scripts/logs/cr_ca/q3.txt`, regenerated for the A3 window).

### Q4 — bars cross-check ($3.75; seed 20261025)

| session | front | roll wk | close diff median | close equal | bar vol / Databento front vol |
|---|---|---|---|---|---|
| 2025-03-18 | ESM5 | yes | +51.25 | 0.000 | 0.29 |
| 2025-03-27 | ESM5 | | 0.00 | 1.000 | 1.000 |
| 2025-05-23 | ESM5 | | 0.00 | 1.000 | 1.000 |
| 2025-06-10 | ESU5 | yes | +53.75 | 0.000 | 83.8 |
| 2025-06-17 | ESU5 | yes | +53.50 | 0.000 | 0.19 |
| 2025-08-05 | ESU5 | | 0.00 | 0.995 | 1.000 |
| 2025-09-12 | ESZ5 | yes | +56.00 | 0.000 | 10.6 |
| 2025-09-29 | ESZ5 | | 0.00 | 1.000 | 1.000 |
| 2025-10-20 | ESZ5 | | 0.00 | 1.000 | 1.000 |
| 2025-11-25 | ESZ5 | | 0.00 | 0.998 | 1.000 |

- **Outside roll weeks** (6 sessions) the Databento-built front minutes equal the 2025 `ironbeam_es_1m_bars` (`source = 'databento'`, `ES.c.0`). Close equal on 99.5–100 % of minutes, max |diff| 0.25, volume ratio 1.000. Same source, same trades.
- **Inside roll weeks** (4 sessions) the bars are the dying contract: +51 to +56 pts against the new front, the dying-contract problem in open question `databento-2025-backfill-calendar-roll-dying-contract`. On 2025-06-10 the bar contract (ESM5) carried 84× the L3 front's (ESU5) volume, and on 2025-09-12 10.6×. On those days the L3 "front" was the thin contract. Both contracts are stored on roll-week sessions, so readers choose. No fix here.

## Load (2026-10-25) — complete, all 31 months PASS

- **DDL** applied once with `DATABASE_URL` (`scripts/cr_ca_ddl.sql`): `es_flow_1s_databento`, PK (ts_utc, symbol), index on trade_date; `dash_backfill_writer` has SELECT, INSERT (no DELETE).
- **Backfill:** `scripts/cr_ca_databento_flow.py`, one `backfill_run(conn, "CR-CA")` per month as `dash_backfill_writer`. 31 runs in `bt_backfill_runs`, all `completed`, 2026-10-25 19:04 → 20:07 UTC. Per-month smoke results are in `smoke_test_results`; log in `scripts/logs/cr_ca/backfill_*.txt`.
- **Raw files (L7):** 739 DBN files, 4.3 GB, in the main checkout's `data/databento/raw/` (git-ignored by this branch's `.gitignore`; untracked in the main checkout until merge). The Q2 and Q4 files were reused.
- **Spend:** priced $305.95 for trades (cap $330, A3.4). The Databento invoice was not checked from here.

**Table:**
- 15,285,418 rows; 643 sessions (every NYSE session 2024-04-03 → 2026-10-23); 11 contracts ESM4 → ESZ6; 757,614,587 contracts of volume.
- `unknown_vol` 3,276 contracts in total (side N, 0.0004 %).
- Size: 3.43 GB total (2.66 GB heap), 224 B/row. DB 26.58 GB after the load (stop 32 GB, A3.5).

| contract | `symbol` | rows | trade_date span |
|---|---|---|---|
| ESM4 | XCME:ES.M24 | 1,166,601 | 2024-04-03 .. 06-21 |
| ESU4 | XCME:ES.U24 | 1,492,432 | 2024-06-10 .. 09-20 |
| ESZ4 | XCME:ES.Z24 | 1,473,088 | 2024-09-09 .. 12-20 |
| ESH5 | XCME:ES.H25 | 1,471,248 | 2024-12-09 .. 2025-03-21 |
| ESM5 | XCME:ES.M25 | 1,583,325 | 2025-03-10 .. 06-20 |
| ESU5 | XCME:ES.U25 | 1,388,577 | 2025-06-09 .. 09-19 |
| ESZ5 | XCME:ES.Z25 | 1,505,359 | 2025-09-08 .. 12-19 |
| ESH6 | XCME:ES.H26 | 1,468,480 | 2025-12-08 .. 2026-03-20 |
| ESM6 | XCME:ES.M26 | 1,548,749 | 2026-03-09 .. 06-18 |
| ESU6 | XCME:ES.U26 | 1,504,477 | 2026-06-08 .. 09-18 |
| ESZ6 | XCME:ES.Z26 | 683,082 | 2026-09-08 .. 10-23 |

**Per-month smoke (all PASS):**
- DB rows = built rows = inserted rows every month.
- Every session has exactly the L3 contract set.
- 0 rows outside [06:15, 13:00) PT; 0 rows with trade_date ≠ UTC date of ts_utc.
- Sign test (front, minute delta vs close − open) +0.47 to +0.70. The lowest values fall in roll months (2026-06 +0.468, 2025-12 +0.485, 2024-09 +0.486), where the L3 front is the thin contract in the roll Monday's first days.
- Front rows per session: median 19.5–23.9 k. Minima are roll Mondays (new front thin: 1.7–6.7 k) and early closes (12–13 k).
- Holdout months (2026-06 → 10) were loaded; only these mechanical checks touched them (A3.3).

**Edge note:** `get_range` bounds requests on the record timestamp (`ts_recv`); rows are bucketed and window-filtered on `ts_event` (A1.3). A trade with `ts_event` < 13:00:00 and `ts_recv` ≥ 13:00:00 PT (microseconds) can be missing from the last second.
