# CR-DJ — Patch the March 2026 (and check June 2026) roll windows on the dying ES contract

Branch `feat/CR-DJ-roll-patch-2026-03` (worktree `../volatility-monorepo-cr-dj`) off `origin/main` `dffa3c8`.
Kickoff: chat 2026-10-28 (Ryan). Data CR.

## Kickoff (frozen)

Read first: CLAUDE.md (Unattended backfill protocol; the CR-BF patched-window convention for 2026-09-13 → 09-18); vault `components/field-conventions.md`; `research/standing-rulings.md` (B5: Databento is the reference). Conformance table required.

**Finding (chat, 2026-10-28, read-only):** `ironbeam_es_1m_bars` (and `es_minutes`, same values) stayed on ESH6 after the March roll. Daily 14–19 UTC average close, Ironbeam minus Databento ESM6: about −50 pts on 2026-03-09 → 03-19; Ironbeam equals ESH6 through 03-19 and ESM6 from 03-20. Ironbeam volume fell to ~350/min on 03-17 → 03-19 (dying contract). CR-CZ cards (`setup_candidates`, set `cr-cz:cp-ranges`) are built on Databento ESM6 from 03-16, so 10 cards (03-16 ×4, 03-18 ×2, 03-19 ×4) sit ~50 pts above the chart. June: Ironbeam switched to ESU6 on 06-15; 06-11 / 06-12 may be on ESM6 after the roll date.

**Step 0 (commit before any write; read-only):**
- a. State the roll convention CR-BF used for its patch (roll date, start / end boundary in UTC) and the convention the CR-CZ card builder used. If they differ, STOP and report both; Ryan picks one before anything is written.
- b. Under that convention, list the exact UTC windows on the dying contract for the 2026-03 and 2026-06 rolls (and confirm 2026-12 hasn't happened yet). Per window: minutes affected, Ironbeam vs Databento front-month offset, volume ratio.
- c. Check that Databento front-month 1-minute bars exist for every affected minute. If any are missing, report the cost to fetch before fetching.
- d. List every table holding bar prices for those windows and which readers are live.
- e. Report which CR-CZ / review cards fall in the windows, and whether any were already labelled.

**Implementation (only after Step 0 is approved):** same method as CR-BF — replace the dying-contract rows in the windows with the front contract from Databento, `source='databento'`, raw_symbol recorded, in both `ironbeam_es_1m_bars` and `es_minutes`. Unattended backfill protocol exactly (`get_backfill_db_conn`, `assert_role_or_die`, `backfill_run "CR-DJ"`, `update_run_smoke`). Insert / update only, no DELETE; if CR-BF needed deletes, report how it did them. Recompute derived tables Step 0d lists, or list them as a follow-up. Update CLAUDE.md's Conventions exception and field-conventions.md.

**Smoke:** each patched window equals Databento's front contract minute for minute; 2026-03-18 11:00–11:15 PT chart shows 6716–6737.5; the 10 CR-CZ cards' boxes sit on the chart's price; `bt_backfill_runs` has the CR-DJ row with smoke results.

**Wrap:** session note, PR (don't merge), logs copied. Don't deploy.

## Standing rulings conformance

| Ruling | Status | Note |
|---|---|---|
| A1–A22 (the setup) | n/a | Data repair; no setup, level, entry, exit or event filter is defined or tested. |
| B1, B2, B3 | n/a | No verdict, parameter sweep or setup optimisation. |
| B4 (data split) | **flag — Ryan to confirm** | No test is run. To answer "check June", Step 0 read June rows (holdout 2026-06-06 → 10-26): daily RTH-window volume per contract in `es_flow_1m_databento` 06-08 → 06-16, and Ironbeam bar counts / >20-pt jumps / flow symbol ranges. Contract-identity aggregates only; no outcome, setup, level or card was read. The March window is train. |
| B5 (Databento is the reference) | honoured | Patch source is Databento GLBX.MDP3 `ohlcv-1m`, outright `ESM6` / `raw_symbol`. |
| B6, B7, B8 | n/a | No exits, placebos or labels. |
| C1, C2 | n/a | No UI or display-feature change. Chart prices for 2026-03-15 22:00 → 03-20 UTC change because the stored bars are corrected (the purpose of the CR). |
| Revoked R1–R7 | none used | |

## Step 0 findings (2026-10-28, read-only)

### a. Roll conventions

| | CR-BF patch (2026-09) | CR-CZ cards (via CR-CP `cr_cp_lib.py`) |
|---|---|---|
| Rule | Calendar: **Monday of expiry week** (2026-09-14, expiry Fri 09-18), "the ADR's CME roll-date rule" | Data-driven: per session, **the symbol with the most RTH-window volume** in `es_flow_1m_databento` (`SYM = vol.groupby("d")…iloc[0]`) |
| Start boundary | Globex open before the roll Monday: **Sun 22:00 UTC** (2026-09-13 22:00) | Session-level (RTH window 06:15–13:00 PT); no ETH boundary |
| End boundary | **Expiry Friday 20:59 UTC** (2026-09-18 20:59) — beyond the dying contract's last minute (13:29 UTC), so the whole week is one source | n/a (next session) |
| Mechanics | Attended owner-role script `scripts/backfill_databento_es.py`: disable trigger → **DELETE** → INSERT → repopulate `es_minutes` → re-enable trigger. Not the unattended protocol (needs DELETE). Backups `*_bak_20260920` (dropped 2026-09-27). | — |

A third rule exists in code: CR-CA `roll_monday()` = third Friday − 11 days = **Monday of the week before expiry week** (2026-03-09); it defines which contracts `es_flow_1s_databento` *stores*, not the front. Not used by either CR above.

**The rules differ in definition but give the same date for both 2026 rolls measured:**

| Roll | Expiry | Monday of expiry week (CR-BF) | Volume crossover (CR-CP) |
|---|---|---|---|
| H6 → M6 | Fri 2026-03-20 | 03-16 | 03-16 (03-13 H 1.39 M / M 0.16 M; 03-16 H 0.57 M / M 0.83 M) |
| M6 → U6 | Fri 2026-06-19 | 06-15 | 06-15 (06-12 M 0.91 M / U 0.31 M; 06-15 M 0.43 M / U 0.74 M) |

Per the kickoff, a differing definition is a STOP: **Ryan picks.** Recommendation: CR-BF's calendar rule (it is the existing patched-window convention and gives an ETH boundary; the volume rule agrees on both dates).

### b. Windows on the dying contract

**2026-03 (H6 → M6).** Ironbeam bars and flow are ESH6 until its last minute, 2026-03-20 13:29 UTC (`ironbeam_es_flow_1s` H26 to 03-20 13:29:59; the 03-20 13:30 bar jumps +41.25), then ESM6.

- Dying-contract span: **2026-03-15 22:00 → 2026-03-20 13:29 UTC**: 6,326 stored minutes (both tables, identical values; `es_minutes` 6,776/6,776 equal to `ironbeam_es_1m_bars` over the full week).
- CR-BF-style window (to Friday 20:59 UTC): **2026-03-15 22:00 → 2026-03-20 20:59 UTC**: 6,900 Globex minutes; 6,776 stored (6,326 H6 + 450 already M6), **124 missing** (03-17 PT: 91, of which 50 RTH; 03-19 PT: 33 ETH).
- Offset (RTH window, Ironbeam − Databento ESM6 mean close): 03-16 −50.41, 03-17 −51.03, 03-18 −51.06, 03-19 −50.20. Ironbeam = ESH6 to the tick on 351–405 / 353–405 minutes per day.
- Volume ratio Ironbeam / Databento ESM6 (RTH window): 03-16 0.687, 03-17 0.183, 03-18 0.129, 03-19 0.115; 03-20 (M6) 0.980.

**2026-06 (M6 → U6).** Ironbeam switched to ESU6 at **Sun 2026-06-14 22:00 UTC** (flow U26 from 06-14 22:00; bar jump +121.75 over the weekend), i.e. exactly at the calendar roll; the volume crossover is also 06-15. 06-11 / 06-12 are on ESM6, which **is the front under both rules**. **No dying-contract window in June; nothing to patch.**

**2026-12 (Z6 → H7):** expiry Fri 2026-12-18, roll Monday 12-14 — has not happened (today 2026-10-28).

### c. Databento coverage and cost

`es_flow_1m_databento` holds ESM6 only for the RTH window [06:15, 13:00) PT, so it does not cover the ETH minutes. Databento GLBX.MDP3 `ohlcv-1m`, `ESM6` / `raw_symbol`, 2026-03-15 22:00 → 2026-03-20 21:00 UTC (metadata calls only, nothing fetched): **6,900 records — every Globex minute of the window — cost $0.0252.** Not fetched; awaiting approval.

### d. Tables holding bar prices for the window, and readers

| Object | Kind | In window | Action |
|---|---|---|---|
| `ironbeam_es_1m_bars` | table | 6,776 rows, all `source='ironbeam'` | patch |
| `es_minutes` | table (AFTER INSERT trigger copy) | 6,776 rows, equal to the above; RTH `bar_index` 0..n−1 per day (03-17: 340 rows, 0..339) | patch |
| `es_minutes_with_features`, `_bt`, `_v2` | views | — | self-heal |
| `ironbeam_es_1m_bars_repair_log` | CR-BZ log | 4 rows (2 bars × 2 tables, 03-19 19:13 / 19:58, symbol H26) | none; this CR logs its own rows |
| Live web readers (`Ironbeam/routes`, `callbacks`, `flow_v2`, `skew_stickiness`, `Bars`, `DayBrowser`, `Analogues`, `SetupV2`, `TodaySetup`, `gamma/*`; `packages/shared/level_placement.py`, `gex_landscape_api.py`, `vol_features.py`) | read on the fly, no stored output | — | self-heal. Flow v2/v3 chooses the contract whose closes match the chart bars (A1.4, `choose_symbol`) → switches H26 → M26 after the patch. Corrected levels for 03-17 → 03-20 use the prior day's 15:45 ET bar (H6 now) → move ~50 pts with the patch. |
| Crons `ironbeam_ingest.py`, `ironbeam_minute_snapshot.py` | writers, current minutes only | — | none |
| `ironbeam_es_flow_1s` | table | 167,090 rows `XCME:ES.H26` (to 03-20 13:29) | none — symbol-tagged; CR-BF/CR-BG precedent leaves them |
| `bt_daily_outcomes` | stored-derived (ES `session_*`) | **84 active rows** (21 trade dates 2025-12-19 → 03-20 × 4 feature versions) whose 60-session horizon includes a window session | **follow-up CR** (CR-BG pattern, `scripts/cr_bg_recompute_missed_roll.py`) |
| `bt_daily_features.vol_risk_premium` | stored-derived (ES daily closes, one-off `cr_d_backfill_vol_features.py`) | 23 non-null rows 03-16 → 04-17 | **follow-up** with the above |
| `bt_signals`, `es_minute_features` | — | 0 rows | none |

### e. Cards in the window

`setup_candidates` 2026-03-16 → 03-20: **12 cards, all CR-CZ / `cr-cz:cp-ranges`** (ids 49–60): 03-16 ×4 (49–52), 03-18 ×2 (53–54), 03-19 ×4 (55–58) — the kickoff's 10 on the dying contract — plus 03-20 ×2 (59–60, entries 09:26 / 09:46 PT, after Ironbeam's switch to M6; already on the chart). **0 reviews on all 12** (CR-CZ has 17 reviews in total, none in the window).

Smoke reference measured now: 2026-03-18 18:00–18:15 UTC (11:00–11:15 PT), Databento ESM6 low / high **6716.5 / 6737.5**; the chart (both tables) shows 6665.25 / 6686.0.

### Blockers for the "insert / update only" method

1. **`dash_backfill_writer` cannot set `source`.** It has INSERT/SELECT on both tables plus column UPDATE on `open, high, low, close, volume` only (CR-BZ grant). Tagging the 6,776 existing rows `source='databento'` needs an owner grant: `GRANT UPDATE (source) ON ironbeam_es_1m_bars, es_minutes TO dash_backfill_writer;`
2. **`es_minutes.bar_index` for the 50 inserted RTH minutes on 03-17.** The trigger gives an inserted RTH bar `MAX(bar_index)+1`, so filling a mid-session gap misorders the index. Renumbering 03-17 needs `GRANT UPDATE (bar_index) ON es_minutes TO dash_backfill_writer;` (or inserting into `es_minutes` first with explicit values, which still leaves the existing 340 rows to renumber).
3. **No `raw_symbol` column** in either table (CR-BF recorded the contract only in docs). Proposal: record it per bar in `ironbeam_es_1m_bars_repair_log.symbol` (`'ESM6'`), with old and new OHLCV, as CR-BZ did; inserted minutes logged with NULL old values.
4. **No DELETE needed:** Databento covers every Globex minute, so every stored minute is updated in place and the 124 missing ones are inserted.

CR-BF needed DELETE and ran attended under the owner role, outside the unattended protocol.

### Observation (out of scope)

`ironbeam_es_1m_bars` 2026-03-23 11:05–11:16 UTC moves +118.5 / +51.5 / +22.25 … in single minutes (M26, post-window). Not checked against Databento here.

### Decisions requested from Ryan

- **D1 Convention:** CR-BF calendar rule (recommended) or CR-CP volume rule — same dates in 2026-03 and 2026-06.
- **D2 Window end:** 2026-03-20 20:59 UTC (CR-BF style, one source for the week; recommended) or 13:29 UTC (dying contract only).
- **D3 Grants** (owner applies): UPDATE (source) on both tables; UPDATE (bar_index) on `es_minutes`. Without D3, the patch can fix OHLCV only and leave `source='ironbeam'`.
- **D4 Fetch:** $0.0252 Databento `ohlcv-1m` ESM6.
- **D5 June:** nothing to patch; confirm.
- **D6 Derived:** `bt_daily_outcomes` (84 rows) and `vol_risk_premium` as a follow-up CR.
- **B4:** confirm the June contract-identity reads.

## Amendment 1 — Ryan's decisions (chat 2026-10-28)

1. **Convention:** CR-BF calendar rule. Window **2026-03-15 22:00 → 2026-03-20 20:59 UTC** (bar starts, inclusive; 6,900 Globex minutes). June: nothing to patch.
2. **Grants, this run only** (applied by Ryan / the owner session, not by this CR's code): `GRANT UPDATE (source) ON ironbeam_es_1m_bars, es_minutes TO dash_backfill_writer; GRANT UPDATE (bar_index) ON es_minutes TO dash_backfill_writer;`. After the run and smoke: REVOKE the same three column privileges and confirm with `has_column_privilege`. Grant and revoke times are recorded here and in the run's `bt_backfill_runs.notes`. The CR-BZ OHLCV grants are unchanged.
3. **Databento fetch approved** ($0.0252): `ohlcv-1m` `ESM6` / `raw_symbol` for the window, plus 2026-03-23 10:35 → 11:36 UTC (±30 min of the 11:05 jump) in the same order; total must stay under $0.05, else stop and report.
4. **B4 deviation approved for the record:** contract-identity / volume aggregates in the holdout window (June roll check); no outcomes or setups read. Logged in `research/standing-rulings.md`.
5. **Repair log:** one row per (table, minute) in `ironbeam_es_1m_bars_repair_log`, `symbol = 'ESM6'` (the raw symbol), old and new OHLCV, as CR-BZ. Inserted minutes have NULL old values.
6. **2026-03-23 11:05 UTC jump:** compared against Databento and reported as a separate finding; not patched here.
7. **Follow-up (not this CR):** recompute the 84 `bt_daily_outcomes` rows and the affected `vol_risk_premium` rows (CR-BG pattern); list past research CRs that read 2026-03-16 → 03-19 bars. Recorded as an open question in the session note.

### Implementation order

- I1 `scripts/cr_dj_fetch_databento.py`: cost gate (< $0.05), fetch once, cache to `scripts/logs/cr_dj/` (git-ignored).
- I2 `scripts/cr_dj_roll_patch.py` (unattended protocol): preconditions (role, column privileges, 6,900 Databento minutes, every stored minute covered, no ETH minute in the 21:00 UTC maintenance hour); one transaction: log → guarded OHLCV + `source` UPDATE of the 6,776 stored minutes in both tables → INSERT of the 124 missing minutes (`es_minutes` first, with explicit `trade_date` / `is_rth` / `bar_index`, then `ironbeam_es_1m_bars`, whose trigger then does nothing) → renumber RTH `bar_index` 0..389 by time on the affected PT dates; after-state smoke in the same run; `update_run_smoke`; grant times in `notes`.
- I3 smoke + wrap.

## Finding F1 — 2026-03-23 11:05 UTC jump (Amendment 1 D6; not patched)

Databento ESM6 `ohlcv-1m` 2026-03-23 10:35 → 11:35 UTC (61 bars, `scripts/logs/cr_dj/databento_esm6_jump.parquet`) vs `ironbeam_es_1m_bars`:

- **Real move, not a bad print or gap.** Databento 11:05 bar: O 6510.50 H 6648.50 L 6509.75 C 6629.00, 13,541 contracts; then 11:06 C 6685.25, 11:07 H 6724.75, 11:14 H 6748.00. About +230 pts (≈ +3.5 %) in 10 minutes from 07:05 ET Monday, on 10–25× the prior minutes' volume; no missing minute.
- Ironbeam agrees on 58 / 61 closes and on every high; its 11:05 bar is a CR-BZ-type **TAIL fragment** (O 6517.50 / L 6517.25 vs 6510.50 / 6509.75, volume 12,796 vs 13,541), and several bars 11:08 → 11:28 carry less volume (e.g. 11:15 3,699 vs 4,841). Ordinary fragment differences; left as is.

## Fetch validation

- Databento marks 2026-03-15 and 03-16 (and 03-21) GLBX.MDP3 as `degraded`. Against the trades-built `es_flow_1m_databento` (RTH window, ESM6): 03-16 equal on 405 / 405 minutes in O, H, L, C and volume; 03-17 → 03-20 high / low 100 %, open / close 99.75 % (one minute a day), volume 98–99 %. 03-15 (Sun 22:00 → 23:59 UTC, 120 minutes) has no second source.
- 03-20 13:30 → 20:59 UTC (Ironbeam already ESM6): 450 / 450 minutes equal in O, H, L, C; volume ratio 1.000.

## Dry run (2026-10-28, `scripts/logs/cr_dj/dryrun.log`)

6,900 Databento minutes (ESM6 only, none in the 21:00 UTC hour); 6,776 stored per table (all `ironbeam`, `es_minutes` equal to the bars); 6,776 updates + 124 inserts per table (50 RTH, all on PT 2026-03-17 → renumber that date); preconditions OK except the three run-only grants. Before-state smoke: 6,326 minutes differ from Databento; 03-18 11:00–11:15 PT chart 6665.25 / 6686.0; CR-CZ cards 49–58 box edges 50.0–59.25 pts above the chart's closes, cards 59–60 (03-20, post-switch) 0.0.

## Run record (2026-10-29 UTC)

One scripted sequence (`scripts/logs/cr_dj/grant_run_revoke.py`, log `grant_run_revoke.log`; no connection string printed):

| Step | UTC | Result |
|---|---|---|
| Owner GRANT UPDATE (source) on both tables, UPDATE (bar_index) on `es_minutes` | **2026-10-29 01:24:43** | `has_column_privilege` false → true on all three |
| Patch run as `dash_backfill_writer` (`--grants-applied-at 2026-10-29T01:24:43Z`) | 01:24:43 → **01:50:24** | run **`2adcc833-5f7f-4406-b645-0ce1b7d61bbc`**, status `completed`; preconditions OK; 6,776 updated + 124 inserted per table; repair log 13,800 rows (`ESM6`, 124 × 2 with NULL old values); `bar_index` renumbered 390 rows (PT 2026-03-17); script exit 1 (in-run smoke, see below) |
| Owner REVOKE (same three) | **2026-10-29 01:50:25** | `has_column_privilege` false on all three; CR-BZ OHLCV grant unchanged (true) |

Times are in `bt_backfill_runs.notes` for the run.

**In-run smoke: FAILED on the smoke's own rule, not the data.** It rebuilt each CR-CZ box over bar starts [formed, qualified] inclusive, but `qualified_ts_pt` is the qualifying bar's **close** (`cr_cz_build_gallery.py`: `ts(d, r.q, close=True)`), so the box is [formed, qualified). Card 50 (top 6756.0 vs 6755.25) and card 54 (bottom 6717.0 vs 6725.0, the 11:15 bar close) failed. CR-CP's own source (`es_flow_1m_databento` ESM6) gives the same values as the patched `es_minutes` under both rules. Script fixed (`83938da`); `--smoke-only` re-run at 2026-10-29 01:51:30 UTC stored beside the original as `smoke_test_results.smoke_rerun`.

**Smoke (re-run): OK.**

| Check | Result |
|---|---|
| `ironbeam_es_1m_bars` vs Databento ESM6 | 6,900 / 6,900 minutes, 0 only-one-side, 0 OHLCV mismatches, all `source='databento'` |
| `es_minutes` vs Databento ESM6 | same: 6,900, 0, 0, all `databento` |
| RTH `bar_index` 03-16 → 03-20 | 390 rows a day, `bar_index` = rank on all; ETH rows −1 |
| 2026-03-18 11:00–11:15 PT (`es_minutes`, which the chart reads) | low **6716.5**, high **6737.5** (was 6665.25 / 6686.0) |
| CR-CZ cards 49–60 box edges vs patched closes | **0.00 on all 12** (49–58 were 50.0–59.25 before) |
| `source='databento'` rows in 2026 outside the CR-BF and CR-DJ windows | 0 |
| `bt_backfill_runs` | CR-DJ row with smoke results, self-assessment and grant / revoke notes |

Chart verified at the data layer (the price chart and REVIEW cards read `es_minutes` / `ironbeam_es_1m_bars` on each request); no local Dash server started (DB connection slots).
