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

