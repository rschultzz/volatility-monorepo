---
type: cr
cr_id: CR-DV
title: Live absorption marks without the cron lag (live read path only)
status: in-progress
run_mode: interactive
data_safety_class: read_only
branch_name: feat/CR-DV-live-absorption-tail
dependencies: [CR-DC, CR-DD, CR-DS]
tags: [dash, cr, order-flow, absorption, live, flow-v3]
---

# CR-DV — Live absorption marks without the cron lag

## Kickoff (Ryan, chat 2026-10-31, verbatim)

```
CR-DV — live absorption marks without the cron lag

Kickoff
1. git fetch && git checkout -b feat/CR-DV-live-absorption-tail origin/main. This is the first action, before any reads, in its own worktree.
2. CR-DR is running and holds the DB-backed server on :8068. Do Step 0 now without starting a local backend (code reading and read-only DB queries only). Don't start any server until CR-DR reports it has released :8068. One DB-backed server at a time.
3. Read:
   - CLAUDE.md
   - Dash/components/field-conventions.md (es_flow_price_1m and ironbeam_es_trades rows)
   - specs/CR-DC-*.md (the absorption rule)
   - specs/CR-DD-*.md (es_flow_price_1m)
4. Write the session note and freeze the spec at specs/CR-DV-live-absorption-tail.md.

Goal
On the current session, absorption marks appear within about 15 s of their minute closing, not 3–8 min. Today they come from es_flow_price_1m, which a cron fills every 5 min for minutes ≥ 3 min old. Ironbeam trade lateness is p99.9 5 s, so the delay is the write path, not the data.

Step 0, diagnosis (answer all of these in the Step-0 commit; then STOP and report)
Q1. Where does the frontend get absorption marks and price-level cells (endpoint, server function, table)? How often does the panel poll? Does anything else read es_flow_price_1m live?
Q2. Measure the current lag on the last full session (2026-10-30 RTH): for each minute, the time it became readable in es_flow_price_1m minus the minute's end. If there's no insert timestamp, say so and derive it from the cron schedule. Report median, p95, max.
Q3. Can cells built from ironbeam_es_trades for a closed minute reproduce the cron's es_flow_price_1m rows exactly (0.25 grid flooring, side mapping, unknown side counted in trade_count only, symbol format)? Check cell for cell on the latest RTH session still in ironbeam_es_trades (2-day retention), using the same method as CR-DD.
Q4. Query cost: time to build the live tail (every closed minute after the last stored one) per poll, using the (symbol, ts_utc DESC) index. Worst case is right after a cron run is skipped (about 8 min of tail). Target < 200 ms.
Q5. Stall detection: propose N, so that "no trades for more than N s during RTH" means the feed has stalled. Use the gap distribution in recent sessions; known stalls are 2026-10-23 07:52–09:56 PT and 09-22 (130 min).
Q6. Symbol: how does the live path pick the front contract today? Must match whatever the cron uses.

Implementation (one commit per item)
1. Server-side live tail. For today's session only, the cell data returned is the stored minutes from es_flow_price_1m, plus every closed minute after the last stored one built from ironbeam_es_trades, tagged source='live_tail'.
   - The minute still forming is excluded.
   - Stored rows always win over live-tail rows for the same minute.
   - Past sessions are unchanged.
2. Absorption marks run unchanged on the merged cells (same CR-DC rule, K 3 / M 3). No change to the rule or its thresholds.
3. Feed-stall flag. When the trades feed is stalled (Q5), the box panel's absorption row reads "flow feed stalled since HH:MM PT" rather than showing nothing. That text is the only visible change; existing row styles; no other layout or colour changes (visible-change discipline).
4. Tests:
   - the live tail equals the cron rows on a replayed session;
   - the forming minute is excluded;
   - stored rows take precedence;
   - the stall flag fires on a fixture gap and not on normal quiet seconds.
5. Smoke + wrap.
   - The mark-appearance lag before vs after (median, p95). On a weekend, replay 2026-10-30 by simulating "now" minute by minute against the stored trades if they're still retained; otherwise say it waits for Monday's live session and smoke the rest.
   - Session note, copy the logs to the main checkout's scripts/logs/cr-dv/, then the PR. Report the PR number and head SHA.

Out of scope: the cron, its schedule, the stored tables, the absorption rule. Live read path only. No writes to any table.
```

## Standing rulings conformance

Live display read path; no test, outcome, verdict or feature statistic is computed.

| Ruling | Status | Note |
|---|---|---|
| A1–A22 | n/a | no setup, level, entry or exit logic; the CR-DC absorption rule is unchanged |
| B1–B3, B6–B8 | n/a | no verdicts, no parameter search, no labels |
| B4 (data split) | honoured | reads only `ironbeam_es_trades` (retained 10-29 → 10-30), `es_flow_price_1m` (`ironbeam`), `ironbeam_es_flow_1s` / `ironbeam_es_1m_bars` (gap counts only) and `bt_backfill_runs`; Step 0 counts are mechanical (lag, equality, gap lengths); no Databento row after 2026-06-05 is read |
| B5 (Databento reference) | n/a | Ironbeam live path only; no pooling |
| C1 (palette) | honoured | the stall text uses the absorption row's existing style; no new colour |
| C2 (off by default) | honoured | Flow v3 layers stay off by default; no other view changes |

Revoked / superseded R1–R7: none used.
