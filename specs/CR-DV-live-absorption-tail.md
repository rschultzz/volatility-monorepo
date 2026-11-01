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

## Step 0 findings (2026-10-31, read-only; no server started, CR-DR holds :8068)

Scripts: `scripts/cr_dv_step0/` (`db.py` read-only connection, `q_schema.py`, `q2_lag.py`, `q2b_tier.py`, `q345.py`,
`q5b.py`, `replay_live.py`); output in `scripts/logs/cr-dv/` (git-ignored). No writes.

### Q1 — where marks and cells come from

- **Endpoint** `GET /api/ironbeam/flow/cells` (`apps/web/modules/Ironbeam/callbacks.py`
  `_build_react_flow_cells_payload`) → `flow_v3.build_cells_payload` → `flow_v3._build`. Per segment (RTH; ETH in FULL view)
  `choose_tier` picks the cell source (DC17): `es_flow_price_1m` **only if its stored minutes cover ≥ 99 % of the
  segment's minutes with flow** (`PRICE_1M_COVERAGE`), else `ironbeam_es_trades` (exact, when the oldest retained trade
  ≤ segment start), else the 1-s table. Bars (`h`/`l` for hold / break) come from `ironbeam_es_flow_1s` via
  `flow_v2._segment_minutes`, written by the `ironbeam_flow_1s_aggregate.py` worker (~1 s loop, 2 s max lateness), so
  they include the minute still forming. Marks: `flow_heat.absorption_marks(cells, bars, seg.start, seg.end, K, M)`.
- **Poll:** `react_price_preview/src/App.jsx` refetches every `FLOW_V2_LIVE_POLL_MS` = **15 s** on the live trade date,
  only while a v3 price-pane layer is on, paused while the chart is being dragged. Live segments use an incremental
  cache (`_segment_cells`: cached to now − 2 min, the rest re-read each poll).
- **Consumers of the marks:** the chart triangles (`chartMarks`), the absorption levels (`absorptionLevels`, confirmed
  by "now" = end of the payload's last bar), and the box panel's Absorption row (`boxLevels` → `boxEvidenceRows`
  key `absorption`).
- **Other live readers of `es_flow_price_1m`:** none. Only `flow_v3` (`choose_tier`, `fetch_price_1m`) reads it in the app.

### Q2 — lag of `es_flow_price_1m` on 2026-10-30 RTH, and what is actually on screen

- The table has **no insert timestamp**. Readable time per minute = `completed_at` of the CR-DD-live run whose smoke
  `[first_minute, last_minute]` covers it (`bt_backfill_runs`, 95 runs 13:00–21:00 UTC, every ~301 s, ~2.3 s each).
  All 390 RTH minutes assigned. **Readable − minute end: median 272 s, p95 399 s, max 412 s** (min 139 s).
- **But the live endpoint never served those rows.** Simulating `choose_tier` at a poll 15 s after each minute close:
  stored minutes trail the flow minutes by 4–5, and 99 % coverage would need ≥ 100 × that many minutes, which RTH
  (390) never reaches. **All 390 polls → `trades` tier; 0 flips** (`q2b_tier.py`). The real `_build` replayed with
  every read cut to what existed at each poll (`replay_live.py`, trades by `received_at`, price_1m by run completion)
  confirms it: every poll was `trades`.
- So the cron lag is **not on screen**. The delay Ryan sees is the rule's own hold window: a mark at minute t is
  confirmed at t + (M + 1) min, i.e. 3 min after its minute closes at M 3.
- **Repaint found.** On a live call `end` = the segment end (13:00 PT), so `confirm ≤ end` always passes and the hold
  window only checks the bars that exist so far, including the forming minute; cells include the forming minute too.
  A mark can appear before its hold has elapsed and vanish on a later poll. Example: sellers 7633.50 at 15:39 UTC
  (08:39 PT) shown at the 15:40:00 poll, not in the final set.
- **Replay at a 15 s cadence (before any change)**, 1,580 polls 13:30:15 → 20:05 UTC (`replay_before_15s.txt`):
  1,521 polls `trades`, 59 `price_1m` (the tier check ran on the real `_segment_minutes`; the 59 are re-checked in the
  smoke). The full session has **5 marks**; live, **all 5 first appeared before their confirm time**: first seen −
  confirm_time median **−195 s** (p5 −207, p95 −183; min −210, max −180), i.e. while the cell's own minute was still
  forming. **19 marks were shown live that are not marks on the full session** (shown on 1–5 polls each, e.g.
  sellers 7633.50 at 15:39 on 2 polls from 15:40:00). Live marks are not late; they show up early and get repainted.

### Q3 — trades → cells reproduce the cron rows

Same SQL as the cron (`floor(price*4)/4`, BUY / SELL sums, `count(*)` incl. side `'0'`), `XCME:ES.Z26`, full outer
join with `es_flow_price_1m` (`ironbeam`):

| window | trades used | cells equal | only one side | differ |
|---|---|---|---|---|
| Globex day 10-30 [10-29 22:00, 10-30 21:00) UTC, 1,380 min | all retained | 13,991 / 13,992 | 0 | 1 (one minute; a print after its minute was stored) |
| same | received ≤ minute end + 15 s | **13,992 / 13,992** | 0 | 0 |
| same | received ≤ minute end + 5 s | 13,990 / 13,992 | 0 | 2 |
| RTH 10-30, 390 min | all / ≤ end + 15 s | **6,266 / 6,266** | 0 | 0 |

0 off-grid prices; one `'0'`-side print (126 contracts) that day, in `trade_count` only. RTH lateness
(`received_at − ts_utc`) p50 0.47 s, p99 1.2 s, p99.9 2.0 s, max 7.3 s; 0 RTH prints later than 15 s after their minute.
**Yes**, exactly: a closed minute built from trades 15 s after it closes equals the cron row.

### Q4 — query cost (`fetch_trade_cells` SQL, index `(symbol, ts_utc DESC)`, Index Scan)

| span | rows | wall (median / max of 5, from the laptop) |
|---|---|---|
| 8-min tail, 10:00–10:08 PT | 206 | 60 / 60 ms (server execution 6.4 ms) |
| 8-min tail at the open | 375 | 66 / 119 ms |
| 15-min tail at the open | 646 | 121 / 193 ms |
| whole RTH (a live cache miss) | 12,043 | 1,141 / 1,577 ms |

Under the 200 ms target for any tail ≤ 15 min. The live cache already reads only ~2–3 min per poll.

### Q5 — stall threshold

RTH gaps between seconds with volume in `ironbeam_es_flow_1s` (the day's main contract), 84 sessions 2026-07-06 → 10-30:
p50 1 s, p99 3 s, p99.9 6 s, p99.99 14 s. Healthy sessions' max gap is 3–22 s. Every gap containing a whole minute
where `ironbeam_es_1m_bars` still traded was a **stall**, and the shortest such gap was **77 s** (10-23 7,408 s; the 09-22
"130 min" is a run of 77–108 s gaps, a degraded feed, not one hole). Labor Day 09-07 (thin, real quiet) max 35 s.
Gaps 60–68 s without a whole minute inside can't be classified from bars (about 10 in 84 sessions).
**Proposal N = 60 s** during RTH: catches every classified stall, never fires on the quiet holiday.

### Q6 — symbol

The live path picks the segment contract with `flow_v2._pick` (A1.4: the contract whose 1-s-built minute closes equal
the chart's `ironbeam_es_1m_bars` on ≥ 90 % of minutes, else the highest volume), then reads cells for that symbol.
The cron writes **every** symbol in `ironbeam_es_trades` (no front choice; `GROUP BY symbol`), Ironbeam format
`XCME:ES.Z26`. Reading the chosen symbol from either source gives the same contract, including roll weeks.

## Amendment 1 (Ryan, chat 2026-10-31, after Step 0; before any implementation)

Ryan, verbatim:

```
Spec amendment, from Ryan via chat, before any implementation.
Step 0 shows the live endpoint already reads ironbeam_es_trades, so the cron lag isn't on screen. The lateness Ryan sees is the 3-min hold window. New scope:

1. Fix the repaint. The hold-window check and the mark candidates use closed minutes only. A confirmed mark never appears before its hold window has fully elapsed. Test it on the 15:39 case.
2. Provisional marks. When a minute closes with qualifying volume at a price (the CR-DC volume test passes), draw a hollow / outlined mark in the same colour, marked provisional in its tooltip. It turns into the normal solid mark when the 3-min hold passes, or is removed if price breaks the level first. The rule and thresholds are unchanged. Hollow vs solid is the only visual difference, and it's approved here (visible-change discipline).
3. Keep the feed-stall flag (N = 60 s) as specced.
4. Drop the "server-side live tail" item as built. Keep only whatever is needed for closed minutes to be served within about 15 s of closing, and confirm that already holds.
Amend the spec and session note first, re-check the rulings table (C1 colours: hollow is a style, not a new colour), commit, then implement. Measure in the smoke: time from minute close to provisional mark, and to confirmed mark.
```

### Amended implementation (replaces Implementation 1–5 above)

1. **Repaint fix (server, `flow_heat.absorption_marks` + `flow_v3._build`).** On a segment that is not complete, marks
   are computed with `end` = the last **closed** minute boundary: `floor((now − 3 s) / 60) × 60`, capped at the
   segment end (DV1). Cells and bars at or after it (the forming minute) are not read by the rule. So a confirmed
   mark has `confirm_time ≤` the last closed minute, and a break is seen on closed bars only. Complete segments and
   `until` replays: `end` unchanged (past sessions unchanged). Test: the 10-30 15:39 UTC sellers 7633.50 case.
2. **Provisional marks.** Same volume test, same per-minute extreme price, same thresholds (K 3, M 3, 15 prior
   cells). A candidate at closed minute t whose hold window has not fully elapsed (`t + (M+1) min > end`) and that
   no closed bar since t has broken (sellers: low < P − 1 tick; buyers: high > P + 1 tick) is returned with
   `provisional: true` (confirmed marks `false`). If a closed bar breaks it before confirmation it is dropped. A
   provisional candidate within 1 tick of an earlier unbroken mark of the same side is not drawn and adds no volume
   until it confirms (then it merges, as today). Chart: provisional = **outlined (unfilled) triangle, same slate
   colour**, same slate line; tooltip first line ends "· provisional" and a line "provisional — hold window ends
   HH:MM". Provisional marks feed neither the absorption levels nor the box panel (they need confirmed marks).
3. **Feed-stall flag.** For the live RTH segment, when now is inside RTH and the segment contract's last trade in
   `ironbeam_es_trades` (ordered read on `(symbol, ts_utc DESC)`) is more than **N = 60 s** old, the payload carries
   `feed_stalled_since` (epoch of that last trade). The box panel's Absorption row then reads
   "flow feed stalled since HH:MM PT" (existing row style; text only).
4. **No live-tail code.** Closed minutes are already served from `ironbeam_es_trades` on every 15 s poll (Q2, Q3);
   the smoke confirms it from the replay.
5. **Tests:** Python — no confirmed mark before its hold has elapsed on every minute + 15 s poll of the recorded 2026-10-30 RTH session (fixture `cr_dv_trades_1030_rth.npz`, read as of each poll)
   (incl. the 15:39 case); provisional appears at the candidate's close, becomes confirmed at confirm time or
   disappears on a break; complete-segment marks byte-identical to before; stall flag fires on a fixture gap
   > 60 s and not on quiet seconds ≤ 60 s or outside RTH / on past sessions. JS — `chartMarks` passes `provisional`;
   `absorptionLevels` and the box levels skip provisional marks; the absorption row shows the stall text.
6. **Smoke + wrap:** replay 2026-10-30 at a 15 s cadence (trades by `received_at`): minute close → provisional
   mark, and → confirmed mark (median, p95), before vs after; repaint count before vs after. Then a browser check
   once CR-DR releases :8068. Session note, logs to the main checkout's `scripts/logs/cr-dv/`, PR.

### Decisions in the amendment (Code's calls, to confirm in review)

| # | Decision | Why |
|---|---|---|
| DV1 | A minute is "closed" for the rule 3 s after its end | RTH lateness p99.9 2.0 s (Q3); the 1-s worker finalises seconds after 2 s. A 15 s poll then shows a closed minute 3–18 s after it closes |
| DV2 | Provisional breaks are checked on closed bars only | the same bars the confirmed rule uses; no flicker from the forming minute |
| DV3 | `feed_stalled_since` only on the live RTH segment, inside 06:30–13:00 PT | Q5's N was measured on RTH; ETH is thinner |

### Standing rulings conformance (re-checked for amendment 1)

| Ruling | Status | Note |
|---|---|---|
| A1–A22 | n/a | the CR-DC rule and thresholds are unchanged; provisional = the same test before its hold completes |
| B1–B3, B6–B8 | n/a | no verdicts, no parameter search, no labels |
| B4 | honoured | as above; the smoke replays 10-30 (live forward log, not holdout-research) mechanically: lags and counts only |
| B5 | n/a | Ironbeam live path only |
| C1 (palette) | honoured | hollow vs solid is a style of the existing slate mark, not a new colour; stall text in the row's existing style |
| C2 (off by default) | honoured | Flow v3 layers stay off by default; approved visible change limited to hollow marks + stall text |

Revoked / superseded R1–R7: none used.

## Smoke (2026-10-31, weekend: 2026-10-30 replayed)

### Replay at a 15 s cadence (`scripts/cr_dv_step0/replay_live.py`, the real `flow_v3._build` with every read cut to the poll)

1,580 polls 13:30:15 → 20:05 UTC; trades by `received_at`, `es_flow_price_1m` by CR-DD-live run completion, 1-s flow
by time. The full session has 5 marks (14:05, 15:20, 15:43, 17:51, 18:16 UTC).

| | before (`origin/main`) | after items 1–3 | after item 4 |
|---|---|---|---|
| final marks first drawn **before** their hold ended | **5 / 5** (median 195 s early, i.e. while the cell's own minute was forming) | 0 / 5 | 0 / 5 |
| marks drawn as confirmed that are not marks on the full session | **19** (1–5 polls each) | **0** | **0** |
| provisional candidates removed (broken before their hold ended) | — | 10 (3–4 polls each) | 10 (3–7 polls each) |
| minute close → provisional mark (median / p95) | — | **15 s / 15 s** | **15 s / 15 s** |
| minute close → confirmed mark (median / p95) | ≈ −15 s (shown at once, then repainted) | **195 s / 195 s** (= M 3 min hold + 15 s) | **195 s / 195 s** |
| hold end → confirmed mark | ≈ −195 s | **15 s / 15 s** (one poll) | **15 s / 15 s** |
| polls on the `price_1m` tier | 59 (18:30:45 → 20:05) | 59 | 13, all from 20:02:00 (session complete: DC17 order, expected); 0 during RTH |

The 15 s figures are the poll cadence: each mark shows on the first poll after its minute closes (3 s grace, DV1).
The two 19:59 candidates stay provisional until the session is complete (20:02) and then drop: their hold window would
end after 13:00 PT, and the unchanged CR-DC rule never confirms those (confirm ≤ segment end).
Logs: `scripts/logs/cr-dv/replay_before_15s.txt`, `replay_after_15s.txt`, `replay_after2_15s.txt` (+ `*_detail.json`).

**Correction to Step 0 Q2.** `q2b_tier.py` said the live endpoint never picks `price_1m`. The replay of the real code
shows **59 polls from 18:30:45** (late in RTH the stored minutes reach 99 % of the flow minutes). On those polls the
newest 3–8 closed minutes were missing, so "closed minutes served within ~15 s" (amendment item 4) did not hold.
**Item 4 fix** (`5641805`): a live Ironbeam segment reads `ironbeam_es_trades` whenever trades cover it; complete
sessions keep DC17's order (es_flow_price_1m first). None of the 5 final marks fell after 18:30, so the before / after
mark lags above weren't affected by this.

### Browser check (smoke backend `scripts/cr_dv_smoke_backend.py` on :8068, simulated now on 2026-10-30)

The backend's `/__crdv/sim?now=…` route makes the cells endpoint answer as if now were that moment (same read cuts as
the replay; `stall_from` injects a gap). Frontend: a `vite build` of this branch served by the same backend (the
folder already had 5 dev servers from other chats). Run before the item-4 fix; all checked moments are 15:3x–15:5x UTC
on the trades tier either way.

- **Hollow → solid:** buyers 7631.75 at 15:43 UTC (08:43 PT) is drawn outlined at 15:44:15 with tooltip
  "Buyers absorbed · 7631.75 · provisional / provisional — hold window ends 08:47"; at 15:47:15 it is solid and the
  tooltip has no "provisional". Canvas pixels at the mark: 14 slate pixels in the centre (outline) vs 66 for a solid mark.
- **Hollow → removed:** buyers 7628.25 at 15:46 is outlined at 15:47:15 ("hold window ends 08:50") and gone at 15:48:15
  (before the change it showed as a confirmed mark on 5 polls).
- **The 15:39 case:** sellers 7633.50 is absent at 15:40:00, outlined at 15:40:15 and 15:40:45 ("hold window ends
  08:43"), gone at 15:41:15. It is never drawn solid (before: solid from the 15:40:00 poll).
- **Stall text:** `stall_from` 15:30:00, now 15:35:15, Edges on, anchor = session open: the current box's Absorption row
  reads "sellers absorbed 7640.00 (08:20) at the bottom · flow feed stalled since 08:29 PT" (08:29 = the last trade
  before the gap); frozen boxes' rows unchanged. Same moment without the gap: "none at the box". No console errors.
- Servers stopped afterwards (`preview_stop`; :8068 free).

### Suites

Python: `test_flow_heat` 24 / 24, Ironbeam suite 108 / 108 (incl. `test_flow_cells` 21 / 21), shared `test_flow*`
56 / 56. JS unit 111 / 111.

## Pre-merge gate and Amendment 2 (Ryan, chat 2026-10-31)

**Gate 1 (Ryan):** warm p95 < 300 ms after 12:00 PT, or don't merge. Measured (`scripts/cr_dv_step0/time_endpoint.py`, real
DB reads cut at the simulated now, 10-30 polls every 15 s 12:00:15 → 13:00 PT, warm caches, laptop timing):
PR p50 554 / p95 755 / max 905 ms, cold 8.6 s; `origin/main` p50 625 / p95 668 / max 865 ms, cold 2.1 s. **Failed; not merged.**
Warm poll split (mean 630 ms): DB 398 ms (`fetch_flow_minutes` 180, `fetch_bar_closes` 108, `fetch_trade_cells` 58,
`_trades_last` 51), compute + JSON 232 ms.

Ryan, verbatim:

```
Gate revised (Ryan via chat): the 300 ms limit was set without a baseline. New gate, same harness, polls 12:00–13:00 PT on 10-30:
- warm p95 ≤ origin/main's warm p95 (668 ms);
- cold first poll ≤ 2.5 s.

Amendment before code (amend the spec and session note first):
1. Replace "live always reads trades" with the original splice: stored es_flow_price_1m minutes up to its last fully written minute + trades for every closed minute after it. Drop the 99 % coverage switch entirely. Re-run the replay and confirm the 59 dropped-minute polls stay at 0 and marks are unchanged.
2. Cache the 1-s flow minutes and bar-close reads incrementally (re-read only the newest minutes), the same pattern as the trades cache.

Re-time both, then report the table vs main. If both gates pass, merge PR #128 with a merge commit, confirm on origin/main, report the SHA, and clean up. Otherwise stop and report.
```

### Amended implementation (replaces item 4's "live always reads trades")

1. **Splice (live Ironbeam segments, tier `splice`, badge "exact: es_flow_price_1m + trades").** W = the end of the last
   minute stored in `es_flow_price_1m` (`ironbeam`, the segment's symbol) inside the segment (the cron writes a minute
   whole in one INSERT, so every visible minute is complete); no stored minute → W = segment start. Cells = stored
   rows for minutes < W + `ironbeam_es_trades` for every **closed** minute in [W, last closed minute) (3 s grace, DV1).
   Stored rows are immutable (insert-only), so they are cached and each poll reads only rows from the cached W on;
   the trades tail [W, closed) is read every poll (3–8 min). No 99 % coverage test on live segments.
2. **Incremental 1-s flow minutes and bar closes (live only).** Both cached up to now − 2 min (`LIVE_RECHECK_MIN`, as
   the cells) and re-read from there each poll; the contract choice (`flow_v2._pick`) runs on the merged rows.
3. Unchanged: complete segments keep DC17's order (es_flow_price_1m if it covers ≥ 99 %, else trades, else 1-s), so
   past sessions are unchanged.

### Decisions (amendment 2)

| # | Decision | Why |
|---|---|---|
| DV4 | "Drop the 99 % switch" applies to live segments; complete segments keep DC17 | the kickoff says past sessions unchanged; DC17's partial-coverage fall-through (e.g. 10-26 before 04:27 UTC) still needs it there |
| DV5 | The minute still forming is no longer in a live segment's cells | "trades for every closed minute" (and kickoff item 1). Consequence: the heat / profile no longer draw the forming minute's partial column; it appears on the first poll after the minute closes |
| DV6 | Mark-appearance lag and the stall flag are unchanged by the splice | stored minutes equal the trades cell for cell (Q3); the stall read is separate |

Re-check of the rulings table: unchanged (no rule, colour or default changes; DV5 removes a partial column, adds nothing).

### Gate ruling (Ryan, chat 2026-11-01)

"Absolute laptop timings are network-bound; the gate is branch vs main timed in the same run." On the concurrent run the
branch **passes**: warm p95 2,047 vs 2,562 ms; cold 4.24 vs 4.08 s (accepted). **Known limit (accepted):** with no
stored `es_flow_price_1m` rows for the session yet (cron not run, or a fresh session), the splice's first poll reads the
whole session's trades — cold 11.4 s in the cron-lag-modelled run (B); later polls are incremental. Approved to merge.
