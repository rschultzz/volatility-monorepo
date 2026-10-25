---
type: cr
cr_id: CR-CC
title: Setup Review Mode on the Price Chart — load candidate setups, replay each one to its entry, label Take / Skip / Unsure with reasons, then reveal the outcome
status: created
run_mode: interactive
size: medium
estimated_days: 2-3
data_safety_class: writes_reviewed
dependencies: [CR-BV, CR-BZ]
soft_dependencies: [CR-CB]
depended_on_by: []
branch_name: feat/CR-CC-setup-review-mode
last_reviewed: 2026-10-25
related_decisions: []
related_open_questions: []
related_sessions: []
tags: [dash, cr, created, review, labeling, discretion, price-chart, frontend]
---

# CR-CC — Setup Review Mode on the Price Chart

## Why

**Ryan (2026-10-25):** the goal is rules for his discretion, so that mid-session he doesn't convince himself of patterns that aren't real, and so that he stops taking the "no" setups he sees all the time.

**The problem.** The order-flow and level tests (CR-BU/BW/BX/BY) never showed Ryan the setups they traded. A test may have refuted a definition that didn't match his eye.

**What this CR adds.** Every pattern CR gets a visual review step:
- Ryan labels candidate setups **blind to the outcome**, on the same chart he trades from;
- his labels tighten the rules;
- then outcomes are compared for Take vs Skip, which is the real-vs-imagined check on his own discretion.

**Existing machinery to reuse.** The Backtests trade log already sends the price chart to a trade:
- `react_backtests_v2/src/components/TradeLog.jsx` POSTs `/api/trade-log/annotation-state`;
- `PriceChart.jsx` polls it every 1.5 s and opens `TradeAnnotationPanel` on that date.

The state is in-memory (`_annotation_state` in `apps/web/modules/TradeLog/routes.py`). This CR adds a persistent review queue next to it, rather than overloading it.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | (a) two new tables; (b) API routes; (c) a REVIEW panel and replay mode in `react_price_preview`; (d) a loader for candidate JSON files. No change to existing routes' default responses (byte-identical tests as in CR-BT/BV), and none to research tables. |
| L2 | Tables | `setup_candidates` (id text PK `<cr>:<set>:<n>`, cr_id, set_name, session_date, cell, direction, level_price, level_net_bn, level_gross_bn, entry_ts_pt, entry_price, stop_price, extreme_price, test_ts_pt, confirm_ts_pt, retest_ts_pt, meta jsonb, outcome jsonb, loaded_at). `setup_reviews` (candidate_id FK, label in {take, skip, unsure}, reasons text[], comment text, reviewed_at, revealed_at). Insert-only, plus UPDATE of `revealed_at`. DDL applied once by Code with `DATABASE_URL` after Ryan's go; grants to the web role as needed. |
| L3 | Loader | `scripts/load_setup_candidates.py` reads a gallery JSON (format below) into `setup_candidates`. `outcome` (P&L / race result) is stored in the table and **never** sent to the frontend before a review exists for that candidate. |
| L4 | API | `GET /api/setup-review/candidates?cr=&set=&cell=&status=unreviewed\|reviewed\|all` (list, no outcomes). `GET /api/setup-review/candidates/<id>` (detail; includes `outcome` only if reviewed). `POST /api/setup-review/candidates/<id>/review` (label, reasons, comment). `POST …/<id>/reveal` (sets `revealed_at`, returns the outcome). `GET /api/setup-review/summary?cr=&set=` (counts, Take vs Skip outcome stats, reasons tally). |
| L5 | Replay mode | `/api/ironbeam/bars` gains an optional `until=<ISO PT>`: bars after it are omitted, and so is the live overlay. Absent = today's behaviour, byte-identical. The flow route gets the same `until`. The landscape and levels are prior-day (no lookahead) and stay as they are. |
| L6 | REVIEW panel | A REVIEW chip next to SMILE / SIGNALS / GEX / ANALOGUES opens a side list, filterable by CR / set / cell / status. Clicking a candidate (or **N** / **P** for next / previous):<br>• loads that session in **Corrected** levels mode, with `until = entry_ts_pt`;<br>• draws the level (line + label with net/gross $bn), the extreme H/L, the entry and the stop as price lines, and markers at the test / confirm / retest / entry times;<br>• shows metadata (cell, overshoot, any CR-specific fields).<br>No P&L or post-entry bars are shown. |
| L7 | Labelling | Buttons **Take / Skip / Unsure** (keys **T / S / U**). Reason chips, multi-select, with a free-text "add reason" that becomes a chip for later candidates: overshoot too big, no clean extreme, level too weak, trend day, chop, too late in day, flow against, other. A comment box. Save is required before reveal. |
| L8 | Reveal | After Save, a **Reveal** button (key **R**) re-fetches without `until` (the full session), draws exit markers and the outcome from `outcome` (e.g. +34 pts / stopped −3.5), and records `revealed_at`. A review can't be edited after reveal; a new review can be added and is flagged as post-reveal. |
| L9 | Summary view | In the panel: per CR/set, counts by label, and for revealed candidates the mean outcome for Take vs Skip vs Unsure, with n, plus the reasons tally on Skips. Chat reads `/summary` (or the tables) to propose rule amendments. |
| L10 | Gallery JSON format | `{cr_id, set_name, generated_at, events: [{id, session_date, cell, direction, level_price, level_net_bn, level_gross_bn, entry_ts_pt, entry_price, stop_price, extreme_price, test_ts_pt, confirm_ts_pt, retest_ts_pt, meta: {...}, outcome: {exit_ts_pt, exit_price, pnl_pts, result, by_exit: {...}}}]}`. Bars are **not** embedded: the chart loads them from the DB with `until`. |
| L11 | Visible-change discipline | Only the REVIEW chip and panel, replay truncation when a candidate is active, and the candidate price lines and markers. Nothing else changes. |

## Step 0 — diagnosis gate

1. **Annotation path.** Confirm how the trade log drives the chart (annotation-state poll, the date switch) and say whether REVIEW should reuse that poll or set the date directly in the price app. It is the same app, so a direct state change is expected.
2. **`until` feasibility.** Where `/api/ironbeam/bars` builds bars and segments, and the smallest change that truncates them, including the live overlay and the flow route. The byte-identity plan.
3. **Price-line API.** Confirm lightweight-charts 5 `createPriceLine` and series markers for the candidate overlays, and that they don't collide with the existing GEX segments.
4. **DDL** text and roles: which DB role the web service uses, and the grants needed.
5. **First data:** whether `research/gallery/cr-cb-stage1.json` (and `cr-bu.json`, `cr-bw.json`) exist yet; if so, validate them against L10.

## Implementation order (one commit each)

1. DDL and loader (L2, L3, L10), with tests.
2. API routes (L4), with tests: outcome never present before review.
3. `until` replay (L5), with byte-identity tests.
4. REVIEW panel, labelling and reveal (L6–L8).
5. Summary view (L9).
6. Smoke: load the CR-CB gallery (or a 5-event fixture if it isn't ready); Playwright walks label → reveal on 2 candidates; screenshots.

**Deploy:** "Dash - MonoRepo" only.

## Kickoff prompt for Claude Code

```
Start CR-CC in its own git worktree: git fetch && git worktree add <scratch>/crcc-wt -b feat/CR-CC-setup-review-mode origin/main (confirm CR-CC is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CC — Setup Review Mode on the Price Chart.md", components/field-conventions.md, specs/CR-BV-landscape-chart-one-space.md, specs/CR-BZ-live-tick-clean-minute-bars.md, and the TradeLog module (apps/web/modules/TradeLog, react_backtests_v2/src/components/TradeLog.jsx, react_price_preview TradeAnnotationPanel/PriceChart annotation code).
Commit 1: spec freeze to specs/CR-CC-setup-review-mode.md, verbatim. Commit 2: Step 0 findings (all five). No DB writes, no DDL yet. Stop and report.
```

---

## Step 0 findings (2026-10-25)

Branch `feat/CR-CC-setup-review-mode` off `origin/main` `59e2fa8` (the CR-BZ merge, PR #77), in its own worktree. CR-CC was unused: no branch, no spec, no commit; only the vault note. The L1–L11 table is still marked "Ryan to confirm"; it was frozen as written, since the kickoff asked for the freeze.

**Method.** Code read at `59e2fa8`. DB reads only: catalog queries (`pg_roles`, `pg_auth_members`, grants, default ACLs), with `default_transaction_read_only=on`. No DDL, no writes, no Render changes, no UI run. The existing bars golden test (`test_bars_levels_toggle`, 9 tests) passes unchanged on this base.

**Verdict: no stop.** Nothing contradicts L1–L11 outright. Six details need Ryan's decision before implementation; they are listed under "Decisions needed". The two that matter most:
- **D1.** The price app's date is a URL parameter, read once, not state.
- **D3.** Post-entry information can reach the screen through more than bars and flow.

### Q1 — Annotation path: date switching is done by Dash, not by the annotation poll

**How the trade log drives the chart today.** `TradeLog.jsx` `handleViewOnChart` makes two POSTs:
1. **`POST /api/backtests-v2/select-trade`** (trade date, setup/entry/exit HH:MM).
   - Dash's `bt2-selection-poll` interval (750 ms, `app.py:471`) reads it via `get_backtests_v2_selection_since`.
   - It then sets `trade-date`, the smile time list, the bar interval and the main tab (`app.py:623–632`).
   - The iframe callback `ib_update_react_preview_src` (`Ironbeam/callbacks.py:4541`) rebuilds the iframe `src` with `?trade_date=…`, which reloads the price app.
2. **`POST /api/trade-log/annotation-state`** (trade fields).
   - `PriceChart.jsx:1062` polls `GET /api/trade-log/annotation-state` every 1.5 s and opens `TradeAnnotationPanel` when the state is non-null.
   - An effect at `:1081` clears the state when the chart's `tradeDate` differs from the trade's.

So the **date switch is the Dash → iframe-reload path**. The annotation state only opens the panel. It is a single global in-memory dict in the Flask process (`routes.py:27`): it does not survive a restart, and with more than one gunicorn worker it would not be shared.

**In the price app the date is not state.**
- `App.jsx:436–438` reads `trade_date` and `expiration_date` from `window.location.search` once, in a `useMemo` with `[]` deps.
- Every data effect is keyed on that constant (bars `:1209/1266/1442`, flow `:1345/1584`, landscape `:877`, analogues `:936`, smile/skew `:1649/1711`, condor `:767`).
- **"Set the date directly in the price app" therefore needs `tradeDate` lifted into React state** (seeded from the URL; REVIEW sets it). The effects already list `tradeDate` as a dependency, so they refetch on change without edits.
- What it does not do: tell Dash. Dash's date picker and smile time list stay on the URL date while a review is active (D1).

**Recommendation.** REVIEW does not reuse the annotation-state poll. That poll is global and in-memory, adds 1.5 s of latency, and needs a Dash round trip and an iframe reload per candidate. Instead, REVIEW sets App state directly: `reviewDate` overrides the URL date while a candidate is active, and closing REVIEW restores the URL date. The alternative is a full reload per candidate via `window.location.replace('?trade_date=D&review=<id>')`, which costs a reload (seconds) per N/P. See D1.

### Q2 — `until` feasibility

**Bars: `/api/ironbeam/bars` → `_build_react_preview_bars_payload` (`callbacks.py:1035`).**
- Per loaded date it calls `_fetch_bars_pt(day_start_pt, day_end_pt, …)` (`:709`), which selects 1-minute rows `datetime >= start AND datetime < end` (bar start, UTC) and resamples to 5 min when asked.
- Everything else is derived from those frames:
  - payload `bars`;
  - `gex_segments` (`_build_gex_segments_for_session`, `:924`);
  - centre `gex_levels`;
  - `expected_move_levels`.
- **Smallest change:** clamp `day_end_pt = min(day_end_pt, until)` before the fetch, and skip a loaded date whose window starts at or after `until`. Everything downstream then sees only pre-`until` bars. In `phase=multi` (`days_either_side` extras) that drops the later sessions, which would otherwise show the outcome.

**Things the clamp changes beyond the bar list** (all intended, but they need to be stated):
- **GEX line selection is range-dependent.** Both `_build_gex_segments_for_session` and the centre `gex_levels` choose which levels to draw from `band = [session low − padding, session high + padding]` and colour-scale by the selected set (`:950–954`, `:1197–1209`).
  - On the full session that band is lookahead: a level appears because price later reached it.
  - With the clamp the band is the pre-entry range, so **the replay can show fewer or different GEX lines than the full-session view**. This is the honest no-lookahead behaviour. Segment `end_time` stays 13:00 PT, so lines still extend to the right; the levels are prior-day data.
- **Expected-move levels** use the 06:30 PT ES open and the 06:33 PT `vol50` (`:1146–1175`). For a candidate with `entry_ts_pt` < 06:33 PT (ETH) they would leak the RTH open. Rule: omit `expected_move_levels` for a session when `until` < 06:33 PT on that session.
- **Last 5-min bar.** With `interval=5min` the bucket containing `until` is built from only the 1-minute bars before `until`: a partial candle, not a leak. `until` semantics: **bars whose start < `until`**, with `until` floored to the minute. So the entry bar itself is excluded when `entry_ts_pt` is a bar start, and included when `entry_ts_pt` is the bar's close time (start + 1 min). The gallery must say which; see D2.
- `payload["symbol"]` and the other keys are unchanged.

**Live overlay (`_build_react_live_trades_overlay_payload`, `:1264`).**
- It returns bars only when the session is today (`is_live_day`).
- The frontend polls it (and the 30 s full-bars refresh) only while `isLiveTradeDate`.
- **Change:** when `until` is present the route returns `bars: []` before any query. The frontend also skips both live polls while a candidate is active. A candidate from today's session is not expected; the loader rejects `session_date` ≥ today.

**Flow (`_build_react_flow_payload`, `:1539`).**
- The view window is RTH 06:30–13:00 or FULL.
- It **reindexes to every second of the window** (`:1617`), so a clamp must happen before the reindex. Otherwise zero-filled seconds after `until` would draw a decaying EMA tail.
- **Change:** `end_utc = min(end_utc, until)`. The window also feeds `_resolve_symbol`; a truncated window can only resolve the same session's contract.

**Byte-identity plan.**
- **Bars:** reuse CR-BT's harness (`tests/bars_fixture.py`, `fixtures/cr_bt_bars_2026-10-23.pkl.gz`, goldens `cr_bt_bars_golden_{center,multi}.json`). Add `until` absent, `until=` (empty) and an unparseable `until` to the suffix loop; each must equal the golden bytes. The golden still passes on `59e2fa8` (9/9, run in this worktree).
- **Live overlay:** no change when `until` is absent; a unit test asserts the early return when it is present.
- **Flow:** there is no flow golden yet. Record one (record/replay, like `record_bars_fixture.py --golden`) for 2026-10-23 RTH 1s and FULL 1m from the unmodified module before step 3 edits `callbacks.py`, then assert byte equality for absent / empty / invalid `until`.
- **With `until`:** no bar `time` ≥ `until` in any loaded session, no later session loaded, and gex/expected-move derived from the truncated frames.

### Q3 — Price-line and marker API: confirmed; no collision

- **Installed version.** `lightweight-charts` 5.1.0 (`package.json` `^5.0.8`).
- **Price lines.** `ISeriesApi.createPriceLine(options)` with `price, color, lineWidth, lineStyle, lineVisible, axisLabelVisible, title, axisLabelColor, axisLabelTextColor` (typings `:3526–3581`).
  - It is already used on the candlestick series for the CR-009 confluence and subtarget lines (`PriceChart.jsx:2055/2074`), tracked in `landscapeLinesRef` and removed with `removePriceLine`.
  - The candidate's lines (level with "net / gross $bn" in `title`, extreme H/L, entry, stop) go in their own ref, created and removed with the active candidate. Price lines are full-width horizontals, which suits a single session.
- **Markers.** v5 moved markers to a plugin: `createSeriesMarkers(series, markers)` (typings `:294`), with shapes `circle | square | arrowUp | arrowDown` and positions `aboveBar | belowBar | inBar` (plus price positions).
  - **Nothing in `PriceChart.jsx` uses markers today** (no `createSeriesMarkers` or `setMarkers`), so one markers plugin on the candlestick series for test / confirm / retest / entry (and, after reveal, exit) collides with nothing.
- **GEX segments** are separate `LineSeries` (`:2391`), one per level and session, with their own data, so candidate price lines on the candle series don't interact with them. A candidate level close to a GEX segment level is drawn on top. They are distinguishable by colour and title, and the candidate line is a price line (axis label) while the segment is not.
- **Time axis.** The chart plots "shifted" epochs, UTC seconds that display PT wall-clock (`utcEpochShowingZoneTime(…, 'America/Los_Angeles')`, `PriceChart.jsx:554`). Marker times must go through the same shift, and must equal a bar `time` present in the series (bar start). So `*_ts_pt` values are snapped to their minute bar's start (or 5-min bucket) before `setMarkers`.
- **Corrected levels.** CR-BT A1.1/A1.2 apply: before 2024-04-03, or with no ES 15:45 ET snapshot, the chart's Corrected GEX lines are absent with a note. The candidate's own `level_price` line is still drawn from the gallery. That value's frame must be declared by the gallery (D4).

### Q4 — DDL and roles

**Which role the web service uses.**
- The local `.env` `DATABASE_URL` logs in as `new_db_cred`. That role is a member of `rschultz` and has `rolconfig role=rschultz`, so **every session runs as `current_user = rschultz`**.
- `rschultz` owns the database, the `public` schema and every application table checked (`trade_log_trades`, `trade_log_fills`, `bt_signals`, `ironbeam_es_1m_bars`, `bt_backfill_runs`, `ironbeam_es_1m_bars_repair_log`).
- **Not verified on Render:** the user name in "Dash - MonoRepo"'s `DATABASE_URL`. Ryan checks that the Render value's user is `new_db_cred` (or `rschultz`). If it is, the web service owns the new tables and **needs no grants**.
- Default privileges: tables created by `rschultz` automatically grant `dash_backfill_writer` `INSERT, SELECT` (sequences: `USAGE`). Postgres 17.9.
- No `setup_*` tables exist yet.

**Consequence for "insert-only".** The web role is the owner, so grants can't enforce L2's "insert-only plus UPDATE of `revealed_at`". Enforcement has to be a trigger: block DELETE, and allow UPDATE only to set `revealed_at` from NULL. See D5.

**Proposed DDL** (to ship as `scripts/cr_cc_ddl.sql`; **not applied**). It adds a surrogate key and a `post_reveal` flag to `setup_reviews`, which L8 implies (several reviews per candidate; a post-reveal review is flagged):

```sql
BEGIN;

CREATE TABLE setup_candidates (
    id              text PRIMARY KEY,                 -- '<cr>:<set>:<n>'
    cr_id           text NOT NULL,
    set_name        text NOT NULL,
    session_date    date NOT NULL,
    cell            text,
    direction       text NOT NULL CHECK (direction IN ('long', 'short')),
    level_price     double precision,
    level_net_bn    double precision,
    level_gross_bn  double precision,
    entry_ts_pt     timestamp without time zone NOT NULL,  -- naive PT wall-clock
    entry_price     double precision,
    stop_price      double precision,
    extreme_price   double precision,
    test_ts_pt      timestamp without time zone,
    confirm_ts_pt   timestamp without time zone,
    retest_ts_pt    timestamp without time zone,
    meta            jsonb NOT NULL DEFAULT '{}'::jsonb,
    outcome         jsonb,                            -- never sent before a review exists
    loaded_at       timestamptz NOT NULL DEFAULT now(),
    CHECK (id = cr_id || ':' || set_name || ':' || split_part(id, ':', 3))
);
CREATE INDEX setup_candidates_cr_set_idx ON setup_candidates (cr_id, set_name, session_date);

CREATE TABLE setup_reviews (
    id            bigserial PRIMARY KEY,
    candidate_id  text NOT NULL REFERENCES setup_candidates (id),
    label         text NOT NULL CHECK (label IN ('take', 'skip', 'unsure')),
    reasons       text[] NOT NULL DEFAULT '{}',
    comment       text,
    reviewed_at   timestamptz NOT NULL DEFAULT now(),
    revealed_at   timestamptz,
    post_reveal   boolean NOT NULL DEFAULT false      -- written after the candidate was revealed
);
CREATE INDEX setup_reviews_candidate_idx ON setup_reviews (candidate_id, reviewed_at);

-- Insert-only, except setting revealed_at once (L2). The web role owns the tables,
-- so grants cannot enforce this; triggers do.
CREATE FUNCTION setup_review_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION '% is insert-only', TG_TABLE_NAME;
    END IF;
    IF TG_TABLE_NAME = 'setup_reviews'
       AND OLD.revealed_at IS NULL AND NEW.revealed_at IS NOT NULL
       AND (NEW.id, NEW.candidate_id, NEW.label, NEW.reasons, NEW.comment,
            NEW.reviewed_at, NEW.post_reveal)
           IS NOT DISTINCT FROM
           (OLD.id, OLD.candidate_id, OLD.label, OLD.reasons, OLD.comment,
            OLD.reviewed_at, OLD.post_reveal) THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION '% is insert-only (only revealed_at may be set, once)', TG_TABLE_NAME;
END $$;

CREATE TRIGGER setup_candidates_guard BEFORE UPDATE OR DELETE ON setup_candidates
    FOR EACH ROW EXECUTE FUNCTION setup_review_guard();
CREATE TRIGGER setup_reviews_guard BEFORE UPDATE OR DELETE ON setup_reviews
    FOR EACH ROW EXECUTE FUNCTION setup_review_guard();

COMMIT;
```

- `TRUNCATE` bypasses row triggers. Only the owner can truncate, and `dash_backfill_writer` has no TRUNCATE.
- **Grants.** None for the web role if Render's user is `new_db_cred` / `rschultz`. The default ACL gives `dash_backfill_writer` `INSERT, SELECT` on both tables and `USAGE` on the sequence, which is enough for the loader if it runs under the backfill protocol (D6).
- Apply once with `DATABASE_URL` after Ryan's go (L2).

**Reveal semantics under this DDL.**
- `POST …/reveal` sets `revealed_at` on the candidate's latest review with `post_reveal = false`. The candidate is "revealed" when any of its reviews has `revealed_at`.
- Before reveal, a re-save inserts a new row (insert-only); the latest pre-reveal row is the label that counts.
- After reveal, a new review is inserted with `post_reveal = true` and is excluded from the Take/Skip/Unsure outcome stats (L9).
- The custom "add reason" chips are `SELECT DISTINCT unnest(reasons)`; no extra table.

### Q5 — First data: no gallery files exist

- `research/gallery/` exists nowhere: not in the main checkout, this worktree, the CR-CB worktree (`crcb-wt`, at its Step 0 commit `137a051`), or `origin/main`.
- `cr-cb-stage1.json`, `cr-bu.json` and `cr-bw.json` are **absent**. Nothing to validate against L10.
- **What exists instead:** per-event outputs of the earlier CRs, machine-local and git-ignored under `scripts/logs/`, each in its own shape:

| CR | file | rows | columns relevant to L10 | missing for L10 |
|---|---|---|---|---|
| CR-BU | `cr_bu/events_primary.csv` | 151 | `d`, `minute` (naive; looks PT, e.g. 06:42, but **unmeasured**), `N` strike, `E` corrected level, `net_bn`, `close_above`, `r10…r40`, `f15` | direction convention, entry/stop/extreme prices, gross $bn, test/confirm/retest times |
| CR-BW | `cr_bw/events_real.pkl` | 1,057 | `d`, `t`, `L`, `net`, `side`, `res_*`, `end_*` | entry/stop prices, gross, time zone of `t` |
| CR-BX | `cr_bx/events_real.pkl` | 1,291 | `d`, `t`, `dirn`, `entry`, `t_entry`, `ext`, `res_*`, `mv_*` | level fields, stop |
| CR-BY | `cr_by/events.pkl` | 608 | `d`, `t`, `dirn`, `box_hi/lo`, `pnl_a/b`, `res` | level fields |

- Each needs a CR-specific converter to L10, and its time-zone and frame conventions measured first (field-conventions rule).
- Several of these frames hold outcome columns (`res_*`, `pnl_*`, `mv_*`, `r10…`). A converter that copies them into `meta` would leak the outcome before reveal (D4).
- **For implementation step 6, the CR-CB gallery is not ready.** CR-CB is at Step 0; Stage 1 has not run. So the smoke uses the spec's fallback: a 5-event fixture. It can be hand-built from real sessions, for example 5 CR-BX events with known prices, with `outcome` from their stored `mv_*`.

### Decisions needed (before implementation)

1. **D1 — How REVIEW changes the date.**
   - Recommended: lift `tradeDate` (and `expirationDate` when the URL had them equal) into App state seeded from the URL. REVIEW overrides it while a candidate is active; closing REVIEW restores the URL date.
   - Consequence: Dash's date picker and smile time list don't follow during review (the smile panel stays on the URL date). L11 allows no other visible change, so the smile panel is left as is.
   - Alternative: an iframe self-reload per candidate (`?trade_date=…&review=<id>`). Simpler, but a full reload on every N/P.
2. **D2 — `until` semantics and the entry bar.** Recommended: bars with start < `until`, where `until = entry_ts_pt` and `entry_ts_pt` is defined in L10 as **the time the entry decision is made, i.e. the close of the signal bar**. So a 07:36-start signal bar has `entry_ts_pt` 07:37:00 and is shown, and the bar starting 07:37 is not. `expected_move_levels` are omitted when `until` < 06:33 PT on that session. Generators must emit times in naive PT on that convention.
3. **D3 — Other lookahead channels while a candidate is active.** L5/L11 cover bars, the live overlay and flow. These also show post-entry information on a historical date:
   - **ATM IV series** (full-day pane, `/api/ironbeam/atm-iv-series`);
   - **SIGNALS** scan (full-day signals);
   - **smile/skew snapshots** at Dash-selected times later than the entry;
   - **condor pricing** (anchored to smile snapshots).

   Recommended: while a candidate is active, truncate the ATM IV series client-side to < `until`, keep SIGNALS collapsed and unscanned, and skip smile/skew/condor fetches for times ≥ `until`; all are restored on reveal. The landscape (prior-day) needs nothing: its spot is the last merged bar's close, which is pre-`until` once bars are clamped.

   This is a visible change beyond L11's list, so it needs explicit spec language.
4. **D4 — Gallery contract additions (L10).**
   - (a) A `level_frame` field (`es_corrected` / `B` / `spx_cash`), so the drawn level is in the chart's Corrected ES frame or flagged.
   - (b) The loader **rejects** `meta` keys that carry outcome information (a deny-list per CR plus `pnl*`, `res*`, `mv_*`, `exit*`) rather than trusting generators.
   - (c) `session_date < today` enforced at load.
   - (d) Times are naive PT on the D2 convention.
5. **D5 — Insert-only enforcement.** Trigger-based (DDL above), because the web role owns the tables. Alternative: a separate least-privilege web role, which is a Render env change and out of L1's scope.
6. **D6 — Loader connection.** `scripts/load_setup_candidates.py` is an attended, insert-only run. Recommended: the backfill protocol anyway (`get_backfill_db_conn` / `assert_role_or_die` / `backfill_run`). `dash_backfill_writer` already gets `INSERT, SELECT` by default ACL and has no DELETE. Rows go in with `ON CONFLICT (id) DO NOTHING`, and the run reports inserted vs skipped. A corrected re-load needs a new `set_name`, because candidates are immutable.

---

## Amendment A1 (Ryan, 2026-10-25, after Step 0)

D1–D6 from the Step 0 findings are accepted as recommended, with these additions. L1–L11 stand except where a clause below says otherwise.

| # | Clause |
|---|---|
| A1.1 (D1) | `tradeDate` becomes React state, seeded from the URL. While a candidate is active a **"Review: <date>"** banner is shown. Leaving review restores the URL date. |
| A1.2 (D2) | `entry_ts_pt` = the close of the signal bar. Replay shows complete bars with start < `entry_ts_pt`. |
| A1.3 (D3) | With `until` active: truncate the ATM IV series and the flow at `until`; hide the SIGNALS, smile/skew and condor pricing panels until reveal; compute the GEX line band from bars up to `until` (not the full session high/low); suppress expected-move lines when `entry_ts_pt` < 06:33 PT. All of it is restored on reveal. Without `until`, every response is byte-identical to today. |
| A1.4 (D4–D6) | As recommended: gallery `level_frame`; the loader rejects outcome-like `meta` keys, `session_date` ≥ today, and non-PT times; insert-only enforced by triggers; the loader runs under the backfill protocol with `ON CONFLICT (id) DO NOTHING`. Gallery converters for CR-BU / CR-BW are **out of scope**. |
| A1.5 (web DB role) | The web service's role is unknown. The DDL must work regardless: grant `SELECT, INSERT` on `setup_candidates` and `setup_reviews`, and `UPDATE (revealed_at)` on `setup_reviews`, to the role that owns the existing app tables (determined from `pg_tables` / `pg_class`, e.g. `trade_log_trades`), and to explicit roles only, never `PUBLIC`. The web service logs the connected role name once at startup (the role name only, never the URL), so it can be confirmed after deploy. If the web role turns out to lack the privileges, the review routes return a clear "permission" error rather than crash. |
| A1.6 (sequence) | Record a golden for the flow route before step 3. Apply the DDL once with `DATABASE_URL`. Implement steps 1–6, one commit each. Smoke with a 5-event fixture from real 2026 train sessions, loaded as set `cc:fixture` (outcome stored, hidden); Playwright walks label → reveal. Push and open a PR (deploy = "Dash - MonoRepo"); do not merge. |
