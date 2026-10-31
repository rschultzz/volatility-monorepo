# CR-DR — Trade log v2: correct import, trades on the price chart, review panel with tags and context

Branch `feat/CR-DR-trade-log-v2` off origin/main 3ec3a42 (CR-DQ merged). Kickoff: Ryan, chat 2026-10-31.

## Why

Ryan wants to log every real trade, see it on the chart, and review it with notes and lessons, to cut trades that break his own checklist. Today: the importer assumes CT timestamps (Ryan's TradingView order-history export is in PT, verified: 2026-10-30 fills match ES minute bars only as PT); trades never show on the price chart; review fields are notes only. `trade_log_trades` holds 3 trades from 2026-04-21.

## Spec

1. **Import fixes.**
   a. Timezone auto-detect: for each upload, test candidate zones (PT, CT, ET, UTC) by checking every fill price against the 1-minute bar range at the converted time (`es_minutes` / `ironbeam_es_1m_bars`; MES and ES share prices). Pick the zone where all fills match; if none or more than one does, refuse with a clear message. Never guess.
   b. Accept TradingView order-history exports including Cancelled / Rejected rows (stored, ignored for pairing), brackets / OCO (TP + stop) and partial fills. FIFO pairing as today.
   c. Re-import is idempotent (duplicate fills skipped). Existing April trades unchanged; report in Step 0 whether their times match the bars (don't change them without Ryan's OK).
2. **Trades on the price chart.** A "My trades" toggle in the Display group, off by default (C2):
   - entry marker (▲ long / ▼ short) and exit marker (×) at the fill time and price;
   - a thin line between them;
   - resting stop / TP lines drawn while the position was open, from the order history.
   Existing palette only (C1): blue for long, slate for short, no red / green.
3. **Trade review panel.** Clicking a trade marker (or "View on chart" in the Trade Log tab) opens a draggable panel styled like the CR-DQ box panel, default bottom-right.
   a. Facts: side, qty, entry / exit time and price, pts and $ (multiplier by contract), time in trade, MFE / MAE in pts (from 1-minute bars over the holding period), and whether a stop was resting at entry (and after how many seconds).
   b. "What the tools showed at entry", computed with the existing until-replay at the entry minute (never using later bars): the CR-DQ verdict and evidence for the box Ryan was trading, if he had an anchor (stored per trade; otherwise "no anchor"), plus Δ share and contracts/pt percentiles, absorption levels near price, and the last 5 minutes' delta and volume.
   c. Ryan's fields (new nullable columns on `trade_log_trades`), autosave with a visible saved state:
      - setup chips: fade top / fade bottom / continuation pullback / break & retest / failed break / second touch / 0DTE pin / other;
      - pre-trade reason (text);
      - mistake chips: no stop / chased / one-minute flicker / against flow / not at edge / event day / late / moved stop / exited early / held too long / target past level;
      - checklist followed: yes / no;
      - lesson (text);
      - grade 1–5.
   d. N / P steps through that day's trades; the chart scrolls to each.
4. **Trade Log tab:** list and filters gain setup chip, mistake chip, checklist followed and grade. The aggregate bar adds totals by setup chip, by mistake chip and by checklist followed yes / no (counts and pts).

## Constraints

- Only one DB-backed local server at a time; check first, never stop another session's server without asking Ryan.
- Never print connection strings: DB errors print the error type only.
- Ryan's broker CSV stays outside the repo and vault (path from Ryan); never committed or copied.
- Read-only backend except the migration Ryan approves at Step 0 (applied as owner). Don't deploy; PR, don't merge.

## Standing rulings conformance

| Ruling | Status | Note |
|---|---|---|
| A1–A22 | n/a | Setup definitions; this CR records Ryan's own trades and shows existing tool readouts, defines no setup. Setup chips are Ryan's labels, not rule definitions. |
| B1–B3, B5–B8 | n/a | No test is run. |
| B4 | honoured | Real trades are forward data (≥ 2026-10-27), recorded, not analysed as a test. The April 2026 trades are Ryan's own fills (no outcome table read). No outcomes read beyond Ryan's own trades. |
| C1 | honoured | Blue for long, slate for short; panel reuses the CR-DQ palette; no red / green. |
| C2 | deviates (approved) | "My trades" off by default; Trade Log tab additions are new controls; visible changes limited to items 2–4, approved by Ryan in chat 2026-10-31. |
| R1–R7 | not used | — |

## Step 0 (diagnosis gate)

- the CSV format the importer expects vs Ryan's export (from the file Ryan provides);
- the timezone check on the 3 April trades and on 2026-10-30;
- where the until-replay and the CR-DQ verdict can be called for one minute;
- the schema change (new columns, all nullable; applied as owner only after Ryan approves).

## Commits

spec freeze → Step 0 → one commit per item → smoke + wrap.

## Smoke

- 2026-10-30 from Ryan's export: detected zone PT; all fills pair; the four visible trades' P&L matches the vault log #3–#6 (+1.75, −7.5, +13.75, −5.25).
- Markers sit on the right bars and prices.
- Panel facts, MFE / MAE and "at entry" context hand-checked on #5 (07:49 short).
- Chips, checklist flag and notes save and filter; the aggregate splits by checklist followed.
- Existing specs green.

## Step 0 findings

### S0.1 Existing trades: timezone and pairing (`scripts/cr_dr_step0_tz.py`, log `scripts/logs/cr-dr/step0_tz_april.txt`)

Each of the 8 stored April fills' raw `Status Time` (the fill time) checked against the `es_minutes` bar (`ts_utc`) high–low in each zone: **PT 8 / 8, CT 0, ET 0, UTC 0.** The stored `*_ts_pt` values are the PT reading of the raw strings (e.g. `4/21/26 8:27` → 15:27 UTC = 08:27 PT), so the stored clock times are right. The current parser (CT → PT) would shift them 2 h if they were re-imported today.

Pairing is not: `parse_tv_csv` sorts and `fifo_pair_fills` time-stamps by **placing** time. By fill time the 8 fills give four trades:

| Fills (order ids) | By fill time | Stored |
|---|---|---|
| 1352254075 B Limit 7123.75 @08:27 → 1352219378 S Stop 7124.75 @08:35 | long +1.00, exit 08:35 | id 5: long +1.00 but exit 08:28 (stop's placing time) |
| 1353055500 S Limit 7124.25 @12:07 → 1353055573 B Market 7127.00 @12:08 | short −2.75 | id 6: same |
| 1353070310 B Limit 7126.00 @12:10 → 1353045914 S Stop 7123.75 @12:14 | **long** −2.25, entry 12:10, exit 12:14 | id 7: **short** −2.25, entry = the stop (placed 12:10), exit = the limit |
| 1353070885 S Limit 7125.50 @12:19 → 1353080861 B Stop Loss 7131.25 @12:21 | short −5.75 | **missing** (no trade row) |

### S0.2 Archive of `trade_log_trades` before CR-DR (all columns, verbatim)

- **id 5**: `trade_date` = 2026-04-21; `symbol` = F.US.MESM26; `direction` = long; `qty` = 1; `entry_order_id` = 1352254075; `entry_ts_pt` = 2026-04-21 15:27:00+00:00; `entry_price` = 7123.7500; `entry_order_type` = Limit; `exit_order_id` = 1352219378; `exit_ts_pt` = 2026-04-21 15:28:00+00:00; `exit_price` = 7124.7500; `exit_order_type` = Stop; `realized_pts` = 1.0000; `realized_pnl_usd` = 5.00; `fees_usd` = 0.00; `net_pnl_usd` = 5.00; `setup_start_ts_pt` = 2026-04-21 15:07:00+00:00; `setup_target_ts_pt` = 2026-04-21 15:13:00+00:00; `setup_direction` = long; `notes` = ; `context_iv_atm_0dte_pct` = 20.2100; `context_target_spx_price` = 7101.8300; `context_skew_delta_put_pct` = 27.7013; `context_skew_delta_call_pct` = 8.4719; `context_skew_delta_atm_iv` = -7.4593; `context_minutes_to_close` = 293.00; `context_computed_at` = 2026-04-25 14:31:30.484111+00:00; `created_at` = 2026-04-25 01:46:52.938320+00:00; `updated_at` = 2026-04-25 14:31:30.933842+00:00
- **id 6**: `trade_date` = 2026-04-21; `symbol` = F.US.MESM26; `direction` = short; `qty` = 1; `entry_order_id` = 1353055500; `entry_ts_pt` = 2026-04-21 19:07:00+00:00; `entry_price` = 7124.2500; `entry_order_type` = Limit; `exit_order_id` = 1353055573; `exit_ts_pt` = 2026-04-21 19:08:00+00:00; `exit_price` = 7127.0000; `exit_order_type` = Market; `realized_pts` = -2.7500; `realized_pnl_usd` = -13.75; `fees_usd` = 0.00; `net_pnl_usd` = -13.75; `setup_start_ts_pt` = 2026-04-21 19:07:00+00:00; `setup_target_ts_pt` = 2026-04-21 19:08:00+00:00; `setup_direction` = NULL; `notes` = NULL; `context_iv_atm_0dte_pct` = NULL; `context_target_spx_price` = NULL; `context_skew_delta_put_pct` = NULL; `context_skew_delta_call_pct` = NULL; `context_skew_delta_atm_iv` = NULL; `context_minutes_to_close` = NULL; `context_computed_at` = 2026-04-25 01:46:57.255241+00:00; `created_at` = 2026-04-25 01:46:52.938320+00:00; `updated_at` = 2026-04-25 01:46:57.698236+00:00
- **id 7**: `trade_date` = 2026-04-21; `symbol` = F.US.MESM26; `direction` = short; `qty` = 1; `entry_order_id` = 1353045914; `entry_ts_pt` = 2026-04-21 19:10:00+00:00; `entry_price` = 7123.7500; `entry_order_type` = Stop; `exit_order_id` = 1353070310; `exit_ts_pt` = 2026-04-21 19:10:00+00:00; `exit_price` = 7126.0000; `exit_order_type` = Limit; `realized_pts` = -2.2500; `realized_pnl_usd` = -11.25; `fees_usd` = 0.00; `net_pnl_usd` = -11.25; `setup_start_ts_pt` = 2026-04-21 19:10:00+00:00; `setup_target_ts_pt` = 2026-04-21 19:10:00+00:00; `setup_direction` = NULL; `notes` = NULL; `context_iv_atm_0dte_pct` = NULL; `context_target_spx_price` = NULL; `context_skew_delta_put_pct` = NULL; `context_skew_delta_call_pct` = NULL; `context_skew_delta_atm_iv` = NULL; `context_minutes_to_close` = NULL; `context_computed_at` = 2026-04-25 01:46:59.524739+00:00; `created_at` = 2026-04-25 01:46:52.938320+00:00; `updated_at` = 2026-04-25 01:46:59.966581+00:00

### S0.3 Where the until-replay and the CR-DQ verdict can be called for one minute

- The verdict is frontend-only: `boxVerdict`, `edgeReadout`, `segmentReadouts` / `readoutShare`, `boxPercentiles`, `absorptionLevels`, `BOX_RULES_DEFAULTS`, `FOMC_2026_DECISION_DAYS` in `react_price_preview/src/flowV3.js`; App wires them in the `v3Edges` memo.
- The until-replay is the `until` query parameter on the flow requests (`/api/ironbeam/flow?mode=v2…&until=`, set from `reviewUntil` in App, CR-CC A1.2): the backend truncates the payload so bars after `until` are never returned.
- "At entry" therefore issues its own flow request with `until` = the entry minute (floored) and the trade's `anchor_ts_pt`, and runs the same pure functions on it. Nothing is computed twice in a second language, and the chart's own replay state is untouched.

### S0.4 Step 0 decisions (Ryan, chat 2026-10-31)

1. **April fix approved:** order, pair and time-stamp by fill (Status) time; rebuild the April trades from their stored fills. Keep notes / setup annotations on any trade whose fills are unchanged; for changed or new trades, list exactly what was carried over or dropped. Old rows archived above (S0.2) before any change.
2. **Migration approved with one change:** `trade_log_fills.fill_price` stays NOT NULL. A new table `trade_log_orders` holds every order row (filled, cancelled, rejected, working); `trade_log_fills` keeps real fills only. Resting stop / TP lines come from `trade_log_orders`. `trade_log_trades` gains the seven columns below. Of the extra fill columns only `detected_tz` is kept.
3. Apply as owner after recording the DDL here. Insert / update only on existing data (no DELETE).
4. CSV read in place; never copied or committed.

### S0.5 Migration (applied as owner, `scripts/cr_dr_migration.sql`)

```sql
BEGIN;
CREATE TABLE IF NOT EXISTS trade_log_orders (
    order_id        text PRIMARY KEY,
    symbol          text NOT NULL,
    side            text NOT NULL,
    order_type      text,
    qty             integer,
    limit_price     numeric,
    stop_price      numeric,
    status          text NOT NULL,
    placing_time_pt timestamptz,
    status_time_pt  timestamptz,
    duration        text,
    detected_tz     text,
    raw_csv_row     jsonb,
    imported_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_tlo_symbol       ON trade_log_orders (symbol);
CREATE INDEX IF NOT EXISTS idx_tlo_placing_time ON trade_log_orders (placing_time_pt);

ALTER TABLE trade_log_fills  ADD COLUMN IF NOT EXISTS detected_tz text;

ALTER TABLE trade_log_trades
    ADD COLUMN IF NOT EXISTS setup_chips        text[],
    ADD COLUMN IF NOT EXISTS pre_trade_reason   text,
    ADD COLUMN IF NOT EXISTS mistake_chips      text[],
    ADD COLUMN IF NOT EXISTS checklist_followed boolean,
    ADD COLUMN IF NOT EXISTS lesson             text,
    ADD COLUMN IF NOT EXISTS grade              smallint,
    ADD COLUMN IF NOT EXISTS anchor_ts_pt       timestamptz;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'trade_log_trades_grade_1_5') THEN
        ALTER TABLE trade_log_trades ADD CONSTRAINT trade_log_trades_grade_1_5 CHECK (grade BETWEEN 1 AND 5);
    END IF;
END $$;
COMMIT;
```

**Applied 2026-10-31** as `rschultz` (table owner) via `scripts/cr_dr_apply_migration.py` (log `scripts/logs/cr-dr/migration_apply.txt`): `trade_log_orders` created (0 rows); `trade_log_fills.detected_tz` added; the seven `trade_log_trades` columns and `trade_log_trades_grade_1_5` added. Row counts unchanged (trades 3, fills 8). The deployed app (origin/main) ignores the new columns and table.

### S0.6 Ryan's export vs the importer (`scripts/cr_dr_step0_csv.py <path>`, log `scripts/logs/cr-dr/step0_csv.txt`)

File: Ryan's TradingView / AMP "order history (all)" export, 2026-10-31 22:08 UTC, read in place (outside the repo; not copied).

| | Importer expects (`parse_tv_csv`) | Ryan's export |
|---|---|---|
| columns | `Symbol, Side, Type, Qty, Fill Qty, Avg Fill Price, Status, Commission, Duration, Order ID, Placing Time, Status Time` (+ `Limit Price, Stop Price, Active At` in the April rows) | `Symbol, Side, Type, Quantity, Limit price, Stop price, Active at, Fill quantity, Avg fill price, Commission, Placing time, Status, Status time, Order ID, Duration` |
| result today | — | **0 fills found** (keys are case-sensitive and `Qty` / `Fill Qty` are renamed): the upload is refused with "No filled orders found" |
| time format | 9 formats incl. `%Y-%m-%d %H:%M:%S`; read as CT | `2026-10-30 09:28:15` (seconds); PT (below) |
| statuses | `cancelled` skipped, rest read as fills if they have a fill qty | Filled 18, Cancelled 12, Rejected 2 |
| order types | free text | Limit 14, Take Profit 6, Stop 5, Stop Loss 4, Market 3 |
| other | — | one symbol `F.US.MESZ26`; `Commission` and `Active at` empty on every row; no partial fills; one price improvement (limit 7720.25 filled 7720.00, 10-27 07:13:40) |

Fix (item 1): header lookup case-insensitive with aliases (`Qty`/`Quantity`, `Fill Qty`/`Fill quantity`, …); both the April layout and this one parse.

### S0.7 Timezone check on 2026-10-30 (and the whole export)

Every filled row's **Status time** (the fill time) in PT / CT / ET / UTC against the `es_minutes` bar (`ts_utc`, minute floor) high–low:

- **PT 17 / 18; CT 0, ET 0, UTC 0.** 2026-10-30: PT 7 / 8.
- The miss: **10-30 06:22:56 sell stop filled 7668.00** (#3's exit); the PT 06:22 ES bar's low is 7668.25. `ironbeam_es_flow_1s` (ES Z26) prints 7668.25 every second 06:22:51 → 06:23:05 PT and never 7668.00 (`scripts/logs/cr-dr/step0_stopfill.txt`). MES is its own order book: the stop filled as a market order one tick through the ES price. **The spec's "MES and ES share prices" holds to ± 1 tick, not exactly**, so a strict "all fills inside the bar" rule refuses Ryan's own export.
- The wrong zones miss by whole points (the nearest: CT 10-30 09:28 at 0.75 below the bar low), so a one-tick tolerance keeps the zones far apart: with ± 0.25, PT 18 / 18, CT / ET / UTC 0 / 18.

### S0.8 FIFO pairing by fill time (whole export)

| Date | Side × qty | Entry (PT) | Exit (PT) | Pts |
|---|---|---|---|---|
| 10-26 | long × 2 | 07:56:49 7731.25 | 08:30:12 7728.50 (Stop) | −2.75 |
| 10-26 | long × 2 | 10:36:12 7717.75 | 10:41:22 7712.50 (Market) | −5.25 |
| 10-27 | long × 1 | 07:13:40 7720.00 | 07:16:44 7715.00 (Stop Loss) | −5.00 |
| 10-27 | long × 1 | 07:18:02 7718.00 | 07:21:43 7712.75 (Stop Loss) | −5.25 |
| 10-27 | long × 1 | 07:23:49 7717.75 | 07:32:11 7728.75 (Stop) | +11.00 (log #2 ✓) |
| 10-30 | long × 1 | 06:11:46 7666.25 | 06:22:56 7668.00 (Stop) | +1.75 (#3 ✓) |
| 10-30 | long × 1 | 07:06:47 7658.75 | 07:10:00 7651.25 (Market) | −7.50 (#4 ✓) |
| 10-30 | short × 1 | 07:49:17 7655.75 | 08:08:12 7642.00 (Limit, TP) | +13.75 (#5 ✓) |
| 10-30 | short × 1 | 09:22:50 7625.50 | 09:28:15 7630.75 (Market) | −5.25 (#6 ✓) |

All 18 fills pair; nothing open. The four 10-30 trades match the vault log #3–#6. Not in the vault log: both 10-26 trades (log #1 records 10-26 as "chat call, no fill recorded") and the first two 10-27 trades (−5.00, −5.25). Reported only; the vault log is not edited by this CR.

Orders behind #5 (answers the log's "Check"): TP 7642 placed 07:51:03, cancelled 07:51:06 and replaced by an OCO pair placed 07:51:06, a Buy Limit 7642 (filled 08:08:12) and a **Buy Stop 7647.75** (cancelled 08:08:12 by the fill). The stop price is below the 7655.75 short entry, as the log says; the export confirms it stayed working, unfilled, for the whole trade.

### S0.9 B4 note: 2026-10-26 is a holdout date

B4 freezes 2026-06-06 → 2026-10-26 and sets the forward log from 2026-10-27; the kickoff says real trades are forward data (≥ 2026-10-27). The export holds two 10-26 trades. **Disclosure:** the S0.7 timezone check read the `es_minutes` high / low of 16 bars on 2026-10-26 (the four 10-26 fill minutes × four zones): bar ranges only, no outcomes, setups, levels or cards. Importing those trades with MFE / MAE and "at entry" context would read more 10-26 bars.

### S0.10 Points for Ryan before implementation

1. **Fill-price tolerance (spec amendment, item 1a):** a fill matches a zone when its price is within the bar's low − 0.25 … high + 0.25 (one tick, MES vs ES books); all fills must match exactly one zone, else refuse. Recommended. The alternative (strict) refuses this export.
2. **10-26 trades (B4):** import them (Ryan's own fills; his trades are recorded in any case), or skip dates ≤ 2026-10-26 down to the train cut (2026-06-05) with a warning? Recommended: import the fills and orders, but show no MFE / MAE or "at entry" context for holdout dates (panel says "holdout date: context not shown", B4).
