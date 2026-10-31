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
