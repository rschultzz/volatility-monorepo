# CR-BT — Corrected Level Placement and Chart Toggle

Spec freeze. Branch `feat/CR-BT-corrected-level-placement` off `origin/main` `e78774f` (the CR-BS merge, PR #70). Read-only against the database; no table writes, no cron.

Source: vault `Dash/CRs/CR-BT — Corrected Level Placement and Chart Toggle.md` (last_reviewed 2026-10-24). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L10 were confirmed by the kickoff prompt (Ryan, 2026-10-24).

---


## Why

The colored GEX lines on the price chart are placed with `discounted_level` = strike × exp((r − q)(dte + 1)/252), using **the option's own DTE** (`apps/cron/job_orats_eod.py:121`). Chat review of the code on 2026-10-24 (no DB run) gives three problems:

1. **The direction is wrong for longer-dated options.** An option's gamma peaks when *its own expiry's forward* equals the strike. So in SPX cash the peak sits *below* the strike, by carry to that option's expiry. `discounted_level` moves it *up* by that carry.
2. **The ES contract's expiry is missing.** On an ES chart, every price sits above SPX cash by carry to the *ES* expiry. The correct ES-chart location is

   **L = K × e^(carry × (T_ES − T_option))**, equivalently **L = K × (ES price ÷ F_option)**, both prices taken at the same snapshot.

   - An option expiring with the ES contract plots exactly at its strike.
   - 0DTE plots near strike + ES basis.
   - Options expiring after ES plot below their strike.
3. **One strike becomes several lines.** `_fetch_gex_grouped_by_level` buckets by `discounted_level`. One strike's 0DTE, weekly and monthly OI are shifted by different, wrong amounts and land in different buckets.

**Chat's rough error estimate** (SPX ≈ 7700, ES Z26 ~55 days out, basis ~25, carry ~0.45 pts per calendar day; unmeasured):

| Expiry | Current shift | Correct shift | Current error |
|---|---|---|---|
| 0DTE | +1 | ~+25 | ~24 too low |
| 7 DTE | +10 | ~+22 | ~12 too low |
| 30 DTE | +38 | ~+11 | ~27 too high |
| 55 DTE (Dec quarterly) | +68 | ~0 | ~68 too high |
| 90 DTE | +112 | ~−16 | ~128 too high |

This was first flagged in CR-BI Step 0 (2026-09-20). On 2026-09-21 Ryan decided the level math must account for both the option's and the ES contract's expiry, built in parallel without touching the current system. This CR does that.

**Ryan's requirement (2026-10-24):** recalculate the levels, leave the old ones exactly as they are, and add a toggle to switch between the two.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | (a) A pure shared function that places each `orats_oi_gamma` row on a target price axis. (b) A price-chart toggle, "Levels: Current \| Corrected", **default Current**. No table writes, no cron changes, no change to `discounted_level`, `orats_gex_landscape`, `bt_daily_features`, `bt_daily_outcomes`, or the landscape panel. |
| L2 | Placement formula | **L = K × (P_target ÷ F_e)**. F_e = the row's own `orats_oi_gamma.stock_price` (per-expiry parity-implied price at the prior session's 15:46 ET snapshot; field-conventions). P_target = the target-frame price at that same snapshot. No rates, no dividend yields, no day counts. |
| L3 | Target frames | One function, two callers. **ES chart:** P_target = ES front-month price at the prior session's 15:46 ET. **SPX cash** (used by CR-BU): P_target = S0 = `stock_price` of the nearest live expiry (CR-BK A1.1). |
| L4 | ES snapshot | Close of the `ironbeam_es_1m_bars` bar starting 15:45 ET on the API trade date. Bars carry the UTC bar *start* (field-conventions); DST is handled through `packages/shared/sessions.py` or `zoneinfo`, never a fixed UTC offset. Step 0 confirms the alignment. |
| L5 | Same selection pipeline | Corrected levels are aggregated and filtered with **exactly** the current machinery, so placement is the only difference: same bucket (`GEX_LEVEL_BUCKET`), same band padding, threshold, minimum spacing, maximum lines, color scale, line width and opacity. Only the level key changes: corrected L instead of `discounted_level`. |
| L6 | Per-expiry popup | The `gex_segments` per-expiration breakdown follows the toggle, built from corrected L, so its rows still sum to the line. |
| L7 | Old path untouched | With the toggle on Current (or the parameter absent), the `/api/ironbeam/bars` response is **byte-identical** to today's for the same inputs. A test enforces this. |
| L8 | Toggle mechanics | A query parameter on the existing route (`levels=corrected`; absent = current). The frontend control sits beside the existing GEX controls. Remembering the choice in `localStorage` is allowed (wrapped in try/catch). No other visible change: no layout, colors or spacing edits (CLAUDE.md visible-change discipline; this spec authorises only the toggle control and the alternate line set). |
| L9 | Intraday | Levels stay fixed through the session, as today (prior-day OI, prior-day snapshot). Intraday recasting is out of scope; see Not in scope. |
| L10 | Holdout | This CR reads no outcomes. Live display of any date is allowed. |

## Step 0 — diagnosis gate (answer all before implementation; contradiction-stop)

1. **`stock_price` is a per-expiry forward.** On 20 random train dates plus 2026-10-23:
   - plot or tabulate F_e / S0 − 1 against calendar days to expiry; it should rise smoothly with DTE;
   - report non-monotone dates and any expiry with a missing or zero `stock_price` (count rows, and share of total |GEX|).
2. **Price-space check against ES** (this is the measurement, per the CLAUDE.md rule):
   - For each sample date, take the SPX expiry nearest the front ES contract's expiry (the quarterly third Friday). Report |F_matched − ES(15:46 ET)|: median, p90 and max.
   - Gate: median ≤ 5 pts. If it fails, stop and report; do not substitute rates.
   - Flag the AM/PM third-Friday overwrite (field-conventions) where it touches the matched expiry.
3. **ES contract identity at the snapshot.** `ironbeam_es_1m_bars` has no contract column.
   - Confirm the live Ironbeam feed is the front month on 2026 dates, including the 2026-09 roll patch window.
   - Count train dates in 2023–2025 roll weeks, where the ES bars are the dying contract (calendar-roll `ES.c.0`).
   - Those dates are display-only caveats for this CR, but CR-BU needs the count.
4. **Worked day 2026-10-23.**
   - For the 10 largest |GEX| rows: strike, expiry, DTE, `discounted_level`, corrected L, and the difference.
   - Then the top selected lines under Current vs Corrected.
   - Compare against the chat estimate table above, and say where it was wrong.
5. **Render path.**
   - Identify exactly which payload fields the React price chart draws: `gex_levels`, `gex_segments`, or both (`react_price_preview/src/App.jsx`, `PriceChart.jsx`).
   - Identify the smallest change that adds the toggle.
   - Name every file to be touched. If anything outside `apps/web/modules/Ironbeam/`, `packages/shared/` and `react_price_preview/src/` is needed, stop and ask.
6. **Byte-identity test plan for L7.** Say how the test captures the current response (fixture date, mocked DB or recorded rows).

## Implementation order (one commit each)

1. `packages/shared/level_placement.py`: `place_levels(rows, p_target) -> rows with level_corrected`, pure, plus unit tests:
   - the matched-expiry row returns its strike to within float tolerance when F_e = P_target;
   - 0DTE shifts by about the basis;
   - an expiry after ES shifts down;
   - empty and NaN inputs are handled.
2. Backend:
   - a `levels` parameter on `/api/ironbeam/bars`;
   - corrected variants of `_fetch_gex_grouped_by_level` / `_fetch_gex_grouped_by_level_and_expir` built on rows + `place_levels`, with the existing functions untouched;
   - the L7 byte-identity test.
3. Frontend toggle (L8).
4. Smoke on 2026-10-23, one 2026 roll-adjacent date, and one third Friday:
   - screenshots of Current vs Corrected;
   - the per-expiry popup sums to its line in both modes.

## Outputs

- Spec `specs/CR-BT-corrected-level-placement.md` with the Step 0 answers and the smoke screenshots.
- Session note in `Dash/sessions/`; new field-conventions row for corrected L once Step 0 measures it.
- Deploy: the web service only. No cron changes.

## Not in scope

- Any change to stored levels, the landscape panel, regimes, outcomes or cards.
- Testing whether the levels work. That is CR-BU.
- Intraday recasting of gamma size (time decay, live IV), CR "C". It depends on CR-BU finding something, and it cannot add new OI.
- The CR-BK repriced-profile toggle on the landscape panel (separate, lower priority).
- The dealer-sign assumption.

## Kickoff prompt for Claude Code

```
Start CR-BT. First action: git fetch && git checkout -b feat/CR-BT-corrected-level-placement origin/main (confirm CR-BT is unused; rename if not).
Read the vault CR at "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BT — Corrected Level Placement and Chart Toggle.md" and components/field-conventions.md.
Commit 1: spec freeze to specs/CR-BT-corrected-level-placement.md, locked decisions transcribed verbatim.
Commit 2: Step 0 findings appended to the spec. Answer all six gate questions; stop and report to me before any implementation; stop immediately if gate 2 fails or anything contradicts the spec.
Read-only against the database. No table writes. Do not modify discounted_level, job_orats_eod.py, orats_gex_landscape, or the existing _fetch_gex_grouped_by_level* functions.
After I approve Step 0: implement steps 1-4, one commit each; with the toggle off the /api/ironbeam/bars response must be byte-identical to today's.
```
