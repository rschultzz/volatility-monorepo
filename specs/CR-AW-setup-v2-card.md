# CR-AW — Setup v2 tab: max price, band evidence, KNN factors

> Authority: vault session note `Dash/sessions/2026-09-07 - CR-AW — Setup v2 Card.md`; design: `Dash/New Functionality/mockups/setup-v2-proposal-card.html` (two tabs: Trade, KNN factors)
> Branch: `feat/CR-AW-setup-v2-card` (off `origin/main` 04e13a2, the CR-AV merge)
> Scope: new Flask module `apps/web/modules/SetupV2/` (`/setup-v2`, `/api/setup-v2/card`), fair-value helper in `probability.py`, v2 frontend in `react_today_setup/`; `/today-setup` and its module untouched. **No deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.

Drafted in chat 2026-09-07. **Authoritative mockup: `Dash/New Functionality/mockups/setup-v2-proposal-card.html`** (two tabs: Trade, KNN factors). It is the design to build; the earlier ledger/scorecard mockups are superseded and not in the vault. Ryan: **new tab beside the existing Today's Setup; old one stays until v2 is trusted.** Unattended through PR. Runs in parallel with CR-AU (worktree).

Terminology: analogues are **every corpus day inside the KNN similarity ceiling** (recency-weighted), not a fixed K. `_K_SAFETY=200` never binds. The card must never display "K="; it says "N analogues within the similarity ceiling".

## What the card must say, in order

1. **Max price to pay** for the proposed spread, from analogue closes. Quote vs max on a gauge; verdict "enter at the open" / "skip".
2. Fair value, expected P&L at the quote (with interval), market-implied (quote ÷ width).
3. **Distance band strip** with the reference cell numbers and today outlined.
4. **Analogue facts** with honest labels: reached the wall; finished above the short strike (full payout); finished above the long strike (any payout); spread worth at close.
5. **How to run it**: enter if under max (don't wait); hold to close; 15 DTE — each with its number. Untested watches (wall half-life, wall trend, vol state) in grey.
6. **Stamp**: reference `cr_id`, cell, n, re-run date, next run, fees included.
7. **Tab 2 — KNN factors**: every feature in the match, today's value, weight, percentile vs the magnet-above corpus, the analogues' middle-half band, a match-quality line.
8. Post-touch bars and the by-minute edge chart behind a detail toggle. No edge ratio anywhere on v2.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Route | New Flask module `apps/web/modules/SetupV2/` serving `/setup-v2` and `/api/setup-v2/card?date=`. New top-nav tab "Setup v2". `/today-setup` and its module untouched. |
| 2 | Fair value | For the walk-forward analogue set already returned by `compute_structural_probability(before_date=date)`: per analogue, `value = clamp(width − d, 0, width)` where `d = final_close_distance_from_target` (points below target; Step 0 confirms sign against the 11% close-above rate). `fair = mean(value)`. **Max price = 20th percentile of the bootstrap distribution of the mean** (1,000 resamples, seed = trade_date as int). Expected P&L = fair − quote; interval = bootstrap 2.5/97.5 of mean minus quote. Only analogues with `outcome_status='computed'`. Fees per shared config subtracted from expected P&L. |
| 3 | Any-payout rate | share of analogues with `d < width`; full-payout = `d ≤ 0`. |
| 4 | Band table | 3 rows + all from `bt_edge_backtest_results` at the latest `cr_id` matching `REF-%`, else `CR-AR`; `structure_type='debit'`, `outcome_type='close'`, `post_touch_pattern IS NULL`. Show `mean_pnl`, `win_rate`, Wilson, n. Today's band from the feature vector's signed distance. |
| 5 | Verdict | "enter at the open" if live quote ≤ max; "skip — quote above max" otherwise; "no listed structure" if the pair snapper raises. Never "wait". |
| 6 | KNN tab | Payload lists every feature key used by the KNN with: today's value, weight (read from the KNN config, not hardcoded), percentile of today vs magnet-above corpus ≤ date (`bt_daily_features_active`), 25th–75th percentile of the same feature across today's analogues, `in_band` bool. Match-quality = count in_band / total. Features with null today (IV rank, VRP) show "not populated". |
| 7 | Untested rows | Vol state, magnet horizon, wall trend: rendered from whatever the payload has (today: implied move percentile; the others null until CR-AT's wall table exists) with a fixed "untested" status. No colour other than grey. |
| 8 | Tests | Fair value on a synthetic analogue set (all above short → fair = width; all far below → 0; mixed); max ≤ fair; verdict for quote above / below / no structure; band selection; percentile computation; bundle grep: no "edge ×", no "supported", no "low-confidence" strings in the v2 bundle. |
| 9 | Not in scope | Changing `/today-setup`; condor card; wall table (CR-AT); weights fitting. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0 — `main` contains CR-AV; sign of `final_close_distance_from_target` confirmed (mean over analogues consistent with ~10% close-above) | yes | STOP |
| G1 — tests per decision 8; suites pass | yes | STOP |
| G2 — `git diff --stat` confined to SetupV2 module, Proposals (payload helpers only), probability.py (fair-value helper only), react_today_setup, tests, spec | yes | STOP |
| G3 — local `/api/setup-v2/card?date=2026-09-03`: fair, max, expected P&L, band, any-payout %, stamp, KNN tab all populated; verdict computed against the stored 06:34 quote | yes | STOP |
| G4 — `/today-setup` payload for 2026-09-03 byte-identical before and after | yes | STOP |

## Kickoff prompt

```
CR-AW — Setup v2 card. Unattended through PR. Authority: vault note
"Dash/sessions/2026-09-07 - CR-AW — Setup v2 Card.md". Read it and open the
mockup "Dash/New Functionality/mockups/setup-v2-proposal-card.html" in a
browser or as source — it is the design. Halt only at STOP gates.
DO NOT deploy. Work in worktree .claude/worktrees/cr-aw.

Branch: git fetch; git checkout -b feat/CR-AW-setup-v2-card origin/main.
Commit 1 — spec freeze. Step 0 → Commit 2: G0 sign check (query + one
  sentence); list the KNN feature keys and their weights from config.
Commit 3 — fair-value helper in probability.py + tests (decision 2, 3).
Commit 4 — SetupV2 module: route, payload (decisions 4–7) + tests.
Commit 5 — frontend: new tab, two-pane card from the mockups, post-touch
  and edge chart behind a toggle, bundle grep test. Build.
Step 1 — G3, G4. Commit 6 with the 2026-09-03 payload (numbers) pasted.
Wrap — Commit 7. Vault: run log; Sessions MOC. Push, open PR
  "CR-AW — Setup v2 tab: max price, band evidence, KNN factors".
  DO NOT MERGE. Print PR URL and the 2026-09-03 card numbers
  (fair, max, expected P&L, any-payout %, band, match quality).
```

## Decisions

- **Max = 20th percentile of the bootstrap mean.** A margin that scales with the analogue sample's uncertainty rather than a fixed discount. Locked here so the number means the same thing every day.
- **Verdict never says "wait".** Waiting selects adverse minutes (CR-AH/AR). Under max → enter; over → skip.
- **New tab, old untouched.** v2 earns trust beside v1; v1 is retired by a separate one-line CR when Ryan says so.

## Links

- [[2026-09-07 - CR-AV — Proposals Card, Evidence Only]], [[2026-09-06 - CR-AR — Pair-Snapping with Width Cap]], [[edge-zone-thresholds-need-recalibration-for-real-implied]], [[position-aware-edge-and-ev-display]], [[2026-09-07 - CR-AU — Live Out-of-Sample Stream]]
