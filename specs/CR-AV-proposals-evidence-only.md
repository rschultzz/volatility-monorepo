# CR-AV — Proposals: credit-fade removed; direction gate advisory

> Authority: vault session note `Dash/sessions/2026-09-07 - CR-AV — Proposals Card, Evidence Only.md`
> Branch: `feat/CR-AV-proposals-evidence-only` (off `origin/main` d8988ad, the CR-AS merge; CR-AR merge 331269e is an ancestor)
> Scope: live Proposals path — credit-fade no longer emitted; `apply_direction_qualification` advisory only; card advisory block. **No deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.


Live path. Runs after CR-AR (merged 331269e), before the web deploy.
Ryan's decision 2026-09-07: credit-fade off the card; direction gate must
not filter.

## Problem
- Credit-fade proposals: refuted CR-AH, held through CR-AM/AN/AP/AR —
  negative P&L and beat in every band.
- Direction gate (apply_direction_qualification) drops/promotes proposals on
  pattern_label. CR-AL: label does not predict P&L; CR-AM: no reversal.
  On 2026-09-03 it dropped the credit proposal.

## Locked decisions
1. Credit-fade no longer emitted as a live proposal. Structure code stays
   (harness uses it). No placeholder on the card.
2. apply_direction_qualification no longer filters or promotes. Returns
   label, sample sizes, fractions; the card renders an advisory block
   ("post-touch pattern: stepping-stone · n=23 · t5 above 61%"), no badge
   implying support or caution. filter_mode stays in payload for audit;
   UI ignores it for display decisions.
3. Edge threshold unchanged.
4. Payload: existing fields kept; new advisory_only: true on the post-touch
   block; frontend reads it.
5. Tests: magnet day yields debit only, no credit in payload; stepping-stone
   and mixed labels produce identical proposal sets with label text present;
   frontend advisory block renders label + n; built-bundle grep finds no
   "supported" / "low-confidence" badge strings.
6. Out of scope: credit in harness/Backtests UI; edge threshold; condor
   card; the setup-ledger redesign.

## Gates
G0 main contains 331269e; every caller of apply_direction_qualification
   listed — STOP on miss.
G1 tests per decision 5 pass; all suites pass — STOP.
G2 git diff --stat confined to Proposals, qualification module, TodaySetup
   frontend, tests, spec — STOP.
G3 local /proposals?date=2026-09-03: one structure (debit), post-touch
   block with advisory_only true, no badge — STOP.

## Kickoff prompt
Branch: git fetch; git checkout -b feat/CR-AV-proposals-evidence-only origin/main.
Commit 1 spec freeze: specs/CR-AV-proposals-evidence-only.md from this note.
Step 0 → Commit 2: G0 caller list; current /proposals payload for
  2026-09-03 run locally (structure names, badges, filter_mode) as "before".
Commit 3 backend (decisions 1, 2, 4) + tests.
Commit 4 frontend advisory block + bundle grep test. Build.
Step 1 G3 local run; paste "after" payload. Commit 5.
Wrap Commit 6: spec What changed / Decisions. Vault: this note's run log;
  engine-qualification-untested-needs-corpus gets "gate demoted to advisory
  2026-09-07 (CR-AV)"; Sessions MOC one-liner. Push, open PR
  "CR-AV — Proposals: credit-fade removed; direction gate advisory".
  DO NOT MERGE. Print PR URL and before/after payload summary.

## Step 0 findings (2026-09-07)

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0 — `main` contains 331269e | yes | `git branch -r --contains 331269e…` → `origin/main`; branch cut from `origin/main` d8988ad (PR #50, CR-AS merge; 331269e is its first parent's ancestor) | PASS |
| G0 — every caller of `apply_direction_qualification` listed | yes | one production caller; table below | PASS |

### Callers of `apply_direction_qualification` (and of the qualification module)

| Caller | What it does today | CR-AV action |
|---|---|---|
| `apps/web/modules/TodaySetup/routes.py:317` (`GET /api/setup/proposals`) | `response["proposals"] = apply_direction_qualification(response["proposals"], structural_probability)` when SP is available | keep the call; the function no longer filters / promotes and marks `post_touch.advisory_only = true` |
| `apps/web/modules/TodaySetup/service.py:36` (definition) | filters to credit-only / debit-only by `pattern_label` + Wilson floor, else both; sets `confidence_badge` ("credit-fade supported", "debit-to-target supported", "mixed pattern — no clear direction", "low-confidence — post-touch sample insufficient", "0DTE corpus insufficient") | rewrite per decision 2 |
| `apps/web/modules/TodaySetup/tests/test_post_touch_qualification.py` | 13 tests pin the filtering / badges | rewrite per decision 5 |
| `packages/shared/post_touch_qualification.py` (`credit_direction_qualifies`, `debit_direction_qualifies`, `_CREDIT_PATTERNS`, `_DEBIT_PATTERNS`) | pure helpers; imported by the service and by `scripts/cr_ah_step4_analysis.py:1282` (harness pattern sets) | untouched in behaviour (harness keeps its import); docstring notes the live path is advisory |
| Frontend | **no component reads `confidence_badge`** (grep of `react_today_setup/src`, `packages/web-shared/src`). The only direction-support text the bundle carries is `SYNTHESIS_LINES` in `StructuralProbabilityBlock.jsx` ("Direction signal: debit-to-target supported." etc.), and `PostTouchSection` branches on `filter_mode` for the thin-corpus notes | replace the synthesis line with the advisory block (decision 2); the bundle-grep test targets the badge phrases |

Credit-fade emission: `packages/shared/strategy_templates.py::_DirectionalSpreadTemplate` (`directional_spread_to_target`) is emitted by `generate_proposals` for every magnet-regime payload alongside the debit template. `generate_proposals` is also called by the CR-AH scripts (`cr_ah_step0_power_check.py`, `cr_ah_step01_coverage_probe.py`) and its tests expect both spreads, and `packages/shared/` is outside G2's file list — so decision 1 is applied in `build_proposals_response` (the live `/api/setup/proposals` builder): the credit template is dropped from the live list there; the template class and `generate_proposals` are unchanged ("structure code stays").

G2 reading: "Proposals" = the `/api/setup/proposals` module (`apps/web/modules/TodaySetup/service.py`, `routes.py`) plus `apps/web/modules/Proposals/` if touched; "qualification module" = `packages/shared/post_touch_qualification.py`; "TodaySetup frontend" = `react_today_setup/`.

### "Before" — `GET /api/setup/proposals?date=2026-09-03&ticker=SPX` run locally (Flask test client on the route, real DB, read-only; scratchpad `cr_av_payload.py`, `cr_av_before.json`)

```
status 200 ok True regime magnet-above
proposals:
  directional_spread_to_target     kind=spread     dte=15 badge='mixed pattern — no clear direction'
  debit_spread_to_target           kind=spread     dte=15 badge='mixed pattern — no clear direction'
structural_probability: regime_kind magnet-above outcome_status ok k 70
post_touch: {'filter_mode': 'strict', 'pattern_label': 'stepping-stone', 'same_bucket_n': 25, 'total_touchers': 50, 'advisory_only': None}
fractions: t1 above 0.56 / t5 above 0.72 / t15 above 0.56
wilson t15 above [0.371, 0.733]; t5 above [0.524, 0.857]
top-level keys: context, ok, proposals, structural_probability
```

Before: **two structures** (credit `directional_spread_to_target` + debit `debit_spread_to_target`, 15 DTE), both badged `mixed pattern — no clear direction` (stepping-stone but t15 `above` Wilson lower bound 0.371 < 0.40 floor under today's walk-forward pool — the note's "dropped the credit proposal on 09-03" was the earlier pool), `filter_mode` strict, pattern stepping-stone, n = 25 same-bucket touchers. No `advisory_only` field.

## Commit 4 — frontend

`StructuralProbabilityBlock.jsx`: the `SYNTHESIS_LINES` table ("Direction signal: debit-to-target supported." …) is gone; `PostTouchSection` renders `post-touch pattern: <label> · n=<same_bucket_n> · <tf> <direction> <pct>` from the server-side `post_touch.advisory` block when `advisory_only` is true (falls back to the raw fields; renders nothing on a legacy payload). `filter_mode` is still read only for the thin-corpus data notes ("Insufficient post-touch sample", "0DTE corpus insufficient", "pooled fallback"), which describe the sample, not a direction. `styles.css`: `.pt-synthesis` → `.pt-advisory`. Tests: `StructuralProbabilityBlock.test.jsx` (5: label + n + fraction; stepping-stone vs mixed differ only in the label; no supported / low-confidence / no-clear-direction / Direction-signal text for any label; raw-field fallback; legacy payload renders nothing) and `bundleBadgeStrings.test.js` (greps `dist/assets/*.js` for the five badge phrases; skips with a warning when `dist/` is absent). Build: `npm run build` → `dist/assets/index-DmgJZy5k.js` (382 kB); grep of the bundle for `debit-to-target supported | credit-fade supported | low-confidence | no clear direction | Direction signal` → **none**; `post-touch pattern` present. Note: the main checkout's `react_today_setup/node_modules` was an x64 install (rollup/rolldown arm64 bindings missing); reinstalled with `npm ci` (the x64 tree moved aside outside the repo).

## Step 1 — G3: `GET /api/setup/proposals?date=2026-09-03&ticker=SPX` run locally, after (scratchpad `cr_av_payload.py`, `cr_av_after.json`)

```
status 200 ok True regime magnet-above
proposals:
  debit_spread_to_target           kind=spread     dte=15 badge=None
structural_probability: regime_kind magnet-above outcome_status ok k 70
post_touch: {'filter_mode': 'strict', 'pattern_label': 'stepping-stone', 'same_bucket_n': 25, 'total_touchers': 50, 'advisory_only': True}
post_touch.advisory: {'pattern_label': 'stepping-stone', 'n': 25, 'n_pooled': 50, 'timeframe': 't15', 'direction': 'above', 'fraction': 0.56, 'wilson_lo': 0.371, 'wilson_hi': 0.733}
fractions: t1 above 0.56 / t5 above 0.72 / t15 above 0.56   (unchanged)
top-level keys: context, ok, proposals, structural_probability   (unchanged)
```

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G3 — one structure (debit) | yes | `debit_spread_to_target` only; `directional_spread_to_target` gone, no placeholder | PASS |
| G3 — post-touch block with `advisory_only` true | yes | `advisory_only: True`, `advisory` = stepping-stone · n=25 · t15 above 56 % (the card text for a 15-DTE trade) | PASS |
| G3 — no badge | yes | no `confidence_badge` on the proposal; bundle carries no badge strings | PASS |
| G2 — `git diff --stat` scope | Proposals / qualification module / TodaySetup frontend / tests / spec | `apps/web/modules/TodaySetup/{service,routes}.py`, `packages/shared/post_touch_qualification.py` (docstring), `apps/web/modules/TodaySetup/tests/*`, `react_today_setup/src/**`, `specs/CR-AV-*.md` | PASS |

### Before / after summary (2026-09-03, magnet-above, stepping-stone, n = 25)

| | before | after |
|---|---|---|
| structures | 2 — credit `directional_spread_to_target` + debit `debit_spread_to_target` (15 DTE) | 1 — debit `debit_spread_to_target` (15 DTE) |
| `confidence_badge` | `mixed pattern — no clear direction` on both | none |
| `filter_mode` | strict | strict (kept, audit only) |
| post-touch block | label / fractions / Wilson CIs | same fields + `advisory_only: true` + `advisory` (label, n, timeframe, direction, fraction, Wilson bounds) |
| card text | "Pattern: stepping-stone. Direction signal: debit-to-target supported." | "post-touch pattern: stepping-stone · n=25 · t15 above 56%" |
