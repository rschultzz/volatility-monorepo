# CR-CQ — GEX Min |GEX| filter defaults to 40 $bn and never silently resets to 0

Spec freeze. Branch `feat/CR-CQ-gex-filter-default-40` off `origin/main` `4075d27` (PR #96 merge, CR-CO), in its own worktree (`../volatility-monorepo-CR-CQ`). Display only: no research data, no DB writes. PR, don't merge; Ryan merges and chat deploys Dash.

Source: the kickoff prompt (Ryan, 2026-10-26), transcribed as K1–K7.

## Why

Ryan: "many times when I refresh the app the filter goes back to 0 and it's too much on the screen."

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Visible change in scope: the default value of the price chart's Min \|GEX\| filter. Nothing else on screen changes (CLAUDE.md visible-change discipline; standing-rulings C2 n/a — an explicit default change Ryan asked for). |
| K2 | Step 0 | Diagnose before fixing: every source of the filter value and their precedence — `react_price_preview/src/App.jsx` (`coerceGexMinAbsB` fallback 10, localStorage `ib-react-gex-min-abs-b`, the `gex_min_abs_b` URL param) and the Dash side (`apps/web/modules/Ironbeam/callbacks.py` ~4730–4761: `indicator_state` `cfg.gex_overlay.min_abs_b` passed into the iframe src, and wherever that store is initialised/persisted). Reproduce the reset to 0. Write the cause into the spec. |
| K3 | Fix 1 | Default 40 everywhere a default exists (React fallback, Dash `gex_overlay` `min_abs_b` default). |
| K4 | Fix 2 | An empty or invalid input never stores 0 — keep the previous value until a valid number is entered. 0 stays allowed only when typed explicitly. |
| K5 | Fix 3 | A refresh shows the last value Ryan set; with nothing stored, 40. |
| K6 | Tests | Unit tests for the coercion/precedence; Playwright: fresh profile shows 40; set 60 → refresh → 60; clear the box → refresh → previous value, not 0; existing suites unchanged apart from the default. |
| K7 | Delivery | PR, don't merge; then Ryan merges and chat deploys Dash. |
