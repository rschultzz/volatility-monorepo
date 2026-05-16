# CR-002: Fix missing useCallback import (hotfix)

## Status
Proposed — 2026-05-16

## Problem
The deployed Price chart on Render renders as a blank React card. Browser
console shows `ReferenceError: useCallback is not defined`. The error
originates in code introduced by CR-001 (chart toggle container refactor),
which added a `useCallback`-wrapped handler without updating the React
imports at the top of the file. The bug is hidden locally by Vite's dev
server but surfaces in production builds.

## Proposed Solution
Locate the file that uses `useCallback` without importing it (most likely
`react_price_preview/src/components/PriceChart.jsx`, possibly
`ChartToggleBar.jsx` or `SignalPanel.jsx`) and add `useCallback` to its
existing `react` import. No behavior changes.

## Affected Files
- One of: `PriceChart.jsx`, `ChartToggleBar.jsx`, `SignalPanel.jsx`
  (locate via grep for `useCallback` then check imports).

## Acceptance Criteria
- `npm run build` in `react_price_preview/` completes without errors.
- Built bundle no longer references an undefined `useCallback`.
- No other code changes.

## Verification
- `npm run build` passes locally.
- Manual reviewer check after Render redeploys: load Price chart,
  confirm it renders, confirm browser console is clear of
  `useCallback is not defined`.

## Out of Scope
- Anything beyond fixing the missing import.
- Investigating the "Failed to fetch" errors (those are expected to
  resolve once the render crash is fixed; if they persist, separate CR).

## Handoff Prompt for Claude Code
> Read `specs/CR-002-fix-useCallback-import.md` and implement it on a
> new branch `feat/CR-002-fix-useCallback-import` (branched from
> Main-Live). Run `npm run build` in `react_price_preview/` to confirm
> the fix.
>
> When complete:
> 1. Push the branch to origin.
> 2. Create a PR against `Main-Live` using `gh pr create`. Title:
>    "CR-002: Fix missing useCallback import (hotfix)". Body should
>    include a summary, link to the spec, the commit, and a manual
>    verification checklist as Markdown checkboxes.
> 3. Print the PR URL.
>
> If you find something that suggests the spec is wrong, stop and ask.