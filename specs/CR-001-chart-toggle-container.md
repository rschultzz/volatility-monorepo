# CR-001: Chart Toggle Button Container

## Status
Proposed — 2026-05-16

## Problem
The Price chart page has three toggle buttons (Smile, Signals, GEX) that
collapse/expand their respective sub-charts. They are currently rendered as
three separate elements at the top level of the Price chart component, with
layout/alignment handled ad-hoc on each one. This made alignment painful to
get right and makes adding a fourth toggle in the future a layout problem
rather than a one-line change.

## Proposed Solution
Extract the three toggle buttons into a single container component
(`ChartToggleBar`) that owns their layout. The parent Price chart component
should render one `<ChartToggleBar />` element instead of three individual
buttons. Adding a new toggle in the future should be a matter of adding it
to the container's children/config, not editing the parent's layout.

- New component lives alongside the Price chart component
  (e.g. `ChartToggleBar.jsx` in the same folder).
- Container handles horizontal layout and spacing between buttons.
- Each toggle still receives its `isOpen` state and toggle handler as props
  from the parent (no state lifted, no behavior changes).
- No business logic moves — this is a structural refactor only.
- Default state on page load: all three sub-charts collapsed (toggles
  closed). This is the current behavior and must be preserved.

## Affected Files
- The Price chart component file (locate via grep for the Smile/Signals/GEX
  button labels).
- A new file for `ChartToggleBar` in the same directory.
- If the toggle buttons currently import from a shared style file, note
  that in the PR description for future cleanup — do not refactor styles
  in this CR.

## Acceptance Criteria
Functional (no regression):
- All three buttons toggle their charts identically to before.
- Default state on page load: all three charts collapsed.
- Button labels and any icons are unchanged.

Structural (the actual goal):
- The Price chart component renders exactly one element for all three
  toggles (one `<ChartToggleBar />`, not three sibling buttons).
- All toggle layout/spacing CSS lives inside `ChartToggleBar`, not in
  the parent.
- Adding a hypothetical fourth toggle would not require editing the
  parent component's JSX or CSS.

Visual:
- The toggle bar looks identical to its current rendered state. Spacing
  between buttons is consistent and matches the current visual.

## Verification
The repo does not currently have an automated test framework. Verification
is therefore a combination of build checks and a manual click-through by
the human reviewer.

Automated (Claude Code runs these):
- Inspect `package.json` and run every check script defined there
  (commonly `build`, `lint`, `typecheck`). Report which exist, which pass,
  and which (if any) fail.
- If no check scripts exist beyond `dev`/`start`, run `npm run build` (or
  equivalent) and confirm it completes cleanly.

Manual (Claude Code writes this checklist into the PR description for the
human reviewer to walk through):
- Start dev server, navigate to Price chart.
- Confirm all three sub-charts are collapsed on initial load.
- Click Smile → its chart expands. Click again → collapses.
- Repeat for Signals and GEX.
- All three buttons visually aligned in a row, spacing consistent.

## Out of Scope
- Changing any of the three sub-charts (Smile, Signals, GEX).
- Changing the toggle behavior, animations, or default states.
- Restyling the buttons themselves.
- Adding a fourth toggle (this CR only enables that, doesn't do it).
- Setting up a test framework — separate concern, separate CR.
- Refactoring shared styles — separate concern, separate CR.

## Handoff Prompt for Claude Code
> Read `specs/CR-001-chart-toggle-container.md` and implement it on the
> current branch (`feat/CR-001-chart-toggle-container`). Follow the spec
> closely. Run all verification commands listed in the Verification
> section. Commit changes in logical chunks with clear messages. When
> finished, write the manual verification checklist into the PR
> description (or print it so I can paste it there).
>
> If you find something in the code that suggests the spec is wrong or
> ambiguous, stop and ask before changing scope.