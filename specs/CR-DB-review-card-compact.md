# CR-DB — Review panel: compact card layout

Frontend display only: `react_price_preview` REVIEW panel (`SetupReviewPanel.jsx`, `ReviewMeta.jsx`) and the review
candidate's lines on `PriceChart.jsx`. No data, API or DB changes. Branch `feat/CR-DB-review-card-compact` off
`origin/main` `0fdcd6d`, own worktree (`../volatility-monorepo-cr-db`). Kickoff: Ryan, 2026-10-27 (chat). Frozen
2026-10-27 before any code change.

---

## Kickoff (verbatim)

```
CR-DB — Review panel: compact card layout so labelling is usable (frontend display fix).

Branch first: git checkout -b feat/CR-DB-review-card-compact off origin/main. Own worktree.
Visible change IS in scope, limited to the REVIEW panel's card detail and action area (CLAUDE.md
visible-change discipline). Standing rulings: C1 (no green/red — note the Take/Skip/Unsure buttons
currently look green/red/amber; bring them into the Dash palette: blue / slate / light grey), C2.
No data or API changes, no DB writes.

Problem (Ryan, 2026-10-27, set cr-cz:cp-ranges): the detail view dumps every meta field in a
two-column grid; long text values (entry_rule, box_basis) wrap one word per line, so the card is
unreadable, the Save / reason controls are pushed out of view, and the trade direction isn't
obvious once a card is open.

Fix:
1. Header, large and always visible: date, time, and the trade direction as a clear label
   ("SHORT the top" / "LONG the bottom"), plus the called edge price.
2. A short "key facts" block, one line each: box high / low (by closes) and width; range start and
   duration; approach leg (direction, points); in-range delta share; progress; contracts/pt ratio.
   Use readable labels, not field names.
3. Everything else (all other meta fields, long rule text) behind a collapsed "details" toggle; long
   text renders as normal wrapped paragraphs, never in the narrow value column.
4. Sticky action footer at the bottom of the panel, always visible without scrolling: Take / Skip /
   Unsure, the reason chips (including "take fade (called side)" and "take continuation (other edge)"),
   and Save. Keyboard shortcuts T / S / U unchanged.
5. On the chart: draw the box (high and low lines) and label both edges, so the fade edge and the
   continuation edge are visible.
6. The same layout for every review set; sets without these fields just show fewer key facts.
Tests: Playwright on one cr-cz card at the panel's normal width and at a narrow width (Save and the
chips visible without scrolling, direction label present); existing review tests pass; nothing
outside the REVIEW panel changes (pixel check of the price chart with the panel closed).
PR, don't merge. Then Ryan merges and chat deploys Dash.
```

## Scope rulings from chat (Ryan, 2026-10-27, before code)

| # | Question | Ryan's answer |
|---|---|---|
| Q1 | Which price is the header's "called edge" on cr-cz cards? | **Box edge + limit**: e.g. `SHORT the top · edge 7736.00 (limit 7735.00)`. Edge = `meta.range.top` / `bottom` (box by closes); limit = `entry_price` (D4, edge ∓ 1). |
| Q2 | Which cards get the chart's box edge lines? | **Any card with `meta.range`**, drawn only in REVIEW mode with a card active. Nothing changes with the panel closed. |
| Q3 | C1 outside the card (queue ▲▼ arrows, queue / summary label colours, fade/break rule chips)? | **Whole panel**: shared label colours follow Take / Skip / Unsure; arrows and rule chips blue / slate. |

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

Display CR; it tests nothing.

| # | Mark | How |
|---|---|---|
| A1–A21 | n/a | no test, no setup definition |
| B1–B7 | n/a | no verdict, no data read for analysis |
| B8 | honoured | the Playwright test opens a cr-cz card and presses Take / chips only as local state; it never clicks Save or Reveal on `cr-cz:cp-ranges`, so no smoke label lands in Ryan's set |
| C1 | honoured | Take = blue, Skip = slate, Unsure = light grey; queue arrows and rule chips moved off green / red (Q3) |
| C2 | deviates — approved | the kickoff itself changes the existing REVIEW panel layout (pre-approved), and Q2 adds edge lines to older sets' cards with `meta.range` (approved in chat, Q2). Both are confined to REVIEW mode with a card active; the chart with the panel closed is unchanged (pixel check). Logged in the rulings change log. |
| Revoked R1–R7 | not used | — |

## Design

**Header (always visible, top of the active card).** Large direction label: short → `SHORT the top`, long → `LONG
the bottom` when the card has `meta.range`; otherwise `SHORT` / `LONG`. Next line: session date · entry time (PT) ·
cell. Then the edge: `edge <range edge> (limit <entry_price>)`; without a range, `entry <entry_price>`. The
candidate id is kept, small and muted (the existing review tests look for it in `review-active`).

**Key facts** (one line each, readable labels, present only when the field exists):
- Box (closes): `low – high · width w pts` from `meta.range` (bottom, top, height).
- Range: `from hh:mm, n min` (formed_ts_pt → qualified_ts_pt, else → entry_ts_pt).
- Approach: `up / down leg, p pts` (`approach`, `approach_pts`).
- Delta in range: `x %` (`delta_share_pct`).
- Progress: `p pts` (`progress_pts`).
- Contracts / pt: `ratio ×` with up / down (`cpp_ratio`, `cpp_up`, `cpp_down`).
- For older sets: level (price, net / gross $bn) and stop when present, so those cards don't lose the facts they had.

**Details** (collapsed by default, a "details" toggle): every other candidate column and meta field via `ReviewMeta`.
Long strings (> 40 chars) render as full-width wrapped paragraphs with their label above, not in the value column.

**Layout.** Panel = header bar, filters, then for the queue tab: card header (fixed), a scrolling middle (key facts,
details, saved line, outcome, queue list), and the action footer (fixed, bottom): Take / Skip / Unsure, reason chips,
add reason, comment, Save / Reveal, errors. The footer is the panel's last flex child, so it stays in view at any
viewport height the card header + footer fit in. Keys unchanged (N / P / T / S / U / R).

**Chart.** For a candidate with `meta.range` top / bottom: two price lines. The edge on the trade's side is the fade
edge (short → top, long → bottom), the other is the continuation edge: titles `fade edge` / `continuation edge`, Dash
blue / slate, dashed. Existing lines, markers, the range box and the zone band are unchanged.

**Panel width** stays 340 px (`PANEL_WIDTH`, which the chart uses to offset). "Narrow width" in the tests = a
narrow and short viewport (e.g. 900 × 600), where the panel is the same width and height is the constraint.

## Tests

- New `tests/e2e/cr-db.spec.js` against the CR-CC smoke backend (port 8061): opens one `cr-cz:cp-ranges` card at
  1440 × 900 and at 900 × 600; asserts the direction label, the Save button and both new chips are inside the
  viewport and the panel without scrolling; T / S / U set the label; details collapsed by default and expandable; the
  chart has the two edge lines. Never Save / Reveal (B8).
- Existing `cr-cc.spec.js` and other review specs pass (selectors adjusted only if they reference removed markup).
- Pixel check: screenshot of the price chart with the panel closed on `main` and on this branch, compared.
