# CR-DK — Review panel + chart small fixes (display only)

Spec copied from the vault session note `sessions/2026-10-28 - CR-DK — Review and Chart Fixes.md` (Ryan's
kickoff, chat 2026-10-28).

Branch `feat/CR-DK-review-and-chart-fixes` off `origin/main` `dffa3c8`, worktree `../volatility-monorepo-cr-dk`.
CR-DJ (data patch) may be running in its own worktree and is not touched. One DB-backed local server at a time.

## Spec (Ryan, chat 2026-10-28)

1. **Reason chips** (SetupReview panel):
   - a. The chip stored as "take continuation (other edge)" is **displayed** as "take opposite".
     - The stored string is unchanged (`apps/web/modules/SetupReview/service.py` and any analysis that counts by
       it); only the label shown changes.
     - Hover tooltip: "You'd take the opposite trade (approach direction, at the other edge of the box)."
   - b. The chip "take fade (called side)" is no longer shown for new labels (Take covers it).
     - Existing saved reviews that carry it still display it (read-only, as saved); nothing in the DB changes.
   - c. Update tests that assert the chip list (`test_routes.py`, `cr-db.spec.js`) to the new display; keep
     assertions that the stored strings are unchanged.
2. **Notes list shows the pre-reveal comment.**
   - When a review has a pre-reveal label comment (`setup_reviews.post_reveal = false`, comment not null), show it
     as the first entry under Notes, tagged "before reveal", and count it ("Notes (2)" for one pre-reveal comment
     plus one post-reveal note).
   - The label stays locked (CR-DE); display only.
   - The "Saved:" line keeps the label and reasons and can drop the duplicated comment text.
3. **Price chart, Levels: Corrected with no corrected lines** (pre-forward era / no ES snapshot / no rows):
   - Show a small notice where the GEX lines would be, e.g. "No corrected GEX levels for this date (pre-forward
     era) — switch Levels to Current".
   - Use the reason the API already returns. Nothing changes on dates that have lines.
4. **Flow-pane crosshair:** snap to the bar under the cursor using the same indexing as CR-DG's price-chart tooltip
   fix (with a day either side the flow pane may sit one bar off). Check and fix if so.
   - **Amendment 1 (Ryan, chat 2026-10-29):** item 4 also covers the flow pane's **scroll**. With a day either side,
     the flow pane sat one bar off the price chart (the price chart's logical range counted GEX time points); the
     shared candle-index mapping (`bdcdd3e`) aligns it. Approved as part of item 4; C2 approval unchanged.

## Standing rulings conformance (UI CR)

Read `research/standing-rulings.md` before the freeze. UI CR, no research claims.

| Ruling | Status | Note |
|---|---|---|
| A1–A22 | n/a | no setup, level, range, entry, exit or outcome logic |
| B1–B7 | n/a | no test of a hypothesis, no data split read |
| B8 — Code smoke labels are never Ryan's | honoured | no labels written by tests to `setup_reviews`; smoke runs against a read-only backend; the chip test captures the request and does not write |
| C1 — no green / red; Dash palette | honoured | no new colours (notice and tag use existing slate / grey text styles) |
| C2 — new display features off by default, existing views unchanged | **deviates — approved by Ryan in chat 2026-10-28** | visible changes limited to items 1–4. Item 1b removes a chip from new labels (an existing view changes); change-log line added to `standing-rulings.md` |

Nothing in "Revoked / superseded" (R1–R7) is used.

## Step 0 (commit before implementation)

- List every place the two chip strings are used (frontend, backend, analysis scripts, tests).
- Confirm the CR-CZ analysis plan counts by stored string.
- Confirm item 4 reproduces on origin/main (2026-10-27, a day either side, cursor on the 07:22 PT candle).

## Commits

spec freeze → Step 0 → one commit per item → smoke + wrap.

## Smoke (read-only backend)

- The chip shows "take opposite" and the API still receives the old string (network capture, no write).
- An old review with "take fade (called side)" still displays it.
- 2026-01-16 09:32 card shows Notes (2) with the pre-reveal comment first.
- 2024-03-15 in Corrected shows the notice.
- The flow-pane crosshair matches the price chart on 2026-10-27 with a day either side.
- Existing specs green.

## Wrap

Session note, PR (don't merge), logs copied. Don't deploy.

## Step 0 findings (read-only; one read-only DB connection, one read-only smoke backend)

### 0a. Every use of the two chip strings

`grep -rn` over the worktree (excluding `node_modules`), plus the main checkout's untracked `scripts/` and the vault:

| Where | Use |
|---|---|
| `apps/web/modules/SetupReview/service.py:34-35` | `DEFAULT_REASONS` (CR-CZ). `known_reasons()` returns `DEFAULT_REASONS` + every other reason used in a non-fixture review, sorted; that list is the panel's chips (`GET /api/setup-review/candidates` → `reasons`) |
| `apps/web/modules/SetupReview/tests/test_routes.py:70-71, 262` | chip-list order (…, take continuation, take fade, other) and a round-trip test that both strings are stored as sent |
| `react_price_preview/tests/e2e/cr-db.spec.js:25` | `CHIPS` — both chips must be visible and unclipped in the footer |
| `react_price_preview/src/components/SetupReviewPanel.jsx` | **no literal**: it renders `body.reasons` as given (chips, "Saved:" line, Notes, Summary "Skip reasons") |
| `specs/CR-CZ-review-cp-ranges.md`, `specs/CR-DB-review-card-compact.md` | prose only |
| analysis scripts | **none** (worktree `scripts/`, main checkout's untracked `scripts/`) |
| vault | CR-CZ session note and the standing-rulings change log (prose) |

DB (read-only): "take continuation (other edge)" is stored on **2** reviews (non-fixture); "take fade (called side)" on **0**.
So no saved review shows "take fade" today; the smoke's "old review still displays it" case is checked by
intercepting the candidate GET in the browser and adding the string to a saved review (no DB write).

**Implementation consequence for 1b.** Removing the string from `DEFAULT_REASONS` alone is not enough:
`known_reasons()` re-adds any reason used in a saved review. 1b therefore needs a retired list that `known_reasons()`
never offers, while saved reviews keep and display the string as stored.

### 0b. CR-CZ analysis counts by stored string

There is **no written analysis plan**: CR-CZ's session note says "Analysis of his labels is a later CR". The only
code that counts reasons is `service.summary()` (Skip-reasons tally), which `unnest`s `setup_reviews.reasons` and
groups by the **stored string**. The stored strings are unchanged by this CR, so that tally and any later analysis
count the same values. The panel's Summary tab will show the display label ("take opposite") for the stored string,
with the count unchanged.

### 0c. Item 4 reproduces on origin/main

`tests/e2e/cr-dk.spec.js` "crosshair" (origin/main on :5180, `CRDK_SIDES=main`; output
`scripts/logs/cr-dk/smoke/crosshair-0722-main.json`). Cursor on the 07:22 PT candle of 2026-10-27, at the left
quarter, centre and right quarter of the bar; price tooltip, and the flow pane's dashed crosshair read back through
the flow chart's own time scale:

| Variant | Price tooltip | Flow crosshair bar | Offset (bars) |
|---|---|---|---|
| Flow current, 0 days | 07:22 | 07:22 | 0 |
| **Flow current, 1 day either side** | 07:22 | **07:23** | **+0.99** |
| Flow v3, 0 days | 07:22 | 07:22 | 0 |
| **Flow v3, 1 day either side** | 07:22 | **07:23** | **+1.01** |

**Root cause.** The price chart reports `{logical, shiftedTime}` with `logical` = its own crosshair logical index,
and every flow pane places the line with `logicalToCoordinate(logical)` on its own chart
(`coordinateForLinkedCrosshair`). The price chart's logical index counts every series' time points (GEX segment ends
between sessions, CR-DG), the flow charts only the candles, so with a day either side the two indexes differ.
**The same mismatch shifts the flow pane's scroll:** the price chart reports its visible *logical* range and the flow
panes apply it as is (`setVisibleLogicalRange`), so with a day either side the whole flow pane sits one bar off
(flow 07:22 drawn ≈ 30 px, one bar, left of price 07:22; the dashed line lands under the cursor but over the 07:23
flow bar).

**Fix (item 4, one mapping for both).** The price chart translates its logical index into candle-index space before
reporting it (`timeScale().timeToIndex` of each candle, piecewise linear between candles, slope 1 beyond the ends):
the crosshair reports the index of the candle at `param.time` (the bar the tooltip shows, CR-DG), and the visible
range is mapped the same way. With no extra days the mapping is the identity, so nothing changes there. Flow panes are
untouched.

Not in scope (recorded as an open question): with 0 days, Flow v3's bars sit a constant ≈ 9 px right of the price
chart's (different price-scale widths); the crosshair still lands on the right bar.

### 0d. Items 2 and 3 (what the code does today)

- **Item 2.** `NoteHistory` lists post-reveal rows only and is drawn only when revealed; the "Saved:" line shows the
  blind label, its reasons and its comment. 2026-01-16 09:32 (`CR-CZ:cr-cz:cp-ranges:18`): pre-reveal row with a
  comment and reasons ["take continuation (other edge)", "flow against"], plus one post-reveal note → today
  "Notes (1)". **Rule as built:** the Notes list (revealed cards, as today) starts with the blind review's comment
  tagged "before reveal" and counts it; the "Saved:" line then drops the comment. Before reveal nothing changes (no
  Notes list; "Saved:" keeps the comment, which is otherwise shown nowhere).
- **Item 3.** `/api/ironbeam/bars?levels=corrected` already returns `levels_notes: [{session_date, reason}]`
  (`pre_forward_era` / `no_es_snapshot` / `no_rows` / `error: …`); the React app ignores it. The notice is shown for
  the **centre date** when its note is present and it has no GEX segments; the text names the reason ("pre-forward
  era", "no ES snapshot", "no rows", "error").

### Standing rulings re-check

Unchanged: C2 deviation approved (items 1–4 visible); item 4's scroll fix is the "flow pane may sit one bar off" the
spec names. C1, B8 honoured.

## Smoke (read-only backend `scripts/cr_dk_smoke_backend.py`, port 8064; branch :5181, origin/main :5180)

One DB-backed server (checked `pg_stat_activity` first: 45/103; no other local DB server running). Logs in
`scripts/logs/cr-dk/smoke/`.

**`tests/e2e/cr-dk.spec.js` 7/7:**

- **Chip.** The panel shows "take opposite" with the tooltip; no "take fade (called side)" chip and no raw
  "take continuation (other edge)" chip. Take + "take opposite" + Save: the captured POST carries
  `reasons: ["take continuation (other edge)"]`; the request was fulfilled in the browser (500, never reached the
  backend), and the card's reviews in the DB are unchanged. The API's chip list holds the stored string and not
  the retired one.
- **Retired chip on a saved review.** No saved review carries "take fade (called side)" (Step 0a), so the 2026-01-16
  card's detail is intercepted in the browser to add it: "Saved:" shows it (and "take opposite"); it is still not a
  chip.
- **Notes.** 2026-01-16 09:32 (`CR-CZ:cr-cz:cp-ranges:18`): "Notes (2)", first entry "before reveal" with the
  pre-reveal comment, then the post-reveal note; "Saved: SKIP · take opposite, flow against" without the comment
  (`notes-0116.png`).
- **Levels notice.** 2024-03-15 Corrected: "No corrected GEX levels for this date (pre-forward era) — switch Levels
  to Current" (`levels-notice-2024-03-15.png`); 2024-03-15 Current and 2026-10-27 Corrected: no notice.
- **Crosshair** (2026-10-27, cursor at three points across the bar; branch vs origin/main):

  | Variant | origin/main flow bar · price-vs-flow px | branch |
  |---|---|---|
  | current, 0 days | 07:22 · 0 | 07:22 · 0 |
  | current, 1 day | **07:23 · 30.2** | 07:22 · 0 |
  | v3, 0 days | 07:22 · −9 | 07:22 · −9 |
  | v3, 1 day | **07:23 · 21.2** | 07:22 · −9.4 |
  | v3 5-min, 0 / 1 day | 07:20 · −9 | 07:20 · −9 |

  The flow crosshair is on the price chart's bar and the flow pane scrolls with the price chart; 0 days and 5-minute
  bars are unchanged (v3's constant −9 px is the price-scale width difference, out of scope).
  The first run of this test after the item-4 change failed (output truncated, not kept); every run since passed
  (branch-only, branch + main, full spec).

**Existing specs** (same backend; review writes intercepted by the specs themselves; `cr-cc` not run, it writes):

| Spec | Result |
|---|---|
| cr-db | 5/5 (chips updated by item 1) |
| cr-de | 7/7 |
| cr-dg | 4 + 1 skipped (live, deferred since CR-DG) |
| cr-di | 6/6 |
| cr-dc | 5/6 on the batch run: "unchanged … pixel-identical to origin/main" failed once on the v2 price chart (2026-05-01); re-run alone twice: **2/2 passed** (both modes). Recorded as a one-off |
| cr-df | 6/6 |
| cr-dh (price-chart tests) | 3/3; its Dashboard test matched the filter and failed (needs CR-DH's Dashboard backend at `/`, not run here, as in CR-DI) |

Unit: `npm run test:unit` 47/47 (4 new). Python: SetupReview 17 passed, 8 DB tests skipped (`CR_CC_DB_TESTS=1` against
the DB, temp tables rolled back, was refused by the session's permission classifier; the new
`test_cr_dk_chip_strings_stored_unchanged` is among the skipped).

The other specs rewrote their tracked assets; copies in `scripts/logs/cr-dk/smoke/other-assets/`, committed assets
restored.
