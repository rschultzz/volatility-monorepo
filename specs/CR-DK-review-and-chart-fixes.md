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
