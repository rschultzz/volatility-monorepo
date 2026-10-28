# CR-DE — Review panel: comments that save, post-reveal notes, bigger comment box

Frontend (`react_price_preview` REVIEW panel) plus a small change to the setup-review API
(`apps/web/modules/SetupReview`). No schema change. Branch `feat/CR-DE-review-comments` off `origin/main`
`51c0287`, own worktree (`.claude/worktrees/cr-de`). Kickoff: Ryan, 2026-10-27 (chat). Frozen 2026-10-27
after the Step 0 report and Ryan's rulings, before any code change.

---

## Kickoff (verbatim)

```
CR-DE — Review panel: comments that save, post-reveal notes, bigger comment box (frontend + small API).

Branch first: git checkout -b feat/CR-DE-review-comments off origin/main. Own worktree.
Visible change in scope, REVIEW panel only (CLAUDE.md discipline; C1 palette). B8: only Ryan's labels
count; never write test reviews to the real tables (intercept or use temp tables, as CR-DB did).

Problem (Ryan, 2026-10-27): comments typed after Reveal don't save. DB check: in the last day there are 2
pre-reveal reviews with no comment and 0 post-reveal reviews. The "Save post-reveal review" button looks
disabled. The comment box is one line tall. Ryan wants to save several notes per card.

Step 0: find why post-reveal saves don't happen (button state, request never sent, API rejects, or
silent error) and why the pre-reveal comment wasn't stored; report before fixing.
Fix:
1. Pre-reveal: the comment typed before Save is stored with the label (verify end-to-end).
2. Post-reveal notes: after Reveal, Ryan can save a note at any time, as many times as he wants. Each
   save INSERTS a new post_reveal row (comment, optional reasons, optional changed label) — setup_reviews
   stays insert-only; nothing is updated. The card shows its note history (time + text) under the
   reveal, newest first.
3. Feedback: a visible "saved" confirmation after every save, and an error message if it fails (never
   silent).
4. Comment box: at least 5 lines tall by default, resizable, wraps; Cmd/Ctrl+Enter saves. Footer still
   fits at 900×600 (scroll inside the panel above the footer if needed, the action buttons stay pinned).
5. Remove the "fixture smoke reason" chip (and any other test-only reason) from the chip list.
6. Set filter: remember the last chosen set (localStorage), and default to the most recently loaded set
   when nothing is stored.
Analysis code that reads labels must use the FIRST pre-reveal review per card as the blind label;
post-reveal rows are notes only (document this in the spec and in the reviews API).
Tests: Playwright with intercepted requests — pre-reveal label + comment sends both; two post-reveal notes
send two inserts; confirmation shows; error shows on a forced 500; chip list has no smoke reason; set
filter persists. Existing review tests pass. PR, don't merge; Ryan merges, chat deploys.
```

## Rulings from chat (Ryan, 2026-10-27, after the Step 0 report, before code)

| # | Question | Ryan's answer |
|---|---|---|
| R1 | `setup_reviews.label` is `NOT NULL` (take / skip / unsure). What does a post-reveal note with no new label store? | **Copy the card's blind label** (its first pre-reveal review) when no new label is picked. No schema change. |
| R2 | With "first pre-reveal review is the label", a second pre-reveal Save would be silently ignored. | **Lock the label after the first pre-reveal Save.** Later input goes in as post-reveal notes, and the panel says so. |
| R3 | Printed DB errors. | **Mask the connection string** in any DB error that is printed or returned (as CR-CA's `redact` does), and test that a failing connection's error output contains no password. |

## Standing rulings conformance (`Dash/research/standing-rulings.md`, read 2026-10-27)

Display and data-entry CR; it tests nothing.

| # | Mark | How |
|---|---|---|
| A1–A21 | n/a | no test, no setup definition |
| B1–B7 | n/a | no test |
| B8 | honoured | every Playwright save / reveal is intercepted in the browser; the API tests use a temp-table fake connection; no review is written to `setup_reviews` |
| C1 | honoured | confirmation, error and note history use the Dash palette (blue / slate / light grey); no green / red |
| C2 | deviates — pre-approved | the kickoff itself changes the existing REVIEW panel (bigger comment box, note history, confirmation, set default). Confined to the REVIEW panel; nothing outside it changes. Recorded, as for CR-CZ / CR-DB |
| Revoked R1–R7 | not used | — |

## Design

### Label rule (API and analysis)

- A card's **blind label** is its **first** pre-reveal review (`post_reveal = false`, earliest `reviewed_at`, then
  lowest `id`). Every reader of labels (the queue's label, Summary, Reveal's target row, any analysis script) uses
  that row. Before CR-DE the service used the latest pre-reveal row; no card had two pre-reveal rows (Step 0), so
  no existing number changes.
- **Post-reveal rows are notes only.** Their `label` is the card's blind label unless Ryan picked a different one
  (R1); a different label is a "changed label" note and is never the card's label.
- **Lock (R2):** once a card has a pre-reveal review, the API rejects another pre-reveal review with 409
  `label_locked`. After reveal, every POST inserts a `post_reveal` row.

### API (`POST /api/setup-review/candidates/<id>/review`)

- Pre-reveal: `{label, reasons[], comment}`; label required; first save only (409 after).
- Post-reveal: `{label?, reasons[], comment}`; label optional (R1: defaults to the blind label); at least one of
  a comment, a reason or a changed label is required (400 otherwise). Always an INSERT; nothing is updated.
- `GET /candidates`: `sets[]` carry `loaded_at` (max per set) so the panel can default to the latest set.
- Reason chips: the default list plus reasons used in reviews on non-fixture sets (`set_name` not ending in
  `:fixture`); the CR-CC smoke reason "fixture smoke reason" is gone.
- DB error messages returned to the browser or logged are passed through a shared `redact()` (R3).

### Panel

- Pre-reveal: label + reasons + comment, Save. After Save: "Saved ✓" confirmation; the label buttons and Save
  lock, with the note "Label locked — reveal to add notes".
- After Reveal: the form stays usable as a note form ("Save note"); the label preselects nothing and the note
  copies the blind label unless a label is picked. Each Save shows "Note saved ✓" and clears the comment; the
  note history (time PT + label change + reasons + text) shows under the outcome, newest first.
- Errors show in a visible bar above the buttons; never silent.
- Comment box: `rows={5}`, wraps, vertical resize; Cmd/Ctrl+Enter saves. The footer's content scrolls inside a
  capped region so the action buttons stay pinned at 900×600.
- Set filter: last chosen set in `localStorage` (`dash.review.set`, try/catch); with nothing stored, default to
  the set with the latest `loaded_at`.

## Implementation order

1. API: first-label rule, lock, post-reveal notes (R1), reason filter, `loaded_at`; docstrings; pytest.
2. Redaction (R3): shared `redact()`, used by the setup-review routes; pytest for a failing connection.
3. Panel: notes, feedback, comment box, set filter, chip list.
4. Playwright `cr-de` (intercepted) + existing read-only review tests.
5. Smoke + wrap.

---

## Step 0 findings (2026-10-27, read-only)

DB (read-only transaction): `setup_reviews` holds 10 rows; 2 carry a comment (both CR-CC smoke on `cc:fixture`);
0 are `post_reveal`; no card has two pre-reveal rows. Ryan's recent reviews (rows 20–24, CR-CB round-1/2, CR-CZ
cp-ranges:9) were each revealed 1.5–3.0 s after Save. Render keeps no request logs for these routes.

1. **Post-reveal saves never reach the API — the request is never sent.** `reveal()` calls `resetForm()`, which
   sets `label` to null; Save is `disabled={!label || busy}`. After Reveal the button reads "Save post-reveal
   review" but stays greyed until a label key is pressed, with no hint why. The API would accept it.
2. **No pre-reveal comment was lost.** With Save → Reveal in 2–3 s, no comment was typed between them; the
   comments were typed after Reveal and blocked by (1). The code does send `comment` with the label; the CR-DB
   test only asserted `comment: ''`, so CR-DE adds an end-to-end check with text.
3. **No confirmation exists**; errors show only in a small grey line in the footer.
4. **"fixture smoke reason"** comes from `known_reasons()`: the default list plus every reason in any review,
   fixture sets included (CR-CC smoke, row 10 on `cc:fixture:2`).
5. **Label rule:** the service used the *latest* pre-reveal review (`_STATE_LATERAL`, `summary`, `reveal`). The
   kickoff's rule is the *first*; with no multi-row cards the switch changes no existing number.
6. **Not run:** the two cr-cc tests that save / reveal on `cc:fixture` write to the real tables (B8).
7. **Redaction gap (R3):** `_run()` returns `str(e)` of a failed connect as `message`, and `_db_error` returns
   `str(e)`; psycopg's conninfo errors include the full URL with password, so a malformed `DATABASE_URL` would
   send it to the browser. Step 0's own ad-hoc query printed one such error to the session output.

---

## Smoke + wrap (2026-10-27)

Servers: `CR_SMOKE_REVIEW_READONLY=1 scripts/cr_cc_smoke_backend.py` (port 8061; review connections read-only,
checked `SHOW default_transaction_read_only` = on), branch frontend on 5172, origin/main (`4db4173`) frontend on 5177.
`setup_reviews` held 10 rows (0 post-reveal, 2 with a comment) before and after every run: nothing was written.

| Suite | Result |
|---|---|
| API `test_routes` (mocked + `CR_CC_DB_TESTS=1` on TEMP tables) | 21 / 21 |
| Frontend unit (`node --test tests/unit`) | 17 / 17 |
| Playwright `cr-de` on the final code | 7 / 7 (5 in one run, the 2 set-list tests re-run with 120 s waits) |
| Playwright `cr-db` (5) + read-only `cr-cc` (5) + `cr-de` (7), run 2 | 16 / 17; the failure was the list race below, fixed in step 4b |
| Same 17, runs 3–4 (after step 4b) | timeouts throughout: network RTT to Render 38–253 ms (avg 113), list call 14–23 s, bars 49 s; environmental, not re-run to a clean pass |

Deltas found during smoke (step 4b, no spec change needed beyond this note):
- The set default touches only the Set; CR stays "all CRs". Defaulting the CR too hid other CRs' sets from the Set
  list (broke cr-cc's set selection, and would have hidden sets from Ryan).
- Stale list responses are ignored (sequence guard): the first-load default and a quick set change raced, and the
  slower response could overwrite the list.
- Save / Reveal are usable as soon as the card refreshes; the queue reload is not awaited. After reveal, the next
  note can be written while the card refreshes (stale card refreshes ignored).
- The two cr-cc tests that save / reveal on `cc:fixture` were not run (B8).
