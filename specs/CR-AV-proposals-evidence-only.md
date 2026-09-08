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
