# CR-CZ — Blind review set from CR-CP's flow-called ranges

Write: `setup_candidates` only, backfill protocol. Train only. Branch `feat/CR-CZ-review-cp-ranges` off `origin/main`
`625d2ec`, own worktree (`.claude/worktrees/cr-cz`). Kickoff: Ryan, 2026-10-27 (chat). Frozen 2026-10-27 before any
CR-CZ data was read.

---

## Kickoff (verbatim)

```
CR-CZ — Load a blind review set from CR-CP's flow-called ranges (write: setup_candidates only,
backfill protocol). Train only.

Branch first: git checkout -b feat/CR-CZ-review-cp-ranges off origin/main. Own worktree.
Read CLAUDE.md (backfill protocol), standing-rulings.md (B4, B8), the CR-CC spec (review tool,
setup_candidates schema, loader scripts/load_setup_candidates.py) and the CR-CP spec.
Set: CR-CP's D2 flow-called ranges on 2026 train (the 235 calls as frozen after A1), random sample
of 120 (seed 20261027), stratified half short-calls / half long-calls. One card per range:
  entry_ts_pt = the decision bar (so replay stops BEFORE the outcome), direction = the D2 side,
  level/zone meta = the box (high/low by closes), range start, approach leg, in-range delta share,
  progress, contracts/pt ratio; outcome = both races (called side AND opposite side, ±10 and ±20),
  hidden until reveal as CR-CC does.
set_name 'cr-cz:cp-ranges', cr_id 'CR-CZ'. Insert-only, run under backfill_run('CR-CZ').
Labels: Ryan uses Take / Skip / Unsure as built; reasons must include two new tags available in the
panel's reason list: "take continuation (other edge)" and "take fade (called side)". If adding reason
tags needs a frontend change, make it the smallest change (reason list only), behind no toggle,
Playwright unchanged elsewhere.
No analysis in this CR. Report counts loaded; PR (loader + any reason-tag change); don't merge.
```

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

This CR builds review cards; it tests nothing. The A block describes setups and tests, so it is n/a except where the
cards carry CR-CP's definitions unchanged.

| # | Mark | How |
|---|---|---|
| A1–A21 | n/a | no test. The cards carry CR-CP's frozen A1 definitions (D0–D5) unchanged; nothing is re-defined or tuned |
| B1, B2, B3 | n/a | no verdict, no parameter search |
| B4 | honoured | train only: every card's session ≤ 2026-06-05 (asserted); CR-CP's train reads only; holdout, R-a, R-b and the forward log not read |
| B5 | honoured | outcomes recomputed from the Databento 1-min bars CR-CP used (`es_flow_1m_databento`); the chart replay uses the review tool's bars as built |
| B6 | honoured | the ±10 / ±20 races are CR-CP's stop-free direction races, labelled as such in `outcome`, not a model of Ryan's exit |
| B7 | n/a | no baseline |
| B8 | honoured | Code writes no reviews; no Playwright writing test is run; cards load unreviewed |
| C1 | honoured | no colours changed |
| C2 | **deviates — pre-approved in the kickoff** | the two new reason chips appear in the existing panel's list for every set, with no toggle ("behind no toggle", Ryan). Recorded, not stopped for |

Nothing from "Revoked / superseded" is used (R1–R7: no entry rule is modelled; the races start at CR-CP's D4 edge fill).
