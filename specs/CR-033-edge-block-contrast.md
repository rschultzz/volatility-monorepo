# CR-033 — Proposal card edge-block contrast bump

Single-file, color-only visible-change CR on `react_today_setup/src/components/ProposalCard.jsx`.
No other files. No layout, sizing, spacing, or font-size changes — color values only.

## Problem

In the "Analogue session-close edge" block on each proposal card, the CI/probability
breakdown and the surrounding chrome are too dark against the card background (#0d1527).
The (struct [CI] vs market) data is #334155 and nearly invisible; headers/N/captions/titles
are #475569 and dim. Make the data comfortably legible and the chrome clearly readable,
without touching the colored headline edge values (green #4ade80 / amber #f59e0b / red #f87171).

## Contradiction-stop rule

If any color value below doesn't match what's in the file (someone changed it since),
STOP and surface it before editing.

## Two-tier target palette

- **Data tier** (the probability/CI numbers users need to read) → `#94a3b8` (slate-400)
- **Chrome tier** (labels, headers, N column, captions, section titles) → `#64748b` (slate-500)
- **Untouched:** all edge-state colors (_STATE_COLORS: #4ade80, #f59e0b, #f87171, #475569 no-data),
  the ~ fails-lb amber, every fontSize, every layout/spacing property.

## Exact edits (all in ProposalCard.jsx)

1. **EdgeCell — CI breakdown span** (the main complaint). The span rendering
   `({_fmtProb(structProb)}{_fmtCi(structCi)} vs {_fmtProb(mktProb)})`:
   change `color: '#334155'` → `color: '#94a3b8'`. Leave `fontSize: 8` and `marginLeft: 3` as-is.

2. **TodaysEdgeBlock — block title** ("Analogue session-close edge — this fixed spread"):
   change `color: '#475569'` → `color: '#64748b'`. Leave `fontSize: 9`, uppercase, letterSpacing as-is.

3. **TodaysEdgeBlock — caption line** ("Session outcome windows… ~ = fails lower-bound"):
   change `color: '#334155'` → `color: '#64748b'`. Leave `fontSize: 8`.

4. **TodaysEdgeBlock — table `<th>` headers** (the `['Sess','Touch edge','Close edge','N']` map):
   change `color: '#475569'` → `color: '#64748b'`. Leave `fontSize: 9` and `borderBottom` as-is.

5. **TodaysEdgeBlock — N column cells** (`<td>` rendering `row.n_close`):
   change `color: '#475569'` → `color: '#64748b'`. Leave `fontSize: 9`.

6. **DeltaBlock — section title** ("Strike deltas by session"):
   change `color: '#475569'` → `color: '#64748b'`.

7. **DeltaBlock — TH style constant** (`color: '#475569'`) → `'#64748b'`.

8. **DeltaBlock — footnote `<tfoot>`** ("Long at proposal expiry only…"):
   change `color: '#1e293b'` → `color: '#475569'` (it's even darker than the rest —
   bring it up to chrome-adjacent but still clearly tertiary).

## Do NOT touch

- The `dim` variable / `low_confidence` `#334155` row-label dimming (intentional state signal)
- The `TDmuted` `#334155` in DeltaBlock (intentional "no data" muting, mirrors no-data edge state)
- `_STATE_COLORS` entries
- The `~` flag amber (`#f59e0b`)
- Short (magnet) / Long row labels (`#64748b` already — leave)
- Any `fontSize`

## Step-0 diagnosis

Contradiction check performed against file state at CR-033 start (2026-06-01):

| Edit | Expected existing color | Found in file | Status |
|------|------------------------|---------------|--------|
| 1 — EdgeCell CI span | `#334155` | `#334155` (line 284) | ✅ |
| 2 — TodaysEdgeBlock title | `#475569` | `#475569` (line 307) | ✅ |
| 3 — TodaysEdgeBlock caption | `#334155` | `#334155` (line 312) | ✅ |
| 4 — TodaysEdgeBlock TH headers | `#475569` | `#475569` (line 325) | ✅ |
| 5 — TodaysEdgeBlock N cells | `#475569` | `#475569` (line 355) | ✅ |
| 6 — DeltaBlock section title | `#475569` | `#475569` (line 387) | ✅ |
| 7 — DeltaBlock TH constant | `#475569` | `#475569` (line 380) | ✅ |
| 8 — DeltaBlock tfoot | `#1e293b` | `#1e293b` (line 429) | ✅ |

All 8 pass. No contradictions. Implementation proceeded immediately.

## Commit structure

1. **Spec freeze** — this file.
2. **Contrast bump** — the 8 edits above to `ProposalCard.jsx`.
3. **Smoke + wrap** — `npm run build` clean; `git diff --stat` shows only `ProposalCard.jsx` + spec; PR via merge commit.

## Outcome

All 8 edits landed as specified. Two-tier palette applied:

| Element | Before | After | Tier |
|---------|--------|-------|------|
| EdgeCell CI breakdown span | `#334155` | `#94a3b8` | Data |
| TodaysEdgeBlock block title | `#475569` | `#64748b` | Chrome |
| TodaysEdgeBlock caption | `#334155` | `#64748b` | Chrome |
| TodaysEdgeBlock TH headers | `#475569` | `#64748b` | Chrome |
| TodaysEdgeBlock N cells | `#475569` | `#64748b` | Chrome |
| DeltaBlock section title | `#475569` | `#64748b` | Chrome |
| DeltaBlock TH constant | `#475569` | `#64748b` | Chrome |
| DeltaBlock tfoot footnote | `#1e293b` | `#475569` | Tertiary lift |

`npm run build` clean (51 modules, no warnings). `git diff --stat origin/main` shows exactly
`ProposalCard.jsx` + this spec — nothing else touched. All untouched items confirmed clean:
`_STATE_COLORS` entries (#4ade80 / #f59e0b / #f87171 / #475569), `~` amber, `TDmuted` (#334155),
`dim` (#334155), every `fontSize`, all layout/spacing properties.

Visual smoke: requires live Flask backend (no-data state in dev). The CI breakdown span
(#94a3b8) is now visually in the same legible tier as the `ExpiryLine` and `NetCostLine`
values elsewhere on the card. The green/amber/red headline edge values (700-weight, full
_STATE_COLORS palette) remain the dominant visual anchor. Chrome labels (#64748b) sit
clearly above the old #475569 dim without competing with the headline data.
