# CR-DW — GEX close-pin test (train, read-only)

Read-only research CR. No DB writes; no local server (CR-DU owns the server queue). Branch
`feat/CR-DW-gex-close-pin` off `origin/main` `4ae205b`, own worktree (`.claude/worktrees/cr-dw`). Kickoff: Ryan,
2026-10-31 (chat). Frozen 2026-10-31 before any CR-DW data was read.

---

## Kickoff (verbatim)

```
CR-DW — GEX close-pin test (train, read-only)

Kickoff: git fetch && git checkout -b feat/CR-DW-gex-close-pin origin/main, in its own worktree, as the first action. No local server; read-only DB only (CR-DU owns the server queue). Read CLAUDE.md, components/field-conventions.md (orats_oi_gamma, settlement rows, CR-BT), research/standing-rulings.md, and the "GEX-bracket short premium" section of research/strategies-proposed/Strategy bench.md. Session note + spec freeze at specs/CR-DW-gex-close-pin.md with the rulings conformance table (research CR).

Hypothesis (Ryan): when large GEX levels sit well inside the day's expected range, SPX tends to settle near them.

Data rules: train only, sessions 2024-04-03 → 2026-06-05 (the forward-like stock_price era, CR-BT). No holdout reads. Settle on SPX cash (spx_cash.settlement_prints, frame A). Levels in strike space (no frame conversion). Exclude AM-settled days and early closes, as the harness does.

Step 0 (all of these; STOP after):
Q1. Reproduce the chart's "yellow / darkest" levels in strike space: which expiries are aggregated, the gross ≥ 40 filter, the |net| ≥ p95 colour span. Show 3 example days, chart vs your reproduction.
Q2. Count the train days with ≥ 1 large level within 0.5 × implied_move_1d of the 06:30 PT SPX open (the condition), and how many large levels per day.
Q3. Propose the null and show it's fair: (a) shift the day's whole level set by a uniform random offset within ±1 × IM, 1,000 draws per day; (b) whether the settlement is nearer the nearest large level than the 06:30 open was.
STOP for approval of Q1–Q3.

Pre-declared primary test: on condition days, the median over days of |settle − nearest large level| / IM, against null (a). One-sided permutation p, plus the effect size in SPX pts. Secondary: (b)'s fraction > 50% with a 90% CI; the same metrics on non-condition days; split by year (2024 / 2025 / 2026).
Report-only: RTH range / IM on condition vs other days.
Results tables, session note, PR. No condor P&L in this CR. If the pin passes, a follow-up CR prices a level-centred condor from cached quotes.
```

---

## Definitions (frozen; Step 0 may amend with Ryan's approval)

- **D1 Session universe.** NYSE sessions (`packages/shared/sessions.py`) 2024-04-03 → 2026-06-05, **subject to the
  B4 stop below**. Excluded: third-Friday AM-settled SPX monthly dates and early closes (no harness settlement), as
  `spx_cash.settlement_prints` / `condor_0dte` do. Holdout (2026-06-06 → 2026-10-26) and forward log never read.
- **D2 Settle.** `spx_cash.settlement_prints` — last clean SPX cash print 12:50–13:00 PT (frame A).
- **D3 Open.** First clean SPX cash print of the session from `spx_cash` (≥ 06:33 PT; the kickoff's "06:30 PT open";
  prints at 06:30–06:32 are mostly the prior close, field-conventions / CR-BH).
- **D4 IM.** `implied_move_1d` from `bt_daily_features` (canonical `v0.6.1-nyse-sessions`), SPX points.
- **D5 Large level.** The chart's yellow (call side) / darkest indigo (put side) levels as reproduced in Q1, keyed
  in **strike space** (kickoff), from `orats_oi_gamma` rows for the session (`trade_date` = session, all live
  expiries `expir_date ≥ trade_date`, ticker SPX): gross = |Σ call GEX| + |Σ put GEX| ≥ 40 $bn, chart band and
  5-pt spacing rule, |net| ≥ the day's p95 of |net| over the filtered set. Exact rule pinned by Q1.
- **D6 Condition day.** ≥ 1 large level within 0.5 × IM of the open (D3).
- **D7 Primary statistic.** On condition days: median over days of |settle − nearest large level| / IM, vs null (a):
  the day's whole large-level set shifted by U(−IM, +IM), 1,000 draws per day; one-sided permutation p (observed
  median smaller than null medians); effect size in SPX pts (observed − null median of |settle − nearest|).
- **D8 Secondary.** (b) fraction of days with |settle − nearest| < |open − nearest|, 90 % Wilson CI vs 50 %; the same
  metrics on non-condition days; by year 2024 / 2025 / 2026. Report-only: SPX cash RTH range / IM, condition vs other.
- **D9 Verdicts (B1).** pass / inconclusive / refuted on the CI / p, defined at the Q3 approval.

---

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-31)

| # | Mark | How |
|---|---|---|
| A1 | honoured | levels are tested as where price may settle (a measurement), not as a trade |
| A2, A3, A4 | n/a | no range, entry or stop; no trade in this CR |
| A5, A6, A14, A16, A17, A19, A20 | n/a | no order flow or skew |
| A7 | n/a | no approach leg |
| A8 | honoured | call-side (yellow) and put-side (darkest) levels reported separately as a readout beside the pooled primary |
| A9, A10, A18, A21 | n/a | not tested |
| A11, A13, A15 | n/a | no target, sizing or exits |
| A12 | **deviates — placement; pre-approved in the kickoff?** | population honoured (yellow / darkest = net beyond the day's p95 \|net\|, gross ≥ 40 display filter). A12 says *Corrected placement*; the kickoff says *strike space (no frame conversion)*. For a 0DTE strike the two agree in SPX cash; for longer expiries the cash-space gamma peak sits at ≈ strike × S / F (below the strike). Treated as Ryan's explicit choice in the kickoff, flagged for confirmation; Q1 measures the gap |
| A22 | honoured | FOMC decision days reported as a separate cell, not pooled into the primary |
| B1 | honoured | pass / inconclusive / refuted |
| B2 | honoured | one pre-declared primary; all secondaries reported |
| B3 | honoured | one hypothesis |
| **B4** | **deviates — STOP** | B4: train ≤ 2026-06-05, but **R-a (2024-04-03 → 2024-12-31) is reserved for confirmation and R-b (2025) is untouched until final confirmation**. The kickoff's window 2024-04-03 → 2026-06-05 reads both. Only 2026-01-02 → 2026-06-05 is plain train. Holdout and forward log are not read. Waiting for Ryan's approval in chat before any 2024 / 2025 data is read |
| B5 | n/a | no ES flow or minute bars used (settle and open are SPX cash, frame A) |
| B6 | n/a | no exits |
| B7 | n/a / partly | the null is a shifted level set on the same day, not a placebo event set; the open's position in the range is identical across null draws by construction |
| B8 | n/a | no labels |
| C1, C2 | n/a | no UI |

Revoked / superseded: R6 (`discounted_level` placement) not used. R1–R5, R7 not relevant.

---

## Step 0 findings (2026-10-31; code at `cf736f6`; read-only; **2026 train sessions only** — B4 stop holds 2024 / 2025)

Scripts: `scripts/cr_dw/q1_levels.py`, `scripts/cr_dw/step0_2026.py` (logs `scripts/logs/cr_dw/`, git-ignored).
No outcome read: `settlement_prints` was used for date membership only; Q3 uses synthetic settles.

**Universe (2026-01-02 → 2026-06-05):** 107 NYSE sessions; 5 have no `implied_move_1d` (first session after a
closure: 01-02, 01-20, 02-17, 04-06, 05-26); 5 third-Friday AM dates excluded (01-16, 02-20, 03-20, 04-17, 05-15);
no early closes. **97 days.** IM p10 / p50 / p90 = 40 / 64 / 116 SPX pts. FOMC decision days in range (A22 cell):
01-28, 03-18, 04-29.

### Q1 — the chart's yellow / darkest levels

Chart rule (Ironbeam `callbacks.py`, React price chart, Levels: Corrected — default since CR-DH):
1. **Expiries:** every `orats_oi_gamma` row with `trade_date` = session, ticker SPX, `expir_date ≥ trade_date`
   (0DTE through ~1-year, 47–48 expiries), summed per level across expiries.
2. **Level key:** CR-BT corrected, ES frame: strike × ES(15:45 ET bar, prior session) / the row's own `stock_price`,
   rounded to 1 pt. Each expiry's strike lands at a different level.
3. **Band:** ES session bars (prior 15:00 PT → 13:00 PT) low − 150 / high + 150. On a past day this uses the whole
   session's range (look-ahead); live at 06:30 it is the bars so far.
4. **Gross filter:** |Σ call GEX| + |Σ put GEX| ≥ 40 $bn (the slider), then greedy 5-pt spacing by gross, max 80.
5. **Colour:** span = p95 of |net| over the drawn lines. Yellow (`#fef08a`) = net ≥ p95; darkest (`#312e81`) = net ≤ −p95.

Re-implementation vs the chart's own functions: identical on all 3 days.

| Day | Chart (ES, corrected) | → SPX cash (− ES−S0 basis) | Strike space (kickoff) | Strike's cash-space gamma peak − strike |
|---|---|---|---|---|
| 2026-01-22 (basis 32.1) | 6869 D, 6925 D, 6975 Y | 6837, 6893, 6943 | 6800 D, 6850 D, 6950 Y | −43, −24, −17 |
| 2026-03-11 (basis 5.2) | 6704 D, 6904 Y | 6699, 6899 | 6600 D, 6700 D, 6800 D | −46, −40, −40 |
| 2026-05-27 (basis 16.2) | 7521 Y, 7546 Y | 7505, 7530 | 7500 Y, 7530 Y, 7600 Y | −41, −2, −34 |

**Strike space is not the chart's level set.** Keyed by strike, all expiries at a round strike stack up (p95 roughly
doubles, 110 → 218 $bn on 01-22), so the large set moves to round 50 / 100 strikes. 2 of 7 chart levels match a
strike level (05-27 7530; 03-11 6700 ≈ 6699). For strikes whose gamma is mostly long-dated, the cash-space gamma peak
is 17–46 pts **below** the strike (the A12 / CR-BT point), comparable to IM (median 64).
The cash equivalent of the chart is CR-BU's placement (strike × S0 / F_e, S0 = nearest expiry's `stock_price`): the
chart's set shifted by the ES−S0 basis, so **same yellow / darkest population in SPX cash**.

Band: a pre-open band (ES bars up to 06:30 PT ± 150) gives the same large set as the full-day band on 87 / 97 days
(cash-corrected) and 83 / 97 (strike). Proposed: pre-open band (what the chart showed at 06:30; no look-ahead).

### Q2 — condition days (2026 only; 2024 / 2025 not counted, B4)

| Placement (pre-open band) | Condition days (≥ 1 large within 0.5 × IM of open) | Large levels per day | Large within 0.5 IM on condition days |
|---|---|---|---|
| Cash-corrected (chart's set) | **53 / 97** (35 yellow-only, 18 darkest-only) | 0: 2, 1: 6, 2: 71, 3: 17, 4: 1 | 1: 43, 2: 9, 3: 1 |
| Strike space | **62 / 97** (37 yellow-only, 25 darkest-only) | 0: 2, 2: 9, 3: 85, 4: 1 | 1: 49, 2: 13 |

Open = first clean `spx_cash` print ≥ 06:33 PT (D3).

### Q3 — the null, checked for fairness

Check: 300 simulated no-pin worlds on the condition days. Settle* = open + IM × z, z ~ N(0, 0.75), independent of the
levels. The primary statistic is tested against each null. A fair null rejects ≈ 5 % at p < 0.05.

| Null | Cash-corrected (53 d) reject rate / median p | Strike (62 d) |
|---|---|---|
| **(a) shift the level set by U(−IM, +IM)** (kickoff) | **23 %** / 0.14 | **15 %** / 0.20 |
| **(c) move swap: settle** = open + IM_d × z_k, z_k drawn from the condition days' own (settle − open) / IM** | 3.7 % / 0.52 | 4.7 % / 0.51 |

**(a) is not fair:** condition days are chosen to have a level near the open, and settles stay near the open, so the
real levels are closer to the settle than shifted ones even with no pin. (a) finds a "pin" 3–5× too often.
**(c) keeps each day's level-to-open geometry** and asks only whether the actual move ends nearer the levels than
the same days' moves would at random. Proposed primary null: (c), one-sided permutation p, 10,000 draws. (a) kept as
a labelled readout.

**(b) settle nearer the nearest level than the open was:** with no pin the expected fraction is **36–37 %**
(p05–p95 25–47 %), not 50 %. That is because the open already sits near a level on condition days. A 50 % bar would
refute a real pin. Proposed: compare (b) with its move-swap null fraction, not 50 %, and keep 50 % as a readout.

STOP: B4 window, placement, null (c) and the (b) bar need Ryan's approval before any 2024 / 2025 read or outcome.
