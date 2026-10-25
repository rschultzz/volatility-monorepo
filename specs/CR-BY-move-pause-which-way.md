# CR-BY — Move, Pause, Then Which Way

Spec freeze. Branch `feat/CR-BY-move-pause-which-way` off `origin/main` `0e943d8`. Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-BY — Move, Pause, Then Which Way.md` (last_reviewed 2026-10-25). The body below is transcribed verbatim from that note, frontmatter omitted. The kickoff prompt (Ryan, 2026-10-25) confirmed:
- locked decisions L1–L9;
- tests T1/T2 and their pass criteria;
- secondary readouts S1–S9;
- Step 0 gates 1–6.

Kickoff additions:
- f under L4 is fixed from counts only;
- no race outcome is computed outside the A/A splits;
- train only, 2026-01-06 .. 2026-06-05; holdout not read.

CR-BX (branch `feat/CR-BX-capitulation-seller-exhaustion`, `5906bd7`) is unmerged, so its code is copied rather than imported.

---

## Why

**Ryan (2026-10-25):** price often moves fast, then hangs around a level before reversing or continuing. He wants to know whether order flow during the pause predicts which, and whether the predicted outcome happens at all, not only the next few ticks.

**CR-BX family B was dropped at Step 0**, with no outcome computed. Its box (range ≤ 0.5·σ5 ≈ 1.1 pts over 3 min) almost never formed: 11 of 1,831 impulses.

**Ryan's corrections for this CR:**
- the pause scales with the move: a 30-pt drop gives a wider pause than a 10-pt drop;
- it hangs around the level;
- if nothing forms within 30 minutes, move on;
- judge by whether the predicted outcome happens at all.

**Multiple-testing note:** this is the fourth order-flow test on the same 88 train sessions. A pass earns one pre-registered holdout read.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. Read-only, no table writes, no cron, no frontend. |
| L2 | Data and sessions | Exactly CR-BX: `ironbeam_es_flow_1s` (ES, front contract), the 88-session train universe 2026-01-06 → 2026-06-05, CR-BT exclusions, event-level flow-gap rule (> 15 s RTH gap in a used window → drop the event). Holdout not read. |
| L3 | Impulse | CR-BX L4/L5 as frozen, with **k = 3.0** (CR-BX Step 0): a new 300 s extreme at least 3·σ5_d beyond the 300 s opposite extreme, both directions, 15-minute cooldown, 06:35–12:45 PT. **Impulse size I** = the distance from the 300 s opposite extreme to the impulse extreme, in points. **Impulse extreme E** = the new low (down) or high (up). |
| L4 | Pause (box) | Search the 30 minutes after the impulse extreme. Box = the **first** window of **≥ 5 minutes** where (a) high − low ≤ **f·I** and (b) price stays near the level: for a down impulse, every price in the window lies in [E − 0.25·I, E + 0.5·I] (no further extension than a quarter of the move, no retrace beyond half); mirror for up. **f** = the smaller of {1/3, 1/2} giving ≥ 150 eligible boxes in every P tercile (L6), fixed in Step 0 from counts only. No pause within 30 minutes → no event, counted. Box event time t_B = the end of the first qualifying 5-minute window. |
| L5 | Outcome (primary): does it happen at all | From t_B, race to the 13:00 PT close. **Reversal** = price reaches the box boundary on the reversal side + 0.5·I in the reversal direction. **Continuation** = price reaches the box boundary on the continuation side + 0.5·I in the continuation direction. Same second → excluded and counted. Neither by the close → **unresolved**: counted and reported, excluded from the rate. |
| L6 | Predictor | P = D_box / V_box: net aggressor volume inside the box window, signed in the impulse direction (selling inside a box after a down move is positive), over total box volume. Terciles of P over all eligible boxes, directions pooled. **High tercile = pressure into the pause without progress** (absorption). |
| L7 | Placebo boxes | For each event box: up to 5 windows in the same session, ending within ±60 min of t_B, with no qualifying impulse in the prior 30 minutes, that meet the **same** range and location rules, using the event's I and the price at the window start as E. Direction from the sign of the 300 s change before the window. Same race and predictor. |
| L8 | Bootstrap | Session-cluster, 10 000 resamples, seed 20260920, 95 % percentile. |
| L9 | Run discipline | Committed before any outcome. One run. |

## Tests and pass criteria (locked)

| Test | Statistic | Pass |
|---|---|---|
| **T1 flow in the pause → reversal** | reversal rate, high P tercile − low P tercile (resolved boxes) | ≥ +8 pts, CI lower > 0, ≥ 150 resolved per tercile |
| **T2 after a move vs any pause** | reversal rate of high-P event boxes − reversal rate of high-P placebo boxes | ≥ +8 pts, CI lower > 0 |

**Verdicts:**
- **Both pass** → candidate edge; next is one pre-registered holdout read.
- **T1 only** → flow in a pause predicts direction, but not specifically after fast moves.
- **Neither** → recorded; the order-flow line on this data stops here.

## Secondary readouts (not gated)

- **S1 time to outcome:** distribution of minutes from t_B to resolution, by tercile; share resolving within 30, 60 and 120 minutes.
- **S2 unresolved:** share unresolved by the close, by tercile.
- **S3 trade readouts** ($4.50 round trip, CR-BW fill rules):
  - (a) stop entry 1 tick beyond the box in the T1-predicted direction (high P → reversal side, low P → continuation side), target 0.5·I, stop at the opposite box boundary + 1 tick;
  - (b) the same with an 8-tick target and stop.
  - Net expectancy per trade in ticks, and break-even win rates.
- **S4 pause length:** boxes still holding at 10 and 20 minutes (extend the first window while all L4 rules hold), split by final length; P recomputed over the full box up to the break.
- **S5 move size:** I < 10 pts, 10–20 pts, ≥ 20 pts.
- **S6 direction:** after down moves vs after up moves.
- **S7 at levels:** boxes within 2 pts of a CR-BW A1.1 level (≥ 40 $bn magnitude) vs not; also ≥ 100 $bn.
- **S8 other f:** results at the f not chosen in L4.
- **S9 end of day:** share of boxes closing the session on the reversal side of the box, by tercile, vs placebo.

## Step 0 — diagnosis gate (answer all before the run; contradiction-stop)

1. **Universe and impulses:** confirm the CR-BX universe and impulse counts at k = 3.0 (1,831 events).
2. **Box counts at f = 1/3 and 1/2** (no outcomes): impulses with a box within 30 minutes, the reasons the rest fail (range vs location), eligible boxes per P tercile, and the f chosen. **Stop if neither f reaches 150 per tercile.**
3. **Geometry:** median I, box range and box duration (points, ticks, minutes); the distribution of target distances 0.5·I.
4. **Placebo feasibility:** placebo boxes per event (median), and events with none.
5. **Worked event** (seed 20261025): the impulse, the box, P, both targets. The outcome stays hidden.
6. **A/A calibration** (CR-BW A1.3 design) on placebo boxes: 200 splits, the T1-style difference. Pass if the CI excludes 0 in ≤ 10 % of splits and the median is within ±3 pts.

## Outputs

- Spec `specs/CR-BY-move-pause-which-way.md`, with Step 0, results and verdict.
- `scripts/cr_by_pause.py`, committed before its run.
- Session note; State of Knowledge update.

## Kickoff prompt for Claude Code

```
Start CR-BY after CR-BX's family A run is finished and committed. First action: git fetch && git checkout -b feat/CR-BY-move-pause-which-way origin/main (confirm CR-BY is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BY — Move, Pause, Then Which Way.md", components/field-conventions.md, and the CR-BX spec (reuse its universe, impulse detector at k = 3.0, flow loader, gap rule, fill/cost model and A/A design; copy code rather than importing from an unmerged branch).
Commit 1: spec freeze to specs/CR-BY-move-pause-which-way.md, verbatim.
Commit 2: Step 0 findings. Answer all six gates; fix f under L4 from counts only. Compute NO race outcomes except inside the A/A splits. Stop and report; stop immediately if any gate fails.
Read-only DB, train only (2026-01-06 .. 2026-06-05), no holdout. Do not push.
```

---

## Step 0 findings (2026-10-25)

Script `scripts/cr_by_step0.py`; log `scripts/logs/cr_by/step0.log` (git-ignored).

**How it ran.**
- Read-only: `get_backfill_db_conn()` (role asserted), `default_transaction_read_only = on` (asserted), `statement_timeout = 600s`, 3 h Python alarm. SELECTs only.
- Train only: sessions ≤ 2026-06-05. Holdout not read.
- CR-BX's universe, σ5_d, impulse detector (k = 3.0), flow loader and gap rule are copied from the unmerged branch `feat/CR-BX-capitulation-seller-exhaustion` (`5906bd7`). The operational choices the spec leaves open (window definition, failure-reason labels, placebo prior window, A/A tercile assignment) are fixed in the script docstring.

**No race outcome was computed.** The script stopped at gate 2, before the placebo and A/A stages.

**Verdict: STOP at gate 2.** Neither f reaches 150 eligible boxes per P tercile. Gates 3–6 were not run, per the kickoff ("stop immediately if any gate fails").

### Gate 1 — universe and impulses — PASS

88 sessions (2026-01-06 → 2026-06-05) and **1,831 impulses** at k = 3.0 (885 down, 946 up). Both match CR-BX Step 0 exactly.

### Gate 2 — box counts (no outcomes) — FAIL

A box is the first 5-minute window within 30 minutes of the impulse extreme that meets both rules:
- (a) range ≤ f·I;
- (b) location: every price within [E − 0.25·I, E + 0.5·I] for a down move, mirrored for up.

Eligibility: no RTH flow gap > 15 s in the box window, and V_box > 0.

| f | impulses with a box | no box: neither rule ever met | no box: range met, location never ("location") | no box: location met, range never ("range") | no box: both met, never in one window | gap drops | eligible | per P tercile |
|---|---|---|---|---|---|---|---|---|
| 1/3 | 95 (5.2 %) | 1,175 | 113 | 383 | 65 | 4 | **91** | 31 / 30 / 30 |
| 1/2 | 339 (18.5 %) | 741 | 547 | 98 | 106 | 8 | **331** | **111 / 110 / 110** |

**f: none.** L4 needs ≥ 150 per tercile, i.e. at least about 450 eligible boxes; the best is 331 at f = 1/2. **STOP.**

**What blocks the pause.**
- **At f = 1/3:** for 1,175 of 1,736 failures, no 5-minute window in the next 30 minutes is narrow enough, and none stays inside the location band either. Range alone is the binding rule on 383 more.
- **At f = 1/2:** the range rule loosens (only 98 fail on range alone). The location band becomes the main constraint: 547 impulses have narrow windows, but none of them stays within 0.25·I beyond the extreme and 0.5·I back. On those days price usually either keeps extending past the band or retraces more than half the move before it pauses.

### Gates 3–6 — not run

They were stopped by the gate 2 failure. No geometry, placebo, worked event or A/A figures were computed.

### What the gate leaves for Ryan (no redefinition proposed)

Under the locked L4, CR-BY cannot reach its pre-registered n on the 88-session train window. Any change to f, the location band, the 5-minute minimum or k is a redefinition and needs a new amendment.

---

## Amendment A1 (Ryan, 2026-10-25, after Step 0 gate 2; no outcome computed)

| # | Clause |
|---|---|
| A1.1 | Replaces L4's location rule (b): the impulse may extend before the pause. E = the most extreme price reached from the impulse until the start of the box window (running extreme); I = the distance from the impulse's 300 s opposite extreme to that E. A window qualifies only if, at no point from the impulse extreme to the window's end, price has retraced more than 0.5·I from the running E. Inside the window, (a) high − low ≤ f·I with the updated I. |

Everything else in L4 (≥ 5-minute window, 30-minute search from the original impulse extreme, f = the smaller of {1/3, 1/2} reaching 150 per tercile, from counts only), and L1–L3, L5–L9, T1/T2 and S1–S9 are unchanged. Targets in L5 use the updated I.

---

## Step 0 under A1 (2026-10-25)

Script `scripts/cr_by_step0_a1.py`: a copy of `cr_by_step0.py` (`1883cc0`) with L4 (b) replaced by A1.1. Log `scripts/logs/cr_by/step0_a1.log` (git-ignored). Read-only (`default_transaction_read_only = on`, `statement_timeout = 600s`), train only, holdout not read. One run.

**No race outcome was computed at any event box.** The L5 race ran only on placebo boxes inside the A/A splits, and only the split statistic is reported.

**Verdict: Step 0 under A1 PASSES.** f = 1/2 (203 / 202 / 203 eligible per P tercile); A/A calibration passes. The run is not started.

### Operational choices fixed before the run (not stated in A1)

- **Running extreme.** From the impulse row t: runE(u) = min low (down) or max high (up) over rows [t, u]. I(u) = |opposite extreme − runE(u)|, where the opposite extreme is the impulse's 300 s opposite extreme.
- **Retrace rule.** Each row u from t to the window end is judged against the move size reached so far: a window fails if any row has retrace(u) > 0.5·I(u). Once the rule is violated, no later window qualifies.
- **Updated I.** For a window ending at e: E_w = runE at the last row at or before e − 300, and I_w = |opposite extreme − E_w|. Rule (a) uses I_w, and so do the L5 targets.
  - Inside the window price may make new extremes, bounded by range ≤ f·I_w.
- **Placebo (L7).** A1 replaced the location rule with a retrace rule anchored on an impulse, which a placebo window does not have. Within any window, retrace from the running extreme cannot exceed the window range, and range ≤ f·I ≤ 0.5·I. So the placebo rule reduces to (a) with the event's I_w.
- **Failure reasons:**
  - "retrace": windows meeting the range rule exist, but all after the first retrace violation;
  - "range": retrace-ok windows exist, but none meets (a);
  - "both": neither kind of window exists.

### Gate 1 — PASS

88 sessions; 1,831 impulses at k = 3.0 (885 down, 946 up).

### Gate 2 — box counts — PASS at f = 1/2

| f | impulses with a box | no box: retrace | no box: range | no box: both | gap drops | eligible | per P tercile |
|---|---|---|---|---|---|---|---|
| 1/3 | 413 (22.6 %) | 840 | 82 | 496 | 7 | 406 | 136 / 135 / 135 |
| **1/2** | **615 (33.6 %)** | 1,087 | 3 | 126 | 7 | **608** | **203 / 202 / 203** |

- **f chosen: 1/2** (L4: the smaller f reaching 150 per tercile; 1/3 reaches 136).
- P cut points −0.0061 / +0.0244. P percentiles 5 / 25 / 50 / 75 / 95: −0.054 / −0.015 / +0.009 / +0.035 / +0.074.
- Eligible down / up: 248 / 360.
- At f = 1/2, the remaining failures are almost all retrace (1,087): price gives back more than half the move before any 5-minute window settles.

### Gate 3 — geometry (608 eligible boxes, f = 1/2)

| | median | p5 / p25 / p75 / p95 |
|---|---|---|
| updated I | **11.50 pts** (46 ticks) | 6.34 / 8.69 / 15.75 / 25.00 |
| box range | 5.50 pts (22 ticks) | 3.00 / 4.00 / 7.50 / 12.41 |
| target distance 0.5·I | **5.75 pts** (23 ticks) | 3.17 / 4.34 / 7.88 / 12.50 |

- Box duration is fixed at 5.0 min (the first qualifying window). Impulse → box end: median 6.5 min (IQR 5.2–8.1).
- **A1 extension.** I_w / I_0 median 1.29 (p75 1.65, p95 2.50); I_0 median 8.50. The typical impulse extends about 30 % further before it pauses.
- Box range / I median 0.49: most boxes sit near the f = 1/2 cap.
- S5 split: I < 10 pts 216, 10–20 pts 311, ≥ 20 pts 81. Boxes ending after 12:30 PT: 26.

### Gate 4 — placebo feasibility

- **475 placebo boxes for 608 event boxes; per event median 0 (mean 0.78).**
- **Events with none: 452.** Events with fewer than 5: 557.
- Placebo per P tercile (event cut points): 187 / 101 / 187. **High-P placebo boxes: 187** (the T2 comparison set).
- Why so few: placebo windows must fall within ±60 min of t_B, have no impulse condition in the 35 minutes before their end, and keep their range ≤ 0.5·I of a large move inside a quiet stretch. Such windows cluster on the few quiet sessions.
- Spread across 150 parent events (A/A), the placebo is concentrated rather than matched event by event.

### Gate 5 — worked event (seed 20261025; outcome hidden)

- **Impulse.** Session 2026-01-26 (`XCME:ES.H26`), σ5_d 2.00. **Up** impulse at 07:48:20 PT. Opposite extreme 6976.00, impulse extreme E0 6983.75, I0 7.75 (3.88 σ5).
- **A1 update.** Running extreme at box start E = 6985.00, updated I = 9.00. Retrace limit 0.5·I(u) (4.50 at box start); range cap f·I = 4.50.
- **Box.** 07:49:44–07:54:44 PT, found 6.4 min after the extreme: low 6982.75, high 6987.25, range 4.50 (at the cap). Price made a new high (6987.25) inside the window, as A1 allows.
- **Predictor.** D_box +686 (buying, signed in the impulse direction), V_box 9,274 → **P = 0.0740, high tercile** (absorption).
- **Targets:**
  - reversal 6978.25 (box low − 4.50);
  - continuation 6991.75 (box high + 4.50).
- The path is in the log, in 1-minute slots from 5 min before the impulse to the box end.

### Gate 6 — A/A calibration — PASS

- Race-window gap drops −6; ties or unresolved by 13:00 −11. **Resolved placebo 458** (per tercile 179 / 98 / 181) in 150 parent events.
- 200 splits by parent event; each placebo box sits in its own P tercile (event cut points). Statistic: [high − low reversal rate](half 1) − (half 2).

| | value |
|---|---|
| median | **−0.85 pts** (gate ±3) |
| mean / sd | −0.71 / 12.04 |
| p5 / p95 | −19.73 / +19.26 |
| **CI excludes 0** | **4.5 % (9 / 200)** (gate ≤ 10 %) |

**Note for the run (not a redefinition).** The split sd is about 12 pts, with roughly 90 high-P placebo per half. With 187 high-P placebo boxes against 203 high-P event boxes, T2's CI will be roughly ±12 pts, so T2 (≥ +8, CI lower > 0) is powered only for large effects.
