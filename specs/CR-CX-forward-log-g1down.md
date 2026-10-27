# CR-CX — Forward log for the G1-down signal (from 2026-10-27; B4 as frozen 2026-10-27)

Spec freeze. Branch `feat/CR-CX-forward-log-g1down` off `origin/main` `625d2ec` (PR #103 merge, CR-CV), in its own worktree (`.claude/worktrees/cr-cx`). Two new insert-only tables, one daily script, tests, a cron deploy plan. PR, don't merge. No display change.

Source: the kickoff prompt (Ryan, 2026-10-27), transcribed as K1–K9. X1–X12 are choices Code fixed before writing any implementation code and before reading any session ≥ 2026-10-27.

---

## Why

B4 (frozen 2026-10-27) makes data from 2026-10-27 onward the **forward log**: recorded daily by a fixed definition, written once, not looked at early, and used only as a pre-registered confirmation. G1-down is the one skew signal that leaned the hypothesised way twice. CR-CL (2026 train, fixed swing 12) gave +17.4 pp [−1.2, +37.3], and CR-CO (2023, vol-scaled swing) gave +13.3 pp [−7.2, +32.0]. Both were INCONCLUSIVE, and neither data set can be re-used to confirm it. This CR starts recording it.

## Kickoff (Ryan, 2026-10-27)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Branch first, own worktree. Read CLAUDE.md (backfill protocol), standing-rulings (B4, A7, A11, A14, A16), the CR-CL, CR-CK and CR-CO specs. Freeze with the conformance table. |
| K2 | Table `forward_log_events` | Insert-only (triggers block UPDATE / DELETE, like `setup_reviews`): signal `'G1-down'`, session_date, leg start / end / confirm times, entry price, ratio, actual / expected skew change, beyond-noise flag, definition version, code SHA, written_at. |
| K3 | Table `forward_log_outcomes` | Insert-only: event id, ±10 race result, ±20 race result, move at +30 / +60, written_at. |
| K4 | Script | `scripts/forward_log_g1down.py`: for ONE session date ≥ 2026-10-27, compute G1-down legs with **CR-CL's frozen code** (vol-scaled swing per CR-CO A1, CR-CK noise band as shipped), insert events, and insert outcomes for any earlier event whose race has resolved. **Refuses any date < 2026-10-27.** |
| K5 | Protocol | Backfill protocol: `dash_backfill_writer`, `backfill_run(conn, "CR-CX")`. |
| K6 | No peeking (B4) | Prints only **counts** (events logged, outcomes logged), never win rates. |
| K7 | Tests | Refuses holdout dates; reproduces CR-CL's events on 3 train dates (computed in memory, not inserted); an idempotent re-run inserts nothing new. |
| K8 | Delivery | PR, don't merge. |
| K9 | Deploy plan (for chat) | Render cron, weekdays 20:30 UTC (13:30 PT), running the script for the current session; needs `BACKFILL_DATABASE_URL` on that service (Ryan sets it). |

## Choices fixed by Code (2026-10-27, before any implementation code and before reading any session ≥ 2026-10-27)

| # | Choice | Value |
|---|---|---|
| X1 | Definition `G1-down v1` | Exactly the CR-CO A1 primary: per session swing_d = 12 × IM_d ÷ IM_train with IM_d = canonical `bt_daily_features.implied_move_1d` and **IM_train = 65.71** (median of the 102 `y26` train sessions; frozen as a constant, re-derived from the DB in a test); legs from `flow_features.swing_legs(H, L, swing_d)`; skew window s = start_i + 1, e = end_i + 1; s ≥ 07:00 PT and **e ≤ 11:00 PT** (primary legs only; late legs are not logged); `skew_expected.stickiness_read` with the shipped `NOISE_BAND_PP`; event = `reason == 'ok'`, beyond noise, **ratio < 0**, **dir = −1** (decline). Hypothesised direction h = +1 (up). Events are computed by calling `scripts/cr_cl_lib.py` `build_legs` / `read` unchanged on a one-session data dict. |
| X2 | Skew input | `orats_monies_minute`, SPX, `expir_date = trade_date`, `expiry_tod = 'pm'`, 06:30 ≤ `snapshot_pt` < 13:00. This is the `cr_cj_lib.load_skew` query restricted to one date. The snapshot pairs with ES at its `quote_date` minute (CR-CJ A1, `skew_minutes`, quotes > 5 min old dropped). The CR-CJ `guard` (train periods only) is not used. Its replacement is the X5 date guard. |
| X3 | ES input | **Open — see Step 0 Q1.** CR-CL reads Databento minute bars (`es_flow_1s_databento`, B5). |
| X4 | Outcome | As CR-CL C9 / C10 and A2.4: P0 = close of bar confirm_i. ±10 race (primary) and ±20 race on bars confirm_i + 1 … 12:59 PT; result ∈ {win, loss, tie, unresolved, gap}. Signed move h × (close(confirm_i + k) − P0) at k = 30, 60 (NULL past 12:59). `cr_cl_lib.race` / `move_at`, unchanged. |
| X5 | Date guard | `session_date < 2026-10-27` → refuse (exit 2, nothing read or written). `session_date` not an NYSE trading day, or an early close → refuse. Run before the session closes (now < 13:15 PT on `session_date`) → refuse. The guard runs before any DB connection. |
| X6 | Completeness gate | Events and outcomes are written once and never changed, so the script writes nothing for a session whose inputs are incomplete. It requires all of: the IM row; **390 / 390** RTH ES bars (amended at Step 0; was ≥ 380); ≥ 80 % of RTH 0DTE PM snapshots. Otherwise it exits non-zero with the reason, so the next run can retry. A session that is still incomplete after the retry window is recorded as a **skip row** (X8) with the reason, so a gap is never silent. |
| X7 | Outcomes timing | The race ends at the 12:59 PT bar, so every event resolves (win / loss / tie / unresolved / gap) on its own session. A run writes outcomes for its own session's events, plus any earlier event without an outcome (catch-up after a failed run). |
| X8 | Tables | `forward_log_events`: id bigserial; signal; definition_version (`'G1-down v1'`); session_date; leg_start_ts_pt / leg_end_ts_pt / confirm_ts_pt (naive PT, bar starts); entry_price (P0); start_px, end_px, swing_pts, im_pts; ratio, actual_pp, expected_pp, band_pp, beyond_noise; es_source; code_sha; run_id; written_at. UNIQUE (signal, definition_version, session_date, leg_end_ts_pt). `forward_log_outcomes`: id; event_id UNIQUE → events; race_10, race_20; move_30, move_60; code_sha; run_id; written_at. `forward_log_sessions` (one row per session processed: event count or skip reason) so that zero-event days are distinguishable from missed days. All three are insert-only by trigger (UPDATE / DELETE raise), DDL applied once by the owner (`dash_backfill_writer` has no CREATE), with SELECT, INSERT granted to `dash_backfill_writer`. |
| X9 | Idempotency | `INSERT … ON CONFLICT DO NOTHING` on the unique keys; a re-run for a logged session inserts 0 rows and reports it. |
| X10 | Printing (K6) | stdout and `bt_backfill_runs.smoke_test_results` carry counts only: events found / inserted, outcomes inserted, sessions skipped. No win, loss, move or rate is printed, logged or aggregated anywhere in the code. |
| X11 | Code SHA | `RENDER_GIT_COMMIT` on Render, else `git rev-parse HEAD`. The run refuses if neither is available. |
| X12 | Tests | (a) guard: each date < 2026-10-27 and today-before-close refused, with no DB call; (b) parity: on 3 `y26` train sessions, the script's event function, fed the same data, returns exactly the G1-down legs that `cr_cl_lib.vol_scaled_legs` gives under `CR_CL_PERIOD=y26` (in memory, nothing inserted); (c) idempotency: against a throwaway schema copy of the DDL, two runs on one fixture session insert n then 0. |

## Steps

- **Step 0** (diagnosis): live feeds for every input; the B5 gap; the DST effect on K9's cron; whether the log can support a confirmation. **Stop for Ryan.**
- **Step 1**: DDL (`scripts/cr_cx_ddl.sql`).
- **Step 2**: `scripts/forward_log_g1down.py` and its lib.
- **Step 3**: tests.
- **Wrap**: session note, PR, deploy plan in chat.

## Standing rulings conformance (at spec freeze)

| Ruling | Status | Note |
|---|---|---|
| A1–A5, A9, A10, A12, A13, A18–A21 | n/a | No levels, ranges, entries, stops or sizing; a signal log. |
| A6 | honoured | G1-down = "price down with put skew falling = bullish". |
| A7 | honoured | Vol-scaled swing (CR-CO A1), as changed 2026-10-26. |
| A8 | n/a | No level population (CR-CL's GEX tag is a readout, not logged). |
| A11 | honoured | ±10 race primary, ±20 logged; +30 / +60 moves logged. MFE is not in K3: see Step 0 Q4. |
| A14 | honoured | CR-CK definitions and the shipped band, unchanged. |
| A15 | honoured | Stop-free outcome. |
| A16 | honoured | Trend Δ over the leg. |
| **A17** | **deviates — approval requested on the CR-CL terms** | The log records skew alone as the signal before the flow-first test. Same deviation as CR-CL / CR-CN / CR-CO, which Ryan approved per CR as a **measurement** (nothing shortlisted). The kickoff does not pre-approve it for CR-CX. |
| B1 | honoured | No verdict in this CR. The log feeds a later pre-registered confirmation. |
| B2, B3 | n/a | Nothing tuned. |
| B4 | honoured | Only sessions ≥ 2026-10-27 are written. The holdout (2026-06-06 → 10-26) is never read: the script refuses earlier dates, and the tests read `y26` train only. No outcome is printed (K6, X10). |
| **B5** | **deviates (Step 0 Q1)** | "Databento is the reference for flow and minute bars in tests." The frozen code reads Databento bars; **no Databento ES feed runs after 2026-10-23**, so a daily log on the frozen inputs needs either a daily Databento pull or Ironbeam bars. |
| B6 | honoured | Stop-free direction signal, labelled not a test of Ryan's trade. |
| B7 | n/a here | The matched baseline lives in the confirmation, not in this log. But see Step 0 Q2: the baseline legs must be logged too, or the confirmation cannot use the B7 match. |
| B8 | n/a | No labels. |
| C1, C2 | n/a | No display. |
| R1–R7 | not used | |

Two "deviates" (A17, B5): **stop for Ryan** before any implementation code.

---

## Step 0 findings (2026-10-27, `scripts/cr_cx_step0.py` @ `7b0dd45` → `scripts/logs/cr_cx/step0.json`; legs only, no outcome computed)

Reads: the latest timestamp in each input table (no values from any session ≥ 2026-06-06), and `y26` train (107 sessions) through the CR-CJ loader and its guard. One launch, exit 0.

### 1. Live inputs (feed heads at 2026-10-27 ~13:00 UTC, before the open)

| Input | Latest row | Live daily? |
|---|---|---|
| `es_flow_1s_databento` / `es_flow_1m_databento` | 2026-10-23 19:59 UTC | **No.** A one-off CR-CA load (`scripts/cr_ca_databento_flow.py`, run by hand); no cron. |
| `ironbeam_es_1m_bars`, `ironbeam_es_flow_1s` | 2026-10-27 12:58 UTC | Yes (Ironbeam cron / trades-ws). |
| `orats_monies_minute` SPX 0DTE PM | 2026-10-26 13:00 PT, 391 rows, `updated_at` 13:30 → 20:00 UTC | Yes, minute by minute through the session. Complete by the close. |
| `bt_daily_features.implied_move_1d` | 2026-10-27 | Yes (13:35 UTC implied-move cron). |

**So CR-CL's frozen code cannot run on its frozen ES input after 2026-10-23.** CR-CL C1 reads Databento minute bars built from `es_flow_1s_databento` (B5).

### 2. Ironbeam vs Databento on the frozen G1-down code (`y26` train, vol-scaled swing, primary legs)

- **Train baseline (Databento): 28 G1-down v1 events in 107 sessions** (0.26 per session; 801 primary legs, 1,033 legs). IM_train re-derived: 65.714 (matches CR-CO's 65.71). 5 sessions have no IM.
- **On sessions both feeds cover fully, the events match:**

| Ironbeam source | Complete sessions (of 107) | G1-down on common sessions: Databento / Ironbeam / identical | All legs identical |
|---|---|---|---|
| `ironbeam_es_flow_1s` → minutes | 87 | 23 / 23 / 22 | 813 of 831 |
| `ironbeam_es_1m_bars` | 92 | 24 / 24 / 23 | 907 of 913 |

  The one mismatch (2026-03-16) is the same leg, same end and confirmation bar, starting one minute later on Ironbeam (a tied low). It is a G1-down event on both feeds.
- **The cost is whole sessions.** `cr_cl_lib.build_legs` skips any session with a single missing RTH bar (`np.isfinite(H).all()`). Ironbeam bars miss ≥ 1 RTH minute on 15 of 107 train sessions (flow: 20), against 0 for Databento. That loses 4 of 28 G1-down events (bars) or 5 (flow). This also means **X6's "≥ 380 of 390 bars" gate is wrong**: the frozen code needs **390 / 390**, or it silently returns no legs. X6 is amended to 390 / 390.

### 3. K9's cron time and DST

20:30 UTC is 13:30 PDT today, but **DST ends 2026-11-01**. From 2026-11-02, 20:30 UTC is **12:30 PST, half an hour before the close**. The race runs to the 12:59 PT bar, so X5's before-close guard would refuse every run from November to March. A fixed UTC time must be ≥ 21:15 UTC (13:15 PST / 14:15 PDT), or the run moves to the next morning.

### 4. Can the log support a confirmation? (B4 "used only as a pre-registered confirmation")

- **Baseline.** CR-CL / CR-CO's estimator is G1-down win rate **minus a matched in-noise baseline** (direction × size tercile × confirmation bucket × EDGE, B7). K2 logs only G1-down events, so the baseline would have to be recomputed later from raw tables. Those are not write-once (Ironbeam bars get repaired, e.g. CR-BZ), and the recomputation would happen after the outcomes are known. Logging the baseline legs (in-noise primary down legs) with their strata keys and outcomes, in the same write-once tables, would fix the comparison at the time it happens.
- **Power.** At train's 0.26 events per session, 40 resolved events take ~150–160 sessions, so about **mid-2027**. CR-CO's 2023 rate (50 in 200, 0.25) agrees.
- **MFE.** CR-CL / CR-CO reported MFE before −10 (A11's favourable-excursion distribution); K3 does not list it. It is one column.

### Questions for Ryan (stop here; no implementation code written)

| # | Decision | Options (Code's recommendation first) |
|---|---|---|
| Q1 | **ES input (B5 deviates)** | **(a) Daily Databento pull** of the session's front-contract trades into `es_flow_1s_databento` with the CR-CA code, then the frozen code unchanged. Honours B5 and K4 exactly. Needs a Databento key on the cron service. Cost is small: CR-CA's cap was $330 for 643 sessions, so under ~$0.5 per session; the actual daily price is not yet measured. Same-day availability of Databento historical data is not yet measured either, so plan the run for the next morning. **(b) Ironbeam `ironbeam_es_1m_bars`** with the frozen code. Events match on complete sessions (23 / 24), but ~14 % of sessions drop and the log's input differs from CR-CL's (a B5 deviation to approve). |
| Q2 | Log the matched baseline? | **Yes:** log every primary vol-scaled leg that is ratio-defined (`grp` ∈ G1 / G2 / base / neither) with strata keys (direction, size, confirmation bucket, EDGE), and outcomes for G1-down and in-noise down legs. **No:** G1-down only, as K2. |
| Q3 | Cron time | With Q1(a): **11:00 UTC next morning** for the previous session (DST-proof, after Databento's daily file). With Q1(b): **21:30 UTC** weekdays (14:30 PDT / 13:30 PST), with the X5 guard. |
| Q4 | MFE column | **Add** `mfe_before_m10` to `forward_log_outcomes` (A11), or leave as K3. |
| Q5 | **A17 (deviates)** | Approve on the CR-CL terms (measurement only, nothing shortlisted), as for CR-CL / CR-CN / CR-CO. |

Amendment to X6, effective now: the completeness gate is **390 / 390 RTH ES bars** (finding 2).
