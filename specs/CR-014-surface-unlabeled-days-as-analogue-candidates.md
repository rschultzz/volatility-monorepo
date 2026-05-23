# CR-014 — Surface Unlabeled Days as Analogue Candidates

Small follow-up to CR-013 that removes a design over-restriction: the analogues endpoint was gated on per-signal human labels in `bt_signals`, but v0.5's actual question ("what days look like today?") is structural — answered by the auto-classified feature vector + price outcomes, with no human-label requirement. Removing the label-gate unblocks immediate analogue surfacing as `bt_daily_features` accumulates via backfill.

### Problem

CR-013's `/api/analogues` endpoint filters candidate days to those with at least one labeled signal in `bt_signals`. The CR-013 spec rationalized this as "keeping the comparison grounded in days where Ryan has a personal read on what happened" (AC #6 stated "Unlabeled days are excluded from results").

In practice this conflates two separate use cases:

- **v0.5 — structural day comparison** ("find days whose landscape feels like today's"). Answered by the feature vector's auto-classified fields (regime, quality, cluster topology, vol regime) and the price outcomes derived from `ironbeam_es_1m_bars`. No human label needed.
- **v1b — strategy backtest** ("which template would have worked on similar days"). Here the per-signal label *is* the validation signal — was this scan-detected setup actually a good trade? This is the use case where the label-gate makes sense.

The label-gate belongs in v1b, not v0.5. As shipped, CR-013's endpoint returns `analogues: []` until a human labels signals — even though the `bt_daily_features` table is populated with 6 days of fully auto-categorized feature vectors and the price outcomes for those days are derivable from cached bar data. The empty-state UX is honest about what's needed, but the underlying gate is the wrong design call.

This blocks a practical use case: backfill `orats_gex_landscape` + `bt_daily_features` for a month of historical data → expect to immediately see analogues for today against ~20 candidate days. Under CR-013's behavior, those 20 backfilled days remain hidden until each gets a labeled signal, which defeats the value of backfilling for comparison purposes.

### Goal

Remove the label-gate from the analogues endpoint. Surface every day in `bt_daily_features` as a candidate (excluding the anchor day). Labels, where present, become additional context on each analogue's modal rather than a gating filter on the list.

### Changes

**`apps/web/modules/Analogues/routes.py`**: candidate-selection query no longer joins-and-filters on `bt_signals`. The query becomes a straight read from `bt_daily_features` filtered by `feature_version`, with the anchor day excluded.

```python
# Before (sketch):
SELECT df.*
FROM bt_daily_features df
INNER JOIN bt_signals bs ON bs.trade_date = df.trade_date
WHERE bs.label IS NOT NULL
  AND df.feature_version = :version
  AND df.trade_date != :anchor_date

# After (sketch):
SELECT df.*
FROM bt_daily_features df
WHERE df.feature_version = :version
  AND df.trade_date != :anchor_date
```

The labels-per-analogue lookup (a separate query, by `trade_date`, returning whatever labels exist) stays — but is now a LEFT-JOIN-style enrichment, not a filter. An analogue day with no labels returns `labeled_signals: []` in its response object; the modal renders a "no labels for this day yet" affordance rather than the day being excluded.

> Implementation note: CR-013's design assigned candidate selection to `service.py`, but the actual code organization puts all DB I/O in `routes.py` (`_load_labeled_candidates`). The change is still localized to a single function; this CR keeps that organization and renames the function to `_load_candidates`.

**`apps/web/modules/Analogues/routes.py` (response shape)**: unchanged. The `labeled_signals` field per analogue now defaults to an empty list when no labels exist (rather than the day being excluded entirely). No new request params. The module docstring updates to reflect that unlabeled days are now included; the `n_candidates` semantics shift from "labeled candidates" to "feature-vector candidates."

**`apps/web/modules/Analogues/tests/test_service.py`**: existing pure-function tests are unaffected by the routes-layer change. Add new unit tests that cover the candidate-loading function directly (using an injectable cursor or stub connection) so that the "unlabeled days surface" behavior is unit-tested. Expected delta: +2 tests, no net loss.

**Frontend — `AnaloguesPanel.jsx`**: empty-state copy updates. The old "no labeled days yet" message becomes contextually accurate to the new behavior:

- When `bt_daily_features` candidate pool is empty (e.g., before backfill runs): "no candidate days yet — run the daily-features backfill to populate"
- When candidate pool exists but no analogues come back within reasonable similarity (extreme outlier today, or all candidates filtered out): "no close analogues — today's landscape is unusual against the current corpus"
- When analogues surface but none have labels yet: list renders normally; modal entries just show "no labels for this day yet" inline

The footer label "N labeled candidate(s)" becomes "N candidate(s)" to match the new semantics.

**Frontend — `AnalogueDetailModal.jsx`**: handle `labeled_signals: []` gracefully — render the day's outcome metrics and landscape summary as normal, with a small "no labels for this day yet" affordance in the label area instead of the existing "No labeled signals." string.

### Acceptance criteria

1. **Backfilled days surface as analogues regardless of label state.** `GET /api/analogues?date=2026-05-22&...&k=5` against production (with 6 backfilled days, 0 labeled) returns up to 5 analogues sorted closest-first by similarity, rather than `[]`.
2. **The anchor day is still excluded** from its own analogue list.
3. **When a labeled signal exists for an analogue day**, the `label / label_note / outcome` fields populate as in CR-013.
4. **When no labeled signal exists for an analogue day**, the analogue still appears in the list with `labeled_signals: []`; the modal renders normally with a "no labels for this day yet" affordance in the label area.
5. **Outcome metrics still populate** (EOD return, intraday range, MFE per direction) for every analogue, regardless of label state. These come from `ironbeam_es_1m_bars`, not from labels.
6. **Backend tests pass.** ~205 → ~207 (existing 205 + 2 new for the unlabeled-candidate case).
7. **`vite build` passes.**
8. **Production smoke** confirms the four CR-011 calibration days as anchors return analogue lists matching the labels-bypassed ranking-math preview from CR-013's wrap-up (5/6 ↔ 5/7 as nearest neighbors, pin ↔ amplification at max distance).

### Verification plan

1. **Unit tests updated** per the changes above.
2. **Production-DB smoke** against the four CR-011 calibration days. Verify the analogue lists now look like CR-013's labels-bypassed preview (which already validated the math). Output should look like:
   ```
   anchor 5/6  → 5/21, 5/18, 5/7, 5/22, 5/20
   anchor 5/7  → 5/6, 5/21, 5/18, 5/22, 5/20
   anchor 5/18 → 5/21, 5/22, 5/6, 5/20, 5/7
   anchor 5/20 → 5/21, 5/18, 5/22, 5/6, 5/7
   ```
3. **Frontend smoke** — toggle ANALOGUES pill on with no labels in `bt_signals`. List shows analogues (not empty state). Click a row → modal opens with outcomes/landscape; label area shows "no labels for this day yet."
4. **No regression on existing functionality** — when a labeled signal *is* present for an analogue day (after Ryan starts labeling), the label info appears in the modal as before.

### Out of scope

- **Backfilling the month** is an ops action, not part of this CR. Once CR-014 ships, run `scripts/backfill_gex_landscape.py --since <date>` then `scripts/backfill_daily_features.py --since <date>` to populate `bt_daily_features` for a month of history.
- **Distance threshold for "no close analogues."** The list always returns up to K analogues regardless of how distant they are. A future CR can add a distance cutoff once we know what "too far" looks like empirically.
- **Per-anchor "explainability"** — why these days vs others. Future CR.
- **Re-ranking based on labels (e.g., upweighting labeled days that share an outcome)** — that's a v1b consideration; out of scope here.

### Data integrity dependencies

- `bt_daily_features` populated for the date range to be served. CR-013's cron + backfill scripts handle this.
- `ironbeam_es_1m_bars` populated for those dates (needed for outcome metrics). Standard data; CR-013 already depended on it.
- `bt_signals` is now an *optional* enrichment, not a hard dependency. If it's empty, the endpoint still works; analogues surface without label info.
