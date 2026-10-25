# Setup registry

One row per setup tested under the CR-CB A10 pipeline: (1) screen on 2026 train → (2) confirm on R-a → (3) optimize → (4) final confirm on R-b → (5) holdout and paper. Losers are recorded too (A10.5), so nothing is re-tested by accident.

Verdicts follow the three-verdict rule:
- **shortlisted / pass**: beats placebo and the baselines with CI lower bound > 0;
- **refuted**: CI upper bound < 0 vs placebo;
- **inconclusive**: otherwise.

Data splits (CR-CB A10.2): train 2026 ≤ 2026-06-05; R-a 2024-04-03 → 2024-12-31; R-b 2025; H after 2026-06-05.

| setup | CR | stage reached | data | verdict | key numbers | date | spec |
|---|---|---|---|---|---|---|---|
| First-touch fade (CR-BW events) | CR-CE | 1 — screen | train, 86 sessions 2026-01-09 → 06-05, Databento flow; 803 events (every re-armed touch, D > 0) | **inconclusive** in all 4 cells and pooled; not shortlisted, Stage 2 not run | +60 min stop-free move, pooled: real −0.03 [−1.46, +1.32]; vs placebo +1.39 [−0.62, +3.43]; vs random +3.15 [+2.14, +4.17]. Per cell vs placebo: put-below +3.89 [−1.82, +9.25], put-above −0.11 [−5.15, +5.45], call-below +1.17 [−3.13, +5.79], call-above −1.48 [−6.15, +3.30]. The vs-random gains reflect a negative random baseline (approach-direction drift), not a level effect. Absorption terciles don't sort the move (DiD −0.60 [−5.37, +4.60]). MAE before +10: median 5.0 pts real vs 5.5 placebo | 2026-10-25 | `specs/CR-CE-first-touch-fade-direction.md` |
