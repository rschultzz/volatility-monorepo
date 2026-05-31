# CR-028 — CR-V: Today's Edge Across Horizons (Touch & Close)

> Source: vault `CRs/CR-V — Today's Edge Across Horizons.md` (2026-05-30 draft)
> Sequence number activated: 028 (next = 29)

## Goal

For a given setup (today's thesis on today's GEX/magnet read), produce and display at **t1 / t5 / t15**, two edge numbers per horizon:

1. **Touch edge** = structural touch probability (analogues touched within N sessions) − market-implied touch probability (≈ min(1.0, 2 × |delta|) at the magnet strike at the tN-DTE expiration).
2. **Close edge** = structural **unconditional** close-past-strike probability (ALL analogues, from `session_close_tN`) − market-implied close probability (|delta| at the magnet strike at the tN-DTE expiration).

Both edges shown with structural CI / analogue count. "Horizon" is a **parameter** — the three horizons {t1, t5, t15} are a v1 default, not a constant. Adding horizons later is data + config only (no logic rewrite).

---

## Step 0 — Diagnosis and design lock (no commits)

### Lock 1: Structure definition + strike placement

- Short strike placed at the magnet (nearest listed SPX 5-pt strike via `compute_spx_strike`).
- Long leg = short strike + 10 SPX points (fixed wing; PARAMETER — `spread_width = 10`, `anchor = magnet`).
- Both structural and market sides measured to the **placed short strike in ES-forward space** (`strike_es`) — not the theoretical magnet float.
- `spread_width` and `anchor` are parameters so a later strike-axis pass extends rather than rewrites.

### Lock 2: Horizon set as config (Critical Constraint #4)

- v1 horizon set = `["t1", "t5", "t15"]` stored as a module-level constant `HORIZONS`.
- The edge core, endpoint schema, and UI all consume this constant — never hard-code "t1/t5/t15" in three separate places.
- Adding t10 later requires: (a) `session_close_t10` column + backfill, (b) add `"t10"` to `HORIZONS` (and `HORIZON_SESSION_DAYS` mapping), (c) find the ~10-DTE expiration for market-side delta. No edge logic, schema shape, or UI structure changes.
- Response schema: **LIST** keyed by horizon, never three fixed fields.

`HORIZON_SESSION_DAYS = {"t1": 1, "t5": 5, "t15": 15}` — maps horizon label to target session count.

### Lock 3: Unconditional close classification (Critical Constraint #1)

**Never** reuse `reached_close` (it is a ±0.25×IM proximity test at the dominant-bucket horizon, not a per-{t1/t5/t15} past-strike test). **Never** use the post-touch bars from `aggregate_post_touch_distribution` (touch-conditional + band-relative). Reuse `_project_close` from `structural_distribution.py`.

```
For each horizon tN:
  denominator = analogues where outcome_status == 'computed' AND session_close_tN is not None
  for each such analogue:
    projected = _project_close(session_close_tN, session_open_t0, implied_move_1d,
                               today_spot, today_implied_move)
    past_strike = (projected >= strike_es)   # magnet-above; flip for magnet-below
  numerator = count(past_strike == True)
  close_rate_tN = numerator / denominator
```

Direction by regime:
- `magnet-above`: projected >= `strike_es` (placed short strike)
- `magnet-below`: projected <= `strike_es`

**Verified (Step 0 code read 2026-05-31):** `_project_close` is pure math, does not check touch status. `session_open_t0` is populated for ALL `outcome_status IN ('computed', 'na_regime')` rows by the CR-G Step 2.5a backfill — not touch-gated. `session_close_tN` is populated by the CR-G Step 0-A backfill for all `computed`/`na_regime` rows (NULL only at corpus end or quarterly Fridays).

### Lock 4: Per-horizon touch rate (structural, path-based, horizon-specific)

Touch is per-horizon: "did price touch the magnet within N sessions?"

```
For each horizon tN:
  k_computed = analogues where outcome_status == 'computed'
  n_touch_within_tN = count where reached_touch AND days_to_reach <= HORIZON_SESSION_DAYS[tN]
  struct_touch_tN = n_touch_within_tN / k_computed
  touch_ci_tN = wilson_ci(n_touch_within_tN, k_computed)
```

`days_to_reach` is available in analogue rows from `_rank_analogues_with_outcomes`.

### Lock 5: Market side — per-horizon stored delta

For each structural horizon tN, the market-implied close probability uses the stored `delta` of the magnet-strike option at the **expiration closest to tN trading days** from `trade_date`.

**Delta is a required field on `OptionMinuteBar` — never null when a bar exists.** Confirmed: `delta: float` (non-Optional) in `models.py`; populated by the ORATS API response and stored in cache.

Horizon → expiration mapping:
- `t15`: use the proposal's own expiration (already fetched by `price_proposal_legs`; surface `bar.delta` for the short/magnet leg).
- `t5`, `t1`: find the nearest available SPX expiration (Mon/Wed/Fri or monthly) to `HORIZON_SESSION_DAYS[tN]` trading days from `trade_date`. Build the OPRA for the magnet-strike at that expiration, fetch from cache (write-through via `fetch_option_bars`).

Helper needed: `nearest_spx_expiration(trade_date, target_session_days)` → `date`. v1 uses a simple rule: step forward trading days, take the first Mon/Wed/Fri (or monthly) at or past the target. SPX weeklies expire Mon/Wed/Fri; monthlies expire 3rd Friday.

```python
market_close_tN = abs(delta)                       # stored ORATS greek
market_touch_tN = min(1.0, 2.0 * market_close_tN) # 2× reflection rule (approximation)
```

**Fallback chain (per horizon, per side):**
1. Stored delta → PRIMARY
2. `spread_cost / spread_width` where `spread_cost = real_pricing.net_debit` — cross-check / fallback for close-market
3. Neither available → both edges `None`, warning added

### Lock 6: Minimum analogue threshold

Per-horizon, per-side guard (not global):
- **Close edge**: `n_close_tN < 8` → `low_confidence = True`
- **Touch edge**: `k_computed < 8` → `low_confidence = True`
- `low_confidence` cell: never bold; warning displayed; value shown (dimmed) or "—" (UI decision in Step 3).
- 8 is a concrete floor; the Wilson CI already conveys the spread for small N — no need for two separate mechanisms.

### Lock 7: Horizon ≠ exact DTE (v1 limitation — state in UI)

The three structural horizons are t1/t5/t15 *sessions* post-trade. A proposal with 10 DTE is bracketed by t5/t15, not exact-t10. This is the explicit cost of building before a per-DTE session-close backfill exists.

UI labels: `"~1d"`, `"~5d"`, `"~15d"` — never the trade's exact DTE.
Flag `2 × |delta|` as approximation in the UI ("reflects-rule approx.").

### Lock 8: Regime scope (v1)

- Supported: `magnet-above`, `magnet-below`
- Not supported (return `None` gracefully): `magnetic-pin`, `bounded`, `amplification`, `untethered`, `broken-magnet`
- Condor / pin / two-sided band edge is later work. The `unconditional_close_past_strike` helper should accept a `direction` parameter so it can be generalized later without rewrite.

---

## Implementation order

### Step 1 — Edge core

**Commit:** `cr-v/step-1: today's-edge core — touch & close vs market, per horizon`

New: `packages/shared/edge_today.py`. Public entry point:

```python
def compute_todays_edge(
    *,
    regime: str,
    strike_es: float,
    today_spot: float,
    today_implied_move: float,
    analogues: list[dict],
    horizons: list[str],                        # e.g. ["t1","t5","t15"] — NEVER hard-coded
    magnet_delta_by_horizon: dict[str, Optional[float]],  # {tN: |delta|}
    spread_cost_by_horizon: dict[str, Optional[float]],   # {tN: net_debit} cross-check
    spread_width: float,
    min_analogues: int = 8,
) -> dict:
```

Internals:
- `_touch_rate_within_sessions(analogues, max_sessions)` — reuses pattern of `_aggregate_outcomes`; filters `outcome_status == 'computed'`; counts `reached_touch AND days_to_reach <= max_sessions`.
- `_unconditional_close_past_strike(analogues, horizon_key, strike_es, direction, today_spot, today_implied_move)` — the ONE new pure function. Uses `_project_close` (imported from `structural_distribution`); `wilson_ci` from `stats`.
- `_market_side(tN, delta, spread_cost, spread_width)` → `(mkt_close, mkt_touch)` or `(None, None)`.
- For each horizon: assemble the per-horizon cell dict.

Output shape:
```python
{
    "regime": str,
    "per_horizon": [
        {
            "horizon": "t1" | "t5" | "t15",
            "struct_touch": float | None,
            "struct_touch_ci": [float, float] | None,
            "n_touch": int,
            "mkt_touch": float | None,
            "touch_edge": float | None,         # struct_touch - mkt_touch
            "struct_close": float | None,
            "struct_close_ci": [float, float] | None,
            "n_close": int,
            "mkt_close": float | None,
            "close_edge": float | None,         # struct_close - mkt_close
            "low_confidence": bool,
        }
    ],
    "warnings": [str],
}
```

Hard gates (enforced by unit tests):
- No import of `implied_distribution` or any BSM pricer.
- Reuse `_project_close` (import from `structural_distribution`), `wilson_ci` (import from `stats`).
- Horizon set is a parameter — never hard-coded.

**Deliverables:** pure unit tests (synthetic analogues); smoke tests 1–10.

### Step 2 — Endpoint + `price_proposal_legs` delta extension

**Commit:** `cr-v/step-2: today's-edge endpoint (per-horizon touch & close)`

Two sub-changes:

**2a. Surface `bar.delta` in `price_proposal_legs`** (`packages/shared/options_cache/pricing.py`):
- Each priced leg already reads `bars[0]`; add `"delta": bar.delta` to the per-leg dict.
- No schema change — new key added to the existing leg dict.

**2b. New helper: `fetch_horizon_delta`** (`packages/shared/options_cache/pricing.py` or sibling):
```python
def fetch_horizon_delta(
    magnet_strike_es: float,
    trade_date: date,
    target_session_days: int,       # 1, 5, or 15
    entry_pt: datetime,             # naive PT
    r: float,
    q: float,
) -> Optional[float]:
    """
    Find the nearest SPX expiration to target_session_days trading days
    from trade_date, build the call OPRA at magnet_strike_spx, fetch from
    cache (writing through on miss), return abs(bar.delta) or None.
    """
```

- `nearest_spx_expiration(trade_date, target_session_days) -> date` — simple rule: step forward trading days (Mon–Fri), take first Mon/Wed/Fri ≥ target. If none within 20 days, return None.
- `compute_spx_strike(magnet_strike_es, dte, r, q)` → magnet strike in SPX space.
- `format_opra("SPX", expir, "C", spx_strike)` → OPRA.
- `fetch_option_bars([opra], entry_pt, entry_pt)` + `get_bars_for_contract` → `bar.delta`.
- Returns `abs(bar.delta)` if bar exists; None otherwise.

**2c. Route extension** (`apps/web/modules/Proposals/routes.py`):
- After `real_pricing = price_proposal_legs(...)`, extract the magnet (short) leg's delta:
  `magnet_delta_t15 = next((l.get("delta") for l in real_pricing["legs"] if l.get("side") == "short"), None)`
- Fetch t1/t5 horizon deltas: `magnet_delta_t1 = fetch_horizon_delta(..., target_session_days=1)`.
- Assemble `magnet_delta_by_horizon` and `spread_cost_by_horizon` dicts.
- Call `compute_todays_edge(...)`.
- Add `"todays_edge": result` to the response JSON (alongside `trade_thesis`, `greeks`, etc.).
- Only compute for `magnet-above` / `magnet-below`; for other regimes return `"todays_edge": None`.

### Step 3 — Frontend edge display

**Commit:** `cr-v/step-3: per-horizon touch & close edge on the proposal card`

- New component or section on the proposal card that iterates `todays_edge.per_horizon` (never three hard-coded columns).
- Each horizon row: touch edge + close edge side-by-side; structural CI / n visible; low-confidence badge.
- Make the touch–close gap legible so "high touch, low close → trade as touch" reads at a glance.
- Horizon labels: `"~1d"`, `"~5d"`, `"~15d"` — never exact DTE.
- 2× touch label: "touch (est.)" or small footnote flag.
- Honest degradation: missing delta AND missing spread cost → both edges show `—` + warning. Low-confidence → dimmed / badge, never bold.
- Reuse CR-T's data-quality-badge pattern.
- No edge-zone chart changes.

### Step 4 — Empirical sanity pass (no commit)

Run for 2–3 real setups. Key checks:
- Unconditional close rate is strictly lower than the post-touch "above" fraction (confirms conditioning fix).
- Touch edge ≥ close edge (by construction of 2× vs 1× rule — verify empirically).
- Structural numbers match what `compute_structural_probability` reports for the same day.
- "The 2026-05-21 anchor (avg_days_to_reach ≈ 9.6d)" is a good test case.

---

## Smoke tests

1. **Unconditional close includes non-touchers.** Synthetic set: some analogues never touched but closed past strike, some touched but closed back — close rate counts ALL.
2. **Conditioning fix.** For a set with touch < 100%, `unconditional_close_past_strike` is strictly ≤ the post-touch "above" fraction.
3. **Event matching.** `touch_edge` uses `2×|delta|` (touch-market); `close_edge` uses `|delta|`. A test that crosses them fails.
4. **Horizon parameterization.** Call with 2 horizons, then 4 — output list correct size; no logic branch on specific horizon names.
5. **Coordinate space.** Close classified in ES-forward (`strike_es`) vs analogue ES close. Mis-spaced comparison caught by hand-built case.
6. **Low-confidence guard.** Horizon with < 8 non-null closes → `low_confidence=True`; frontend marks it.
7. **Delta-null fallback + missing-market degradation.** Null delta → fall back to `cost÷width`; neither → both edges `None` + warning.
8. **No B-L / no BSM.** Static check: `edge_today.py` imports neither `implied_distribution` nor a BSM pricer.
9. **Reuse, not reimplementation.** `_project_close` imported from `structural_distribution`; `wilson_ci` from `stats`.
10. **2× cap.** `mkt_touch = min(1.0, 2 × |delta|)` — never exceeds 1.0.

---

## Wrap criteria

- Steps 1–3 committed; Step 4 sanity findings in status updates.
- All smoke tests pass.
- A real setup returns per-horizon (t1/t5/t15) touch edge AND close edge, structural side unconditional + real, market side from stored ORATS delta (cost÷width cross-check), CIs surfaced, thin cells flagged, horizons labeled approximate.
- Horizon set is demonstrably a parameter (smoke test 4).
- Roadmap: `active_cr → null`, `next_sequence_number → 29`, CR-V complete in queue.
- Decision note `2026-05-30 - Trade-First Edge via Analogue Backtest` Phase-1 status advanced.

---

## Out of scope

- Per-DTE horizons beyond t1/t5/t15 (new `session_close_tN` columns + outcomes backfill)
- Decision-rule backtest (does edge signal predict?)
- Realized P&L / held-to-expiration backtest
- Pin / bounded / condor (two-sided band) edge
- Strike-axis sweep / variable width / delta-anchored long leg
- Any B-L or edge-zone changes

---

## Step 0 — Diagnosis gate (2026-05-31 code read findings)

### `_project_close` on non-touchers (CONFIRMED clean)

`_project_close` in `structural_distribution.py` is pure math; it takes `(close, anchor_spot, anchor_implied_move, today_spot, today_implied_move)` and makes no reference to touch status. `compute_terminal_prob_in_range` already calls it on ALL analogues where the three inputs are non-null, regardless of `reached_touch`. `session_open_t0` is populated by the CR-G Step 2.5a backfill for ALL `outcome_status IN ('computed', 'na_regime')` rows — it is the analogue's own RTH open on its trade_date, not touch-gated. `session_close_tN` is populated by CR-G Step 0-A backfill for the same set (NULL only at corpus end or quarterly Fridays).

**Verdict**: `_project_close` generalizes cleanly to all computed analogues. No modification needed.

### Delta null-rate (CONFIRMED binary, never null on a hit)

`OptionMinuteBar.delta` is declared `delta: float` (non-Optional) in `packages/shared/options_cache/models.py`. The repository's `_BAR_COLUMNS` tuple includes `"delta"` and `_row_to_bar` maps it directly. If a bar exists in the cache, delta is guaranteed non-null. The fallback (cost÷width) fires only on a cache miss (zero bars returned) — not on null delta. For the t1 and t5 horizons, the cache miss rate depends on ORATS historical coverage for those expirations; CR-T Step 0-B showed good coverage for real proposals. If the t1/t5 expiration OPRA is cold, `fetch_option_bars` will attempt a live fetch and write to cache on success.

**Verdict**: delta null-rate is 0% on any cache hit; fallback is cache-miss, not null-field.

---

## Related files

Read: `packages/shared/probability.py`, `packages/shared/structural_distribution.py`, `packages/shared/stats.py`, `packages/shared/options_cache/pricing.py`, `packages/shared/options_cache/models.py`, `packages/shared/options_cache/repository.py`, `packages/shared/forward_math.py`, `apps/web/modules/Proposals/routes.py`, `apps/web/modules/Proposals/service.py`

Write (new): `packages/shared/edge_today.py`, `packages/shared/tests/test_edge_today.py`

Write (extend): `packages/shared/options_cache/pricing.py` (surface `bar.delta` in `price_proposal_legs`; add `fetch_horizon_delta`), `apps/web/modules/Proposals/routes.py` (assemble and call edge, add to response), frontend proposal card component
