# CR-015 — Day Setup Recommendations (Backtesting v1a)

**Status**: in-progress  
**Branch**: `feat/CR-015-day-setup-recommendations`  
**Base**: `Main-Live` @ `3c603c1`

---

## Problem

The four-stage framework's signal stage shipped in v0.5 (`bt_daily_features`, regime
classification, quality tags, cluster topology). It surfaces analogous days for context
but says nothing about what to *do* with today's setup.

The next stage — trade construction — needs to turn structural signal into concrete trade
proposals: given a pin cluster at 7400 with FWHM 24pt, what's the butterfly's body strike
and wing distances? Given a target cluster at 7480 above spot 7440, what's the call
spread's short/long strikes?

This is decision-support, not validation. There's no backtest in v1a — Ryan looks at the
proposals and decides. Validation comes in v1b (point-based simulation) once the proposal
interface is stable.

---

## Goal

Build a `/today-setup` page (new React app, `react_today_setup/`, served from a new
`apps/web/modules/TodaySetup/` backend module) that displays the recommendation card for
a given trade date. The card surfaces all applicable trade proposals from a fixed set of 6
templates, with no ranking — trader scans and picks.

The page is date-driven so historical dates work too. The page is laid out as a grid of
cards; v1a populates one card (recommendation); future CRs add more cards without page
refactors.

---

## Architecture

### 1. Strategy templates (`packages/shared/strategy_templates.py`)

Six templates:

| `template_id` | `template_kind` | Applies when | Wing/width source |
|---|---|---|---|
| `pin_butterfly_tight` | `butterfly` | regime in (pinned, magnetic-pin), cluster quality == "pin" | half-FWHM |
| `pin_butterfly_medium` | `butterfly` | same | full FWHM |
| `pin_butterfly_wide` | `butterfly` | same | 1σ × implied_move |
| `directional_spread_to_target` | `spread` | regime in (magnet-above, magnet-below) + drift_target in regime_block | fixed 10pt width |
| `bounded_iron_condor` | `condor` | regime == bounded + containment_zone in regime_block | fixed 10pt wing width |
| `feature_no_trade` | `no_trade` | regime in (feature, untethered, amplification, broken-magnet) | n/a |

**Template Protocol**:

```python
class Template(Protocol):
    template_id: str
    template_kind: str

    def applies_to(
        self,
        regime_block: dict,       # full regime dict (not just the string)
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
    ) -> bool: ...

    def propose(
        self,
        regime_block: dict,
        cluster: dict | None,
        clusters_all: list[dict],
        spot: float,
        implied_move: float,
        anchor_strategy: str = "cluster_centered",
    ) -> TradeProposal: ...
```

`generate_proposals(landscape_payload, spot, implied_move)` iterates over every
(template, cluster) pair, calls `applies_to`, and collects proposals from those that
match. Returns `list[TradeProposal]`.

### 2. Strike anchor strategies (`packages/shared/strike_anchors.py`)

```python
class AnchorStrategy(Protocol):
    name: str

    def butterfly_strikes(
        self,
        cluster: dict,
        wing_distance_recipe: str,  # "half_fwhm" / "full_fwhm" / "sigma_1x"
        implied_move: float,
    ) -> tuple[float, float, float]: ...  # (lower_wing, body, upper_wing)

    def spread_strikes(
        self,
        short_strike: float,
        spot: float,
        width_pts: float,
    ) -> tuple[float, float, str]: ...  # (short, long, direction)

    def condor_strikes(
        self,
        lower_price: float,
        upper_price: float,
        wing_width_pts: float,
    ) -> tuple[float, float, float, float]: ...  # (long_put, short_put, short_call, long_call)
```

`ClusterCenteredAnchor`:
- `butterfly_strikes`: body = `cluster["center_price"]`; wing recipe as in spec
- `spread_strikes`: short = `short_strike`; long = short ± width; direction = "call" if short > spot else "put"
- `condor_strikes`: short_put = `lower_price`, short_call = `upper_price`, longs = shorts ± wing_width

### 3. Backend module (`apps/web/modules/TodaySetup/`)

Endpoint: `GET /api/setup/proposals?date=YYYY-MM-DD&spot=<float>&implied_move=<float>&ticker=SPX`

### 4. Frontend (`react_today_setup/`)

New Vite React app, served at `/today-setup`. Same scaffolding as `react_backtests_v2/`.

---

## Step-0 Validation Notes (2026-05-22)

### Regime distribution (37 days in corpus)

| Regime | Count | Sample dates |
|---|---|---|
| magnet-above | ~18 | 2026-05-22, 2026-05-21, 2026-05-19 |
| bounded | ~7 | 2026-05-15, 2026-05-08, 2026-05-01 |
| magnetic-pin | 3 | 2026-05-07, 2026-04-28, 2026-04-17 |
| untethered | ~4 | 2026-05-18, 2026-04-27, 2026-04-21 |
| amplification | 2 | 2026-05-20, 2026-04-01 |

### Smoke-test dates

- **Pin**: 2026-05-07 — magnetic-pin, spot=7362.8, 2 pin clusters (7353/109pt FWHM, 7372/165pt FWHM)
- **Magnet-above**: 2026-05-22 — magnet-above, spot=7444.3, drift_target=7524.3
- **Bounded**: 2026-05-15 — bounded, spot=7505.5, containment lower=7502.5 / upper=7543.5
- **Untethered**: 2026-05-18 — untethered, spot=7421.3

### avg_fwhm field

Confirmed present on `confluences` clusters: pin day 2026-05-07 shows `avg_fwhm=109.0` and
`avg_fwhm=165.5`. Field name matches spec exactly.

### Spec amendments (two)

**Amendment A — directional_spread_to_target**:
- Original applies_to: `regime in (magnet-above, magnet-below) AND cluster quality == "target"`
- **Amended**: `regime in (magnet-above, magnet-below) AND "drift_target" in regime_block`
- Template uses `regime_block["drift_target"]` as the short strike, not a cluster from confluences.
- Why: Magnet-above days show dominant wall GEX of ~492B — below the 550B "target" threshold.
  The "target" in this template name refers to the regime's price target (drift_target), not
  the GEX quality tier. No "target" quality clusters appear in current magnet-above corpus.
- Impact: `directional_spread_to_target` now fires for all real magnet days. Correct behavior.

**Amendment B — bounded_iron_condor**:
- Original applies_to: `regime == bounded AND ≥2 clusters straddling spot`
- **Amended**: `regime == bounded AND "containment_zone" in regime_block`
- Template uses `containment_zone["lower_price"]` and `containment_zone["upper_price"]` as
  short put/call strikes.
- Why: Bounded days may show only 1 confluence or confluences not straddling spot, but
  always have `containment_zone` when regime is bounded. The containment zone IS the bounded
  definition (competitive positive walls on both sides).
- Impact: Condor template reliably fires for all bounded days. Correct behavior.

**Protocol change**: `applies_to` and `propose` receive `regime_block: dict` (not `regime: str`)
so templates can access `drift_target` / `containment_zone` / `drift_direction`.

### React scaffolding pattern

`react_backtests_v2/vite.config.js` — simple static base:
```js
base: '/backtests-v2-preview/'
```
`react_price_preview/vite.config.js` — conditional base (dev vs build):
```js
base: command === 'serve' ? '/' : '/react-preview/'
```
Use `react_backtests_v2` pattern (static base `/today-setup/`). The price_preview
conditional base is not needed here.

---

## Implementation Order

1. **Spec** — this file ✓
2. **Anchor strategies** — `packages/shared/strike_anchors.py` + tests
3. **Templates** — `packages/shared/strategy_templates.py` + tests
4. **Backend module** — `apps/web/modules/TodaySetup/` + tests + Flask wiring
5. **Endpoint smoke** — hit 4 regime dates against production DB
6. **React scaffold** — `react_today_setup/` skeleton
7. **Card components** — proposal rendering
8. **Nav wiring** — link from main dashboard

---

## Acceptance Criteria

1. `/api/setup/proposals?date=<pin_day>` returns 3 butterfly variants per pin cluster.
2. `/api/setup/proposals?date=<magnet-above_day>` returns directional_spread_to_target with
   short at drift_target and long 10pt above (direction=="call").
3. `/api/setup/proposals?date=<bounded_day>` returns bounded_iron_condor using containment_zone.
4. `/api/setup/proposals?date=<untethered_day>` returns single feature_no_trade.
5. Anchor strategy field populated as "cluster_centered" on all non-no_trade proposals.
6. Expiry DTE bucket field populated per cluster bucket.
7. No cross-template leakage.
8. ~235+ backend tests passing.
9. `vite build` passes for react_today_setup.
10. `/today-setup` page loads with date picker, context strip, proposal cards.
11. Nav link from main dashboard.

---

## Out of Scope (v1a)

- Vol surface in proposals
- Anchor strategies beyond cluster_centered
- Concrete expiry resolution (DTE target only)
- Sizing recommendations (quantity: 1 throughout)
- P&L simulation
- Persisting proposals to DB
- Risk metrics card
- Journaling card
