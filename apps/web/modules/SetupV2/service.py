"""SetupV2 service — pure helpers for GET /api/setup-v2/card (CR-AW).

No DB I/O in this module. All DB access is in routes.py; everything here is
synthetic-testable.

Functions:
    select_reference_rows   — pick the band rows from bt_edge_backtest_results rows
    pick_reference_cr_id    — latest cr_id matching REF-%, else CR-AR
    band_for_sigma          — harness distance band (near / mid / far)
    percentile_rank         — percentile of a value against a corpus (kind='mean')
    quantile                — linear-interpolated quantile of a list
    knn_factor_rows         — one row per KNN feature: today, weight, percentile,
                              analogue middle-half band, in_band
    match_quality           — count in_band / populated
    expected_pnl            — fair − quote − fees, with the bootstrap interval
    verdict                 — enter / skip / no listed structure / no quote
    horizon_mix             — analogue outcome-horizon counts
    next_reference_run      — 1st of the month after a re-run date
    fee_points              — round-trip fees for a vertical, in SPX points
    net_debit_by_minute     — spread price per minute from per-leg bars (cache only)

Terminology (CR-AW): analogues are every corpus day inside the KNN similarity
ceiling (recency-weighted), never a fixed K. Nothing here emits "K=".
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from packages.shared.backtest.models import distance_band
from packages.shared.backtest.quote_validity import leg_quote_is_valid
from packages.shared.day_features import FEATURE_NAMES

# CR-AW A6: the shared fee constant is created by CR-AU (packages/shared/config.py).
# Import it when present; fall back to the confirmed Schwab value so the two CRs
# do not overlap on a file. $ per contract per leg (one side).
try:  # pragma: no cover - exercised once CR-AU lands
    from packages.shared.config import FEE_PER_CONTRACT_PER_LEG  # type: ignore
except ImportError:  # pragma: no cover
    FEE_PER_CONTRACT_PER_LEG = 0.65

# The card's quote minute (A4): first clean minute after the open used by the
# CR-AS / CR-AU capture conventions (06:33 spot, 06:30–06:45 window).
ENTRY_MINUTE_PT = dt.time(6, 34)

REFERENCE_FALLBACK_CR_ID = "CR-AR"
REFERENCE_PREFIX = "REF-"

# Card structure filters (decision 4).
_STRUCTURE_TYPE = "debit"
_OUTCOME_TYPE = "close"

_BANDS = ("near", "mid", "far")


# ── Fees ─────────────────────────────────────────────────────────────────────


def fee_points(n_legs: int, fee_per_contract_per_leg: float = FEE_PER_CONTRACT_PER_LEG) -> float:
    """Round-trip fees for an n-leg vertical in SPX points (1 pt = $100).

    n_legs × 2 sides × fee / 100. A 2-leg spread at $0.65 → 0.026 pt.
    """
    return round(n_legs * 2 * fee_per_contract_per_leg / 100.0, 6)


# ── Reference cells (decision 4) ─────────────────────────────────────────────


def pick_reference_cr_id(cr_ids: list[tuple[str, dt.datetime]]) -> Optional[str]:
    """Latest `cr_id` matching REF-% (by created_at), else CR-AR, else None.

    `cr_ids` is a list of (cr_id, latest_created_at) pairs.
    """
    refs = [(c, t) for (c, t) in cr_ids if c and c.startswith(REFERENCE_PREFIX)]
    if refs:
        return max(refs, key=lambda x: x[1])[0]
    if any(c == REFERENCE_FALLBACK_CR_ID for (c, _) in cr_ids):
        return REFERENCE_FALLBACK_CR_ID
    return None


def select_reference_rows(rows: list[dict]) -> dict[str, dict]:
    """Pick one cell per band (near / mid / far / all) from result rows.

    Rows are dicts with at least: cr_id, structure_type, outcome_type,
    post_touch_pattern, distance_band, partition, threshold, mean_pnl,
    win_rate, wilson_lo, wilson_hi, n_settled, baseline_mean, beat_baseline,
    created_at, run_id.

    Filters: structure_type='debit', outcome_type='close',
    post_touch_pattern IS NULL (the pooled cells, not the two-axis tags).
    When several partitions / thresholds remain for a band, 'train' wins over
    anything else and the lowest threshold wins — deterministic, and the
    lowest threshold is the closest thing to "enter at the open" the
    persisted cells carry (the un-gated baseline is in `baseline_mean`).
    """
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("structure_type") != _STRUCTURE_TYPE:
            continue
        if r.get("outcome_type") != _OUTCOME_TYPE:
            continue
        if r.get("post_touch_pattern") is not None:
            continue
        band = r.get("distance_band")
        if band not in (*_BANDS, "all"):
            continue
        cur = out.get(band)
        if cur is None or _cell_rank(r) < _cell_rank(cur):
            out[band] = r
    return out


def _cell_rank(r: dict) -> tuple:
    part = 0 if r.get("partition") == "train" else 1
    thr = r.get("threshold")
    return (part, float(thr) if thr is not None else float("inf"))


def band_for_sigma(sigma: Optional[float]) -> Optional[str]:
    """Harness distance band for the wall's σ-distance (A3); None when unknown."""
    if sigma is None:
        return None
    return distance_band(float(sigma))


def reference_cell_dict(r: Optional[dict]) -> Optional[dict]:
    """JSON-ready view of one reference cell; None when absent."""
    if r is None:
        return None
    return {
        "band":          r.get("distance_band"),
        "mean_pnl":      _f(r.get("mean_pnl")),
        "win_rate":      _f(r.get("win_rate")),
        "wilson_lo":     _f(r.get("wilson_lo")),
        "wilson_hi":     _f(r.get("wilson_hi")),
        "n":             r.get("n_settled"),
        "n_dates":       r.get("n_dates"),
        "baseline_mean": _f(r.get("baseline_mean")),
        "beat_baseline": _f(r.get("beat_baseline")),
        "threshold":     _f(r.get("threshold")),
        "partition":     r.get("partition"),
        "mean_width_actual": _f(r.get("mean_width_actual")),
    }


def _f(v) -> Optional[float]:
    return float(v) if v is not None else None


# ── Percentiles (decision 6) ─────────────────────────────────────────────────


def percentile_rank(corpus: list[float], x: float) -> Optional[float]:
    """Percentile (0–100) of `x` against `corpus`, kind='mean' — the average of
    the strict (<) and weak (≤) percentiles, matching vol_features._pct_of.
    None when the corpus is empty.
    """
    vals = [float(v) for v in corpus if v is not None]
    n = len(vals)
    if n == 0:
        return None
    lt = sum(1 for v in vals if v < x)
    le = sum(1 for v in vals if v <= x)
    return round(100.0 * (lt + le) / (2.0 * n), 2)


def quantile(values: list[float], p: float) -> Optional[float]:
    """Linear-interpolated quantile (p in [0, 1]) of `values`; None if empty."""
    vals = sorted(float(v) for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return vals[0]
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


# ── KNN factor rows (decision 6) ─────────────────────────────────────────────

# Display metadata per feature key: label, group, low-end / high-end words.
# Every FEATURE_NAMES key is covered (checked by a test); the order of the
# groups is the mockup's.
FACTOR_META: dict[str, dict] = {
    # Regime / structure
    "is_magnet_day":          {"label": "Magnet day",            "group": "Regime",          "lo": "no",       "hi": "yes"},
    "is_pin_day":             {"label": "Pin day",               "group": "Regime",          "lo": "no",       "hi": "yes"},
    "is_bounded_day":         {"label": "Bounded day",           "group": "Regime",          "lo": "no",       "hi": "yes"},
    "is_untethered_day":      {"label": "Untethered day",        "group": "Regime",          "lo": "no",       "hi": "yes"},
    "is_amplification_day":   {"label": "Amplification day",     "group": "Regime",          "lo": "no",       "hi": "yes"},
    "magnet_direction_signed": {"label": "Magnet direction",     "group": "Regime",          "lo": "below",    "hi": "above"},
    # Wall geometry
    "cluster_1_signed_distance_sigma": {"label": "Distance to wall",  "group": "Wall geometry", "lo": "closer",  "hi": "farther"},
    "cluster_1_max_gex":      {"label": "Wall size",             "group": "Wall geometry",   "lo": "smaller",  "hi": "larger"},
    "cluster_1_quality_ordinal": {"label": "Wall quality",       "group": "Wall geometry",   "lo": "diffuse",  "hi": "sharp"},
    "top_cluster_fraction_of_total_max_gex": {"label": "Wall dominance", "group": "Wall geometry", "lo": "crowded", "hi": "one wall"},
    "cluster_2_signed_distance_sigma": {"label": "2nd cluster distance", "group": "Wall geometry", "lo": "closer", "hi": "farther"},
    "cluster_2_max_gex":      {"label": "2nd cluster size",      "group": "Wall geometry",   "lo": "smaller",  "hi": "larger"},
    "cluster_2_quality_ordinal": {"label": "2nd cluster quality", "group": "Wall geometry",  "lo": "diffuse",  "hi": "sharp"},
    "cluster_3_signed_distance_sigma": {"label": "3rd cluster distance", "group": "Wall geometry", "lo": "closer", "hi": "farther"},
    "cluster_3_max_gex":      {"label": "3rd cluster size",      "group": "Wall geometry",   "lo": "smaller",  "hi": "larger"},
    "cluster_3_quality_ordinal": {"label": "3rd cluster quality", "group": "Wall geometry",  "lo": "diffuse",  "hi": "sharp"},
    # Cluster counts
    "n_clusters_total":       {"label": "Clusters total",        "group": "Cluster counts",  "lo": "few",      "hi": "many"},
    "n_clusters_above_spot":  {"label": "Clusters above spot",   "group": "Cluster counts",  "lo": "few",      "hi": "many"},
    "n_clusters_below_spot":  {"label": "Clusters below spot",   "group": "Cluster counts",  "lo": "few",      "hi": "many"},
    "n_pin":                  {"label": "Pin clusters",          "group": "Cluster counts",  "lo": "few",      "hi": "many"},
    "n_target":               {"label": "Target clusters",       "group": "Cluster counts",  "lo": "few",      "hi": "many"},
    "n_feature":              {"label": "Feature clusters",      "group": "Cluster counts",  "lo": "few",      "hi": "many"},
    # Gamma by expiry
    "dominance_0DTE":         {"label": "0DTE share",            "group": "Gamma by expiry", "lo": "little",   "hi": "lots"},
    "dominance_1_7":          {"label": "1–7 DTE share",         "group": "Gamma by expiry", "lo": "little",   "hi": "lots"},
    "dominance_8_30":         {"label": "8–30 DTE share",        "group": "Gamma by expiry", "lo": "little",   "hi": "lots"},
    "dominance_30plus":       {"label": "30+ DTE share",         "group": "Gamma by expiry", "lo": "little",   "hi": "lots"},
    # Negative zones
    "n_neg_zones":            {"label": "Negative zones",        "group": "Negative gamma",  "lo": "few",      "hi": "many"},
    "nearest_neg_signed_distance_sigma": {"label": "Nearest negative zone", "group": "Negative gamma", "lo": "close below", "hi": "far below"},
    "total_neg_max_gex":      {"label": "Negative gamma total",  "group": "Negative gamma",  "lo": "small",    "hi": "large"},
    # Volatility
    "implied_move_1d":        {"label": "Implied move",          "group": "Volatility surface", "lo": "quiet", "hi": "volatile"},
    "term_structure_slope":   {"label": "Term structure slope",  "group": "Volatility surface", "lo": "inverted", "hi": "steep"},
    "skew_percentile":        {"label": "Skew",                  "group": "Volatility surface", "lo": "flat",  "hi": "steep"},
    "smile_convexity":        {"label": "Smile convexity",       "group": "Volatility surface", "lo": "flat",  "hi": "curved"},
    "atm_iv_percentile":      {"label": "IV rank",               "group": "Volatility surface", "lo": "low",   "hi": "high"},
    "vol_risk_premium":       {"label": "Vol risk premium",      "group": "Volatility surface", "lo": "IV < RV", "hi": "IV > RV"},
}

GROUP_ORDER = ("Regime", "Wall geometry", "Cluster counts", "Gamma by expiry", "Negative gamma", "Volatility surface")


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def knn_factor_rows(
    today_features: dict,
    feature_weights: Optional[dict],
    corpus_vectors: list[dict],
    analogue_vectors: list[dict],
) -> list[dict]:
    """One row per KNN feature key (decision 6).

    today      — today's value (None → "not populated")
    weight     — from the KNN config (missing key → 1.0, the config default)
    percentile — today vs `corpus_vectors` (the magnet-above corpus before today)
    band_lo / band_hi — 25th / 75th percentile of the feature across
                 `analogue_vectors`, expressed in value space
    band_lo_pct / band_hi_pct — the same band placed on the corpus percentile
                 scale so the bar can shade it
    in_band    — band_lo ≤ today ≤ band_hi (None when today or the band is missing)
    populated  — today is not None
    """
    weights = feature_weights or {}
    rows: list[dict] = []
    for key in FEATURE_NAMES:
        meta = FACTOR_META.get(key, {"label": key, "group": "Other", "lo": "low", "hi": "high"})
        today = _num(today_features.get(key))
        corpus = [x for x in (_num(v.get(key)) for v in corpus_vectors) if x is not None]
        analog = [x for x in (_num(v.get(key)) for v in analogue_vectors) if x is not None]
        pct = percentile_rank(corpus, today) if today is not None else None
        b_lo = quantile(analog, 0.25)
        b_hi = quantile(analog, 0.75)
        in_band: Optional[bool] = None
        if today is not None and b_lo is not None and b_hi is not None:
            in_band = b_lo <= today <= b_hi
        rows.append({
            "key":          key,
            "label":        meta["label"],
            "group":        meta["group"],
            "lo_label":     meta["lo"],
            "hi_label":     meta["hi"],
            "today":        today,
            "populated":    today is not None,
            "weight":       float(weights.get(key, 1.0)),
            "percentile":   pct,
            "corpus_n":     len(corpus),
            "band_lo":      b_lo,
            "band_hi":      b_hi,
            "band_lo_pct":  percentile_rank(corpus, b_lo) if b_lo is not None else None,
            "band_hi_pct":  percentile_rank(corpus, b_hi) if b_hi is not None else None,
            "analogue_n":   len(analog),
            "in_band":      in_band,
        })
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    rows.sort(key=lambda r: (order.get(r["group"], len(order)), FEATURE_NAMES.index(r["key"])))
    return rows


def match_quality(rows: list[dict]) -> dict:
    """{in_band, total, outliers: [keys]} over rows with a computable in_band."""
    scored = [r for r in rows if r.get("in_band") is not None]
    return {
        "in_band":  sum(1 for r in scored if r["in_band"]),
        "total":    len(scored),
        "outliers": [r["key"] for r in scored if not r["in_band"]],
    }


# ── P&L and verdict (decisions 2, 5) ─────────────────────────────────────────


def expected_pnl(fv: dict, quote: Optional[float], fee_pts: float) -> dict:
    """Expected P&L at the quote: fair − quote − fees; interval from the
    bootstrap 2.5 / 97.5 of the mean, minus the same. None when either side
    is missing.
    """
    fair = fv.get("fair") if fv else None
    if fair is None or quote is None:
        return {"expected": None, "lo": None, "hi": None, "fee_pts": fee_pts}
    return {
        "expected": round(fair - quote - fee_pts, 4),
        "lo":       round(fv["boot_lo"] - quote - fee_pts, 4) if fv.get("boot_lo") is not None else None,
        "hi":       round(fv["boot_hi"] - quote - fee_pts, 4) if fv.get("boot_hi") is not None else None,
        "fee_pts":  fee_pts,
    }


def verdict(quote: Optional[float], max_price: Optional[float], *, listed: bool) -> dict:
    """Decision 5. Never "wait".

    enter        — quote ≤ max
    skip         — quote > max
    no_structure — the pair snapper found nothing listed within the cap
    no_quote     — structure listed but no valid quote at the entry minute,
                   or no analogue fair value to compare against
    """
    if not listed:
        return {"code": "no_structure", "text": "no listed structure at this expiry"}
    if quote is None:
        return {"code": "no_quote", "text": "no valid quote at the entry minute — nothing to compare"}
    if max_price is None:
        return {"code": "no_quote", "text": "no analogue fair value — nothing to compare"}
    if quote <= max_price:
        return {"code": "enter", "text": "quote is under the max — enter at the open"}
    return {"code": "skip", "text": "skip — quote above max"}


# ── Analogue facts ───────────────────────────────────────────────────────────


def horizon_mix(outcome_rows: list[dict]) -> dict[str, int]:
    """Counts of `horizon_sessions` across computed analogue outcome rows,
    keyed by the session count as a string (JSON-safe), ascending."""
    mix: dict[int, int] = {}
    for r in outcome_rows:
        h = r.get("horizon_sessions")
        if h is None:
            continue
        mix[int(h)] = mix.get(int(h), 0) + 1
    return {str(k): mix[k] for k in sorted(mix)}


def next_reference_run(after: Optional[dt.date]) -> Optional[str]:
    """First day of the month after `after` (CR-AU decision 3: the reference
    re-run cron fires on the 1st). None when there is no prior run date."""
    if after is None:
        return None
    y, m = after.year, after.month
    if m == 12:
        return dt.date(y + 1, 1, 1).isoformat()
    return dt.date(y, m + 1, 1).isoformat()


def seed_for_date(d: dt.date) -> int:
    """Bootstrap seed = the trade date as YYYYMMDD (decision 2)."""
    return d.year * 10000 + d.month * 100 + d.day


# ── By-minute quote strip (detail toggle, A8) ────────────────────────────────

# Entry window the daily capture stores (CR-AU: 06:30–06:45 PT) plus the v1
# pl-data minute (07:00). Cache only — never a fetch.
QUOTE_WINDOW_PT = (dt.time(6, 30), dt.time(7, 0))


def net_debit_by_minute(bars_by_side: dict[str, list], width: Optional[float]) -> list[dict]:
    """Spread price per minute from per-leg minute bars already in the cache.

    bars_by_side: {"long": [bar, ...], "short": [bar, ...]} where each bar has
    snapshot_pt, bid_price, ask_price. A minute is valid only when both legs
    pass the CR-AN leg rule and the debit lies in [0, width] (the CR-AN range
    rule); invalid minutes are kept with net_debit=None so the strip shows
    the gap rather than hiding it.
    Returns [{minute: 'HH:MM', net_debit, valid}] ascending.
    """
    def _by_minute(bars):
        out = {}
        for b in bars or []:
            key = b.snapshot_pt.replace(second=0, microsecond=0)
            out[key] = b
        return out

    long_ = _by_minute(bars_by_side.get("long"))
    short = _by_minute(bars_by_side.get("short"))
    rows = []
    for minute in sorted(set(long_) | set(short)):
        lb, sb = long_.get(minute), short.get(minute)
        net = None
        valid = False
        if lb is not None and sb is not None \
                and leg_quote_is_valid(lb.bid_price, lb.ask_price) \
                and leg_quote_is_valid(sb.bid_price, sb.ask_price):
            lm = (float(lb.bid_price) + float(lb.ask_price)) / 2.0
            sm = (float(sb.bid_price) + float(sb.ask_price)) / 2.0
            net = round(lm - sm, 4)
            valid = (0.0 <= net <= float(width)) if width else net >= 0.0
            if not valid:
                net = None
        rows.append({"minute": minute.strftime("%H:%M"), "net_debit": net, "valid": valid})
    return rows
