#!/usr/bin/env python3
"""
explore_gex_landscape.py
========================

Phase 0 prototype for the GEX landscape visualization. NO DB writes, NO cron
changes, NO frontend touch. Reads orats_oi_gamma, computes the landscape via
DTE-scaled Gaussian spread, finds walls + containment zones, writes PNGs to
outputs/ for visual validation.

Goal: confirm the continuous landscape adds signal beyond the discrete GEX
lines currently rendered, BEFORE investing in the cron/endpoint/frontend build.

Pipeline
--------
orats_oi_gamma (per-strike, per-expiration, in $-space):
    gex_call = gamma × S² × call_oi × 100
    gex_put  = gamma × S² × put_oi  × 100
    discounted_level = strike × exp((r-q) × (dte+1)/252)

The script:
1.  Queries every row for trade_date where expir_date >= trade_date.
2.  For each candidate spot price S in [spot ± range_pts]:
        GEX(S) = Σ over rows [ net_gex × exp(-(S - level)² / (2 × σ²)) ]
    where σ = spread_coef × sqrt(max(dte, 0.5)).
    Net_gex = gex_call - |gex_put|  (matches frontend convention).
3.  Decomposes the landscape into 4 DTE buckets for the stacked view.
4.  scipy.signal.find_peaks on |landscape| → walls (positive + negative).
5.  For each adjacent pair of POSITIVE walls → containment zone with
    score = (min_strength × valley_depth) / width.

Env vars (reads .env from repo root):
    DATABASE_URL

Usage
-----
    # Single date:
    python scripts/explore_gex_landscape.py --date 2026-05-19

    # Multiple dates:
    python scripts/explore_gex_landscape.py --dates 2026-05-15,2026-05-16,2026-05-19

    # Tune the Gaussian spread coefficient (default 8.0):
    #   c=6  → tighter peaks (0DTE ~4.2pt, 30DTE ~33pt)
    #   c=8  → balanced     (0DTE ~5.7pt, 30DTE ~44pt)
    #   c=12 → broader      (0DTE ~8.5pt, 30DTE ~66pt)
    python scripts/explore_gex_landscape.py --date 2026-05-19 --spread-coef 6

    # Skip cache (force re-query):
    python scripts/explore_gex_landscape.py --date 2026-05-19 --no-cache

Outputs per date:
    outputs/landscape_<date>_compare.png  — side-by-side: GEX lines vs landscape
    outputs/landscape_<date>_stacked.png  — landscape decomposed by DTE bucket

The script prints structured stdout for each date so you can scan results
without opening every PNG. Walls and the active containment zone are printed.

Iteration loop
--------------
1. Run on 5-10 days you remember (a heavy-pin day, a trend day, an FOMC day, etc).
2. Look at the right panel of each *_compare.png. Is the wall structure obvious?
   Does the active zone bracket where price actually traded?
3. If walls are too "spiky" → bump spread_coef. If they're too washed out → lower it.
4. Once one coefficient looks good across diverse days → proceed to Phase 1 (cron).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ─── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
CACHE_DIR  = SCRIPT_DIR / ".cache"
OUTPUT_DIR = REPO_ROOT / "outputs"


# ─── Env loading (matches gamma_pin_study.py pattern) ───────────────────────

def load_env():
    for parent in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(env_path)
                return env_path
            except ImportError:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                return env_path
    return None


# ─── DB query ───────────────────────────────────────────────────────────────

def _make_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine.url import make_url

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")
    url = make_url(db_url)
    if url.get_backend_name() == "postgresql" and url.get_driver_name() in (None, "", "psycopg2"):
        url = url.set(drivername="postgresql+psycopg")
    return create_engine(url, pool_pre_ping=True)


def fetch_strike_data(trade_date: dt.date, ticker: str, use_cache: bool) -> pd.DataFrame:
    """
    Pull per-strike, per-expiration GEX rows for one trade_date.

    Matches the filter convention from _fetch_gex_grouped_by_level in
    apps/web/modules/Ironbeam/callbacks.py (expir_date >= trade_date) so we
    work with live contracts only — no already-expired snapshot residue.

    Returns columns: discounted_level, strike, expir_date, dte, stock_price,
                     call_oi, put_oi, gamma, gex_call, gex_put
    """
    from sqlalchemy import text

    cache_key = CACHE_DIR / f"oi_gamma_full_{ticker}_{trade_date.isoformat()}.parquet"
    if use_cache and cache_key.exists():
        try:
            return pd.read_parquet(cache_key)
        except Exception:
            pass  # fall through to re-query

    engine = _make_engine()
    sql = text("""
        SELECT
            discounted_level,
            strike,
            expir_date,
            dte,
            stock_price,
            call_oi,
            put_oi,
            gamma,
            gex_call,
            gex_put
        FROM orats_oi_gamma
        WHERE ticker = :tkr
          AND trade_date = :d
          AND expir_date >= :d
          AND discounted_level IS NOT NULL
        ORDER BY expir_date, discounted_level
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"tkr": ticker, "d": trade_date.isoformat()})

    if use_cache and not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache_key)
        except Exception as e:
            print(f"  [cache] save failed: {e}", file=sys.stderr)

    return df


# ─── Shared analytical module (CR-007) ──────────────────────────────────────
#
# The landscape math now lives in packages/shared/gex_landscape.py so the same
# functions back this CLI script, the EOD cron, the backfill script, and a
# future endpoint. This script keeps the CLI, the DB query (fetch_strike_data),
# the matplotlib plotting, and the run_one_date orchestration.

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.gex_landscape import (  # noqa: E402
    _annotate_distance_class,
    analyze_confluence,
    classify_distance,
    classify_per_bucket,
    classify_regime,
    compute_implied_move,
    compute_landscape,
    find_intraday_subtarget,
    find_proximate_negative_zones,
    find_walls,
    score_containment_zones,
    summarize_per_bucket,
)


# ─── Plotting ───────────────────────────────────────────────────────────────

# Dark theme matching the dashboard
BG_OUTER = "#0b1220"
BG_INNER = "#1f2937"
GRID     = "#475569"
TEXT_HI  = "#f1f5f9"
TEXT_MID = "#cbd5e1"
TEXT_LO  = "#94a3b8"
SPOT_C   = "#fbbf24"  # amber
POS_C    = "#fbbf24"  # warm — positive GEX (contains)
NEG_C    = "#06b6d4"  # cool — negative GEX (amplifies)
ACTIVE_C = "#10b981"  # green — active containment zone


def _style_axis(ax):
    ax.set_facecolor(BG_INNER)
    ax.tick_params(colors=TEXT_LO)
    for s in ax.spines.values():
        s.set_color(GRID)


def plot_compare(
    landscape: pd.DataFrame,
    df_strikes: pd.DataFrame,
    walls: pd.DataFrame,
    zones: pd.DataFrame,
    spot: float,
    trade_date: dt.date,
    spread_coef: float,
    out_path: Path,
    regime: Optional[dict] = None,
):
    """
    Side-by-side: left = discrete GEX lines (the current viz, but unfiltered
    so you see everything the cron has), right = continuous landscape with
    walls and zones marked. Shared Y axis.

    If a regime dict is passed, a single-line summary is added as a subtitle.
    """
    fig, (ax_lines, ax_land) = plt.subplots(
        1, 2, figsize=(15, 10),
        gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.03},
        sharey=True,
    )
    fig.patch.set_facecolor(BG_OUTER)
    _style_axis(ax_lines)
    _style_axis(ax_land)

    y_min, y_max = landscape["price"].min(), landscape["price"].max()

    # ─── Left: discrete GEX lines, ALL strikes (not the filtered subset) ───
    # Aggregate per integer level (matches the cron's bucket=1)
    by_level = (
        df_strikes.assign(level=df_strikes["discounted_level"].round().astype(int))
        .groupby("level")
        .agg(
            call_gamma=("gex_call", "sum"),
            put_gamma=("gex_put", lambda x: -np.abs(x).sum()),
        )
        .reset_index()
    )
    by_level["net"] = by_level["call_gamma"] + by_level["put_gamma"]
    by_level = by_level[(by_level["level"] >= y_min) & (by_level["level"] <= y_max)]

    max_abs = float(by_level["net"].abs().max()) if len(by_level) else 1.0
    max_abs = max(max_abs, 1.0)

    for _, r in by_level.iterrows():
        net = float(r["net"])
        norm = min(1.0, abs(net) / max_abs)
        color = POS_C if net > 0 else NEG_C
        lw    = 0.4 + 3.6 * norm
        alpha = 0.15 + 0.70 * norm
        ax_lines.axhline(r["level"], color=color, linewidth=lw, alpha=alpha)

    ax_lines.axhline(spot, color=SPOT_C, linewidth=1.6, linestyle="--",
                     label=f"Spot {spot:.1f}", zorder=10)
    ax_lines.set_xlim(0, 1)
    ax_lines.set_xticks([])
    ax_lines.set_title(
        "Current viz · GEX lines\n(all strikes, unfiltered)",
        color=TEXT_MID, fontsize=11, pad=10,
    )
    ax_lines.set_ylabel("Price (discounted)", color=TEXT_MID)
    ax_lines.legend(loc="upper left", facecolor=BG_OUTER, edgecolor=GRID,
                    labelcolor=TEXT_MID, fontsize=9, framealpha=0.9)

    # ─── Right: landscape profile (proposed viz) ───
    price = landscape["price"].to_numpy()
    gex_b = landscape["gex_total"].to_numpy() / 1e9

    pos_mask = gex_b >= 0
    neg_mask = gex_b < 0

    # Use np.where to safely fill (preserve NaN where mask is false)
    pos_vals = np.where(pos_mask, gex_b, 0.0)
    neg_vals = np.where(neg_mask, gex_b, 0.0)

    ax_land.fill_betweenx(price, 0, pos_vals, color=POS_C, alpha=0.55,
                          label="positive GEX (contains)")
    ax_land.fill_betweenx(price, neg_vals, 0, color=NEG_C, alpha=0.55,
                          label="negative GEX (amplifies)")
    ax_land.plot(gex_b, price, color=TEXT_HI, linewidth=1.0, alpha=0.85)
    ax_land.axvline(0, color=GRID, linewidth=0.6, alpha=0.7)

    # Containment zone highlights — top 3, with active one outlined
    xlim_right = float(np.nanmax(np.abs(gex_b))) * 1.10 if len(gex_b) else 1.0
    xlim_right = max(xlim_right, 0.1)
    ax_land.set_xlim(-xlim_right, xlim_right)

    if not zones.empty:
        for i, z in zones.head(3).iterrows():
            is_active = bool(z["contains_spot"])
            color = ACTIVE_C if is_active else GRID
            alpha = 0.22 if is_active else 0.08
            ax_land.axhspan(z["lower_price"], z["upper_price"],
                            color=color, alpha=alpha, zorder=1)
            label = "ACTIVE" if is_active else f"#{i + 1}"
            ax_land.text(
                xlim_right * 0.97,
                (z["lower_price"] + z["upper_price"]) / 2,
                f"{label}  w={z['width_pts']:.0f}pt",
                ha="right", va="center",
                color=color, fontsize=8.5, fontweight="bold", alpha=0.95,
            )
            if is_active:
                ax_land.axhline(z["equilibrium_price"], color=ACTIVE_C,
                                linewidth=1.0, linestyle=":", alpha=0.85,
                                zorder=2)

    # Wall markers
    for _, w in walls.iterrows():
        gex_at_wall = float(w["gex"]) / 1e9
        marker_color = POS_C if w["sign"] > 0 else NEG_C
        ax_land.scatter([gex_at_wall], [w["price"]],
                        color=marker_color, edgecolor=BG_OUTER,
                        s=70, zorder=5, linewidths=1.2)

    ax_land.axhline(spot, color=SPOT_C, linewidth=1.6, linestyle="--", zorder=10)
    ax_land.set_title(
        f"Proposed viz · GEX landscape\n(continuous · walls · containment zones)",
        color=TEXT_MID, fontsize=11, pad=10,
    )
    ax_land.set_xlabel("GEX ($B)", color=TEXT_MID)
    ax_land.legend(loc="lower right", facecolor=BG_OUTER, edgecolor=GRID,
                   labelcolor=TEXT_MID, fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"GEX landscape · SPX · {trade_date.isoformat()} · spread_coef={spread_coef}",
        color=TEXT_HI, fontsize=13, fontweight="bold", y=0.995,
    )

    if regime is not None:
        # Compact one-line summary under the suptitle, color-coded by regime
        regime_tag = regime.get("regime", "?")
        regime_colors = {
            "magnetic-pin":  ACTIVE_C,
            "magnet-above":  "#fbbf24",
            "magnet-below":  "#fbbf24",
            "bounded":       ACTIVE_C,
            "amplification": NEG_C,
            "broken-magnet": "#a78bfa",  # purple — to stand out as a momentum regime
            "untethered":    TEXT_LO,
        }
        col = regime_colors.get(regime_tag, TEXT_MID)

        # Build short summary string
        bits = [f"REGIME: {regime_tag}"]
        if "crossing_direction" in regime:
            bits.append(
                f"crossed wall at {regime['dominant_wall']['price']:.0f} "
                f"({regime.get('prior_spot', 0):.0f} → {regime['spot']:.0f}, "
                f"{regime['crossing_direction']})"
            )
        elif "drift_target" in regime and "drift_direction" in regime:
            bits.append(
                f"→ target {regime['drift_target']:.0f} "
                f"(drift {regime['drift_direction']} "
                f"{abs(regime['spot'] - regime['drift_target']):.0f}pt)"
            )
        if "containment_zone" in regime:
            z = regime["containment_zone"]
            bits.append(
                f"→ zone {z['lower_price']:.0f}–{z['upper_price']:.0f} "
                f"(eq {z['equilibrium']:.0f}, w {z['width_pts']:.0f}pt)"
            )
        if "amplification_zone" in regime:
            z = regime["amplification_zone"]
            bits.append(f"⚠ neg zone at {z['price']:.0f} ({z['distance']:.0f}pt away)")
        summary = "  ·  ".join(bits)

        fig.text(
            0.5, 0.955, summary,
            color=col, fontsize=11, fontweight="bold",
            ha="center", va="top",
        )

    plt.savefig(out_path, dpi=130, facecolor=BG_OUTER, bbox_inches="tight")
    plt.close(fig)


def plot_stacked(
    landscape: pd.DataFrame,
    spot: float,
    trade_date: dt.date,
    out_path: Path,
    per_bucket: Optional[dict] = None,
    confluences: Optional[list] = None,
):
    """
    Single panel: landscape decomposed by DTE bucket. Shows which expirations
    drive each wall. 0DTE = sharp/intraday, structural = broad/multi-week.

    If per_bucket is provided, legend labels include dominance % and regime
    tag so the eye-roll mapping "which timeframe controls today" is immediate.

    If confluences is provided, horizontal dashed lines mark levels where
    multiple buckets have peaks at the same price (confluence points are
    structurally stronger than any single bucket's peak).
    """
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor(BG_OUTER)
    _style_axis(ax)

    price = landscape["price"].to_numpy()

    layers = [
        ("0DTE",      landscape["gex_0dte"].to_numpy()   / 1e9, "#ef4444", "gex_0dte"),
        ("1-7 DTE",   landscape["gex_near"].to_numpy()   / 1e9, "#f59e0b", "gex_near"),
        ("8-30 DTE",  landscape["gex_med"].to_numpy()    / 1e9, "#10b981", "gex_med"),
        ("30+ DTE",   landscape["gex_struct"].to_numpy() / 1e9, "#3b82f6", "gex_struct"),
    ]

    # Identify dominant bucket if per_bucket info is present
    dominant_label = None
    if per_bucket is not None:
        dominant_label = max(per_bucket.items(),
                            key=lambda x: x[1]["dominance_pct"])[0]

    for name, vals, color, _col in layers:
        # If per_bucket is given, augment the label with dominance + regime
        if per_bucket is not None and name in per_bucket:
            r = per_bucket[name]
            marker = "●" if name == dominant_label else " "
            label = (
                f"{marker} {name:8s} {r['dominance_pct']:>4.0f}% · "
                f"{r['regime']}"
            )
            # Bolder line for dominant bucket
            lw = 2.6 if name == dominant_label else 1.4
            alpha_line = 1.0 if name == dominant_label else 0.75
        else:
            label = name
            lw = 1.6
            alpha_line = 0.9

        ax.plot(vals, price, color=color, linewidth=lw, label=label,
                alpha=alpha_line)
        ax.fill_betweenx(price, 0, vals, color=color, alpha=0.15)

    ax.plot(landscape["gex_total"].to_numpy() / 1e9, price,
            color=TEXT_HI, linewidth=1.0, linestyle="--",
            alpha=0.55, label="total")

    ax.axvline(0, color=GRID, linewidth=0.6, alpha=0.7)
    ax.axhline(spot, color=SPOT_C, linewidth=1.6, linestyle="--",
               label=f"Spot {spot:.1f}", zorder=10)

    # Confluence lines: horizontal markers at price levels where multiple
    # buckets agree on a peak. Width and opacity scale with n_buckets;
    # line style reflects quality grade.
    if confluences:
        xlim = ax.get_xlim()
        x_right = xlim[1]
        for i, c in enumerate(confluences[:4]):
            n = c["n_buckets"]
            # Color from confluence strength: 2 buckets = yellow, 3 = orange, 4 = bright green
            colors = {2: "#fbbf24", 3: "#fb923c", 4: ACTIVE_C}
            line_color = colors.get(n, ACTIVE_C)
            line_alpha = 0.35 + 0.20 * (n - 2)
            # Line style by quality: pin solid, target dashed, feature dotted
            quality = c.get("quality", "feature")
            style_map = {"pin": "-", "target": "--", "feature": ":"}
            line_style = style_map.get(quality, ":")
            line_width = {"pin": 2.0, "target": 1.5, "feature": 1.0}.get(
                quality, 1.0
            )
            ax.axhline(c["center_price"], color=line_color,
                       linewidth=line_width, linestyle=line_style,
                       alpha=line_alpha, zorder=8)
            # Annotation on the right edge — small but readable
            quality_short = {"pin": "PIN", "target": "TGT",
                             "feature": "soft"}.get(quality, "")
            ax.text(
                x_right * 0.985,
                c["center_price"],
                f"{c['center_price']:.0f}  {'★' * n}  {quality_short}",
                ha="right", va="center",
                color=line_color, fontsize=8.5, fontweight="bold",
                alpha=0.9,
                bbox=dict(facecolor=BG_OUTER, edgecolor="none", pad=2, alpha=0.7),
            )

    ax.set_xlabel("GEX ($B) — contribution by DTE bucket", color=TEXT_MID)
    ax.set_ylabel("Price (discounted)", color=TEXT_MID)

    ax.legend(loc="upper right", facecolor=BG_OUTER, edgecolor=GRID,
              labelcolor=TEXT_MID, fontsize=10, framealpha=0.9,
              prop={"family": "monospace"})

    fig.suptitle(
        f"GEX landscape by DTE bucket · SPX · {trade_date.isoformat()}",
        color=TEXT_HI, fontsize=13, fontweight="bold", y=0.995,
    )

    plt.savefig(out_path, dpi=130, facecolor=BG_OUTER, bbox_inches="tight")
    plt.close(fig)


# ─── Per-date orchestration ─────────────────────────────────────────────────

def run_one_date(trade_date: dt.date, args, out_dir: Path):
    print(f"\n=== {trade_date.isoformat()} ===")

    df = fetch_strike_data(trade_date, args.ticker, use_cache=not args.no_cache)
    if df.empty:
        print(f"  [skip] no orats_oi_gamma rows for ticker={args.ticker} trade_date={trade_date}")
        return

    table_spot = float(df["stock_price"].dropna().iloc[0])
    prior_spot: Optional[float] = None
    if args.spot is not None:
        spot = float(args.spot)
        prior_spot = table_spot   # used for broken-magnet detection
        spot_source = f"override (table reference: {table_spot:.2f})"
    else:
        spot = table_spot
        spot_source = f"orats_oi_gamma stock_price (SPX cash, prior EOD)"

    # Implied move resolution: --implied-move takes priority, then --iv
    implied_move = 0.0
    iv_source = ""
    if args.implied_move is not None:
        implied_move = float(args.implied_move)
        iv_source = f"--implied-move {implied_move:.1f}pt"
    elif args.iv is not None:
        implied_move = compute_implied_move(spot, float(args.iv))
        iv_source = f"--iv {args.iv:.4f} → {implied_move:.1f}pt (1σ daily)"
    n_expir = df["expir_date"].nunique()
    n_strikes = df["strike"].nunique()
    print(f"  spot:        {spot:.2f}  [{spot_source}]")
    if args.spot is None:
        print(f"  note:        levels are ES-forward; consider --spot <actual ES price>")
    if implied_move > 0:
        print(f"  implied:     ±{implied_move:.1f}pt 1σ  [{iv_source}]")
        print(f"               in-range:       {spot - implied_move:.0f} → {spot + implied_move:.0f}  (1σ)")
        print(f"               intraday reach: {spot - 1.5*implied_move:.0f} → {spot + 1.5*implied_move:.0f}  (1.5σ)")
    else:
        print(f"  implied:     not provided — targets won't be classified by realistic range")
        print(f"               (pass --implied-move or --iv to enable)")
    print(f"  rows:        {len(df):,} (strikes × expirations)")
    print(f"  expirations: {n_expir}")
    print(f"  strikes:     {n_strikes}")
    print(f"  dte range:   {int(df['dte'].min())} – {int(df['dte'].max())}")

    landscape = compute_landscape(
        df, spot,
        range_pts=args.range_pts,
        step_pts=args.step_pts,
        spread_coef=args.spread_coef,
    )

    walls = find_walls(landscape, min_prominence_pct=args.prominence)
    zones = score_containment_zones(walls, landscape, spot)
    regime = classify_regime(landscape, spot, prior_spot=prior_spot,
                             implied_move=implied_move)
    regime = _annotate_distance_class(regime, implied_move)
    per_bucket = classify_per_bucket(landscape, spot, prior_spot=prior_spot,
                                     implied_move=implied_move)
    bucket_summary = summarize_per_bucket(per_bucket)
    confluence_result = analyze_confluence(landscape)

    n_pos = int((walls["sign"] > 0).sum()) if not walls.empty else 0
    n_neg = int((walls["sign"] < 0).sum()) if not walls.empty else 0
    print(f"  walls:       {len(walls)}  (pos={n_pos}, neg={n_neg})")
    if not walls.empty:
        pos_walls = walls[walls["sign"] > 0].nlargest(5, "gex")
        for _, w in pos_walls.iterrows():
            print(f"    + {w['price']:>6.0f}   gex={w['gex']/1e9:+7.1f}B")
        neg_walls = walls[walls["sign"] < 0].nsmallest(3, "gex")
        for _, w in neg_walls.iterrows():
            print(f"    – {w['price']:>6.0f}   gex={w['gex']/1e9:+7.1f}B")

    print(f"  zones:       {len(zones)}")
    if not zones.empty:
        active = zones[zones["contains_spot"]]
        if not active.empty:
            z = active.iloc[0]
            print(
                f"  active:      {z['lower_price']:.0f} → {z['upper_price']:.0f}  "
                f"(width {z['width_pts']:.0f}pt, "
                f"eq {z['equilibrium_price']:.1f}, "
                f"spot−eq {spot - z['equilibrium_price']:+.1f})"
            )
        else:
            print(f"  active:      none — spot {spot:.1f} not inside any positive-wall pair")
        print("  top zones by containment score:")
        for i, z in zones.head(3).iterrows():
            marker = "*" if z["contains_spot"] else " "
            print(
                f"    {marker} #{i + 1}: {z['lower_price']:.0f}–{z['upper_price']:.0f}  "
                f"w={z['width_pts']:>3.0f}pt  "
                f"min_strength={min(z['lower_gex'], z['upper_gex'])/1e9:.1f}B  "
                f"score={z['containment_score']:.2e}"
            )

    # Regime classification — the high-level "what kind of day is this" summary
    print()
    print(f"  REGIME:      {regime['regime'].upper()}")
    for note in regime.get("notes", []):
        # Wrap long notes to 80 chars-ish for terminal readability
        words = note.split()
        line_chars = 0
        line_parts = []
        for w in words:
            if line_chars + len(w) > 72:
                print(f"               {' '.join(line_parts)}")
                line_parts = [w]
                line_chars = len(w)
            else:
                line_parts.append(w)
                line_chars += len(w) + 1
        if line_parts:
            print(f"               {' '.join(line_parts)}")
    if "drift_target" in regime:
        cls_info = regime.get("target_classification", {})
        cls_str = ""
        if cls_info.get("sigma") is not None:
            cls_str = f"  [{cls_info['class']}, {cls_info['sigma']:.1f}σ]"
        print(
            f"               target: {regime['drift_target']:.1f}  "
            f"(direction: {regime.get('drift_direction', '?')}, "
            f"distance: {abs(regime['spot'] - regime['drift_target']):.0f}pt)"
            f"{cls_str}"
        )
    if "containment_zone" in regime:
        z = regime["containment_zone"]
        print(
            f"               zone:   {z['lower_price']:.0f} → {z['upper_price']:.0f}  "
            f"(eq {z['equilibrium']:.1f}, width {z['width_pts']:.0f}pt)"
        )
    if "amplification_zone" in regime:
        z = regime["amplification_zone"]
        print(
            f"               neg:    {z['price']:.0f} "
            f"({z['gex']/1e9:+.0f}B, {z['distance']:.0f}pt away)"
        )
    if "crossing_direction" in regime:
        print(
            f"               crossed: {regime.get('prior_spot', 0):.0f} → "
            f"{regime['spot']:.0f} through wall at {regime['dominant_wall']['price']:.0f}  "
            f"(direction: {regime['crossing_direction']}, "
            f"distance past: {regime['distance_past_magnet']:.0f}pt)"
        )

    # Per-DTE-bucket breakdown — which timeframe is in control?
    print()
    print(f"  PER-DTE BREAKDOWN  (intensity = sum of |GEX| within ±30pt of spot):")
    for label, r in per_bucket.items():
        marker = "●" if label == bucket_summary["primary_bucket"] else " "
        # Compact one-line summary per bucket
        extras = []
        if "drift_target" in r:
            extras.append(f"target {r['drift_target']:.0f}")
        if "containment_zone" in r:
            z = r["containment_zone"]
            extras.append(f"zone {z['lower_price']:.0f}–{z['upper_price']:.0f}")
        if "amplification_zone" in r:
            z = r["amplification_zone"]
            extras.append(f"neg {z['price']:.0f}")
        if "crossing_direction" in r:
            extras.append(f"crossed {r['crossing_direction']}")
        extra_str = (" → " + ", ".join(extras)) if extras else ""

        print(
            f"   {marker} {label:8s}  "
            f"dom={r['dominance_pct']:>5.1f}%  "
            f"intensity={r['local_intensity']/1e9:>6.1f}B  "
            f"net={r['local_net']/1e9:>+6.1f}B  "
            f"regime={r['regime']}"
            f"{extra_str}"
        )

    print()
    print(
        f"  DOMINANT BUCKET: {bucket_summary['primary_bucket']} "
        f"({bucket_summary['primary_dominance']:.0f}%) — "
        f"regime: {bucket_summary['primary_regime']}"
    )
    if bucket_summary["secondary_dominance"] >= 15.0:
        print(
            f"  RUNNER-UP:       {bucket_summary['secondary_bucket']} "
            f"({bucket_summary['secondary_dominance']:.0f}%) — "
            f"regime: {bucket_summary['secondary_regime']}"
        )
    if not bucket_summary["consensus"] and len(bucket_summary["meaningful_regimes"]) > 1:
        print(
            f"  DIVERGENCE:      meaningful buckets disagree — "
            f"{' / '.join(bucket_summary['meaningful_regimes'])}"
        )
        print(
            f"                   (intraday signal may diverge from structural — "
            f"the dominant bucket usually controls; watch for regime shift if "
            f"price moves out of dominant bucket's range)"
        )

    # Confluence: peaks where multiple DTE buckets agree on the same price level
    confluences = confluence_result["confluences"]
    print()
    if confluences:
        print(f"  CONFLUENCE  (peaks aligned across buckets — confluence = stronger pin):")
        for i, c in enumerate(confluences[:5]):
            marker = "●" if i == 0 else " "
            stars = "★" * c["n_buckets"]
            buckets_str = ", ".join(b.replace(" DTE", "").replace("DTE", "0d")
                                    for b in c["buckets"])
            distance = c["center_price"] - spot
            # Add distance classification if implied_move available
            if implied_move > 0:
                cls = classify_distance(abs(distance), implied_move)
                cls_tag = f"  [{cls['class']}, {cls['sigma']:.1f}σ]"
            else:
                cls_tag = ""
            quality_tag = f"  [{c['quality']}]"
            print(
                f"   {marker} {c['center_price']:>6.0f}  {stars:<4s}  "
                f"({c['n_buckets']} buckets: {buckets_str})  "
                f"max={c['max_gex']/1e9:>5.0f}B  "
                f"fwhm={c['avg_fwhm']:>4.0f}pt  "
                f"score={c['score']/1e9:.1f}B/pt"
                f"{quality_tag}  "
                f"[{distance:+.0f}pt from spot]{cls_tag}"
            )
    else:
        print(f"  CONFLUENCE:   none — no peaks aligned across multiple buckets")
        print(f"                (a no-confluence day means no clear pin level — "
              f"contributing to chop/uncertainty)")

    # Intraday subtarget — when primary target is structural, point to a closer
    # actionable level within ~1.5σ that's in the same direction.
    if implied_move > 0 and "drift_target" in regime:
        target_cls = regime.get("target_classification", {}).get("class", "")
        if target_cls in ("stretch", "multi-day", "far"):
            direction = regime.get("drift_direction")
            subtarget = find_intraday_subtarget(
                confluences, walls, spot, implied_move,
                max_sigma=1.5, direction=direction,
            )
            print()
            if subtarget:
                print(
                    f"  INTRADAY ACTIONABLE:  target {subtarget['price']:.0f}  "
                    f"({subtarget['type']}, {subtarget['sigma_distance']:.1f}σ from spot)"
                )
                print(
                    f"                        the {regime['drift_target']:.0f} magnet is "
                    f"{regime['target_classification']['sigma']:.1f}σ away "
                    f"({target_cls}) — that's a structural pull, not today's range. "
                    f"Today, watch {subtarget['price']:.0f}."
                )
            else:
                print(
                    f"  INTRADAY ACTIONABLE:  no significant level within "
                    f"1.5σ ({1.5*implied_move:.0f}pt) in the direction of drift"
                )
                print(
                    f"                        the {regime['drift_target']:.0f} magnet is "
                    f"structural pull only — today's range likely stays within "
                    f"{spot-1.5*implied_move:.0f}–{spot+1.5*implied_move:.0f} unless catalyst"
                )

    # High-volatility zones: significant negative GEX walls within 1.5σ.
    # These aren't magnets but they ARE high-attention strikes that act as
    # accelerators or bounce candidates depending on arrival conditions.
    if implied_move > 0:
        dom_strength_for_neg = (
            regime["dominant_wall"]["gex"]
            if "dominant_wall" in regime else None
        )
        neg_zones = find_proximate_negative_zones(
            walls, spot, implied_move,
            dom_strength=dom_strength_for_neg,
        )
        if neg_zones:
            print()
            print(f"  HIGH-VOL ZONES  (negative GEX within 1.5σ — accelerators, not magnets):")
            for z in neg_zones[:3]:
                arrow = "↓" if z["direction"] == "below" else "↑"
                print(
                    f"   ⚠ {z['price']:>6.0f} ({z['gex']/1e9:+5.0f}B, "
                    f"{z['distance']:.0f}pt {z['direction']} spot, "
                    f"{z['sigma']:.1f}σ)  {arrow}"
                )
                # Conditional arrival framing
                if z["direction"] == "below":
                    print(f"               grind-down approach → acceleration through "
                          f"(trapdoor)")
                    print(f"               gap-down to here    → bounce candidate "
                          f"(put profit-taking)")
                else:
                    print(f"               grind-up approach   → acceleration through "
                          f"(squeeze)")
                    print(f"               gap-up to here      → exhaustion / fade "
                          f"candidate")

    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"landscape_{trade_date.isoformat()}_compare.png"
    plot_compare(landscape, df, walls, zones, spot, trade_date, args.spread_coef, p1,
                 regime=regime)
    print(f"  → {p1.relative_to(REPO_ROOT) if p1.is_relative_to(REPO_ROOT) else p1}")

    p2 = out_dir / f"landscape_{trade_date.isoformat()}_stacked.png"
    plot_stacked(landscape, spot, trade_date, p2, per_bucket=per_bucket,
                 confluences=confluence_result["confluences"])
    print(f"  → {p2.relative_to(REPO_ROOT) if p2.is_relative_to(REPO_ROOT) else p2}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--date", help="Single trade date YYYY-MM-DD")
    ap.add_argument("--dates", help="Comma-separated list of trade dates")
    ap.add_argument("--ticker", default="SPX")
    ap.add_argument("--spot", type=float, default=None,
                    help="Override the spot used for regime classification and "
                         "plot centering. Default reads stock_price from "
                         "orats_oi_gamma, which is SPX cash from the PRIOR EOD "
                         "(when ORATS pulled the chain) — usually NOT where the "
                         "session you're analyzing actually opens. GEX levels "
                         "are in ES-forward coordinates (discounted_level), so "
                         "pass the actual ES price at the moment you want to "
                         "analyze. E.g. --spot 7400 for 5/7 session open.")
    ap.add_argument("--implied-move", type=float, default=None,
                    help="Today's expected 1-sigma move in points (e.g. ATM "
                         "straddle width). Used to classify each target as "
                         "in-range/intraday-reach/stretch/multi-day/far. "
                         "Strongly recommended — without it, the model can "
                         "report 3-sigma 'targets' that are actually multi-day "
                         "structural pulls, not today's range.")
    ap.add_argument("--iv", type=float, default=None,
                    help="Alternative to --implied-move: ATM IV as decimal "
                         "(e.g. 0.107 for 10.7%%). Implied move computed as "
                         "spot × iv × sqrt(1/252).")
    ap.add_argument("--spread-coef", type=float, default=8.0,
                    help="Gaussian spread coef: σ = c × √DTE  (default 8.0)")
    ap.add_argument("--range-pts", type=float, default=200.0,
                    help="Price range ± from spot (default 200)")
    ap.add_argument("--step-pts", type=float, default=1.0,
                    help="Price step in landscape grid (default 1)")
    ap.add_argument("--prominence", type=float, default=0.10,
                    help="Min peak prominence as fraction of max |GEX| (default 0.10)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Skip parquet cache; force DB re-query")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR),
                    help=f"Output directory (default {OUTPUT_DIR})")
    args = ap.parse_args()

    if not args.date and not args.dates:
        ap.error("Provide --date or --dates")

    env_path = load_env()
    if env_path:
        print(f"[env] loaded {env_path}")

    if not os.environ.get("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    dates: list[dt.date] = []
    if args.date:
        dates.append(dt.date.fromisoformat(args.date))
    if args.dates:
        dates.extend(dt.date.fromisoformat(d.strip()) for d in args.dates.split(",") if d.strip())

    out_dir = Path(args.out_dir)
    print(f"[config] spread_coef={args.spread_coef}  range_pts={args.range_pts}  "
          f"prominence={args.prominence}")

    for d in dates:
        try:
            run_one_date(d, args, out_dir)
        except Exception as e:
            print(f"  [error] {d}: {e}")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()