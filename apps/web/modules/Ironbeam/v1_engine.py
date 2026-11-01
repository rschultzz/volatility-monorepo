"""CR-DU — inputs for the v1 rules engine (`react_price_preview/src/v1Engine.js`), read-only.

`GET /api/ironbeam/v1-engine/inputs?trade_date=` (callbacks.py) and `scripts/cr_du_backtest_v1.py` both call
`build_inputs`, so the overlay, the live panel and the train backtest feed the engine the same data. The cells payload
itself (bars, flow, absorption marks) is the chart's `/api/ironbeam/flow/cells` (flow_v3); this adds what the engine
needs on top, all as of the session's minutes (Step 0 Q1 fixes 3 and 5, Amendment 1):

  - `blocked`: why the overlay must not run on this date (holdout, reserved R-a / R-b, set 2 in progress, warm-up);
  - `baseline` / `contra`: train dates — the prior-sessions-only box baseline and contradiction-bar percentiles
    (`data/cr_du_prior_baselines.json`, `scripts/cr_du_prior_baselines.py`); forward dates — no baseline (the frontend's
    fixed train `boxBaseline.json`) and Ironbeam contradiction percentiles from all train sessions;
  - `swing`: the A7 swing, 12 × the session's implied move ÷ the 2026-train median (CR-CP `swing_of`; 12 when missing);
  - `gex`: the strong (A12 yellow / darkest) corrected levels per RTH minute — the chart's `_select_levels` (gross ≥ 40
    $bn) over the band of the session-window bars up to that minute ± `GEX_LEVEL_PADDING`, strong = |net| ≥ the colour
    span (`_compute_color_span`); one entry per band change;
  - `seconds`: the RTH segment's 1-s OHLC (same table and contract as the segment), for same-minute fills and exits.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from sqlalchemy import text

from modules.Ironbeam import flow_v2

TRAIN_FIRST, TRAIN_END = dt.date(2026, 1, 2), dt.date(2026, 6, 5)
HOLDOUT = (dt.date(2026, 6, 6), dt.date(2026, 10, 26))
SET2_NAME = "set-2:cp-calls"
SWING_REF, IM_TRAIN_MEDIAN = 12.0, 65.71          # A7 / CR-CP C2 (2026-train median implied move)
GROSS_MIN = 40e9                                  # the chart's level filter
CANONICAL_FEATURE_VERSION = "v0.6.1-nyse-sessions"
PRIOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cr_du_prior_baselines.json")

BLOCK_HOLDOUT = "holdout date: not shown"
BLOCK_RESERVED = "reserved date (R-a / R-b): not shown"
BLOCK_SET2 = "set 2 in progress"
BLOCK_WARMUP = "warm-up session: no prior-sessions baseline"


@lru_cache(maxsize=1)
def prior_baselines() -> dict:
    with open(PRIOR_FILE) as f:
        return json.load(f)


def blocked_reason(d: dt.date, set2_open_dates: set[dt.date], warmup: set[dt.date] | None = None) -> str | None:
    """Why the v1-trades overlay is disabled on session `d`, or None. `set2_open_dates`: every set-2 card's session
    while any set-2 card has no review (empty once set 2 is reviewed)."""
    if HOLDOUT[0] <= d <= HOLDOUT[1]:
        return BLOCK_HOLDOUT
    if d < TRAIN_FIRST:
        return BLOCK_RESERVED
    if d in set2_open_dates:
        return BLOCK_SET2
    if warmup is not None and d in warmup:
        return BLOCK_WARMUP
    return None


def set2_open_dates(engine) -> set[dt.date]:
    q = text("""SELECT DISTINCT c.session_date FROM setup_candidates c WHERE c.set_name = :s
                AND EXISTS (SELECT 1 FROM setup_candidates x WHERE x.set_name = :s
                            AND NOT EXISTS (SELECT 1 FROM setup_reviews r WHERE r.candidate_id = x.id))""")
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(q, {"s": SET2_NAME})}


def swing_pts(engine, d: dt.date) -> tuple[float, str | None]:
    q = text("""SELECT (feature_vector->>'implied_move_1d')::float FROM bt_daily_features
                WHERE feature_version = :v AND ticker = 'SPX' AND active AND trade_date = :d LIMIT 1""")
    with engine.connect() as conn:
        im = conn.execute(q, {"v": CANONICAL_FEATURE_VERSION, "d": d}).scalar()
    if im is None or not np.isfinite(im) or im <= 0:
        return SWING_REF, "no implied move for the session: fixed 12-pt swing"
    return SWING_REF * float(im) / IM_TRAIN_MEDIAN, None


def gex_schedule(ib, d: dt.date, rth_bar_t: list[int], until_utc: dt.datetime | None) -> dict:
    """Strong corrected levels per RTH minute as of that minute (`ib` = modules.Ironbeam.callbacks)."""
    try:
        df_c, _, why = ib._corrected_gex_tables(d)
    except Exception as e:                                    # error type only (no URLs)
        return {"reason": f"error: {type(e).__name__}", "schedule": []}
    if df_c is None:
        return {"reason": why, "schedule": []}
    pt_tz = flow_v2.PT_TZ
    start, end = ib._session_window_pt(d, pt_tz)
    if until_utc is not None:
        end = min(end, until_utc.astimezone(pt_tz))
    bars = ib._fetch_bars_pt(start, end, "1m", pt_tz)
    if bars is None or bars.empty:
        return {"reason": "no_bars", "schedule": []}
    bt = np.array([int(x.timestamp()) for x in bars["datetime_pt"]], dtype=np.int64)   # bar starts, UTC epochs
    hi = np.maximum.accumulate(bars["high"].to_numpy(dtype=float))
    lo = np.minimum.accumulate(bars["low"].to_numpy(dtype=float))
    pad = float(ib.GEX_LEVEL_PADDING)
    out, last = [], None
    for t in rth_bar_t:                                       # decision at the close of bar t: bars starting <= t
        k = int(np.searchsorted(bt, t, side="right")) - 1
        if k < 0:
            continue
        band = (float(lo[k]) - pad, float(hi[k]) + pad)
        if band == last:
            continue
        last = band
        df = ib._select_levels(df_c, band[0], band[1], GROSS_MIN)
        levels = []
        if df is not None and not df.empty:
            _, span = ib._compute_color_span(df["net_gamma"].to_numpy(dtype=float))
            levels = [{"price": float(x.level), "net": float(x.net_gamma), "side": "call" if x.net_gamma > 0 else "put"}
                      for x in df.itertuples() if abs(float(x.net_gamma)) >= span]
        out.append({"from": int(t), "band": [band[0], band[1]], "levels": sorted(levels, key=lambda v: v["price"])})
    return {"reason": None, "schedule": out}


def fetch_seconds_ohlc(engine, table: str, symbol: str, a: dt.datetime, b: dt.datetime) -> dict:
    q = text(f"""SELECT extract(epoch FROM ts_utc)::bigint AS ts, open::float8 AS o, high::float8 AS h, low::float8 AS l,
                        close::float8 AS c
                 FROM {table} WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b ORDER BY ts_utc""")
    with engine.connect() as conn:
        r = pd.read_sql(q, conn, params={"s": symbol, "a": a.astimezone(flow_v2.UTC), "b": b.astimezone(flow_v2.UTC)})
    return {"t": r["ts"].astype(int).tolist(), **{k: r[k].astype(float).tolist() for k in ("o", "h", "l", "c")}}


def build_inputs(engine, ib, d: dt.date, cells_payload: dict, *, until_pt: dt.datetime | None = None,
                 seconds: bool = True, check_blocked: bool = True) -> dict:
    """The engine's inputs for session `d` on top of its cells payload (`flow_v3.build_cells_payload`)."""
    pri = prior_baselines()
    warmup = {dt.date.fromisoformat(x) for x in pri["warmup_sessions"]}
    out = {"mode": "v1-engine-inputs", "trade_date": d.isoformat(), "blocked": None}
    if check_blocked:
        out["blocked"] = blocked_reason(d, set2_open_dates(engine), warmup)
        if out["blocked"]:
            return out
    rth = next((s for s in cells_payload.get("segments", []) if s.get("name") == "rth"), None)
    out["source"] = rth.get("source") if rth else None
    if d <= TRAIN_END:
        ses = pri["sessions"].get(d.isoformat())
        out["baseline"] = None if ses is None else {"meta": pri["meta"], "databento": ses["databento"]}
        out["contra"] = None if ses is None else {"source": "databento", "buckets": ses["contra"]["databento"]}
    else:
        out["baseline"] = None                                # the frontend's fixed train baseline (boxBaseline.json)
        out["contra"] = {"source": "ironbeam", "buckets": pri["forward"]["contra"]["ironbeam"]}
    out["swing"], out["swing_note"] = swing_pts(engine, d)
    until_utc = None if until_pt is None else until_pt.astimezone(flow_v2.UTC)
    out["gex"] = gex_schedule(ib, d, rth["bars"]["t"] if rth else [], until_utc)
    out["seconds"] = None
    if seconds and rth:
        table = flow_v2.DATABENTO_TABLE if rth["source"] == "databento" else flow_v2.IRONBEAM_TABLE
        a = dt.datetime.fromtimestamp(rth["start"], flow_v2.UTC)
        b = dt.datetime.fromtimestamp(rth["end"], flow_v2.UTC)
        if until_utc is not None:
            b = min(b, until_utc)
        out["seconds"] = fetch_seconds_ohlc(engine, table, rth["symbol"], a, b)
    return out
