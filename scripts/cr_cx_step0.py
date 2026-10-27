"""CR-CX Step 0 — can the forward log run CR-CL's frozen G1-down code on live inputs? READ-ONLY, no outcomes.

Spec: specs/CR-CX-forward-log-g1down.md (Step 0). Reads:
  1. feed heads: latest row in each input table (no values from any session >= 2026-06-06 are read);
  2. y26 train only (2026-01-02 .. 06-05, the CR-CJ loader and its guard): the vol-scaled G1-down legs
     (CR-CO A1) on the frozen Databento bars vs the same code on Ironbeam bars, two ways —
     (a) ironbeam_es_flow_1s aggregated to minutes (front symbol, as the CR-CJ Databento query),
     (b) ironbeam_es_1m_bars (no symbol column; the stored front-month series).
     Legs and counts only; no race, move or outcome is computed.

Run: CR_CL_PERIOD=y26 apps/web/.venv/bin/python scripts/cr_cx_step0.py  ->  scripts/logs/cr_cx/step0.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CR_CL_PERIOD"] = "y26"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_cj_lib as L  # noqa: E402
import cr_cl_lib as C  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_cx"
OUT.mkdir(parents=True, exist_ok=True)


def feed_heads() -> dict:
    sqls = {
        "es_flow_1s_databento": "SELECT max(ts_utc) FROM es_flow_1s_databento",
        "es_flow_1m_databento": "SELECT max(minute_utc) FROM es_flow_1m_databento",
        "ironbeam_es_1m_bars": "SELECT max(datetime) FROM ironbeam_es_1m_bars",
        "ironbeam_es_flow_1s": "SELECT max(ts_utc) FROM ironbeam_es_flow_1s",
        "orats_monies_minute (SPX)": "SELECT max(snapshot_pt) FROM orats_monies_minute WHERE ticker = 'SPX'",
        "bt_daily_features implied_move_1d": """SELECT max(trade_date) FROM bt_daily_features
                                                 WHERE feature_vector ? 'implied_move_1d'""",
    }
    return {k: str(L.q(s).iloc[0, 0]) for k, s in sqls.items()}


def ironbeam_flow_minutes() -> pd.DataFrame:
    lo, hi = L.PERIODS["y26"]
    L.guard(lo, hi)
    r = L.q("""
        WITH s AS (
            SELECT symbol, ts_utc, open, high, low, close, volume,
                   (ts_utc AT TIME ZONE 'America/Los_Angeles') AS ts_pt
            FROM ironbeam_es_flow_1s
            WHERE ts_utc >= %s AND ts_utc < %s
              AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
              AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        )
        SELECT ts_pt::date AS d, symbol, date_trunc('minute', ts_pt) AS m_pt,
               (array_agg(open ORDER BY ts_utc))[1]::float AS o, max(high)::float AS h, min(low)::float AS l,
               (array_agg(close ORDER BY ts_utc DESC))[1]::float AS c, sum(volume)::float AS vol
        FROM s GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """, (L.pt(lo, 6, 30), L.pt(hi, 13, 0)))
    assert r["d"].min() >= lo and r["d"].max() <= hi
    return r


def ironbeam_bar_minutes() -> pd.DataFrame:
    lo, hi = L.PERIODS["y26"]
    L.guard(lo, hi)
    a, b = L.pt(lo, 6, 30).astimezone(L.UTC).replace(tzinfo=None), L.pt(hi, 13, 0).astimezone(L.UTC).replace(tzinfo=None)
    r = L.q("""
        SELECT ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::date AS d, 'IB' AS symbol,
               ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles') AS m_pt,
               open::float AS o, high::float AS h, low::float AS l, close::float AS c, volume::float AS vol
        FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime < %s
          AND ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
          AND ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        ORDER BY datetime
    """, (a, b))
    assert r["d"].min() >= lo and r["d"].max() <= hi
    return r


def g1down(W: pd.DataFrame) -> pd.DataFrame:
    m = C.group_mask(W, "G1", -1) & (~W["late"])
    return W[m]


def key(W: pd.DataFrame) -> set:
    return set(zip(W["d"], W["s"], W["e"], W["ci"]))


def with_bars(D: dict, es: pd.DataFrame) -> dict:
    sym, bars = L.front_bars(es)
    sess = [d for d in D["SESS"] if d in bars]
    px = {d: L.es_close_at_snapshot(bars, d, D["MIN"]) for d in sess}
    return dict(D, BARS=bars, SYM=sym, SESS=sess, PX=px)


def complete(D: dict) -> list:
    return [d for d in D["SESS"] if np.isfinite(D["BARS"][d]["h"]).all() and np.isfinite(D["BARS"][d]["l"]).all()]


def main():
    out = {"feed_heads": feed_heads()}
    D = C.load()
    ref, im_train, no_im = C.vol_scaled_legs(D)
    out["im_train"] = im_train
    out["sessions"] = len(D["SESS"])
    out["no_im"] = [str(d) for d in no_im]
    ref_g = g1down(ref)
    out["databento"] = dict(complete_sessions=len(complete(D)), legs=len(ref), primary=int((~ref["late"]).sum()),
                            g1down=len(ref_g))
    for name, es in (("ironbeam_flow", ironbeam_flow_minutes()), ("ironbeam_bars", ironbeam_bar_minutes())):
        Dx = with_bars(D, es)
        W, _, _ = C.vol_scaled_legs(Dx)
        g = g1down(W)
        both = set(Dx["SESS"]) & set(complete(Dx)) & set(complete(D))
        rk = {k for k in key(ref_g) if k[0] in both}
        xk = {k for k in key(g) if k[0] in both}
        sess_lost = sorted(set(complete(D)) - set(complete(Dx)))
        lost_g = sorted(str(k[0]) for k in key(ref_g) if k[0] in sess_lost)
        out[name] = dict(sessions_with_bars=len(Dx["SESS"]), complete_sessions=len(complete(Dx)),
                         sessions_incomplete_vs_databento=[str(d) for d in sess_lost],
                         legs=len(W), g1down=len(g),
                         on_common_sessions=dict(n_sessions=len(both), databento=len(rk), ironbeam=len(xk),
                                                 identical=len(rk & xk), only_databento=len(rk - xk),
                                                 only_ironbeam=len(xk - rk),
                                                 only_db=[list(map(str, k)) for k in sorted(rk - xk)],
                                                 only_ib=[list(map(str, k)) for k in sorted(xk - rk)]),
                         g1down_legs_on_lost_sessions=lost_g)
        allr = {k for k in key(ref) if k[0] in both}
        allx = {k for k in key(W) if k[0] in both}
        out[name]["all_legs_common"] = dict(databento=len(allr), ironbeam=len(allx), identical=len(allr & allx))
    (OUT / "step0.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
