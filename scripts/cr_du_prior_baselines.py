"""CR-DU (Step 0 Q1 fix 3, Q5) — prior-sessions-only baselines for the v1 engine on train dates. Read-only DB access.

For every train trading day d (2026-01-02 → 2026-06-05), from the CR-DQ baseline sessions strictly before d (FOMC
decision days and the March 2026 expiry week left out, as `scripts/cr_dq_step0.py`), when there are ≥ 20 of them:
  - the box panel baseline in `boxBaseline.json`'s shape (|Δ share| by length bucket × time of day, contracts/pt by
    length bucket; 101-point grids), Databento only (B5: train dates are scored on Databento);
  - contradiction-bar percentiles (item 4) per 30-min RTH bucket: minute signed delta p10 / p20 / p80 / p90 and minute
    volume p80, Databento.
Plus `forward`: the same contradiction percentiles from all train baseline sessions on Ironbeam, for forward / live
dates (their box baseline is the fixed `boxBaseline.json`). No outcome is read.

Writes apps/web/modules/Ironbeam/data/cr_du_prior_baselines.json (committed; served by the v1-engine inputs route).
Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_prior_baselines.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_dq_step0 as s0  # noqa: E402  (loads .env, read-only PGOPTIONS)

import psycopg  # noqa: E402

from packages.shared.backfill_safety import connect_redacted  # noqa: E402

from packages.shared.trading_calendar import is_trading_day  # noqa: E402

OUT = os.path.join(s0.REPO, "apps", "web", "modules", "Ironbeam", "data", "cr_du_prior_baselines.json")
WARMUP = 20
NB30 = 13


def grids(win):
    A, L, S, C = (np.concatenate(x) for x in zip(*[(w[0], w[1], w[2], w[3]) for w in win]))
    share, cpp = {}, {}
    for name, lo, hi in s0.BUCKETS:
        m = (L >= lo) & (L <= hi)
        cpp[name] = [round(float(v)) for v in np.percentile(C[m], np.arange(101))]
        for tname, t0, t1 in s0.TOD:
            mt = m & (A >= t0) & (A < t1)
            share.setdefault(tname.split()[0], {})[name] = [round(float(v), 6) for v in np.percentile(S[mt], np.arange(101))]
    return share, cpp


def contra(arrs):
    out = []
    for b in range(NB30):
        dl = np.concatenate([a[2, b * 30:(b + 1) * 30] for a in arrs])
        v = np.concatenate([a[3, b * 30:(b + 1) * 30] for a in arrs])
        ok = ~np.isnan(dl)
        d10, d20, d80, d90 = (float(x) for x in np.percentile(dl[ok], [10, 20, 80, 90]))
        out.append({"from": b * 30, "d10": round(d10, 1), "d20": round(d20, 1), "d80": round(d80, 1),
                    "d90": round(d90, 1), "v80": round(float(np.percentile(v[ok], 80)), 1)})
    return out


def main():
    days = s0.sessions()
    try:
        with connect_redacted(os.environ["BACKFILL_DATABASE_URL"], connect_timeout=20) as conn:
            db = s0.load(conn, "databento", days)
            ib = s0.load(conn, "ironbeam", days)
    except Exception as e:  # never print the connection string
        print("DB error:", type(e).__name__)
        sys.exit(1)
    win = {d: s0.windows(arr) for d, (sym, arr) in db.items()}
    base_days = sorted(d for d in db if win[d] is not None)
    out = {"meta": {
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "script": "scripts/cr_du_prior_baselines.py",
        "train": [str(s0.FIRST), str(s0.LAST)], "warmup": WARMUP,
        "excluded": sorted(str(d) for d in s0.EXCLUDE if s0.FIRST <= d <= s0.LAST),
        "buckets": [{"name": n, "min": lo, "max": hi} for n, lo, hi in s0.BUCKETS],
        "tod": [{"name": n.split()[0], "fromMin": a, "toMin": b} for n, a, b in s0.TOD],
        "tick": s0.TICK, "maxL": s0.MAX_L,
    }, "sessions": {}, "warmup_sessions": []}
    d = s0.FIRST
    while d <= s0.LAST:
        if is_trading_day(d):
            prior = [x for x in base_days if x < d]
            if len(prior) < WARMUP:
                out["warmup_sessions"].append(str(d))
            else:
                share, cpp = grids([win[x] for x in prior])
                out["sessions"][str(d)] = {
                    "databento": {"sessions": len(prior), "first": str(prior[0]), "last": str(prior[-1]),
                                  "absShare": share, "cpp": cpp},
                    "contra": {"databento": contra([db[x][1] for x in prior])},
                }
        d += dt.timedelta(days=1)
    out["forward"] = {"contra": {"ironbeam": contra([a for _, a in ib.values()])}, "ironbeam_sessions": len(ib)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    sc = sorted(out["sessions"])
    print(f"wrote {os.path.relpath(OUT, s0.REPO)} ({os.path.getsize(OUT)} bytes): {len(sc)} sessions {sc[0]} → {sc[-1]}, "
          f"warm-up {len(out['warmup_sessions'])} ({out['warmup_sessions'][0]} → {out['warmup_sessions'][-1]}), "
          f"forward contra from {len(ib)} Ironbeam sessions")


if __name__ == "__main__":
    main()
