"""CR-DQ item 5 — the box panel's percentile baseline. Writes react_price_preview/src/data/boxBaseline.json (committed;
the frontend imports it). Refreshed only by re-running this script. Read-only DB access.

Windows, sessions, exclusions and contract choice are Step 0's (`scripts/cr_dq_step0.py`, spec S0.4): train sessions
2026-01-02 → 2026-06-05 only (B4), FOMC decision days (A22) and the March 2026 expiry week left out; every RTH window
(anchor a, length L = 3 … 240 min) with all minutes present; |Δ share| = |Σ delta| ÷ Σ volume over a+1 … a+L;
contracts/pt = Σ volume ÷ max(|close[a+L] − close[a]|, 0.25).

Per source (Ryan, Step 0 decision 2: same-source baselines, no adjustment) and length bucket, a 101-point quantile grid
(p0 … p100) of:
  - |Δ share|, split by the window's start time (decision 1): open 06:30–07:30, mid 07:30–11:00, late 11:00–13:00 PT;
  - contracts/pt, all day.
The panel interpolates linearly in the grid (as numpy.interp), so a hand check against the JSON matches.

Usage: apps/web/.venv/bin/python scripts/cr_dq_baseline.py [--out react_price_preview/src/data/boxBaseline.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr_dq_step0 as s0  # noqa: E402  (loads .env, read-only PGOPTIONS)

import psycopg  # noqa: E402

OUT = os.path.join(s0.REPO, "react_price_preview", "src", "data", "boxBaseline.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    days = s0.sessions()
    try:
        with psycopg.connect(s0.review_routes._normalize_db_url(os.environ["BACKFILL_DATABASE_URL"]), connect_timeout=20) as conn:
            data = {src: s0.load(conn, src, days) for src in s0.TABLES}
    except Exception as e:  # never print the connection string
        print("DB error:", type(e).__name__)
        sys.exit(1)

    out = {
        "meta": {
            "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "script": "scripts/cr_dq_baseline.py",
            "train": [str(s0.FIRST), str(s0.LAST)],
            "excluded": sorted(str(d) for d in s0.EXCLUDE if s0.FIRST <= d <= s0.LAST),
            "buckets": [{"name": n, "min": lo, "max": hi} for n, lo, hi in s0.BUCKETS],
            "tod": [{"name": n.split()[0], "fromMin": a, "toMin": b} for n, a, b in s0.TOD],  # minutes after 06:30 PT
            "tick": s0.TICK,
            "maxL": s0.MAX_L,
        },
    }
    for src, per in data.items():
        A, L, S, C = [], [], [], []
        for d, (sym, arr) in sorted(per.items()):
            w = s0.windows(arr)
            if w is None:
                continue
            A.append(w[0])
            L.append(w[1])
            S.append(w[2])
            C.append(w[3])
        A, L, S, C = (np.concatenate(x) for x in (A, L, S, C))
        share = {}
        cpp = {}
        counts = {}
        for name, lo, hi in s0.BUCKETS:
            m = (L >= lo) & (L <= hi)
            cpp[name] = [round(float(v)) for v in np.percentile(C[m], np.arange(101))]
            counts[name] = int(m.sum())
            for tname, t0, t1 in s0.TOD:
                key = tname.split()[0]
                mt = m & (A >= t0) & (A < t1)
                share.setdefault(key, {})[name] = [round(float(v), 6) for v in np.percentile(S[mt], np.arange(101))]
        out[src] = {
            "sessions": len(per), "first": str(min(per)), "last": str(max(per)),
            "windows": counts, "absShare": share, "cpp": cpp,
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {os.path.relpath(args.out, s0.REPO)} ({os.path.getsize(args.out)} bytes):",
          {src: {k: out[src][k] for k in ('sessions', 'first', 'last')} for src in s0.TABLES})


if __name__ == "__main__":
    main()
