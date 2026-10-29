#!/usr/bin/env python3
"""CR-DJ I1 — fetch Databento ESM6 1-minute bars for the March 2026 roll patch.

Two ranges, one raw symbol (ESM6, stype raw_symbol), GLBX.MDP3 ohlcv-1m:
  - patch window   2026-03-15 22:00 -> 2026-03-20 21:00 UTC (6,900 Globex minutes)
  - jump check     2026-03-23 10:35 -> 11:36 UTC (±30 min of the 11:05 Ironbeam jump; not patched)

Cost gate: the summed metadata cost must be < $0.05 (Amendment 1 D3), else exit with nothing
fetched. Output: scripts/logs/cr_dj/databento_esm6_{window,jump}.parquet (git-ignored; the
patch script reads the window file, so the billed fetch runs once).

Usage:
    apps/web/.venv/bin/python scripts/cr_dj_fetch_databento.py [--cost-only]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import databento as db  # noqa: E402

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOL = "ESM6"
STYPE_IN = "raw_symbol"
MAX_COST = 0.05
RANGES = {
    "window": ("2026-03-15T22:00:00Z", "2026-03-20T21:00:00Z"),
    "jump": ("2026-03-23T10:35:00Z", "2026-03-23T11:36:00Z"),
}
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_dj"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-only", action="store_true", help="print the cost and stop (free)")
    args = ap.parse_args()

    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    kw = dict(dataset=DATASET, schema=SCHEMA, symbols=[SYMBOL], stype_in=STYPE_IN)
    costs = {name: client.metadata.get_cost(start=a, end=b, **kw) for name, (a, b) in RANGES.items()}
    total = sum(costs.values())
    print(f"[CR-DJ] cost {', '.join(f'{k} ${v:.4f}' for k, v in costs.items())}; total ${total:.4f} (gate ${MAX_COST})")
    if total >= MAX_COST:
        print("[CR-DJ] cost gate failed; nothing fetched.")
        return 1
    if args.cost_only:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (a, b) in RANGES.items():
        df = client.timeseries.get_range(start=a, end=b, **kw).to_df()
        if df.empty:
            print(f"[CR-DJ] {name}: 0 bars; stopping.")
            return 1
        df = df.reset_index()[["ts_event", "symbol", "open", "high", "low", "close", "volume"]]
        path = OUT_DIR / f"databento_esm6_{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"[CR-DJ] {name}: {len(df)} bars {df.ts_event.min()} -> {df.ts_event.max()}, "
              f"symbols {sorted(df.symbol.unique())}, volume {int(df.volume.sum()):,} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
