"""CR-CX Step 0b (Ryan's ruling 1) — price one session's Databento pull and check availability timing. No data pulled.

Metadata calls only (`metadata.get_cost`, `get_billable_size`, `get_dataset_range`): no trades are downloaded and
nothing is written, so no holdout value is read. Priced: the trades the frozen code needs for one session, i.e. the
CR-CA window [06:15, 13:00) PT for every contract CR-CA stores that day (front, plus the dying contract in roll
weeks). Sessions priced: 2026-10-26 and 2026-10-23 (normal weeks, front only) and 2026-09-15 (ESU6 → ESZ6 roll
week: two contracts, the most a session costs).

Ruling 1 gate: STOP if cost > $1 per session or the data is not available by 10:30 UTC the next morning.

Run: apps/web/.venv/bin/python scripts/cr_cx_step0b.py  ->  scripts/logs/cr_cx/step0b.json
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_ca_step0 import DATASET, client, contracts_for, raw_symbol, window_utc  # noqa: E402

OUT = Path(__file__).resolve().parent / "logs" / "cr_cx"
OUT.mkdir(parents=True, exist_ok=True)
CAP_USD = 1.0


def price(cl, d: date) -> dict:
    s, e = window_utc(d)
    parts = []
    for c in contracts_for(d):
        sym = raw_symbol(*c)
        kw = dict(dataset=DATASET, schema="trades", symbols=[sym], stype_in="raw_symbol",
                  start=s.isoformat(), end=e.isoformat())
        parts.append(dict(raw_symbol=sym, usd=float(cl.metadata.get_cost(**kw)),
                          bytes=int(cl.metadata.get_billable_size(**kw))))
    return dict(session=str(d), window_utc=[s.isoformat(), e.isoformat()], contracts=parts,
                usd=sum(p["usd"] for p in parts))


def main():
    cl = client()
    now = datetime.now(timezone.utc)
    rng = cl.metadata.get_dataset_range(dataset=DATASET)
    out = dict(checked_at_utc=now.isoformat(), dataset=DATASET, dataset_range=rng)
    out["sessions"] = [price(cl, date(2026, 10, 26))]
    out["sessions"].append(price(cl, date(2026, 10, 23)))
    out["sessions"].append(price(cl, date(2026, 9, 15)))
    out["max_usd"] = max(s["usd"] for s in out["sessions"])
    out["cost_gate"] = out["max_usd"] <= CAP_USD
    (OUT / "step0b.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
