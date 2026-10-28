"""CR-DI Step 0: dump /flow/cells payloads (read-only, one connection, no server) for the edge-effort series.

    apps/web/.venv/bin/python -u scripts/cr_di_step0/dump_cells.py OUT_DIR TRADE_DATE SESSION [UNTIL_PT ...]

UNTIL_PT is a naive PT wall clock ("2026-10-27T12:55"); "-" means no until (live / complete session).
"""
import datetime as dt
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
for p in (REPO, os.path.join(REPO, "apps", "web")):
    sys.path.insert(0, p)

import sqlalchemy  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.Ironbeam import flow_v2, flow_v3  # noqa: E402
from packages.shared import flow_heat as fh  # noqa: E402

out_dir, trade_date, session, *untils = sys.argv[1:]
os.makedirs(out_dir, exist_ok=True)
engine = sqlalchemy.create_engine(cb._get_db_url(), pool_size=1, max_overflow=0)
d = dt.date.fromisoformat(trade_date)
for u in untils or ["-"]:
    until = None if u == "-" else dt.datetime.fromisoformat(u).replace(tzinfo=flow_v2.PT_TZ)
    payload, status = flow_v3.build_cells_payload(engine, trade_date, session=session, until_pt=until,
                                                  params=fh.AbsorbParams(), session_date=d, session_note=None)
    tag = "live" if u == "-" else u.replace(":", "")
    path = os.path.join(out_dir, f"cells_{trade_date}_{session}_{tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print(status, path, [(s["name"], len(s["bars"]["t"]), len(s["marks"] or [])) for s in payload["segments"]])
engine.dispose()
