"""Record the /api/gex-landscape fixture (and, with --golden, the Current-mode golden bodies).

Read-only: connects with BACKFILL_DATABASE_URL and default_transaction_read_only=on.

    apps/web/.venv/bin/python apps/web/modules/Ironbeam/tests/record_landscape_fixture.py --golden
        (run once on the unmodified route/builder: records GOLDEN_QS and writes the golden bodies)
    apps/web/.venv/bin/python apps/web/modules/Ironbeam/tests/record_landscape_fixture.py
        (re-records GOLDEN_QS + CORRECTED_QS into the fixture; never rewrites the golden)

Both modes check that the live body equals the replayed body for every request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import landscape_fixture as lf  # noqa: E402

for line in (lf.REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from sqlalchemy import create_engine  # noqa: E402


def _engine():
    url = os.environ["BACKFILL_DATABASE_URL"]
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, connect_args={"options": "-c default_transaction_read_only=on"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", action="store_true")
    args = ap.parse_args()

    eng = _engine()
    with eng.connect() as c:
        assert c.exec_driver_sql("SHOW default_transaction_read_only").scalar() == "on"

    rec = lf.RecordingEngine(eng)
    live = lf.make_client(rec)
    requests = dict(lf.GOLDEN_QS)
    if not args.golden:
        requests.update(lf.CORRECTED_QS)
    bodies = {}
    with lf.frozen_clock():
        for name, qs in requests.items():
            r = lf.get(live, qs)
            assert r.status_code == 200, (name, r.status_code, r.data[:200])
            bodies[name] = r.data
    lf.save_store(rec.store)

    replay = lf.make_client(lf.ReplayEngine(lf.load_store()))
    with lf.frozen_clock():
        for name, qs in requests.items():
            assert lf.get(replay, qs).data == bodies[name], f"replay != live for {name}"
            print(f"{name:14s} {len(bodies[name]):>8,d} B  sha256 {hashlib.sha256(bodies[name]).hexdigest()[:16]}")

    if args.golden:
        lf.GOLDEN_PATH.write_text(json.dumps({k: v.decode() for k, v in bodies.items()}, indent=0))
        print(f"golden written: {lf.GOLDEN_PATH}")
    else:
        golden = lf.load_golden()
        for name in lf.GOLDEN_QS:
            assert bodies[name].decode() == golden[name], f"live Current body != golden for {name}"
        print("live Current bodies == golden: OK")
    print(f"fixture: {lf.FIXTURE_PATH} ({len(rec.store)} recorded queries)")


if __name__ == "__main__":
    main()
