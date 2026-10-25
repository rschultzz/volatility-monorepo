"""Record the CR-CD /api/ironbeam/flow?mode=v2 fixture (live DB, backfill role, read-only).

    apps/web/.venv/bin/python -m apps.web.modules.Ironbeam.tests.record_flow_v2_fixture

Writes fixtures/cr_cd_flow_v2.pkl.gz (every DB read made while serving flow_v2_fixture.QUERIES)
and fixtures/cr_cd_flow_v2_golden_0501.json (the replayed 2026-05-01 FULL response), and checks
live bytes == replay bytes for every query.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"

from apps.web.modules.Ironbeam.tests import flow_v2_fixture as vf  # noqa: E402

ROUTE = "/api/ironbeam/flow?"


def main() -> None:
    client = vf.make_client()
    store: dict = {}
    live = {}
    for name, qs in vf.QUERIES.items():
        with vf.record(store):
            r = client.get(ROUTE + qs)
        assert r.status_code == 200, (name, r.status_code, r.data[:300])
        live[name] = r.data
    vf.save(store, vf.FIXTURE)
    for name, qs in vf.QUERIES.items():
        with vf.replay(store):
            r = client.get(ROUTE + qs)
        assert r.data == live[name], f"replay != live for {name}"
    (vf.FIXTURE_DIR / "cr_cd_flow_v2_golden_0501.json").write_bytes(live["0501"])
    print(f"{len(store['calls'])} calls -> {vf.FIXTURE} ({vf.FIXTURE.stat().st_size} B); replay == live")


if __name__ == "__main__":
    main()
