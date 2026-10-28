"""Record the CR-DC /api/ironbeam/flow/cells fixture (live DB, backfill role, read-only).

    apps/web/.venv/bin/python -m apps.web.modules.Ironbeam.tests.record_flow_cells_fixture

Writes fixtures/cr_dc_flow_cells.pkl.gz (every DB read made while serving flow_cells_fixture.QUERIES)
and fixtures/cr_dc_flow_cells_golden_0501.json, and checks live bytes == replay bytes for every query.
2026-10-27 is served from `ironbeam_es_trades`, so it records only while that session is retained (2 days).
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

from apps.web.modules.Ironbeam.tests import flow_cells_fixture as cf  # noqa: E402

ROUTE = "/api/ironbeam/flow/cells?"


def main() -> None:
    client = cf.make_client()
    store: dict = {}
    live = {}
    for name, qs in cf.QUERIES.items():
        with cf.record(store):
            r = client.get(ROUTE + qs)
        assert r.status_code == 200, (name, r.status_code, r.data[:300])
        live[name] = r.data
    cf.save(store, cf.FIXTURE)
    for name, qs in cf.QUERIES.items():
        with cf.replay(store):
            r = client.get(ROUTE + qs)
        assert r.data == live[name], f"replay != live for {name}"
    (cf.FIXTURE_DIR / "cr_dc_flow_cells_golden_0501.json").write_bytes(live["0501"])
    print(f"{len(store['calls'])} calls -> {cf.FIXTURE} ({cf.FIXTURE.stat().st_size} B); replay == live")


if __name__ == "__main__":
    main()
