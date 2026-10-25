"""Record the CR-CC /api/ironbeam/flow fixture (live DB, SELECTs only).

    apps/web/.venv/bin/python -m apps.web.modules.Ironbeam.tests.record_flow_fixture [--golden]

Writes fixtures/cr_cc_flow_2026-10-23.pkl.gz: every DB-reader call made while
serving flow_fixture.QUERIES. --golden also writes the replayed response bodies to
fixtures/cr_cc_flow_golden_{full,rth}.json. Run --golden only on the callbacks.py
that predates CR-CC's `until` (origin/main 59e2fa8). Either way the script checks
live bytes == replay bytes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from apps.web.modules.Ironbeam.tests import flow_fixture as ff  # noqa: E402

FIXTURE = ff.FIXTURE_DIR / "cr_cc_flow_2026-10-23.pkl.gz"
ROUTE = "/api/ironbeam/flow?"
# Extra windows the CR-CC `until` tests replay (truncated windows read the DB with other args).
UNTIL_QUERIES = {
    "full_until": ff.FULL_QS + "&until=2026-10-23T07:37:00",
    "rth_until": ff.RTH_QS + "&until=2026-10-23T07:37:00",
}


def main() -> None:
    golden = "--golden" in sys.argv
    client = ff.make_client()
    store: dict = {}
    queries = dict(ff.QUERIES)
    if not golden:
        queries.update(UNTIL_QUERIES)

    live = {}
    with ff.record(store):
        for name, qs in queries.items():
            r = client.get(ROUTE + qs)
            assert r.status_code == 200, (name, r.status_code)
            live[name] = r.data
    ff.save(store, FIXTURE)
    print(f"recorded {len(store['calls'])} calls → {FIXTURE}")

    with ff.replay(ff.load(FIXTURE)):
        for name, qs in queries.items():
            body = client.get(ROUTE + qs).data
            assert body == live[name], f"{name}: replay differs from live"
            print(f"{name}: live == replay ({len(body):,} bytes)")
            if golden:
                Path(ff.FIXTURE_DIR / f"cr_cc_flow_golden_{name}.json").write_bytes(body)
    if golden:
        print("golden files written")


if __name__ == "__main__":
    main()
