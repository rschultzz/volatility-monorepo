"""Record the CR-BT /api/ironbeam/bars fixture (live DB, SELECTs only).

    apps/web/.venv/bin/python -m apps.web.modules.Ironbeam.tests.record_bars_fixture [--golden]

Writes fixtures/cr_bt_bars_2026-10-23.pkl.gz: every DB-reader call made while
serving CENTER_QS, MULTI_QS and their `levels=corrected` variants.

--golden additionally writes the raw response bodies of CENTER_QS / MULTI_QS
under replay to fixtures/cr_bt_bars_golden_{center,multi}.json. Run --golden only
on the unmodified callbacks.py (origin/main e78774f) — those files are the L7
reference. Either way the script checks live bytes == replay bytes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
# Load .env before bars_fixture, which defaults DATABASE_URL to a dummy for replay.
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from apps.web.modules.Ironbeam.tests import bars_fixture as bf  # noqa: E402

FIXTURE = bf.FIXTURE_DIR / "cr_bt_bars_2026-10-23.pkl.gz"
ROUTE = "/api/ironbeam/bars?"


def main() -> None:
    golden = "--golden" in sys.argv
    client = bf.make_client()
    store: dict = {}
    queries = {"center": bf.CENTER_QS, "multi": bf.MULTI_QS}
    if not golden:
        queries.update({k + "_corrected": v + "&levels=corrected" for k, v in list(queries.items())})

    live = {}
    with bf.record(store):
        for name, qs in queries.items():
            r = client.get(ROUTE + qs)
            assert r.status_code == 200, (name, r.status_code)
            live[name] = r.data
    bf.save(store, FIXTURE)
    print(f"recorded {len(store['calls'])} calls → {FIXTURE}")

    with bf.replay(bf.load(FIXTURE)):
        for name, qs in queries.items():
            body = client.get(ROUTE + qs).data
            assert body == live[name], f"{name}: replay differs from live"
            print(f"{name}: live == replay ({len(body):,} bytes)")
            if golden:
                Path(bf.FIXTURE_DIR / f"cr_bt_bars_golden_{name}.json").write_bytes(body)
    if golden:
        print("golden files written")


if __name__ == "__main__":
    main()
