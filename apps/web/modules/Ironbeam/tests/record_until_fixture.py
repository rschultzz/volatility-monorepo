"""Record the CR-CC `until` replay fixture for /api/ironbeam/bars (live DB, SELECTs only).

    apps/web/.venv/bin/python -m apps.web.modules.Ironbeam.tests.record_until_fixture

Writes fixtures/cr_cc_bars_until_2026-10-23.pkl.gz: every DB-reader call made while
serving UNTIL_QUERIES (and the no-cutoff CENTER_QS / MULTI_QS, for comparisons), and
checks live bytes == replay bytes. The flow `until` calls are recorded by
record_flow_fixture (without --golden).
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

from apps.web.modules.Ironbeam.tests import bars_fixture as bf  # noqa: E402

FIXTURE = bf.FIXTURE_DIR / "cr_cc_bars_until_2026-10-23.pkl.gz"
ROUTE = "/api/ironbeam/bars?"
UNTIL_RTH = "2026-10-23T07:37:00"
UNTIL_ETH = "2026-10-23T05:10:00"
UNTIL_QUERIES = {
    "center": bf.CENTER_QS,
    "multi": bf.MULTI_QS,
    "center_until": bf.CENTER_QS + "&until=" + UNTIL_RTH,
    "multi_until": bf.MULTI_QS + "&until=" + UNTIL_RTH,
    "center_until_corrected": bf.CENTER_QS + "&levels=corrected&until=" + UNTIL_RTH,
    "center_until_eth": bf.CENTER_QS + "&until=" + UNTIL_ETH,
    "center_until_5min": bf.CENTER_QS.replace("interval=1min", "interval=5min") + "&until=" + UNTIL_RTH,
}


def main() -> None:
    client = bf.make_client()
    store: dict = {}
    live = {}
    with bf.record(store):
        for name, qs in UNTIL_QUERIES.items():
            r = client.get(ROUTE + qs)
            assert r.status_code == 200, (name, r.status_code)
            live[name] = r.data
    bf.save(store, FIXTURE)
    print(f"recorded {len(store['calls'])} calls → {FIXTURE}")
    with bf.replay(bf.load(FIXTURE)):
        for name, qs in UNTIL_QUERIES.items():
            body = client.get(ROUTE + qs).data
            assert body == live[name], f"{name}: replay differs from live"
            print(f"{name}: live == replay ({len(body):,} bytes)")


if __name__ == "__main__":
    main()
