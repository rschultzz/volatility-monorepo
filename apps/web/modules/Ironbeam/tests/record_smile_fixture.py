"""Record the CR-CC A2 smile/skew fixture (live DB, SELECTs only).

    apps/web/.venv/bin/python -m apps.web.modules.Ironbeam.tests.record_smile_fixture [--golden]

Writes fixtures/cr_cc_smile_2026-10-23.pkl.gz: every ORATS-reader call made while serving
smile_fixture.QUERIES (plus, without --golden, the `until` requests the A2 tests replay).
--golden also writes the replayed bodies to fixtures/cr_cc_smile_golden_<name>.json. Run
--golden only on the callbacks.py that predates A2 (origin/main f11b1de). Either way the
script checks live bytes == replay bytes.
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

from apps.web.modules.Ironbeam.tests import smile_fixture as sf  # noqa: E402

FIXTURE = sf.FIXTURE_DIR / "cr_cc_smile_2026-10-23.pkl.gz"
UNTIL = "2026-10-23T08:00:00"
UNTIL_QUERIES = {
    "smile_until": sf.QUERIES["smile"] + "&until=" + UNTIL,
    "skew_until": sf.QUERIES["skew"] + "&until=" + UNTIL,
    "skew_until_noexp": sf.QUERIES["skew_noexp"] + "&until=" + UNTIL,
    # nothing selected / all selected times after the cutoff -> latest snapshot <= until
    "smile_until_default": "/api/ironbeam/smile-data?trade_date=2026-10-23&expiration_date=2026-10-23&until=" + UNTIL,
    "skew_until_late": "/api/ironbeam/skew-data?trade_date=2026-10-23&expiration_date=2026-10-23&times=10:00,11:00&until=" + UNTIL,
    # cutoff before the first snapshot of the day (ETH entry) -> no snapshot
    "smile_until_eth": "/api/ironbeam/smile-data?trade_date=2026-10-23&expiration_date=2026-10-23&times=07:00&until=2026-10-23T05:10:00",
}


def main() -> None:
    golden = "--golden" in sys.argv
    client = sf.make_client()
    store: dict = {}
    queries = dict(sf.QUERIES)
    if not golden:
        queries.update(UNTIL_QUERIES)
    live = {}
    with sf.record(store):
        for name, url in queries.items():
            r = client.get(url)
            assert r.status_code == 200, (name, r.status_code)
            live[name] = r.data
    sf.save(store, FIXTURE)
    print(f"recorded {len(store['calls'])} calls → {FIXTURE}")
    with sf.replay(sf.load(FIXTURE)):
        for name, url in queries.items():
            body = client.get(url).data
            assert body == live[name], f"{name}: replay differs from live"
            print(f"{name}: live == replay ({len(body):,} bytes)")
            if golden:
                Path(sf.FIXTURE_DIR / f"cr_cc_smile_golden_{name}.json").write_bytes(body)
    if golden:
        print("golden files written")


if __name__ == "__main__":
    main()
