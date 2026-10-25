"""CR-CD A4.1 / A4.5 — identity of the stored 1-minute tables with Flow v2's on-the-fly aggregation,
and the L9 cold-load check. READ-ONLY (backfill role, default_transaction_read_only=on).

    apps/web/.venv/bin/python scripts/cr_cd_a4_identity.py identity
    apps/web/.venv/bin/python scripts/cr_cd_a4_identity.py l9

identity: 3 sessions per source (one a two-contract roll-week session each). For each, the windows
Flow v2 reads (ETH for Ironbeam, RTH with warm-up, a ragged mid-minute `until` window, and the
20-session baseline ranges) through `fetch_flow_minutes` with the 1-minute tables vs forced on the fly
(cutoff None): frames must be equal exactly. Then the whole v2 payload both ways: JSON must be equal.
l9: 5 random train sessions (seed 20261025, 2026-01-02 .. 2026-06-05, NYSE): per session a fresh DB
connection (engine disposed) and every in-process cache cleared, then one timed /api/ironbeam/flow
?mode=v2&session=FULL request through the Flask route. Postgres' own buffer cache cannot be cleared
from here; the first request per session is still the first read of that session's 1-minute rows.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
for line in (REPO / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
for p in (str(REPO), str(REPO / "apps" / "web")):
    sys.path.insert(0, p)

import pandas as pd  # noqa: E402

import dash  # noqa: E402
import modules.Ironbeam.callbacks as cb  # noqa: E402
import modules.Ironbeam.flow_v2 as v2  # noqa: E402
from packages.shared import flow_features as ff  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

UTC = dt.timezone.utc
SESSIONS = {
    "databento": [dt.date(2024, 6, 12), dt.date(2025, 3, 18), dt.date(2026, 5, 1)],   # 2025-03-18: roll week
    "ironbeam": [dt.date(2026, 1, 14), dt.date(2026, 3, 17), dt.date(2026, 10, 23)],  # 2026-03-17: two contracts
}


def clear():
    v2._CACHE.clear()
    v2._WATERMARK.clear()
    v2._TABLE_STATE.clear()


def windows(source, d):
    rth_a = v2._pt(d, 6, 30).astimezone(UTC)
    out = {"rth_warm": [((v2._pt(d, 6, 30) - dt.timedelta(minutes=5)).astimezone(UTC), v2._pt(d, 13, 0).astimezone(UTC))],
           "until_ragged": [(rth_a, v2._pt(d, 7, 39).astimezone(UTC) + dt.timedelta(seconds=17))]}
    if source == "ironbeam":
        eth_a = dt.datetime.combine(d - dt.timedelta(days=1), dt.time(15, 0), tzinfo=v2.PT_TZ).astimezone(UTC)
        out["eth"] = [(eth_a, rth_a)]
    prior = [x for x in v2._prior_sessions(d, source)][:v2.BASELINE_CHUNK]
    out["baseline_chunk"] = [(v2._pt(x, 6, 30).astimezone(UTC), v2._pt(x, 13, 0).astimezone(UTC)) for x in prior]
    return out


def identity():
    table = {"databento": v2.DATABENTO_TABLE, "ironbeam": v2.IRONBEAM_TABLE}
    report = []
    for source, days in SESSIONS.items():
        for d in days:
            for name, ranges in windows(source, d).items():
                clear()
                via_1m = v2.fetch_flow_minutes(cb.engine, table[source], ranges)
                cut = v2._cutoff_1m(cb.engine, v2.flow_1m.TABLE_1M[table[source]], dt.datetime.now(UTC))
                r1m, r1s = v2.split_ranges(ranges, cut)
                with patch.object(v2, "_cutoff_1m", return_value=None):
                    on_fly = v2.fetch_flow_minutes(cb.engine, table[source], ranges)
                pd.testing.assert_frame_equal(via_1m, on_fly, check_exact=True)
                report.append((source, str(d), name, len(via_1m), len(r1m), len(r1s), "equal"))
            for session in ("FULL", "RTH"):
                clear()
                kw = dict(session=session, until_pt=None, params=ff.MarkerParams(), session_date=d, session_note=None)
                p1, _ = v2.build_flow_v2_payload(cb.engine, str(d), **kw)
                clear()
                with patch.object(v2, "_cutoff_1m", return_value=None):
                    p0, _ = v2.build_flow_v2_payload(cb.engine, str(d), **kw)
                assert json.dumps(p1, sort_keys=True) == json.dumps(p0, sort_keys=True), (source, d, session)
                report.append((source, str(d), f"payload {session}", len(p1["bars"]), len(p1["markers"]), "", "equal"))
    for row in report:
        print(*row, sep="\t")
    print("IDENTITY PASS:", len(report), "comparisons")


def l9():
    days = [d for d in (dt.date(2026, 1, 2) + dt.timedelta(n) for n in range(155))
            if d <= dt.date(2026, 6, 5) and is_trading_day(d)]
    pick = sorted(random.Random(20261025).sample(days, 5))
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    cb.register_ironbeam_callbacks(app)
    client = app.server.test_client()
    # one warm-up request on a session outside the sample (Python first-call costs, not DB coldness)
    client.get("/api/ironbeam/flow?trade_date=2026-01-05&session=FULL&mode=v2")
    rows = []
    for d in pick:
        clear()
        cb.engine.dispose()
        t = time.perf_counter()
        cb.engine.connect().close()            # the fresh connection (TCP + TLS + auth), returned to the pool
        conn_s = time.perf_counter() - t
        t = time.perf_counter()
        r = client.get(f"/api/ironbeam/flow?trade_date={d}&session=FULL&mode=v2")
        req_s = time.perf_counter() - t
        p = r.get_json()
        rows.append((conn_s, req_s))
        print(f"{d}\tstatus {r.status_code}\tconnect {conn_s:.2f} s + request {req_s:.2f} s = {conn_s + req_s:.2f} s"
              f"\tsource {p.get('source')}\tbars {len(p.get('bars', []))}\tmarkers {len(p.get('markers', []))}"
              f"\tbaseline {(p.get('baseline') or {}).get('sessions')}")
    tot = [a + b for a, b in rows]
    print(f"cold total max {max(tot):.2f} s, median {sorted(tot)[2]:.2f} s; request only max {max(b for _, b in rows):.2f} s; "
          f"L9 (<= 1.5 s cold total): {'PASS' if max(tot) <= 1.5 else 'FAIL'}")


if __name__ == "__main__":
    {"identity": identity, "l9": l9}[sys.argv[1]]()
