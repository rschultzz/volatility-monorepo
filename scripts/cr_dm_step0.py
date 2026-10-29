"""CR-DM Step 0 (read-only): the volume field Flow v3 draws, unknown-side volume, and the effort-legs baseline.

    apps/web/.venv/bin/python -u scripts/cr_dm_step0.py

Q1 — per session: Σ volume vs Σ (buy + sell) vs Σ unknown_vol in the 1-minute tables the v2 payload reads
     (RTH 06:30-13:00 PT), and minutes where volume != buy + sell + unknown.
Q3 — the v2 payload's confirmed legs (effort=1) for 2026-10-27 and 2026-10-29 (until 07:30 PT, so the file is
     stable while the session runs) and a few replay cut-offs, written to scripts/logs/cr-dm/step0/ as the baseline
     the implementation is compared against.

Backfill role, default_transaction_read_only=on. DB errors print the error type only.
"""
import datetime as dt
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from sqlalchemy import text  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.Ironbeam import flow_v2  # noqa: E402
from packages.shared import flow_features as ff  # noqa: E402

OUT = os.environ.get("CRDM_OUT") or os.path.join(REPO, "scripts", "logs", "cr-dm", "step0")
PT = ff.PT

Q1_SQL = """
    SELECT count(*) AS minutes, sum(volume) AS vol, sum(buy_vol + sell_vol) AS bs, sum(unknown_vol) AS unk,
           sum(CASE WHEN volume <> buy_vol + sell_vol + unknown_vol THEN 1 ELSE 0 END) AS mismatch,
           sum(CASE WHEN unknown_vol > 0 THEN 1 ELSE 0 END) AS unk_minutes
    FROM {table} WHERE minute_utc >= :a AND minute_utc < :b {sym}"""


def rth(d):
    return (dt.datetime.combine(d, dt.time(6, 30), tzinfo=PT).astimezone(dt.timezone.utc),
            dt.datetime.combine(d, dt.time(13, 0), tzinfo=PT).astimezone(dt.timezone.utc))


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        engine = sqlalchemy.create_engine(cb._get_db_url(), pool_size=2, max_overflow=0)
        with engine.connect() as c:
            print("Q1 — volume vs buy + sell + unknown, RTH, 1-minute tables")
            for table, d, sym in [("es_flow_1m_databento", dt.date(2026, 5, 1), "AND raw_symbol = 'ESM6'"),
                                  ("es_flow_1m_databento", dt.date(2026, 1, 6), "AND raw_symbol = 'ESH6'"),
                                  ("ironbeam_es_flow_1m", dt.date(2026, 10, 27), ""),
                                  ("ironbeam_es_flow_1m", dt.date(2026, 10, 28), ""),
                                  ("ironbeam_es_flow_1m", dt.date(2026, 10, 29), "")]:
                a, b = rth(d)
                r = c.execute(text(Q1_SQL.format(table=table, sym=sym)), {"a": a, "b": b}).mappings().one()
                vol, bs, unk = float(r["vol"] or 0), float(r["bs"] or 0), float(r["unk"] or 0)
                print(f"  {table} {d}: minutes {r['minutes']}, volume {vol:,.0f}, buy+sell {bs:,.0f}, "
                      f"unknown {unk:,.0f} ({unk / max(vol, 1):.5%}), minutes with unknown {r['unk_minutes']}, "
                      f"volume != buy+sell+unknown on {r['mismatch']}")

        print("\nQ3 — confirmed legs baseline (v2 payload, effort=1, default swing 8 — the App's default)")
        params = ff.MarkerParams(swing=8.0, min_prev_push=500.0, min_div_cvd=300.0, fade_min_share=0.1, gap_min=30)
        cases = [("2026-10-27", None), ("2026-10-27", "2026-10-27T08:00:00"), ("2026-10-27", "2026-10-27T10:15:00"),
                 ("2026-10-29", "2026-10-29T07:30:00"), ("2026-10-29", "2026-10-29T07:00:00")]
        for day, until in cases:
            d = dt.date.fromisoformat(day)
            until_pt = None if until is None else dt.datetime.fromisoformat(until).replace(tzinfo=PT)
            payload, status = flow_v2.build_flow_v2_payload(engine, day, session="RTH", until_pt=until_pt, params=params,
                                                            session_date=d, session_note=None, effort=True)
            legs = payload.get("legs", [])
            vol_bars = [(b["time"], b["vol"], b["buy"], b["sell"]) for b in payload["bars"]]
            tag = day + ("" if until is None else "_until_" + until[11:16].replace(":", ""))
            with open(os.path.join(OUT, f"legs_{tag}.json"), "w") as f:
                json.dump({"status": status, "source": payload.get("source"), "symbol": payload.get("symbol"),
                           "legs": legs, "live_leg": payload.get("live_leg"), "bars_vol_buy_sell": vol_bars}, f, indent=1)
            print(f"  {tag}: status {status}, source {payload.get('source')}, {len(legs)} legs, {len(vol_bars)} bars")
        engine.dispose()
    except Exception as e:  # never print connection strings
        print("DB error:", type(e).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()
