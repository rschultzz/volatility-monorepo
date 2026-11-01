#!/usr/bin/env python3
"""CR-DU item 5 — the v1 engine's verdict at each card's decision minute, stored blind in `setup_engine_verdicts`.

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_set2_verdicts.py migrate
    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python scripts/cr_du_set2_verdicts.py run [--set NAME] [--dry-run]

- Cards: `setup_candidates` of the set (default `set-2:cp-calls`); only id, session_date and entry_ts_pt are read (no
  outcome). Decision minute = the bar whose close is the card's entry_ts_pt (as the review replay ends).
- Engine: react_price_preview/src/v1Engine.js via scripts/v1EngineCli.mjs `at` queries — 06:30 anchor, the engine's own
  trades that day through that minute (rules 19 / 20), the same inputs as the backtest (scripts/logs/cr-du/sessions/,
  built read-only by scripts/cr_du_backtest_v1.py build). Sessions without a prior-sessions baseline (warm-up, before
  2026-02-03) can't be evaluated: stored as NO ACTION with failed rule `warmup`.
- Write: INSERT ... ON CONFLICT DO NOTHING under the backfill protocol (dash_backfill_writer, backfill_run 'CR-DU-item5').
  Not shown in any UI. Prints counts only (SHORT / LONG / NO ACTION), never per card.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_du_backtest_v1 as BT  # noqa: E402  (loads .env, paths)

SQL = Path(__file__).resolve().parent / "cr_du_engine_verdicts.sql"


def migrate():
    import psycopg
    try:
        conn = psycopg.connect(os.environ["DATABASE_URL"].replace("+psycopg", ""), autocommit=True,
                               options="-c default_transaction_read_only=off")
    except Exception as e:
        print("connect failed:", type(e).__name__)
        sys.exit(1)
    conn.execute(SQL.read_text())
    cols = conn.execute("""SELECT column_name, data_type FROM information_schema.columns
                           WHERE table_name = 'setup_engine_verdicts' ORDER BY ordinal_position""").fetchall()
    print("setup_engine_verdicts", [f"{c}:{t}" for c, t in cols], "rows", conn.execute("SELECT count(*) FROM setup_engine_verdicts").fetchone()[0])


def run(set_name: str, dry: bool):
    with BT.db() as c:
        cards = c.execute("SELECT id, session_date, entry_ts_pt FROM setup_candidates WHERE set_name = %s ORDER BY id",
                          (set_name,)).fetchall()
    assert cards, f"no cards in {set_name}"
    at, warm = [], []
    for cid, d, e in cards:
        f = BT.SES / f"{d}.json"
        (at if f.exists() else warm).append({"id": cid, "file": str(f), "t": BT.decision_bar(e)})
    out = BT.node({"variants": {}, "sessions": [], "at": at})
    rows = [(a["id"], out["version"], a["verdict"] or "NO ACTION", json.dumps(a["failed"]), json.dumps(a.get("evidence"), default=str))
            for a in out["at"]]
    rows += [(w["id"], out["version"], "NO ACTION",
              json.dumps([{"rule": None, "code": "warmup", "text": "no prior-sessions baseline (warm-up session)"}]), None) for w in warm]
    cnt = Counter(r[2] for r in rows)
    print(f"{set_name}: {len(cards)} cards · engine {out['version']} · SHORT {cnt.get('SHORT', 0)} · LONG {cnt.get('LONG', 0)} · "
          f"NO ACTION {cnt.get('NO ACTION', 0)} (of which warm-up, not evaluated: {len(warm)})")
    if dry:
        print("dry run: nothing written")
        return
    from packages.shared.backfill_safety import assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke
    os.environ.pop("PGOPTIONS", None)          # the reads above were read-only (cr_du_backtest_v1); this one writes
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    with backfill_run(conn, "CR-DU-item5") as run_id:
        with conn.cursor() as cur:
            cur.executemany("""INSERT INTO setup_engine_verdicts (candidate_id, engine_version, verdict, failed_rules, evidence)
                               VALUES (%s, %s, %s, %s::jsonb, %s::jsonb) ON CONFLICT (candidate_id, engine_version) DO NOTHING""", rows)
            n = cur.execute("SELECT count(*) FROM setup_engine_verdicts v JOIN setup_candidates c ON c.id = v.candidate_id "
                            "WHERE c.set_name = %s AND v.engine_version = %s", (set_name, out["version"])).fetchone()[0]
        conn.commit()
        smoke = {"set": set_name, "cards": len(cards), "rows_for_set": n, "short": cnt.get("SHORT", 0), "long": cnt.get("LONG", 0),
                 "no_action": cnt.get("NO ACTION", 0), "warmup": len(warm), "engine_version": out["version"]}
        update_run_smoke(conn, run_id, smoke, f"CR-DU item 5: {n} engine verdicts for {set_name} (counts only)")
    print(f"run {run_id}: rows for the set now {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["migrate", "run"])
    ap.add_argument("--set", default="set-2:cp-calls")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    migrate() if a.cmd == "migrate" else run(a.set, a.dry_run)
