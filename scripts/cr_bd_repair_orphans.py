#!/usr/bin/env python3
"""CR-BD decision 4 — repair the holiday-mis-stamped rows and the orphaned sessions.

Steps, in order (each command is logged before it runs):
  a. deactivate the mis-stamped feature / outcome rows for the closed days
     (2026-06-19, 2026-07-03, 2026-09-07): active = false, deactivated_reason
     'holiday-mis-stamp'. Features need the owner URL (dash_backfill_writer has
     no UPDATE on bt_daily_features.active); outcomes use the backfill role.
  b. re-run apps/cron/job_orats_eod.py with --date <ORATS date> and
     FORCE_STORE_DATE=<true next trading day>: 06-18→06-22, 07-02→07-06, 09-04→09-08.
     (The job itself runs on DATABASE_URL — owner — and ORATS_TOKEN, as on Render.)
  c. scripts/cr_ab_open_implied_move.py --date <d> and
     scripts/cr_b_backfill_outcomes.py --from-date <d> --to-date <d> for the three sessions.
  verify: the G2 query — per date, active feature rows, landscape rows, active outcome rows.

Step (d), the daily capture for 2026-09-08, is CR-BD Step 2 / G3 and is run separately.
The run is recorded in bt_backfill_runs under cr_id CR-BD-repair (the deactivation
counts and every subprocess exit code land in smoke_test_results).

Usage:
    apps/web/.venv/bin/python scripts/cr_bd_repair_orphans.py --dry-run
    apps/web/.venv/bin/python scripts/cr_bd_repair_orphans.py                # a, b, c, verify
    apps/web/.venv/bin/python scripts/cr_bd_repair_orphans.py --only verify
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TICKER = "SPX"
PY = sys.executable
CR_ID = "CR-BD-repair"

# (closed day, feature_version of the mis-stamped row) — from Step 0's before-state
MISSTAMPED = [
    (date(2026, 6, 19), "v0.5.0-rebuilt"),
    (date(2026, 7, 3), "v0.5.0-rebuilt"),
    (date(2026, 9, 7), "v0.6.0-openiv"),
]
# (ORATS api date, true next trading day)
ORPHANS = [
    (date(2026, 6, 18), date(2026, 6, 22)),
    (date(2026, 7, 2), date(2026, 7, 6)),
    (date(2026, 9, 4), date(2026, 9, 8)),
]
ALL_DATES = [d for d, _ in MISSTAMPED] + [s for _, s in ORPHANS]

_G2_SQL = """
SELECT d::date AS trade_date,
       (SELECT count(*) FROM bt_daily_features f WHERE f.ticker = %(t)s AND f.trade_date = d::date AND f.active) AS active_feature_rows,
       (SELECT string_agg(f.feature_version || CASE WHEN f.active THEN '' ELSE '(inactive)' END, ',') FROM bt_daily_features f WHERE f.ticker = %(t)s AND f.trade_date = d::date) AS feature_versions,
       (SELECT count(*) FROM orats_gex_landscape l WHERE l.ticker = %(t)s AND l.trade_date = d::date) AS landscape_rows,
       (SELECT count(*) FROM bt_daily_outcomes o WHERE o.ticker = %(t)s AND o.trade_date = d::date AND o.active) AS active_outcome_rows,
       (SELECT string_agg(o.outcome_status || CASE WHEN o.active THEN '' ELSE '(inactive)' END, ',') FROM bt_daily_outcomes o WHERE o.ticker = %(t)s AND o.trade_date = d::date) AS outcomes,
       (SELECT (f.feature_vector->>'implied_move_1d')::float FROM bt_daily_features f WHERE f.ticker = %(t)s AND f.trade_date = d::date AND f.active LIMIT 1) AS im
FROM unnest(%(dates)s::date[]) AS d
ORDER BY 1
"""


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def verify(conn) -> list[tuple]:
    rows = conn.execute(_G2_SQL, {"t": TICKER, "dates": ALL_DATES}).fetchall()
    print("  date        | act.feat | feature versions                 | landscape | act.outc | outcomes                 | IM")
    for r in rows:
        print(f"  {r[0]}  | {r[1]:>8} | {str(r[2] or '-'):<32} | {r[3]:>9} | {r[4]:>8} | {str(r[5] or '-'):<24} | {r[6]}")
    return rows


def g2_pass(rows: list[tuple]) -> tuple[bool, list[str]]:
    """Decision 4 / G2: orphans have exactly one active feature + landscape + outcome row; closed days none active."""
    closed = {d for d, _ in MISSTAMPED}
    problems = []
    for td, n_feat, _, n_land, n_out, _, _ in rows:
        if td in closed:
            if n_feat or n_out:
                problems.append(f"{td}: closed day still has active rows (features {n_feat}, outcomes {n_out})")
        else:
            if not (n_feat == 1 and n_land >= 1 and n_out == 1):
                problems.append(f"{td}: expected 1/≥1/1 (feature/landscape/outcome), got {n_feat}/{n_land}/{n_out}")
    return not problems, problems


def run(cmd: list[str], env_extra: dict | None = None, dry_run: bool = False) -> int:
    env = {**os.environ, **(env_extra or {}), "PYTHONUNBUFFERED": "1"}
    shown = " ".join(f"{k}={'<owner url>' if 'URL' in k else v}" for k, v in (env_extra or {}).items())
    log(f"$ {shown + ' ' if shown else ''}{' '.join(cmd)}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
    out = proc.stdout + proc.stderr
    for k in ("DATABASE_URL", "BACKFILL_DATABASE_URL"):
        v = env.get(k, "")
        if v:
            out = out.replace(v, f"<{k}>")
    tail = [l for l in out.splitlines() if l.strip() and "sqlalchemy" not in l][-12:]
    for l in tail:
        print("    " + l)
    log(f"exit={proc.returncode}")
    return proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CR-BD decision 4 repair")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and the current G2 state; write nothing")
    ap.add_argument("--only", default=None, help="comma-separated subset of a,b,c,verify (one run row)")
    args = ap.parse_args(argv)
    _load_env()
    from packages.shared.backfill_safety import assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke
    import psycopg

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    owner_url = os.environ.get("DATABASE_URL", "").strip()
    if not owner_url or owner_url == os.environ.get("BACKFILL_DATABASE_URL", "").strip():
        sys.exit("ERROR: DATABASE_URL (owner) is required for step (a) feature deactivation and for the EOD job.")
    if not os.environ.get("ORATS_TOKEN", "").strip():
        sys.exit("ERROR: ORATS_TOKEN is required by apps/cron/job_orats_eod.py.")

    steps = [x.strip() for x in args.only.split(",")] if args.only else ["a", "b", "c", "verify"]
    assert all(x in ("a", "b", "c", "verify") for x in steps), steps
    # apps/cron/db.py accepts postgres:// and postgresql:// (what Render supplies); the local .env carries the
    # SQLAlchemy scheme postgresql+psycopg://, which psycopg rejects — hand the EOD subprocess the plain form.
    owner_url_plain = "postgresql://" + owner_url.split("://", 1)[1] if owner_url.startswith("postgresql+") else owner_url
    log(f"CR-BD repair  dry_run={args.dry_run}  steps={steps}")
    log("G2 before:")
    before = verify(conn)
    if args.dry_run:
        for closed, ver in MISSTAMPED:
            log(f"(a) would UPDATE bt_daily_features SET active=false, deactivated_reason='holiday-mis-stamp' WHERE {TICKER} {closed} {ver} (owner) "
                f"and bt_daily_outcomes likewise (backfill role)")
        for api_d, store_d in ORPHANS:
            run([PY, "apps/cron/job_orats_eod.py", "--date", api_d.isoformat()], {"FORCE_STORE_DATE": store_d.isoformat()}, dry_run=True)
        for _, store_d in ORPHANS:
            run([PY, "scripts/cr_ab_open_implied_move.py", "--date", store_d.isoformat()], dry_run=True)
            run([PY, "scripts/cr_b_backfill_outcomes.py", "--from-date", store_d.isoformat(), "--to-date", store_d.isoformat()], dry_run=True)
        log("dry-run: nothing written, no run row.")
        return 0

    smoke: dict = {"steps": steps, "deactivated_features": {}, "deactivated_outcomes": {}, "exit_codes": {}}
    with backfill_run(conn, CR_ID) as run_id:
        log(f"Run ID: {run_id}")
        if "a" in steps:
            with psycopg.connect(owner_url.replace("postgresql+psycopg://", "postgresql://")) as oc:
                for closed, ver in MISSTAMPED:
                    log(f"(a) UPDATE bt_daily_features SET active=false, deactivated_at=now(), deactivated_reason='holiday-mis-stamp' "
                        f"WHERE ticker={TICKER} trade_date={closed} feature_version={ver} AND active  [owner]")
                    cur = oc.execute(
                        "UPDATE bt_daily_features SET active = false, deactivated_at = now(), deactivated_reason = 'holiday-mis-stamp' "
                        "WHERE ticker = %s AND trade_date = %s AND feature_version = %s AND active",
                        (TICKER, closed, ver))
                    smoke["deactivated_features"][closed.isoformat()] = cur.rowcount
                    log(f"    rows={cur.rowcount}")
                oc.commit()
            for closed, ver in MISSTAMPED:
                log(f"(a) UPDATE bt_daily_outcomes SET active=false, deactivated_at=now(), deactivated_reason='holiday-mis-stamp' "
                    f"WHERE ticker={TICKER} trade_date={closed} feature_version={ver} AND active  [backfill role]")
                cur = conn.execute(
                    "UPDATE bt_daily_outcomes SET active = false, deactivated_at = now(), deactivated_reason = 'holiday-mis-stamp' "
                    "WHERE ticker = %s AND trade_date = %s AND feature_version = %s AND active",
                    (TICKER, closed, ver))
                smoke["deactivated_outcomes"][closed.isoformat()] = cur.rowcount
                log(f"    rows={cur.rowcount}")
        if "b" in steps:
            for api_d, store_d in ORPHANS:
                rc = run([PY, "apps/cron/job_orats_eod.py", "--date", api_d.isoformat()],
                         {"FORCE_STORE_DATE": store_d.isoformat(), "DATABASE_URL": owner_url_plain})
                smoke["exit_codes"][f"eod {api_d}->{store_d}"] = rc
        if "c" in steps:
            for _, store_d in ORPHANS:
                rc = run([PY, "scripts/cr_ab_open_implied_move.py", "--date", store_d.isoformat()])
                smoke["exit_codes"][f"fill {store_d}"] = rc
                rc = run([PY, "scripts/cr_b_backfill_outcomes.py", "--from-date", store_d.isoformat(), "--to-date", store_d.isoformat()])
                smoke["exit_codes"][f"outcomes {store_d}"] = rc
        log("G2 after:")
        after = verify(conn)
        ok, problems = g2_pass(after)
        smoke["g2_pass"] = ok
        smoke["g2_problems"] = problems
        smoke["g2_after"] = [[str(x) for x in r] for r in after]
        update_run_smoke(conn, run_id, smoke,
                         f"CR-BD repair: deactivated features {smoke['deactivated_features']} outcomes {smoke['deactivated_outcomes']}; "
                         f"exit codes {smoke['exit_codes']}; G2 {'PASS' if ok else 'FAIL: ' + '; '.join(problems)}")
    log(f"G2 {'PASS' if ok else 'FAIL'}" + ("" if ok else ": " + "; ".join(problems)))
    return 0 if ok and all(v == 0 for v in smoke["exit_codes"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
