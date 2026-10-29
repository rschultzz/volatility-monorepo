#!/usr/bin/env python3
"""CR-DL — recompute the derived rows the CR-DJ bar patch changed (2026-03-16 → 03-19 bars).

Approved (Ryan, chat 2026-10-29): D1 all three ES-based versions; D2 only the columns CR-DJ
changes; D3 the 18 vol_risk_premium rows. 43 bt_daily_outcomes UPDATEs: v0.5.0-rebuilt 03-11 /
03-12 are left as stored (their changed t5 columns were never filled — CR-AA sweep run 5cef55c5;
CR-DL corrects values, it doesn't fill gaps).

The plan is frozen from the Step 0 dry run (scripts/logs/cr_dl/step0_dryrun.json): for each
(version, trade_date, column) in the CR-DJ diff whose stored value is not NULL, before = stored,
after = post (current bars). Before writing, the post recompute (same writer code) must equal
every planned after, and every planned column must still equal its before.

Unattended protocol: dash_backfill_writer, backfill_run("CR-DL"); the complete old rows are
archived to bt_backfill_runs.smoke_test_results before the write; one transaction of guarded,
column-scoped UPDATEs (a row changed since the archive rolls the whole write back). Outcome rows
get backfill_run_id / computed_at stamped (CR-BG precedent); feature rows don't (CR-D convention).

Smoke (after commit): the post recompute again — written columns equal the dry run's expected
values; every row's remaining stored-vs-post difference equals the dry run's minus the written
columns; labels and every unwritten column equal the archive; row counts unchanged; -spxcash
untouched; vol_risk_premium equals the dry run's post.

B4: every bar read is capped at 2026-06-05; rows are trade_date ≤ 2026-03-20.

Usage:
    apps/web/.venv/bin/python scripts/cr_dl_recompute_derived.py --dry-run
    apps/web/.venv/bin/python scripts/cr_dl_recompute_derived.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)
from scripts.cr_b_backfill_outcomes import _fetch_landscape  # noqa: E402
from scripts.cr_d_backfill_vol_features import _compute_vol_features  # noqa: E402
from scripts.cr_dl_step0_dryrun import (  # noqa: E402
    LOG_DIR, OWNED, POS_COLUMNS, ROWS_FROM, ROWS_TO, TICKER, TRAIN_END, VRP_FROM, VRP_TO,
    _diff, _j, _same, compute_legacy, compute_nyse,
)

CR_ID = "CR-DL"
PLAN_PATH = LOG_DIR / "step0_dryrun.json"
EXPECTED_UPDATES = {"v0.6.1-nyse-sessions": 15, "v0.6.0-openiv": 15, "v0.5.0-rebuilt": 13}
EXPECTED_VRP = 18
LEFT_AS_STORED = {("v0.5.0-rebuilt", "2026-03-11"), ("v0.5.0-rebuilt", "2026-03-12")}
LABELS = ("outcome_status", "reached_touch", "reached_close", "days_to_reach",
          "max_excursion_in_direction", "horizon_sessions", "horizon_end_date") + POS_COLUMNS
STAMP_COLUMNS = ("backfill_run_id", "computed_at")
EXACT = 1e-9


def load_plan() -> tuple[dict, dict, dict]:
    """(outcome plan {version: {date: {col: (before, after)}}}, vrp plan {date: (before, after)}, dry run)."""
    dry = json.loads(PLAN_PATH.read_text())
    plan: dict = {}
    for version, v in dry["versions"].items():
        plan[version] = {}
        for d, r in v["rows"].items():
            if not r["crdj"]:
                continue
            assert not set(r["crdj"]) & set(r["crbz"]), (version, d, "CR-BZ also changes a CR-DJ column")
            cols = {c: (r["stored_to_post"][c]["from"], r["stored_to_post"][c]["to"])
                    for c in r["crdj"] if r["stored_to_post"][c]["from"] is not None}
            if cols:
                plan[version][d] = cols
            else:
                assert (version, d) in LEFT_AS_STORED, (version, d)
        assert len(plan[version]) == EXPECTED_UPDATES[version], (version, len(plan[version]))
    vrp = {d: (x["stored"], x["post"]) for d, x in dry["vol_risk_premium"]["rows"].items() if x["crdj"]}
    assert len(vrp) == EXPECTED_VRP, len(vrp)
    return plan, vrp, dry


def fetch_outcomes(conn, version) -> dict:
    cur = conn.execute(
        "SELECT * FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s AND active "
        "AND trade_date BETWEEN %s AND %s ORDER BY trade_date", (TICKER, version, ROWS_FROM, ROWS_TO))
    names = [c.name for c in cur.description]
    return {str(r[names.index("trade_date")]): dict(zip(names, r)) for r in cur.fetchall()}


def compute_post(conn, version, stored) -> dict:
    """The writer's own code on the current bars (the dry run's post state), keyed by ISO date."""
    by_date = {s["trade_date"]: s for s in stored.values()}
    feats = {d: (reg, fv) for d, reg, fv in conn.execute(
        "SELECT trade_date, regime_at_classification, feature_vector FROM bt_daily_features "
        "WHERE ticker=%s AND feature_version=%s AND active AND trade_date = ANY(%s)",
        (TICKER, version, list(by_date))).fetchall()}
    assert set(feats) == set(by_date), version
    landscape = _fetch_landscape(conn, TICKER, list(by_date))
    out = (compute_nyse(conn, by_date, feats, landscape) if version == "v0.6.1-nyse-sessions"
           else compute_legacy(conn, version, by_date, feats, landscape, by_date))
    return {str(d): row for d, row in out.items()}


def fetch_vrp(conn) -> dict:
    return {str(d): v for d, v in conn.execute(
        "SELECT trade_date, vol_risk_premium FROM bt_daily_features WHERE ticker=%s AND "
        "feature_version='v0.5.0-rebuilt' AND active AND trade_date BETWEEN %s AND %s",
        (TICKER, VRP_FROM, VRP_TO)).fetchall()}


def counts(conn) -> dict:
    rows = conn.execute(
        "SELECT feature_version, count(*), count(*) FILTER (WHERE active), max(computed_at) "
        "FROM bt_daily_outcomes WHERE ticker=%s GROUP BY 1 ORDER BY 1", (TICKER,)).fetchall()
    return {v: {"rows": n, "active": a, "max_computed_at": _j(m)} for v, n, a, m in rows}


def preflight(conn, plan, vrp_plan) -> tuple[dict, dict, dict, dict, list]:
    """Stored rows, post recompute, current VRP and fresh VRP; problems if the plan no longer holds."""
    problems, stored, post = [], {}, {}
    for version in OWNED:
        stored[version] = fetch_outcomes(conn, version)
        post[version] = compute_post(conn, version, stored[version])
        for d, cols in plan[version].items():
            for c, (before, after) in cols.items():
                if not _same(stored[version][d][c], before, EXACT):
                    problems.append(f"{version} {d}.{c}: stored {stored[version][d][c]!r} != planned before {before!r}")
                if not _same(post[version][d][c], after, EXACT):
                    problems.append(f"{version} {d}.{c}: recomputed {post[version][d][c]!r} != planned after {after!r}")
    vrp_now = fetch_vrp(conn)
    vrp_new = {}
    for d, (before, after) in vrp_plan.items():
        vrp_new[d] = _compute_vol_features(conn, dt.date.fromisoformat(d), TICKER)["vol_risk_premium"]
        if not _same(vrp_now.get(d), before, EXACT):
            problems.append(f"vrp {d}: stored {vrp_now.get(d)!r} != planned before {before!r}")
        if not _same(vrp_new[d], after, EXACT):
            problems.append(f"vrp {d}: recomputed {vrp_new[d]!r} != planned after {after!r}")
    return stored, post, vrp_now, vrp_new, problems


def _f(v):
    return None if v is None else float(v)


def write(conn, run_id, plan, stored, post, vrp_now, vrp_new) -> int:
    n = 0
    # Guard parameters are cast to the column type: psycopg reads `real` as text, so the Python
    # float (0.07837067) is not the float4 widened to double, and an uncast guard never matches.
    types = dict(conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'bt_daily_outcomes'").fetchall())
    with conn.transaction():
        for version, rows in plan.items():
            for d, cols in rows.items():
                names = list(cols)
                sets = ", ".join(f"{c} = %s" for c in names)
                guards = " AND ".join(f"{c} IS NOT DISTINCT FROM %s::{types[c]}" for c in names)
                cur = conn.execute(
                    f"UPDATE bt_daily_outcomes SET {sets}, backfill_run_id = %s, computed_at = NOW() "
                    f"WHERE ticker = %s AND trade_date = %s AND feature_version = %s AND active "
                    f"AND computed_at IS NOT DISTINCT FROM %s AND {guards}",
                    [_f(post[version][d][c]) for c in names]
                    + [run_id, TICKER, dt.date.fromisoformat(d), version, stored[version][d]["computed_at"]]
                    + [stored[version][d][c] for c in names])
                if cur.rowcount != 1:
                    raise RuntimeError(f"guard failed for {version} {d} (rowcount={cur.rowcount}); "
                                       "transaction rolled back, nothing written")
                n += 1
        for d in sorted(vrp_new):
            cur = conn.execute(
                "UPDATE bt_daily_features SET vol_risk_premium = %s WHERE ticker = %s AND trade_date = %s "
                "AND feature_version = 'v0.5.0-rebuilt' AND active AND vol_risk_premium IS NOT DISTINCT FROM %s",
                (_f(vrp_new[d]), TICKER, dt.date.fromisoformat(d), vrp_now[d]))
            if cur.rowcount != 1:
                raise RuntimeError(f"guard failed for vrp {d} (rowcount={cur.rowcount}); "
                                   "transaction rolled back, nothing written")
            n += 1
    return n


def _stored_equals(stored, expected) -> bool:
    """Written value read back: session_* / *_over_im are `real` (float4), so ~7 significant digits."""
    if stored is None or expected is None:
        return stored is None and expected is None
    return abs(float(stored) - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))


def _same_diff(a: dict, b: dict) -> bool:
    def eq(x, y):
        if isinstance(x, str) or isinstance(y, str):
            return str(x) == str(y)
        return _same(x, y, EXACT)
    return set(a) == set(b) and all(eq(a[c][k], b[c][k]) for c in a for k in ("from", "to"))


def smoke(conn, run_id, plan, vrp_plan, dry, stored_old, vrp_old, counts_before) -> tuple[dict, list]:
    problems, written = [], 0
    for version, cols_owned in OWNED.items():
        after = fetch_outcomes(conn, version)
        post = compute_post(conn, version, after)
        if set(after) != set(stored_old[version]):
            problems.append(f"{version}: row set changed")
        for d, a in after.items():
            planned = plan[version].get(d, {})
            for c, (_, exp) in planned.items():
                if not _stored_equals(a[c], exp):
                    problems.append(f"{version} {d}.{c}: stored {a[c]!r} != expected {exp!r}")
            expected_rest = {c: x for c, x in dry["versions"][version]["rows"][d]["stored_to_post"].items()
                             if c not in planned}
            now_rest = _diff(cols_owned, a, post[d])
            if not _same_diff(now_rest, expected_rest):
                problems.append(f"{version} {d}: stored-vs-post now {now_rest} != dry run minus written {expected_rest}")
            old = stored_old[version][d]
            for c, v in a.items():
                if c in planned or c in STAMP_COLUMNS:
                    continue
                if v != old[c]:
                    problems.append(f"{version} {d}.{c}: changed {old[c]!r} → {v!r}" + (" (LABEL)" if c in LABELS else ""))
            if planned:
                written += 1
                if str(a["backfill_run_id"]) != run_id:
                    problems.append(f"{version} {d}: backfill_run_id not stamped")
            elif a["backfill_run_id"] != old["backfill_run_id"] or a["computed_at"] != old["computed_at"]:
                problems.append(f"{version} {d}: unplanned row stamped")
    vrp_after = fetch_vrp(conn)
    for d, v in vrp_after.items():
        exp = vrp_plan[d][1] if d in vrp_plan else vrp_old[d]
        if not _same(v, exp, EXACT):
            problems.append(f"vrp {d}: {v!r} != expected {exp!r}")
    counts_after = counts(conn)
    for v, c in counts_after.items():
        b = counts_before.get(v)
        if b is None or (c["rows"], c["active"]) != (b["rows"], b["active"]):
            problems.append(f"{v}: row counts {b} → {c}")
        if v not in OWNED and c != b:
            problems.append(f"{v}: changed (should be untouched) {b} → {c}")
    return {"outcome_rows_written": written, "vrp_rows_checked": len(vrp_after),
            "counts_before": counts_before, "counts_after": counts_after}, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CR-DL derived recompute for the CR-DJ patched window")
    ap.add_argument("--dry-run", action="store_true", help="preflight and print the plan; write nothing, no run row")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)

    plan, vrp_plan, dry = load_plan()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    assert max(dt.date.fromisoformat(d) for v in plan.values() for d in v) <= ROWS_TO <= TRAIN_END

    counts_before = counts(conn)
    stored, post, vrp_now, vrp_new, problems = preflight(conn, plan, vrp_plan)
    n_out = sum(len(v) for v in plan.values())
    print(f"=== {CR_ID} === plan: {n_out} outcome UPDATEs "
          f"({', '.join(f'{v} {len(r)}' for v, r in plan.items())}) + {len(vrp_plan)} vol_risk_premium; "
          f"preflight {'OK' if not problems else 'FAIL'}")
    for p in problems:
        print("  " + p)
    if problems:
        conn.close()
        return 1
    if args.dry_run:
        for version, rows in plan.items():
            for d, cols in rows.items():
                print(f"  {version} {d}: " + "; ".join(f"{c} {b} → {a}" for c, (b, a) in cols.items()))
        for d, (b, a) in sorted(vrp_plan.items()):
            print(f"  vol_risk_premium {d}: {b} → {a}")
        print("[DRY RUN — nothing written, no run row]")
        conn.close()
        return 0

    archive = {
        "bt_daily_outcomes": {v: {d: {k: _j(x) for k, x in stored[v][d].items()} for d in rows}
                              for v, rows in plan.items()},
        "bt_daily_features.vol_risk_premium": {d: vrp_now[d] for d in sorted(vrp_plan)},
        "plan": {v: {d: {c: list(ba) for c, ba in cols.items()} for d, cols in rows.items()}
                 for v, rows in plan.items()},
        "left_as_stored": sorted(f"{v} {d}" for v, d in LEFT_AS_STORED),
        "plan_source": str(PLAN_PATH.relative_to(REPO_ROOT)) + " generated " + dry["generated_utc"],
    }
    archive = json.loads(json.dumps(archive, default=str))  # UUID backfill_run_id etc.
    ok = False
    with backfill_run(conn, CR_ID) as run_id:
        print(f"Run ID: {run_id}")
        results: dict = {"archive": archive}
        update_run_smoke(conn, run_id, results, "archive written; UPDATE pending")
        n = write(conn, run_id, plan, stored, post, vrp_now, vrp_new)
        update_run_progress(conn, run_id, 0)  # UPDATE-only run
        print(f"committed {n} UPDATEs ({n_out} outcome rows + {len(vrp_plan)} vol_risk_premium)")
        summary, problems = smoke(conn, run_id, plan, vrp_plan, dry, stored, vrp_now, counts_before)
        ok = not problems
        results["smoke"] = {**summary, "problems": problems, "ok": ok}
        update_run_smoke(
            conn, run_id, results,
            f"CR-DL: {n_out} bt_daily_outcomes rows (CR-DJ columns only) + {len(vrp_plan)} vol_risk_premium "
            f"UPDATEd in place from the patched M26 bars; v0.5.0-rebuilt 03-11 / 03-12 left as stored; "
            f"smoke {'OK' if ok else 'FAIL: ' + '; '.join(problems[:20])}")
    (LOG_DIR / "write_smoke.json").write_text(json.dumps({"run_id": run_id, **results}, indent=1, default=str))
    print(f"smoke: {'OK' if ok else 'FAIL'}  ({summary})")
    for p in problems:
        print("  " + p)
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
