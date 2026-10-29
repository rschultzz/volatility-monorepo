#!/usr/bin/env python3
"""CR-DO — recompute the derived rows CR-BZ's fragment repair changed (train full, holdout blind).

Approved (Ryan, chat 2026-10-29): D1 blind-write the R-b row; D2 all planned holdout values blind;
D3 all three ES-based versions; D4 the 10 CR-DJ residual ratio values too; every comparison exact
in the column's own type; D5 archive to the insert-only BLIND table bt_derived_repair_archive.
No vol_risk_premium write (0 changes).

The plan is re-derived in process by the Step 0 code (scripts/cr_do_step0_dryrun.analyse) and must
equal the Step 0 exact run (scripts/logs/cr_do/step0_run4.json, reproduced by run5) before anything
is written: train cell by cell with values; holdout / R-b / forward by counts and md5 only.

Unattended protocol: dash_backfill_writer, backfill_run("CR-DO").
  1. archive: every planned row's complete old row → bt_derived_repair_archive (own transaction,
     committed first); train old rows also in smoke_test_results (values allowed), blind groups
     only as counts + md5 of the archived payload;
  2. one transaction of guarded, column-scoped UPDATEs: PK + active + computed_at + every written
     column IS NOT DISTINCT FROM its archived value cast to the column type; rowcount must be 1 or
     the whole write rolls back. Stamps backfill_run_id / computed_at (CR-BG / CR-DL precedent).
  3. smoke (fresh connection): analyse(run_id=…) re-derives the written cells from the bars alone
     (CR-BZ columns ∪ D4 residual where stored now equals post) and every group's counts / md5s must
     equal Step 0's; row counts, the other feature versions, rows before 2025-09-01 and
     vol_risk_premium are unchanged (md5 computed in SQL).
  --smoke-only RUN_ID --tag T runs step 3 again in a separate process (second-run equality) and
  records it under smoke_test_results.smoke_T. Smoke 1 of run 642e0d88 failed on a definition
  gap that resolve_written() closes (spec Amendment 3); smokes 2 and 3 are the corrected re-runs.

BLIND output rule (B4): nothing this script prints, logs or stores in bt_backfill_runs contains a
holdout / R-b / forward value, row, date or diff — counts, column names and md5 only. Error
messages name the version and group, not the row. DB errors print the error type only.

Usage:
    apps/web/.venv/bin/python scripts/cr_do_recompute_derived.py --dry-run
    apps/web/.venv/bin/python scripts/cr_do_recompute_derived.py
    apps/web/.venv/bin/python scripts/cr_do_recompute_derived.py --smoke-only <run_id>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import psycopg  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from packages.shared.backfill_safety import backfill_run, update_run_smoke  # noqa: E402
from scripts.cr_do_step0_dryrun import (  # noqa: E402
    BLIND, CRBZ_RUN, FLIP_LABELS, GROUPS, LOG_DIR, OWNED, ROWS_FROM, STAMP_COLUMNS, TICKER,
    _j, analyse, canon, column_types, connect, md5, measure, step0_summary,
)

CR_ID = "CR-DO"
ARCHIVE = "bt_derived_repair_archive"
PLAN_PATH = LOG_DIR / "step0_run4.json"
PLAN_REPRO_PATH = LOG_DIR / "step0_run5.json"
EXPECTED_VALUES = {   # Amendment 2 (exact run4 = run5): planned values per version and group
    "v0.6.1-nyse-sessions": {"train": 167, "holdout": 172, "rb": 2, "fwd": 0},
    "v0.6.0-openiv": {"train": 184, "holdout": 196, "rb": 0, "fwd": 0},
    "v0.5.0-rebuilt": {"train": 106, "holdout": 12, "rb": 0, "fwd": 0},
}
MEASURE_KEYS = ("rows", "planned_rows", "planned_values", "planned_cols", "flip_label_changes",
                "differs_after_write", "md5_keys", "md5_before", "md5_after", "md5_all_post",
                "md5_untouched", "md5_labels", "md5_differs_after_write")
MAX_NULL_CANDIDATES = 4


def resolve_written(version, rows, types, exp) -> tuple[dict, list]:
    """Smoke: pin the written cell set from the stored state, without reading the (blind) archive.

    analyse(run_id=…) re-derives the written cells as CR-BZ / D4 columns of stamped rows where stored
    now equals post. Two classes can't be told apart from the stored state alone (Amendment 3):
      - a planned cell whose recompute is NULL, now written NULL: it looks like a never-filled cell
        (stored NULL); candidates = stamped-row cells with stored NULL, post NULL, pre not NULL;
      - a reproduction-excluded CR-BZ cell that already equalled post: it looks like a written cell.
    The set is pinned by matching Step 0's md5_keys over (any subset of the NULL candidates) ×
    (at most one re-derived cell removed); every other checksum is then checked against Step 0.
    r['planned'] becomes the pinned set; NULL-written cells count as filled (they were before the write).
    """
    notes, problems = {}, []
    for g in GROUPS:
        rs = {d: r for d, r in rows.items() if r["regime"] == g and r["written"]}
        base = sorted((d, c) for d, r in rs.items() for c in r["planned"])
        nulls = sorted((d, c) for d, r in rs.items() for c in OWNED[version]
                       if r["stored"][c] is None and canon(r["post"][c], types[c]) == "NULL"
                       and canon(r["pre"][c], types[c]) != "NULL")
        key = lambda cells: md5(f"{version}|{d}|{c}" for d, c in cells)  # noqa: E731
        found = None
        if len(nulls) <= MAX_NULL_CANDIDATES:
            subsets = [[x for i, x in enumerate(nulls) if mask >> i & 1] for mask in range(2 ** len(nulls))]
            for sub in sorted(subsets, key=len, reverse=True):
                for drop in [None] + base:
                    cells = sorted(set(base) - {drop} | set(sub))
                    if key(cells) == exp[g]["md5_keys"]:
                        found = (cells, len(sub), drop is not None)
                        break
                if found:
                    break
        if not found:
            problems.append(f"{version} {g}: written cell set cannot be matched to Step 0 md5_keys "
                            f"(re-derived {len(base)}, NULL candidates {len(nulls)})")
            notes[g] = {"null_written": None, "excluded_equal_post": None}
            continue
        cells, n_null, dropped = found
        for d, r in rs.items():
            r["planned"] = sorted(c for d2, c in cells if d2 == d)
            r["filled"] = sorted(set(r["filled"]) | {c for d2, c in cells if d2 == d and r["stored"][c] is None})
        # the planned 'before': pre for CR-BZ cells (they reproduced), Step 0's train 'from' for the D4 cells
        detail = exp["train"]["rows_detail"] if g == "train" else {}
        before = [f"{version}|{d}|{c}|" + (detail[str(d)]["planned"][c]["from"] if c in rows[d]["resid"]
                                          else canon(rows[d]["pre"][c], types[c])) for d, c in cells]
        if md5(before) != exp[g]["md5_before"]:
            problems.append(f"{version} {g}: planned 'before' (pre / D4 from) differs from Step 0 md5_before")
        notes[g] = {"null_written": n_null, "excluded_equal_post": int(dropped),
                    "null_written_cols": sorted({c for d, c in cells if rows[d]["stored"][c] is None})}
    return notes, problems


# ── plan source ───────────────────────────────────────────────────────────────────────────────
def load_expected() -> dict:
    a, b = json.loads(PLAN_PATH.read_text()), json.loads(PLAN_REPRO_PATH.read_text())
    for version in OWNED:
        for g in GROUPS:
            x, y = a["versions"][version][g], b["versions"][version][g]
            assert all(x[k] == y[k] for k in MEASURE_KEYS), (version, g, "run4 != run5")
            assert x["planned_values"] == EXPECTED_VALUES[version][g], (version, g, x["planned_values"])
            assert x["flip_label_changes"] == 0, (version, g)
    assert all(a["versions"][v]["train"]["rows_detail"] == b["versions"][v]["train"]["rows_detail"] for v in OWNED)
    return a


# ── table-level fingerprints (md5 computed in SQL; no value leaves the database) ─────────────
def fingerprints(conn) -> dict:
    owned = list(OWNED)
    out = {}
    for tk, fv, n, a, h in conn.execute(
            "SELECT ticker, feature_version, count(*), count(*) FILTER (WHERE active), "
            "md5(string_agg(o::text, E'\\n' ORDER BY o::text) FILTER "
            "(WHERE NOT (active AND trade_date >= %s AND ticker = %s AND feature_version = ANY(%s)))) "
            "FROM bt_daily_outcomes o GROUP BY 1, 2 ORDER BY 1, 2", (ROWS_FROM, TICKER, owned)).fetchall():
        out[f"bt_daily_outcomes {tk} {fv}"] = {"rows": n, "active": a, "md5_outside_scope": h}
    n, h = conn.execute(
        "SELECT count(*), md5(string_agg(concat_ws('|', ticker, feature_version, trade_date, active, "
        "vol_risk_premium), E'\\n' ORDER BY ticker, feature_version, trade_date, active, vol_risk_premium)) "
        "FROM bt_daily_features").fetchone()
    out["bt_daily_features vol_risk_premium"] = {"rows": n, "md5": h}
    return out


# ── preflight: the in-process plan must equal Step 0 ──────────────────────────────────────────
def build(conn_factory, types, crbz_done, expected, run_id=None, heartbeat=None) -> tuple[dict, dict, list]:
    """analyse() every version (fresh connection each); compare its measure to Step 0's."""
    rows_by_v, meas, problems = {}, {}, []
    for version in OWNED:
        conn = conn_factory()
        conn.read_only = True
        rows = analyse(conn, version, types, crbz_done, run_id=run_id)
        conn.close()
        if heartbeat:
            heartbeat()
        exp = expected["versions"][version]
        if run_id:
            notes, more = resolve_written(version, rows, types, exp)
            problems += more
        m = measure(version, rows, types) if run_id else step0_summary(version, rows, types)
        if run_id:
            for g in GROUPS:
                m[g].update(notes[g])
        for g in GROUPS:
            for k in MEASURE_KEYS:
                if run_id and k == "md5_before":   # stored at the written cells now = Step 0's after
                    ok = m[g][k] == exp[g]["md5_after"]
                else:
                    ok = m[g][k] == exp[g][k]
                if not ok:
                    problems.append(f"{version} {g}: {k} differs from Step 0" +
                                    (f" ({m[g][k]} != {exp[g][k]})" if g == "train" and not k.startswith("md5") else ""))
        if run_id is None:
            if m["train"]["rows_detail"] != exp["train"]["rows_detail"]:
                bad = sorted(d for d in exp["train"]["rows_detail"]
                             if m["train"]["rows_detail"].get(d) != exp["train"]["rows_detail"][d])
                problems.append(f"{version} train: rows_detail differs from Step 0 on {bad}")
            for g in GROUPS:
                if not m[g]["resid_ok"]:
                    problems.append(f"{version} {g}: D4 residual proof fails")
        else:
            for g in GROUPS:
                rs = [r for r in rows.values() if r["regime"] == g]
                stamped, planned = sum(r["written"] for r in rs), sum(bool(r["planned"]) for r in rs)
                m[g]["stamped_rows"] = stamped
                if stamped != planned or stamped != exp[g]["planned_rows"]:
                    problems.append(f"{version} {g}: stamped rows {stamped}, re-derived written rows {planned}, "
                                    f"Step 0 planned rows {exp[g]['planned_rows']}")
            # train cells, with values: stored now equals Step 0's planned 'to'
            for d, r in rows.items():
                if r["regime"] != "train":
                    continue
                for c, ft in exp["train"]["rows_detail"].get(str(d), {}).get("planned", {}).items():
                    now = canon(r["stored"][c], types[c])
                    if now != ft["to"]:
                        problems.append(f"{version} train {d}.{c}: stored {now} != planned {ft['to']}")
        rows_by_v[version] = rows
        meas[version] = {g: {k: v for k, v in m[g].items() if k != "rows_detail"} for g in GROUPS}
    return rows_by_v, meas, problems


def _param(v, typ):
    """The value in its column type's canonical text (exact float4 / float8), cast in SQL."""
    c = canon(v, typ)
    return None if c == "NULL" else c


class BlindDbError(RuntimeError):
    """A DB error with its message dropped: a 'failing row' DETAIL could carry holdout values."""


def _sanitised(fn, *a):
    try:
        return fn(*a)
    except psycopg.Error as e:
        raise BlindDbError(f"DB error during {fn.__name__}: {type(e).__name__}") from None


# ── archive and write ─────────────────────────────────────────────────────────────────────────
def archive(conn, run_id, rows_by_v) -> dict:
    """Insert-only: one archive row per planned row (complete old row). Returns counts / md5s per group."""
    summary = {g: {"rows": 0, "md5": []} for g in GROUPS}
    train_rows = {}
    with conn.transaction():
        for version, rows in rows_by_v.items():
            for d, r in rows.items():
                if not r["planned"]:
                    continue
                old = {k: _j(v) for k, v in r["stored"].items()}
                old = json.loads(json.dumps(old, default=str))
                conn.execute(
                    f"INSERT INTO {ARCHIVE} (run_id, cr_id, table_name, ticker, feature_version, trade_date, "
                    "regime, columns_written, old_row) VALUES (%s, %s, 'bt_daily_outcomes', %s, %s, %s, %s, %s, %s)",
                    (run_id, CR_ID, TICKER, version, d, r["regime"], r["planned"], Jsonb(old)))
                g = summary[r["regime"]]
                g["rows"] += 1
                g["md5"].append(f"{version}|{d}|{json.dumps(old, sort_keys=True)}")
                if r["regime"] == "train":
                    train_rows.setdefault(version, {})[str(d)] = old
    return {"archive_table": ARCHIVE, "train_old_rows": train_rows,
            "groups": {g: {"rows": s["rows"], "md5_payload": hashlib.md5("\n".join(sorted(s["md5"])).encode()).hexdigest()}
                       for g, s in summary.items()}}


def write(conn, run_id, rows_by_v, types) -> dict:
    """One transaction of guarded, column-scoped UPDATEs; any miss rolls everything back."""
    n = {g: 0 for g in GROUPS}
    with conn.transaction():
        for version, rows in rows_by_v.items():
            for d, r in rows.items():
                cols = r["planned"]
                if not cols:
                    continue
                s, post = r["stored"], r["post"]
                sets = ", ".join(f"{c} = %s::{types[c]}" for c in cols)
                guards = " AND ".join(f"{c} IS NOT DISTINCT FROM %s::{types[c]}" for c in cols)
                cur = conn.execute(
                    f"UPDATE bt_daily_outcomes SET {sets}, backfill_run_id = %s, computed_at = NOW() "
                    f"WHERE ticker = %s AND trade_date = %s AND feature_version = %s AND active "
                    f"AND computed_at IS NOT DISTINCT FROM %s AND {guards}",
                    [_param(post[c], types[c]) for c in cols]
                    + [run_id, TICKER, d, version, s["computed_at"]]
                    + [_param(s[c], types[c]) for c in cols])
                if cur.rowcount != 1:
                    where = f"{d}" if r["regime"] == "train" else "a blind row"
                    raise RuntimeError(f"guard failed on {version} {r['regime']} {where} (rowcount={cur.rowcount}); "
                                       "transaction rolled back, nothing written")
                n[r["regime"]] += 1
    return n


# ── smoke ─────────────────────────────────────────────────────────────────────────────────────
def smoke(types, crbz_done, expected, run_id, fp_before, heartbeat=None) -> tuple[dict, list]:
    _, meas, problems = build(connect, types, crbz_done, expected, run_id=run_id, heartbeat=heartbeat)
    conn = connect()
    conn.read_only = True
    fp_after = fingerprints(conn)
    conn.close()
    if fp_before is not None and fp_after != fp_before:
        problems += [f"{k}: fingerprint changed" for k in sorted(set(fp_before) | set(fp_after))
                     if fp_before.get(k) != fp_after.get(k)]
    return {"measure": meas, "fingerprints": fp_after}, problems


def _print_measure(meas) -> None:
    for version, gs in meas.items():
        for g, m in gs.items():
            print(f"  {version} {g:7} rows {m['rows']}; planned/written rows {m['planned_rows']} values "
                  f"{m['planned_values']}; status/touch/reached/days written {m['flip_label_changes']}; "
                  f"differ from post {m['differs_after_write']}" +
                  (f"; stamped {m['stamped_rows']}" if "stamped_rows" in m else "") +
                  (f"; NULL-written {m['null_written']} {m.get('null_written_cols')}; excluded cell equal to post "
                   f"{m['excluded_equal_post']}" if "null_written" in m else ""))
            print(f"    md5 keys {m['md5_keys']} after {m['md5_after']} all_post {m['md5_all_post']} "
                  f"untouched {m['md5_untouched']} labels {m['md5_labels']} differs {m['md5_differs_after_write']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CR-DO derived recompute for CR-BZ's repaired bars")
    ap.add_argument("--dry-run", action="store_true", help="preflight and print the plan; write nothing, no run row")
    ap.add_argument("--smoke-only", metavar="RUN_ID", help="independent smoke of a completed run")
    ap.add_argument("--tag", default="2", help="--smoke-only: label stored as smoke_<tag>")
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    expected = load_expected()
    try:
        conn = connect()
    except Exception as e:  # never echo the connection string
        print(f"DB connect failed: {type(e).__name__}")
        return 1
    types = column_types(conn)
    crbz_done = conn.execute("SELECT completed_at FROM bt_backfill_runs WHERE run_id = %s", (CRBZ_RUN,)).fetchone()[0]

    if args.smoke_only:
        run_id, tag = args.smoke_only, args.tag
        status, results = conn.execute("SELECT status, smoke_test_results FROM bt_backfill_runs "
                                       "WHERE run_id = %s AND cr_id = %s", (run_id, CR_ID)).fetchone()
        fp1 = results["smoke"]["fingerprints"]
        out, problems = smoke(types, crbz_done, expected, run_id, fp1, heartbeat=lambda: conn.execute("SELECT 1"))
        ok = not problems
        digest = hashlib.md5(json.dumps(out["measure"], sort_keys=True).encode()).hexdigest()
        print(f"=== {CR_ID} smoke {tag} for run {run_id} ({status}) ===")
        _print_measure(out["measure"])
        print(f"fingerprints == smoke1: {out['fingerprints'] == fp1}; md5 of the whole measure {digest}")
        for p in problems:
            print("  " + p)
        entry = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "measure": out["measure"],
                 "md5_measure": digest, "problems": problems, "ok": ok}
        conn.execute("UPDATE bt_backfill_runs SET smoke_test_results = jsonb_set(smoke_test_results, %s, %s) "
                     "WHERE run_id = %s", ([f"smoke_{tag}"], Jsonb(entry), run_id))
        (LOG_DIR / f"smoke_{tag}.json").write_text(json.dumps({"run_id": run_id, **out, "md5_measure": digest,
                                                               "problems": problems, "ok": ok}, indent=1, default=str))
        print(f"smoke {tag}: {'OK' if ok else 'FAIL'}")
        conn.close()
        return 0 if ok else 1

    hour = dt.datetime.now(dt.timezone.utc).hour
    if 11 <= hour < 14 and not args.dry_run:
        print("refusing: 11:00–14:00 UTC is the CR-AA sweep / CR-022 insert window")
        return 1
    fp_before = fingerprints(conn)
    conn.close()
    rows_by_v, meas, problems = build(connect, types, crbz_done, expected)
    total = {g: sum(meas[v][g]["planned_values"] for v in OWNED) for g in GROUPS}
    nrows = {g: sum(meas[v][g]["planned_rows"] for v in OWNED) for g in GROUPS}
    print(f"=== {CR_ID} === plan: rows {nrows}, values {total}; preflight {'OK' if not problems else 'FAIL'}")
    for p in problems:
        print("  " + p)
    if problems:
        return 1
    if args.dry_run:
        for version in OWNED:
            for d, x in expected["versions"][version]["train"]["rows_detail"].items():
                if x["planned"]:
                    print(f"  {version} train {d}: " + "; ".join(f"{c} {ft['from']} → {ft['to']}" for c, ft in x["planned"].items()))
        _print_measure(meas)
        print("[DRY RUN — nothing written, no run row]")
        return 0

    conn = connect()
    ok, summary = False, {}
    with backfill_run(conn, CR_ID) as run_id:
        print(f"Run ID: {run_id}")
        results: dict = {"plan_source": f"{PLAN_PATH.name} (= {PLAN_REPRO_PATH.name}) generated {expected['generated_utc']}",
                         "comparison": "exact in column type", "planned_rows": nrows, "planned_values": total,
                         "fingerprints_before": fp_before}
        results["archive"] = _sanitised(archive, conn, run_id, rows_by_v)
        print(f"archived {sum(a['rows'] for a in results['archive']['groups'].values())} rows to {ARCHIVE}: "
              + ", ".join(f"{g} {a['rows']}" for g, a in results["archive"]["groups"].items()))
        update_run_smoke(conn, run_id, results, "archive written; UPDATE pending")
        n = _sanitised(write, conn, run_id, rows_by_v, types)
        conn.execute("UPDATE bt_backfill_runs SET rows_updated = %s WHERE run_id = %s", (sum(n.values()), run_id))
        print(f"committed {sum(n.values())} UPDATEs: {n}")
        out, problems = smoke(types, crbz_done, expected, run_id, fp_before, heartbeat=lambda: conn.execute("SELECT 1"))
        ok = not problems
        results["smoke"] = {**out, "rows_updated": n, "problems": problems, "ok": ok}
        update_run_smoke(
            conn, run_id, results,
            f"CR-DO: bt_daily_outcomes rows UPDATEd in place from CR-BZ-repaired bars (CR-BZ columns + D4 CR-DJ "
            f"residual), rows {n}, values {total}; holdout / R-b blind (md5 only); archive in {ARCHIVE}; "
            f"smoke {'OK' if ok else 'FAIL: ' + '; '.join(problems[:20])}")
    (LOG_DIR / "write_smoke.json").write_text(json.dumps({"run_id": run_id, **results}, indent=1, default=str))
    print("smoke1:")
    _print_measure(out["measure"])
    print(f"smoke1: {'OK' if ok else 'FAIL'}")
    for p in problems:
        print("  " + p)
    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except psycopg.Error as e:   # never echo a connection string or a row
        print(f"DB error: {type(e).__name__}")
        sys.exit(1)
