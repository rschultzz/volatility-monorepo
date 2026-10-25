"""CR-CC L3 — load a setup gallery JSON (L10) into `setup_candidates`.

    apps/web/.venv/bin/python -m scripts.load_setup_candidates research/gallery/<file>.json [--dry-run]

Validation (A1.4 / D4) rejects the whole file on any error; nothing is inserted:
  - top level: cr_id, set_name, level_frame (es_corrected | es | B | spx_cash), events[];
  - each event: id, session_date (< today in PT), direction (long | short), entry_ts_pt;
  - *_ts_pt values are naive PT wall-clock ISO strings (no offset, no 'Z') inside the
    event's ES session window (prior day 15:00 PT -> session_date 13:00 PT);
    entry_ts_pt is the close of the signal bar (A1.2);
  - `meta` keys that look like outcome information (pnl_*, res_*, result*, mv_*, exit*,
    outcome, by_exit, win*/loss*, return*, r<N>, f<N>) are rejected: outcomes go in
    `outcome` only.

Rows are immutable: INSERT ... ON CONFLICT (id) DO NOTHING; the run reports inserted vs
skipped. A corrected re-load needs a new set_name. Runs under the backfill protocol
(dash_backfill_writer has INSERT/SELECT on setup_candidates and no DELETE).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
CR_ID = "CR-CC"
LEVEL_FRAMES = ("es_corrected", "es", "B", "spx_cash")
TS_FIELDS = ("entry_ts_pt", "test_ts_pt", "confirm_ts_pt", "retest_ts_pt")
FLOAT_FIELDS = (
    "level_price", "level_net_bn", "level_gross_bn",
    "entry_price", "stop_price", "extreme_price",
)
COLUMNS = (
    "id", "cr_id", "set_name", "session_date", "cell", "direction",
    *FLOAT_FIELDS[:3], "entry_ts_pt", *FLOAT_FIELDS[3:],
    "test_ts_pt", "confirm_ts_pt", "retest_ts_pt", "meta", "outcome",
)
# First token of the key (up to '_', a digit or the end): pnl_a, res_4, mv_30m, exit_price,
# r10, f15 match; reason, retest_ts, window, resistance do not.
OUTCOME_KEY_RE = re.compile(
    r"^(pnl|res|result|mv|exit|outcome|by_exit|win|winner|loss|loser|return|r\d+|f\d+)(_|\d|$)",
    re.IGNORECASE,
)
_NAIVE_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$")

INSERT_SQL = (
    f"INSERT INTO setup_candidates ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(COLUMNS))}) "
    "ON CONFLICT (id) DO NOTHING"
)


class GalleryError(ValueError):
    """The gallery file failed validation; the message lists every problem."""


def today_pt() -> dt.date:
    return dt.datetime.now(PT).date()


def _parse_ts(raw, field: str, where: str, errors: list[str]) -> dt.datetime | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not _NAIVE_TS_RE.match(raw.strip()):
        errors.append(f"{where}: {field} {raw!r} is not a naive PT timestamp (YYYY-MM-DDTHH:MM[:SS], no offset)")
        return None
    return dt.datetime.fromisoformat(raw.strip().replace(" ", "T"))


def _float(raw, field: str, where: str, errors: list[str]) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        errors.append(f"{where}: {field} {raw!r} is not a number")
        return None


def validate_gallery(doc: dict, today: dt.date | None = None) -> list[dict]:
    """Return one row dict per event (keys = COLUMNS), or raise GalleryError."""
    today = today or today_pt()
    errors: list[str] = []
    if not isinstance(doc, dict):
        raise GalleryError("gallery must be a JSON object")

    cr_id = doc.get("cr_id")
    set_name = doc.get("set_name")
    level_frame = doc.get("level_frame")
    events = doc.get("events")
    if not isinstance(cr_id, str) or not cr_id.strip():
        errors.append("cr_id is required")
    if not isinstance(set_name, str) or not set_name.strip():
        errors.append("set_name is required")
    if level_frame not in LEVEL_FRAMES:
        errors.append(f"level_frame must be one of {LEVEL_FRAMES}, got {level_frame!r}")
    if not isinstance(events, list) or not events:
        errors.append("events must be a non-empty list")
    if errors:
        raise GalleryError("; ".join(errors))

    rows: list[dict] = []
    seen: set[str] = set()
    for k, ev in enumerate(events):
        where = f"events[{k}]"
        if not isinstance(ev, dict):
            errors.append(f"{where}: not an object")
            continue
        n = ev.get("id")
        if n is None or str(n).strip() == "":
            errors.append(f"{where}: id is required")
            continue
        cand_id = f"{cr_id}:{set_name}:{n}"
        where = f"{where} ({cand_id})"
        if cand_id in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(cand_id)

        try:
            session_date = dt.date.fromisoformat(str(ev.get("session_date")))
        except ValueError:
            errors.append(f"{where}: session_date {ev.get('session_date')!r} is not YYYY-MM-DD")
            continue
        if session_date >= today:
            errors.append(f"{where}: session_date {session_date} is not before today ({today})")

        direction = ev.get("direction")
        if direction not in ("long", "short"):
            errors.append(f"{where}: direction must be 'long' or 'short', got {direction!r}")

        ts = {f: _parse_ts(ev.get(f), f, where, errors) for f in TS_FIELDS}
        if ts["entry_ts_pt"] is None and not ev.get("entry_ts_pt"):
            errors.append(f"{where}: entry_ts_pt is required")
        win_start = dt.datetime.combine(session_date - dt.timedelta(days=1), dt.time(15, 0))
        win_end = dt.datetime.combine(session_date, dt.time(13, 0))
        for f, v in ts.items():
            if v is not None and not (win_start <= v <= win_end):
                errors.append(f"{where}: {f} {v.isoformat()} is outside the session window {win_start} → {win_end} PT")

        meta = ev.get("meta") or {}
        if not isinstance(meta, dict):
            errors.append(f"{where}: meta must be an object")
            meta = {}
        bad = sorted(key for key in meta if OUTCOME_KEY_RE.match(str(key)))
        if bad:
            errors.append(f"{where}: meta carries outcome-like keys {bad}; move them to `outcome`")
        outcome = ev.get("outcome")
        if outcome is not None and not isinstance(outcome, dict):
            errors.append(f"{where}: outcome must be an object")

        floats = {f: _float(ev.get(f), f, where, errors) for f in FLOAT_FIELDS}
        rows.append({
            "id": cand_id,
            "cr_id": cr_id,
            "set_name": set_name,
            "session_date": session_date,
            "cell": None if ev.get("cell") is None else str(ev.get("cell")),
            "direction": direction,
            **floats,
            **ts,
            "meta": {**meta, "level_frame": level_frame},
            "outcome": outcome,
        })

    if errors:
        raise GalleryError("\n".join(errors))
    return rows


def insert_rows(conn, rows: list[dict]) -> tuple[int, int]:
    """Insert rows (ON CONFLICT DO NOTHING); return (inserted, skipped)."""
    from psycopg.types.json import Jsonb

    inserted = 0
    for r in rows:
        params = [Jsonb(r[c]) if c in ("meta", "outcome") and r[c] is not None else r[c] for c in COLUMNS]
        cur = conn.execute(INSERT_SQL, params)
        inserted += cur.rowcount or 0
    return inserted, len(rows) - inserted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gallery", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="validate only; no DB connection")
    args = ap.parse_args(argv)

    doc = json.loads(args.gallery.read_text())
    try:
        rows = validate_gallery(doc)
    except GalleryError as e:
        print(f"REJECTED {args.gallery}:\n{e}", file=sys.stderr)
        return 2
    print(f"valid: {len(rows)} events, cr_id={doc['cr_id']} set_name={doc['set_name']} level_frame={doc['level_frame']}")
    if args.dry_run:
        return 0

    from dotenv import load_dotenv
    from packages.shared.backfill_safety import (
        assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke,
    )

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    with backfill_run(conn, CR_ID) as run_id:
        with conn.transaction():
            inserted, skipped = insert_rows(conn, rows)
        n_set = conn.execute(
            "SELECT count(*) FROM setup_candidates WHERE cr_id = %s AND set_name = %s",
            (doc["cr_id"], doc["set_name"]),
        ).fetchone()[0]
        smoke = {"gallery": str(args.gallery), "events": len(rows), "inserted": inserted,
                 "skipped_existing": skipped, "rows_in_set": n_set}
        update_run_smoke(conn, run_id, smoke,
                         f"loaded {inserted} new / {skipped} existing candidates into {doc['cr_id']}:{doc['set_name']}")
    print(f"run {run_id}: inserted {inserted}, skipped {skipped} (already present); set now holds {n_set}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
