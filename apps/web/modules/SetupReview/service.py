"""SetupReview service — SQL for the CR-CC review queue (L4, L8, L9).

Every function takes an open psycopg connection and never commits; the route layer
owns the transaction. `outcome` leaves the database only through `get_candidate`
(when the candidate has at least one review), `reveal` (which requires one) and
`summary` (revealed candidates only). `list_candidates` never selects it.

Label semantics (CR-DE): a candidate's blind label is its FIRST pre-reveal review
(post_reveal = false; earliest reviewed_at, then lowest id). Any code that reads
labels for analysis must use that row. The label locks once it is saved: a second
pre-reveal review is rejected (409 label_locked). A candidate is revealed when any
of its reviews carries revealed_at. Rows written after reveal are notes only: each
POST inserts a post_reveal row (comment, reasons, optional changed label; with no
new label the row copies the blind label, since label is NOT NULL). Post-reveal
rows never move the Take / Skip / Unsure outcome stats.
"""
from __future__ import annotations

import datetime as dt

from psycopg.rows import dict_row

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION

LABELS = ("take", "skip", "unsure")
DEFAULT_REASONS = (
    "overshoot too big",
    "no clean extreme",
    "level too weak",
    "trend day",
    "chop",
    "too late in day",
    "flow against",
    "take continuation (other edge)",   # CR-CZ
    "take fade (called side)",          # CR-CZ
    "other",
)
STATUSES = ("unreviewed", "reviewed", "all")
FIXTURE_SET_SUFFIX = ":fixture"   # CR-CC smoke sets; their reasons are not offered as chips

_CANDIDATE_COLS = (
    "c.id, c.cr_id, c.set_name, c.session_date, c.cell, c.direction, "
    "c.level_price, c.level_net_bn, c.level_gross_bn, c.entry_ts_pt, c.entry_price, "
    "c.stop_price, c.extreme_price, c.test_ts_pt, c.confirm_ts_pt, c.retest_ts_pt, c.meta"
)

# Review state per candidate: first pre-reveal label (the blind label), review count, revealed flag.
_STATE_LATERAL = """
    LEFT JOIN LATERAL (
        SELECT
            count(*)                                   AS n_reviews,
            bool_or(r.revealed_at IS NOT NULL)         AS revealed,
            max(r.revealed_at)                         AS revealed_at,
            (array_agg(r.label ORDER BY r.reviewed_at, r.id)
                FILTER (WHERE NOT r.post_reveal))[1]   AS label
        FROM setup_reviews r
        WHERE r.candidate_id = c.id
    ) s ON true
"""


def _iso(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v


def _clean(row: dict) -> dict:
    return {k: _iso(v) for k, v in row.items()}


def _state(row: dict) -> dict:
    n = int(row.pop("n_reviews") or 0)
    row["reviewed"] = n > 0
    row["revealed"] = bool(row.get("revealed"))
    row["n_reviews"] = n
    return row


def list_candidates(conn, *, cr=None, set_name=None, cell=None, status="all") -> list[dict]:
    """Candidates without outcomes, ordered by session and entry time."""
    where, params = [], []
    if cr:
        where.append("c.cr_id = %s")
        params.append(cr)
    if set_name:
        where.append("c.set_name = %s")
        params.append(set_name)
    if cell:
        where.append("c.cell = %s")
        params.append(cell)
    if status == "unreviewed":
        where.append("COALESCE(s.n_reviews, 0) = 0")
    elif status == "reviewed":
        where.append("COALESCE(s.n_reviews, 0) > 0")
    sql = (
        f"SELECT {_CANDIDATE_COLS}, s.n_reviews, s.revealed, s.revealed_at, s.label "
        f"FROM setup_candidates c {_STATE_LATERAL} "
        + (f"WHERE {' AND '.join(where)} " if where else "")
        + "ORDER BY c.session_date, c.entry_ts_pt, c.id"
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return [_clean(_state(dict(r))) for r in cur.fetchall()]


def list_sets(conn) -> list[dict]:
    """(cr_id, set_name) pairs with candidate counts and latest loaded_at, for the panel's filters
    (CR-DE: the panel defaults to the most recently loaded set)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT cr_id, set_name, count(*) AS n, array_agg(DISTINCT cell) "
            "FILTER (WHERE cell IS NOT NULL) AS cells, max(loaded_at) AS loaded_at "
            "FROM setup_candidates GROUP BY cr_id, set_name ORDER BY cr_id, set_name"
        )
        return [_clean(dict(r)) for r in cur.fetchall()]


def known_reasons(conn) -> list[str]:
    """The default chips plus every reason used in a saved review (L7), except reviews on
    fixture sets (CR-DE: smoke-test reasons such as "fixture smoke reason" are not chips)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT unnest(r.reasons) FROM setup_reviews r "
            "JOIN setup_candidates c ON c.id = r.candidate_id WHERE c.set_name NOT LIKE %s",
            (f"%{FIXTURE_SET_SUFFIX}",),
        )
        used = {r[0] for r in cur.fetchall() if r[0]}
    extra = sorted(used - set(DEFAULT_REASONS))
    return list(DEFAULT_REASONS) + extra


def _reviews(conn, candidate_id: str) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, label, reasons, comment, reviewed_at, revealed_at, post_reveal "
            "FROM setup_reviews WHERE candidate_id = %s ORDER BY reviewed_at, id",
            (candidate_id,),
        )
        return [_clean(dict(r)) for r in cur.fetchall()]


def get_candidate(conn, candidate_id: str) -> dict | None:
    """Detail; `outcome` is present only when the candidate has a review (L4)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_CANDIDATE_COLS}, s.n_reviews, s.revealed, s.revealed_at, s.label, "
            "CASE WHEN COALESCE(s.n_reviews, 0) > 0 THEN c.outcome END AS outcome "
            f"FROM setup_candidates c {_STATE_LATERAL} WHERE c.id = %s",
            (candidate_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    out = _clean(_state(dict(row)))
    if not out["reviewed"]:
        out.pop("outcome", None)
    out["reviews"] = _reviews(conn, candidate_id)
    out["implied_move_1d"] = implied_move(conn, row["session_date"], row["entry_ts_pt"])
    return out


IM_IV_TIME_PT = dt.time(6, 33)


def implied_move(conn, session_date, entry_ts_pt) -> float | None:
    """CR-CD A3: the session's canonical `implied_move_1d` (bt_daily_features, SPX points), used to
    size the replay's price-axis fit. None when the entry is before 06:33 PT (the IV print it is
    built from comes after the cutoff; CR-CC A1.3) or when there is no canonical row."""
    if isinstance(entry_ts_pt, dt.datetime) and entry_ts_pt.time() < IM_IV_TIME_PT:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT (feature_vector->>'implied_move_1d')::float FROM bt_daily_features "
            "WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date = %s LIMIT 1",
            (CANONICAL_FEATURE_VERSION, session_date),
        )
        r = cur.fetchone()
    v = r[0] if r else None
    return float(v) if v is not None and v > 0 else None


class ReviewError(ValueError):
    """Bad review input or a state conflict; `status` is the HTTP status."""

    def __init__(self, message: str, status: int = 400, code: str = "bad_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def add_review(conn, candidate_id: str, body: dict) -> dict:
    """Insert a review (insert-only; nothing is updated).

    Before reveal: the blind label, `label` required, first save only (409 label_locked after).
    After reveal: a note, flagged post_reveal; `label` optional and copies the blind label when
    absent (R1); a note needs a comment, a reason or a changed label.
    """
    label = str(body.get("label") or "").strip().lower() or None
    if label is not None and label not in LABELS:
        raise ReviewError(f"label must be one of {LABELS}")
    reasons_raw = body.get("reasons") or []
    if not isinstance(reasons_raw, list):
        raise ReviewError("reasons must be a list of strings")
    reasons, seen = [], set()
    for r in reasons_raw:
        r = str(r).strip()
        if r and r.lower() not in seen:
            seen.add(r.lower())
            reasons.append(r[:80])
    comment = body.get("comment")
    comment = (str(comment).strip()[:4000] or None) if comment is not None else None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM setup_candidates WHERE id = %s) AS found, "
            "EXISTS (SELECT 1 FROM setup_reviews WHERE candidate_id = %s AND revealed_at IS NOT NULL) AS revealed, "
            "(SELECT label FROM setup_reviews WHERE candidate_id = %s AND NOT post_reveal "
            " ORDER BY reviewed_at, id LIMIT 1) AS blind_label",
            (candidate_id, candidate_id, candidate_id),
        )
        st = cur.fetchone()
        if not st["found"]:
            raise ReviewError("candidate not found", 404)
        revealed = bool(st["revealed"])
        if not revealed:
            if st["blind_label"] is not None:
                raise ReviewError("label is locked after the first save; reveal to add notes", 409, "label_locked")
            if label is None:
                raise ReviewError(f"label must be one of {LABELS}")
        else:
            if label is None or label == st["blind_label"]:
                if not comment and not reasons:
                    raise ReviewError("a note needs a comment, a reason or a changed label")
                label = label or st["blind_label"]
            if label is None:
                raise ReviewError(f"label must be one of {LABELS}")
        cur.execute(
            "INSERT INTO setup_reviews (candidate_id, label, reasons, comment, post_reveal) "
            "VALUES (%s, %s, %s, %s, %s) "
            "RETURNING id, label, reasons, comment, reviewed_at, revealed_at, post_reveal",
            (candidate_id, label, reasons, comment, revealed),
        )
        return _clean(dict(cur.fetchone()))


def reveal(conn, candidate_id: str) -> dict:
    """Set revealed_at on the blind-label (first pre-reveal) review and return the outcome (L8).

    Requires a saved review. Idempotent: a revealed candidate returns its outcome and
    the original revealed_at without another UPDATE.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT outcome FROM setup_candidates WHERE id = %s", (candidate_id,))
        cand = cur.fetchone()
        if cand is None:
            raise ReviewError("candidate not found", 404)
        cur.execute(
            "SELECT max(revealed_at) AS revealed_at FROM setup_reviews WHERE candidate_id = %s",
            (candidate_id,),
        )
        revealed_at = cur.fetchone()["revealed_at"]
        if revealed_at is None:
            cur.execute(
                "SELECT id FROM setup_reviews WHERE candidate_id = %s AND NOT post_reveal "
                "ORDER BY reviewed_at, id LIMIT 1",
                (candidate_id,),
            )
            first = cur.fetchone()
            if first is None:
                raise ReviewError("save a review before reveal", 409)
            cur.execute(
                "UPDATE setup_reviews SET revealed_at = now() WHERE id = %s RETURNING revealed_at",
                (first["id"],),
            )
            revealed_at = cur.fetchone()["revealed_at"]
    return {"id": candidate_id, "revealed_at": _iso(revealed_at), "outcome": cand["outcome"]}


def summary(conn, *, cr=None, set_name=None) -> list[dict]:
    """Per CR/set: counts by label; for revealed candidates the mean outcome
    (outcome.pnl_pts) for Take / Skip / Unsure with n; reasons tally on Skips (L9)."""
    where, params = [], []
    if cr:
        where.append("c.cr_id = %s")
        params.append(cr)
    if set_name:
        where.append("c.set_name = %s")
        params.append(set_name)
    sql = f"""
        WITH st AS (
            SELECT c.cr_id, c.set_name, c.id, s.label, COALESCE(s.revealed, false) AS revealed,
                   CASE WHEN s.revealed THEN (c.outcome->>'pnl_pts')::double precision END AS pnl,
                   (SELECT r.reasons FROM setup_reviews r
                     WHERE r.candidate_id = c.id AND NOT r.post_reveal
                     ORDER BY r.reviewed_at, r.id LIMIT 1) AS reasons
            FROM setup_candidates c {_STATE_LATERAL}
            {('WHERE ' + ' AND '.join(where)) if where else ''}
        )
        SELECT cr_id, set_name, label, count(*) AS n,
               count(*) FILTER (WHERE revealed) AS n_revealed,
               count(pnl) AS n_outcome,
               avg(pnl) AS mean_pnl_pts,
               count(*) FILTER (WHERE revealed AND pnl > 0) AS n_win
        FROM st GROUP BY cr_id, set_name, label ORDER BY cr_id, set_name, label NULLS FIRST
    """
    reasons_sql = f"""
        WITH st AS (
            SELECT c.cr_id, c.set_name, s.label,
                   (SELECT r.reasons FROM setup_reviews r
                     WHERE r.candidate_id = c.id AND NOT r.post_reveal
                     ORDER BY r.reviewed_at, r.id LIMIT 1) AS reasons
            FROM setup_candidates c {_STATE_LATERAL}
            {('WHERE ' + ' AND '.join(where)) if where else ''}
        )
        SELECT cr_id, set_name, reason, count(*) AS n
        FROM st, unnest(reasons) AS reason
        WHERE label = 'skip'
        GROUP BY cr_id, set_name, reason ORDER BY cr_id, set_name, n DESC, reason
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.execute(reasons_sql, params)
        reason_rows = cur.fetchall()

    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["cr_id"], r["set_name"])
        g = out.setdefault(key, {
            "cr_id": key[0], "set_name": key[1], "n_candidates": 0, "n_reviewed": 0,
            "n_revealed": 0, "by_label": {}, "skip_reasons": [],
        })
        n = int(r["n"])
        g["n_candidates"] += n
        g["n_revealed"] += int(r["n_revealed"])
        if r["label"] is None:
            g["unreviewed"] = n
            continue
        g["n_reviewed"] += n
        mean = r["mean_pnl_pts"]
        g["by_label"][r["label"]] = {
            "n": n,
            "n_revealed": int(r["n_revealed"]),
            "n_outcome": int(r["n_outcome"]),
            "mean_pnl_pts": None if mean is None else round(float(mean), 2),
            "n_win": int(r["n_win"]),
        }
    for r in reason_rows:
        g = out.get((r["cr_id"], r["set_name"]))
        if g is not None:
            g["skip_reasons"].append({"reason": r["reason"], "n": int(r["n"])})
    for g in out.values():
        g.setdefault("unreviewed", 0)
    return list(out.values())


def current_role(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT current_user")
        return cur.fetchone()[0]
