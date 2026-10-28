"""CR-CC step 2 — /api/setup-review/* (L4, L8, L9, A1.5).

Two layers:
  * Route tests with the connection mocked (no DB): error mapping, input checks.
  * TestAgainstDb (opt-in, CR_CC_DB_TESTS=1): the real SQL against DATABASE_URL on ONE
    connection whose commit/close are no-ops and which is rolled back at the end. Since CR-DE
    (B8) the SQL runs against TEMP copies of setup_candidates / setup_reviews (with the
    insert-only guard trigger), which shadow the real tables on that connection; setUp
    asserts both resolve to temp tables, so no test row is ever written to the real ones.

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.SetupReview.tests.test_routes
    CR_CC_DB_TESTS=1 apps/web/.venv/bin/python -m unittest apps.web.modules.SetupReview.tests.test_routes
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flask
import psycopg

from apps.web.modules.SetupReview import routes, service

P = "/api/setup-review"


def _app():
    app = flask.Flask(__name__)
    routes.register_setup_review_routes(app)
    return app.test_client()


class TestRoutesMocked(unittest.TestCase):
    def setUp(self):
        self.client = _app()
        self.conn = MagicMock()
        p = patch.object(routes, "_conn", return_value=self.conn)
        p.start()
        self.addCleanup(p.stop)

    def test_permission_error_is_403_not_a_crash(self):
        with patch.object(service, "list_candidates",
                          side_effect=psycopg.errors.InsufficientPrivilege("permission denied for table setup_candidates")), \
                patch.object(service, "current_role", return_value="some_web_role"):
            r = self.client.get(f"{P}/candidates")
        self.assertEqual(r.status_code, 403)
        body = r.get_json()
        self.assertEqual(body["error"], "permission")
        self.assertEqual(body["role"], "some_web_role")
        self.assertIn("setup review tables", body["message"])
        self.assertNotIn("postgresql://", json.dumps(body))

    def test_known_reasons_include_cr_cz_tags(self):
        cur = MagicMock()
        cur.fetchall.return_value = [("trend day",), ("my custom",)]
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        got = service.known_reasons(conn)
        self.assertEqual(got[:len(service.DEFAULT_REASONS)], list(service.DEFAULT_REASONS))
        self.assertEqual(got[-1], "my custom")
        i = got.index("take continuation (other edge)")
        self.assertEqual(got[i + 1:i + 3], ["take fade (called side)", "other"])

    def test_missing_tables_is_503(self):
        with patch.object(service, "summary", side_effect=psycopg.errors.UndefinedTable("relation does not exist")):
            r = self.client.get(f"{P}/summary")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json()["error"], "missing_tables")

    def test_guard_trigger_is_409(self):
        with patch.object(service, "reveal", side_effect=psycopg.errors.RaiseException("setup_reviews is insert-only")):
            r = self.client.post(f"{P}/candidates/a:b:1/reveal")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.get_json()["error"], "insert_only")

    def test_known_reasons_skip_fixture_sets(self):
        cur = MagicMock()
        cur.fetchall.return_value = []
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        service.known_reasons(conn)
        sql, params = cur.execute.call_args.args
        self.assertIn("NOT LIKE", sql)
        self.assertEqual(params, ("%:fixture",))

    def test_label_locked_is_409_with_code(self):
        with patch.object(service, "add_review", side_effect=service.ReviewError("locked", 409, "label_locked")):
            r = self.client.post(f"{P}/candidates/a:b:1/review", json={"label": "take"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.get_json()["error"], "label_locked")
        self.conn.commit.assert_not_called()

    def test_bad_status_and_label(self):
        self.assertEqual(self.client.get(f"{P}/candidates?status=nope").status_code, 400)
        r = self.client.post(f"{P}/candidates/a:b:1/review", json={"label": "maybe"})
        self.assertEqual(r.status_code, 400)
        self.conn.commit.assert_not_called()

    def test_candidate_id_with_colons_routes_to_detail(self):
        with patch.object(service, "get_candidate", return_value={"id": "CR-CC:cc:fixture:3"}) as g:
            r = self.client.get(f"{P}/candidates/CR-CC%3Acc%3Afixture%3A3")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(g.call_args.args[1], "CR-CC:cc:fixture:3")

    def test_write_routes_commit(self):
        with patch.object(service, "add_review", return_value={"id": 1}):
            r = self.client.post(f"{P}/candidates/a:b:1/review", json={"label": "take"})
        self.assertEqual(r.status_code, 200)
        self.conn.commit.assert_called_once()


SECRET = "s3cretPw0rd"


class TestRedaction(unittest.TestCase):
    """CR-DE R3: no DB error that is returned or logged carries the connection string."""

    def test_redact_forms(self):
        from packages.shared.redact import redact
        msg = (f'invalid connection option "postgresql+psycopg://user:{SECRET}@db.example.com/x?sslmode" '
               f"and postgres://u:{SECRET}@h:5432/d and host=h password={SECRET} dbname=d")
        out = redact(msg)
        self.assertNotIn(SECRET, out)
        self.assertIn("<postgres-url>", out)

    def test_redact_env_value(self):
        from packages.shared.redact import redact
        url = f"postgresql+psycopg://user:{SECRET}@db.example.com/x"
        with patch.dict(os.environ, {"DATABASE_URL": url}):
            self.assertEqual(redact(f"bad: {url}"), "bad: <DATABASE_URL>")

    def _assert_clean(self, r, logs):
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.get_json()["error"], "db_connect")
        self.assertNotIn(SECRET, r.get_data(as_text=True))
        self.assertNotIn(SECRET, "\n".join(logs.output))

    def test_failing_connection_output_has_no_password(self):
        # a real psycopg connect that fails: nothing listens on port 1
        url = f"postgresql://dash_user:{SECRET}@127.0.0.1:1/curve?connect_timeout=2"
        with patch.dict(os.environ, {"DATABASE_URL": url}), self.assertLogs(routes.log, "ERROR") as logs:
            r = _app().get(f"{P}/candidates")
        self._assert_clean(r, logs)

    def test_malformed_url_error_has_no_password(self):
        # psycopg echoes the whole conninfo when it cannot parse it (the Step 0 leak)
        url = f"postgresql+psycopg://dash_user:{SECRET}@db.example.com/curve"
        with patch.dict(os.environ, {"DATABASE_URL": url}), \
                patch.object(routes, "_normalize_db_url", side_effect=lambda u: u), \
                self.assertLogs(routes.log, "ERROR") as logs:
            r = _app().get(f"{P}/candidates")
        self._assert_clean(r, logs)

    def test_query_error_message_has_no_password(self):
        conn = MagicMock()
        err = psycopg.OperationalError(f"server closed: postgresql://u:{SECRET}@h/d")
        with patch.object(routes, "_conn", return_value=conn), \
                patch.object(service, "list_candidates", side_effect=err), \
                self.assertLogs(routes.log, "ERROR") as logs:
            r = _app().get(f"{P}/candidates")
        self.assertEqual(r.status_code, 500)
        self.assertNotIn(SECRET, r.get_data(as_text=True))
        self.assertNotIn(SECRET, "\n".join(logs.output))


# TEMP copies of the CR-CC tables (scripts/cr_cc_ddl.sql) that shadow the real ones on the
# test connection (pg_temp is searched first), with the same insert-only guard trigger.
_TEMP_DDL = """
CREATE TEMP TABLE setup_candidates (
    id text PRIMARY KEY, cr_id text NOT NULL, set_name text NOT NULL, session_date date NOT NULL,
    cell text, direction text NOT NULL CHECK (direction IN ('long', 'short')),
    level_price double precision, level_net_bn double precision, level_gross_bn double precision,
    entry_ts_pt timestamp without time zone NOT NULL, entry_price double precision,
    stop_price double precision, extreme_price double precision,
    test_ts_pt timestamp without time zone, confirm_ts_pt timestamp without time zone,
    retest_ts_pt timestamp without time zone, meta jsonb NOT NULL DEFAULT '{}'::jsonb,
    outcome jsonb, loaded_at timestamptz NOT NULL DEFAULT now()
) ON COMMIT DROP;
CREATE TEMP TABLE setup_reviews (
    id bigserial PRIMARY KEY, candidate_id text NOT NULL REFERENCES setup_candidates (id),
    label text NOT NULL CHECK (label IN ('take', 'skip', 'unsure')),
    reasons text[] NOT NULL DEFAULT '{}', comment text,
    reviewed_at timestamptz NOT NULL DEFAULT clock_timestamp(), revealed_at timestamptz,
    post_reveal boolean NOT NULL DEFAULT false
) ON COMMIT DROP;
CREATE TRIGGER setup_reviews_guard BEFORE UPDATE OR DELETE ON setup_reviews
    FOR EACH ROW EXECUTE FUNCTION public.setup_review_guard();
"""


class _NoCommit:
    """Wrap a psycopg connection so the routes cannot commit or close it."""

    def __init__(self, conn):
        self._c = conn

    def commit(self):
        pass

    def close(self):
        pass

    def rollback(self):
        pass

    def __getattr__(self, name):
        return getattr(self._c, name)


def _db_url():
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            return routes._normalize_db_url(line.split("=", 1)[1].strip())
    raise unittest.SkipTest("DATABASE_URL not in .env")


@unittest.skipUnless(os.getenv("CR_CC_DB_TESTS") == "1", "set CR_CC_DB_TESTS=1 to run against the DB (rolled back)")
class TestAgainstDb(unittest.TestCase):
    CR, SET = "CR-CC-TEST", "rollback"

    def setUp(self):
        self.raw = psycopg.connect(_db_url())
        self.addCleanup(self.raw.close)
        self.addCleanup(self.raw.rollback)
        self.raw.execute(_TEMP_DDL)
        for t in ("setup_candidates", "setup_reviews"):   # B8: the SQL below must hit the temp copies
            persistence = self.raw.execute(
                "SELECT relpersistence FROM pg_class WHERE oid = %s::regclass", (t,)).fetchone()[0]
            self.assertEqual(persistence, "t", f"{t} does not resolve to a temp table")
        for n, (d, direction, pnl) in enumerate([("2026-03-10", "long", 12.5), ("2026-03-11", "short", -3.5),
                                                  ("2026-03-12", "long", 7.0)], start=1):
            self.raw.execute(
                "INSERT INTO setup_candidates (id, cr_id, set_name, session_date, cell, direction, "
                "level_price, entry_ts_pt, entry_price, stop_price, meta, outcome) "
                "VALUES (%s, %s, %s, %s, 'c1', %s, 6700, %s, 6702, 6695, '{}', %s)",
                (f"{self.CR}:{self.SET}:{n}", self.CR, self.SET, d, direction, f"{d} 07:37:00",
                 json.dumps({"pnl_pts": pnl, "result": "x", "exit_price": 6710})),
            )
        self.client = _app()
        p = patch.object(routes, "_conn", return_value=_NoCommit(self.raw))
        p.start()
        self.addCleanup(p.stop)

    def cid(self, n):
        return f"{self.CR}:{self.SET}:{n}"

    def test_outcome_hidden_until_review_then_reveal(self):
        lst = self.client.get(f"{P}/candidates?cr={self.CR}&set={self.SET}").get_json()
        self.assertEqual(len(lst["candidates"]), 3)
        self.assertTrue(all("outcome" not in c for c in lst["candidates"]))
        self.assertNotIn("pnl_pts", json.dumps(lst))
        self.assertIn("trend day", lst["reasons"])
        for tag in ("take continuation (other edge)", "take fade (called side)"):   # CR-CZ
            self.assertIn(tag, lst["reasons"])

        det = self.client.get(f"{P}/candidates/{self.cid(1)}").get_json()["candidate"]
        self.assertNotIn("outcome", det)
        self.assertFalse(det["reviewed"])

        r = self.client.post(f"{P}/candidates/{self.cid(1)}/reveal")
        self.assertEqual(r.status_code, 409)  # save is required before reveal

        rv = self.client.post(f"{P}/candidates/{self.cid(1)}/review",
                              json={"label": "take", "reasons": ["clean retest", "clean retest"], "comment": " ok "})
        self.assertEqual(rv.status_code, 200)
        review = rv.get_json()["review"]
        self.assertEqual(review["reasons"], ["clean retest"])
        self.assertFalse(review["post_reveal"])

        rev = self.client.post(f"{P}/candidates/{self.cid(1)}/reveal").get_json()
        self.assertEqual(rev["outcome"]["pnl_pts"], 12.5)
        again = self.client.post(f"{P}/candidates/{self.cid(1)}/reveal").get_json()
        self.assertEqual(again["revealed_at"], rev["revealed_at"])  # idempotent

        post = self.client.post(f"{P}/candidates/{self.cid(1)}/review", json={"label": "skip"}).get_json()["review"]
        self.assertTrue(post["post_reveal"])   # a changed label alone is a valid note

        lst = self.client.get(f"{P}/candidates?cr={self.CR}&set={self.SET}&status=unreviewed").get_json()
        self.assertEqual([c["id"] for c in lst["candidates"]], [self.cid(2), self.cid(3)])
        self.assertIn("clean retest", lst["reasons"])

    def test_summary_take_vs_skip(self):
        for n, label, reasons in ((1, "take", []), (2, "skip", ["chop", "trend day"]), (3, "skip", ["chop"])):
            self.client.post(f"{P}/candidates/{self.cid(n)}/review", json={"label": label, "reasons": reasons})
        for n in (1, 2):
            self.client.post(f"{P}/candidates/{self.cid(n)}/reveal")
        # a post-reveal relabel must not move the stats
        self.client.post(f"{P}/candidates/{self.cid(2)}/review", json={"label": "take"})

        g = self.client.get(f"{P}/summary?cr={self.CR}&set={self.SET}").get_json()["groups"][0]
        self.assertEqual((g["n_candidates"], g["n_reviewed"], g["n_revealed"], g["unreviewed"]), (3, 3, 2, 0))
        self.assertEqual(g["by_label"]["take"], {"n": 1, "n_revealed": 1, "n_outcome": 1, "mean_pnl_pts": 12.5, "n_win": 1})
        # skip: 2 labelled, 1 revealed (-3.5); the unrevealed one's outcome is not averaged
        self.assertEqual(g["by_label"]["skip"], {"n": 2, "n_revealed": 1, "n_outcome": 1, "mean_pnl_pts": -3.5, "n_win": 0})
        self.assertEqual(g["skip_reasons"], [{"reason": "chop", "n": 2}, {"reason": "trend day", "n": 1}])

    # ---- CR-DE ----

    def test_label_locks_after_first_pre_reveal_save(self):
        ok = self.client.post(f"{P}/candidates/{self.cid(1)}/review", json={"label": "take", "comment": "first"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.get_json()["review"]["comment"], "first")
        again = self.client.post(f"{P}/candidates/{self.cid(1)}/review", json={"label": "skip"})
        self.assertEqual(again.status_code, 409)
        self.assertEqual(again.get_json()["error"], "label_locked")
        det = self.client.get(f"{P}/candidates/{self.cid(1)}").get_json()["candidate"]
        self.assertEqual(det["label"], "take")
        self.assertEqual(len(det["reviews"]), 1)

    def test_blind_label_is_first_pre_reveal_row(self):
        # rows written directly (bypassing the lock) to pin the reader rule: first wins
        for label in ("skip", "take"):
            self.raw.execute("INSERT INTO setup_reviews (candidate_id, label) VALUES (%s, %s)", (self.cid(1), label))
        lst = self.client.get(f"{P}/candidates?cr={self.CR}&set={self.SET}").get_json()["candidates"]
        self.assertEqual(lst[0]["label"], "skip")
        self.client.post(f"{P}/candidates/{self.cid(1)}/reveal")
        rows = self.raw.execute("SELECT label, revealed_at IS NOT NULL FROM setup_reviews ORDER BY id").fetchall()
        self.assertEqual(rows, [("skip", True), ("take", False)])   # reveal stamps the blind-label row
        g = self.client.get(f"{P}/summary?cr={self.CR}&set={self.SET}").get_json()["groups"][0]
        self.assertEqual(g["by_label"]["skip"]["n"], 1)
        self.assertNotIn("take", g["by_label"])

    def test_post_reveal_notes_insert_and_copy_blind_label(self):
        self.client.post(f"{P}/candidates/{self.cid(2)}/review", json={"label": "skip", "reasons": ["chop"]})
        self.client.post(f"{P}/candidates/{self.cid(2)}/reveal")
        n1 = self.client.post(f"{P}/candidates/{self.cid(2)}/review", json={"comment": "note one"})
        n2 = self.client.post(f"{P}/candidates/{self.cid(2)}/review",
                              json={"comment": "note two", "reasons": ["trend day"]})
        n3 = self.client.post(f"{P}/candidates/{self.cid(2)}/review", json={"label": "take", "comment": "changed"})
        for r in (n1, n2, n3):
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()["review"]["post_reveal"])
        self.assertEqual(n1.get_json()["review"]["label"], "skip")   # R1: copies the blind label
        self.assertEqual(n2.get_json()["review"]["reasons"], ["trend day"])
        self.assertEqual(n3.get_json()["review"]["label"], "take")
        empty = self.client.post(f"{P}/candidates/{self.cid(2)}/review", json={"comment": "  "})
        self.assertEqual(empty.status_code, 400)
        same = self.client.post(f"{P}/candidates/{self.cid(2)}/review", json={"label": "skip"})
        self.assertEqual(same.status_code, 400)   # not a changed label and no text
        det = self.client.get(f"{P}/candidates/{self.cid(2)}").get_json()["candidate"]
        self.assertEqual(det["label"], "skip")
        self.assertEqual([r["comment"] for r in det["reviews"]], [None, "note one", "note two", "changed"])
        n = self.raw.execute("SELECT count(*) FROM setup_reviews WHERE candidate_id = %s", (self.cid(2),)).fetchone()[0]
        self.assertEqual(n, 4)
        g = self.client.get(f"{P}/summary?cr={self.CR}&set={self.SET}").get_json()["groups"][0]
        self.assertEqual(g["by_label"]["skip"]["n"], 1)
        self.assertNotIn("take", g["by_label"])
        with self.assertRaises(psycopg.errors.RaiseException):   # still insert-only
            self.raw.execute("UPDATE setup_reviews SET comment = 'x' WHERE candidate_id = %s", (self.cid(2),))

    def test_fixture_set_reasons_are_not_chips(self):
        self.raw.execute(
            "INSERT INTO setup_candidates (id, cr_id, set_name, session_date, direction, entry_ts_pt) "
            "VALUES ('CR-CC:cc:fixture:2', 'CR-CC', 'cc:fixture', '2026-03-10', 'long', '2026-03-10 07:37')")
        self.raw.execute("INSERT INTO setup_reviews (candidate_id, label, reasons) "
                         "VALUES ('CR-CC:cc:fixture:2', 'skip', '{fixture smoke reason}')")
        self.client.post(f"{P}/candidates/{self.cid(1)}/review", json={"label": "skip", "reasons": ["mine"]})
        reasons = self.client.get(f"{P}/candidates").get_json()["reasons"]
        self.assertNotIn("fixture smoke reason", reasons)
        self.assertIn("mine", reasons)

    def test_sets_carry_loaded_at(self):
        sets = self.client.get(f"{P}/candidates").get_json()["sets"]
        mine = [x for x in sets if x["set_name"] == self.SET][0]
        self.assertTrue(mine["loaded_at"])


if __name__ == "__main__":
    unittest.main()
