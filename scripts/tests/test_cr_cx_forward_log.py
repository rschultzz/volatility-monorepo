"""CR-CX (Y11) — forward log of G1-down: date guard, parity with the frozen CR-CL / CR-CO code, idempotency.

  TestGuard        no DB: every date < 2026-10-27 is refused before a connection is made.
  TestParity       DB + the CR-CJ y26 caches (scripts/logs/cr_cj/{skew,es}_y26.parquet): on 3 train sessions the
                   script's one-session loader and leg builder return exactly cr_cl_lib.vol_scaled_legs's legs
                   (in memory; nothing inserted). Also re-derives IM_TRAIN and TERCILE_CUTS.
  TestIdempotency  DB: the DDL on session-local pg_temp copies of the three tables (never the real ones; asserted
                   before any insert). One train session logged twice inserts n, then 0; UPDATE / DELETE raise.

Run with:
    CR_CL_PERIOD=y26 apps/web/.venv/bin/python -m unittest scripts.tests.test_cr_cx_forward_log -v
"""
from __future__ import annotations

import io
import os
import re
import sys
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

os.environ.setdefault("CR_CL_PERIOD", "y26")
SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import cr_cj_lib as L  # noqa: E402
import forward_log_g1down as F  # noqa: E402
import numpy as np  # noqa: E402

HAVE_DB = bool(os.environ.get("BACKFILL_DATABASE_URL"))
HAVE_CACHE = all((L.OUT_DIR / f"{k}_y26.parquet").exists() for k in ("skew", "es"))
OUTCOME_WORDS = ("race", "win", "loss", "move_", "mfe")


class TestGuard(unittest.TestCase):
    REFUSED = [date(2024, 5, 1), date(2026, 3, 2), date(2026, 6, 5), date(2026, 6, 8), date(2026, 9, 15),
               date(2026, 10, 26)]

    def test_refuses_train_and_holdout_without_db(self):
        boom = mock.Mock(side_effect=AssertionError("DB touched"))
        with mock.patch.object(F, "get_backfill_db_conn", boom), mock.patch.object(F, "code_sha", boom):
            for d in self.REFUSED:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = F.main(["--session", d.isoformat()])
                self.assertEqual(rc, 2, d)
                self.assertIn("REFUSED", buf.getvalue())
        boom.assert_not_called()

    def test_check_date(self):
        now = datetime(2026, 10, 28, 11, 0, tzinfo=timezone.utc)
        for d in self.REFUSED:
            with self.assertRaises(F.Refused):
                F.check_date(d, now)
        F.check_date(date(2026, 10, 27), now)                       # first forward session, available
        with self.assertRaises(F.Refused):
            F.check_date(date(2026, 10, 28), now)                   # same day: not yet available
        with self.assertRaises(F.Refused):
            F.check_date(date(2026, 10, 31), now)                   # Saturday

    def test_default_targets_never_before_start(self):
        now = datetime(2026, 11, 3, 11, 0, tzinfo=timezone.utc)
        t = F.default_targets(set(), now)
        self.assertEqual(t[0], F.FORWARD_START)
        self.assertTrue(all(d >= F.FORWARD_START for d in t))
        self.assertEqual(t[-1], date(2026, 11, 2))
        self.assertNotIn(date(2026, 10, 27), F.default_targets({date(2026, 10, 27)}, now))


def _train_conn():
    from packages.shared.backfill_safety import get_backfill_db_conn
    c = get_backfill_db_conn()
    return c


@unittest.skipUnless(HAVE_DB and HAVE_CACHE, "needs BACKFILL_DATABASE_URL and the CR-CJ y26 caches")
class TestParity(unittest.TestCase):
    KEYS = ["s", "e", "ci", "dir", "grp", "start_px", "end_px", "size", "p0", "cbucket", "edge", "ratio", "actual",
            "expected", "band", "beyond"]

    @classmethod
    def setUpClass(cls):
        import cr_cl_lib as C
        assert C.PERIOD == "y26"
        cls.C = C
        D = C.load()
        cls.ref, cls.im_train, _ = C.vol_scaled_legs(D)
        prim = cls.ref[(~cls.ref["late"]) & (cls.ref["reason"] == "ok")]
        g1d = sorted(prim[(prim["grp"] == "G1") & (prim["dir"] == -1)]["d"].unique())
        cls.days = [g1d[0], g1d[len(g1d) // 2], g1d[-1]]            # 3 train sessions with a G1-down event
        cls.conn = _train_conn()
        cls.conn.read_only = True

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_im_train_and_tercile_cuts_rederived(self):
        self.assertAlmostEqual(self.im_train, F.IM_TRAIN, places=9)
        _, cuts = self.C.add_terciles(self.ref)
        self.assertEqual(tuple(cuts), F.TERCILE_CUTS)

    def test_reproduces_cr_cl_legs_on_three_train_sessions(self):
        self.assertTrue(all(d <= date(2026, 6, 5) for d in self.days))
        for d in self.days:
            D = F.load_session(self.conn, d)
            self.assertIsNone(F.skip_reason(D, d), d)
            W, Q, _ = F.session_legs(D, d)
            R = self.ref[self.ref["d"] == d]
            self.assertEqual(len(W), len(R), d)
            got = W[self.KEYS].reset_index(drop=True)
            want = R[self.KEYS].reset_index(drop=True)
            for k in self.KEYS:
                a, b = got[k].to_numpy(), want[k].to_numpy()
                if a.dtype.kind == "f" or b.dtype.kind == "f":
                    np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, err_msg=f"{d} {k}")
                else:
                    self.assertEqual(list(a), list(b), f"{d} {k}")
            refq = R[(~R["late"]) & (R["reason"] == "ok")]
            self.assertEqual(list(Q["cell"] == "G1-down"),
                             list((refq["grp"] == "G1") & (refq["dir"] == -1)), d)
            self.assertGreater(int((Q["cell"] == "G1-down").sum()), 0, d)


DDL = SCRIPTS / "cr_cx_ddl.sql"


def temp_ddl() -> str:
    """The DDL rewritten onto pg_temp: temp tables, a pg_temp trigger function, no grants, no transaction."""
    sql = DDL.read_text()
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = sql.replace("CREATE TABLE IF NOT EXISTS", "CREATE TEMP TABLE")
    sql = sql.replace("FUNCTION forward_log_guard()", "FUNCTION pg_temp.forward_log_guard()")
    sql = re.sub(r"REVOKE ALL.*?;|GRANT .*?;|BEGIN;|COMMIT;", "", sql, flags=re.S)
    return sql


@unittest.skipUnless(HAVE_DB, "needs BACKFILL_DATABASE_URL")
class TestIdempotency(unittest.TestCase):
    DAY = date(2026, 3, 16)            # a y26 train session (holds a G1-down leg in the CR-CL set)

    def setUp(self):                                               # fresh temp tables per test
        self.conn = _train_conn()
        self.conn.execute(temp_ddl())
        for t in ("forward_log_sessions", "forward_log_events", "forward_log_outcomes"):
            ns = self.conn.execute("""SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                                      WHERE c.oid = to_regclass(%s)""", (t,)).fetchone()[0]
            assert ns.startswith("pg_temp"), (t, ns)               # never the real tables

    def tearDown(self):
        self.conn.close()                                           # drops the temp tables

    def test_second_run_inserts_nothing(self):
        D = F.load_session(self.conn, self.DAY)
        r1 = F.log_session(self.conn, self.DAY, D, "test-sha", None)
        self.assertEqual(r1["status"], "logged")
        self.assertGreater(r1["legs_inserted"], 0)
        self.assertGreater(r1["outcomes_inserted"], 0)
        self.assertEqual(r1["session_row_inserted"], 1)
        n_ev = self.conn.execute("SELECT count(*) FROM forward_log_events").fetchone()[0]
        n_elig = self.conn.execute("SELECT count(*) FROM forward_log_events WHERE signal = ANY(%s)",
                                   (list(F.OUTCOME_CELLS),)).fetchone()[0]
        self.assertEqual(n_ev, r1["legs_inserted"])
        self.assertEqual(n_elig, r1["outcomes_inserted"])
        r2 = F.log_session(self.conn, self.DAY, F.load_session(self.conn, self.DAY), "test-sha", None)
        self.assertEqual((r2["legs_inserted"], r2["outcomes_inserted"], r2["session_row_inserted"]), (0, 0, 0))
        self.assertEqual(F.catch_up_outcomes(self.conn, "test-sha", None), 0)
        # counts only (B4): nothing outcome-like is returned for printing
        for r in (r1, r2):
            flat = repr(r).lower()
            self.assertFalse(any(w in flat for w in OUTCOME_WORDS), r)

    def test_insert_only(self):
        import psycopg
        D = F.load_session(self.conn, self.DAY)
        F.log_session(self.conn, self.DAY, D, "test-sha", None)
        for sql in ("UPDATE forward_log_events SET entry_price = entry_price + 1",
                    "DELETE FROM forward_log_outcomes",
                    "UPDATE forward_log_sessions SET status = 'skipped'"):
            with self.assertRaises(psycopg.errors.RaiseException, msg=sql):
                self.conn.execute(sql)

    def test_incomplete_session_is_skipped(self):
        D = F.load_session(self.conn, self.DAY)
        D["BARS"][self.DAY]["h"][100] = np.nan                       # one missing RTH minute
        self.assertEqual(F.skip_reason(D, self.DAY), "incomplete ES bars (389/390)")


if __name__ == "__main__":
    unittest.main()
