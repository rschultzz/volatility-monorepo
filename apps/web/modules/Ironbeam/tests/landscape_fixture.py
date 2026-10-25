"""Record/replay harness for /api/gex-landscape (CR-BV byte-identity test).

The route reads the DB only through `engine.connect()` → `conn.execute(sql, params)`.
`RecordingEngine` wraps a live (read-only) engine and stores every
(sql text, params) -> result rows; `ReplayEngine` serves the stored rows, so the
route can be exercised with no DB. Results are kept in a form that supports the
three access patterns the route and builder use: `.mappings().first()`,
`.mappings().all()` and `.fetchone()`.
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "cr_bv_landscape_2026-10-23.pkl.gz"
GOLDEN_PATH = FIXTURE_DIR / "cr_bv_landscape_golden.json"

# Requests the golden covers (query strings, without the route prefix). Captured
# from the unmodified route/builder (origin/main 0e943d8) before any CR-BV edit.
GOLDEN_QS = {
    "low_iv": "ticker=SPX&date=2026-10-23&spot=7660.75&iv=0.14",
    "low_noiv": "ticker=SPX&date=2026-10-23&spot=7660.75",
    "high_iv": "ticker=SPX&date=2026-10-23&spot=7660.75&iv=0.14&accuracy=high",
    "rth_open_iv": "ticker=SPX&date=2026-10-23&iv=0.14",
}
# Corrected-mode requests recorded for the offline Corrected tests.
CORRECTED_QS = {
    "corrected_iv": "ticker=SPX&date=2026-10-23&spot=7660.75&iv=0.14&levels=corrected",
}


FROZEN_NOW = "2026-10-25T12:00:00+00:00"


@contextmanager
def frozen_clock():
    """Pin the builder's `recomputed_at` (dt.datetime.now on the request-time path) so bodies compare."""
    import datetime as dt
    import types
    from unittest.mock import patch

    class _FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromisoformat(FROZEN_NOW).astimezone(tz)

    fake_dt = types.SimpleNamespace(date=dt.date, datetime=_FrozenDateTime,
                                    timezone=dt.timezone, timedelta=dt.timedelta)
    with patch("packages.shared.gex_landscape_api.dt", fake_dt):
        yield


def _key(sql, params) -> tuple:
    return (" ".join(str(sql).split()), tuple(sorted((params or {}).items())))


class _Result:
    def __init__(self, keys, rows):
        self._keys = list(keys)
        self._rows = [tuple(r) for r in rows]

    def mappings(self):
        return self

    def first(self):
        return dict(zip(self._keys, self._rows[0])) if self._rows else None

    def all(self):
        return [dict(zip(self._keys, r)) for r in self._rows]

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _RecordingConn:
    def __init__(self, conn, store):
        self._conn, self._store = conn, store

    def execute(self, sql, params=None):
        res = self._conn.execute(sql, params or {})
        keys = list(res.keys())
        rows = [tuple(r) for r in res.fetchall()]
        self._store[_key(sql, params)] = (keys, rows)
        return _Result(keys, rows)


class RecordingEngine:
    def __init__(self, engine):
        self._engine, self.store = engine, {}

    @contextmanager
    def connect(self):
        with self._engine.connect() as conn:
            yield _RecordingConn(conn, self.store)


class _ReplayConn:
    def __init__(self, store):
        self._store = store

    def execute(self, sql, params=None):
        k = _key(sql, params)
        if k not in self._store:
            raise KeyError(f"no recorded result for {k[0][:80]!r} {k[1]!r}")
        keys, rows = self._store[k]
        return _Result(keys, rows)


class ReplayEngine:
    def __init__(self, store):
        self.store = store

    @contextmanager
    def connect(self):
        yield _ReplayConn(self.store)


def load_store() -> dict:
    with gzip.open(FIXTURE_PATH, "rb") as fh:
        return pickle.load(fh)


def save_store(store: dict) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(FIXTURE_PATH, "wb") as fh:
        pickle.dump(store, fh, protocol=4)


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def make_client(engine):
    from flask import Flask

    from apps.web.modules.Ironbeam.routes import register_gex_landscape_route

    server = Flask(__name__)
    register_gex_landscape_route(server, engine)
    return server.test_client()


def get(client, qs: str):
    return client.get(f"/api/gex-landscape?{qs}")
