"""Record/replay harness for /api/ironbeam/smile-data and /skew-data (CR-CC A2 byte-identity test).

Same pattern as bars_fixture: the two payload builders read ORATS only through
`fetch_data_from_db`, `fetch_skew_data` (imported into callbacks) and, under `until`,
`_fetch_latest_snapshot_hhmm`. `record()` wraps them against the live DB (SELECTs only);
`replay()` serves the stored values.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from apps.web.modules.Ironbeam.tests import bars_fixture as bf

FIXTURE_DIR = bf.FIXTURE_DIR
RECORDED_FUNCS = ("fetch_data_from_db", "fetch_skew_data")
OPTIONAL_FUNCS = ("_fetch_latest_snapshot_hhmm",)  # added by CR-CC A2

BASE = "trade_date=2026-10-23&expiration_date=2026-10-23&times=07:00,07:30,08:15,10:00"
QUERIES = {
    "smile": "/api/ironbeam/smile-data?" + BASE,
    "smile_noexp": "/api/ironbeam/smile-data?" + BASE + "&expected=off",
    "skew": "/api/ironbeam/skew-data?" + BASE,
    "skew_noexp": "/api/ironbeam/skew-data?" + BASE + "&expected=off",
}

callbacks_module = bf.callbacks_module
make_client = bf.make_client
save = bf.save
load = bf.load


@contextmanager
def record(store: dict):
    cb = callbacks_module()
    store.setdefault("calls", {})
    with ExitStack() as stack:
        for name in RECORDED_FUNCS + OPTIONAL_FUNCS:
            real = getattr(cb, name, None)
            if real is None:
                continue

            def wrapper(*args, __real=real, __name=name, **kwargs):
                out = __real(*args, **kwargs)
                store["calls"][bf._key(__name, args, kwargs)] = out
                return out

            stack.enter_context(patch.object(cb, name, wrapper))
        yield store


@contextmanager
def replay(store: dict):
    """Serve the recorded results; any unrecorded call raises (fails loudly)."""
    cb = callbacks_module()
    with ExitStack() as stack:
        for name in RECORDED_FUNCS + OPTIONAL_FUNCS:
            if not hasattr(cb, name):
                continue

            def fake(*args, __name=name, **kwargs):
                key = bf._key(__name, args, kwargs)
                if key not in store["calls"]:
                    raise KeyError(f"unrecorded call {key}")
                out = store["calls"][key]
                return out.copy() if hasattr(out, "copy") else out

            stack.enter_context(patch.object(cb, name, fake))
        yield
