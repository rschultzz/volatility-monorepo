"""Record/replay harness for /api/ironbeam/flow (CR-CC byte-identity test).

Same pattern as bars_fixture: `_build_react_flow_payload` reads the DB only through
`_resolve_symbol` and `_fetch_flow_utc`; `record()` wraps them against the live DB
(SELECTs only), `replay()` serves the stored values. The env-derived FLOW_* constants
are stored and patched on replay so the result does not depend on the local .env.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from apps.web.modules.Ironbeam.tests import bars_fixture as bf

FIXTURE_DIR = bf.FIXTURE_DIR
RECORDED_FUNCS = ("_resolve_symbol", "_fetch_flow_utc")
RECORDED_CONSTANTS = ("TRADES_SYMBOL", "FLOW_SYMBOL", "FLOW_SESSION", "FLOW_RESAMPLE", "FLOW_EMA_LEN")

# Requests the fixture covers: the price app's default (FULL, 1m) and an RTH 15s view.
FULL_QS = "trade_date=2026-10-23&session=FULL&resample=1m&ema_len=14"
RTH_QS = "trade_date=2026-10-23&session=RTH&resample=15s&ema_len=56"
QUERIES = {"full": FULL_QS, "rth": RTH_QS}

callbacks_module = bf.callbacks_module
make_client = bf.make_client
save = bf.save
load = bf.load


@contextmanager
def record(store: dict):
    cb = callbacks_module()
    store.setdefault("calls", {})
    store["constants"] = {k: getattr(cb, k) for k in RECORDED_CONSTANTS}
    with ExitStack() as stack:
        for name in RECORDED_FUNCS:
            real = getattr(cb, name)

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
        for k, v in store["constants"].items():
            stack.enter_context(patch.object(cb, k, v))
        for name in RECORDED_FUNCS:

            def fake(*args, __name=name, **kwargs):
                key = bf._key(__name, args, kwargs)
                if key not in store["calls"]:
                    raise KeyError(f"unrecorded call {key}")
                out = store["calls"][key]
                return out.copy() if hasattr(out, "copy") else out

            stack.enter_context(patch.object(cb, name, fake))
        yield
