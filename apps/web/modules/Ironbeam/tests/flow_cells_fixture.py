"""Record/replay harness for /api/ironbeam/flow/cells (CR-DC step 2).

`modules.Ironbeam.flow_v3` reads the DB only through its `fetch_*`, `_exists_price_1m` and `_trades_oldest`
functions, plus Flow v2's `fetch_flow_minutes`, `fetch_bar_closes` and `_watermark_1m` (contract choice).
`record()` wraps them against the live DB (SELECTs only); `replay()` serves the stored results. Keys leave
out the engine argument. Both modules' in-process caches are cleared on entry.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from apps.web.modules.Ironbeam.tests import bars_fixture as bf
from apps.web.modules.Ironbeam.tests import flow_v2_fixture as vf

FIXTURE_DIR = bf.FIXTURE_DIR
FIXTURE = FIXTURE_DIR / "cr_dc_flow_cells.pkl.gz"
V3_FUNCS = ("fetch_price_1m", "fetch_trade_cells", "fetch_seconds", "_exists_price_1m", "_trades_oldest")

QS_0501 = "trade_date=2026-05-01&session=FULL"
QS_0501_RTH = "trade_date=2026-05-01&session=RTH"
QS_0501_UNTIL = "trade_date=2026-05-01&session=RTH&until=2026-05-01T07:39:00"
QS_1027 = "trade_date=2026-10-27&session=FULL"        # exact (trades) while retained
QS_1027_UNTIL = "trade_date=2026-10-27&session=RTH&until=2026-10-27T08:15:00"
QS_1023 = "trade_date=2026-10-23&session=RTH"         # after 2026-06-05, beyond retention: Ironbeam 1-s
QUERIES = {"0501": QS_0501, "0501_rth": QS_0501_RTH, "0501_until": QS_0501_UNTIL, "1027": QS_1027,
           "1027_until": QS_1027_UNTIL, "1023": QS_1023}

make_client = bf.make_client
save = bf.save
load = bf.load


def v3_module():
    bf.callbacks_module()
    import modules.Ironbeam.flow_v3 as v3  # noqa: WPS433
    return v3


def _clear(v2, v3):
    for c in (v2._CACHE, v2._WATERMARK, v2._TABLE_STATE, v3._CACHE, v3._STATE):
        c.clear()


def _targets():
    v2, v3 = vf.v2_module(), v3_module()
    return v2, v3, [(v2, n) for n in vf.RECORDED_FUNCS] + [(v3, n) for n in V3_FUNCS]


@contextmanager
def record(store: dict):
    v2, v3, targets = _targets()
    _clear(v2, v3)
    store.setdefault("calls", {})
    with ExitStack() as stack:
        for mod, name in targets:
            real = getattr(mod, name)

            def wrapper(*args, __real=real, __name=name, **kwargs):
                out = __real(*args, **kwargs)
                store["calls"][bf._key(__name, args[1:], kwargs)] = out
                return out

            stack.enter_context(patch.object(mod, name, wrapper))
        yield store


@contextmanager
def replay(store: dict, overrides: dict | None = None):
    """Serve the recorded results; any unrecorded call raises. `overrides` {name: fn} replaces a read."""
    v2, v3, targets = _targets()
    _clear(v2, v3)
    calls = []
    with ExitStack() as stack:
        for mod, name in targets:

            def fake(*args, __name=name, **kwargs):
                calls.append((__name, args[1:], kwargs))
                if overrides and __name in overrides:
                    return overrides[__name](*args[1:], **kwargs)
                key = bf._key(__name, args[1:], kwargs)
                if key not in store["calls"]:
                    raise KeyError(f"unrecorded call {key}")
                out = store["calls"][key]
                return out.copy() if hasattr(out, "copy") else out

            stack.enter_context(patch.object(mod, name, fake))
        yield calls
    _clear(v2, v3)
