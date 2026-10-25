"""Record/replay harness for /api/ironbeam/flow?mode=v2 (CR-CD step 2).

`modules.Ironbeam.flow_v2` reads the DB only through `fetch_flow_minutes` and `fetch_bar_closes`.
`record()` wraps them against the live DB (SELECTs only); `replay()` serves the stored results.
Keys leave out the engine argument (its repr carries the DB user and host). The in-process
minute cache is cleared on entry to both, so every read is recorded / replayed.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from apps.web.modules.Ironbeam.tests import bars_fixture as bf

FIXTURE_DIR = bf.FIXTURE_DIR
FIXTURE = FIXTURE_DIR / "cr_cd_flow_v2.pkl.gz"
RECORDED_FUNCS = ("fetch_flow_minutes", "fetch_bar_closes")

QS_0501 = "trade_date=2026-05-01&session=FULL&resample=1m&ema_len=14&mode=v2"
QS_0501_UNTIL = QS_0501 + "&until=2026-05-01T07:39:00"
QS_1023 = "trade_date=2026-10-23&session=FULL&resample=1m&ema_len=14&mode=v2"
QS_0317 = "trade_date=2026-03-17&session=RTH&resample=1m&ema_len=14&mode=v2"
QUERIES = {"0501": QS_0501, "0501_until": QS_0501_UNTIL, "1023": QS_1023, "0317": QS_0317}

make_client = bf.make_client
save = bf.save
load = bf.load


def v2_module():
    bf.callbacks_module()
    import modules.Ironbeam.flow_v2 as v2  # noqa: WPS433
    return v2


def _key(name, args, kwargs):
    return bf._key(name, args[1:], kwargs)


@contextmanager
def record(store: dict):
    v2 = v2_module()
    v2._CACHE.clear()
    store.setdefault("calls", {})
    with ExitStack() as stack:
        for name in RECORDED_FUNCS:
            real = getattr(v2, name)

            def wrapper(*args, __real=real, __name=name, **kwargs):
                out = __real(*args, **kwargs)
                store["calls"][_key(__name, args, kwargs)] = out
                return out

            stack.enter_context(patch.object(v2, name, wrapper))
        yield store


@contextmanager
def replay(store: dict):
    """Serve the recorded results; any unrecorded call raises (fails loudly)."""
    v2 = v2_module()
    v2._CACHE.clear()
    with ExitStack() as stack:
        for name in RECORDED_FUNCS:

            def fake(*args, __name=name, **kwargs):
                key = _key(__name, args, kwargs)
                if key not in store["calls"]:
                    raise KeyError(f"unrecorded call {key}")
                out = store["calls"][key]
                return out.copy() if hasattr(out, "copy") else out

            stack.enter_context(patch.object(v2, name, fake))
        yield
    v2._CACHE.clear()
