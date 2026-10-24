"""Record/replay harness for /api/ironbeam/bars (CR-BT L7 byte-identity test).

The payload builder in modules.Ironbeam.callbacks reads the DB only through the
functions in RECORDED_FUNCS. `record()` wraps them against the live DB (SELECTs
only) and stores every (function, args) -> return value; `replay()` patches them
to serve the stored values, so the route can be exercised with no DB.

The env-derived module constants that shape the payload (symbol, GEX bucket /
padding / threshold / width / colour settings) are stored too and patched on
replay, so the result does not depend on the local .env.

callbacks.py imports `modules.Ironbeam...`, so apps/web must be on sys.path and
the module must be imported as `modules.Ironbeam.callbacks` (not
`apps.web.modules...`, which would be a second copy).
"""
from __future__ import annotations

import gzip
import os
import pickle
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[5]
for p in (str(REPO_ROOT), str(REPO_ROOT / "apps" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)
# callbacks.py builds a (lazy) engine at import; replay never connects.
os.environ.setdefault("DATABASE_URL", "postgresql://replay@localhost/replay")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

RECORDED_FUNCS = (
    "_fetch_bars_pt",
    "_fetch_gex_grouped_by_level",
    "_fetch_gex_grouped_by_level_and_expir",
    "_fetch_available_trade_dates",
    "fetch_skew_data",
)
# Added by CR-BT step 2; absent from fixtures recorded before it existed.
OPTIONAL_FUNCS = (
    "_fetch_gex_rows_for_placement",
    "_fetch_es_snapshot_close",
)
RECORDED_CONSTANTS = (
    "TRADES_SYMBOL",
    "TICKER",
    "GEX_LEVEL_PADDING",
    "GEX_ABS_THRESHOLD_DEFAULT",
    "GEX_COLOR_ABS_MAX",
    "GEX_COLOR_PERCENTILE",
    "GEX_MAX_LEVELS_PER_DAY",
    "GEX_MIN_LEVEL_SPACING",
    "GEX_LEVEL_BUCKET",
    "GEX_LEVEL_LINE_WIDTH",
    "GEX_LEVEL_LINE_WIDTH_SCALE",
    "GEX_LEVEL_LINE_WIDTH_MAX",
    "GEX_LEVEL_LINE_OPACITY",
)

# Requests the fixture covers (query strings, without the route prefix).
CENTER_QS = "trade_date=2026-10-23&interval=1min&phase=center&gex_enabled=1&gex_min_abs_b=10"
MULTI_QS = "trade_date=2026-10-23&interval=1min&phase=multi&gex_enabled=1&gex_min_abs_b=10&days_either_side=1"


def callbacks_module():
    import modules.Ironbeam.callbacks as cb  # noqa: WPS433
    return cb


def make_client():
    import dash

    cb = callbacks_module()
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = dash.html.Div()
    cb.register_ironbeam_callbacks(app)
    app.server.config["PROPAGATE_EXCEPTIONS"] = True
    return app.server.test_client()


def _key(name, args, kwargs):
    return (name, repr(args), repr(sorted(kwargs.items())))


@contextmanager
def record(store: dict):
    """Wrap the DB readers so every call's result lands in `store`."""
    cb = callbacks_module()
    store.setdefault("calls", {})
    store["constants"] = {k: getattr(cb, k) for k in RECORDED_CONSTANTS}
    with ExitStack() as stack:
        for name in RECORDED_FUNCS + OPTIONAL_FUNCS:
            real = getattr(cb, name, None)
            if real is None:
                continue

            def wrapper(*args, __real=real, __name=name, **kwargs):
                out = __real(*args, **kwargs)
                store["calls"][_key(__name, args, kwargs)] = out
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
        for name in RECORDED_FUNCS + OPTIONAL_FUNCS:
            if not hasattr(cb, name):
                continue

            def fake(*args, __name=name, **kwargs):
                key = _key(__name, args, kwargs)
                if key not in store["calls"]:
                    raise KeyError(f"unrecorded call {key}")
                out = store["calls"][key]
                return out.copy() if hasattr(out, "copy") else out

            stack.enter_context(patch.object(cb, name, fake))
        yield


def save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
