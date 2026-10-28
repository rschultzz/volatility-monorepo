"""Record/replay harness for the Dashboard GEX chart (CR-DH).

`render_gex` reads the DB only through the functions in RECORDED (gamma callbacks + the two
Ironbeam CR-BT fetch helpers behind `_corrected_gex_tables`). `record()` wraps them against the
live DB (SELECTs only); `replay()` patches them to serve the stored values, so the callback runs
with no DB. Env-derived constants that shape the figure are stored and patched too.

Modules are imported as `modules.gamma.callbacks` / `modules.Ironbeam.callbacks` (apps/web on
sys.path), the same names app.py uses.
"""
from __future__ import annotations

import gzip
import os
import pickle
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[5]
for p in (str(REPO_ROOT), str(REPO_ROOT / "apps" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)
# Ironbeam callbacks builds a (lazy) engine at import; replay never connects.
os.environ.setdefault("DATABASE_URL", "postgresql://replay@localhost/replay")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE = FIXTURE_DIR / "cr_dh_gex_levels.pkl.gz"
GOLDEN_DIR = FIXTURE_DIR

# Corrected defined + ES snapshot present; pre-forward era; no ES snapshot (2025-11-28 early close).
DATES = ("2026-10-27", "2024-03-15", "2025-12-01")

RECORDED = {
    "gamma": ("_fetch_gex_grouped_by_level", "_fetch_es_open_at_1501_prior_day"),
    "ib": ("_fetch_gex_rows_for_placement", "_fetch_es_snapshot_close"),
}
CONSTANTS = {
    "gamma": ("TICKER", "ZOOM_PCT", "PUT_COLOR", "CALL_COLOR"),
    "ib": ("TICKER", "GEX_LEVEL_BUCKET"),
}


def modules():
    from modules.gamma import callbacks as gamma
    from modules.Ironbeam import callbacks as ib
    return {"gamma": gamma, "ib": ib}


def _key(name, args):
    return (name, tuple(a.isoformat() if hasattr(a, "isoformat") else a for a in args))


@contextmanager
def replay(store, extra_gamma_modules=()):
    mods = modules()
    targets = {"gamma": [mods["gamma"], *extra_gamma_modules], "ib": [mods["ib"]]}
    with ExitStack() as st:
        for grp, names in RECORDED.items():
            for name in names:
                def fake(*args, _n=name):
                    return store["calls"][_key(_n, args)]
                for m in targets[grp]:
                    st.enter_context(patch.object(m, name, fake))
        for grp, names in CONSTANTS.items():
            for name in names:
                for m in targets[grp]:
                    st.enter_context(patch.object(m, name, store["constants"][grp][name]))
        yield


def run_render(mod, date_iso, mode=None, *, mode_arg=True):
    """Call render_gex outside a Dash request (no trigger = a date change)."""
    import dash
    fake_ctx = SimpleNamespace(triggered=[], triggered_id=None)
    with patch.object(dash, "ctx", fake_ctx):
        fn = getattr(mod.render_gex, "__wrapped__", mod.render_gex)
        if mode_arg:
            return fn(date_iso, 0, mode, None)
        return fn(date_iso, 0, None)


def load(path=FIXTURE):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
