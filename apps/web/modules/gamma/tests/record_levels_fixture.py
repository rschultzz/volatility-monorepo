"""Record the CR-DH Dashboard fixture and the origin/main golden figures (live DB, SELECTs only).

    apps/web/.venv/bin/python apps/web/modules/gamma/tests/record_levels_fixture.py

Goldens: origin/main's modules/gamma/callbacks.py (read with `git show`) rendered on the
recorded data, one JSON per date — the bytes the Current chart must keep.
"""
from __future__ import annotations

import gzip
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from apps.web.modules.gamma.tests import levels_fixture as lf  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(lf.REPO_ROOT / ".env", override=True)


def origin_gamma_module():
    src = subprocess.run(
        ["git", "show", "origin/main:apps/web/modules/gamma/callbacks.py"],
        cwd=lf.REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    path = Path(lf.FIXTURE_DIR / "_origin_gamma_callbacks.py")
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("origin_gamma_callbacks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path.unlink()
    return mod


def main():
    import datetime as dt
    mods = lf.modules()
    store = {"calls": {}, "constants": {g: {n: getattr(mods[g], n) for n in names}
                                        for g, names in lf.CONSTANTS.items()}}
    for d_iso in lf.DATES:
        d = dt.date.fromisoformat(d_iso)
        store["calls"][lf._key("_fetch_gex_grouped_by_level", (d,))] = mods["gamma"]._fetch_gex_grouped_by_level(d)
        store["calls"][lf._key("_fetch_es_open_at_1501_prior_day", (d,))] = mods["gamma"]._fetch_es_open_at_1501_prior_day(d)
        api = mods["ib"].prev_trading_day(d)
        store["calls"][lf._key("_fetch_es_snapshot_close", (api,))] = mods["ib"]._fetch_es_snapshot_close(api)
        if mods["ib"].corrected_levels_defined(d) and store["calls"][lf._key("_fetch_es_snapshot_close", (api,))]:
            store["calls"][lf._key("_fetch_gex_rows_for_placement", (d,))] = mods["ib"]._fetch_gex_rows_for_placement(d)
    lf.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(lf.FIXTURE, "wb") as f:
        pickle.dump(store, f)

    origin = origin_gamma_module()
    with lf.replay(store, extra_gamma_modules=(origin,)):
        for d_iso in lf.DATES:
            fig = lf.run_render(origin, d_iso, mode_arg=False)
            (lf.GOLDEN_DIR / f"cr_dh_origin_main_{d_iso}.json").write_text(fig.to_json())
    print("recorded", lf.FIXTURE, "and", len(lf.DATES), "goldens")


if __name__ == "__main__":
    main()
