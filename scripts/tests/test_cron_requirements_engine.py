"""CR-BJ — the cron build (apps/cron/requirements.txt) must be able to build the
options_cache engine.

Two layers:
  * static (always runs): the DBAPI driver SQLAlchemy picks for a plain
    postgresql:// URL is listed in the cron requirements.
  * clean-venv smoke (opt-in, RUN_CRON_VENV_SMOKE=1; slow, needs network): a fresh
    venv with ONLY the cron requirements imports the repository and builds the
    engine. No database connection is made — create_engine() is lazy, but it does
    import the DBAPI module, which is the failure this guards against.
"""
import os
import re
import subprocess
import sys
import venv
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

_ROOT = Path(__file__).resolve().parents[2]
CRON_REQS = _ROOT / "apps" / "cron" / "requirements.txt"
PLAIN_URL = "postgresql://u:p@localhost:5432/db"     # same scheme as the Render URLs; never connected to

# DBAPI module -> requirement names that provide it
_PROVIDERS = {"psycopg2": {"psycopg2", "psycopg2-binary"}, "psycopg": {"psycopg"}}


def _requirement_names():
    names = set()
    for line in CRON_REQS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[\[<>=!~; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_cron_requirements_list_the_default_postgres_driver():
    driver = make_url(PLAIN_URL).get_dialect().driver
    assert driver in _PROVIDERS, f"unexpected default driver {driver!r}"
    assert _PROVIDERS[driver] & _requirement_names(), (
        f"get_engine() on a plain postgresql:// URL needs the {driver!r} DBAPI, "
        f"which {CRON_REQS.relative_to(_ROOT)} does not install")


@pytest.mark.skipif(os.environ.get("RUN_CRON_VENV_SMOKE") != "1",
                    reason="slow clean-venv install; set RUN_CRON_VENV_SMOKE=1")
def test_clean_cron_venv_builds_the_options_cache_engine(tmp_path):
    env_dir = tmp_path / "cronvenv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    py = env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", str(CRON_REQS)], check=True)

    code = ("from packages.shared.options_cache import repository as repo\n"
            "e = repo.get_engine()\n"
            "print(e.dialect.name, e.dialect.driver)\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    env["DATABASE_URL"] = PLAIN_URL
    out = subprocess.run([str(py), "-c", code], cwd=str(_ROOT), env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["postgresql", "psycopg2"]
