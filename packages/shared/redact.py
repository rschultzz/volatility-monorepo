"""Mask database connection strings in text that is printed, logged or returned (CR-DE R3).

Same rule as CR-CA's `scripts/cr_ca_step0.py::redact`: the exact DATABASE_URL /
BACKFILL_DATABASE_URL values become `<DATABASE_URL>` / `<BACKFILL_DATABASE_URL>`, and any other
postgres URL becomes `<postgres-url>`. Also masks a libpq keyword `password=...`. psycopg puts the
whole conninfo in some error messages (e.g. `invalid connection option "postgresql+psycopg://..."`),
so pass `str(e)` of any DB error through this before it leaves the process.
"""
from __future__ import annotations

import os
import re

_URL_RE = re.compile(r"postgres(?:ql)?(?:\+\w+)?://[^\s'\"]+")
_KW_PASSWORD_RE = re.compile(r"(password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE)
_ENV_KEYS = ("DATABASE_URL", "BACKFILL_DATABASE_URL")


def redact(text) -> str:
    """`text` with every DB connection string and password masked."""
    text = str(text)
    for k in _ENV_KEYS:
        v = (os.environ.get(k) or "").strip()
        if v:
            for form in {v, v.replace("+psycopg", "")}:
                text = text.replace(form, f"<{k}>")
    text = _URL_RE.sub("<postgres-url>", text)
    return _KW_PASSWORD_RE.sub(r"\1<redacted>", text)
