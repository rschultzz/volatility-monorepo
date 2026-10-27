#!/usr/bin/env python3
"""CR-CR — CR-CP's definitions on R-a with the effort call rule. READ-ONLY, R-a only.

Spec: specs/CR-CR-effort-range-ra.md (E1-E10). Sets CR_CP_PERIOD=ra before cr_cp_lib is imported, so every read is
guarded to [2024-04-01, 2025-01-01) UTC and logs go to scripts/logs/cr_cr. The primary is CR-CP's `cpp_pushes`
variant exactly (E5); readouts swap one field each (E6).
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

assert os.environ.setdefault("CR_CP_PERIOD", "ra") == "ra"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cr_cp_lib as C  # noqa: E402

assert C.PERIOD == "ra" and C.OUT.name == "cr_cr" and C.D_FROM.year == 2024 and C.CAP.year == 2025

PRIMARY = replace(C.PRIMARY, rule="cpp")                         # E5 = CR-CP VARIANTS["cpp_pushes"]
assert PRIMARY == C.VARIANTS["cpp_pushes"]
VARIANTS = {
    "primary": PRIMARY,
    "ratio13": replace(PRIMARY, cpp_ratio=1.3), "ratio20": replace(PRIMARY, cpp_ratio=2.0),
    "swing_fixed12": replace(PRIMARY, swing=12.0),
}
