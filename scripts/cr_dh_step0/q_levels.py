"""CR-DH Step 0 a/b (read-only): Dashboard discounted vs corrected levels, and the price chart's corrected lines.

Run from the repo root:  apps/web/.venv/bin/python scripts/cr_dh_step0/q_levels.py
Writes scripts/logs/cr-dh/step0_levels.json and prints a summary.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "web"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from modules.gamma import callbacks as gamma  # noqa: E402
from modules.Ironbeam import callbacks as ib  # noqa: E402
from packages.shared.trading_calendar import prev_trading_day  # noqa: E402

OUT = ROOT / "scripts" / "logs" / "cr-dh"
OUT.mkdir(parents=True, exist_ok=True)


def top10(df):
    d = df.assign(abs_net=df["net_gamma"].abs()).sort_values("abs_net", ascending=False).head(10)
    return [
        {"level": float(r.level), "net_bn": round(r.net_gamma / 1e9, 3),
         "call_bn": round(r.call_gamma / 1e9, 3), "put_bn": round(r.put_gamma / 1e9, 3)}
        for r in d.itertuples()
    ]


def chart_lines(date_iso: str, mode: str):
    payload, status = ib._build_react_preview_bars_payload(date_iso, "1min", levels_mode=mode, phase="center")
    segs = [s for s in payload.get("gex_segments", []) if s.get("session_date") == date_iso]
    return status, payload.get("levels_notes"), segs


report: dict = {}

# --- a. 2026-10-27 ---
d = dt.date(2026, 10, 27)
disc = gamma._fetch_gex_grouped_by_level(d)
corr, corr_ex, reason = ib._corrected_gex_tables(d)
status, notes, segs = chart_lines(d.isoformat(), "corrected")
corr_by_level = {float(r.level): r for r in corr.itertuples()} if corr is not None else {}
mismatch = []
for s in segs:
    r = corr_by_level.get(float(s["level"]))
    if r is None or abs(r.net_gamma - s["net_gamma"]) > 1e-3 or abs(r.call_gamma - s["call_gamma"]) > 1e-3 \
            or abs(r.put_gamma - s["put_gamma"]) > 1e-3:
        mismatch.append(s["level"])
report["a"] = {
    "date": d.isoformat(),
    "discounted_rows": len(disc), "corrected_rows": None if corr is None else len(corr), "reason": reason,
    "es_snapshot_close": ib._fetch_es_snapshot_close(prev_trading_day(d)),
    "top10_discounted": top10(disc),
    "top10_corrected": top10(corr) if corr is not None else None,
    "price_chart_status": status, "price_chart_notes": notes,
    "price_chart_corrected_lines": sorted(
        [{"level": s["level"], "net_bn": round(s["net_gamma"] / 1e9, 3)} for s in segs],
        key=lambda x: -abs(x["net_bn"])),
    "price_chart_lines_not_matching_corrected_table": mismatch,
}

# --- b. pre-forward-era and no-ES-snapshot dates ---
b = {}
for label, iso in (("pre_forward_era", "2024-03-15"), ("no_es_snapshot", "2025-12-01")):
    dd = dt.date.fromisoformat(iso)
    _, _, rsn = ib._corrected_gex_tables(dd)
    st, nt, sg = chart_lines(iso, "corrected")
    st_cur, _, sg_cur = chart_lines(iso, "current")
    b[label] = {"date": iso, "corrected_reason": rsn, "status": st, "levels_notes": nt,
                "corrected_lines": len(sg), "current_lines": len(sg_cur),
                "es_snapshot_close_prev_day": ib._fetch_es_snapshot_close(prev_trading_day(dd))}
report["b"] = b

(OUT / "step0_levels.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
