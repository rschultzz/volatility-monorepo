"""CR-BR shared definitions — close-time term structure (CR-BL method), roll-safe ES nights (CR-BO rules).

Imported by scripts/cr_br_step0.py and the run scripts so Step 0 and the run use one definition of every input
and exclusion (specs/CR-BR-term-structure-slope.md, L1–L4).

READ-ONLY. Monies reads are limited to the latest ≤ 12:46 PT snapshot of t (PM nodes, atmiv and vol50) and the
12:59 PT stock_price used only by CR-BO's contract-switch detector. No open-time monies field is read.
ES bars are read only up to 2026-06-05 13:00 PT (holdout never read).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from scripts import cr_bo_common as BO  # noqa: E402   (session_pairs, roll_week, EARLY_CLOSES, window grid)
from scripts.cr_bo_extract import detect_switches, es_side, es_ok  # noqa: E402   (CR-BO ES-side rules)
from scripts.cr_bl_backfill import total_variance_at  # noqa: E402   (CR-BL constant-maturity method)

SPLIT = dt.date(2026, 6, 5)
START = dt.date(2023, 5, 1)
SNAP_CUTOFF = dt.time(12, 46)
TENORS = (7, 14, 30, 60, 90)
E1_START = dt.date(2024, 3, 1)          # CR-BP A1.1 era split
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_br"

_SNAP_SQL = """SELECT max(snapshot_pt) FROM orats_monies_minute WHERE trade_date = %s AND snapshot_pt <= %s"""   # CR-BL verbatim
_NODES_SQL = """
    SELECT expir_date, atmiv, vol50 FROM orats_monies_minute
    WHERE trade_date = %s AND snapshot_pt = %s AND ticker = 'SPX'
      AND expiry_tod = 'pm' AND atmiv > 0 AND expir_date > trade_date
"""   # CR-BL _NODES_SQL + vol50 (Step 0 era ratio only)
_SWITCH_PX_SQL = """
    SELECT trade_date, expir_date, dte, snapshot_pt, stock_price FROM orats_monies_minute
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND expiry_tod = 'pm' AND dte BETWEEN 1 AND 16
      AND snapshot_pt::time = '12:59'
"""
_ES_SQL = "SELECT datetime, close, source FROM ironbeam_es_1m_bars WHERE datetime >= %s AND datetime < %s ORDER BY datetime"


def connect():
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    conn.execute("SET statement_timeout = '600s'")
    return conn


def sessions() -> list[dt.date]:
    return BO.train_sessions()


def load_snapshots(conn, days: list[dt.date]) -> dict:
    """{t: (snapshot_pt, atm nodes [(days, atmiv)], vol50 nodes [(days, vol50)])}; nodes ≥ 1 calendar day (CR-BL)."""
    out = {}
    for d in days:
        sp = conn.execute(_SNAP_SQL, (str(d), dt.datetime.combine(d, SNAP_CUTOFF))).fetchone()[0]
        if sp is None:
            continue
        rows = conn.execute(_NODES_SQL, (str(d), sp)).fetchall()
        a = sorted(((dt.date.fromisoformat(e) - d).days, float(v)) for e, v, _ in rows)
        b = sorted(((dt.date.fromisoformat(e) - d).days, float(w)) for e, _, w in rows if w is not None and w > 0)
        out[d] = (sp, [n for n in a if n[0] >= 1], [n for n in b if n[0] >= 1])
    return out


def cm_iv(nodes, T):
    """CR-BL: constant-maturity vol at T calendar days, linear in total variance; (iv, lo_days, hi_days) or None."""
    r = total_variance_at(nodes, T)
    if r is None:
        return None
    w, t1, t2 = r
    return float(np.sqrt(w / T)), t1, t2


def load_es(conn) -> pd.DataFrame:
    """ironbeam_es_1m_bars as CR-BO's load_es (ET stamps, r = ln close/prev close), cut at SPLIT 13:00 PT."""
    a = dt.datetime.combine(START - dt.timedelta(days=5), dt.time(0))
    b = pd.Timestamp(dt.datetime.combine(SPLIT, dt.time(13, 0)), tz="America/Los_Angeles").tz_convert("UTC").tz_localize(None).to_pydatetime()
    es = pd.read_sql(_ES_SQL, conn, params=(a, b)).drop_duplicates("datetime").sort_values("datetime")
    es["et"] = es.datetime.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    es = es.reset_index(drop=True)
    es["r"] = np.log(es.close / es.close.shift())
    es["hm"] = es.et.dt.strftime("%H:%M"); es["d"] = es.et.dt.date      # as CR-BO main() before detect_switches
    return es


def load_switch_px(conn) -> pd.DataFrame:
    m = pd.read_sql(_SWITCH_PX_SQL, conn, params=(START.isoformat(), SPLIT.isoformat()))
    m["td"] = pd.to_datetime(m.trade_date).dt.date
    m["et"] = m.snapshot_pt + pd.Timedelta(hours=3)
    m["hm"] = m.et.dt.strftime("%H:%M")
    return m


def es_nights(es: pd.DataFrame, pm: pd.DataFrame) -> tuple[dict, list]:
    """{t: CR-BO es_side dict + RV (13:00 PT t → 13:00 PT t+1) + ok} for every session pair (t, t+1 ≤ SPLIT)."""
    switches = detect_switches(es, pm)
    minutes = set(es.et)
    out = {}
    for t, t1 in BO.session_pairs():
        x = es_side(es, t, t1, switches, minutes)
        x["t1"] = t1
        x["RV"] = x["RV_on"] + x["RV_rth"]
        x["ok"] = es_ok(x)
        out[t] = x
    return out, switches
