"""CR-DA shared code — universe, quotes, ES prints and the per-night instrument rows (no outcomes).

Spec: specs/CR-DA-overnight-premium-prices.md (D1–D7). Used by scripts/cr_da_step0.py and scripts/cr_da_run.py so
Step 0 and the run read one definition of every exclusion. READ-ONLY: SELECTs only, no run row, no API calls.

Reused unchanged: CR-BP's Black / K-aware solve (scripts/cr_bp_step0.py `black`, `solve_v`), CR-BO's ES loader
(scripts/cr_bo_step0.py `load_es`) and event calendar (scripts/cr_bo_common.py), the quote-validity rule
(packages/shared/backtest/quote_validity.py `leg_quote_is_valid`).
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
sys.path.insert(0, str(REPO_ROOT))

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from packages.shared.backtest.quote_validity import leg_quote_is_valid
from scripts import cr_bo_common as C
from scripts.cr_bp_step0 import black, solve_v

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

NIGHTS_CSV = REPO_ROOT / "scripts" / "cr_cy_nights.csv"
CRBO_PAIRS = REPO_ROOT / "scripts" / "output" / "cr_bo" / "step0_pairs.csv"
CRCY_LOGS = REPO_ROOT / "scripts" / "logs" / "cr_cy"          # git-ignored; copied from the main checkout
OUT_DIR = REPO_ROOT / "scripts" / "output" / "cr_da"
LOG_DIR = REPO_ROOT / "scripts" / "logs" / "cr_da"

CLOSE_PT, OPEN_PT = time(12, 55), time(6, 35)                 # D4: 15:55 / 09:35 ET
CLOSE_ET, OPEN_ET = time(15, 55), time(9, 35)
OFFSETS = (0, -1, 1, -2, 2)                                   # D3 fallback order, ties to the earlier minute
WING = 5.0                                                    # R-DA3
ES_TICK_RT = 0.25                                             # readout 9: ES round-trip cost per point of hedge

QUOTES_SQL = """
SELECT trade_date_d, expir_date_d, strike, option_type, snapshot_pt, bid_price, ask_price
FROM orats_options_minute
WHERE ticker = 'SPX' AND expiry_tod = 'pm' AND expir_date_d = ANY(%s)
  AND ((trade_date_d < expir_date_d AND snapshot_pt::time BETWEEN '12:50' AND '13:00')
    OR (trade_date_d = expir_date_d AND snapshot_pt::time BETWEEN '06:30' AND '06:45'))
"""


def load_universe() -> pd.DataFrame:
    """D1 + D2(a): the 621 nights with the CR-CY plan and window completeness."""
    n = pd.read_csv(NIGHTS_CSV)
    n["t"] = pd.to_datetime(n.t).dt.date; n["t1"] = pd.to_datetime(n.t1).dt.date
    s0 = pd.read_csv(CRBO_PAIRS, parse_dates=["t"])
    keep = set(s0[s0.keep].t.dt.date)
    if set(n.t) != keep or len(n) != 621:
        raise SystemExit(f"universe gate: cr_cy_nights {len(n)} vs CR-BO keep {len(keep)}")
    plan = pd.read_csv(CRCY_LOGS / "full_plan.csv")
    plan["t"] = pd.to_datetime(plan.t).dt.date
    w = pd.read_csv(CRCY_LOGS / "full_windows.csv")
    w["t"] = pd.to_datetime(w.t).dt.date
    cw = w.pivot_table(index="t", columns="window", values="contracts_with_rows", aggfunc="first")
    n = n.merge(plan[["t", "basis", "S_c", "K_c", "K_o"]], on="t", how="left")
    n["n_close"] = n.t.map(cw["close"]); n["n_open"] = n.t.map(cw["open"])
    n["cy_complete"] = (n.n_close == 6) & (n.n_open == 6)
    n["K0_c"] = n.K_c.map(lambda s: float(s.split("/")[1])); n["K0_o"] = n.K_o.map(lambda s: float(s.split("/")[1]))
    n["cal_gap"] = [(b - a).days for a, b in zip(n.t, n.t1)]
    n["weekday"] = [a.strftime("%a") for a in n.t]
    n["year"] = [a.year for a in n.t]
    n["event_tags"] = [",".join(C.event_tags(b)) for b in n.t1]
    n["event_0830"] = n.event_tags != ""
    n["fomc_t1"] = [b in C.FOMC_DAYS for b in n.t1]
    n["nonevent_weeknight"] = (n.cal_gap == 1) & n.weekday.isin(["Mon", "Tue", "Wed", "Thu"]) & ~n.event_0830
    n["tau_c"] = [C.tau_years(datetime.combine(a, CLOSE_ET), b) for a, b in zip(n.t, n.t1)]
    n["tau_o"] = [C.tau_years(datetime.combine(b, OPEN_ET), b) for b in n.t1]
    return n


def load_quotes(conn, t1s: list[date]) -> dict:
    """{(t1, 'close'|'open', K, cp, 'HH:MM' PT): (bid, ask)} for close rows on t and open rows on t+1."""
    q = pd.read_sql(QUOTES_SQL, conn, params=(list(t1s),))
    q["hm"] = q.snapshot_pt.dt.strftime("%H:%M")
    q["win"] = np.where(q.trade_date_d == q.expir_date_d, "open", "close")
    return q


def quote_map(q: pd.DataFrame, t_of: dict) -> dict:
    """Close rows must be from t itself (the session before t+1), not an earlier day."""
    out = {}
    for r in q.itertuples():
        if r.win == "close" and r.trade_date_d != t_of.get(r.expir_date_d):
            continue
        out[(r.expir_date_d, r.win, float(r.strike), r.option_type, r.hm)] = (r.bid_price, r.ask_price)
    return out


def pick(Q: dict, t1: date, win: str, K: float, cp: str) -> tuple | None:
    """D3: (bid, ask, offset, zero_zero_at_target) at the nearest valid minute within ±2, else None."""
    base = datetime.combine(t1, CLOSE_PT if win == "close" else OPEN_PT)
    zz = False
    for off in OFFSETS:
        hm = (base + timedelta(minutes=off)).strftime("%H:%M")
        r = Q.get((t1, win, float(K), cp, hm))
        if off == 0 and r is not None and r[0] == 0 and r[1] == 0:
            zz = True
        if r is not None and leg_quote_is_valid(r[0], r[1]):
            return (float(r[0]), float(r[1]), off, zz)
    return None


def es_print(es_close: pd.Series, at: datetime, back: int = 5) -> float:
    """Price at `at` (ET) = close of the bar starting at − 1 min; earlier bars up to `back` min as fallback."""
    for k in range(1, back + 1):
        v = es_close.get(at - timedelta(minutes=k))
        if v is not None and not np.isnan(v):
            return float(v)
    return np.nan


def rv(es: pd.DataFrame, lo: datetime, hi: datetime) -> float:
    """Σ r² over bars with lo ≤ start < hi (CR-BO L4 arithmetic)."""
    a, b = es.et.searchsorted(lo, side="left"), es.et.searchsorted(hi, side="left")
    return float((es.r.values[a:b] ** 2).sum())


def straddle_delta(F: float, K: float, v: float) -> float:
    d1 = (np.log(F / K) + v / 2) / np.sqrt(v)
    return 2 * norm.cdf(d1) - 1


def instruments(n: pd.DataFrame, Q: dict, es_close: pd.Series) -> pd.DataFrame:
    """Per night: ATM straddle and ±5 strangle quotes and V_px at both ends, spreads, the tradeable entry / exit.
    No realised variance and no premium here (Step 0 reads this frame)."""
    rows = []
    for r in n.itertuples():
        rec = dict(t=r.t, t1=r.t1)
        S_c = float(r.S_c)
        S_o = es_print(es_close, datetime.combine(r.t1, OPEN_ET)) - float(r.basis)
        es_c = es_print(es_close, datetime.combine(r.t, CLOSE_ET)); es_o = es_print(es_close, datetime.combine(r.t1, OPEN_ET))
        rec.update(S_o=S_o, es_c=es_c, es_o=es_o)   # S_c comes from the CR-CY plan
        for end, K, S in (("c", r.K0_c, S_c), ("o", r.K0_o, S_o)):
            win = "close" if end == "c" else "open"
            legs = {"atmC": (K, "C"), "atmP": (K, "P"), "wingP": (K - WING, "P"), "wingC": (K + WING, "C")}
            got = {k: pick(Q, r.t1, win, *v) for k, v in legs.items()}
            for k, g in got.items():
                rec[f"{k}_{end}_ok"] = g is not None
                rec[f"{k}_{end}_off"] = g[2] if g else np.nan
                rec[f"{k}_{end}_zz"] = g[3] if g else bool(_zz(Q, r.t1, win, *legs[k]))
            st_ok = got["atmC"] is not None and got["atmP"] is not None
            sg_ok = got["wingP"] is not None and got["wingC"] is not None
            mid = lambda g: (g[0] + g[1]) / 2
            if st_ok:
                st = mid(got["atmC"]) + mid(got["atmP"])
                rec[f"st_{end}"] = st
                rec[f"spr_{end}"] = (got["atmC"][1] - got["atmC"][0]) + (got["atmP"][1] - got["atmP"][0])
                rec[f"bid_{end}"] = got["atmC"][0] + got["atmP"][0]
                rec[f"ask_{end}"] = got["atmC"][1] + got["atmP"][1]
                rec[f"V_{end}"] = solve_v(st, S, [(K, "C"), (K, "P")]) if np.isfinite(S) else np.nan
                rec[f"Fpar_{end}"] = K + mid(got["atmC"]) - mid(got["atmP"])
            else:
                rec[f"st_{end}"] = rec[f"spr_{end}"] = rec[f"bid_{end}"] = rec[f"ask_{end}"] = rec[f"V_{end}"] = rec[f"Fpar_{end}"] = np.nan
            if sg_ok and np.isfinite(S):
                rec[f"Vstr_{end}"] = solve_v(mid(got["wingP"]) + mid(got["wingC"]), S, [(K - WING, "P"), (K + WING, "C")])
            else:
                rec[f"Vstr_{end}"] = np.nan
            rec[f"st_ok_{end}"] = st_ok; rec[f"sg_ok_{end}"] = sg_ok
        # tradeable: sell K0_c at the close bids, buy K0_c back at the open asks (quote, else model mark)
        Kc = float(r.K0_c)
        bc, bp = pick(Q, r.t1, "open", Kc, "C"), pick(Q, r.t1, "open", Kc, "P")
        if bc is not None and bp is not None:
            rec.update(exit_ask=bc[1] + bp[1], exit_mark="quote")
        elif np.isfinite(rec["V_o"]) and np.isfinite(S_o) and np.isfinite(rec["spr_o"]):
            model = black(S_o, Kc, rec["V_o"], "C") + black(S_o, Kc, rec["V_o"], "P")
            rec.update(exit_ask=model + rec["spr_o"] / 2, exit_mark="model")
        else:
            rec.update(exit_ask=np.nan, exit_mark="none")
        rec["hedge_h"] = straddle_delta(S_c, Kc, rec["V_c"]) if np.isfinite(rec["V_c"]) and rec["V_c"] > 0 else np.nan
        rows.append(rec)
    return n.merge(pd.DataFrame(rows), on=["t", "t1"])


def _zz(Q, t1, win, K, cp) -> bool:
    r = Q.get((t1, win, float(K), cp, (CLOSE_PT if win == "close" else OPEN_PT).strftime("%H:%M")))
    return r is not None and r[0] == 0 and r[1] == 0


def exclusions(df: pd.DataFrame) -> pd.DataFrame:
    """D2 (a)–(c), in order; adds `excl` (first failing reason or '') and `final`."""
    reason = np.where(~df.cy_complete, "a_cy_incomplete",
             np.where(~(df.st_ok_c & df.st_ok_o), "b_no_valid_atm_quote",
             np.where(~(np.isfinite(df.V_c) & np.isfinite(df.V_o) & (df.V_c > 0) & (df.V_o > 0)), "c_solve_failed", "")))
    df = df.copy(); df["excl"] = reason; df["final"] = df.excl == ""
    return df
