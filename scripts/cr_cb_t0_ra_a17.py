#!/usr/bin/env python3
"""CR-CB Amendment A17 — second T0 read on R-a (2024-04-03 .. 2024-12-31), after a mechanical fix. READ-ONLY.

Spec: specs/CR-CB-failed-retest-at-level.md, Amendment A17 (A17.1–A17.7); everything not changed there is A16 and
is copied unchanged from scripts/cr_cb_t0_ra.py (guard < 2025-01-01, universe and exclusions incl. early closes and
the contract check, A1.1 single levels, A15.6 placebo, frozen train O_c, restricted-mean time to escape, zone
dwell, seeds). Price paths only; no trade outcome. One run; JSON written before printing.

A17 changes, fixed here:
  - A17.1 one test per level per session: the first armed test (either side; its side sets the cell).
  - A17.2 arming distance = max(0.10·IM_d, 4 pts) (> the 2-pt band always).
  - A17.3 session-weighted: per (session, cell, arm) mean of the measure; cell estimate = mean over sessions with
    >= 1 test in that arm. Bootstrap: the same 10 000 × n multinomial weights (seed 20260920) applied to both arms;
    estimate per resample = Σ w_s m_s / Σ w_s over sessions present in the arm. Holm one-sided α 0.025 over 4 cells +
    pooled on the primary; one-sided p = share of resamples with diff <= 0 (floor 1e-4).
  - A17.4 flag: implied_move_1d < 25 pts. Primary includes flagged sessions; readout excludes them (weights zeroed),
    no gate.
  - A17.5 IM cross-check per flagged session, and as ratios to IM over all usable R-a sessions:
      atm_recomp = atmiv × stock_price × √(1/252) and vol50_recomp = vol50 × stock_price × √(1/252), from the first
        orats_monies_minute row with snapshot_pt >= 06:33 PT and the smallest dte > 0 (ticker SPX/SPXW; the
        monies stock_price stands in for table_spot);
      straddle = call mid + put mid of the 0DTE PM strike nearest stock_price that has both legs, at the first
        orats_options_minute snapshot in [06:33, 06:45] PT (None where not cached);
      range = ES RTH high − low from the session's flow.
  - A17.7 disclosed as the second T0 read on R-a.

Usage: apps/web/.venv/bin/python -u scripts/cr_cb_t0_ra_a17.py | tee scripts/logs/cr_cb/t0_ra_a17.log
"""
from __future__ import annotations

import json
import os
import signal
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn  # noqa: E402
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.level_placement import (  # noqa: E402
    bucket_levels, es_snapshot_bar_start_utc, place_levels)
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day  # noqa: E402

signal.alarm(3 * 3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2024, 4, 3), date(2024, 12, 31)
LIMIT_DATE = date(2025, 1, 1)
TICK = 0.25
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N = 30, 40e9, 5.0, 1.0, 6
ARM_IM, ARM_MIN, BAND, IM_FLAG = 0.10, 4.0, 2.0, 25.0                   # A17.2, A17.4
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 4
O_C = {("put", 1): 9.75, ("put", -1): 8.25, ("call", 1): 6.25, ("call", -1): 5.25}
ESC_IM, ESC_CAP, DWELL_S = 0.25, 120, 3600
SEED_BOOT, N_BOOT, ALPHA = 20260920, 10_000, 0.025
CELLS = [("put", 1), ("put", -1), ("call", 1), ("call", -1)]
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cb"

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '600s'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def third_friday(y, m):
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d):
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def pt(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def ep(x):
    return int(x.timestamp())


SESSIONS, d = [], D_FROM
while d <= D_TO:
    if is_trading_day(d):
        SESSIONS.append(d)
    d += timedelta(days=1)
QEXP = {third_friday(2024, m) for m in (3, 6, 9, 12)}
CAP_TS = pt(D_TO, 13, 0)
assert CAP_TS < datetime(2025, 1, 1, tzinfo=UTC)

# ----------------------------------------------------------------------------------------------- loads
sym = q("""
    SELECT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date AS d, symbol, sum(volume) AS vol
    FROM es_flow_1s_databento
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2
""", (pt(prev_trading_day(D_FROM), 0, 0), CAP_TS)).sort_values("vol")
assert sym["d"].max() < LIMIT_DATE
SYM = dict(zip(sym["d"], sym["symbol"]))


def load_flow(d_, symbol):
    f = q("""SELECT ts_utc, open, high, low, close FROM es_flow_1s_databento
             WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s ORDER BY ts_utc""",
          (symbol, pt(d_, 6, 15), pt(d_, 13, 0), CAP_TS))
    return dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64), o=f["open"].to_numpy(float),
                h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))


FLOW = {d_: load_flow(d_, SYM[d_]) for d_ in SESSIONS if d_ in SYM}

oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + %s
""", (D_FROM - timedelta(days=7), D_TO, DTE_MAX + 7))
assert oig["trade_date"].max() < LIMIT_DATE
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d_ <= e <= d_ + timedelta(days=DTE_MAX) for e, d_ in zip(oig["expir_date"], oig["d"])]]
oig["call"] = oig["gex_call"].astype(float)
oig["put"] = -oig["gex_put"].abs().astype(float)

feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
feat = feat[~feat.index.duplicated()]

starts = {es_snapshot_bar_start_utc(prev_trading_day(d_)): d_ for d_ in SESSIONS}
assert max(starts) < datetime(2025, 1, 1)                     # naive UTC bar starts
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)", (list(starts),))
PTGT = {starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
        if c is not None and np.isfinite(c) and c > 0}

# flow close of A's front symbol in the 15:45 ET minute
ACHK = {}
for st, d_ in starts.items():
    a_ = prev_trading_day(d_)
    if a_ not in SYM:
        continue
    r = q("""SELECT close FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
             ORDER BY ts_utc DESC LIMIT 1""", (SYM[a_], st.replace(tzinfo=UTC), st.replace(tzinfo=UTC) + timedelta(minutes=1)))
    if len(r):
        ACHK[d_] = (SYM[a_], float(r["close"][0]))


def rth_slice(F, d_):
    return np.searchsorted(F["t"], ep(pt(d_, 6, 30))), np.searchsorted(F["t"], ep(pt(d_, 13, 0)))


OPEN, LAST = {}, {}
for d_, F in FLOW.items():
    a, b = rth_slice(F, d_)
    if b > a:
        OPEN[d_], LAST[d_] = float(F["o"][a]), int(F["t"][b - 1])

U, funnel = list(SESSIONS), []
for name, rule in [
    ("roll week", lambda d_: roll_week(d_)),
    ("no ES 15:45 ET bar on A", lambda d_: d_ not in PTGT),
    ("quarterly-expiry Friday", lambda d_: d_ in QEXP),
    ("no Databento RTH flow", lambda d_: d_ not in OPEN),
    ("early close (last RTH print < 12:55 PT)", lambda d_: LAST[d_] < ep(pt(d_, 12, 55))),
    ("no canonical IM / IM <= 0", lambda d_: not (d_ in feat.index and pd.notna(feat[d_]) and feat[d_] > 0)),
    ("no orats_oi_gamma rows", lambda d_: d_ not in set(oig["d"])),
    ("contract check (A symbol != D symbol, or |P_target − A flow 15:45 close| > 2)",
     lambda d_: d_ not in ACHK or ACHK[d_][0] != SYM[d_] or abs(ACHK[d_][1] - PTGT[d_]) > 2.0),
]:
    rm = [d_ for d_ in U if rule(d_)]
    U = [d_ for d_ in U if not rule(d_)]
    funnel.append([name, len(rm), len(U), [str(x) for x in rm]])
USABLE = U
IM = {d_: float(feat[d_]) for d_ in USABLE}
SIDX = {d_: k for k, d_ in enumerate(USABLE)}


# ---------------------------------------------------------------------------------------------- levels
def select_levels(g, ptgt, o, im):
    g = place_levels(g, ptgt)
    g = g[g["level_corrected"].notna()].copy()
    g["level"] = bucket_levels(g["level_corrected"].to_numpy(), 1.0)
    bk = g.groupby("level").agg(call=("call", "sum"), put=("put", "sum")).reset_index()
    bk["mag"] = bk["call"].abs() + bk["put"].abs()
    bk["net"] = bk["call"] + bk["put"]
    bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
    kept = []
    for r in bk.itertuples():
        if all(abs(r.level - k.level) >= SPACING for k in kept):
            kept.append(r)
    win = [k for k in kept if abs(k.level - o) <= WIN_IM * im]
    return [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:TOP_N]]


LEVELS = {d_: select_levels(g, PTGT[d_], OPEN[d_], IM[d_]) for d_, g in oig[oig["d"].isin(USABLE)].groupby("d")}


def placebo(d_):
    o, im = OPEN[d_], IM[d_]
    real = np.array([x["L"] for x in LEVELS[d_]])
    pool = {}
    for x in LEVELS[d_]:
        s = np.sign(x["L"] - o)
        lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
        cand = np.arange(lo, hi + 0.5, PLC_BIN)
        cand = cand[(np.sign(cand - o) == s) & (np.min(np.abs(cand[:, None] - real[None, :]), axis=1) >= PLC_SEP)]
        cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
        for b in cand:
            pool.setdefault(float(b), dict(L=float(b), mag=x["mag"], net=x["net"]))
    return list(pool.values())


PLV = {d_: placebo(d_) for d_ in USABLE}


# ------------------------------------------------------------------------------------------------ tests
def tests_for(F, d_, L, im):
    a, b = rth_slice(F, d_)
    h, l = F["h"], F["l"]
    arm = max(ARM_IM * im, ARM_MIN)
    assert arm > BAND
    below = np.nonzero(h[a:b] <= L - arm)[0] + a
    above = np.nonzero(l[a:b] >= L + arm)[0] + a
    hit_b = np.nonzero(h[a:b] >= L - BAND)[0] + a
    hit_a = np.nonzero(l[a:b] <= L + BAND)[0] + a
    out, pos = [], a

    def nxt(arr, k):
        i = np.searchsorted(arr, k)
        return int(arr[i]) if i < len(arr) else None

    while True:
        ib, ia = nxt(below, pos), nxt(above, pos)
        if ib is None and ia is None:
            break
        if ia is None or (ib is not None and ib < ia):
            jt, side = nxt(hit_b, ib), 1
        else:
            jt, side = nxt(hit_a, ia), -1
        if jt is None:
            break
        out.append((jt, side))
        pos = jt + 1
    return out[:1]                                  # A17.1: first test only


def measures(levels):
    rows = []
    for d_ in USABLE:
        F = FLOW[d_]
        _, b = rth_slice(F, d_)
        t, h, l, c = F["t"], F["h"], F["l"], F["c"]
        close13 = ep(pt(d_, 13, 0))
        for lv in levels[d_]:
            ty = "put" if lv["net"] < 0 else "call"
            for j, side in tests_for(F, d_, lv["L"], IM[d_]):
                t0 = int(t[j])
                esc = np.nonzero((h[j:b] >= lv["L"] + ESC_IM * IM[d_]) | (l[j:b] <= lv["L"] - ESC_IM * IM[d_]))[0]
                esc_min = (t[j + int(esc[0])] - t0) / 60 if len(esc) else np.inf
                cap = min(ESC_CAP, (close13 - t0) / 60)
                O = O_C[(ty, side)]
                g = np.arange(t0, min(t0 + DWELL_S, close13))
                kk = np.searchsorted(t, g, side="right") - 1
                px = c[np.clip(kk, 0, None)]
                rows.append(dict(d=d_, L=lv["L"], type=ty, side=side, esc=min(esc_min, cap), censored=not np.isfinite(esc_min),
                                 capped=bool(esc_min > cap), dwell=int(np.sum(np.abs(px - lv["L"]) <= O))))
    return pd.DataFrame(rows)


R = measures(LEVELS)
P = measures(PLV)

# ------------------------------------------------------------------------------------------- statistics
nS = len(USABLE)
W = np.random.default_rng(SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)
FLAG = np.array([IM[x] < IM_FLAG for x in USABLE])


def sess_means(E, m):
    s_ = np.zeros((nS, 2))
    np.add.at(s_, ([SIDX[x] for x in E["d"]], 0), E[m].to_numpy(float))
    np.add.at(s_, ([SIDX[x] for x in E["d"]], 1), 1)
    pres = s_[:, 1] > 0
    mean = np.where(pres, s_[:, 0] / np.maximum(s_[:, 1], 1), 0.0)
    return mean, pres.astype(float)


def compare(sel, m, mask=None):
    keep = np.ones(nS) if mask is None else mask.astype(float)
    (mr, pr), (mp, pp) = sess_means(sel(R), m), sess_means(sel(P), m)
    pr, pp = pr * keep, pp * keep
    Wk = W * keep
    with np.errstate(invalid="ignore", divide="ignore"):
        br = (Wk @ (mr * pr)) / (Wk @ pr)
        bp = (Wk @ (mp * pp)) / (Wk @ pp)
        bd = br - bp
    rv = (mr * pr).sum() / pr.sum() if pr.sum() else np.nan
    pv = (mp * pp).sum() / pp.sum() if pp.sum() else np.nan
    lo, hi = np.nanpercentile(bd, [2.5, 97.5])
    return dict(n_real_tests=int(len(sel(R)) if mask is None else sel(R)["d"].map(lambda x: keep[SIDX[x]] > 0).sum()),
                n_placebo_tests=int(len(sel(P)) if mask is None else sel(P)["d"].map(lambda x: keep[SIDX[x]] > 0).sum()),
                n_real_sessions=int(pr.sum()), n_placebo_sessions=int(pp.sum()),
                real=round(float(rv), 4), placebo=round(float(pv), 4), diff=round(float(rv - pv), 4),
                lo=round(float(lo), 4), hi=round(float(hi), 4),
                p_one_sided=max(float(np.nanmean(bd <= 0)), 1.0 / N_BOOT))


HYP = {f"{ty}/{'below' if s == 1 else 'above'}": (lambda E, ty=ty, s=s: E[(E["type"] == ty) & (E["side"] == s)])
       for ty, s in CELLS}
HYP["pooled"] = lambda E: E


def holm(block):
    order = sorted(block, key=lambda k: block[k]["p_one_sided"])
    stop = False
    for i, k in enumerate(order):
        thr = ALPHA / (len(order) - i)
        rej = (not stop) and block[k]["p_one_sided"] <= thr
        stop = stop or not rej
        b = block[k]
        b.update(holm_threshold=thr, holm_rejected=bool(rej), passed=bool(rej and b["lo"] > 0),
                 verdict="pass" if (rej and b["lo"] > 0) else ("refuted" if b["hi"] < 0 else "inconclusive"))


RES = dict(funnel=funnel, sessions=dict(n=nS, first=str(USABLE[0]), last=str(USABLE[-1]),
                                        symbols=sorted({SYM[x] for x in USABLE})),
           levels=dict(real=sum(len(v) for v in LEVELS.values()), placebo=sum(len(v) for v in PLV.values()),
                       real_put=sum(x["net"] < 0 for v in LEVELS.values() for x in v),
                       real_call=sum(x["net"] > 0 for v in LEVELS.values() for x in v)),
           tests=dict(real=len(R), placebo=len(P), real_censored=int(R["censored"].sum()),
                      placebo_censored=int(P["censored"].sum()), real_capped=int(R["capped"].sum()),
                      placebo_capped=int(P["capped"].sum()),
                      real_per_session=dict(R.groupby("d").size().describe().round(2)),
                      placebo_per_session=dict(P.groupby("d").size().describe().round(2))),
           primary={k: compare(f, "esc") for k, f in HYP.items()},
           secondary={k: compare(f, "dwell") for k, f in HYP.items()})
holm(RES["primary"])
RES["A17_pass"] = any(v["passed"] for v in RES["primary"].values())
RES["verdict"] = "pass" if RES["A17_pass"] else "fail"
RES["low_im"] = dict(n=int(FLAG.sum()), sessions={str(x): round(IM[x], 3) for x in USABLE if IM[x] < IM_FLAG})
RES["readout_excl_low_im"] = dict(primary={k: compare(f, "esc", ~FLAG) for k, f in HYP.items()},
                                  secondary={k: compare(f, "dwell", ~FLAG) for k, f in HYP.items()})


# ------------------------------------------------------------------------------------- A17.5 IM check
def im_check(d_):
    mm = q("""SELECT snapshot_pt, dte, atmiv::float a, vol50::float v, stock_price::float s FROM orats_monies_minute
              WHERE ticker IN ('SPX', 'SPXW') AND trade_date = %s AND snapshot_pt >= %s AND dte > 0
              ORDER BY dte, snapshot_pt LIMIT 1""", (str(d_), datetime.combine(d_, time(6, 33))))
    out = dict(im=IM[d_])
    if len(mm):
        sp = mm["s"][0]
        out.update(monies_t=str(mm["snapshot_pt"][0]), dte=int(mm["dte"][0]), spot=sp,
                   atm_recomp=mm["a"][0] * sp * np.sqrt(1 / 252), vol50_recomp=mm["v"][0] * sp * np.sqrt(1 / 252))
    op = q("""SELECT snapshot_pt, strike::float k, option_type, bid_price::float b, ask_price::float a,
                     stock_price::float s
              FROM orats_options_minute
              WHERE ticker IN ('SPX', 'SPXW') AND trade_date_d = %s AND expir_date_d = %s AND expiry_tod = 'pm'
                AND snapshot_pt >= %s AND snapshot_pt <= %s""",
           (d_, d_, datetime.combine(d_, time(6, 33)), datetime.combine(d_, time(6, 45))))
    out["straddle"] = None
    if len(op):
        t0 = op["snapshot_pt"].min()
        o = op[op["snapshot_pt"] == t0].copy()
        o["kind"] = o["option_type"].astype(str).str[0].str.lower()
        o["mid"] = (o["b"] + o["a"]) / 2
        piv = o.pivot_table(index="k", columns="kind", values="mid", aggfunc="first").dropna()
        if len(piv) and {"c", "p"} <= set(piv.columns):
            k_ = piv.index[np.argmin(np.abs(piv.index - o["s"].iloc[0]))]
            out.update(straddle=float(piv.loc[k_, "c"] + piv.loc[k_, "p"]), straddle_strike=float(k_),
                       straddle_t=str(t0))
    F = FLOW[d_]
    a, b = rth_slice(F, d_)
    out["es_range"] = float(F["h"][a:b].max() - F["l"][a:b].min())
    return out


IMC = {x: im_check(x) for x in USABLE}
RES["im_check_flagged"] = {str(x): IMC[x] for x in USABLE if IM[x] < IM_FLAG}


def ratio_dist(key, flagged=None):
    v = [IMC[x][key] / IMC[x]["im"] for x in USABLE if IMC[x].get(key) is not None
         and (flagged is None or (IM[x] < IM_FLAG) == flagged)]
    if not v:
        return dict(n=0)
    v = np.array(v)
    return dict(n=len(v), p10=round(float(np.percentile(v, 10)), 3), p50=round(float(np.median(v)), 3),
                p90=round(float(np.percentile(v, 90)), 3))


RES["im_check_ratios_all"] = {k: ratio_dist(k) for k in ("atm_recomp", "vol50_recomp", "straddle", "es_range")}
RES["im_check_ratios_unflagged"] = {k: ratio_dist(k, False) for k in ("atm_recomp", "vol50_recomp", "straddle", "es_range")}
RES["im_dist"] = dict(p10=round(float(np.percentile(list(IM.values()), 10)), 2),
                      p50=round(float(np.median(list(IM.values()))), 2),
                      p90=round(float(np.percentile(list(IM.values()), 90)), 2))

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "t0_ra_a17.json").write_text(json.dumps(RES, indent=1, default=str))
print(json.dumps({k: v for k, v in RES.items() if k != "funnel"}, indent=1, default=str))
for f_ in funnel:
    print(f_[:3])
for title, blk in (("PRIMARY time to escape, session-weighted (min, real − placebo)", RES["primary"]),
                   ("READOUT excl. IM < 25", RES["readout_excl_low_im"]["primary"])):
    print(f"\n{title}:")
    for k, v in blk.items():
        print(f"  {k:12s} sess {v['n_real_sessions']}/{v['n_placebo_sessions']} tests {v['n_real_tests']}/"
              f"{v['n_placebo_tests']}  {v['real']:.2f} vs {v['placebo']:.2f}  diff {v['diff']:+.2f} "
              f"[{v['lo']:+.2f}, {v['hi']:+.2f}]  p {v['p_one_sided']:.4f}"
              + (f"  Holm thr {v['holm_threshold']:.4f}  {v['verdict']}" if "verdict" in v else ""))
print("A17:", RES["verdict"].upper())
