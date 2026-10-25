#!/usr/bin/env python3
"""CR-BY run — move, pause, then which way: T1, T2, S1–S9. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BY-move-pause-which-way.md (L1–L9, Amendment A1, Step 0 under A1: f = 1/2). One run. Every
statistic is computed and written to scripts/logs/cr_by/results.json before anything is printed.

Universe, impulses, A1 box finder, eligibility, P, placebo and the L5 race are copied unchanged from
scripts/cr_by_step0_a1.py (commit c101f83); the script asserts the Step 0 counts (88 sessions, 1,831 impulses,
608 boxes 203/202/203, 475 placebo boxes) before any race. Additional definitions fixed here before the run:
  - L5 race on event and placebo boxes: from the row after t_B to 13:00 PT; target distance 0.5·I (updated I).
    Same-second -> tie (excluded, counted); neither by 13:00 -> unresolved (excluded, counted). Race-window gap
    rule (A2-style, L2): drop if [t_B, resolution or 13:00] has an RTH gap > 15 s; counted per tercile.
  - T1: reversal rate high − low P tercile (resolved boxes). Pass: >= +8, CI lower > 0, >= 150 resolved per tercile.
  - T2: reversal rate of high-P event boxes − high-P placebo boxes. Pass: >= +8, CI lower > 0.
  - L8 bootstrap: session-cluster, 10 000 multinomial resamples, seed 20260920; one weight matrix for every
    statistic; event and placebo resampled jointly by session.
  - S1 / S2: minutes from t_B to resolution (resolved boxes), share resolved within 30 / 60 / 120 min and share
    unresolved / tied, as a share of boxes kept after the gap rule; by tercile, event and placebo.
  - S3 (a): stop entry 1 tick beyond the box in the T1-predicted direction (high P -> reversal side, low P ->
    continuation side; mid tercile not traded), filled + 1 tick slippage; target entry ± 0.5·I (hit on a trade
    1 tick through); stop at the opposite box boundary ∓ 1 tick, filled ∓ 1 tick. (b): the same entry, 8-tick
    target and 8-tick stop. Entry row counts a stop, not a target; same row both -> loss; no trigger by 13:00 ->
    no trade; unresolved -> excluded; race-window gap rule. Net expectancy per trade = mean P&L − 0.36 tick;
    break-even win rate = (avg loss + 0.36) / (avg win + avg loss). Event and placebo boxes.
  - S4: the box window is extended (start fixed at t_B − 300) row by row while range <= f·I_w and the A1 retrace
    rule hold; final length = last qualifying row − window start. Holding at 10 / 20 min counted. P recomputed
    over the extended box; the L5 race then runs from the last qualifying row with the extended box boundaries
    and I_w; T1-style difference by final length (< 10, 10–20, >= 20 min) using the L6 cut points.
  - S5: I < 10, 10–20, >= 20 pts (placebo by the parent's I). S6: down vs up impulse (placebo by its own
    direction). S7: box range ± 2 pts contains a CR-BW A1.1 level (>= 40 $bn), vs not; and >= 100 $bn.
    T1 and T2 within each.
  - S8: the same T1 / T2 at f = 1/3 (own boxes, cut points and placebo).
  - S9: share of boxes whose 13:00 PT close is on the reversal side of the box (down impulse: close > box high),
    by tercile, event vs placebo; high − low and high event − high placebo with CI.

Usage: apps/web/.venv/bin/python -u scripts/cr_by_pause.py | tee scripts/logs/cr_by/run.log
"""
from __future__ import annotations

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
from packages.shared.level_placement import es_snapshot_bar_start_utc  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day  # noqa: E402

pd.set_option("display.width", 220)
signal.alarm(3 * 3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2025, 12, 26), date(2026, 6, 5)
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
TICK, GAP_MAX, K = 0.25, 15, 3.0
COOLDOWN = 900
BOX_W, SEARCH, F_LIST = 300, 1800, (1 / 3, 1 / 2)
EXT_FRAC, RET_FRAC, TGT_FRAC = 0.25, 0.5, 0.5
PLC_SPAN, PLC_PRIOR, PLC_N = 3600, 1800, 5
SCALE_N, SCALE_MIN = 20, 5
SEED, SEED_BOOT, N_SPLIT, N_BOOT = 20261025, 20260920, 200, 2_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '600s'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100, flush=True)


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d: date) -> bool:
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def trading_days(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def pt(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def ep(x: datetime) -> int:
    return int(x.timestamp())


def hms(t: int) -> str:
    return datetime.fromtimestamp(int(t), PT).strftime("%H:%M:%S")


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2025, 2026) for m in (3, 6, 9, 12)}

# ---------------------------------------------------------------------------------------------- loads
sym = q("""
    SELECT trade_date, symbol, sum(volume) AS vol
    FROM ironbeam_es_flow_1s
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2
""", (pt(D_FROM, 0, 0), HOLDOUT_TS))
assert not sym.duplicated("trade_date").any(), "more than one RTH symbol on a session"
SYM = dict(zip(sym["trade_date"], sym["symbol"]))

FLOW: dict[date, dict] = {}
for d in SESSIONS:
    if d not in SYM:
        continue
    f = q("""
        SELECT ts_utc, volume, buy_vol, sell_vol, open, high, low, close
        FROM ironbeam_es_flow_1s
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (SYM[d], pt(d, 6, 30), pt(d, 13, 0), HOLDOUT_TS))
    if f.empty:
        continue
    FLOW[d] = dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64),
                   vol=f["volume"].to_numpy(np.int64), buy=f["buy_vol"].to_numpy(np.int64),
                   sell=f["sell_vol"].to_numpy(np.int64), o=f["open"].to_numpy(float),
                   h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))

oig_days = set(q("""SELECT DISTINCT trade_date FROM orats_oi_gamma
                    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s""",
                 (D_FROM - timedelta(days=7), D_TO))["trade_date"])
oig_days = {t if is_trading_day(t) else next_trading_day(t) for t in oig_days}
feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im
    FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
feat = feat[~feat.index.duplicated()]
es_starts = {es_snapshot_bar_start_utc(prev_trading_day(d)): d for d in SESSIONS}
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
       (list(es_starts.keys()),))
PTGT = {es_starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
        if c is not None and np.isfinite(c) and c > 0}


def max_gap(d, t0, t1):
    f = FLOW[d]["t"]
    t0 = max(int(t0), ep(pt(d, 6, 30)))
    t1 = min(int(t1), ep(pt(d, 13, 0)))
    if t1 <= t0:
        return 0
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(np.diff(np.concatenate([[t0], f[a:b], [t1]])).max())


ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in PTGT}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
SCALE_ELIG = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c and d in FLOW]


def abs_changes(d, w):
    f = FLOW[d]
    t0 = ep(pt(d, 6, 30))
    out = []
    for k in range(23400 // w):
        ws, we = t0 + w * k, t0 + w * (k + 1)
        a, b = np.searchsorted(f["t"], ws), np.searchsorted(f["t"], we)
        if a == 0 or b <= a or max_gap(d, ws, we) > GAP_MAX:
            continue
        out.append(abs(f["c"][b - 1] - f["c"][a - 1]))
    return np.array(out)


CH = {d: abs_changes(d, 300) for d in SCALE_ELIG}
SIG = {}
for d in SESSIONS:
    prior = [x for x in SCALE_ELIG if x < d][-SCALE_N:]
    if len(prior) >= SCALE_MIN:
        SIG[d] = float(np.median(np.concatenate([CH[x] for x in prior])))
USABLE = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c and d in FLOW and d in feat.index
          and pd.notna(feat[d]) and feat[d] > 0 and d in oig_days and d in SIG]

PREP = {}
for d in USABLE:
    f = FLOW[d]
    idx = pd.to_datetime(f["t"], unit="s")
    PREP[d] = dict(lo300=pd.Series(f["l"], idx).rolling("300s", closed="both").min().to_numpy(),
                   hi300=pd.Series(f["h"], idx).rolling("300s", closed="both").max().to_numpy(),
                   whi=pd.Series(f["h"], idx).rolling(f"{BOX_W}s").max().to_numpy(),
                   wlo=pd.Series(f["l"], idx).rolling(f"{BOX_W}s").min().to_numpy(),
                   inwin=(f["t"] >= ep(pt(d, 6, 35))) & (f["t"] <= ep(pt(d, 12, 45))),
                   net=np.concatenate([[0], np.cumsum(f["buy"] - f["sell"])]),
                   vol=np.concatenate([[0], np.cumsum(f["vol"])]))


def cum_between(d, key, t0, t1):
    f, c = FLOW[d]["t"], PREP[d][key]
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(c[b] - c[a])


# -------------------------------------------------------------------------------------------- impulses
IMP, COND = [], {}
for d in USABLE:
    f, p = FLOW[d], PREP[d]
    s = SIG[d]
    down = p["inwin"] & (f["l"] == p["lo300"]) & (f["l"] <= p["hi300"] - K * s)
    up = p["inwin"] & (f["h"] == p["hi300"]) & (f["h"] >= p["lo300"] + K * s)
    COND[d] = f["t"][down | up]
    for dirn, mask in ((-1, down), (1, up)):
        last = -10**12
        for i in np.nonzero(mask)[0]:
            t = int(f["t"][i])
            if t - last < COOLDOWN:
                continue
            last = t
            E = f["l"][i] if dirn == -1 else f["h"][i]
            opp = p["hi300"][i] if dirn == -1 else p["lo300"][i]
            IMP.append(dict(d=d, i=int(i), t=t, dirn=dirn, E=float(E), I=float(abs(E - opp)), opp=float(opp)))
IMP = pd.DataFrame(IMP).sort_values(["d", "t"]).reset_index(drop=True)



def find_box(x, frac):
    """A1: first 300 s window ending in [t + 300, t + 1800] with no retrace violation up to its end and
    range <= frac·I_w."""
    d = x.d
    f, p = FLOW[d], PREP[d]
    i0 = x.i
    end = np.searchsorted(f["t"], min(x.t + SEARCH, ep(pt(d, 13, 0))), side="right")
    if x.dirn == -1:
        runE = np.minimum.accumulate(f["l"][i0:end])
        retr = f["h"][i0:end] - runE
    else:
        runE = np.maximum.accumulate(f["h"][i0:end])
        retr = runE - f["l"][i0:end]
    Iu = np.abs(x.opp - runE)
    viol = np.nonzero(retr > RET_FRAC * Iu)[0]
    v_t = int(f["t"][i0 + viol[0]]) if len(viol) else 10**12
    a = np.searchsorted(f["t"], x.t + BOX_W)
    js = np.arange(a, end)
    ks = np.searchsorted(f["t"], f["t"][js] - BOX_W, side="right") - 1 - i0
    Ew = runE[np.clip(ks, 0, None)] if len(js) else np.array([])
    Iw = np.abs(x.opp - Ew)
    ra = (p["whi"][js] - p["wlo"][js]) <= frac * Iw
    rt = f["t"][js] < v_t
    both = np.nonzero(ra & rt)[0]
    if len(both):
        k = int(both[0])
        j = int(js[k])
        e = int(f["t"][j])
        V = cum_between(d, "vol", e - BOX_W + 1, e)
        D = x.dirn * cum_between(d, "net", e - BOX_W + 1, e)
        return dict(found=True, reason="", j=j, tB=e, box_hi=float(p["whi"][j]), box_lo=float(p["wlo"][j]),
                    Ew=float(Ew[k]), Iw=float(Iw[k]), V=V, P=D / V if V > 0 else np.nan,
                    gap=max_gap(d, e - BOX_W, e))
    reason = "both" if not ra.any() and not rt.any() else "retrace" if ra.any() else "range"
    return dict(found=False, reason=reason)


def boxes(frac):
    B = pd.DataFrame([{**x._asdict(), **find_box(x, frac)} for x in IMP.itertuples(index=False)])
    B["elig"] = B["found"] & (B["gap"] <= GAP_MAX) & (B["V"] > 0)
    return B


def terc_of(x, cut):
    return np.where(x <= cut[0], 0, np.where(x <= cut[1], 1, 2))



def placebo(BE, frac):
    rng = np.random.default_rng(SEED)
    rows, none = [], 0
    for x in BE.sort_values(["d", "tB"]).itertuples():
        d = x.d
        f, p = FLOW[d], PREP[d]
        lo_t = max(x.tB - PLC_SPAN, ep(pt(d, 6, 30)) + BOX_W)
        hi_t = min(x.tB + PLC_SPAN, ep(pt(d, 13, 0)))
        a, b = np.searchsorted(f["t"], lo_t), np.searchsorted(f["t"], hi_t, side="right")
        idx = np.arange(a, b)
        te = f["t"][idx]
        keep = np.abs(te - x.tB) >= BOX_W
        ca = COND[d]
        k1, k2 = np.searchsorted(ca, te - BOX_W - PLC_PRIOR), np.searchsorted(ca, te, side="right")
        keep &= (k2 - k1) == 0
        s0 = np.searchsorted(f["t"], te - BOX_W, side="right") - 1
        s1 = np.searchsorted(f["t"], te - 2 * BOX_W, side="right") - 1
        keep &= (s0 >= 0) & (s1 >= 0)
        Ep = np.where(s0 >= 0, f["c"][np.maximum(s0, 0)], np.nan)
        dirn = np.sign(Ep - np.where(s1 >= 0, f["c"][np.maximum(s1, 0)], np.nan))
        keep &= dirn != 0
        hi, lo = p["whi"][idx], p["wlo"][idx]
        keep &= (hi - lo) <= frac * x.I
        cand = []
        for k in np.nonzero(keep)[0]:
            e = int(te[k])
            if max_gap(d, e - BOX_W, e) > GAP_MAX:
                continue
            V = cum_between(d, "vol", e - BOX_W + 1, e)
            if V <= 0:
                continue
            cand.append((int(idx[k]), e, int(dirn[k]), float(Ep[k]), V))
        chosen = []
        for k in rng.permutation(len(cand)):
            if all(abs(cand[k][1] - c[1]) >= BOX_W for c in chosen):
                chosen.append(cand[k])
            if len(chosen) == PLC_N:
                break
        if not chosen:
            none += 1
        for j, e, dn, Ep_, V in chosen:
            P_ = dn * cum_between(d, "net", e - BOX_W + 1, e) / V
            rows.append(dict(d=d, parent=x.pid, j=j, tB=e, dirn=dn, E=Ep_, I=x.I, box_hi=float(p["whi"][j]),
                             box_lo=float(p["wlo"][j]), P=P_))
    return pd.DataFrame(rows), none

# ===================================================================================== run (no printing)
import json  # noqa: E402

from packages.shared.level_placement import bucket_levels, place_levels  # noqa: E402

F_RUN, F_OTHER = 1 / 2, 1 / 3
COST_TICKS, N_BOOT_RUN, NEAR_PTS = 4.50 / 12.50, 10_000, 2.0
DTE_MAX, MAG_MIN, STRONG, SPACING, TOP_N = 30, 40e9, 100e9, 5.0, 6
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_by"
INFO: dict = dict(sessions=len(USABLE), impulses=len(IMP))
assert len(USABLE) == 88 and len(IMP) == 1831, INFO


def build(frac):
    Bx = boxes(frac)
    be = Bx[Bx["elig"]].copy().reset_index(drop=True)
    cut = np.percentile(be["P"], [100 / 3, 200 / 3])
    be["terc"] = terc_of(be["P"].to_numpy(), cut)
    be["pid"] = np.arange(len(be))
    be["I0"], be["E0"] = be["I"], be["E"]
    be["I"], be["E"] = be["Iw"], be["Ew"]
    pb, _ = placebo(be, frac)
    pb["terc"] = terc_of(pb["P"].to_numpy(), cut)
    return be, pb, cut


BE, PB, CUT = build(F_RUN)
INFO.update(eligible=len(BE), per_tercile=np.bincount(BE["terc"], minlength=3).tolist(), cut=CUT.tolist(),
            placebo=len(PB), placebo_per_tercile=np.bincount(PB["terc"], minlength=3).tolist())
# Step 0 under A1 consistency (commit c101f83): fail before any outcome if anything differs
assert len(BE) == 608 and INFO["per_tercile"] == [203, 202, 203] and len(PB) == 475, INFO


def race(d, j, dirn, bhi, blo, I):
    """L5 from the row after j: 1 reversal / 0 continuation / -1 same-second tie / -2 unresolved; end time."""
    f = FLOW[d]
    j = int(j)
    up_t, dn_t = bhi + TGT_FRAC * I, blo - TGT_FRAC * I
    ui = np.nonzero(f["h"][j + 1:] >= up_t)[0]
    di = np.nonzero(f["l"][j + 1:] <= dn_t)[0]
    if not len(ui) and not len(di):
        return -2, ep(pt(d, 13, 0))
    fu, fd = (ui[0] if len(ui) else 10**9), (di[0] if len(di) else 10**9)
    if fu == fd:
        return -1, int(f["t"][j + 1 + fu])
    return (1 if (fu < fd) == (dirn == -1) else 0), int(f["t"][j + 1 + min(fu, fd)])


def trade(d, j, long, bhi, blo, tgt_dist, stop_mode):
    """S3 stop-entry trade from the row after j. long: buy stop at box_hi + 1 tick, filled + 1 tick slippage.
    stop_mode 'box': stop at the opposite boundary ∓ 1 tick, filled ∓ 1 tick; 'fixed8': stop 8 ticks from entry.
    Target entry ± tgt_dist, hit on a trade 1 tick through. Entry row counts a stop, not a target; same row both
    -> loss. Returns (1 win / 0 loss / -2 unresolved / -3 no entry, pnl ticks before cost, entry t, end t)."""
    f = FLOW[d]
    j = int(j)
    if long:
        trig = bhi + TICK
        hit = np.nonzero(f["h"][j + 1:] >= trig)[0]
    else:
        trig = blo - TICK
        hit = np.nonzero(f["l"][j + 1:] <= trig)[0]
    if not len(hit):
        return -3, np.nan, None, None
    k = j + 1 + int(hit[0])
    E = trig + (TICK if long else -TICK)
    if stop_mode == "box":
        stp = (blo - TICK) if long else (bhi + TICK)
    else:
        stp = (E - 8 * TICK) if long else (E + 8 * TICK)
    sfill = stp - TICK if long else stp + TICK
    tgt = E + tgt_dist if long else E - tgt_dist
    win_t = (tgt - E) / TICK if long else (E - tgt) / TICK
    loss_t = (sfill - E) / TICK if long else (E - sfill) / TICK
    if (long and f["l"][k] <= stp) or (not long and f["h"][k] >= stp):
        return 0, loss_t, int(f["t"][k]), int(f["t"][k])
    if long:
        th, sh = f["h"][k + 1:] >= tgt + TICK, f["l"][k + 1:] <= stp
    else:
        th, sh = f["l"][k + 1:] <= tgt - TICK, f["h"][k + 1:] >= stp
    ti, si = np.nonzero(th)[0], np.nonzero(sh)[0]
    if not len(ti) and not len(si):
        return -2, np.nan, int(f["t"][k]), ep(pt(d, 13, 0))
    m = min(ti[0] if len(ti) else 10**9, si[0] if len(si) else 10**9)
    w = len(ti) and (not len(si) or ti[0] < si[0])
    return (1, win_t, int(f["t"][k]), int(f["t"][k + 1 + m])) if w else (0, loss_t, int(f["t"][k]), int(f["t"][k + 1 + m]))


def close_1300(d):
    f = FLOW[d]
    return f["c"][-1]


def annotate(E):
    rr = [race(x.d, x.j, x.dirn, x.box_hi, x.box_lo, x.I) for x in E.itertuples()]
    E["res"], E["end"] = [r[0] for r in rr], [r[1] for r in rr]
    E["gapdrop"] = [max_gap(x.d, x.tB, x.end) > GAP_MAX for x in E.itertuples()]
    E["mins"] = (E["end"] - E["tB"]) / 60.0
    cl = np.array([close_1300(d) for d in E["d"]])
    E["eod_rev"] = np.where(E["dirn"] == -1, cl > E["box_hi"], cl < E["box_lo"])
    for lab, tgt, mode in (("a", None, "box"), ("b", 8 * TICK, "fixed8")):
        out = []
        for x in E.itertuples():
            # high P -> reversal side, low P -> continuation side; mid: no trade
            if x.terc == 1:
                out.append((-4, np.nan, None, None))
                continue
            rev_long = x.dirn == -1
            long = rev_long if x.terc == 2 else not rev_long
            out.append(trade(x.d, x.j, long, x.box_hi, x.box_lo, TGT_FRAC * x.I if tgt is None else tgt, mode))
        E[f"tr_{lab}"] = [o[0] for o in out]
        E[f"pnl_{lab}"] = [o[1] for o in out]
        E[f"trgap_{lab}"] = [o[2] is not None and max_gap(x.d, x.tB, o[3]) > GAP_MAX
                             for x, o in zip(E.itertuples(), out)]
    E["sidx"] = [USABLE.index(d) for d in E["d"]]
    return E


BE, PB = annotate(BE), annotate(PB)

# ------------------------------------------------------------------------------------------ S7 levels
OPEN = {d: float(FLOW[d]["o"][0]) for d in USABLE}
IMd = {d: float(feat[d]) for d in USABLE}
ptg = {d: PTGT[d] for d in USABLE}
oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + %s
""", (D_FROM - timedelta(days=7), D_TO, DTE_MAX + 7))
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d <= e <= d + timedelta(days=DTE_MAX) for e, d in zip(oig["expir_date"], oig["d"])]]
oig["call"] = oig["gex_call"].astype(float)
oig["put"] = -oig["gex_put"].abs().astype(float)
LV = {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    g = place_levels(g, ptg[d])
    g = g[g["level_corrected"].notna()]
    g["level"] = bucket_levels(g["level_corrected"].to_numpy(), 1.0)
    bk = g.groupby("level").agg(call=("call", "sum"), put=("put", "sum")).reset_index()
    bk["mag"] = bk["call"].abs() + bk["put"].abs()
    bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
    kept = []
    for r in bk.itertuples():
        if all(abs(r.level - k.level) >= SPACING for k in kept):
            kept.append(r)
    win = [k for k in kept if abs(k.level - OPEN[d]) <= IMd[d]][:TOP_N]
    LV[d] = (np.array([k.level for k in win]), np.array([k.mag for k in win]))


def near(x, min_mag):
    lv, mg = LV.get(x.d, (np.array([]), np.array([])))
    m = (lv >= x.box_lo - NEAR_PTS) & (lv <= x.box_hi + NEAR_PTS) & (mg >= min_mag)
    return bool(m.any())


for E in (BE, PB):
    E["near40"] = [near(x, MAG_MIN) for x in E.itertuples()]
    E["near100"] = [near(x, STRONG) for x in E.itertuples()]

# -------------------------------------------------------------------------------------- S4 extension


def extend(x, frac):
    """Extend the box window (start fixed at tB − 300) while range <= frac·I_w and the retrace rule holds."""
    d = x.d
    f = FLOW[d]
    i0 = int(IMP.loc[(IMP["d"] == d) & (IMP["t"] == x.t), "i"].iloc[0])
    s = x.tB - BOX_W
    a = np.searchsorted(f["t"], s, side="right")
    end = len(f["t"])
    if x.dirn == -1:
        runE = np.minimum.accumulate(f["l"][i0:end])
        retr = f["h"][i0:end] - runE
    else:
        runE = np.maximum.accumulate(f["h"][i0:end])
        retr = runE - f["l"][i0:end]
    ok_rt = retr <= RET_FRAC * np.abs(x.opp - runE)
    hi = np.maximum.accumulate(f["h"][a:end])
    lo = np.minimum.accumulate(f["l"][a:end])
    ok = ((hi - lo) <= frac * x.I) & ok_rt[a - i0:]
    bad = np.nonzero(~ok)[0]
    last = (a + int(bad[0]) - 1) if len(bad) else end - 1
    last = max(last, x.j)
    e = int(f["t"][last])
    k = last - a
    V = cum_between(d, "vol", s + 1, e)
    P = x.dirn * cum_between(d, "net", s + 1, e) / V if V > 0 else np.nan
    return dict(len_min=(e - s) / 60.0, j_ext=last, bhi_ext=float(hi[k]), blo_ext=float(lo[k]), P_ext=P)


ext = pd.DataFrame([extend(x, F_RUN) for x in BE.itertuples()])
BE = pd.concat([BE, ext], axis=1)
rr = [race(x.d, x.j_ext, x.dirn, x.bhi_ext, x.blo_ext, x.I) for x in BE.itertuples()]
BE["res_ext"], BE["end_ext"] = [r[0] for r in rr], [r[1] for r in rr]
BE["gap_ext"] = [max_gap(x.d, int(FLOW[x.d]["t"][x.j_ext]), x.end_ext) > GAP_MAX for x in BE.itertuples()]
BE["terc_ext"] = terc_of(BE["P_ext"].to_numpy(), CUT)

# ------------------------------------------------------------------------------------------ bootstrap
nS = len(USABLE)
W = np.random.default_rng(SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT_RUN).astype(float)


def cnt(E, mask=None, res="res", gap="gapdrop", terc="terc"):
    m = ~E[gap] & (E[res] >= 0)
    if mask is not None:
        m &= mask
    e = E[m]
    c = np.zeros((nS, 3, 2))
    np.add.at(c, (e["sidx"].to_numpy(), e[terc].to_numpy(), 0), e[res].to_numpy())
    np.add.at(c, (e["sidx"].to_numpy(), e[terc].to_numpy(), 1), 1)
    return c


def rate(c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 0] / c[..., 1]


def ci(fn, *cs):
    obs = fn(*[c.sum(0) for c in cs])
    b = fn(*[np.einsum("bs,sij->bij", W, c) for c in cs])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=round(float(obs) * 100, 2), lo=round(float(lo) * 100, 2), hi=round(float(hi) * 100, 2))


def t1(c):
    r = rate(c)
    return r[..., 2] - r[..., 0]


def t2(ce, cp):
    return rate(ce)[..., 2] - rate(cp)[..., 2]


def block(E=None, P=None, me=None, mp=None, res="res", gap="gapdrop", terc="terc"):
    """T1 on event boxes, T1 on placebo boxes, T2; event columns res/gap/terc, placebo always the L5 columns."""
    E = BE if E is None else E
    P = PB if P is None else P
    ce, cp = cnt(E, me, res, gap, terc), cnt(P, mp)
    return dict(event_n=ce.sum(0)[:, 1].astype(int).tolist(), event_rev=(rate(ce.sum(0)) * 100).round(2).tolist(),
                placebo_n=cp.sum(0)[:, 1].astype(int).tolist(), placebo_rev=(rate(cp.sum(0)) * 100).round(2).tolist(),
                T1=ci(t1, ce), T1_placebo=ci(t1, cp), T2=ci(t2, ce, cp))


RES: dict = dict(info=INFO)
for name, E in (("event", BE), ("placebo", PB)):
    RES[f"counts_{name}"] = dict(
        gap_drop=[int(((E["terc"] == k) & E["gapdrop"]).sum()) for k in range(3)],
        tie=[int(((E["terc"] == k) & ~E["gapdrop"] & (E["res"] == -1)).sum()) for k in range(3)],
        unresolved=[int(((E["terc"] == k) & ~E["gapdrop"] & (E["res"] == -2)).sum()) for k in range(3)],
        resolved=[int(((E["terc"] == k) & ~E["gapdrop"] & (E["res"] >= 0)).sum()) for k in range(3)])
PRIM = block()
RES["primary"] = PRIM
n_ok = min(PRIM["event_n"]) >= 150
RES["T1_parts"] = dict(est_ge_8=PRIM["T1"]["est"] >= 8, ci_lo_gt_0=PRIM["T1"]["lo"] > 0, n_ge_150=n_ok)
RES["T2_parts"] = dict(est_ge_8=PRIM["T2"]["est"] >= 8, ci_lo_gt_0=PRIM["T2"]["lo"] > 0)
RES["T1_pass"], RES["T2_pass"] = all(RES["T1_parts"].values()), all(RES["T2_parts"].values())
RES["verdict"] = ("Both pass: candidate edge; next is one pre-registered holdout read" if RES["T1_pass"] and RES["T2_pass"]
                  else "T1 only: flow in a pause predicts direction, but not specifically after fast moves "
                       "(T2 underpowered: A/A split sd ≈ 12 pts)" if RES["T1_pass"]
                  else "T2 only (not a verdict-table case): recorded" if RES["T2_pass"]
                  else "Neither: recorded; the order-flow line on this data stops here")

# S1 time to outcome, S2 unresolved
s1 = {}
for name, E in (("event", BE), ("placebo", PB)):
    rows = []
    for k in range(3):
        e = E[(E["terc"] == k) & ~E["gapdrop"]]
        r = e[e["res"] >= 0]
        rows.append(dict(n=len(e), median_min=round(float(r["mins"].median()), 1) if len(r) else None,
                         p25=round(float(r["mins"].quantile(.25)), 1) if len(r) else None,
                         p75=round(float(r["mins"].quantile(.75)), 1) if len(r) else None,
                         within_30=round(float(((e["res"] >= 0) & (e["mins"] <= 30)).mean()) * 100, 2),
                         within_60=round(float(((e["res"] >= 0) & (e["mins"] <= 60)).mean()) * 100, 2),
                         within_120=round(float(((e["res"] >= 0) & (e["mins"] <= 120)).mean()) * 100, 2),
                         unresolved=round(float((e["res"] == -2).mean()) * 100, 2),
                         tie=round(float((e["res"] == -1).mean()) * 100, 2)))
    s1[name] = rows
RES["S1_S2"] = s1


# S3 trades
def s3(E, lab, tgt_desc):
    out = {}
    for k, side in ((2, "high P → reversal side"), (0, "low P → continuation side")):
        e = E[(E["terc"] == k) & ~E[f"trgap_{lab}"]]
        tr = e[e[f"tr_{lab}"] >= 0]
        wins, losses = tr[tr[f"tr_{lab}"] == 1][f"pnl_{lab}"], tr[tr[f"tr_{lab}"] == 0][f"pnl_{lab}"]
        net = tr[f"pnl_{lab}"] - COST_TICKS
        Wb, Lb = (wins.mean() if len(wins) else np.nan), (-losses.mean() if len(losses) else np.nan)
        out[side] = dict(boxes=len(e), no_entry=int((e[f"tr_{lab}"] == -3).sum()),
                         unresolved=int((e[f"tr_{lab}"] == -2).sum()), trades=len(tr),
                         win_rate=round(float((tr[f"tr_{lab}"] == 1).mean()) * 100, 2) if len(tr) else None,
                         avg_win_ticks=round(float(Wb), 2), avg_loss_ticks=round(float(Lb), 2),
                         net_exp_ticks=round(float(net.mean()), 3) if len(tr) else None,
                         breakeven_wr=round(float((Lb + COST_TICKS) / (Wb + Lb)) * 100, 2) if len(tr) else None)
    return dict(target=tgt_desc, **out)


RES["S3"] = {f"{name}_{lab}": s3(E, lab, desc) for name, E in (("event", BE), ("placebo", PB))
             for lab, desc in (("a", "0.5·I target, stop at opposite box boundary ∓ 1 tick"), ("b", "8-tick target and stop"))}

# S4 pause length
lens = pd.cut(BE["len_min"], [0, 10, 20, 10**6], right=False, labels=["<10", "10-20", ">=20"])
RES["S4"] = dict(holding_10=int((BE["len_min"] >= 10).sum()), holding_20=int((BE["len_min"] >= 20).sum()),
                 median_len_min=round(float(BE["len_min"].median()), 2),
                 buckets={str(b): block(me=(lens == b), res="res_ext", gap="gap_ext", terc="terc_ext")["T1"] | dict(
                     n=cnt(BE, lens == b, res="res_ext", gap="gap_ext", terc="terc_ext").sum(0)[:, 1].astype(int).tolist(),
                     rev=(rate(cnt(BE, lens == b, res="res_ext", gap="gap_ext", terc="terc_ext").sum(0)) * 100)
                     .round(2).tolist()) for b in ["<10", "10-20", ">=20"]})
# S5 move size, S6 direction, S7 levels
RES["S5"] = {lab: block(me=m(BE["I"]), mp=m(PB["I"])) for lab, m in
             (("I<10", lambda x: x < 10), ("10<=I<20", lambda x: (x >= 10) & (x < 20)), ("I>=20", lambda x: x >= 20))}
RES["S6"] = {lab: block(me=BE["dirn"] == v, mp=PB["dirn"] == v) for lab, v in (("after_down", -1), ("after_up", 1))}
RES["S7"] = dict(near40=block(me=BE["near40"], mp=PB["near40"]), not_near40=block(me=~BE["near40"], mp=~PB["near40"]),
                 near100=block(me=BE["near100"], mp=PB["near100"]),
                 counts=dict(near40=int(BE["near40"].sum()), near100=int(BE["near100"].sum())))
# S8 other f
BE3, PB3, CUT3 = build(F_OTHER)
BE3, PB3 = annotate(BE3), annotate(PB3)
RES["S8"] = dict(f="1/3", eligible=len(BE3), per_tercile=np.bincount(BE3["terc"], minlength=3).tolist(),
                 placebo=len(PB3), **block(E=BE3, P=PB3))


# S9 end of day
def s9():
    def c(E):
        m = ~E["gapdrop"]
        e = E[m]
        out = np.zeros((nS, 3, 2))
        np.add.at(out, (e["sidx"].to_numpy(), e["terc"].to_numpy(), 0), e["eod_rev"].to_numpy().astype(float))
        np.add.at(out, (e["sidx"].to_numpy(), e["terc"].to_numpy(), 1), 1)
        return out
    ce, cp = c(BE), c(PB)
    return dict(event=(rate(ce.sum(0)) * 100).round(2).tolist(), placebo=(rate(cp.sum(0)) * 100).round(2).tolist(),
                high_minus_low=ci(t1, ce), high_event_minus_high_placebo=ci(t2, ce, cp))


RES["S9"] = s9()

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "results.json").write_text(json.dumps(RES, indent=1, default=str))
try:
    BE.to_pickle(OUT_DIR / "events.pkl")
    PB.to_pickle(OUT_DIR / "placebo.pkl")
except Exception as exc:  # convenience dumps; results.json is the record
    print("event dump failed:", exc)

# ------------------------------------------------------------------------------------------- printing
print(json.dumps(RES, indent=1, default=str))
p = RES["primary"]
print(f"\nT1 (high − low P tercile reversal rate): {p['T1']['est']:+.2f} pts [{p['T1']['lo']:+.2f}, {p['T1']['hi']:+.2f}]; "
      f"resolved {p['event_n']} -> {'PASS' if RES['T1_pass'] else 'FAIL'} {RES['T1_parts']}")
print(f"T2 (high-P event − high-P placebo): {p['T2']['est']:+.2f} pts [{p['T2']['lo']:+.2f}, {p['T2']['hi']:+.2f}] -> "
      f"{'PASS' if RES['T2_pass'] else 'FAIL'} {RES['T2_parts']}")
print("VERDICT:", RES["verdict"])
