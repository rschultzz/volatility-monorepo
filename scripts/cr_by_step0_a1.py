#!/usr/bin/env python3
"""CR-BY Step 0 under Amendment A1 — move, pause, then which way. READ-ONLY: SELECTs only, no API calls.

Copy of scripts/cr_by_step0.py (commit 1883cc0) with L4 (b) replaced by A1.1. Changes are marked "A1".
  - A1 running extreme: from the impulse row t, runE(u) = min low (down) / max high (up) over rows [t, u];
    opp = the impulse's 300 s opposite extreme (hi300(t) down, lo300(t) up); I(u) = |opp − runE(u)|.
  - A1 retrace rule: a window ending at e qualifies only if no row u in [t, e] has retrace(u) > 0.5·I(u), with
    retrace(u) = high(u) − runE(u) (down) / runE(u) − low(u) (up), each row judged against the move size
    reached so far (I(u)). Once violated, no later window qualifies.
  - A1 updated I for a window: E_w = runE at the last row at or before the window start (e − 300);
    I_w = |opp − E_w|. (a) uses high − low <= f·I_w. L5 targets use I_w.
  - A1 failure reasons: "retrace" (range-qualifying windows exist, all after the first retrace violation);
    "range" (retrace-ok windows exist, none meets (a)); "both" (neither exists).
  - A1 placebo (L7 "same range and location rules"): the location rule is now the retrace rule from an
    impulse, which a placebo window does not have; within a window, retrace from the running extreme cannot
    exceed the window range, and range <= f·I <= 0.5·I, so the placebo rule reduces to (a) with the event's
    updated I. E' (price at window start) sets direction context only.

Spec: specs/CR-BY-move-pause-which-way.md (Step 0 gates 1–6; kickoff: f under L4 from counts only).

NO race is run at any event box. The L5 race runs only on placebo boxes inside the A/A splits, and only the
split-statistic distribution is printed.

Universe, σ5_d, impulse detector (k = 3.0), flow loader and gap rule are copied from scripts/cr_bx_step0.py /
scripts/cr_bx_capitulation.py (unmerged branch feat/CR-BX-capitulation-seller-exhaustion, commit 5906bd7).

Definitions fixed here before the run:
  - connection: get_backfill_db_conn() + default_transaction_read_only = on (asserted); statement_timeout 600 s;
    Python alarm 3 h. Holdout guard: sessions <= 2026-06-05; flow ts_utc < 2026-06-05 21:00 UTC.
  - universe and impulses exactly as CR-BX (asserted: 88 sessions, 1,831 impulses at k = 3.0).
  - L3: down impulse at row t: E = low(t), I = hi300(t) − low(t); up: E = high(t), I = high(t) − lo300(t).
  - L4 windows: 300 s windows (e − 300, e] ending at a flow row e; max high / min low over the window's rows.
    Search: e − 300 >= t (starts after the impulse extreme), e <= t + 1800, e <= 13:00 PT. Box = the first e with
    (a) high − low <= f·I and (b) location: down: low >= E − 0.25·I and high <= E + 0.5·I; up: high <= E + 0.25·I
    and low >= E − 0.5·I. t_B = e. Box = [min low, max high] of that window.
    Failure reasons for impulses with no box: "range" (no window meets (a) ∧ (b) but some window meets (b));
    "location" (some window meets (a), none meets (b)); "both" (no window meets either); "never jointly"
    (windows meet (a) and windows meet (b), but none both).
  - eligibility: RTH flow gap <= 15 s over [t_B − 300, t_B] and V_box > 0. (The race-window part of the gap
    rule needs the race end — an outcome — so for event boxes it is applied in the run; in the A/A it is applied.)
  - L6 P = D_box / V_box, D signed in the impulse direction (down: sell − buy), V = `volume` (buy + sell +
    unknown) over the window rows. Terciles of P over eligible boxes.
  - f: the smaller of 1/3, 1/2 with >= 150 eligible boxes in every P tercile.
  - L7 placebo: windows ending at rows e' with |e' − t_B| <= 3600 and |e' − t_B| >= 300 (no overlap with the event
    box), e' − 300 >= 06:30 PT, e' <= 13:00 PT; no L5 condition row (either direction, k = 3.0) in
    [e' − 300 − 1800, e'] (the 30 minutes before the window, and the window); E' = close of the last row at or
    before e' − 300; direction = sign(E' − close of the last row at or before e' − 600) (0 -> not eligible);
    I = the event's I; same (a) and (b) with E', and the same eligibility. Up to 5, pairwise |e'1 − e'2| >= 300,
    drawn at random (default_rng(20261025), events in time order). Placebo P terciles use the event cut points.
  - L5 race (A/A only): from the row after t_B to 13:00 PT. Down impulse: reversal when high >= box_high + 0.5·I,
    continuation when low <= box_low − 0.5·I; up mirrored. Same row both -> excluded; neither -> unresolved
    (excluded). Race-window gap rule: drop if [t_B, resolution or 13:00] has an RTH gap > 15 s.
  - Gate 6 A/A: 200 splits of the placebo boxes into two halves by parent event (default_rng(20261025) fresh;
    first half = ceil(n/2)); statistic [high − low P-tercile reversal rate](half 1) − (half 2), each placebo box
    in its own P tercile; session-cluster bootstrap 2 000 per split, one default_rng(20260920) stream.
    Pass: CI excludes 0 in <= 10 % of splits and |median| <= 3 pts.
  - Gate 5: default_rng(20261025) fresh, uniform over eligible event boxes at the chosen f.

Usage: apps/web/.venv/bin/python -u scripts/cr_by_step0_a1.py | tee scripts/logs/cr_by/step0_a1.log
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

hdr("GATE 1 — UNIVERSE AND IMPULSES")
print(f"usable sessions {len(USABLE)} ({USABLE[0]} .. {USABLE[-1]}); impulses at k = {K}: {len(IMP)} "
      f"(down {int((IMP['dirn'] == -1).sum())}, up {int((IMP['dirn'] == 1).sum())})")
ok1 = len(USABLE) == 88 and len(IMP) == 1831
print(f"GATE 1 (CR-BX universe 88 sessions, 1,831 impulses): {'PASS' if ok1 else 'FAIL — STOP'}")
if not ok1:
    sys.exit("STOP: gate 1")


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


hdr("GATE 2 — BOX COUNTS (no outcomes)")
BOX = {}
for frac in F_LIST:
    B = boxes(frac)
    BOX[frac] = B
    be = B[B["elig"]]
    cut = np.percentile(be["P"], [100 / 3, 200 / 3]) if len(be) >= 3 else [np.nan] * 2
    tc = np.bincount(terc_of(be["P"].to_numpy(), cut), minlength=3).tolist() if len(be) >= 3 else [len(be), 0, 0]
    rs = B.loc[~B["found"], "reason"].value_counts().to_dict()
    print(f"f = {frac:.3f}: impulses {len(B)}; with a box within 30 min {int(B['found'].sum())} "
          f"({B['found'].mean():.1%}); no box {int((~B['found']).sum())} by reason {rs}; gap drops "
          f"{int((B['found'] & (B['gap'] > GAP_MAX)).sum())}; V = 0 {int((B['found'] & ~(B['V'] > 0)).sum())}; "
          f"eligible {len(be)}; per P tercile {tc}")
F = next((fr for fr in F_LIST if BOX[fr]["elig"].sum() >= 3 and
          min(np.bincount(terc_of(BOX[fr].loc[BOX[fr]["elig"], "P"].to_numpy(),
                                  np.percentile(BOX[fr].loc[BOX[fr]["elig"], "P"], [100 / 3, 200 / 3])),
                          minlength=3)) >= 150), None)
print(f"\nf CHOSEN (L4, counts only): {F if F is None else round(F, 4)}")
if F is None:
    sys.exit("STOP: neither f reaches 150 per tercile")
B = BOX[F]
BE = B[B["elig"]].copy().reset_index(drop=True)
BE["I0"], BE["E0"] = BE["I"], BE["E"]
BE["I"], BE["E"] = BE["Iw"], BE["Ew"]                      # A1: updated I and running E
CUT = np.percentile(BE["P"], [100 / 3, 200 / 3])
BE["terc"] = terc_of(BE["P"].to_numpy(), CUT)
BE["pid"] = np.arange(len(BE))
print(f"P cut points {CUT.round(4)}; P quantiles 5/25/50/75/95: "
      f"{np.percentile(BE['P'], [5, 25, 50, 75, 95]).round(4).tolist()}; eligible by direction down/up "
      f"{int((BE['dirn'] == -1).sum())}/{int((BE['dirn'] == 1).sum())}")

hdr("GATE 3 — GEOMETRY (eligible boxes at the chosen f)")
BE["range"] = BE["box_hi"] - BE["box_lo"]
BE["wait"] = BE["tB"] - BE["t"]
for lab, s, unit in (("I (impulse size)", BE["I"], "pts"), ("box range", BE["range"], "pts"),
                     ("target distance 0.5·I", 0.5 * BE["I"], "pts")):
    print(f"{lab}: median {s.median():.2f} {unit} ({s.median() / TICK:.0f} ticks); p5/p25/p75/p95 "
          f"{s.quantile(.05):.2f}/{s.quantile(.25):.2f}/{s.quantile(.75):.2f}/{s.quantile(.95):.2f}")
print(f"box duration: fixed 5.0 min (first qualifying window); impulse → box end median {BE['wait'].median() / 60:.1f} min "
      f"(IQR {BE['wait'].quantile(.25) / 60:.1f}–{BE['wait'].quantile(.75) / 60:.1f})")
print(f"A1 extension before the pause: I_w / I_0 median {(BE['I'] / BE['I0']).median():.3f}, p75 "
      f"{(BE['I'] / BE['I0']).quantile(.75):.3f}, p95 {(BE['I'] / BE['I0']).quantile(.95):.3f}; I_0 median {BE['I0'].median():.2f}")
print(f"S5 move-size split: I < 10 {int((BE['I'] < 10).sum())}, 10–20 {int(((BE['I'] >= 10) & (BE['I'] < 20)).sum())}, "
      f">= 20 {int((BE['I'] >= 20).sum())}; box range / I median {(BE['range'] / BE['I']).median():.3f}; "
      f"boxes ending after 12:30 PT {int((BE['tB'] > [ep(pt(x, 12, 30)) for x in BE['d']]).sum())}")


# ----------------------------------------------------------------------------------- gate 4 placebo
def placebo(BE):
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
        keep &= (hi - lo) <= F * x.I
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


hdr("GATE 4 — PLACEBO FEASIBILITY")
PB, none = placebo(BE)
PB["terc"] = terc_of(PB["P"].to_numpy(), CUT)
nb = PB.groupby("parent").size().reindex(BE["pid"], fill_value=0)
print(f"placebo boxes {len(PB)} for {len(BE)} event boxes; per event median {nb.median():.0f}, mean {nb.mean():.2f}; "
      f"events with none {none}; with < 5 {int((nb < 5).sum())}; placebo per P tercile (event cuts) "
      f"{np.bincount(PB['terc'], minlength=3).tolist()}; high-P placebo {int((PB['terc'] == 2).sum())}")

hdr("GATE 5 — WORKED EVENT (random eligible event box, seed 20261025); outcome hidden")
ev = BE.iloc[int(np.random.default_rng(SEED).integers(len(BE)))]
d = ev["d"]
f = FLOW[d]
dn = int(ev["dirn"])
print(f"session {d}; {SYM[d]}; σ5_d {SIG[d]:.3f}; impulse {hms(ev['t'])} PT {'DOWN' if dn == -1 else 'UP'}: "
      f"opposite extreme {ev['opp']:.2f}, impulse extreme E0 {ev['E0']:.2f}, I0 {ev['I0']:.2f} pts "
      f"({ev['I0'] / SIG[d]:.2f} σ5)")
print(f"A1 running extreme at box start E {ev['E']:.2f}; updated I {ev['I']:.2f}; retrace limit 0.5·I(u) "
      f"(at box start {0.5 * ev['I']:.2f}); range cap f·I = {F * ev['I']:.2f}")
print(f"box {hms(ev['tB'] - BOX_W)}–{hms(ev['tB'])} PT (found {ev['wait'] / 60:.1f} min after the extreme): "
      f"low {ev['box_lo']:.2f}, high {ev['box_hi']:.2f}, range {ev['range']:.2f}")
V = cum_between(d, "vol", int(ev["tB"]) - BOX_W + 1, int(ev["tB"]))
Dn = dn * cum_between(d, "net", int(ev["tB"]) - BOX_W + 1, int(ev["tB"]))
print(f"D_box {Dn:,} (signed in the impulse direction), V_box {V:,}, P {ev['P']:.4f} → tercile "
      f"{['low', 'mid', 'high (absorption)'][int(ev['terc'])]}")
rev = ev["box_hi"] + 0.5 * ev["I"] if dn == -1 else ev["box_lo"] - 0.5 * ev["I"]
con = ev["box_lo"] - 0.5 * ev["I"] if dn == -1 else ev["box_hi"] + 0.5 * ev["I"]
print(f"targets: reversal {rev:.2f}, continuation {con:.2f} (0.5·I = {0.5 * ev['I']:.2f} beyond each boundary)")
a = np.searchsorted(f["t"], int(ev["t"]) - 300)
b = np.searchsorted(f["t"], int(ev["tB"]), side="right")
path = pd.DataFrame(dict(t=f["t"][a:b], lo=f["l"][a:b], hi=f["h"][a:b], c=f["c"][a:b], buy=f["buy"][a:b],
                         sell=f["sell"][a:b]))
path["m"] = (path["t"] - path["t"].iloc[0]) // 60
agg = path.groupby("m").agg(t=("t", "first"), low=("lo", "min"), high=("hi", "max"), close=("c", "last"),
                            buy=("buy", "sum"), sell=("sell", "sum"))
agg["t"] = [hms(x) for x in agg["t"]]
print("path, 1-minute slots from 5 min before the impulse to the box end:\n" + agg.to_string(index=False))


# ------------------------------------------------------------------------------------------ gate 6
def race(d, j, dirn, bhi, blo, I):
    """L5 from the row after j: 1 reversal / 0 continuation / -1 excluded; end time. A/A only."""
    f = FLOW[d]
    up_t, dn_t = bhi + TGT_FRAC * I, blo - TGT_FRAC * I
    ui = np.nonzero(f["h"][j + 1:] >= up_t)[0]
    di = np.nonzero(f["l"][j + 1:] <= dn_t)[0]
    if not len(ui) and not len(di):
        return -1, ep(pt(d, 13, 0))
    fu, fd = (ui[0] if len(ui) else 10**9), (di[0] if len(di) else 10**9)
    if fu == fd:
        return -1, int(f["t"][j + 1 + fu])
    first_up = fu < fd
    return (1 if first_up == (dirn == -1) else 0), int(f["t"][j + 1 + min(fu, fd)])


hdr("GATE 6 — A/A CALIBRATION (placebo boxes only)")
rr = [race(x.d, x.j, x.dirn, x.box_hi, x.box_lo, x.I) for x in PB.itertuples()]
PB["res"] = [r[0] for r in rr]
PB["race_gap"] = [max_gap(x.d, x.tB, r[1]) for x, r in zip(PB.itertuples(), rr)]
n0 = len(PB)
PBk = PB[PB["race_gap"] <= GAP_MAX]
n1 = len(PBk)
PBk = PBk[PBk["res"] >= 0].copy()
print(f"placebo boxes {n0}; race-window gap drops −{n0 - n1}; excluded (tie or unresolved by 13:00) −{n1 - len(PBk)}; "
      f"resolved {len(PBk)}; per P tercile {np.bincount(PBk['terc'], minlength=3).tolist()}")
S = {d: k for k, d in enumerate(USABLE)}
nS = len(USABLE)
PBk["sidx"] = [S[d] for d in PBk["d"]]
parents = sorted(set(PBk["parent"]))
pmap = {p_: k for k, p_ in enumerate(parents)}
PBk["pidx"] = [pmap[p_] for p_ in PBk["parent"]]
rng, brng = np.random.default_rng(SEED), np.random.default_rng(SEED_BOOT)


def counts(m):
    c = np.zeros((nS, 4))
    sub = PBk[m]
    for col, (tt, what) in enumerate([(2, "w"), (2, "n"), (0, "w"), (0, "n")]):
        s = sub[sub["terc"] == tt]
        c[:, col] = np.bincount(s["sidx"], weights=s["res"] if what == "w" else None, minlength=nS)
    return c


def diff(c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 0] / c[..., 1] - c[..., 2] / c[..., 3]


st, excl = [], 0
for _ in range(N_SPLIT):
    perm = rng.permutation(len(parents))
    h1 = np.zeros(len(parents), bool)
    h1[perm[: (len(parents) + 1) // 2]] = True
    m1 = h1[PBk["pidx"].to_numpy()]
    c1, c2 = counts(m1), counts(~m1)
    obs = diff(c1.sum(0)) - diff(c2.sum(0))
    w = np.stack([np.bincount(brng.integers(nS, size=nS), minlength=nS) for _ in range(N_BOOT)]).astype(float)
    lo, hi = np.nanpercentile(diff(w @ c1) - diff(w @ c2), [2.5, 97.5])
    st.append(obs)
    excl += not (lo <= 0 <= hi)
st = np.array(st) * 100
share = excl / N_SPLIT
ok6 = share <= 0.10 and abs(np.median(st)) <= 3
print(f"{len(parents)} parent events; statistic (high − low P tercile reversal rate, half 1 − half 2): median "
      f"{np.median(st):+.2f}, mean {st.mean():+.2f}, sd {st.std(ddof=1):.2f}, p5 {np.percentile(st, 5):+.2f}, "
      f"p95 {np.percentile(st, 95):+.2f}; CI excludes 0 in {share:.1%} ({excl}/{N_SPLIT})")
print(f"GATE 6: {'PASS' if ok6 else 'FAIL — STOP'}")
print("\nDONE")
