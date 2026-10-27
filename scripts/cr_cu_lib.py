#!/usr/bin/env python3
"""CR-CU shared definitions — CR-CT's second-touch reversal on 2023-05-01 .. 2024-04-02. READ-ONLY.

Spec: specs/CR-CU-second-touch-2023.md (C1, U1-U10). Everything is CR-CT as shipped (scripts/cr_ct_lib.py: levels,
arming / touches, approach speed and relvol, classes, random baseline) except what this module replaces:

  data (C1, U1): ironbeam_es_1m_bars source = 'databento' (ES.c.0, the only Databento ES series before 2024-04-03),
                 ext window 06:15 .. 12:59 PT; volume = ES.c.0 bar volume. Every read inside
                 [2023-05-01 00:00 PT, 2024-04-02 13:00 PT].
  universe (U2): CR-CT / CR-CP funnel, then CR-CO's exclusions (2023-09, roll-contaminated sessions).
  F1 / F2 (U3): missing across a quarterly ES expiry (no symbol column to compare).
  priors (U4): roll-contaminated sessions are not relvol priors.
  touch and race (C1): entry = L at the touch bar j (R2 level-entry variant); race on 1-min bars from j + 1.
Importing computes no outcome; race1m / mfe1m / intrabar are only called by the run script.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_ct_lib as T  # noqa: E402  (pure functions only; its 2026 loaders are never called here)
import cr_cp_lib as C  # noqa: E402  (conn / q / pt; loads .env)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, prev_trading_day  # noqa: E402

PT, UTC = C.PT, C.UTC
D_FROM, D_TO = date(2023, 5, 1), date(2024, 4, 2)
FLOOR, CAP = C.pt(D_FROM, 0, 0), C.pt(D_TO, 13, 0)         # U1; R-a starts 2024-04-03
NB, PRE, NE = T.NB, T.PRE, T.NE
X1, X2 = T.X1, T.X2
SEED, N_BOOT, SEED_AA, N_AA = T.SEED, T.N_BOOT, T.SEED_AA, T.N_AA
MIN_CELL, MIN_SESS = T.MIN_CELL, T.MIN_SESS
W_PRIMARY, W_READ = T.W_PRIMARY, T.W_READ
# U5: CR-CT Step 0 cuts (scripts/logs/cr_ct/step0.json), frozen; never re-fit
CUTS = {"10": [0.007250746319479294, 0.014000729018632885],
        "5": [0.010157656953575242, 0.021024723131091827],
        "20": [0.004038630778980846, 0.009603635524005163]}
# U2: CR-CO exclusions (scripts/logs/cr_co/exclusions.json), verbatim
EXCL_QUOTE = [date(2023, 9, d) for d in (1, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29)]
EXCL_ROLL = [date(2023, 3, 15), date(2023, 3, 16), date(2023, 3, 17), date(2023, 6, 14), date(2023, 6, 15),
             date(2023, 6, 16), date(2023, 9, 12), date(2023, 9, 13), date(2023, 9, 14), date(2023, 9, 15),
             date(2023, 12, 12), date(2023, 12, 13), date(2023, 12, 14), date(2023, 12, 15), date(2024, 3, 13),
             date(2024, 3, 14), date(2024, 3, 15)]
EXPIRY_FRIDAYS = [date(2023, 6, 16), date(2023, 9, 15), date(2023, 12, 15), date(2024, 3, 15)]   # U3
# U8 sensitivity: expiry-week Mondays / Tuesdays (kept by CR-CO; ES.c.0 is the dying contract at 0.16-0.78x volume)
EXPIRY_MON_TUE = [f - timedelta(days=k) for f in EXPIRY_FRIDAYS for k in (4, 3)]
OUT = C.REPO_ROOT / "scripts" / "logs" / "cr_cu"
OUT.mkdir(parents=True, exist_ok=True)


def guard(lo, hi):
    assert FLOOR <= lo <= hi <= CAP, (lo, hi)
    return lo, hi


def naive(ts):
    return ts.astimezone(UTC).replace(tzinfo=None)                # ironbeam_es_1m_bars.datetime is naive UTC


def check_frozen():
    """U2 / U5: the embedded constants equal CR-CT's / CR-CO's logs where those files exist (main checkout copy)."""
    out = {}
    for root in (C.REPO_ROOT, Path("/Users/ryan/code/volatility-monorepo")):
        p = root / "scripts" / "logs" / "cr_ct" / "step0.json"
        if p.exists() and "cuts" not in out:
            s = json.loads(p.read_text())["cuts"]
            for w, c in CUTS.items():
                assert np.allclose(s[w]["cuts"], c), (w, s[w]["cuts"], c)
            out["cuts"] = str(p)
        p = root / "scripts" / "logs" / "cr_co" / "exclusions.json"
        if p.exists() and "excl" not in out:
            e = json.loads(p.read_text())
            assert sorted(e["quote_time"]) == [str(d) for d in EXCL_QUOTE]
            assert sorted(e["roll"]) == [str(d) for d in EXCL_ROLL]
            out["excl"] = str(p)
    return out


# ------------------------------------------------------------------------------------------------ data (C1, U1)
def load_m1() -> pd.DataFrame:
    path = OUT / "m1_ext.parquet"
    if path.exists():
        return pd.read_parquet(path)
    a, b = guard(C.pt(D_FROM, 6, 15), C.pt(D_TO, 13, 0))
    r = C.q("""
        SELECT (datetime AT TIME ZONE 'UTC') AS minute_utc, source,
               open::float AS o, high::float AS h, low::float AS l, close::float AS c, volume::float AS vol
        FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime < %s
          AND ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::time >= '06:15'
          AND ((datetime AT TIME ZONE 'UTC') AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        ORDER BY datetime
    """, (naive(a), naive(b)))
    t = pd.to_datetime(r["minute_utc"], utc=True)
    assert t.min() >= a and t.max() < b
    r.to_parquet(path)
    return r


def ext_arrays(m1: pd.DataFrame):
    """As cr_ct_lib.ext_arrays with one series (ES.c.0): raw / flat-filled OHLC (NE,), volume (NE,), last bar."""
    m1 = m1[m1["source"] == "databento"].copy()
    ts = pd.to_datetime(m1["minute_utc"], utc=True).dt.tz_convert(PT)
    m1["d"] = ts.dt.date
    m1["e"] = (ts.dt.hour * 60 + ts.dt.minute - 375).astype(int)
    RAW, FILL, TOT, LAST = {}, {}, {}, {}
    for d, g in m1.groupby("d"):
        tot = np.zeros(NE)
        np.add.at(tot, g["e"].to_numpy(), g["vol"].to_numpy(float))
        raw = {k: np.full(NE, np.nan) for k in ("o", "h", "l", "c")}
        for k in raw:
            raw[k][g["e"].to_numpy()] = g[k].to_numpy(float)
        have = np.isfinite(raw["c"])
        fl = {k: raw[k].copy() for k in raw}
        if have.any():
            first, last = int(np.argmax(have)), int(NE - 1 - np.argmax(have[::-1]))
            for j in range(first + 1, last + 1):
                if not have[j]:
                    for k in fl:
                        fl[k][j] = fl["c"][j - 1]
            LAST[d] = last
        RAW[d], FILL[d], TOT[d] = raw, fl, tot
    return dict(SYM={d: "ES.c.0" for d in RAW}, RAW=RAW, FILL=FILL, TOT=TOT, LAST=LAST)


def load_im() -> pd.Series:
    guard(C.pt(D_FROM, 0, 0), C.pt(D_TO, 13, 0))
    f = C.q("""
        SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
        WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
    """, (CANONICAL_FEATURE_VERSION, D_FROM, D_TO))
    return f.drop_duplicates("trade_date").set_index("trade_date")["im"]


def sessions():
    out, d = [], D_FROM
    while d <= D_TO:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def rth_bars(raw):
    """CR-CP build: (NB,) RTH arrays with flat fill inside [first, last], and the missing-minute stats."""
    arr = {k: raw[k][PRE:].copy() for k in ("o", "h", "l", "c")}
    have = np.isfinite(arr["c"])
    if not have.any():
        return None, None
    first, last = int(np.argmax(have)), int(NB - 1 - np.argmax(have[::-1]))
    miss = ~have
    miss[:first] = False
    run, best = 0, 0
    for x in miss[first:last + 1]:
        run = run + 1 if x else 0
        best = max(best, run)
    for j in range(first + 1, last + 1):
        if miss[j]:
            for k in arr:
                arr[k][j] = arr["c"][j - 1]
    return arr, dict(first=first, last=last, n_missing=int(miss[first:last + 1].sum()), max_run=best)


def build():
    """Universe (U2), ext arrays, priors (U4), IM."""
    m1 = load_m1()
    src = m1["source"].value_counts().to_dict()
    E = ext_arrays(m1)
    im = load_im()
    BARS, MISS = {}, {}
    for d, raw in E["RAW"].items():
        b, m = rth_bars(raw)
        if b is not None:
            BARS[d], MISS[d] = b, m
    U, funnel = sessions(), []
    excl_q, excl_r = set(EXCL_QUOTE), set(EXCL_ROLL)
    for nm, rule in [
        ("no Databento RTH bars", lambda d: d not in BARS),
        ("starts after 06:31 PT", lambda d: MISS[d]["first"] > 1),
        ("early close (last RTH bar < 12:55 PT)", lambda d: MISS[d]["last"] < C.EARLY_LAST),
        ("missing-minute run > 5 (C18)", lambda d: MISS[d]["max_run"] > C.MISS_RUN_MAX),
        ("no canonical IM / IM <= 0", lambda d: not (d in im.index and pd.notna(im[d]) and im[d] > 0)),
        ("CR-CO roll-contaminated", lambda d: d in excl_r),
        ("CR-CO quote-time month 2023-09", lambda d: d in excl_q),
    ]:
        rm = [d for d in U if rule(d)]
        U = [d for d in U if not rule(d)]
        funnel.append([nm, len(rm), len(U), [str(x) for x in rm]])
    alld = sorted(d for d in E["RAW"] if is_trading_day(d))
    PRI = [d for d in alld if E["LAST"].get(d, -1) >= C.EARLY_LAST + PRE and d not in excl_r]   # U4
    return dict(SESS=U, BARS={d: BARS[d] for d in U}, SYM=E["SYM"], MISS=MISS, IM={d: float(im[d]) for d in U},
                E=E, ALLD=alld, PRI=PRI, funnel=funnel, sources=src)


# ---------------------------------------------------------------------------------------------- levels (U3)
def crosses_expiry(prev, d):
    return any(prev <= f <= d for f in EXPIRY_FRIDAYS)


def overnight(d, prev, raw_d):
    """CR-CT C3 F2 on ironbeam_es_1m_bars; the 06:29 check compares the series with itself here (U3)."""
    t0, t1 = guard(C.pt(prev, 15, 0), C.pt(d, 6, 30))
    r = C.q("""SELECT max(high)::float h, min(low)::float l, count(*) n,
                      count(*) FILTER (WHERE source IS DISTINCT FROM 'databento') n_other
               FROM ironbeam_es_1m_bars WHERE datetime >= %s AND datetime < %s""", (naive(t0), naive(t1)))
    if not r["n"][0]:
        return None, "no overnight bars"
    if r["n_other"][0]:
        return None, "non-Databento overnight bars"
    ib = C.q("SELECT close::float c FROM ironbeam_es_1m_bars WHERE datetime = %s", (naive(C.pt(d, 6, 29)),))
    dc = raw_d["c"][PRE - 1]
    if not len(ib) or not np.isfinite(dc):
        return None, "no 06:29 bar"
    if abs(ib["c"][0] - dc) > 2.0:
        return None, f"06:29 {ib['c'][0]} vs {dc}"
    return dict(h=float(r["h"][0]), l=float(r["l"][0])), None


def levels(D, d, why: dict):
    """cr_ct_lib.levels with the U3 contract guard in place of the symbol check; no read before 2023-05-01."""
    E, B = D["E"], D["BARS"][d]
    o, im = float(B["o"][0]), D["IM"][d]
    lv = []
    prev = prev_trading_day(d)
    if prev < D_FROM or prev not in E["RAW"]:
        why.setdefault("F1", []).append((str(d), "prior session not read / no bars"))
        why.setdefault("F2", []).append((str(d), "prior session not read / no bars"))
        prev = None
    elif crosses_expiry(prev, d):
        why.setdefault("F1", []).append((str(d), "quarterly expiry between prior session and today"))
        why.setdefault("F2", []).append((str(d), "quarterly expiry between prior session and today"))
        prev = None
    if prev is not None:
        R = E["RAW"][prev]
        hh, ll, cc = R["h"][PRE:], R["l"][PRE:], R["c"][PRE:]
        fin = np.where(np.isfinite(cc))[0]
        if not len(fin):
            why.setdefault("F1", []).append((str(d), "prior session no RTH bars"))
        else:
            for kind, L_ in (("prior_high", np.nanmax(hh)), ("prior_low", np.nanmin(ll)), ("prior_close", cc[fin[-1]])):
                lv.append(dict(fam="F1", kind=kind, L=float(L_), known=0))
        on, err = overnight(d, prev, E["RAW"][d])
        if on is None:
            why.setdefault("F2", []).append((str(d), err))
        else:
            lv += [dict(fam="F2", kind="on_high", L=on["h"], known=0), dict(fam="F2", kind="on_low", L=on["l"], known=0)]
    lv = [x for x in lv if abs(x["L"] - o) <= T.WIN_IM * im]
    lv = T.merge([x for x in lv if x["fam"] == "F1"], T.MERGE_FAM) + T.merge([x for x in lv if x["fam"] == "F2"], T.MERGE_FAM)
    return lv + T.run_levels(B["h"], B["l"])


# --------------------------------------------------------------------------------------------- events
def events(D):
    """First and second touches per level with CR-CT's approach features (windows 10 / 5 / 20). No outcome."""
    rows, why = [], {}
    for d in D["SESS"]:
        B, arm = D["BARS"][d], T.arm_of(D, d)
        for lv in levels(D, d, why):
            for n, (j, s, a) in enumerate(T.touches(B, lv["L"], lv["known"], arm)):
                b = -s
                r = dict(d=d, fam=lv["fam"], kind=lv["kind"], L=lv["L"], known=lv["known"], touch=n + 1, j=j,
                         b=b, arm_bar=a, edge=T.edge_at(B, j, lv["L"]), first_hour=j < T.FIRST_HOUR_END,
                         mon_tue_expiry=d in EXPIRY_MON_TUE)
                for w in (W_PRIMARY,) + W_READ:
                    sp, rv, wy = T.approach(D, d, j, b, w)
                    r[f"speed{w}"], r[f"relvol{w}"], r[f"why{w}"] = sp, rv, wy
                rows.append(r)
    ev = pd.DataFrame(rows)
    ev["dup"] = False
    keep = {}
    order = ev.assign(pr=ev["fam"].map(T.FAM_PRIORITY)).sort_values(["d", "touch", "j", "pr"]).index
    for i in order:                                          # CR-CT C5: pooled dedupe, per touch number
        r = ev.loc[i]
        key = (r["d"], r["touch"], r["j"], r["b"])
        if any(abs(r["L"] - x) <= T.MERGE_POOL for x in keep.get(key, [])):
            ev.at[i, "dup"] = True
        else:
            keep.setdefault(key, []).append(r["L"])
    for w, c in CUTS.items():
        ev[f"cls{w}"] = T.classify(ev, f"speed{w}", f"relvol{w}", c)
    return ev, why


# --------------------------------------------------------------------------------------------- outcomes (C1)
def _fav_adv(B, j, L_, b):
    H, Lo = B["h"][j + 1:], B["l"][j + 1:]
    return ((H - L_), (L_ - Lo)) if b == 1 else ((L_ - Lo), (H - L_))


def race1m(B, j, L_, b, X):
    """C1: entry L at touch bar j; 1-min bars j + 1 .. 12:59 PT. 'win' / 'loss' / 'tie' / 'unresolved'."""
    fav, adv = _fav_adv(B, j, L_, b)
    fw, aw = fav >= X, adv >= X
    i_f = int(np.argmax(fw)) if fw.any() else None
    i_a = int(np.argmax(aw)) if aw.any() else None
    if i_f is None and i_a is None:
        return "unresolved"
    if i_a is None or (i_f is not None and i_f < i_a):
        return "win"
    if i_f is None or i_a < i_f:
        return "loss"
    return "tie"


def mfe1m(B, j, L_, b, adv_x=X1):
    """Largest favourable excursion before the first -adv_x bar (that bar's favourable side not credited)."""
    fav, adv = _fav_adv(B, j, L_, b)
    aw = adv >= adv_x
    stop = int(np.argmax(aw)) if aw.any() else len(fav)
    return float(max(0.0, np.max(fav[:stop]))) if stop > 0 else 0.0


def intrabar(B, j, L_, b, X=X1):
    """C1 disclosure: did bar j itself reach +X / -X from L (order inside the minute unknown; not credited)."""
    fav = (B["h"][j] - L_) if b == 1 else (L_ - B["l"][j])
    adv = (L_ - B["l"][j]) if b == 1 else (B["h"][j] - L_)
    return ("fav" if fav >= X else "") + ("adv" if adv >= X else "") or "none"
