"""Tests for packages/shared/flow_features.py (CR-CD step 1).

Run with:
    python -m unittest packages.shared.tests.test_flow_features -v

Part A identity: the building blocks equal CR-CB's own code (a verbatim excerpt of
cr_cb_step0_v2.py, executed as an oracle) on 3 recorded Databento sessions and 8 synthetic ones.
Part B: the pane series and markers on synthetic minutes, and causality on a real session.
"""
from __future__ import annotations

import math
import sys
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared import flow_features as ff  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"
REAL_DAYS = (date(2026, 1, 6), date(2026, 3, 2), date(2026, 5, 1))
SYN_DAYS = tuple(date(2025, 12, 1) + timedelta(days=k) for k in range(10) if (date(2025, 12, 1) + timedelta(days=k)).weekday() < 5)


def load_real():
    z = np.load(FIX / "cr_cd_flow_1s_3sessions.npz")
    return {d: {k: z[f"{d.isoformat()}/{k}"] for k in ("t", "buy", "sell", "vol", "o", "h", "l", "c")}
            for d in REAL_DAYS}


def synthetic_session(d, seed):
    """[06:15, 13:00) PT, ~85 % of seconds with trades, a few > 15 s holes, random-walk prices."""
    rng = np.random.default_rng(seed)
    t0, t1 = ff.ep(ff.pt(d, 6, 15)), ff.ep(ff.pt(d, 13, 0))
    t = np.arange(t0, t1)
    keep = rng.random(len(t)) < 0.85
    for h in rng.integers(0, len(t) - 40, size=4):
        keep[h:h + 30] = False
    t = t[keep]
    buy = rng.poisson(6, len(t)).astype(np.int64)
    sell = rng.poisson(6, len(t)).astype(np.int64)
    c = 6000 + np.round(np.cumsum(rng.normal(0, 0.25, len(t)) + 0.00004 * (buy - sell)) * 4) / 4
    o = np.concatenate([[c[0]], c[:-1]])
    return dict(t=t.astype(np.int64), buy=buy, sell=sell, vol=buy + sell + rng.poisson(0.01, len(t)),
                o=o, h=np.maximum(o, c) + 0.25 * (rng.random(len(t)) < 0.3), l=np.minimum(o, c), c=c)


def oracle(flow, usable):
    """Execute the verbatim CR-CB excerpt with the globals it reads."""
    ns = dict(np=np, pd=pd, date=date, datetime=datetime, time=time, timedelta=timedelta,
              PT=ZoneInfo("America/Los_Angeles"), UTC=timezone.utc, BASE_W=300, GAP_MAX=15, BASE_N=20,
              BASE_MIN=5, PUSH_W=120, RETREAT=2.0, FLOW=flow, USABLE=list(usable), SKEW={}, IM={})
    exec(compile((FIX / "cr_cb_step0_v2_excerpt.py.txt").read_text(), "cr_cb_step0_v2_excerpt", "exec"), ns)
    return ns


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def events_for(F, d):
    """Deterministic edge events spread over the session (valid index orderings; no outcomes)."""
    out = []
    t = F["t"]
    for hh in (7, 8, 9, 10, 11, 12):
        c = int(np.searchsorted(t, ff.ep(ff.pt(d, hh, 0))))
        w0 = int(np.searchsorted(t, t[c] - 1800))
        for side in (1, -1):
            Hs, Los, _, _ = ff.space(F, side)
            hi_i = w0 + int(np.argmax(Hs[w0:c + 1]))
            H = float(Hs[hi_i])
            pl_i = hi_i + int(np.argmin(Los[hi_i:c + 1]))
            PL = float(Los[pl_i])
            after = np.nonzero(Hs[c + 1:] >= H - 1.0)[0]
            top_i = c + 1 + int(after[0]) if len(after) else len(t) - 1
            after_b = np.nonzero(Los[c + 1:] <= PL + 1.0)[0]
            bot_i = c + 1 + int(after_b[0]) if len(after_b) else len(t) - 1
            e = dict(side=side, L=side * (H - 3.0), j=hi_i)
            a = dict(top_i=top_i, bot_i=bot_i, hi_i=hi_i, pl_i=pl_i, H=H, PL=PL, tol=1.0, c=c)
            out.append((e, a))
    return out


class TestPartAIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow = {d: synthetic_session(d, k) for k, d in enumerate(SYN_DAYS)}
        cls.flow.update(load_real())
        cls.days = sorted(cls.flow)
        cls.ns = oracle(cls.flow, cls.days)

    def test_session_impact_samples_identical(self):
        for d in self.days:
            dc, imp = ff.session_impact_samples(self.flow[d], d)
            o = self.ns["STATS"][d]
            self.assertTrue(np.array_equal(dc, o[0]), d)
            self.assertTrue(np.array_equal(imp, o[1]), d)

    def test_impact_baseline_identical(self):
        samples = {d: ff.session_impact_samples(self.flow[d], d) for d in self.days}
        n = 0
        for d in self.days:
            got = ff.impact_baseline(samples, self.days, d)
            if d in self.ns["BD"]:
                self.assertEqual(got, (self.ns["SIG5"][d], self.ns["BD"][d]), d)
                n += 1
            else:
                self.assertIsNone(got, d)
        self.assertGreaterEqual(n, 5)

    def test_edge_flow_features_identical(self):
        keys = ("P", "A_rel", "Q", "n_push", "PD", "range_s", "LE", "RA", "DV", "SS_share", "SS_vwap")
        n = 0
        for d in REAL_DAYS:
            F = self.flow[d]
            bd = self.ns["BD"].get(d)
            for e, a in events_for(F, d):
                for edge in ("top", "bot"):
                    o = self.ns["edge_features"](F, d, e, dict(a), edge)
                    g = ff.edge_flow_features(F, d, e, dict(a), edge, bd)
                    for k in keys:
                        self.assertTrue(same(g[k], o[k]), (d, edge, k, g[k], o[k]))
                    n += 1
        self.assertEqual(n, 3 * 6 * 2 * 2)

    def test_flow_sum_and_max_gap_identical(self):
        F, d = self.flow[REAL_DAYS[2]], REAL_DAYS[2]
        for hh in range(7, 13):
            t0, t1 = ff.ep(ff.pt(d, hh, 0)), ff.ep(ff.pt(d, hh, 7))
            self.assertEqual(ff.flow_sum(F, t0, t1), self.ns["flow_sum"](F, t0, t1))
            self.assertEqual(ff.max_gap(F, d, t0, t1), self.ns["max_gap"](F, d, t0, t1))


def frame(closes, deltas, vols=None, spread=0.25, start=0):
    """Synthetic minutes: close path, net delta per bar, high/low = close ± spread."""
    c = np.asarray(closes, float)
    dl = np.asarray(deltas, float)
    v = np.asarray(vols if vols is not None else np.abs(dl) + 400.0, float)
    buy, sell = (v + dl) / 2, (v - dl) / 2
    idx = np.arange(start, start + 60 * len(c), 60, dtype=np.int64)
    return pd.DataFrame({"o": np.concatenate([[c[0]], c[:-1]]), "h": c + spread, "l": c - spread, "c": c,
                         "buy": buy, "sell": sell, "vol": v}, index=idx)


def kinds(marks, kind):
    return [m for m in marks if m["kind"] == kind]


class TestPartBPane(unittest.TestCase):
    def run_pane(self, m, base=None, p=ff.A1_AS_WRITTEN):
        s = ff.pane_series(m, int(m.index[0]), base, absorption=base is not None)
        return s, ff.pane_markers(m, s, int(m.index[0]), p)

    def test_bullish_divergence_against_confirmed_swing(self):
        down1 = np.linspace(110, 100, 11)                 # swing low 100 at bar 10
        up = np.linspace(100.5, 105, 10)                  # >= 4-pt retrace confirms it
        down2 = np.linspace(104.5, 99, 12)                # new low below 100
        closes = np.concatenate([down1, up, down2])
        d_sell = np.concatenate([np.full(11, -300.0), np.full(10, 100.0), np.full(12, -300.0)])
        d_buy = np.concatenate([np.full(11, -300.0), np.full(10, 100.0), np.full(12, 150.0)])
        _, m_none = self.run_pane(frame(closes, d_sell))
        _, m_div = self.run_pane(frame(closes, d_buy))
        self.assertEqual(kinds(m_none, "divergence"), [])
        div = kinds(m_div, "divergence")
        self.assertEqual(len(div), 1)
        self.assertEqual(div[0]["side"], "bull")
        self.assertEqual(div[0]["detail"]["swing_price"], 100 - 0.25)
        first_new_low = 21 + int(np.argmax(down2 - 0.25 < 99.75))
        self.assertEqual(div[0]["time"], 60 * first_new_low)

    def test_push_decay_at_retest_of_same_swing(self):
        up = np.linspace(100, 110, 11)                    # swing high 110, pushed by strong buying
        back = np.linspace(109, 104, 6)                   # retreat > 4 pts: confirms and arms
        retest = np.linspace(105, 109, 5)                 # retest within 2 pts
        closes = np.concatenate([up, back, retest, [108.0, 107.0]])
        strong = np.concatenate([np.full(11, 500.0), np.full(6, -100.0), np.full(5, 500.0), [0.0, 0.0]])
        weak = np.concatenate([np.full(11, 500.0), np.full(6, -100.0), np.full(5, 60.0), [0.0, 0.0]])
        _, m_strong = self.run_pane(frame(closes, strong))
        _, m_weak = self.run_pane(frame(closes, weak))
        self.assertEqual(kinds(m_strong, "push_decay"), [])
        pdm = kinds(m_weak, "push_decay")
        self.assertEqual(len(pdm), 1)
        self.assertEqual(pdm[0]["side"], "bear")
        self.assertLessEqual(pdm[0]["detail"]["push"], 0.6 * pdm[0]["detail"]["prev_push"])

    def test_fading(self):
        closes = np.linspace(100, 115, 65)
        deltas = np.concatenate([np.full(40, 300.0), np.full(25, 20.0)])   # steady (LE 0), then weaker
        _, marks = self.run_pane(frame(closes, deltas))
        f = kinds(marks, "fading")
        self.assertTrue(f)
        self.assertEqual(f[0]["side"], "bear")
        self.assertIn("Buyers fading", f[0]["text"])

    def test_absorption(self):
        closes = np.concatenate([np.linspace(100, 110, 10), np.full(10, 110.0)])
        deltas = np.concatenate([np.full(10, 200.0), np.full(10, 600.0)])
        m = frame(closes, deltas, vols=np.full(20, 1000.0))
        base = ff.absorption_baseline([frame(np.linspace(100, 130, 60), np.full(60, 200.0),
                                             vols=np.full(60, 1000.0))])
        s, _ = self.run_pane(m, base)
        self.assertFalse(s["absorb"].iloc[:10].any())
        self.assertTrue(s["absorb"].iloc[15:].all())    # big buying, price flat, share 0.6

    def test_cap_one_per_kind_per_15_min(self):
        marks = [dict(i=i, kind="fading", side="bull", text="", detail={}) for i in (0, 5, 14, 15, 29, 31)]
        self.assertEqual([m["i"] for m in ff.cap_markers(marks, 15)], [0, 15, 31])
        self.assertEqual([m["i"] for m in ff.cap_markers(marks, 30)], [0, 31])

    def test_default_floors_suppress_small_signals(self):
        """Defaults (setting F): divergence needs an 8-pt swing and a CVD change of at least 300."""
        down1 = np.linspace(110, 100, 11)
        up = np.linspace(100.5, 109, 10)                  # 9-pt retrace: confirms an 8-pt swing
        down2 = np.linspace(108.5, 99, 12)
        closes = np.concatenate([down1, up, down2])
        small = np.concatenate([np.full(11, -300.0), np.full(10, 10.0), np.full(12, 10.0)])    # dCVD ~ 200
        big = np.concatenate([np.full(11, -300.0), np.full(10, 10.0), np.full(12, 150.0)])     # dCVD > 300
        _, m_small = self.run_pane(frame(closes, small), p=ff.MarkerParams())
        _, m_big = self.run_pane(frame(closes, big), p=ff.MarkerParams())
        self.assertEqual(kinds(m_small, "divergence"), [])
        self.assertEqual(len(kinds(m_big, "divergence")), 1)
        # the 5-pt retrace of the A1 test does not confirm an 8-pt swing
        closes4 = np.concatenate([np.linspace(110, 100, 11), np.linspace(100.5, 105, 10), np.linspace(104.5, 99, 12)])
        _, m4 = self.run_pane(frame(closes4, big), p=ff.MarkerParams())
        self.assertEqual(kinds(m4, "divergence"), [])

    def test_push_decay_floor(self):
        up = np.linspace(100, 110, 11)
        back = np.linspace(109, 100, 10)                  # > 8-pt retrace
        retest = np.linspace(101, 109, 9)
        closes = np.concatenate([up, back, retest, [108.0, 107.0]])
        def deltas(first):
            return np.concatenate([np.full(11, first), np.full(10, -100.0), np.full(9, 20.0), [0.0, 0.0]])
        _, small_prev = self.run_pane(frame(closes, deltas(200.0)), p=ff.MarkerParams())   # prev push 400 < 500
        _, big_prev = self.run_pane(frame(closes, deltas(400.0)), p=ff.MarkerParams())     # prev push 800
        self.assertEqual(kinds(small_prev, "push_decay"), [])
        self.assertEqual(len(kinds(big_prev, "push_decay")), 1)

    def test_causal_on_real_session(self):
        F = load_real()[date(2026, 5, 1)]
        d = date(2026, 5, 1)
        start = ff.ep(ff.pt(d, 6, 30))
        rows = ff.minutes_from_seconds(F)
        m = ff.complete_minutes(rows, ff.ep(ff.pt(d, 6, 15)), ff.ep(ff.pt(d, 13, 0)))
        base = ff.absorption_baseline([m])
        full_s = ff.pane_series(m, start, base, True)
        full_m = {k: ff.pane_markers(m, full_s, start, p) for k, p in
                  (("F", ff.MarkerParams()), ("A1", ff.A1_AS_WRITTEN))}
        self.assertTrue(full_m["F"])
        self.assertGreater(len(full_m["A1"]), len(full_m["F"]))
        for hh, mm in ((7, 39), (8, 2), (9, 30), (11, 0), (12, 45)):
            cut = ff.ep(ff.pt(d, hh, mm))
            mp = m[m.index < cut]
            s = ff.pane_series(mp, start, base, True)
            pd.testing.assert_frame_equal(s, full_s[full_s.index < cut])
            for k, p in (("F", ff.MarkerParams()), ("A1", ff.A1_AS_WRITTEN)):
                self.assertEqual(ff.pane_markers(mp, s, start, p), [x for x in full_m[k] if x["time"] < cut])

    def test_minutes_from_seconds_matches_definition(self):
        F = load_real()[date(2026, 1, 6)]
        rows = ff.minutes_from_seconds(F)
        k = rows.index[100]
        sel = (F["t"] // 60) * 60 == k
        self.assertEqual(rows.loc[k, "h"], F["h"][sel].max())
        self.assertEqual(rows.loc[k, "o"], F["o"][sel][0])
        self.assertEqual(rows.loc[k, "c"], F["c"][sel][-1])
        self.assertEqual(rows.loc[k, "buy"], F["buy"][sel].sum())


if __name__ == "__main__":
    unittest.main()
