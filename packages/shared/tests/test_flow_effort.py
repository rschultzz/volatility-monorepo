"""Tests for the swing-effort building blocks in packages/shared/flow_features.py (CR-CD A5).

Run with:
    python -m unittest packages.shared.tests.test_flow_effort -v

Synthetic minutes with hand-checked legs and stats, edge cases (first leg in a direction has no
ratio, zero-point legs, zero-volume previous leg, two legs confirmed on one bar), and causality:
a leg's stats neither change nor appear before its end is confirmed (synthetic and a real session).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared import flow_features as ff  # noqa: E402
from packages.shared.tests.test_flow_features import REAL_DAYS, load_real  # noqa: E402

T0 = 1_777_642_200   # 2026-05-01 06:30 PT


def frame(H, L, O=None, vol=None, buy=None, sell=None, t0=T0):
    """Minutes from explicit highs / lows (closes at the bar's midpoint unless O is given)."""
    H, L = np.asarray(H, float), np.asarray(L, float)
    n = len(H)
    C = (H + L) / 2
    O = np.asarray(O, float) if O is not None else np.concatenate([[C[0]], C[:-1]])
    vol = np.asarray(vol if vol is not None else np.full(n, 100.0), float)
    buy = np.asarray(buy if buy is not None else vol * 0.5, float)
    sell = np.asarray(sell if sell is not None else vol - buy, float)
    return pd.DataFrame({"o": O, "h": H, "l": L, "c": C, "buy": buy, "sell": sell, "vol": vol},
                        index=pd.Index(t0 + 60 * np.arange(n, dtype=np.int64), name="time"))


def path(points, step=1.0):
    """Bar highs / lows walking through the turning points `points`, one bar per `step` pts; each bar
    spans [p, p + step] on the way up and [p - step, p] on the way down."""
    H, L = [], []
    for a, b in zip(points[:-1], points[1:]):
        sg = 1 if b > a else -1
        p = a
        while (p < b) if sg > 0 else (p > b):
            q = p + sg * step
            H.append(max(p, q))
            L.append(min(p, q))
            p = q
    return H, L


def random_minutes(seed, n=390):
    rng = np.random.default_rng(seed)
    c = 6000 + np.round(np.cumsum(rng.normal(0, 1.6, n)) * 4) / 4
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + np.round(rng.random(n) * 4) / 4
    l = np.minimum(o, c) - np.round(rng.random(n) * 4) / 4
    vol = rng.poisson(2000, n).astype(float)
    buy = np.round(vol * rng.uniform(0.35, 0.65, n))
    vol[rng.random(n) < 0.03] = 0.0                       # a few quiet minutes
    buy = np.minimum(buy, vol)
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "buy": buy, "sell": vol - buy, "vol": vol},
                        index=pd.Index(T0 + 60 * np.arange(n, dtype=np.int64), name="time"))


class TestSwingLegs(unittest.TestCase):
    def test_zigzag_legs_and_confirmation(self):
        H, L = path([100, 120, 105, 125, 110, 118])      # last move (+8) never confirmed by a retrace
        m = frame(H, L, O=[100] * len(H))
        legs = ff.swing_legs(m["h"].to_numpy(), m["l"].to_numpy(), 8.0)
        self.assertEqual([(lg["dir"], lg["start_px"], lg["end_px"]) for lg in legs],
                         [(1, 100.0, 120.0), (-1, 120.0, 105.0), (1, 105.0, 125.0), (-1, 125.0, 110.0)])
        # the 120 high is bar 19 (H = 120); the first bar with L <= 112 is bar 27 (L = 112)
        self.assertEqual((legs[0]["end_i"], legs[0]["confirm_i"]), (19, 27))
        self.assertEqual(legs[0]["start_i"], 0)          # the 100 low (bar 0), confirmed at bar 8 (H = 109)
        for lg in legs:
            self.assertGreater(lg["confirm_i"], lg["end_i"])
        self.assertEqual(legs[1]["start_i"], legs[0]["end_i"])

    def test_retrace_smaller_than_swing_does_not_end_a_leg(self):
        H, L = path([100, 110, 103, 121, 112])           # 7-pt dip: no leg; then 9-pt retrace confirms 121
        m = frame(H, L, O=[100] * len(H))
        legs = ff.swing_legs(m["h"].to_numpy(), m["l"].to_numpy(), 8.0)
        self.assertEqual([(lg["dir"], lg["start_px"], lg["end_px"]) for lg in legs], [(1, 100.0, 121.0)])

    def test_first_leg_down(self):
        H, L = path([100, 90, 99])
        m = frame(H, L, O=[100] * len(H))
        legs = ff.swing_legs(m["h"].to_numpy(), m["l"].to_numpy(), 8.0)
        self.assertEqual([(lg["dir"], lg["start_px"], lg["end_px"]) for lg in legs], [(-1, 100.0, 90.0)])

    def test_wide_first_bar_is_both_extremes(self):
        # bar 0 spans 100-110; bar 1 confirms both its high (L 101 <= 102) and its low (H 109 >= 108).
        # The earlier extreme goes first (tie: the high), so the first leg is down from 110.
        H, L = [110, 109], [100, 101]
        h2, l2 = path([101, 90, 99])
        H, L = np.array(H + h2, float), np.array(L + l2, float)
        legs = ff.swing_legs(H, L, 8.0)
        self.assertEqual([(lg["dir"], lg["start_px"], lg["end_px"], lg["start_i"]) for lg in legs],
                         [(-1, 110.0, 90.0, 0)])

    def test_empty(self):
        self.assertEqual(ff.swing_legs(np.array([]), np.array([]), 8.0), [])
        self.assertEqual(ff.pane_effort_legs(frame([], [], O=[]), T0, 8.0), [])


class TestLegEffort(unittest.TestCase):
    def test_stats_and_ratio(self):
        H, L = path([100, 110, 100, 110, 100, 112, 100])
        n = len(H)
        vol = np.full(n, 100.0)
        vol[20:30] = 300.0                               # the second up leg (bars 20..29) is heavier
        buy = vol * 0.6
        m = frame(H, L, O=[100] * n, vol=vol, buy=buy)
        legs = ff.pane_effort_legs(m, T0, 8.0)
        self.assertEqual([lg["dir"] for lg in legs], ["up", "down", "up", "down", "up"])
        up1, dn1, up2, dn2, up3 = legs
        # up1 starts at the 100 low (bar 0): bars 1..9, 10 pts, vol 900, delta 0.2 * 900
        self.assertEqual((up1["start_px"], up1["pts"], up1["vol"]), (100.0, 10.0, 900.0))
        self.assertAlmostEqual(up1["delta"], 180.0)
        self.assertEqual(up1["cpp"], 90.0)
        self.assertAlmostEqual(up1["dpp"], 18.0)
        self.assertIsNone(up1["ratio"])                  # first up leg
        # dn1: bars 10..19, -10 pts
        self.assertEqual((dn1["pts"], dn1["vol"], dn1["cpp"]), (-10.0, 1000.0, 100.0))
        self.assertAlmostEqual(dn1["dpp"], 20.0)         # delta per point keeps the delta's sign
        self.assertIsNone(dn1["ratio"])                  # first down leg
        # up2: bars 20..29 at 300 each
        self.assertEqual((up2["pts"], up2["vol"], up2["cpp"]), (10.0, 3000.0, 300.0))
        self.assertAlmostEqual(up2["ratio"], 300.0 / 90.0)
        self.assertAlmostEqual(dn2["ratio"], 1.0)
        # up3: 12 pts on 12 bars of 100 -> cpp 100, ratio vs up2 = 1/3
        self.assertEqual((up3["pts"], up3["cpp"]), (12.0, 100.0))
        self.assertAlmostEqual(up3["ratio"], 100.0 / 300.0)
        # the legs' (start, end] spans tile bars 1 .. last end without overlap
        last_bar = (legs[-1]["end_time"] - T0) // 60
        self.assertEqual(sum(lg["vol"] for lg in legs), float(vol[1:last_bar + 1].sum()))
        for a, b in zip(legs[:-1], legs[1:]):
            self.assertEqual(b["start_time"], a["end_time"])
            self.assertEqual(b["start_px"], a["end_px"])

    def test_every_leg_moves_at_least_the_swing(self):
        for seed in range(6):
            for swing in (1.0, 4.0, 8.0):
                legs = ff.pane_effort_legs(random_minutes(seed), T0, swing)
                with self.subTest(seed=seed, swing=swing):
                    self.assertTrue(legs)
                    self.assertTrue(all(abs(lg["pts"]) >= swing for lg in legs))
                    self.assertTrue(all(a["dir"] != b["dir"] for a, b in zip(legs[:-1], legs[1:])))

    def test_zero_point_leg(self):
        # swing_legs never yields a zero-point leg; leg_effort still guards one (cpp / dpp None) and the
        # next leg in that direction gets no ratio (its previous cpp is None)
        legs = [dict(dir=1, start_i=0, end_i=2, confirm_i=3, start_px=100.0, end_px=100.0),
                dict(dir=-1, start_i=2, end_i=4, confirm_i=5, start_px=100.0, end_px=90.0),
                dict(dir=1, start_i=4, end_i=6, confirm_i=7, start_px=90.0, end_px=100.0),
                dict(dir=1, start_i=6, end_i=8, confirm_i=9, start_px=100.0, end_px=105.0)]
        vol, delta = np.full(10, 10.0), np.full(10, 2.0)
        out = ff.leg_effort(legs, vol, delta)
        self.assertEqual((out[0]["pts"], out[0]["cpp"], out[0]["dpp"], out[0]["ratio"]), (0.0, None, None, None))
        self.assertEqual(out[2]["cpp"], 2.0)
        self.assertIsNone(out[2]["ratio"])
        self.assertAlmostEqual(out[3]["ratio"], (20.0 / 5.0) / 2.0)

    def test_zero_volume_previous_leg_gives_no_ratio(self):
        H, L = path([100, 110, 100, 110, 100])
        n = len(H)
        vol = np.full(n, 100.0)
        vol[:10] = 0.0                                   # the first up leg trades nothing
        m = frame(H, L, O=[100] * n, vol=vol)
        legs = ff.pane_effort_legs(m, T0, 8.0)
        self.assertEqual(legs[0]["cpp"], 0.0)
        self.assertIsNone(legs[2]["ratio"])

    def test_swing_size_parameter(self):
        H, L = path([100, 110, 104, 116, 100])           # a 6-pt retrace is a leg at swing 4, not at 8
        m = frame(H, L, O=[100] * len(H))
        self.assertEqual(len(ff.pane_effort_legs(m, T0, 4.0)), 3)
        self.assertEqual(len(ff.pane_effort_legs(m, T0, 8.0)), 1)

    def test_bars_before_start_are_ignored(self):
        H, L = path([90, 100, 90, 110, 100])
        m = frame(H, L, O=[90] * len(H))
        start = T0 + 60 * 20                              # starts at the 90 low (bar 20)
        legs = ff.pane_effort_legs(m, start, 8.0)
        self.assertTrue(all(lg["start_time"] >= start for lg in legs))
        self.assertEqual((legs[0]["start_time"], legs[0]["start_px"], legs[0]["end_px"]), (start, 90.0, 110.0))


class TestCausality(unittest.TestCase):
    """A leg appears only once its confirm bar is in the data, and its stats never change after."""

    def check_prefixes(self, m, swing):
        full = ff.pane_effort_legs(m, int(m.index[0]), swing)
        self.assertGreater(len(full), 3)
        times = m.index.to_numpy()
        for k in range(1, len(m) + 1):
            part = ff.pane_effort_legs(m.iloc[:k], int(m.index[0]), swing)
            self.assertEqual(part, [lg for lg in full if lg["confirm_time"] <= times[k - 1]], f"prefix {k}")
        for lg in full:
            self.assertLess(lg["end_time"], lg["confirm_time"])

    def test_causal_synthetic(self):
        for seed in range(6):
            for swing in (4.0, 8.0):
                with self.subTest(seed=seed, swing=swing):
                    self.check_prefixes(random_minutes(seed), swing)

    def test_causal_real_session(self):
        F = load_real()[REAL_DAYS[2]]                     # 2026-05-01, Databento
        m = ff.minutes_from_seconds(F)
        m = m[m.index >= T0]
        self.check_prefixes(m, 8.0)


class TestLiveLeg(unittest.TestCase):
    """CR-DM: the leg in progress from swing_legs' own end state, with leg_effort's numbers."""

    def test_swing_legs_is_the_scan(self):
        for seed in range(4):
            m = random_minutes(seed)
            H, L = m["h"].to_numpy(float), m["l"].to_numpy(float)
            self.assertEqual(ff.swing_legs(H, L, 8.0), ff._swing_scan(H, L, 8.0)[0])

    def test_hand_example(self):
        # 99 low (bar 0) confirmed at bar 2 (H 110 >= 107); up 99 -> 110 confirmed at bar 3 (L 102 <= 102);
        # down 110 -> 100 (bar 4) confirmed at bar 6 (H 109 >= 108); then up from 100 to 109 so far (bars 7-8
        # neither retrace to 101 nor make a new high).
        H = [100, 105, 110, 104, 102, 105, 109, 106, 108]
        L = [99, 101, 106, 102, 100, 101, 105, 104, 105]
        vol = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        m = frame(H, L, vol=vol, buy=[v * 0.75 for v in vol])
        legs = ff.pane_effort_legs(m, T0, 8.0)
        self.assertEqual([(lg["dir"], lg["start_px"], lg["end_px"]) for lg in legs],
                         [("up", 99.0, 110.0), ("down", 110.0, 100.0)])
        live = ff.pane_live_leg(m, T0, 8.0)
        # bars (4, 6]: vol 60 + 70, delta half of it
        self.assertEqual((live["dir"], live["start_px"], live["end_px"], live["pts"]), ("up", 100.0, 109.0, 9.0))
        self.assertEqual((live["start_time"], live["end_time"], live["last_time"]), (T0 + 240, T0 + 360, T0 + 480))
        self.assertEqual((live["vol"], live["delta"], live["minutes"]), (130.0, 65.0, 2))
        self.assertAlmostEqual(live["cpp"], 130 / 9)
        self.assertAlmostEqual(live["dpp"], 65 / 9)
        self.assertAlmostEqual(live["prev_cpp"], 50 / 11)                 # the up leg, bars (0, 2]
        self.assertAlmostEqual(live["ratio"], (130 / 9) / (50 / 11))
        self.assertEqual(ff.pane_effort_legs(m, T0, 8.0), legs)          # confirmed legs untouched

    def test_padded_live_session(self):
        # a live payload pads to 13:00 with flat zero-volume minutes: same leg, last_time = the last traded minute
        H = [100, 105, 110, 104, 102, 105, 109, 106, 108]
        L = [99, 101, 106, 102, 100, 101, 105, 104, 105]
        m = frame(H, L, vol=[10, 20, 30, 40, 50, 60, 70, 80, 90])
        pad = frame(H + [106.5] * 5, L + [106.5] * 5, vol=[10, 20, 30, 40, 50, 60, 70, 80, 90] + [0] * 5)
        a, b = ff.pane_live_leg(m, T0, 8.0), ff.pane_live_leg(pad, T0, 8.0)
        self.assertEqual(a, b)
        self.assertEqual(b["last_time"], T0 + 480)

    def test_none_before_the_first_confirmation(self):
        m = frame([100, 101, 102], [99, 100, 101])
        self.assertIsNone(ff.pane_live_leg(m, T0, 8.0))
        self.assertIsNone(ff.pane_live_leg(m.iloc[:0], T0, 8.0))

    def check_handoff(self, m, swing):
        """On the bars just before a leg is confirmed, the live leg IS that leg (same start, extreme, bars, cpp,
        ratio); prev_cpp is the previous same-direction leg's cpp; the live leg never reads a later bar."""
        full = ff.pane_effort_legs(m, int(m.index[0]), swing)
        times = m.index.to_numpy()
        keys = ("dir", "start_time", "end_time", "start_px", "end_px", "pts", "vol", "delta", "cpp", "dpp", "ratio")
        checked = 0
        for j, lg in enumerate(full):
            k = int(np.searchsorted(times, lg["confirm_time"]))      # bars [0, k) end just before confirmation
            live = ff.pane_live_leg(m.iloc[:k], int(m.index[0]), swing)
            if j == 0:
                continue   # the first leg starts at the first confirmed extreme, so it is not a tail before it
            self.assertEqual({x: live[x] for x in keys}, {x: lg[x] for x in keys}, f"leg {j}")
            prev = next((p for p in reversed(full[:j]) if p["dir"] == lg["dir"]), None)
            self.assertEqual(live["prev_cpp"], None if prev is None else prev["cpp"])
            self.assertEqual(live["last_time"], int(times[:k][m["vol"].to_numpy()[:k] > 0][-1]))
            checked += 1
        self.assertGreater(checked, 2)

    def test_handoff_synthetic(self):
        for seed in range(6):
            for swing in (4.0, 8.0):
                with self.subTest(seed=seed, swing=swing):
                    self.check_handoff(random_minutes(seed), swing)

    def test_handoff_real_session(self):
        F = load_real()[REAL_DAYS[2]]                     # 2026-05-01, Databento
        m = ff.minutes_from_seconds(F)
        m = m[m.index >= T0]
        self.check_handoff(m, 8.0)


if __name__ == "__main__":
    unittest.main()
