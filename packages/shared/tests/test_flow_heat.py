"""Tests for packages/shared/flow_heat.py (CR-DC): price cells and absorption marks.

Run with:
    python -m unittest packages.shared.tests.test_flow_heat -v

The 1-s rule (sell at the second's low, buy at its high), exact trade cells, `es_flow_price_1m`
flooring, the absorption rule on hand-built minutes (qualify, hold, break, merge, mirror, `until`),
and causality: on every prefix, the marks equal the full session's marks confirmed in the prefix,
with breaks seen only before the prefix end (synthetic and a recorded Databento session).
CR-DV: live marks on closed minutes only, provisional marks, and the 2026-10-30 replay (recorded Ironbeam trades).
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

from packages.shared import flow_heat as fh  # noqa: E402

T0 = 1_777_642_200   # 2026-05-01 06:30 PT
FIXTURE = REPO_ROOT / "packages/shared/tests/fixtures/cr_cd_flow_1s_3sessions.npz"
FIXTURE_1030 = REPO_ROOT / "packages/shared/tests/fixtures/cr_dv_trades_1030_rth.npz"   # CR-DV (record_fixture_1030.py)


def bars(H, L, t0=T0):
    return pd.DataFrame({"h": H, "l": L}, index=pd.Index([t0 + 60 * i for i in range(len(H))]))


def cells(rows):
    """rows: (minute index, price, buy, sell)."""
    return pd.DataFrame({"t": [T0 + 60 * r[0] for r in rows], "p": fh.to_ticks([r[1] for r in rows]),
                         "buy": [float(r[2]) for r in rows], "sell": [float(r[3]) for r in rows]})


def background(n_minutes, price=100.0, vol=10.0):
    """One small sell and buy cell per minute (the running median's population)."""
    return [(i, price + 2.0, vol, vol) for i in range(n_minutes)]


class TestAssignment(unittest.TestCase):
    def test_seconds_sell_at_low_buy_at_high(self):
        ts = np.array([T0, T0 + 1, T0 + 61])
        c = fh.cells_from_seconds(ts, high=[100.5, 100.25, 101.0], low=[100.0, 100.25, 100.75],
                                  buy=[3, 4, 5], sell=[7, 2, 1])
        got = {(int(r.t), r.p * fh.TICK): (r.buy, r.sell) for r in c.itertuples()}
        self.assertEqual(got, {(T0, 100.0): (0.0, 7.0), (T0, 100.25): (4.0, 2.0), (T0, 100.5): (3.0, 0.0),
                               (T0 + 60, 100.75): (0.0, 1.0), (T0 + 60, 101.0): (5.0, 0.0)})

    def test_seconds_single_price_second_puts_both_sides_on_it(self):
        c = fh.cells_from_seconds([T0], high=[99.75], low=[99.75], buy=[2], sell=[3])
        self.assertEqual(c[["p", "buy", "sell"]].values.tolist(), [[399, 2.0, 3.0]])

    def test_seconds_volume_is_conserved(self):
        rng = np.random.default_rng(1)
        n = 500
        ts = T0 + np.sort(rng.integers(0, 3600, n))
        lo = 5000 + rng.integers(0, 40, n) * 0.25
        hi = lo + rng.integers(0, 4, n) * 0.25
        b, s = rng.integers(0, 50, n), rng.integers(0, 50, n)
        c = fh.cells_from_seconds(ts, hi, lo, b, s)
        self.assertEqual(c["buy"].sum(), b.sum())
        self.assertEqual(c["sell"].sum(), s.sum())
        self.assertTrue(c["t"].isin(fh.minute_of(ts)).all())

    def test_trades_exact_and_unknown_dropped(self):
        ts = np.array([T0 + 5, T0 + 6, T0 + 7, T0 + 65])
        c = fh.cells_from_trades(ts, price=[100.25, 100.25, 100.0, 100.5], size=[2, 3, 9, 1],
                                 side=["BUY", "SELL", "0", "SELL"])
        got = {(int(r.t), r.p * fh.TICK): (r.buy, r.sell) for r in c.itertuples()}
        self.assertEqual(got, {(T0, 100.25): (2.0, 3.0), (T0 + 60, 100.5): (0.0, 1.0)})

    def test_price_1m_floors_off_grid(self):
        c = fh.cells_from_price_1m([T0, T0], price=[100.30, 100.25], buy=[1, 2], sell=[0, 4])
        self.assertEqual(c[["p", "buy", "sell"]].values.tolist(), [[401, 3.0, 4.0]])


class TestAbsorption(unittest.TestCase):
    P = fh.AbsorbParams(k=3.0, m=3, min_cells=15)

    def setUp(self):
        n = 40
        self.H = [104.0] * n
        self.L = [101.0] * n

    def test_sellers_absorbed_qualifies_holds_and_breaks(self):
        H, L = self.H, self.L
        L[20] = 100.0                       # the absorbing minute's low = P
        L[30] = 99.5                        # first bar below P - 1 tick: the break
        rows = background(40) + [(20, 100.0, 4, 50)]
        m = fh.absorption_marks(cells(rows), bars(H, L), T0, T0 + 60 * 40, self.P)
        self.assertEqual(len(m), 1)
        mk = m[0]
        self.assertEqual((mk["kind"], mk["price"], mk["aggressor"], mk["opposite"]), ("sellers_absorbed", 100.0, 50.0, 4.0))
        self.assertEqual(mk["time"], T0 + 1200)
        self.assertEqual(mk["confirm_time"], T0 + 1200 + 4 * 60)
        self.assertEqual(mk["break_time"], T0 + 1800)
        self.assertEqual(mk["held_min"], 10)
        self.assertEqual(mk["median"], 10.0)

    def test_one_tick_below_is_still_a_hold(self):
        L = self.L
        L[20], L[22] = 100.0, 99.75       # P - 1 tick: allowed
        m = fh.absorption_marks(cells(background(40) + [(20, 100.0, 0, 50)]), bars(self.H, L), T0, T0 + 2400, self.P)
        self.assertEqual(len(m), 1)
        self.assertIsNone(m[0]["break_time"])
        self.assertEqual(m[0]["held_min"], 20)

    def test_two_ticks_below_inside_window_is_no_mark(self):
        L = self.L
        L[20], L[23] = 100.0, 99.5
        m = fh.absorption_marks(cells(background(40) + [(20, 100.0, 0, 50)]), bars(self.H, L), T0, T0 + 2400, self.P)
        self.assertEqual(m, [])

    def test_below_k_times_median_is_no_mark(self):
        L = self.L
        L[20] = 100.0
        m = fh.absorption_marks(cells(background(40) + [(20, 100.0, 0, 29)]), bars(self.H, L), T0, T0 + 2400, self.P)
        self.assertEqual(m, [])

    def test_warmup_min_cells(self):
        L = self.L
        L[10] = 100.0
        # 10 prior minutes x 1 sell cell = 10 < 15: no mark at minute 10
        m = fh.absorption_marks(cells(background(40) + [(10, 100.0, 0, 500)]), bars(self.H, L), T0, T0 + 2400, self.P)
        self.assertEqual(m, [])

    def test_lowest_qualifying_price_per_minute_and_merge(self):
        L = self.L
        L[20] = L[21] = 100.0
        rows = background(40) + [(20, 100.0, 0, 60), (20, 100.5, 0, 60), (21, 100.25, 0, 70)]
        m = fh.absorption_marks(cells(rows), bars(self.H, L), T0, T0 + 2400, self.P)
        self.assertEqual([x["price"] for x in m], [100.0])          # minute 21 at +1 tick merges
        self.assertEqual(m[0]["merged_vol"], 70.0)
        self.assertEqual(m[0]["merges"], [[T0 + 21 * 60 + 240, 70.0]])  # CR-DU: (confirm_time, contracts)

    def test_buyers_absorbed_mirror(self):
        H = self.H
        H[20] = 105.0
        H[33] = 105.5
        rows = background(40, price=103.0) + [(20, 105.0, 50, 2)]
        m = fh.absorption_marks(cells(rows), bars(H, self.L), T0, T0 + 2400, self.P)
        self.assertEqual([(x["kind"], x["price"], x["opposite"], x["break_time"]) for x in m],
                         [("buyers_absorbed", 105.0, 2.0, T0 + 33 * 60)])

    def test_until_hides_unconfirmed_and_later_breaks(self):
        L = self.L
        L[20], L[30] = 100.0, 99.5
        c, b = cells(background(40) + [(20, 100.0, 0, 50)]), bars(self.H, L)
        self.assertEqual(fh.absorption_marks(c, b, T0, T0 + 23 * 60, self.P), [])   # confirm at 24 min
        m = fh.absorption_marks(c, b, T0, T0 + 25 * 60, self.P)
        self.assertEqual(len(m), 1)
        self.assertIsNone(m[0]["break_time"])
        self.assertEqual(m[0]["held_min"], 5)

    def test_params(self):
        L = self.L
        L[20] = 100.0
        c, b = cells(background(40) + [(20, 100.0, 0, 50)]), bars(self.H, L)
        self.assertEqual(fh.absorption_marks(c, b, T0, T0 + 2400, fh.AbsorbParams(k=6.0)), [])
        L[24] = 99.5      # outside m=3's window (bars 20..23), inside m=5's
        self.assertEqual(len(fh.absorption_marks(c, bars(self.H, L), T0, T0 + 2400, fh.AbsorbParams(m=3))), 1)
        self.assertEqual(fh.absorption_marks(c, bars(self.H, L), T0, T0 + 2400, fh.AbsorbParams(m=5)), [])


def _causal_view(full, end):
    out = []
    for mk in full:
        if mk["confirm_time"] > end:
            continue
        brk = mk["break_time"] if mk["break_time"] is not None and mk["break_time"] < end else None
        merged = sum(v for at, v in mk["merges"] if at <= end)          # CR-DU: merged_vol as of `end`
        out.append((mk["kind"], mk["time"], mk["p"], mk["aggressor"], mk["opposite"], brk, merged))
    return out


def _view(marks):
    return [(mk["kind"], mk["time"], mk["p"], mk["aggressor"], mk["opposite"], mk["break_time"], mk["merged_vol"])
            for mk in marks]


class TestCausality(unittest.TestCase):
    def check(self, c, b, start, end, params):
        full = fh.absorption_marks(c, b, start, end, params)
        for e in range(start + 60, end + 1, 60 * 7):
            pre = fh.absorption_marks(c[c["t"] < e], b[b.index < e], start, e, params)
            self.assertEqual(_view(pre), _causal_view(full, e), f"prefix end {e}")
        return full

    def test_random_sessions(self):
        rng = np.random.default_rng(7)
        total = 0
        for seed in range(4):
            n = 240
            px = 5000 + np.cumsum(rng.normal(0, 1.0, n))
            H = np.round((px + rng.uniform(0.25, 2, n)) * 4) / 4
            L = np.round((px - rng.uniform(0.25, 2, n)) * 4) / 4
            rows = []
            for i in range(n):
                for p in np.arange(L[i], H[i] + 1e-9, 0.25):
                    rows.append((i, p, rng.exponential(20), rng.exponential(20) * (8 if p == L[i] and rng.random() < 0.15 else 1)))
            total += len(self.check(cells(rows), bars(list(H), list(L)), T0, T0 + 60 * n, fh.AbsorbParams()))
        self.assertGreater(total, 0)

    @unittest.skipUnless(FIXTURE.exists(), "recorded Databento fixture missing")
    def test_recorded_sessions_1s_cells(self):
        z = np.load(FIXTURE)
        for d in ("2026-01-06", "2026-03-02", "2026-05-01"):
            F = {k: z[f"{d}/{k}"] for k in ("t", "buy", "sell", "h", "l")}
            t = F["t"]
            c = fh.cells_from_seconds(t, F["h"], F["l"], F["buy"], F["sell"])
            self.assertEqual(c["sell"].sum(), F["sell"].sum())
            m = pd.DataFrame({"t": fh.minute_of(t), "h": F["h"], "l": F["l"]}).groupby("t").agg(
                h=("h", "max"), l=("l", "min"))
            start = int((t.min() // 60 + 15) * 60)      # 06:30 PT (the fixture starts 06:15)
            end = int((t.max() // 60) * 60)
            full = self.check(c, m, start, end, fh.AbsorbParams())
            self.assertGreater(len(full), 0, d)


def _key(mk):
    return (mk["kind"], mk["time"], mk["p"])


class TestLiveMarks(unittest.TestCase):
    """CR-DV: `end` = the last closed minute; candidates inside their hold window come back provisional."""
    P = fh.AbsorbParams(k=3.0, m=3, min_cells=15)

    def setUp(self):
        n = 40
        self.H = [104.0] * n
        self.L = [101.0] * n
        self.L[20] = 100.0
        self.rows = background(40) + [(20, 100.0, 4, 50)]

    def live(self, end, provisional=True, L=None):
        c = cells(self.rows)
        b = bars(self.H, L or self.L)
        return fh.absorption_marks(c[c["t"] < end], b[b.index < end], T0, end, self.P, provisional=provisional)

    def test_provisional_until_the_hold_elapses_then_confirmed(self):
        self.assertEqual(self.live(T0 + 60 * 20), [])                      # minute 20 not closed yet
        for k in (21, 22, 23):                                              # closed, hold window not elapsed
            m = self.live(T0 + 60 * k)
            self.assertEqual([(mk["provisional"], mk["price"], mk["confirm_time"]) for mk in m],
                             [(True, 100.0, T0 + 60 * 24)])
            self.assertEqual(self.live(T0 + 60 * k, provisional=False), [])
        m = self.live(T0 + 60 * 24)
        self.assertEqual([(mk["provisional"], mk["confirm_time"]) for mk in m], [(False, T0 + 60 * 24)])

    def test_old_call_confirms_before_the_hold_elapsed(self):
        # the repaint: end = segment end with only bars so far (cells / bars cut at minute 21) confirms at once
        c, b = cells(self.rows), bars(self.H, self.L)
        old = fh.absorption_marks(c[c["t"] < T0 + 60 * 21], b[b.index < T0 + 60 * 21], T0, T0 + 60 * 40, self.P)
        self.assertEqual(len(old), 1)
        self.assertNotIn(True, [mk["provisional"] for mk in old])

    def test_provisional_removed_when_broken_first(self):
        L = list(self.L)
        L[22] = 99.5                                                        # breaks inside the hold window
        self.assertEqual(len(self.live(T0 + 60 * 22, L=L)), 1)
        self.assertEqual(self.live(T0 + 60 * 23, L=L), [])
        self.assertEqual(self.live(T0 + 60 * 40, L=L), [])

    def test_provisional_adds_no_volume_until_it_confirms(self):
        self.rows.append((26, 100.25, 0, 60))                              # within 1 tick of the 20 mark
        self.L[26] = 100.25
        m = self.live(T0 + 60 * 27)
        self.assertEqual([(mk["time"], mk["merged_vol"], mk["provisional"]) for mk in m], [(T0 + 1200, 0.0, False)])
        self.assertEqual(m[0]["merges"], [])                                # CR-DU merges: no entry while provisional
        m = self.live(T0 + 60 * 30)
        self.assertEqual([(mk["time"], mk["merged_vol"]) for mk in m], [(T0 + 1200, 60.0)])
        self.assertEqual(m[0]["merges"], [[T0 + 60 * 26 + (self.P.m + 1) * 60, 60.0]])   # added once it confirms

    def test_complete_call_unchanged_by_default(self):
        c, b = cells(self.rows), bars(self.H, self.L)
        self.assertEqual(fh.absorption_marks(c, b, T0, T0 + 60 * 22, self.P), [])   # unconfirmed at end: not returned


@unittest.skipUnless(FIXTURE_1030.exists(), "recorded 2026-10-30 trades fixture missing")
class TestLiveReplay1030(unittest.TestCase):
    """2026-10-30 RTH (XCME:ES.Z26) polled every minute + 15 s, each read as of that moment: cells received by then,
    1-minute bars from seconds received by then (the flow-1s worker, 2 s lateness)."""
    A, B = 1_793_367_000, 1_793_390_400                                    # 13:30, 20:00 UTC

    @classmethod
    def setUpClass(cls):
        z = np.load(FIXTURE_1030)
        cls.z = {k: z[k] for k in z.files}
        cls.full = fh.absorption_marks(cls.cells(cls.B + 3600), cls.bars(cls.B + 3600), cls.A, cls.B, fh.AbsorbParams())

    @classmethod
    def cells(cls, T):
        z = cls.z
        k = z["cell_rcv"] <= T
        return pd.DataFrame({"t": z["cell_t"][k], "p": z["cell_p"][k], "buy": z["cell_buy"][k], "sell": z["cell_sell"][k]})

    @classmethod
    def bars(cls, T):
        z = cls.z
        k = (z["sec_ts"] < T - 2) & (z["sec_rcv"] <= T)
        df = pd.DataFrame({"t": fh.minute_of(z["sec_ts"][k]), "h": z["sec_h"][k], "l": z["sec_l"][k]})
        return df.groupby("t").agg(h=("h", "max"), l=("l", "min"))

    @classmethod
    def live(cls, T):
        end = min(cls.B, (T - 3) // 60 * 60)
        c, b = cls.cells(T), cls.bars(T)
        return end, fh.absorption_marks(c[c["t"] < end], b[b.index < end], cls.A, end, fh.AbsorbParams(), provisional=True)

    def test_full_session_has_marks(self):
        self.assertGreater(len(self.full), 0)

    def test_no_confirmed_mark_before_its_hold_and_prefix_equal(self):
        for T in range(self.A + 75, self.B + 300, 60):
            end, m = self.live(T)
            conf = [mk for mk in m if not mk["provisional"]]
            for mk in conf:
                self.assertLessEqual(mk["confirm_time"], end)
                self.assertLessEqual(mk["confirm_time"], T - 3)
            self.assertEqual(_view(conf), _causal_view(self.full, end), f"T {T}")
            prov = {_key(mk) for mk in m if mk["provisional"]}
            due = {_key(mk) for mk in self.full if mk["time"] + 60 <= end < mk["confirm_time"]}
            self.assertLessEqual(due, prov, f"T {T}")                        # every later-confirmed mark shows hollow first
            self.assertFalse(prov & {_key(mk) for mk in self.full if mk["confirm_time"] <= end})

    def test_1539_sellers_case(self):
        t, p = 1_793_374_740, fh.to_ticks(7633.5)                          # 15:39 UTC (08:39 PT), sellers 7633.50
        key = ("sellers_absorbed", t, int(p))
        self.assertNotIn(key, {_key(mk) for mk in self.full})               # not a mark on the full session
        T = t + 60                                                          # the 15:40:00 poll
        c, b = self.cells(T), self.bars(T)
        old = fh.absorption_marks(c, b, self.A, self.B, fh.AbsorbParams())  # pre-CR-DV live call: end = 13:00 PT
        self.assertIn(key, {_key(mk) for mk in old})                        # shown as confirmed: the repaint
        self.assertEqual(self.live(T)[1], [mk for mk in self.live(T)[1] if mk["time"] < t])   # 15:39 not closed yet
        shown = {}
        for T in range(t + 60, t + 6 * 60, 15):
            for mk in self.live(T)[1]:
                if _key(mk) == key:
                    shown[T] = mk["provisional"]
        self.assertTrue(shown)
        self.assertTrue(all(shown.values()), shown)                          # only ever provisional (hollow)
        self.assertLess(max(shown), t + 4 * 60)                              # removed before its hold would end


if __name__ == "__main__":
    unittest.main()
