"""Tests for packages/shared/flow_heat.py (CR-DC): price cells and absorption marks.

Run with:
    python -m unittest packages.shared.tests.test_flow_heat -v

The 1-s rule (sell at the second's low, buy at its high), exact trade cells, `es_flow_price_1m`
flooring, the absorption rule on hand-built minutes (qualify, hold, break, merge, mirror, `until`),
and causality: on every prefix, the marks equal the full session's marks confirmed in the prefix,
with breaks seen only before the prefix end (synthetic and a recorded Databento session).
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
        out.append((mk["kind"], mk["time"], mk["p"], mk["aggressor"], mk["opposite"], brk))
    return out


def _view(marks):
    return [(mk["kind"], mk["time"], mk["p"], mk["aggressor"], mk["opposite"], mk["break_time"]) for mk in marks]


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

if __name__ == "__main__":
    unittest.main()
