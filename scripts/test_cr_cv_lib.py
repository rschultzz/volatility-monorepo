"""CR-CV C13: unit tests for the 1-s fills, race and statistic (synthetic arrays, no DB)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import math

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cr_cv_lib as V  # noqa: E402


class approx(float):
    def __eq__(self, other):
        return math.isclose(float(self), float(other), rel_tol=1e-9, abs_tol=1e-12)


def mk(rows, start=0):
    """rows: (o, h, l, c) per consecutive second from `start`."""
    a = np.array(rows, float)
    return dict(sec=np.arange(start, start + len(a)), o=a[:, 0], h=a[:, 1], l=a[:, 2], c=a[:, 3])


def test_race_win_loss_tie_unresolved():
    S = mk([(100, 100, 100, 100), (100, 102, 100, 102), (102, 102, 97, 97)])
    assert V.race(S, 1, 100.0, 1, 2.0) == "win"
    assert V.race(S, 1, 100.0, -1, 2.0) == "loss"
    assert V.race(S, 1, 100.0, 1, 3.0) == "loss"          # 97 = -3 before +3
    T = mk([(100, 100, 100, 100), (100, 102, 98, 100)])
    assert V.race(T, 1, 100.0, 1, 2.0) == "tie"
    assert V.race(T, 1, 100.0, 1, 4.0) == "unresolved"


def test_entry_second_rule_is_loss_only():
    S = mk([(100, 104, 97.75, 100), (100, 100, 100, 100)])
    assert V.race(S, 1, 100.0, 1, 2.0, entry_k=0) == "loss"            # 2.25 adverse inside the entry second
    S2 = mk([(100, 104, 99, 100), (100, 100, 100, 100)])
    assert V.race(S2, 1, 100.0, 1, 2.0, entry_k=0) == "unresolved"     # favourable side of the entry second not credited


def test_market_fill_slippage_sign():
    S = mk([(100, 100, 100, 100)] * 5 + [(101, 101, 101, 101)], start=55)   # seconds 55..60
    k, f = V.market_fill(S, 0, 1)
    assert S["sec"][k] == 60 and f == 101.25
    k, f = V.market_fill(S, 0, -1)
    assert f == 100.75
    assert V.market_fill(S, 1, 1) is None
    assert V.bar_close_start(S, 0) == k


def test_level_fill_trade_through_and_stop_side():
    S = mk([(99, 99.75, 99, 99.5), (99.5, 100, 99.5, 100), (100, 100.25, 100, 100.25)], start=60)
    # b = +1, level 100: touch at second 61, trade-through (>= 100.25) at second 62
    k, f = V.level_fill(S, 1, 100.0, 1, -1, 0.0)
    assert S["sec"][k] == 61 and f == 100.0
    k, f = V.level_fill(S, 1, 100.0, 1, -1, V.TICK)
    assert S["sec"][k] == 62 and f == 100.0                 # fade side: resting limit at L
    k, f = V.level_fill(S, 1, 100.0, 1, 1, V.TICK)
    assert S["sec"][k] == 62 and f == 100.25                # break side: stop one tick through
    assert V.level_fill(S, 1, 100.5, 1, 1, V.TICK) is None


def test_edge_fill_trade_through_continues_past_touch():
    p = V.C.PRIMARY
    hi, lo, t = 110.0, 100.0, 0
    B = {k: np.full(V.NB, 105.0) for k in "ohlc"}
    # minute 1 touches the long limit (lo + 1 = 101) but does not trade through; minute 2 trades 100.75
    rows = [(105, 105, 105, 105)] * 60 + [(102, 102, 101, 101)] * 60 + [(101, 101, 100.75, 101)] * 60
    S = mk(rows)
    st, j, k, px = V.edge_fill(B, S, hi, lo, t, 1, 0.0, p)
    assert (st, j, px) == ("fill", 1, 101.0)
    st, j, k, px = V.edge_fill(B, S, hi, lo, t, 1, V.TICK, p)
    assert (st, j, px) == ("fill", 2, 101.0) and S["sec"][k] == 120
    B["c"][2] = hi + p.void + 0.25                            # minute 2 closes > 2 beyond the other edge -> void first
    st, j, k, px = V.edge_fill(B, S, hi, lo, t, 1, V.TICK, p)
    assert st == "no fill: box broke" and j == 2


def test_minute_agg():
    S = mk([(1, 3, 0.5, 2), (2, 4, 1, 3)] + [(5, 5, 5, 5)], start=58)
    m = V.minute_agg(S)
    assert (m["o"][0], m["h"][0], m["l"][0], m["c"][0]) == (1, 4, 0.5, 3)
    assert m["o"][1] == 5 and np.isnan(m["o"][2])


def test_expectancy_break_even_holm():
    assert V.expectancy(0.5, 4, 0.5) == -0.5
    assert V.break_even(4, 0.5) == approx(0.5625)
    assert V.break_even(2, 0.5) == approx(0.625)
    assert V.expectancy(V.break_even(2, 0.5), 2, 0.5) == approx(0.0)
    rng = np.random.default_rng(0)
    cells = [dict(p=1.0, n=100, E_draws=rng.normal(0, 1, 2000), ci95=[-2, 2]) for _ in range(16)]
    cells[0].update(p=0.0001, E_draws=rng.normal(5, 0.5, 2000), ci95=[4, 6])
    cells[1].update(n=10)
    cells[2].update(ci95=[-3, -0.1])
    out = V.holm(cells)
    assert out[0]["verdict"] == "PASS" and out[0]["p_holm"] == approx(0.0016)
    assert out[1]["verdict"] == "UNDERPOWERED" and out[2]["verdict"] == "REFUTED"
    assert out[3]["verdict"] == "INCONCLUSIVE"


def test_bootstrap_weights_shape_and_seed():
    a, b = V.boot_weights(5, 10), V.boot_weights(5, 10)
    assert a.shape == (10, 5) and (a.sum(1) == 5).all() and (a == b).all()


if __name__ == "__main__":                 # apps/web/.venv has no pytest; same tests, plain runner
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
