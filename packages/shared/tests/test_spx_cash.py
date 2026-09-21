"""CR-BH / CR-BI — SPX-cash hybrid series: segment choice, lag shift, filter, settlement print."""
import datetime as dt

import pandas as pd

from packages.shared.spx_cash import (
    CUTOVER, build_series, daily_ohlc, is_lagged, normalize_minutes, series_segment, settlement_prints,
)


def _day(d, spot, stock, start=(6, 30), end=(13, 0)):
    t0, t1 = dt.datetime.combine(d, dt.time(*start)), dt.datetime.combine(d, dt.time(*end))
    mins = pd.date_range(t0, t1, freq="1min")
    return pd.DataFrame({"minute": mins, "spot": [spot(i) for i in range(len(mins))],
                         "stock": [stock(i) for i in range(len(mins))]})


def test_segments_and_lag_windows():
    assert series_segment(dt.date(2023, 11, 8)) == "spot_shifted_2023"
    assert series_segment(CUTOVER) == "stock_price"
    assert is_lagged(dt.date(2023, 9, 7)) and not is_lagged(dt.date(2023, 9, 8))
    assert is_lagged(dt.date(2023, 10, 24)) and not is_lagged(dt.date(2023, 11, 9))


def test_stock_segment_uses_stock_price_and_window_starts_0633():
    d = dt.date(2025, 3, 4)
    raw = normalize_minutes(_day(d, spot=lambda i: 9999.0 + i, stock=lambda i: 5000.0 + 0.1 * i))
    clean, _ = build_series(raw)
    o = daily_ohlc(clean).loc[d]
    assert o["first"] == dt.time(6, 33) and abs(o["open"] - 5000.3) < 1e-9
    assert abs(o["close"] - (5000.0 + 0.1 * 390)) < 1e-9


def test_lagged_day_shifts_15_minutes_and_closes_on_stock_median():
    d = dt.date(2023, 6, 1)
    # spot frozen for the first 16 prints, then a ramp; stock flat at 4200 in the last 5 minutes
    raw = normalize_minutes(_day(d, spot=lambda i: 4100.0 if i < 16 else 4100.0 + 0.2 * i, stock=lambda i: 4200.0))
    clean, dropped = build_series(raw)
    assert (dropped["reason"] == "frozen_open").sum() == 16
    o = daily_ohlc(clean).loc[d]
    assert o["last"] == dt.time(13, 0) and o["close"] == 4200.0          # median stock 12:56–13:00
    # print stamped 06:48 (i=18) is the index at 06:33 after the shift
    assert o["first"] == dt.time(6, 33) and abs(o["open"] - (4100.0 + 0.2 * 18)) < 1e-9


def test_isolated_spike_dropped_but_trend_kept():
    d = dt.date(2025, 3, 5)
    def stock(i):
        if i == 100:
            return 5000.0 + 0.05 * i + 60.0                       # one-print spike, > 0.3 %
        return 5000.0 + 0.05 * i + (30.0 if i >= 200 else 0.0)     # level shift at i=200 is real
    clean, dropped = build_series(normalize_minutes(_day(d, spot=lambda i: 1.0, stock=stock)))
    assert list(dropped["reason"]) == ["spike"]
    assert daily_ohlc(clean).loc[d, "high"] < 5000.0 + 0.05 * 390 + 30.0 + 1e-6


def test_settlement_print_is_last_print_in_window_and_absent_on_early_close():
    d1, d2 = dt.date(2025, 3, 6), dt.date(2025, 7, 3)
    raw = pd.concat([_day(d1, lambda i: 1.0, lambda i: 5000.0 + 0.1 * i),
                     _day(d2, lambda i: 1.0, lambda i: 6000.0 + 0.1 * i, end=(10, 15))])
    clean, _ = build_series(normalize_minutes(raw))
    s = settlement_prints(clean)
    assert abs(s[d1] - (5000.0 + 0.1 * 390)) < 1e-9 and d2 not in s.index
