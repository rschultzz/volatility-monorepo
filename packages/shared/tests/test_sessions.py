"""CR-BI — shared session definition: NYSE-calendar sessions, T+N OHLC, post-touch positions."""
import datetime as dt

import pandas as pd

from packages.shared.sessions import post_touch_positions, session_ohlc_at, trading_sessions_only


def _frame(dates, base=100.0):
    rows = {d: {"open": base + i, "high": base + i + 2, "low": base + i - 2, "close": base + i + 1}
            for i, d in enumerate(dates)}
    return pd.DataFrame.from_dict(rows, orient="index")


def test_trading_sessions_only_drops_holiday_and_weekend():
    # Thu 2024-11-28 = Thanksgiving (ES prints RTH-window bars); Sat 2026-02-21 = stray bars
    dates = [dt.date(2024, 11, 27), dt.date(2024, 11, 28), dt.date(2024, 11, 29), dt.date(2026, 2, 21)]
    out = trading_sessions_only(_frame(dates))
    assert list(out.index) == [dt.date(2024, 11, 27), dt.date(2024, 11, 29)]   # early close 11-29 is a session


def test_session_ohlc_at_counts_sessions_not_calendar_days():
    dates = [dt.date(2024, 11, 27), dt.date(2024, 11, 29), dt.date(2024, 12, 2)]
    f = _frame(dates)
    out = session_ohlc_at(f, dt.date(2024, 11, 27), (1, 2, 5))
    assert out["session_close_t1"] == f.loc[dt.date(2024, 11, 29), "close"]
    assert out["session_open_t2"] == f.loc[dt.date(2024, 12, 2), "open"]
    assert out["session_close_t5"] is None


def test_session_ohlc_at_none_when_trade_date_not_a_session():
    f = _frame([dt.date(2024, 11, 27), dt.date(2024, 11, 29)])
    assert all(v is None for v in session_ohlc_at(f, dt.date(2024, 11, 28)).values())


def test_post_touch_positions_indexes_from_trade_date():
    dates = [dt.date(2024, 12, 2) + dt.timedelta(days=i) for i in range(4)]
    f = _frame(dates)                                   # closes 101, 102, 103, 104
    pos = post_touch_positions(f, dates[0], days_to_reach=1, drift_target=103.0, tolerance=0.5, timeframes=(1, 2, 5))
    assert pos == {1: 0, 2: 1, 5: None}                 # close 103 at, 104 above, beyond frame
    assert post_touch_positions(f, dates[0], None, 103.0, 0.5) == {1: None, 5: None, 15: None}
