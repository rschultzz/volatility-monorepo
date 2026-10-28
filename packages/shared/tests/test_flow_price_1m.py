from datetime import datetime, timedelta, timezone

import pandas as pd

from packages.shared import flow_price_1m as F

UTC = timezone.utc


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_floor_tick():
    assert F.floor_tick(6812.0) == 6812.0
    assert F.floor_tick(6812.25) == 6812.25
    assert F.floor_tick(6812.30) == 6812.25
    assert list(F.floor_tick(pd.Series([1.74, 1.75, 1.99]))) == [1.5, 1.75, 1.75]


def test_grace_three_minutes():
    m = t("2026-10-27 15:00")
    assert F.is_held(m, m + timedelta(minutes=2, seconds=59))
    assert not F.is_held(m, m + timedelta(minutes=3))


def test_close_hold_edt_and_est():
    # EDT: close 17:00 ET = 21:00 UTC, reopen 18:00 ET = 22:00 UTC
    last = t("2026-10-27 20:59")
    assert F.is_held(last, t("2026-10-27 21:55"))
    assert not F.is_held(last, t("2026-10-27 22:00"))
    assert F.is_held(t("2026-10-27 20:45"), t("2026-10-27 21:30"))
    assert not F.is_held(t("2026-10-27 20:44"), t("2026-10-27 20:47"))  # before the hold window
    # EST (after 2026-11-01): close 22:00 UTC, reopen 23:00 UTC
    assert F.is_held(t("2026-11-02 21:59"), t("2026-11-02 22:30"))
    assert not F.is_held(t("2026-11-02 21:59"), t("2026-11-02 23:00"))
    assert not F.is_held(t("2026-11-02 20:59"), t("2026-11-02 21:05"))  # 15:59 ET: not held in EST


def test_window_minute_aligned():
    since, end = F.window(t("2026-10-28 03:31:27"))
    assert since == t("2026-10-26 04:31") and end == t("2026-10-28 03:29")
    assert not F.is_held(end - timedelta(minutes=1), t("2026-10-28 03:31:27"))
    assert F.is_held(end, t("2026-10-28 03:31:27"))


def test_aggregate_trades_cells():
    tr = pd.DataFrame({
        "ts": pd.to_datetime(["2026-01-06 15:00:01.5", "2026-01-06 15:00:59.9", "2026-01-06 15:00:30",
                              "2026-01-06 15:01:00", "2026-01-06 15:00:10"], utc=True, format="ISO8601"),
        "price": [6900.25, 6900.25, 6900.5, 6900.25, 6900.25],
        "size": [3, 2, 7, 1, 4],
        "side": ["B", "A", "A", "B", "N"],
    })
    out = F.aggregate_trades(tr, "XCME:ES.H26", "ts", "side", "B", "A", "databento")
    assert list(out.columns) == F.COLS
    rows = {(r.minute_utc.strftime("%H:%M"), r.price): (r.buy_vol, r.sell_vol, r.trade_count)
            for r in out.itertuples()}
    assert rows == {("15:00", 6900.25): (3, 2, 3), ("15:00", 6900.5): (0, 7, 1), ("15:01", 6900.25): (1, 0, 1)}
