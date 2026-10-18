"""CR-BN — 0DTE quote readers select PM-settled rows only.

On third Fridays ORATS files the AM-settled SPX monthly under the same OPRA
symbol as the PM SPXW; `orats_options_minute.expiry_tod` is the only tell.
The predicate is checked against an in-memory table with the same columns.
"""
import sqlite3
from datetime import datetime

from packages.shared.backtest.condor_0dte import PM_SETTLED_SQL, build_box, first_valid_entry

OPRA = "SPX   230721P04500000"


def _table(rows):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE orats_options_minute (opra_symbol TEXT, snapshot_pt TEXT, strike REAL, "
                "option_type TEXT, bid_price REAL, ask_price REAL, expiry_tod TEXT, "
                "PRIMARY KEY (opra_symbol, snapshot_pt))")
    db.executemany("INSERT INTO orats_options_minute VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return db


def test_predicate_keeps_pm_drops_am_and_null():
    db = _table([
        (OPRA, "2023-07-21 06:30:00", 4500.0, "P", 1.0, 1.2, "pm"),
        (OPRA, "2023-07-21 06:31:00", 4500.0, "P", 9.0, 9.4, "am"),
        (OPRA, "2023-07-21 06:32:00", 4500.0, "P", 1.1, 1.3, None),
        (OPRA, "2023-07-21 06:33:00", 4500.0, "P", 1.2, 1.4, "pm"),
    ])
    got = db.execute(f"SELECT snapshot_pt FROM orats_options_minute WHERE opra_symbol = ? AND {PM_SETTLED_SQL} "
                     "ORDER BY snapshot_pt", (OPRA,)).fetchall()
    assert [r[0] for r in got] == ["2023-07-21 06:30:00", "2023-07-21 06:33:00"]


def test_am_minute_cannot_set_the_entry():
    """A box whose only valid minute is an AM-contract quote gets no entry once
    the AM rows are filtered — the AM mid is never priced as the PM leg."""
    td = datetime(2023, 7, 21)
    box = build_box(td.date(), open_px=4500.0, implied_move=40.0, k_im=0.5)   # 4470/4480/4520/4530
    legs = [(box.long_put, "P"), (box.short_put, "P"), (box.short_call, "C"), (box.long_call, "C")]
    am_quotes = {(box.long_put, "P"): (0.4, 0.6), (box.short_put, "P"): (1.4, 1.6),
                 (box.short_call, "C"): (1.4, 1.6), (box.long_call, "C"): (0.4, 0.6)}
    rows = [(f"O{k:g}{t}", "2023-07-21 06:33:00", k, t, *am_quotes[(k, t)], "am") for k, t in legs]
    # PM minute: short put quote crossed → no valid credit at 06:34
    rows += [(f"O{k:g}{t}", "2023-07-21 06:34:00", k, t, *(am_quotes[(k, t)] if (k, t) != (box.short_put, "P") else (1.6, 1.4)), "pm")
             for k, t in legs]
    db = _table(rows)

    def minutes(where):
        by = {}
        for snap, k, t, bid, ask in db.execute(
                f"SELECT snapshot_pt, strike, option_type, bid_price, ask_price FROM orats_options_minute {where} "
                "ORDER BY snapshot_pt"):
            by.setdefault(datetime.fromisoformat(snap), []).append((k, t, bid, ask))
        return sorted(by.items())

    assert first_valid_entry(box, minutes("")) is not None            # unfiltered: AM minute prices the box
    assert first_valid_entry(box, minutes(f"WHERE {PM_SETTLED_SQL}")) is None
