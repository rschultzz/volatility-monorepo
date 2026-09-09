"""CR-BD decision 1 / G0 — the EOD job stamps onto the next NYSE trading day and no
weekend-only helper survives in the cron path."""
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for p in (str(_ROOT), str(_ROOT / "apps" / "cron")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:1/test")    # db.get_conn is never called at import

import job_orats_eod as eod        # noqa: E402  (apps/cron module; logs a START line on import)


def test_store_date_is_next_trading_day_not_next_weekday():
    assert eod.next_trading_day(date(2026, 9, 4)) == date(2026, 9, 8)     # Labor Day skipped
    assert eod.next_trading_day(date(2026, 6, 18)) == date(2026, 6, 22)   # Juneteenth skipped
    assert eod.next_trading_day(date(2026, 7, 2)) == date(2026, 7, 6)     # July 4 observed skipped
    assert eod.next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27) # Thanksgiving skipped
    assert eod.next_trading_day(date(2026, 12, 24)) == date(2026, 12, 28) # Christmas skipped


def test_weekend_only_helper_is_gone_from_the_cron():
    assert not hasattr(eod, "next_business_day")
    src = Path(eod.__file__).read_text()
    assert "next_business_day(" not in src
    assert "next_trading_day(api_trade_date)" in src


def test_lookback_probe_starts_from_the_previous_trading_day():
    """previous_business_day_with_data must never probe a closed day (A2)."""
    probed = []
    class _Sess: pass
    eod.has_data_for_date = lambda session, token, ticker, d: (probed.append(d) or d == date(2026, 9, 4))
    import datetime as dt
    class _FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 8, 6, 1, tzinfo=tz)
    real = eod.dt.datetime
    eod.dt.datetime = _FakeDT
    try:
        assert eod.previous_business_day_with_data(_Sess(), "tok", "SPX") == date(2026, 9, 4)
    finally:
        eod.dt.datetime = real
    assert probed == [date(2026, 9, 4)]                                    # not 09-07 (holiday), not 09-06/05 (weekend)
