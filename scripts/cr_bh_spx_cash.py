"""CR-BH SPX-cash series builder — moved to packages/shared/spx_cash.py (CR-BI). Re-export for the CR-BH scripts."""
from packages.shared.spx_cash import *  # noqa: F401,F403
from packages.shared.spx_cash import (  # noqa: F401
    CUTOVER, LAGGED_WINDOWS, build_series, daily_ohlc, fetch_minutes, is_lagged,
    load_spx_daily_bars, normalize_minutes, series_segment,
)
