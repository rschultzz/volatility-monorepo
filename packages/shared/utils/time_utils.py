import pandas as pd
from datetime import datetime
import pytz

def is_market_hours(dt: datetime = None) -> bool:
    """
    Checks if the given datetime is within the US stock market hours (9:30 AM to 4:00 PM ET).

    Args:
        dt: The datetime to check. If None, the current time is used.

    Returns:
        True if the datetime is within market hours, False otherwise.
    """
    if dt is None:
        dt = datetime.now(pytz.utc)

    eastern = pytz.timezone('US/Eastern')
    dt_et = dt.astimezone(eastern)

    # Market hours are 9:30 AM to 4:00 PM ET
    market_open = dt_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if it's a weekday and within market hours
    return market_open.time() <= dt_et.time() <= market_close.time() and dt_et.weekday() < 5
