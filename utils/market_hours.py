"""
NYSE market hours guard.

is_market_open() returns True only when the US equity market is open
(weekdays, 09:30–15:59 Eastern Time). Does not account for market holidays
(a complete holiday calendar requires a dedicated library like `trading_calendars`).
"""
from datetime import datetime, time
import pytz

EASTERN = pytz.timezone('America/New_York')
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)   # exclusive: 15:59:59 is open, 16:00 is not


def is_market_open(dt=None) -> bool:
    """
    Returns True if the given datetime falls within NYSE trading hours.

    Args:
        dt: Timezone-aware datetime to check. Defaults to now.

    Returns:
        True if Monday–Friday, 09:30–15:59:59 Eastern Time.
    """
    if dt is None:
        dt = datetime.now(tz=EASTERN)
    elif dt.tzinfo is None:
        dt = EASTERN.localize(dt)
    else:
        dt = dt.astimezone(EASTERN)

    if dt.weekday() >= 5:   # Saturday=5, Sunday=6
        return False

    t = dt.time()
    return MARKET_OPEN <= t < MARKET_CLOSE
