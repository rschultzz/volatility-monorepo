"""CR-BO shared definitions — universe calendar, event dates, roll weeks, ES window grid.

Imported by scripts/cr_bo_step0.py and scripts/cr_bo_extract.py so Step 0 and the run use one
definition of every exclusion (specs/CR-BO-session-premium.md, L1/L2/L6).

No DB access here.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

from packages.shared.trading_calendar import NYSE_HOLIDAYS, is_trading_day, next_trading_day, nth_trading_day

TRAIN_START = date(2023, 5, 1)       # first orats_monies_minute date (CR-BL Step 0)
TRAIN_END = date(2026, 6, 5)         # t+1 ≤ TRAIN_END (holdout never read)

# NYSE early closes in range (13:00 ET); trading days per trading_calendar, excluded by L1.
EARLY_CLOSES = frozenset({
    date(2023, 7, 3), date(2023, 11, 24), date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
})

# ---------------------------------------------------------------- L6(c) event calendar
# Source schedules (fetched 2026-10-18):
#   CPI   https://www.bls.gov/bls/news-release/cpi.htm      (archive; dates from release file names)
#   PPI   https://www.bls.gov/bls/news-release/ppi.htm
#   NFP   https://www.bls.gov/bls/news-release/empsit.htm   (Employment Situation)
#   PCE   BEA Personal Income and Outlays:
#         2023 https://apps.bea.gov/scb/issues/2022/12-december/1222-news-releases-2023.htm
#         2024 https://apps.bea.gov/scb/issues/2023/12-december/1223-news-releases-2024.htm
#         2025 https://www.bea.gov/news/schedule/full-2025
#         2026 https://bea.gov/index.php/news/blog/2026-01-15/economic-release-schedule-updates-gdp-personal-income-and-outlays
#              (Jan, Feb 2026 reports), https://www.bea.gov/news/2026/personal-income-and-outlays-december-2025
#              (Dec 2025 report, 2026-02-20), https://www.bea.gov/index.php/news/schedule/full (Mar–May 2026)
#   FOMC  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (statement on the last day)
# BLS releases are 08:30 ET. BEA PCE is 08:30 ET except the ones listed in PCE_NOT_0830.
# October 2025 CPI / NFP / PPI were not published (2025 appropriations lapse).

def _d(s: str) -> date:
    return date.fromisoformat(s)


CPI = [_d(x) for x in """
2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11
2024-10-10 2024-11-13 2024-12-11 2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11
2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18 2026-01-13 2026-02-13 2026-03-11 2026-04-10
2026-05-12""".split()]

PPI = [_d(x) for x in """
2023-05-11 2023-06-14 2023-07-13 2023-08-11 2023-09-14 2023-10-11 2023-11-15 2023-12-13
2024-01-12 2024-02-16 2024-03-14 2024-04-11 2024-05-14 2024-06-13 2024-07-12 2024-08-13 2024-09-12
2024-10-11 2024-11-14 2024-12-12 2025-01-14 2025-02-13 2025-03-13 2025-04-11 2025-05-15 2025-06-12
2025-07-16 2025-08-14 2025-09-10 2025-11-25 2026-01-14 2026-01-30 2026-02-27 2026-03-18 2026-04-14
2026-05-13""".split()]

NFP = [_d(x) for x in """
2023-05-05 2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06
2024-10-04 2024-11-01 2024-12-06 2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06
2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16 2026-01-09 2026-02-11 2026-03-06 2026-04-03
2026-05-08 2026-06-05""".split()]

PCE_ALL = [_d(x) for x in """
2023-05-26 2023-06-30 2023-07-28 2023-08-31 2023-09-29 2023-10-27 2023-11-30 2023-12-22
2024-01-26 2024-02-29 2024-03-29 2024-04-26 2024-05-31 2024-06-28 2024-07-26 2024-08-30 2024-09-27
2024-10-31 2024-11-27 2024-12-20 2025-01-31 2025-02-28 2025-03-28 2025-04-30 2025-05-30 2025-06-27
2025-07-31 2025-08-29 2025-09-26 2025-12-05 2026-01-22 2026-02-20 2026-03-13 2026-04-09 2026-04-30
2026-05-28""".split()]
# Released at 10:00 ET, not 08:30 — not an L6 event night on their own.
PCE_NOT_0830 = frozenset({_d("2024-11-27"), _d("2025-04-30"), _d("2025-12-05"), _d("2026-01-22")})
PCE = [d for d in PCE_ALL if d not in PCE_NOT_0830]

FOMC = [_d(x) for x in """
2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29""".split()]
# 2025-08-22 unscheduled notation vote: no scheduled statement → not flagged.

EVENT_0830 = {"CPI": frozenset(CPI), "PPI": frozenset(PPI), "NFP": frozenset(NFP), "PCE": frozenset(PCE)}
FOMC_DAYS = frozenset(FOMC)


def event_tags(d: date) -> list[str]:
    return [k for k, s in EVENT_0830.items() if d in s]


# ---------------------------------------------------------------- L1 roll weeks
def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d: date) -> bool:
    """ES quarterly roll week, the CR-BM definition: Monday of the week before the third Friday
    through expiry Friday, Mar/Jun/Sep/Dec. ES.c.0 (Databento, 2023–2025) and the Ironbeam
    front-month ingest (2026) both carry the dying contract through it."""
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


# ---------------------------------------------------------------- sessions and pairs
def train_sessions() -> list[date]:
    out, d = [], TRAIN_START
    while d <= TRAIN_END:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def session_pairs() -> list[tuple[date, date]]:
    s = train_sessions()
    return [(a, b) for a, b in zip(s, s[1:]) if b == next_trading_day(a)]


def spans_holiday(t: date, t1: date) -> bool:
    d = t + timedelta(days=1)
    while d < t1:
        if d in NYSE_HOLIDAYS:
            return True
        d += timedelta(days=1)
    return False


def bucket_expiries(t: date) -> dict[str, object]:
    """L2: S1 = expiry on the next session, S2 = two sessions out; S7 is chosen from the listed
    expiries (nearest ≥ 7 calendar days) by the caller."""
    return {"S1": next_trading_day(t), "S2": nth_trading_day(t, 2)}


# ---------------------------------------------------------------- ES Globex minute grid (ET)
def globex_open(m: datetime) -> bool:
    """Standard CME equity Globex minute (bar-start, ET): Sun 18:00 → Fri 17:00, daily halt
    17:00–17:59. Holiday schedules are not encoded; see expected_minutes()."""
    wd, hm = m.weekday(), m.time()
    if time(17, 0) <= hm < time(18, 0):
        return False
    if wd == 5:
        return False
    if wd == 4 and hm >= time(17, 0):
        return False
    if wd == 6 and hm < time(18, 0):
        return False
    return True


def window_minutes(start: datetime, end: datetime, skip_dates: frozenset[date] = frozenset()) -> list[datetime]:
    """Expected bar-start minutes in [start, end) on the standard grid, dropping every minute whose
    ET calendar date is in skip_dates (NYSE holidays inside a holiday-spanning night: CME runs a
    holiday schedule there that the grid does not model)."""
    out, m = [], start
    while m < end:
        if globex_open(m) and m.date() not in skip_dates:
            out.append(m)
        m += timedelta(minutes=1)
    return out


def overnight_bounds(t: date, t1: date) -> tuple[datetime, datetime]:
    return datetime.combine(t, time(16, 0)), datetime.combine(t1, time(9, 30))


def rth_bounds(t1: date) -> tuple[datetime, datetime]:
    return datetime.combine(t1, time(9, 30)), datetime.combine(t1, time(16, 0))


def holiday_dates_between(t: date, t1: date) -> frozenset[date]:
    out, d = set(), t + timedelta(days=1)
    while d < t1:
        if d in NYSE_HOLIDAYS:
            out.add(d)
        d += timedelta(days=1)
    return frozenset(out)


YEAR_SECONDS = 365.0 * 86400.0   # L3 day-count: calendar time / 365 (Step 0 Q3)


def tau_years(snapshot_et: datetime, expir: date) -> float:
    exp = datetime.combine(expir, time(16, 0))
    return (exp - snapshot_et).total_seconds() / YEAR_SECONDS
