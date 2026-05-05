"""Tests for utils/market_hours.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
import pytz
from utils.market_hours import is_market_open

EASTERN = pytz.timezone('America/New_York')


def _est(year, month, day, hour, minute=0):
    return EASTERN.localize(datetime(year, month, day, hour, minute))


def test_open_mid_morning_tuesday():
    assert is_market_open(_est(2024, 1, 16, 10, 30)) is True


def test_open_at_930_exactly():
    assert is_market_open(_est(2024, 1, 16, 9, 30)) is True


def test_closed_one_minute_before_open():
    assert is_market_open(_est(2024, 1, 16, 9, 29)) is False


def test_open_at_1559():
    assert is_market_open(_est(2024, 1, 16, 15, 59)) is True


def test_closed_at_1600_exactly():
    assert is_market_open(_est(2024, 1, 16, 16, 0)) is False


def test_closed_saturday():
    assert is_market_open(_est(2024, 1, 20, 12, 0)) is False


def test_closed_sunday():
    assert is_market_open(_est(2024, 1, 21, 12, 0)) is False


def test_uses_current_time_when_no_arg():
    # Just verify it doesn't crash (result depends on when test runs)
    result = is_market_open()
    assert isinstance(result, bool)
