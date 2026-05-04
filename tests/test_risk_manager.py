"""Tests for utils/risk_manager.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from utils.risk_manager import RiskManager


@pytest.fixture
def rm():
    return RiskManager(
        risk_per_trade=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        max_drawdown_pct=0.15,
        max_daily_trades=3,
    )


def test_normal_buy_signal_allowed(rm):
    result = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=10_000.0, peak_capital=10_000.0, daily_trades=0,
    )
    assert result['can_trade'] is True
    assert result['position_size'] == pytest.approx(10_000.0 * 0.02 / 100.0, rel=0.01)
    assert result['stop_loss'] == pytest.approx(98.0, rel=0.01)
    assert result['take_profit'] == pytest.approx(104.0, rel=0.01)


def test_neutral_signal_blocked(rm):
    result = rm.evaluate_trade(
        signal=0, current_price=100.0,
        current_capital=10_000.0, peak_capital=10_000.0, daily_trades=0,
    )
    assert result['can_trade'] is False
    assert 'neutral' in result['reason'].lower() or 'signal' in result['reason'].lower()


def test_max_drawdown_exceeded_blocks_trade(rm):
    # 16% drawdown exceeds 15% limit
    result = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=8_400.0, peak_capital=10_000.0, daily_trades=0,
    )
    assert result['can_trade'] is False
    assert 'drawdown' in result['reason'].lower()


def test_drawdown_at_limit_does_not_block(rm):
    # Exactly at limit: 14.9% drawdown — should still allow
    result = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=8_510.0, peak_capital=10_000.0, daily_trades=0,
    )
    assert result['can_trade'] is True


def test_max_daily_trades_blocks(rm):
    result = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=10_000.0, peak_capital=10_000.0, daily_trades=3,
    )
    assert result['can_trade'] is False
    assert 'daily' in result['reason'].lower()


def test_position_size_scales_with_capital(rm):
    result_low = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=5_000.0, peak_capital=5_000.0, daily_trades=0,
    )
    result_high = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=10_000.0, peak_capital=10_000.0, daily_trades=0,
    )
    assert result_high['position_size'] == pytest.approx(result_low['position_size'] * 2, rel=0.01)


def test_result_has_all_required_keys(rm):
    result = rm.evaluate_trade(
        signal=1, current_price=100.0,
        current_capital=10_000.0, peak_capital=10_000.0, daily_trades=0,
    )
    for key in ['can_trade', 'position_size', 'stop_loss', 'take_profit', 'reason']:
        assert key in result
