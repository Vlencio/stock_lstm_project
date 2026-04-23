"""Tests for utils/feature_engineering.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from utils.feature_engineering import add_technical_indicators, FEATURE_COLS

EXPECTED_INDICATOR_COLS = [
    'EMA_9', 'EMA_21', 'EMA_50',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI_14', 'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14',
    'OBV', 'Volume_SMA_20',
]

@pytest.fixture
def sample_ohlcv():
    """100 days of synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        'Open': close - 0.5,
        'High': close + 1.0,
        'Low': close - 1.0,
        'Close': close,
        'Volume': np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))


def test_adds_all_expected_columns(sample_ohlcv):
    result = add_technical_indicators(sample_ohlcv.copy())
    for col in EXPECTED_INDICATOR_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_drops_nan_rows(sample_ohlcv):
    result = add_technical_indicators(sample_ohlcv.copy())
    assert result.isnull().sum().sum() == 0, "Result contains NaN values"


def test_row_count_reduced_due_to_nan_drop(sample_ohlcv):
    result = add_technical_indicators(sample_ohlcv.copy())
    # EMA_50 needs 50 rows + Stochastic needs 14 + signal needs 9 = at least 50 rows dropped
    assert len(result) < len(sample_ohlcv)
    assert len(result) > 0


def test_ohlcv_columns_preserved(sample_ohlcv):
    result = add_technical_indicators(sample_ohlcv.copy())
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        assert col in result.columns


def test_feature_cols_constant_matches_output(sample_ohlcv):
    result = add_technical_indicators(sample_ohlcv.copy())
    for col in FEATURE_COLS:
        assert col in result.columns, f"FEATURE_COLS references missing column: {col}"


def test_requires_ohlcv_columns():
    df_missing = pd.DataFrame({'Close': [1, 2, 3]})
    with pytest.raises(KeyError):
        add_technical_indicators(df_missing)
