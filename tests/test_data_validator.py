"""Tests for utils/data_validator.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest
import pandas as pd
import numpy as np
from utils.data_validator import validate_no_lookahead


@pytest.fixture
def clean_df():
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        'feat_a': np.random.randn(n),
        'feat_b': np.random.randn(n),
        'target': np.random.randn(n),
    })

def test_passes_with_uncorrelated_features(clean_df):
    # Should complete without raising
    validate_no_lookahead(clean_df, feature_cols=['feat_a', 'feat_b'], target_col='target')


def test_warns_when_feature_highly_correlated_with_target(caplog):
    np.random.seed(42)
    n = 300
    base = np.random.randn(n)
    df = pd.DataFrame({
        'leaky': base,
        'target': base + np.random.randn(n) * 0.01,  # almost identical
    })
    with caplog.at_level(logging.WARNING):
        validate_no_lookahead(df, feature_cols=['leaky'], target_col='target')
    messages = ' '.join(r.message for r in caplog.records)
    assert 'leaky' in messages or 'WARNING' in messages.upper() or '0.9' in messages


def test_raises_if_target_col_missing():
    df = pd.DataFrame({'feat': [1, 2, 3], 'other': [4, 5, 6]})
    with pytest.raises(KeyError):
        validate_no_lookahead(df, feature_cols=['feat'], target_col='missing_col')


def test_raises_if_feature_col_missing():
    df = pd.DataFrame({'feat': [1, 2, 3], 'target': [4, 5, 6]})
    with pytest.raises(KeyError):
        validate_no_lookahead(df, feature_cols=['nonexistent'], target_col='target')
