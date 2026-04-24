"""
Dataset utilities for LSTM stock prediction.

StockDataset wraps a list of parquet files into a PyTorch Dataset.
Each sample is (window_of_features, next_close_price), both in normalized scale.
The scaler is fit ONLY on training data to prevent data leakage.
"""
import torch
from torch.utils.data import Dataset, Subset
import polars as pl
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.scaler import TimeSeriesScaler


class StockDataset(Dataset):
    def __init__(self, data_source, window=30, feature_cols=None,
                 target_col='Close', scaler=None, fit_scaler=True):
        """
        Args:
            data_source: List of parquet file paths OR a numpy array (n_rows, n_features).
            window: Number of past timesteps fed to the LSTM.
            feature_cols: Column names to use as features (must include target_col).
            target_col: Column to predict (must be in feature_cols).
            scaler: Pre-fitted TimeSeriesScaler. If None, creates and fits one.
            fit_scaler: Whether to fit the scaler on this data (set False for val/test).
        """
        if isinstance(data_source, (list, tuple)):
            df = pl.concat([pl.read_parquet(f) for f in sorted(data_source)])
            if feature_cols is None:
                feature_cols = ['Close']
            raw_features = df.select(feature_cols).to_numpy()
            self.target_idx = feature_cols.index(target_col)
        else:
            # Accept numpy array directly (used internally by create_datasets_with_scaler).
            # Caller must set dataset.target_idx after construction if target is not col 0.
            raw_features = data_source
            self.target_idx = 0

        self.window = window
        # Store feature_cols even in the numpy path so callers can introspect column names.
        self.feature_cols = feature_cols if feature_cols is not None else []

        if scaler is None:
            self.scaler = TimeSeriesScaler(scaler_type='minmax', feature_range=(0, 1))
            if fit_scaler:
                self.scaler.fit(raw_features)
        else:
            self.scaler = scaler

        self.features = self.scaler.transform(raw_features)

    def __len__(self):
        return len(self.features) - self.window

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.window]            # (window, n_features)
        y = self.features[idx + self.window, self.target_idx]  # scalar: next Close
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(float(y), dtype=torch.float32),
        )

    def get_scaler(self):
        return self.scaler

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def create_time_series_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    """Temporal train/val/test split — never shuffled."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    return {
        'train': Subset(dataset, list(range(0, train_size))),
        'val': Subset(dataset, list(range(train_size, train_size + val_size))),
        'test': Subset(dataset, list(range(train_size + val_size, total_size))),
    }


def create_datasets_with_scaler(
    parquet_files,
    window=30,
    train_ratio=0.7,
    val_ratio=0.15,
    feature_cols=None,
    target_col='Close',
    scaler_type='minmax',
):
    """
    Creates train/val/test datasets with scaler fitted only on training data.

    Args:
        parquet_files: List of parquet file paths (loaded and concatenated).
        window: Lookback window size.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        feature_cols: Feature column names. Defaults to ['Close'].
        target_col: Column to predict (scalar output). Must be in feature_cols.
        scaler_type: 'minmax' or 'standard'.

    Returns:
        Dict with keys: train, val, test (Subset objects), scaler, dataset.
    """
    if feature_cols is None:
        feature_cols = ['Close']

    all_data = pl.concat([pl.read_parquet(f) for f in sorted(parquet_files)])
    raw_features = all_data.select(feature_cols).to_numpy()
    target_idx = feature_cols.index(target_col)

    # Fit scaler ONLY on training data to prevent data leakage.
    train_size = int(len(raw_features) * train_ratio)
    scaler = TimeSeriesScaler(scaler_type=scaler_type, feature_range=(0, 1))
    scaler.fit(raw_features[:train_size])

    # Build dataset using pre-fitted scaler (no re-fitting).
    dataset = StockDataset(
        data_source=raw_features,
        window=window,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=scaler,
        fit_scaler=False,
    )
    dataset.target_idx = target_idx

    splits = create_time_series_splits(dataset, train_ratio, val_ratio)

    return {
        'train': splits['train'],
        'val': splits['val'],
        'test': splits['test'],
        'scaler': scaler,
        'dataset': dataset,
    }
