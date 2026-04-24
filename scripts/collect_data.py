"""
Downloads OHLCV data via yfinance for a given symbol, enriches it with
technical indicators, and saves to data/processed/<SYMBOL>.parquet.

Raw data (unchanged format) is still saved to data/raw/ for backward
compatibility with existing scripts (evaluate.py, inference.py).

Usage:
    python scripts/collect_data.py --symbol AAPL --start 2018-01-01 --end 2024-01-01
"""
import argparse
import sys
import os
from pathlib import Path

import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.feature_engineering import add_technical_indicators, FEATURE_COLS

# ── Configurable defaults ──────────────────────────────────────────────────────
DEFAULT_INTERVAL = '1d'       # Daily candles
RAW_DIR = 'data/raw'          # Kept for backward compat; split by year as before
PROCESSED_DIR = 'data/processed'  # Single parquet per symbol with all features
# ──────────────────────────────────────────────────────────────────────────────


def download_stock_data(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Download OHLCV from yfinance and return a clean, flat-column DataFrame."""
    data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True)

    # yfinance 0.2.x returns MultiIndex columns for a single ticker.
    # Flatten to standard names: Close, High, Low, Open, Volume.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Keep only the 5 OHLCV columns we need.
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    data.index.name = 'Date'
    return data


def save_raw_by_year(data: pd.DataFrame, out_dir: str) -> None:
    """Preserve the original raw split-by-year format."""
    years = data.index.year
    for year in years.unique():
        mask = years == year
        file_path = os.path.join(out_dir, f'{year}.parquet')
        data[mask].to_parquet(file_path)


def save_processed(data: pd.DataFrame, symbol: str, out_dir: str) -> None:
    """Apply feature engineering and save single enriched parquet."""
    enriched = add_technical_indicators(data.copy())
    file_path = os.path.join(out_dir, f'{symbol}.parquet')
    enriched.to_parquet(file_path, index=True)
    print(f"Saved processed data ({len(enriched)} rows, {len(enriched.columns)} cols) → {file_path}")
    print(f"Feature columns: {FEATURE_COLS}")


def main():
    parser = argparse.ArgumentParser(description='Download stock data and run feature engineering.')
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--interval', type=str, default=DEFAULT_INTERVAL)
    parser.add_argument('--raw_dir', type=str, default=RAW_DIR)
    parser.add_argument('--processed_dir', type=str, default=PROCESSED_DIR)
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    print(f"Downloading {args.symbol} from {args.start} to {args.end}...")
    data = download_stock_data(args.symbol, args.start, args.end, args.interval)
    print(f"Downloaded {len(data)} rows.")

    save_raw_by_year(data, args.raw_dir)
    print(f"Raw data saved to {args.raw_dir}/")

    save_processed(data, args.symbol, args.processed_dir)


if __name__ == '__main__':
    main()
