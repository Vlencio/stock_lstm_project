"""
Enriches OHLCV DataFrames with technical indicators for LSTM feature input.

Constants:
    FEATURE_COLS: Ordered list of all columns the model expects as input features.
                  Keeping this constant here is the single source of truth for both
                  feature_engineering and dataset/train modules.
"""
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Single source of truth: ordered feature columns fed to the LSTM.
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_9', 'EMA_21', 'EMA_50',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI_14', 'Stoch_K', 'Stoch_D',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR_14',
    'OBV', 'Volume_SMA_20',
]

TARGET_COL = 'Close'


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 15 technical indicator columns to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume.

    Returns:
        Enriched DataFrame with NaN rows dropped.
    """
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise KeyError(f"Required OHLCV column missing: {col}")

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Trend
    df['EMA_9'] = EMAIndicator(close=close, window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(close=close, window=21).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=close, window=50).ema_indicator()

    macd_ind = MACD(close=close)
    df['MACD'] = macd_ind.macd()
    df['MACD_Signal'] = macd_ind.macd_signal()
    df['MACD_Hist'] = macd_ind.macd_diff()

    # Momentum
    df['RSI_14'] = RSIIndicator(close=close, window=14).rsi()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    # Volatility
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    df['ATR_14'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # Volume
    df['OBV'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()

    df = df.dropna().reset_index(drop=False)
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    return df
