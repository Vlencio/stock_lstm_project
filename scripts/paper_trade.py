"""
Live paper trading loop using Alpaca sandbox.

Runs every 15 minutes during NYSE market hours (09:30–16:00 EST).
On each cycle: fetches recent OHLCV, runs feature engineering + LSTM inference,
consults RiskManager, and submits orders to Alpaca paper account.

All decisions are logged to logs/paper_trade.log.

Usage:
    python scripts/paper_trade.py --symbol AAPL --checkpoint checkpoints/best_model.pth
"""
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model import StockLSTM
from utils.feature_engineering import add_technical_indicators, FEATURE_COLS, TARGET_COL
from utils.scaler import TimeSeriesScaler
from utils.risk_manager import RiskManager
from utils.market_hours import is_market_open

load_dotenv()

# ── Configurable defaults ──────────────────────────────────────────────────────
SCHEDULE_INTERVAL_MINUTES = 15   # How often to run the trading cycle
LOOKBACK_DAYS = 90                # Days of history to fetch for indicator calculation
SIGNAL_THRESHOLD = 0.005          # Predicted return threshold for a buy signal (0.5%)
# ──────────────────────────────────────────────────────────────────────────────

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trade.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _get_alpaca_api():
    """Initialize Alpaca REST client from environment variables."""
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        raise ImportError("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")

    api_key = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    base_url = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not api_key or not secret_key or api_key == 'your_key_here':
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file."
        )

    return tradeapi.REST(api_key, secret_key, base_url)


def fetch_recent_ohlcv(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Download recent OHLCV and return flat-column DataFrame."""
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    data = yf.download(symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


def load_model_and_scaler(checkpoint_path: str, scaler_path: str, device: str):
    """Load LSTM from checkpoint and scaler from file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['args']

    model = StockLSTM(
        input_size=model_args.get('input_size', len(FEATURE_COLS)),
        hidden_size=model_args['hidden_size'],
        num_layers=model_args['num_layers'],
        dropout=model_args['dropout'],
        output_size=1,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = TimeSeriesScaler(scaler_type='minmax')
    scaler.load(scaler_path)

    window = model_args['window']
    return model, scaler, window


def run_inference(model, scaler: TimeSeriesScaler, df: pd.DataFrame,
                  window: int, device: str) -> tuple:
    """
    Returns (current_close, predicted_next_close) in original price scale.
    """
    enriched = add_technical_indicators(df.copy())
    if len(enriched) < window:
        raise ValueError(f"Not enough data after feature engineering: {len(enriched)} < {window}")

    features = enriched[FEATURE_COLS].values
    scaled = scaler.transform(features)
    input_seq = scaled[-window:]   # (window, n_features)

    with torch.no_grad():
        x = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred_normalized = model(x).cpu().item()

    close_idx = FEATURE_COLS.index(TARGET_COL)
    dummy = np.zeros((1, len(FEATURE_COLS)))
    dummy[0, close_idx] = pred_normalized
    predicted_price = scaler.inverse_transform(dummy)[0, close_idx]

    current_price = float(enriched['Close'].iloc[-1])
    return current_price, predicted_price


def get_current_position(api, symbol: str) -> float:
    """Returns current quantity held for symbol, or 0 if no position."""
    try:
        pos = api.get_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def submit_order(api, symbol: str, qty: float, side: str) -> bool:
    """Submits a market order. Returns True on success."""
    try:
        api.submit_order(
            symbol=symbol,
            qty=round(qty, 0),
            side=side,
            type='market',
            time_in_force='day',
        )
        logger.info("ORDER SUBMITTED: %s %s %.2f shares", side.upper(), symbol, qty)
        return True
    except Exception as exc:
        logger.error("Order submission failed: %s", exc)
        return False


def get_account_capital(api) -> float:
    """Returns current portfolio equity."""
    try:
        account = api.get_account()
        return float(account.equity)
    except Exception as exc:
        logger.error("Could not fetch account equity: %s", exc)
        return 0.0


def trading_cycle(symbol: str, model, scaler, window: int, device: str,
                  risk_manager: RiskManager, api, peak_capital: list) -> None:
    """Single trading cycle: fetch → predict → signal → risk check → order."""
    logger.info("── Trading cycle start ── symbol=%s", symbol)

    if not is_market_open():
        logger.info("Market is closed. Skipping cycle.")
        return

    try:
        ohlcv = fetch_recent_ohlcv(symbol, LOOKBACK_DAYS)
        current_price, predicted_price = run_inference(model, scaler, ohlcv, window, device)

        predicted_return = (predicted_price - current_price) / current_price
        signal = 1 if predicted_return > SIGNAL_THRESHOLD else 0

        current_capital = get_account_capital(api)
        peak_capital[0] = max(peak_capital[0], current_capital)
        current_qty = get_current_position(api, symbol)

        logger.info(
            "current=%.2f predicted=%.2f return=%.4f signal=%d capital=%.2f",
            current_price, predicted_price, predicted_return, signal, current_capital,
        )

        decision = risk_manager.evaluate_trade(
            signal=signal,
            current_price=current_price,
            current_capital=current_capital,
            peak_capital=peak_capital[0],
            daily_trades=0,
        )

        logger.info("RiskManager: can_trade=%s reason=%s", decision['can_trade'], decision['reason'])

        if decision['can_trade'] and current_qty == 0:
            submit_order(api, symbol, decision['position_size'], 'buy')
        elif not decision['can_trade'] and current_qty > 0:
            submit_order(api, symbol, current_qty, 'sell')
        else:
            logger.info("No order action required.")

    except Exception as exc:
        logger.error("Cycle error: %s", exc, exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='Alpaca paper trading with LSTM signals')
    parser.add_argument('--symbol', type=str, default='AAPL')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--scaler_path', type=str, default='checkpoints/scaler.pkl')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    logger.info("Initializing paper trading for %s", args.symbol)

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        raise ImportError("apscheduler not installed. Run: pip install apscheduler")

    api = _get_alpaca_api()
    model, scaler, window = load_model_and_scaler(args.checkpoint, args.scaler_path, args.device)
    risk_manager = RiskManager()
    peak_capital = [get_account_capital(api)]

    logger.info("Model loaded. Window=%d. Starting scheduler every %d min.",
                window, SCHEDULE_INTERVAL_MINUTES)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        trading_cycle,
        'cron',
        day_of_week='mon-fri',
        hour='9-15',
        minute=f'*/{SCHEDULE_INTERVAL_MINUTES}',
        args=[args.symbol, model, scaler, window, args.device, risk_manager, api, peak_capital],
    )
    scheduler.start()


if __name__ == '__main__':
    main()
