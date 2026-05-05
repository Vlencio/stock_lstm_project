"""
Vectorbt backtester for the LSTM trading strategy.

Loads a trained LSTM checkpoint, generates price predictions on test data,
converts them to buy/sell signals, and simulates the strategy with realistic
transaction costs. Results are saved to logs/backtest_results.json and plots
to plots/backtest/.

Usage:
    python scripts/backtest.py \\
      --checkpoint checkpoints/best_model.pth \\
      --data_dir data/processed \\
      --symbol AAPL \\
      --threshold 0.005 \\
      --capital 10000 \\
      --fee 0.001 \\
      --slippage 0.0005
"""
import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # headless rendering
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.model import StockLSTM
from scripts.dataset import create_datasets_with_scaler
from utils.feature_engineering import FEATURE_COLS, TARGET_COL
from utils.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Configurable defaults ──────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.005   # Predicted return threshold to trigger a signal (0.5%)
DEFAULT_CAPITAL = 10_000.0  # Starting capital in USD
DEFAULT_FEE = 0.001         # Brokerage fee per trade (0.1%)
DEFAULT_SLIPPAGE = 0.0005   # Price slippage per trade (0.05%)
RISK_FREE_RATE = 0.05       # Annual risk-free rate for Sharpe calculation
# ──────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load StockLSTM from checkpoint. Returns (model, model_args dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint['args']
    input_size = model_args.get('input_size', len(FEATURE_COLS))
    model = StockLSTM(
        input_size=input_size,
        hidden_size=model_args['hidden_size'],
        num_layers=model_args['num_layers'],
        dropout=model_args['dropout'],
        output_size=1,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, model_args


def generate_predictions(model, data_dict: dict, device: str) -> np.ndarray:
    """Run inference on the full dataset; return (n_samples,) array of predicted Close prices."""
    dataset = data_dict['dataset']
    scaler = data_dict['scaler']
    close_idx = FEATURE_COLS.index(TARGET_COL)

    predictions = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred_normalized = model(x).cpu().item()

            # Inverse-transform just the Close column using a dummy full-width array.
            dummy = np.zeros((1, len(FEATURE_COLS)))
            dummy[0, close_idx] = pred_normalized
            pred_price = scaler.inverse_transform(dummy)[0, close_idx]
            predictions.append(pred_price)

    return np.array(predictions)


def predictions_to_signals(predictions: np.ndarray, actual_closes: np.ndarray,
                            threshold: float) -> np.ndarray:
    """Convert predicted returns to {1, 0} buy signals."""
    predicted_returns = (predictions - actual_closes) / actual_closes
    return np.where(predicted_returns > threshold, 1, 0)


def run_vectorbt_backtest(close_prices: pd.Series, signals: np.ndarray,
                           capital: float, fee: float, slippage: float,
                           risk_manager: RiskManager) -> dict:
    """Simulate the trading strategy with vectorbt."""
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("vectorbt not installed. Run: pip install vectorbt")

    # Apply RiskManager filter: gate each signal through risk rules.
    filtered_signals = np.zeros_like(signals)
    capital_sim = capital
    peak_capital = capital
    daily_trades = 0
    last_date = None

    for i, (sig, price) in enumerate(zip(signals, close_prices)):
        current_date = close_prices.index[i].date() if hasattr(close_prices.index[i], 'date') else None
        if current_date != last_date:
            daily_trades = 0
            last_date = current_date

        if sig == 1:
            decision = risk_manager.evaluate_trade(
                signal=1,
                current_price=float(price),
                current_capital=capital_sim,
                peak_capital=peak_capital,
                daily_trades=daily_trades,
            )
            if decision['can_trade']:
                filtered_signals[i] = 1
                daily_trades += 1

        # Rough capital update for simulation.
        if i > 0 and filtered_signals[i - 1] == 1:
            ret = (float(close_prices.iloc[i]) - float(close_prices.iloc[i - 1])) / float(close_prices.iloc[i - 1])
            capital_sim *= (1 + ret * risk_manager.risk_per_trade)
            peak_capital = max(peak_capital, capital_sim)

    entries = pd.Series(filtered_signals == 1, index=close_prices.index)
    exits = pd.Series(filtered_signals != 1, index=close_prices.index)

    pf = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=exits,
        init_cash=capital,
        fees=fee,
        slippage=slippage,
        freq='D',
    )

    stats = pf.stats()

    def _get(key, fallback=float('nan')):
        try:
            return float(stats[key])
        except (KeyError, TypeError):
            return fallback

    # Compute profit factor manually.
    try:
        trade_returns = pf.trades.returns.values
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    except Exception:
        trade_returns = np.array([])
        profit_factor = float('nan')

    # Manual Sharpe ratio using daily risk-free rate.
    daily_rf = RISK_FREE_RATE / 252
    pf_returns = pf.returns()
    excess = pf_returns - daily_rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else float('nan')

    return {
        'portfolio': pf,
        'trade_returns': trade_returns,
        'stats': {
            'total_return_pct': _get('Total Return [%]'),
            'annualized_return_pct': _get('Annualized Return [%]'),
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': _get('Max Drawdown [%]'),
            'win_rate_pct': _get('Win Rate [%]'),
            'profit_factor': profit_factor,
            'total_trades': int(_get('Total Trades', 0)),
            'period_start': str(close_prices.index[0]),
            'period_end': str(close_prices.index[-1]),
        },
    }


def save_results(stats: dict, log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    out = Path(log_dir) / 'backtest_results.json'
    with open(out, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Results saved to %s", out)


def save_plots(pf, trade_returns: np.ndarray, close_prices: pd.Series, plot_dir: str) -> None:
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Equity curve vs Buy & Hold
    fig, ax = plt.subplots(figsize=(12, 5))
    pf.value().rename('Strategy').plot(ax=ax)
    bh = close_prices / close_prices.iloc[0] * pf.init_cash
    bh.rename('Buy & Hold').plot(ax=ax, linestyle='--')
    ax.set_title('Equity Curve: Strategy vs Buy & Hold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    fig.savefig(f'{plot_dir}/equity_curve.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Drawdown
    fig, ax = plt.subplots(figsize=(12, 3))
    pf.drawdown().plot(ax=ax, color='red')
    ax.set_title('Drawdown Over Time')
    ax.set_ylabel('Drawdown (%)')
    fig.savefig(f'{plot_dir}/drawdown.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Trade returns distribution
    if len(trade_returns) > 0:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(trade_returns * 100, bins=30, edgecolor='black')
            ax.set_title('Trade Return Distribution')
            ax.set_xlabel('Return per Trade (%)')
            ax.set_ylabel('Frequency')
            fig.savefig(f'{plot_dir}/trade_returns.png', dpi=120, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning("Could not save trade_returns plot: %s", e)

    logger.info("Plots saved to %s/", plot_dir)


def main():
    parser = argparse.ArgumentParser(description='Backtest LSTM trading strategy with vectorbt')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--symbol', type=str, default='AAPL')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--capital', type=float, default=DEFAULT_CAPITAL)
    parser.add_argument('--fee', type=float, default=DEFAULT_FEE)
    parser.add_argument('--slippage', type=float, default=DEFAULT_SLIPPAGE)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--plot_dir', type=str, default='plots/backtest')
    args = parser.parse_args()

    logger.info("Loading model from %s", args.checkpoint)
    model, model_args = load_model(args.checkpoint, args.device)
    window = model_args['window']

    logger.info("Loading data from %s/%s.parquet", args.data_dir, args.symbol)
    parquet_files = [f'{args.data_dir}/{args.symbol}.parquet']
    data_dict = create_datasets_with_scaler(
        parquet_files, window=window,
        feature_cols=FEATURE_COLS, target_col=TARGET_COL,
    )

    # Generate predictions across the full dataset, then isolate test indices.
    logger.info("Generating predictions...")
    all_preds = generate_predictions(model, data_dict, args.device)

    dataset = data_dict['dataset']
    scaler = data_dict['scaler']
    close_idx = FEATURE_COLS.index(TARGET_COL)
    test_indices = data_dict['test'].indices

    # Reconstruct actual Close prices for the test window.
    test_closes = []
    for idx in test_indices:
        _, y_norm = dataset[idx]
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, close_idx] = y_norm.item()
        actual_price = scaler.inverse_transform(dummy)[0, close_idx]
        test_closes.append(actual_price)

    # Reconstruct dates from the parquet index.
    df_full = pd.read_parquet(parquet_files[0])
    if not isinstance(df_full.index, pd.DatetimeIndex):
        if 'Date' in df_full.columns:
            df_full = df_full.set_index('Date')
    date_index = df_full.index[window:]
    test_date_index = date_index[test_indices]

    test_closes_series = pd.Series(test_closes, index=test_date_index, name='Close')
    test_preds = all_preds[test_indices]

    signals = predictions_to_signals(test_preds, np.array(test_closes), args.threshold)
    logger.info("Signals: %d buys, %d holds out of %d",
                signals.sum(), (signals == 0).sum(), len(signals))

    risk_manager = RiskManager()
    result = run_vectorbt_backtest(
        test_closes_series, signals, args.capital, args.fee, args.slippage, risk_manager,
    )

    print("\n=== BACKTEST RESULTS ===")
    for k, v in result['stats'].items():
        print(f"  {k}: {v}")

    save_results(result['stats'], args.log_dir)
    save_plots(result['portfolio'], result['trade_returns'], test_closes_series, args.plot_dir)


if __name__ == '__main__':
    main()
