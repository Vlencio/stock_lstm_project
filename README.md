# Stock LSTM — Quantitative Trading System

An end-to-end quantitative trading system built with PyTorch. Collects historical OHLCV data, engineers 20 technical features, trains an LSTM model to predict next-close prices, backtests the strategy with vectorbt, manages risk with fixed-fractional position sizing, and executes paper trades via Alpaca — all monitored through a Streamlit dashboard.

---

## Features

- **Feature Engineering** — 20 technical indicators (EMA 9/21/50, MACD, RSI, Stochastic, Bollinger Bands, ATR, OBV, Volume SMA)
- **Leak-free training** — scaler fitted only on training data; temporal splits (70/15/15), never shuffled
- **Backtesting** — vectorbt-powered simulation with equity curve, drawdown, and trade return plots
- **Risk Management** — fixed-fractional position sizing, max drawdown circuit breaker, daily trade limit
- **Paper Trading** — Alpaca API integration, APScheduler cron (every 15 min during market hours)
- **Dashboard** — Streamlit 4-page app: Overview, Trades, Model Performance, Logs
- **Data Validation** — correlation-based lookahead leak detector

---

## Project Structure

```
stock_lstm_project/
├── scripts/
│   ├── collect_data.py     # Download OHLCV + compute indicators → data/processed/
│   ├── dataset.py          # StockDataset, temporal splits, scaler wiring
│   ├── model.py            # StockLSTM (configurable layers/hidden size)
│   ├── train.py            # Training loop with checkpointing and logging
│   ├── backtest.py         # vectorbt backtest + RiskManager signal gate
│   └── paper_trade.py      # Live paper trading via Alpaca + APScheduler
├── utils/
│   ├── feature_engineering.py  # FEATURE_COLS, TARGET_COL, add_technical_indicators()
│   ├── scaler.py               # TimeSeriesScaler (MinMax / Standard)
│   ├── risk_manager.py         # RiskManager — position sizing + circuit breakers
│   ├── market_hours.py         # NYSE hours guard (America/New_York)
│   ├── data_validator.py       # Lookahead leak correlation check
│   ├── logger.py               # TrainingLogger
│   ├── reporter.py             # TrainingReporter
│   ├── visualizer.py           # Loss / prediction plots
│   └── progress.py             # Colored progress bars
├── dashboard/
│   └── app.py              # Streamlit monitoring dashboard
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_data_validator.py
│   ├── test_risk_manager.py
│   └── test_market_hours.py
├── data/
│   ├── raw/                # Per-year parquet files from yfinance
│   └── processed/          # Enriched parquet (DatetimeIndex, 20 feature cols)
├── checkpoints/            # Model checkpoints + scaler (.pkl)
├── logs/                   # Training logs + backtest_results.json
└── plots/                  # Training loss curve + backtest charts
```

---

## Installation

```bash
git clone https://github.com/Vlencio/stock_lstm_project.git
cd stock_lstm_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Collect & engineer data

```bash
python scripts/collect_data.py --symbol AAPL --start 2018-01-01 --end 2024-01-01
```

Saves `data/processed/AAPL.parquet` with a DatetimeIndex and 20 feature columns.

### 2. Train the model

```bash
python -m scripts.train --symbol AAPL --epochs 50 --hidden_size 128 --num_layers 2
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--symbol` | `AAPL` | Ticker (must match a file in `--data_dir`) |
| `--epochs` | `50` | Training epochs |
| `--hidden_size` | `128` | LSTM hidden units |
| `--num_layers` | `2` | LSTM layers |
| `--dropout` | `0.2` | Dropout rate |
| `--lr` | `0.001` | Learning rate |
| `--scaler_type` | `minmax` | `minmax` or `standard` |

Saves `checkpoints/best_model.pth` and `checkpoints/scaler.pkl`.

### 3. Backtest

```bash
python scripts/backtest.py --checkpoint checkpoints/best_model.pth
```

Outputs:
- `logs/backtest_results.json` — return, Sharpe, drawdown, win rate
- `plots/backtest/equity_curve.png`
- `plots/backtest/drawdown.png`
- `plots/backtest/trade_returns.png`

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
```

Opens a 4-page Streamlit app: Overview, Trades, Model Performance, Logs.

### 5. Paper trading (optional)

Create a `.env` file in the project root:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Then run:

```bash
python scripts/paper_trade.py --checkpoint checkpoints/best_model.pth --symbol AAPL
```

The scheduler fires every 15 minutes on weekdays between 09:30–16:00 Eastern. Trades are gated by `RiskManager` before submission.

---

## Tests

```bash
pytest tests/ -v
```

25 tests across feature engineering, data validation, risk management, and market hours.

---

## Architecture Notes

- `FEATURE_COLS` in `utils/feature_engineering.py` is the single source of truth for feature names and order — imported by `collect_data`, `dataset`, and `train`.
- The scaler is fit **only on training data** (first 70% of rows) to prevent data leakage. Val and test sets are transformed with the same fitted scaler.
- Inverse-transforming a single predicted value requires a full-width dummy array (`np.zeros((1, 20))`), since the MinMaxScaler was fit on all 20 features.
- `args.input_size` is stored inside each checkpoint so the model can be reconstructed at inference time without re-specifying flags.
