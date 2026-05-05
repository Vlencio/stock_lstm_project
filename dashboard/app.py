"""
Streamlit monitoring dashboard for the LSTM trading system.

Pages:
  1. Overview   — Key performance metrics from backtest_results.json
  2. Trades     — Trade-level details from vectorbt portfolio stats
  3. Model Performance — Price chart with indicators from processed data
  4. Logs       — Live tail of paper_trade.log with risk alerts highlighted

Run with:
    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configurable paths ─────────────────────────────────────────────────────────
BACKTEST_RESULTS_PATH = Path('logs/backtest_results.json')
PAPER_TRADE_LOG_PATH = Path('logs/paper_trade.log')
PROCESSED_DATA_DIR = Path('data/processed')
LOG_TAIL_LINES = 50
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='LSTM Trading Dashboard', layout='wide', page_icon='📈')


def load_backtest_results() -> dict | None:
    if not BACKTEST_RESULTS_PATH.exists():
        return None
    with open(BACKTEST_RESULTS_PATH) as f:
        return json.load(f)


def load_log_tail(n: int = LOG_TAIL_LINES) -> list:
    if not PAPER_TRADE_LOG_PATH.exists():
        return []
    with open(PAPER_TRADE_LOG_PATH) as f:
        lines = f.readlines()
    return lines[-n:]


def page_overview():
    st.header('Portfolio Overview')
    results = load_backtest_results()

    if results is None:
        st.warning('No backtest results found. Run scripts/backtest.py first.')
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Return', f"{results.get('total_return_pct', float('nan')):.2f}%")
    col2.metric('Sharpe Ratio', f"{results.get('sharpe_ratio', float('nan')):.3f}")
    col3.metric('Max Drawdown', f"{results.get('max_drawdown_pct', float('nan')):.2f}%")
    col4.metric('Total Trades', str(results.get('total_trades', 'N/A')))

    col5, col6, col7 = st.columns(3)
    col5.metric('Annualized Return', f"{results.get('annualized_return_pct', float('nan')):.2f}%")
    col6.metric('Win Rate', f"{results.get('win_rate_pct', float('nan')):.1f}%")
    col7.metric('Profit Factor', f"{results.get('profit_factor', float('nan')):.2f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"Period: {results.get('period_start', '?')} → {results.get('period_end', '?')}")
    with col_b:
        if BACKTEST_RESULTS_PATH.exists():
            mtime = datetime.fromtimestamp(BACKTEST_RESULTS_PATH.stat().st_mtime)
            st.caption(f"Last run: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    equity_img = Path('plots/backtest/equity_curve.png')
    if equity_img.exists():
        st.image(str(equity_img), caption='Equity Curve vs Buy & Hold')

    dd_img = Path('plots/backtest/drawdown.png')
    if dd_img.exists():
        st.image(str(dd_img), caption='Drawdown Over Time')


def page_trades():
    st.header('Trade Analysis')
    results = load_backtest_results()
    if results is None:
        st.warning('No backtest results found.')
        return

    st.subheader('Summary Statistics')
    df_stats = pd.DataFrame([{
        'Metric': k.replace('_', ' ').title(),
        'Value': v,
    } for k, v in results.items()])
    st.dataframe(df_stats, use_container_width=True)

    trade_dist_img = Path('plots/backtest/trade_returns.png')
    if trade_dist_img.exists():
        st.image(str(trade_dist_img), caption='Trade Return Distribution')
    else:
        st.info('Trade distribution plot not available. Re-run backtest to generate it.')


def page_model_performance():
    st.header('Model Prediction Performance')

    symbol_files = list(PROCESSED_DATA_DIR.glob('*.parquet'))
    if not symbol_files:
        st.warning('No processed data found. Run collect_data.py first.')
        return

    symbol = st.selectbox('Symbol', [f.stem for f in symbol_files])
    df = pd.read_parquet(PROCESSED_DATA_DIR / f'{symbol}.parquet')

    if 'Close' not in df.columns:
        st.error("'Close' column not found in processed data.")
        return

    if 'Date' in df.columns:
        df = df.set_index('Date')

    n_display = st.slider('Days to display', min_value=30, max_value=min(365, len(df)), value=90)
    df_recent = df.tail(n_display)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Close'],
                             mode='lines', name='Close Price'))
    if 'EMA_21' in df_recent.columns:
        fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['EMA_21'],
                                 mode='lines', name='EMA 21', line=dict(dash='dash')))
    if 'BB_Upper' in df_recent.columns:
        fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['BB_Upper'],
                                 mode='lines', name='BB Upper',
                                 line=dict(dash='dot', color='grey')))
        fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['BB_Lower'],
                                 mode='lines', name='BB Lower',
                                 line=dict(dash='dot', color='grey'),
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))

    fig.update_layout(title=f'{symbol} Price + Indicators',
                      xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig, use_container_width=True)

    if 'RSI_14' in df_recent.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_recent.index, y=df_recent['RSI_14'],
                                     mode='lines', name='RSI 14'))
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='red',
                          annotation_text='Overbought')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='green',
                          annotation_text='Oversold')
        fig_rsi.update_layout(title='RSI (14)', yaxis_title='RSI', yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)


def page_logs():
    st.header('Live Paper Trading Logs')

    if not PAPER_TRADE_LOG_PATH.exists():
        st.warning('No paper trading log found. Start paper_trade.py first.')
        return

    lines = load_log_tail(LOG_TAIL_LINES)
    st.caption(f"Showing last {LOG_TAIL_LINES} lines of {PAPER_TRADE_LOG_PATH}")

    log_text = ''
    for line in reversed(lines):
        line = line.strip()
        if 'CRITICAL' in line or 'ERROR' in line or 'RISK HALT' in line:
            log_text += f':red[{line}]\n\n'
        elif 'WARNING' in line:
            log_text += f':orange[{line}]\n\n'
        elif 'ORDER SUBMITTED' in line:
            log_text += f':green[{line}]\n\n'
        else:
            log_text += f'{line}\n\n'

    st.markdown(log_text)

    if st.button('Refresh logs'):
        st.rerun()


def main():
    st.title('📈 LSTM Trading System Dashboard')

    page = st.sidebar.radio(
        'Navigation',
        ['Overview', 'Trades', 'Model Performance', 'Logs'],
    )

    if page == 'Overview':
        page_overview()
    elif page == 'Trades':
        page_trades()
    elif page == 'Model Performance':
        page_model_performance()
    elif page == 'Logs':
        page_logs()


if __name__ == '__main__':
    main()
