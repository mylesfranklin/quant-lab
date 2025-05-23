#!/usr/bin/env python3
"""Example trading strategy using VectorBT - cleaned version without prints"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import talib
from pathlib import Path

# Load data
df = pd.read_parquet("data/BTC_1m.parquet")

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
df = df.set_index('datetime')

# Convert price columns to float
for col in ['open', 'high', 'low', 'close', 'vol']:
    df[col] = df[col].astype(float)

# Calculate indicators
# RSI
df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

# Bollinger Bands
df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
    df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
)

# Moving averages
df['sma_20'] = talib.SMA(df['close'].values, timeperiod=20)
df['sma_50'] = talib.SMA(df['close'].values, timeperiod=50)

# Simple RSI strategy
# Buy when RSI < 30 (oversold)
# Sell when RSI > 70 (overbought)
entries = df['rsi'] < 30
exits = df['rsi'] > 70

# Run backtest with VectorBT
portfolio = vbt.Portfolio.from_signals(
    df['close'],
    entries,
    exits,
    init_cash=10000,
    fees=0.001,  # 0.1% trading fee
    freq='1min'
)

# Save results
results = {
    'total_return': portfolio.total_return(),
    'sharpe_ratio': portfolio.sharpe_ratio(),
    'max_drawdown': portfolio.max_drawdown(),
    'win_rate': portfolio.trades.win_rate(),
    'num_trades': portfolio.trades.count(),
    'final_value': portfolio.final_value()
}

results_df = pd.DataFrame([results])
results_df.to_csv('results/rsi_strategy_results.csv', index=False)

# Plot (save to file)
fig = portfolio.plot()
fig.write_html('results/portfolio_plot.html')