#!/usr/bin/env python3
"""Example trading strategy using VectorBT"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import talib
from pathlib import Path

# Load data
print("Loading BTC data...")
df = pd.read_parquet("data/BTC_1m.parquet")

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
df = df.set_index('datetime')

# Convert price columns to float
for col in ['open', 'high', 'low', 'close', 'vol']:
    df[col] = df[col].astype(float)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Calculate indicators
print("\nCalculating indicators...")

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
print("\nGenerating signals...")
entries = df['rsi'] < 30
exits = df['rsi'] > 70

# Run backtest with VectorBT
print("\nRunning backtest...")
portfolio = vbt.Portfolio.from_signals(
    df['close'],
    entries,
    exits,
    init_cash=10000,
    fees=0.001,  # 0.1% trading fee
    freq='1min'
)

# Print results
print("\n=== Backtest Results ===")
print(f"Total Return: {portfolio.total_return():.2%}")
print(f"Sharpe Ratio: {portfolio.sharpe_ratio():.2f}")
print(f"Max Drawdown: {portfolio.max_drawdown():.2%}")
print(f"Win Rate: {portfolio.trades.win_rate():.2%}")
print(f"Number of Trades: {portfolio.trades.count()}")
print(f"Average Trade Duration: {portfolio.trades.duration.mean()}")
print(f"Final Value: ${portfolio.final_value():.2f}")

# Save results
print("\nSaving results...")
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
print("Results saved to results/rsi_strategy_results.csv")

# Plot (save to file)
print("\nGenerating plots...")
fig = portfolio.plot()
fig.write_html('results/portfolio_plot.html')
print("Portfolio plot saved to results/portfolio_plot.html")

print("\n=== Strategy analysis complete ===")