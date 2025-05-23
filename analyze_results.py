#!/usr/bin/env python
"""Analyze batch backtest results"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load results
results_file = Path("results/metrics.json")
if not results_file.exists():
    print("No results file found. Run batch_backtest.py first.")
    exit(1)

with open(results_file) as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Filter out failed backtests
df = df[df['num_trades'] > 0]

print("=== BATCH BACKTEST ANALYSIS ===")
print(f"\nTotal parameter combinations tested: {len(results)}")
print(f"Successful backtests: {len(df)}")
print(f"Failed backtests: {len(results) - len(df)}")

# Group by strategy
print("\n=== STRATEGY PERFORMANCE ===")
for strategy in df['strategy'].unique():
    strat_df = df[df['strategy'] == strategy]
    print(f"\n{strategy}:")
    print(f"  Combinations tested: {len(strat_df)}")
    print(f"  Best return: {strat_df['return_pct'].max():.2f}%")
    print(f"  Average return: {strat_df['return_pct'].mean():.2f}%")
    print(f"  Best Sharpe: {strat_df['sharpe'].max():.2f}")
    print(f"  Average trades: {strat_df['num_trades'].mean():.1f}")

# Top 10 results
print("\n=== TOP 10 RESULTS BY RETURN ===")
top_10 = df.nlargest(10, 'return_pct')
for idx, row in top_10.iterrows():
    params = {k: v for k, v in row.items() 
              if k not in ['return_pct', 'sharpe', 'max_dd', 'win_rate', 
                          'num_trades', 'avg_trade_duration', 'profit_factor', 
                          'final_value', 'strategy', 'timestamp']}
    param_str = ', '.join(f'{k}={v}' for k, v in params.items())
    print(f"{row['strategy']}: {row['return_pct']:.2f}% (Sharpe: {row['sharpe']:.2f}) - {param_str}")

# Best Sharpe ratios
print("\n=== TOP 10 RESULTS BY SHARPE RATIO ===")
top_sharpe = df.nlargest(10, 'sharpe')
for idx, row in top_sharpe.iterrows():
    params = {k: v for k, v in row.items() 
              if k not in ['return_pct', 'sharpe', 'max_dd', 'win_rate', 
                          'num_trades', 'avg_trade_duration', 'profit_factor', 
                          'final_value', 'strategy', 'timestamp']}
    param_str = ', '.join(f'{k}={v}' for k, v in params.items())
    print(f"{row['strategy']}: Sharpe {row['sharpe']:.2f} (Return: {row['return_pct']:.2f}%) - {param_str}")

# Most active strategies (by trade count)
print("\n=== MOST ACTIVE STRATEGIES ===")
most_active = df.nlargest(10, 'num_trades')
for idx, row in most_active.iterrows():
    params = {k: v for k, v in row.items() 
              if k not in ['return_pct', 'sharpe', 'max_dd', 'win_rate', 
                          'num_trades', 'avg_trade_duration', 'profit_factor', 
                          'final_value', 'strategy', 'timestamp']}
    param_str = ', '.join(f'{k}={v}' for k, v in params.items())
    print(f"{row['strategy']}: {row['num_trades']} trades (Return: {row['return_pct']:.2f}%) - {param_str}")

# Parameter analysis
print("\n=== PARAMETER INSIGHTS ===")

# RSI strategy analysis
if 'strategy_rsi_bands' in df['strategy'].values:
    rsi_df = df[df['strategy'] == 'strategy_rsi_bands']
    
    # Best RSI parameters
    best_rsi = rsi_df.loc[rsi_df['return_pct'].idxmax()]
    print(f"\nBest RSI+Bands parameters:")
    print(f"  RSI Period: {best_rsi['rsi_period']}")
    print(f"  RSI Lower: {best_rsi['rsi_lower']}")
    print(f"  RSI Upper: {best_rsi['rsi_upper']}")
    print(f"  BB Period: {best_rsi['bb_period']}")
    print(f"  BB Std: {best_rsi['bb_std']}")

# MA strategy analysis
if 'strategy_ma_cross' in df['strategy'].values:
    ma_df = df[df['strategy'] == 'strategy_ma_cross']
    
    # Best MA parameters
    best_ma = ma_df.loc[ma_df['return_pct'].idxmax()]
    print(f"\nBest MA Crossover parameters:")
    print(f"  Fast Period: {best_ma['fast_period']}")
    print(f"  Slow Period: {best_ma['slow_period']}")
    print(f"  MA Type: {best_ma['ma_type']}")
    print(f"  Volume Multiplier: {best_ma['volume_mult']}")
    print(f"  Volume Period: {best_ma['volume_period']}")
    
    # MA type comparison
    print(f"\nMA Type Comparison:")
    for ma_type in ['SMA', 'EMA']:
        type_df = ma_df[ma_df['ma_type'] == ma_type]
        if len(type_df) > 0:
            print(f"  {ma_type}: Avg Return {type_df['return_pct'].mean():.2f}%, Best {type_df['return_pct'].max():.2f}%")

# Save best parameters
best_params = {}
for strategy in df['strategy'].unique():
    strat_df = df[df['strategy'] == strategy]
    best_row = strat_df.loc[strat_df['return_pct'].idxmax()]
    
    params = {k: v for k, v in best_row.items() 
              if k not in ['return_pct', 'sharpe', 'max_dd', 'win_rate', 
                          'num_trades', 'avg_trade_duration', 'profit_factor', 
                          'final_value', 'strategy', 'timestamp']}
    
    best_params[strategy] = {
        'parameters': params,
        'return_pct': float(best_row['return_pct']),
        'sharpe': float(best_row['sharpe']),
        'num_trades': int(best_row['num_trades'])
    }

# Save best parameters
with open('results/best_parameters.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("\n\nBest parameters saved to results/best_parameters.json")