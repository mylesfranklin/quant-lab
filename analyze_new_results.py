#!/usr/bin/env python
"""Analyze new batch backtest results"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load results
results_file = Path("results/new_strategies_metrics.json")
with open(results_file) as f:
    results = json.load(f)

# Convert to DataFrame and handle Infinity/NaN
df = pd.DataFrame(results)
df = df.replace([np.inf, -np.inf], np.nan)

# Filter to only strategies that made trades
df_with_trades = df[df['num_trades'] > 0]

print("=== NEW STRATEGIES BATCH BACKTEST ANALYSIS ===")
print(f"\nTotal parameter combinations tested: {len(results)}")
print(f"Combinations with trades: {len(df_with_trades)}")
print(f"Combinations with no trades: {len(df) - len(df_with_trades)}")

# Group by strategy
strategies = df['strategy'].unique()
for strategy in strategies:
    strat_df = df[df['strategy'] == strategy]
    strat_with_trades = strat_df[strat_df['num_trades'] > 0]
    
    print(f"\n{strategy}:")
    print(f"  Total combinations: {len(strat_df)}")
    print(f"  Combinations with trades: {len(strat_with_trades)}")
    
    if len(strat_with_trades) > 0:
        print(f"  Best return: {strat_with_trades['return_pct'].max():.2f}%")
        print(f"  Worst return: {strat_with_trades['return_pct'].min():.2f}%")
        print(f"  Average return: {strat_with_trades['return_pct'].mean():.2f}%")
        print(f"  Average trades: {strat_with_trades['num_trades'].mean():.1f}")
        print(f"  Best win rate: {strat_with_trades['win_rate'].max():.1f}%")

# Strategy details
print("\n=== STRATEGY DETAILS ===")

print("\n1. EMA ATR Mean Reversion:")
ema_df = df[df['strategy'] == 'ema_atr_mean_reversion']
if len(ema_df[ema_df['num_trades'] > 0]) == 0:
    print("   No trades generated - parameters may be too conservative")
    print(f"   Tested parameters: fast_len={ema_df['fast_len'].unique()}, slow_len={ema_df['slow_len'].unique()}")

print("\n2. Bollinger OBV Breakout:")
bb_df = df_with_trades[df_with_trades['strategy'] == 'bollinger_obv_breakout']
if len(bb_df) > 0:
    best_bb = bb_df.loc[bb_df['return_pct'].idxmax()]
    print(f"   Best parameters: bb_window={best_bb['bb_window']}, std_mul={best_bb['std_mul']}")
    print(f"   Best return: {best_bb['return_pct']:.2f}%")
    print(f"   Win rate: {best_bb['win_rate']:.1f}%")
    print(f"   Trades: {best_bb['num_trades']}")

print("\n3. MACD RSI Trend Follower:")
macd_df = df_with_trades[df_with_trades['strategy'] == 'macd_rsi_trend_follower']
if len(macd_df) > 0:
    best_macd = macd_df.loc[macd_df['return_pct'].idxmax()]
    print(f"   Best parameters: fast={best_macd['macd_fast']}, slow={best_macd['macd_slow']}, signal={best_macd['macd_signal']}")
    print(f"   Best return: {best_macd['return_pct']:.2f}%")
    print(f"   Win rate: {best_macd['win_rate']:.1f}%")
    print(f"   Trades: {best_macd['num_trades']}")

# Overall performance
print("\n=== OVERALL PERFORMANCE ===")
if len(df_with_trades) > 0:
    print(f"Best overall return: {df_with_trades['return_pct'].max():.2f}%")
    print(f"Average return (strategies with trades): {df_with_trades['return_pct'].mean():.2f}%")
    print(f"Total trades across all strategies: {df_with_trades['num_trades'].sum()}")
    
    # Best performing combination
    best_idx = df_with_trades['return_pct'].idxmax()
    best = df_with_trades.loc[best_idx]
    print(f"\nBest performing combination:")
    print(f"  Strategy: {best['strategy']}")
    print(f"  Return: {best['return_pct']:.2f}%")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  Max DD: {best['max_dd']:.2f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Trades: {best['num_trades']}")
    
    # Extract parameters
    param_cols = [col for col in best.index if col not in ['return_pct', 'sharpe', 'max_dd', 
                                                            'win_rate', 'num_trades', 'avg_trade_duration',
                                                            'profit_factor', 'final_value', 'strategy', 'timestamp']]
    print("  Parameters:")
    for col in param_cols:
        if pd.notna(best[col]):
            print(f"    {col}: {best[col]}")

print("\n=== RECOMMENDATIONS ===")
print("1. EMA ATR Mean Reversion generated no trades - consider:")
print("   - Reducing ATR multiplier for wider bands")
print("   - Using shorter EMA periods")
print("   - Testing on more volatile data")
print("\n2. Bollinger OBV Breakout shows promise but negative returns:")
print("   - May work better in trending markets")
print("   - Consider adding trend filters")
print("\n3. MACD RSI Trend Follower has low trade count:")
print("   - RSI threshold of 30 may be too conservative")
print("   - Consider higher RSI entry levels (35-40)")
print("\n4. General observations:")
print("   - All strategies show negative returns on this data")
print("   - The 1-minute timeframe may be too noisy")
print("   - Consider testing on higher timeframes (5m, 15m, 1h)")