# Backtest Results Analysis

## Summary
- Total combinations tested: 48
- Combinations with trades: 29
- Combinations with no trades: 19

## Strategy Performance
| Strategy | Combinations | With Trades | Best Return | Avg Return | Best Sharpe | Avg Trades |
|----------|--------------|-------------|-------------|------------|-------------|------------|
| ma_cross_vectorized | 20 | 19 | 3.60 | 0.30 | 11.24 | 27.8 |
| bollinger_obv_breakout | 6 | 6 | -0.93 | -2.53 | -6.25 | 65.7 |
| macd_rsi_trend_follower | 4 | 4 | -0.10 | -0.74 | -1.48 | 4.8 |

## Top 10 Results by Return
| Strategy | Return % | Sharpe | Trades | Parameters |
|----------|----------|--------|--------|------------|
| ma_cross_vectorized | 3.60 | 11.24 | 16 | fast_ma_len=20.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 2.33 | 7.40 | 25 | fast_ma_len=10.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 2.26 | 7.18 | 15 | fast_ma_len=30.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 1.74 | 5.56 | 24 | fast_ma_len=10.0, slow_ma_len=200.0 |
| ma_cross_vectorized | 1.69 | 5.39 | 16 | fast_ma_len=40.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 1.65 | 5.46 | 33 | fast_ma_len=10.0, slow_ma_len=100.0 |
| ma_cross_vectorized | 1.57 | 5.00 | 17 | fast_ma_len=20.0, slow_ma_len=200.0 |
| ma_cross_vectorized | 1.57 | 5.21 | 26 | fast_ma_len=30.0, slow_ma_len=100.0 |
| ma_cross_vectorized | 1.30 | 4.10 | 14 | fast_ma_len=40.0, slow_ma_len=200.0 |
| ma_cross_vectorized | 1.22 | 3.86 | 11 | fast_ma_len=50.0, slow_ma_len=200.0 |

## Top 10 Results by Sharpe Ratio
| Strategy | Sharpe | Return % | Max DD % | Parameters |
|----------|--------|----------|----------|------------|
| ma_cross_vectorized | 11.24 | 3.60 | 2.43 | fast_ma_len=20.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 7.40 | 2.33 | 2.78 | fast_ma_len=10.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 7.18 | 2.26 | 2.33 | fast_ma_len=30.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 5.56 | 1.74 | 3.18 | fast_ma_len=10.0, slow_ma_len=200.0 |
| ma_cross_vectorized | 5.46 | 1.65 | 3.09 | fast_ma_len=10.0, slow_ma_len=100.0 |
| ma_cross_vectorized | 5.39 | 1.69 | 2.46 | fast_ma_len=40.0, slow_ma_len=150.0 |
| ma_cross_vectorized | 5.21 | 1.57 | 2.36 | fast_ma_len=30.0, slow_ma_len=100.0 |
| ma_cross_vectorized | 5.00 | 1.57 | 2.75 | fast_ma_len=20.0, slow_ma_len=200.0 |
| ma_cross_vectorized | 4.10 | 1.30 | 2.65 | fast_ma_len=40.0, slow_ma_len=200.0 |
| ma_cross_vectorized | 3.86 | 1.22 | 3.39 | fast_ma_len=50.0, slow_ma_len=200.0 |

## Best Overall Result
- **Strategy**: ma_cross_vectorized
- **Return**: 3.60%
- **Sharpe Ratio**: 11.24
- **Max Drawdown**: 2.43%
- **Win Rate**: 50.0%
- **Number of Trades**: 16
- **Parameters**: fast_ma_len=20.0, slow_ma_len=150.0