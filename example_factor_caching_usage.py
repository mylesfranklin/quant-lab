#!/usr/bin/env python
"""
Complete example of using the factor caching system
Shows how to create a cached strategy and achieve 10x performance
"""
from rich import print
from strategy_framework_cached import CachedBaseStrategy, StrategyConfig
import cached_indicators as ci
from backtester_cached import CachedBacktester
import pandas as pd

# 1. Define a new cached strategy
class MyTradingStrategy(CachedBaseStrategy):
    """Example strategy using multiple cached indicators"""
    
    config = StrategyConfig(
        name="My Cached Strategy",
        version="1.0",
        description="Multi-indicator strategy with factor caching"
    )
    
    # Define parameter grid for optimization
    param_grid = {
        'rsi_period': [14, 21, 28],
        'ema_short': [10, 20],
        'ema_long': [50, 100],
        'bb_period': [20],
        'bb_std': [2.0, 2.5]
    }
    
    # Define which indicators to pre-cache (optional but recommended)
    cache_indicators = {
        'rsi': [14, 21, 28],
        'ema': [10, 20, 50, 100],
        'bb': [(20, 2.0), (20, 2.5)]
    }
    
    def calculate_indicators_cached(self, data):
        """Calculate all indicators using cache"""
        # These are computed only once and cached!
        return {
            'rsi': ci.rsi(period=self.rsi_period),
            'ema_short': ci.ema(period=self.ema_short),
            'ema_long': ci.ema(period=self.ema_long),
            'bb': ci.bollinger_bands(period=self.bb_period, std_dev=self.bb_std)
        }
    
    def generate_signals(self, data, indicators):
        """Generate trading signals"""
        # Entry conditions:
        # - RSI oversold (<30)
        # - Price below lower Bollinger Band
        # - Short EMA above long EMA (uptrend)
        entries = (
            (indicators['rsi'] < 30) &
            (data.close < indicators['bb']['lower']) &
            (indicators['ema_short'] > indicators['ema_long'])
        )
        
        # Exit conditions:
        # - RSI overbought (>70)
        # - Price above upper Bollinger Band
        exits = (
            (indicators['rsi'] > 70) |
            (data.close > indicators['bb']['upper'])
        )
        
        return entries, exits

# 2. Register the strategy (optional)
from strategy_framework_cached import CACHED_STRATEGY_REGISTRY
CACHED_STRATEGY_REGISTRY['my_strategy'] = MyTradingStrategy

# 3. Run backtesting with caching
def main():
    print("[bold cyan]Factor Caching Example[/bold cyan]\n")
    
    # Initialize backtester
    bt = CachedBacktester()
    
    # Load data
    print("Loading data...")
    data = bt.load_data('BTC', '1m')
    print(f"✓ Loaded {len(data)} data points\n")
    
    # Warm cache (optional but recommended)
    print("Warming cache...")
    bt.warm_cache('my_strategy', data)
    
    # Run optimization
    print("\nRunning optimization with factor caching...")
    results = bt.optimize_strategy(
        'my_strategy',
        data,
        n_jobs=1  # Sequential for best cache performance
    )
    
    # Show results
    if len(results) > 0:
        print(f"\n[green]Optimization complete![/green]")
        print(f"Total combinations tested: {len(results)}")
        
        # Best result
        best = results.iloc[0]
        print(f"\nBest parameters:")
        print(f"  RSI Period: {best['rsi_period']}")
        print(f"  EMA Short: {best['ema_short']}")
        print(f"  EMA Long: {best['ema_long']}")
        print(f"  BB Period: {best['bb_period']}")
        print(f"  BB Std: {best['bb_std']}")
        print(f"\nPerformance:")
        print(f"  Return: {best['return_pct']:.2f}%")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  Max DD: {best['max_dd']:.2f}%")
        print(f"  Trades: {best['num_trades']}")
        
        # Show cache performance
        cache_report = bt.get_cache_report()
        print(f"\n[cyan]Cache Performance:[/cyan]")
        print(f"  Hit rate: {cache_report['overall_hit_rate']:.0f}%")
        print(f"  Factors cached: {cache_report['cache_stats']['total_factors']}")
        print(f"  Average compute time saved: ~{cache_report['cache_stats']['avg_compute_time_ms']:.0f}ms per indicator")
    
    # Clean up
    bt.dm.close()

if __name__ == "__main__":
    main()