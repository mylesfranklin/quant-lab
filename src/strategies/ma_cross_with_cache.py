"""
Simple MA crossover strategy that uses the consolidated factor info helpers.
Shows the minimal code needed to leverage factor caching.
"""

import pandas as pd
from src.strategies.base_strategy import BaseStrategy
from src.backtesting.cached_indicators import ma
from src.backtesting.factor_utils import factor_info, FactorCache


class MACrossoverCached(BaseStrategy):
    """MA crossover strategy with factor cache awareness."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals using cached MAs."""
        
        # Calculate MAs (automatically cached)
        fast_ma = ma(data, period=self.fast_period)
        slow_ma = ma(data, period=self.slow_period)
        
        # Get factor info if needed
        symbol = getattr(data, 'symbol', 'BTC/USDT')
        timeframe = getattr(data, 'timeframe', '1h')
        
        # Check fast MA factor
        fast_factor_id = f"ma_{symbol}_{timeframe}_{{'period': {self.fast_period}}}"
        fast_info = factor_info(fast_factor_id)
        
        if fast_info and fast_info.get('is_partial'):
            print(f"WARNING: Fast MA has partial data!")
        
        # Generate crossover signals
        signals = pd.Series(0, index=data.index)
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1
        
        # Only trade on crossovers
        return signals.diff().fillna(0)
    
    def before_backtest(self, data: pd.DataFrame) -> None:
        """Clean up old factors."""
        # Clean factors older than 3 days
        removed = FactorCache.cleanup(days=3)
        if removed > 0:
            print(f"Cleaned up {removed} old factors")


# Example usage
if __name__ == "__main__":
    # Quick demo
    strategy = MACrossoverCached(fast_period=10, slow_period=20)
    print("Strategy created with factor caching support!")