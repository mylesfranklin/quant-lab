"""
Example strategy demonstrating the consolidated factor info helpers.
Shows how to use factor_info, find_factors, and validate_factor in a real strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.strategies.base_strategy import BaseStrategy
from src.backtesting.factor_store_enhanced import (
    factor_info, find_factors, validate_factor, invalidate_stale_factors
)
from src.backtesting.cached_indicators import ma, rsi, bb, macd


class FactorInfoStrategy(BaseStrategy):
    """Strategy that demonstrates factor info usage."""
    
    def __init__(self, ma_period: int = 20, rsi_period: int = 14):
        super().__init__()
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals with factor info awareness."""
        
        # Get symbol and timeframe from data attributes (assumed to be set)
        symbol = getattr(data, 'symbol', 'BTC/USDT')
        timeframe = getattr(data, 'timeframe', '1h')
        
        # Check what factors are already cached for this symbol
        cached_mas = find_factors('ma', symbol, timeframe)
        cached_rsis = find_factors('rsi', symbol, timeframe)
        
        print(f"\nFound {len(cached_mas)} cached MA factors for {symbol} {timeframe}")
        print(f"Found {len(cached_rsis)} cached RSI factors for {symbol} {timeframe}")
        
        # Calculate indicators with caching
        ma_values = ma(data, period=self.ma_period)
        rsi_values = rsi(data, period=self.rsi_period)
        
        # Get info about the factors we just used
        ma_factor_id = f"ma_{symbol}_{timeframe}_{{'period': {self.ma_period}}}"
        rsi_factor_id = f"rsi_{symbol}_{timeframe}_{{'period': {self.rsi_period}}}"
        
        ma_info = factor_info(ma_factor_id)
        rsi_info = factor_info(rsi_factor_id)
        
        if ma_info:
            print(f"\nMA Factor Info:")
            print(f"  - Rows: {ma_info['rows']}")
            print(f"  - Dtype: {ma_info['dtype']}")
            print(f"  - Is Partial: {ma_info['is_partial']}")
            print(f"  - Created: {ma_info['created_at']}")
            
        if rsi_info:
            print(f"\nRSI Factor Info:")
            print(f"  - Rows: {rsi_info['rows']}")
            print(f"  - Dtype: {rsi_info['dtype']}")
            print(f"  - Is Partial: {rsi_info['is_partial']}")
            print(f"  - Created: {rsi_info['created_at']}")
        
        # Validate factors are still valid
        if ma_info and not validate_factor(ma_factor_id, data):
            print(f"WARNING: MA factor {ma_factor_id} is stale!")
            
        if rsi_info and not validate_factor(rsi_factor_id, data):
            print(f"WARNING: RSI factor {rsi_factor_id} is stale!")
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy when price crosses above MA and RSI < 30
        buy_condition = (
            (data['close'] > ma_values) & 
            (data['close'].shift(1) <= ma_values.shift(1)) &
            (rsi_values < 30)
        )
        
        # Sell when price crosses below MA or RSI > 70
        sell_condition = (
            (data['close'] < ma_values) & 
            (data['close'].shift(1) >= ma_values.shift(1)) |
            (rsi_values > 70)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def before_backtest(self, data: pd.DataFrame) -> None:
        """Clean up stale factors before running backtest."""
        # Remove factors older than 7 days
        removed = invalidate_stale_factors(max_age_days=7)
        if removed > 0:
            print(f"Cleaned up {removed} stale factors")
            
    def after_backtest(self, results: Dict[str, Any]) -> None:
        """Report on factor cache usage after backtest."""
        symbol = results.get('symbol', 'BTC/USDT')
        timeframe = results.get('timeframe', '1h')
        
        # Find all factors used
        all_factors = []
        for indicator in ['ma', 'rsi', 'bb', 'macd']:
            all_factors.extend(find_factors(indicator, symbol, timeframe))
            
        print(f"\nTotal cached factors for {symbol} {timeframe}: {len(all_factors)}")
        
        # Check for partial factors
        partial_count = 0
        for factor_id in all_factors:
            info = factor_info(factor_id)
            if info and info.get('is_partial'):
                partial_count += 1
                
        if partial_count > 0:
            print(f"WARNING: {partial_count} factors have partial data!")


def demonstrate_factor_helpers():
    """Demonstrate the factor info helpers in action."""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'open': 50000 + np.random.randn(1000).cumsum() * 100,
        'high': 50100 + np.random.randn(1000).cumsum() * 100,
        'low': 49900 + np.random.randn(1000).cumsum() * 100,
        'close': 50000 + np.random.randn(1000).cumsum() * 100,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Add metadata
    data.symbol = 'BTC/USDT'
    data.timeframe = '1h'
    
    # Create and run strategy
    strategy = FactorInfoStrategy(ma_period=20, rsi_period=14)
    
    print("=== Running Strategy with Factor Info ===")
    strategy.before_backtest(data)
    signals = strategy.generate_signals(data)
    
    # Simulate results
    results = {
        'symbol': data.symbol,
        'timeframe': data.timeframe,
        'total_trades': signals.abs().sum(),
        'final_value': 10000 * (1 + np.random.randn() * 0.1)
    }
    
    strategy.after_backtest(results)
    
    print(f"\nStrategy completed with {results['total_trades']} trades")
    print(f"Final portfolio value: ${results['final_value']:.2f}")
    
    # Show how to check specific factors
    print("\n=== Checking Specific Factors ===")
    
    # Look for all MA factors
    ma_factors = find_factors('ma', 'BTC/USDT', '1h')
    print(f"\nAll MA factors for BTC/USDT 1h:")
    for factor_id in ma_factors[:3]:  # Show first 3
        info = factor_info(factor_id)
        if info:
            print(f"  - {factor_id}: {info['rows']} rows, dtype={info['dtype']}")
    
    # Validate a specific factor
    if ma_factors:
        factor_id = ma_factors[0]
        is_valid = validate_factor(factor_id, data)
        print(f"\nFactor {factor_id} is {'valid' if is_valid else 'INVALID'}")


if __name__ == "__main__":
    demonstrate_factor_helpers()