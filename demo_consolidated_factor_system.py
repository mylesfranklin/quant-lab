"""
Comprehensive demo of the consolidated factor caching system.
Shows how strategies can easily use factor_info and related helpers.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.backtesting.factor_store_enhanced import factor_info, find_factors
from src.backtesting.cached_indicators import ma, rsi, bb
from src.backtesting.factor_utils import FactorCache, print_factor_report


def main():
    print("=== QuantLab Factor Caching System Demo ===\n")
    
    # 1. Create sample data
    print("1. Creating sample data...")
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
    
    # 2. Compute some indicators (will be cached)
    print("\n2. Computing indicators (with automatic caching)...")
    ma20 = ma(data, period=20)
    ma50 = ma(data, period=50)
    rsi14 = rsi(data, period=14)
    bb20 = bb(data, period=20, std=2.0)
    
    print("✓ Computed MA(20), MA(50), RSI(14), BB(20,2)")
    
    # 3. Use factor_info to check what was cached
    print("\n3. Checking cached factors with factor_info()...")
    
    # Check MA(20)
    ma20_id = f"ma_BTC/USDT_1h_{{'period': 20}}"
    ma20_info = factor_info(ma20_id)
    
    if ma20_info:
        print(f"\nMA(20) Factor Info:")
        print(f"  - Factor ID: {ma20_id}")
        print(f"  - Rows: {ma20_info['rows']}")
        print(f"  - Data Type: {ma20_info['dtype']}")
        print(f"  - Is Partial: {ma20_info['is_partial']}")
        print(f"  - Created: {ma20_info['created_at']}")
        print(f"  - Fingerprint: {ma20_info['data_fingerprint'][:16]}...")
    
    # 4. Search for factors
    print("\n4. Searching for cached factors...")
    
    all_ma_factors = find_factors('ma', 'BTC/USDT', '1h')
    all_rsi_factors = find_factors('rsi', 'BTC/USDT', '1h')
    
    print(f"\nFound {len(all_ma_factors)} MA factors for BTC/USDT 1h:")
    for factor_id in all_ma_factors:
        print(f"  - {factor_id}")
    
    print(f"\nFound {len(all_rsi_factors)} RSI factors for BTC/USDT 1h:")
    for factor_id in all_rsi_factors:
        print(f"  - {factor_id}")
    
    # 5. Use FactorCache utilities
    print("\n5. Using FactorCache utilities...")
    
    # Get summary
    summary = FactorCache.get_factor_summary('BTC/USDT', '1h')
    print(f"\nFactor Summary for BTC/USDT 1h:")
    print(f"  - Total Factors: {summary['total_factors']}")
    print(f"  - By Indicator: {summary['by_indicator']}")
    print(f"  - Memory Cached: {summary['memory_cached']}")
    print(f"  - Disk Cached: {summary['disk_cached']}")
    
    # Get cache stats
    stats = FactorCache.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  - Memory Cache Size: {stats['memory_cache_size']}")
    print(f"  - DB Total Factors: {stats['db_total_factors']}")
    print(f"  - DB Total Rows: {stats['db_total_rows']:,}")
    
    # 6. Demonstrate cache reuse
    print("\n6. Demonstrating cache reuse...")
    print("Computing MA(20) again (should use cache)...")
    
    import time
    start = time.time()
    ma20_cached = ma(data, period=20)
    cache_time = time.time() - start
    
    print(f"✓ Retrieved from cache in {cache_time*1000:.1f}ms")
    print(f"✓ Values match: {np.array_equal(ma20.values, ma20_cached.values)}")
    
    # 7. Show how strategies use this
    print("\n7. Strategy Usage Example:")
    print("-" * 50)
    
    code_example = '''
# In your strategy file:
from src.backtesting.factor_store_enhanced import factor_info
from src.backtesting.cached_indicators import ma, rsi

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Compute indicators (automatically cached)
        ma20 = ma(data, period=20)
        rsi14 = rsi(data, period=14)
        
        # Check factor info if needed
        factor_id = f"ma_{data.symbol}_{data.timeframe}_{{'period': 20}}"
        info = factor_info(factor_id)
        
        if info and info.get('is_partial'):
            self.logger.warning("MA has partial data!")
        
        # Generate signals...
        return signals
'''
    print(code_example)
    
    # 8. Print full report
    print("\n8. Full Factor Report:")
    print_factor_report('BTC/USDT', '1h')
    
    print("\n✅ Demo completed successfully!")
    print("\nKey Takeaways:")
    print("- Use factor_info() to get detailed information about any cached factor")
    print("- Use find_factors() to search for cached factors")
    print("- Use FactorCache utilities for high-level operations")
    print("- All caching happens automatically when using cached_indicators")
    print("- The system handles edge cases like partial data and type normalization")


if __name__ == "__main__":
    main()