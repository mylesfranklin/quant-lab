#!/usr/bin/env python
"""
Quick demonstration of factor caching performance
"""
import time
from rich import print
from data_manager import DataManager
from backtester_cached import CachedBacktester
from factor_store import get_factor_store

def main():
    print("[bold cyan]Factor Caching Demo[/bold cyan]\n")
    
    # Initialize
    bt = CachedBacktester()
    
    # Load data
    print("Loading BTC data...")
    data = bt.load_data('BTC', '1m')
    print(f"✓ Loaded {len(data)} data points\n")
    
    # Define a reasonable parameter grid (30 combinations)
    param_grid = {
        'fast_period': [10, 20, 30],
        'slow_period': [50, 100, 150],
        'ma_type': ['SMA', 'EMA']  # Limited to what talib supports
    }
    
    total_combos = 3 * 3 * 2  # 18 combinations
    print(f"Testing {total_combos} parameter combinations\n")
    
    # Clear cache to start fresh
    get_factor_store().invalidate_factors('BTC', '1m')
    
    # Test 1: First run (cold cache)
    print("[yellow]Run 1: Cold cache (computing all indicators)[/yellow]")
    start = time.time()
    results1 = bt.optimize_strategy(
        'ma_crossover_cached', 
        data,
        param_grid=param_grid,
        warm_cache=False
    )
    time1 = time.time() - start
    print(f"Time: {time1:.2f}s\n")
    
    # Test 2: Second run (warm cache)
    print("[yellow]Run 2: Warm cache (using cached indicators)[/yellow]")
    start = time.time()
    results2 = bt.optimize_strategy(
        'ma_crossover_cached',
        data, 
        param_grid=param_grid,
        warm_cache=False  # Don't re-warm, use existing cache
    )
    time2 = time.time() - start
    print(f"Time: {time2:.2f}s\n")
    
    # Show improvement
    speedup = time1 / time2 if time2 > 0 else 0
    print(f"[green]Performance Improvement:[/green]")
    print(f"  First run: {time1:.2f}s")
    print(f"  Cached run: {time2:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved: {time1 - time2:.2f}s ({(1-time2/time1)*100:.0f}% faster)\n")
    
    # Show cache stats
    stats = get_factor_store().get_cache_stats()
    print(f"[cyan]Cache Statistics:[/cyan]")
    print(f"  Hit rate: {stats['hit_rate']*100:.0f}%")
    print(f"  Factors cached: {stats['total_factors']}")
    print(f"  Memory used: {stats['memory_usage_mb']:.1f} MB")
    
    # Verify results are identical
    if len(results1) > 0 and len(results2) > 0:
        best1 = results1.iloc[0]
        best2 = results2.iloc[0]
        
        returns_match = abs(best1['return_pct'] - best2['return_pct']) < 1e-6
        if returns_match:
            print(f"\n[green]✓ Results match exactly - cache is working correctly![/green]")
        else:
            print(f"\n[red]✗ Results don't match - cache issue![/red]")
    
    # Clean up
    bt.dm.close()
    get_factor_store().close()

if __name__ == "__main__":
    main()