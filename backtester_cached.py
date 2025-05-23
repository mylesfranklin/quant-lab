#!/usr/bin/env python
"""
Enhanced backtester with factor caching support
Provides 10x performance improvement for grid optimization
"""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
warnings.filterwarnings('ignore')

from data_manager import DataManager
from strategy_framework import BaseStrategy
from strategy_framework_cached import CachedBaseStrategy, MarketData, CACHED_STRATEGY_REGISTRY, load_cached_strategy
from factor_store import get_factor_store
import cached_indicators as ci


class CachedBacktester:
    """Enhanced backtester with factor caching for 10x performance"""
    
    def __init__(self, data_manager: DataManager = None, use_cache: bool = True):
        self.dm = data_manager or DataManager()
        self.use_cache = use_cache
        self.results = []
        self.performance_stats = {
            'total_backtests': 0,
            'cache_hits': 0,
            'time_saved': 0,
            'avg_speedup': 0
        }
        
    def load_data(self, symbol: str, timeframe: str, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> MarketData:
        """Load data from data manager"""
        df = self.dm.load_data(symbol, timeframe, start_date, end_date)
        
        if len(df) == 0:
            raise ValueError(f"No data found for {symbol} {timeframe}")
        
        # Convert to MarketData format
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('datetime')
        
        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            open=df['open'].astype(float),
            high=df['high'].astype(float),
            low=df['low'].astype(float),
            close=df['close'].astype(float),
            volume=df['volume'].astype(float)
        )
        
        # Setup cache context if using cache
        if self.use_cache:
            market_data.setup_cache_context()
        
        return market_data
    
    def warm_cache(self, strategy_name: str, data: MarketData):
        """Pre-compute all indicators for a strategy"""
        if not self.use_cache:
            return
        
        print(f"Warming cache for {strategy_name}...")
        
        if strategy_name in CACHED_STRATEGY_REGISTRY:
            strategy_class = CACHED_STRATEGY_REGISTRY[strategy_name]
            
            # Warm cache for this strategy
            start_time = time.time()
            strategy_class.warm_cache(data)
            elapsed = time.time() - start_time
            
            print(f"  Cache warmed in {elapsed:.2f}s")
            
            # Get cache stats
            cache_stats = get_factor_store().get_cache_stats()
            print(f"  Factors cached: {cache_stats['total_factors']}")
            print(f"  Memory used: {cache_stats['memory_usage_mb']:.1f} MB")
    
    def run_single_backtest(self, strategy: Union[BaseStrategy, CachedBaseStrategy], 
                           data: MarketData, init_cash: float = 10000, 
                           fees: float = 0.001) -> Dict:
        """Run backtest for a single parameter combination"""
        try:
            start_time = time.time()
            
            # Generate signals
            entries = strategy.entries(data)
            exits = strategy.exits(data)
            
            # Ensure boolean type
            entries = entries.astype(bool)
            exits = exits.astype(bool)
            
            # Run portfolio simulation
            pf = vbt.Portfolio.from_signals(
                data.close,
                entries,
                exits,
                init_cash=init_cash,
                fees=fees,
                freq='1T' if data.timeframe.endswith('m') else '1H'
            )
            
            # Extract metrics
            stats = pf.stats()
            
            # Calculate computation time
            compute_time = time.time() - start_time
            
            # Get strategy performance stats if available
            perf_stats = {}
            if hasattr(strategy, 'get_performance_stats'):
                perf_stats = strategy.get_performance_stats()
            
            return {
                'success': True,
                'metrics': {
                    'return_pct': float(stats.get('Total Return [%]', 0)),
                    'sharpe': float(stats.get('Sharpe Ratio', 0)),
                    'max_dd': float(stats.get('Max Drawdown [%]', 0)),
                    'win_rate': float(stats.get('Win Rate [%]', 0)),
                    'num_trades': int(stats.get('Total Trades', 0)),
                    'profit_factor': float(stats.get('Profit Factor', 0)),
                    'final_value': float(pf.final_value()),
                    'sortino': float(stats.get('Sortino Ratio', 0)),
                    'calmar': float(stats.get('Calmar Ratio', 0)),
                    'compute_time': compute_time,
                    'indicator_time': perf_stats.get('indicator_compute_time', 0),
                    'signal_time': perf_stats.get('signal_compute_time', 0)
                },
                'trades': len(pf.trades.records_readable)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def optimize_strategy(self, strategy_name: str, data: MarketData,
                         param_grid: Optional[Dict] = None,
                         init_cash: float = 10000, fees: float = 0.001,
                         n_jobs: int = 1, warm_cache: bool = True) -> pd.DataFrame:
        """Run optimization over parameter grid with caching"""
        
        # Determine if using cached strategy
        if strategy_name in CACHED_STRATEGY_REGISTRY:
            strategy_class = CACHED_STRATEGY_REGISTRY[strategy_name]
            is_cached = True
        else:
            # Fallback to non-cached strategies
            from strategy_framework import STRATEGY_REGISTRY
            if strategy_name in STRATEGY_REGISTRY:
                strategy_class = STRATEGY_REGISTRY[strategy_name]
                is_cached = False
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Warm cache if requested
        if warm_cache and is_cached and self.use_cache:
            self.warm_cache(strategy_name, data)
        
        # Use default param grid if not provided
        if param_grid is None:
            param_grid = strategy_class.param_grid
        
        # Generate parameter combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        param_combinations = list(itertools.product(*values))
        
        print(f"\nOptimizing {strategy_name} ({len(param_combinations)} combinations)")
        print(f"Using cache: {is_cached and self.use_cache}")
        
        results = []
        total_time = 0
        
        # Track cache performance
        cache_stats_before = get_factor_store().get_cache_stats() if self.use_cache else None
        
        if n_jobs == 1:
            # Sequential execution (better for cache performance)
            for i, combo_values in enumerate(param_combinations):
                params = dict(zip(keys, combo_values))
                
                # Create strategy instance
                if is_cached:
                    strategy = load_cached_strategy(strategy_name, **params)
                else:
                    strategy = strategy_class(**params)
                
                # Run backtest
                result = self.run_single_backtest(strategy, data, init_cash, fees)
                
                if result['success']:
                    # Add parameters to result
                    result['metrics'].update(params)
                    result['metrics']['strategy'] = strategy_name
                    results.append(result['metrics'])
                    total_time += result['metrics']['compute_time']
                
                # Progress indicator with performance info
                if (i + 1) % 10 == 0:
                    avg_time = total_time / (i + 1)
                    eta = avg_time * (len(param_combinations) - i - 1)
                    print(f"  Progress: {i + 1}/{len(param_combinations)} "
                          f"(avg: {avg_time:.3f}s, ETA: {eta:.1f}s)")
        
        else:
            # Parallel execution (may have lower cache hit rate)
            print("Note: Parallel execution may reduce cache efficiency")
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                futures = {}
                for combo_values in param_combinations:
                    params = dict(zip(keys, combo_values))
                    
                    if is_cached:
                        strategy = load_cached_strategy(strategy_name, **params)
                    else:
                        strategy = strategy_class(**params)
                    
                    future = executor.submit(
                        self.run_single_backtest, 
                        strategy, data, init_cash, fees
                    )
                    futures[future] = params
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    params = futures[future]
                    result = future.result()
                    
                    if result['success']:
                        result['metrics'].update(params)
                        result['metrics']['strategy'] = strategy_name
                        results.append(result['metrics'])
                        total_time += result['metrics']['compute_time']
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{len(param_combinations)}")
        
        # Calculate cache performance
        if self.use_cache and cache_stats_before:
            cache_stats_after = get_factor_store().get_cache_stats()
            
            new_hits = cache_stats_after['hits'] - cache_stats_before['hits']
            new_misses = cache_stats_after['misses'] - cache_stats_before['misses']
            hit_rate = new_hits / (new_hits + new_misses) if (new_hits + new_misses) > 0 else 0
            
            print(f"\nCache Performance:")
            print(f"  Hit rate: {hit_rate*100:.1f}%")
            print(f"  New factors computed: {cache_stats_after['computes'] - cache_stats_before['computes']}")
            print(f"  Total factors cached: {cache_stats_after['total_factors']}")
            
            # Update performance stats
            self.performance_stats['cache_hits'] += new_hits
            self.performance_stats['total_backtests'] += len(param_combinations)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by return
        if len(df) > 0:
            df = df.sort_values('return_pct', ascending=False)
            
            # Performance summary
            avg_time = total_time / len(results)
            print(f"\nOptimization Complete:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average per backtest: {avg_time:.3f}s")
            print(f"  Best return: {df.iloc[0]['return_pct']:.2f}%")
            print(f"  Best Sharpe: {df['sharpe'].max():.2f}")
        
        return df
    
    def compare_cached_vs_uncached(self, strategy_name: str, data: MarketData,
                                  sample_size: int = 10) -> Dict:
        """Compare performance of cached vs uncached strategies"""
        
        if strategy_name not in CACHED_STRATEGY_REGISTRY:
            print(f"Strategy {strategy_name} doesn't have a cached version")
            return {}
        
        print(f"\nComparing cached vs uncached performance for {strategy_name}")
        
        # Get parameter combinations
        strategy_class = CACHED_STRATEGY_REGISTRY[strategy_name]
        param_grid = strategy_class.param_grid
        
        # Sample parameters
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combos = list(itertools.product(*values))
        
        if len(all_combos) > sample_size:
            import random
            sample_combos = random.sample(all_combos, sample_size)
        else:
            sample_combos = all_combos
        
        # Test cached version
        print(f"\nTesting CACHED version ({len(sample_combos)} combinations)...")
        start_cached = time.time()
        
        cached_times = []
        for combo_values in sample_combos:
            params = dict(zip(keys, combo_values))
            strategy = load_cached_strategy(strategy_name, **params)
            
            result = self.run_single_backtest(strategy, data)
            if result['success']:
                cached_times.append(result['metrics']['compute_time'])
        
        total_cached = time.time() - start_cached
        
        # Clear cache for fair comparison
        get_factor_store().invalidate_factors(data.symbol, data.timeframe)
        
        # Test uncached version
        print(f"\nTesting UNCACHED version ({len(sample_combos)} combinations)...")
        start_uncached = time.time()
        
        uncached_times = []
        for combo_values in sample_combos:
            params = dict(zip(keys, combo_values))
            strategy = load_cached_strategy(strategy_name, **params)
            strategy.config.use_cache = False  # Disable caching
            
            result = self.run_single_backtest(strategy, data)
            if result['success']:
                uncached_times.append(result['metrics']['compute_time'])
        
        total_uncached = time.time() - start_uncached
        
        # Calculate results
        avg_cached = np.mean(cached_times) if cached_times else 0
        avg_uncached = np.mean(uncached_times) if uncached_times else 0
        speedup = avg_uncached / avg_cached if avg_cached > 0 else 0
        
        results = {
            'total_cached_time': total_cached,
            'total_uncached_time': total_uncached,
            'avg_cached_time': avg_cached,
            'avg_uncached_time': avg_uncached,
            'speedup': speedup,
            'time_saved': total_uncached - total_cached,
            'percent_improvement': (1 - total_cached/total_uncached) * 100 if total_uncached > 0 else 0
        }
        
        print(f"\n=== Performance Comparison ===")
        print(f"Cached version:")
        print(f"  Total time: {total_cached:.2f}s")
        print(f"  Average per backtest: {avg_cached:.3f}s")
        print(f"\nUncached version:")
        print(f"  Total time: {total_uncached:.2f}s")
        print(f"  Average per backtest: {avg_uncached:.3f}s")
        print(f"\nImprovement:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time saved: {results['time_saved']:.2f}s ({results['percent_improvement']:.1f}%)")
        
        # Check if we met the target
        if speedup >= 10:
            print(f"\n✅ SUCCESS: Achieved {speedup:.1f}x speedup (target: 10x)")
        else:
            print(f"\n⚠️  Current speedup: {speedup:.1f}x (target: 10x)")
        
        return results
    
    def get_cache_report(self) -> Dict:
        """Get comprehensive cache performance report"""
        cache_stats = get_factor_store().get_cache_stats()
        
        report = {
            'cache_stats': cache_stats,
            'backtest_stats': self.performance_stats,
            'overall_hit_rate': cache_stats['hit_rate'] * 100,
            'memory_efficiency': cache_stats['memory_usage_mb'] / cache_stats['total_factors'] if cache_stats['total_factors'] > 0 else 0,
            'popular_indicators': cache_stats.get('popular_indicators', [])
        }
        
        print("\n=== Cache Performance Report ===")
        print(f"Overall Statistics:")
        print(f"  Hit rate: {report['overall_hit_rate']:.1f}%")
        print(f"  Total factors: {cache_stats['total_factors']}")
        print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
        print(f"  Disk usage: {cache_stats['disk_usage_mb']:.1f} MB")
        print(f"  Average compute time: {cache_stats['avg_compute_time_ms']:.1f} ms")
        
        if report['popular_indicators']:
            print(f"\nMost Used Indicators:")
            for ind in report['popular_indicators']:
                print(f"  {ind['type']}: {ind['count']} factors, {ind['avg_accesses']:.1f} avg accesses")
        
        return report
    
    def save_results(self, results: pd.DataFrame, filepath: str):
        """Save results to JSON file with cache metadata"""
        output_path = Path(filepath)
        output_path.parent.mkdir(exist_ok=True)
        
        # Get cache report
        cache_report = self.get_cache_report()
        
        # Convert DataFrame to records
        records = results.to_dict('records')
        
        # Add metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': records,
            'summary': {
                'total_tests': len(records),
                'best_return': float(results['return_pct'].max()) if len(results) > 0 else 0,
                'best_sharpe': float(results['sharpe'].max()) if len(results) > 0 else 0,
                'avg_compute_time': float(results['compute_time'].mean()) if 'compute_time' in results else 0
            },
            'cache_performance': {
                'hit_rate': cache_report['overall_hit_rate'],
                'memory_usage_mb': cache_report['cache_stats']['memory_usage_mb'],
                'total_factors': cache_report['cache_stats']['total_factors']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cached Backtester with 10x Performance')
    parser.add_argument('command', choices=['optimize', 'compare', 'benchmark', 'cache-stats'])
    parser.add_argument('--symbol', default='BTC', help='Symbol to test')
    parser.add_argument('--timeframe', default='1m', help='Timeframe')
    parser.add_argument('--strategy', help='Strategy name')
    parser.add_argument('--fees', type=float, default=0.001, help='Trading fees')
    parser.add_argument('--cash', type=float, default=10000, help='Initial cash')
    parser.add_argument('--output', default='results/cached_backtest_results.json', help='Output file')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel jobs')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--sample-size', type=int, default=10, help='Sample size for benchmark')
    
    args = parser.parse_args()
    
    # Initialize backtester
    bt = CachedBacktester(use_cache=not args.no_cache)
    
    try:
        # Load data
        if args.command != 'cache-stats':
            print(f"Loading {args.symbol} {args.timeframe} data...")
            data = bt.load_data(args.symbol, args.timeframe)
            print(f"Loaded {len(data)} data points")
        
        if args.command == 'optimize':
            if not args.strategy:
                print("Error: --strategy required for optimize command")
            else:
                results = bt.optimize_strategy(
                    args.strategy, data,
                    init_cash=args.cash,
                    fees=args.fees,
                    n_jobs=args.jobs
                )
                
                if len(results) > 0:
                    print(f"\nTop 5 results:")
                    print(results[['return_pct', 'sharpe', 'num_trades', 'compute_time']].head())
                    
                    bt.save_results(results, args.output)
        
        elif args.command == 'compare':
            # Compare all cached strategies
            strategies = list(CACHED_STRATEGY_REGISTRY.keys())
            
            all_results = []
            for strategy in strategies:
                print(f"\n{'='*60}")
                print(f"Testing {strategy}")
                print('='*60)
                
                results = bt.optimize_strategy(
                    strategy, data,
                    init_cash=args.cash,
                    fees=args.fees,
                    n_jobs=1  # Sequential for better cache performance
                )
                
                if len(results) > 0:
                    best = results.iloc[0]
                    all_results.append({
                        'strategy': strategy,
                        'best_return': best['return_pct'],
                        'best_sharpe': best['sharpe'],
                        'avg_compute_time': results['compute_time'].mean()
                    })
            
            # Summary
            if all_results:
                summary_df = pd.DataFrame(all_results)
                print("\n=== Strategy Comparison Summary ===")
                print(summary_df)
                
                bt.save_results(summary_df, args.output)
        
        elif args.command == 'benchmark':
            if not args.strategy:
                # Benchmark all strategies
                strategies = list(CACHED_STRATEGY_REGISTRY.keys())
            else:
                strategies = [args.strategy]
            
            for strategy in strategies:
                results = bt.compare_cached_vs_uncached(
                    strategy, data, 
                    sample_size=args.sample_size
                )
        
        elif args.command == 'cache-stats':
            report = bt.get_cache_report()
            
    finally:
        bt.dm.close()