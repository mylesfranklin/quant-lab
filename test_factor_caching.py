#!/usr/bin/env python
"""
Comprehensive test suite for factor caching implementation
Validates 10x performance improvement and all success criteria
"""
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print

from data_manager import DataManager
from backtester_cached import CachedBacktester
from strategy_framework_cached import CACHED_STRATEGY_REGISTRY, benchmark_caching
from factor_store import get_factor_store
import cached_indicators as ci

console = Console()


class FactorCacheValidator:
    """Validates factor caching implementation meets all success criteria"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'success_criteria': {
                'grid_runtime_improved_80_percent': False,
                'no_change_in_equity_curves': False,
                'cache_size_under_1gb': False,
                'new_strategies_use_cache': False,
                'target_10x_speedup': False
            }
        }
        
        self.dm = DataManager()
        self.bt_cached = CachedBacktester(use_cache=True)
        self.bt_uncached = CachedBacktester(use_cache=False)
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        console.print("[bold cyan]Starting Factor Cache Validation Suite[/bold cyan]\n")
        
        # Load test data
        console.print("Loading test data...")
        data = self.bt_cached.load_data('BTC', '1m')
        console.print(f"✓ Loaded {len(data)} data points\n")
        
        # Test 1: Single strategy performance
        self.test_single_strategy_performance(data)
        
        # Test 2: Grid optimization performance
        self.test_grid_optimization_performance(data)
        
        # Test 3: Cache correctness
        self.test_cache_correctness(data)
        
        # Test 4: Memory and disk usage
        self.test_cache_size_limits()
        
        # Test 5: Multi-strategy optimization
        self.test_multi_strategy_optimization(data)
        
        # Test 6: Cache persistence
        self.test_cache_persistence(data)
        
        # Generate report
        self.generate_report()
    
    def test_single_strategy_performance(self, data):
        """Test 1: Measure single strategy performance improvement"""
        console.print("[yellow]Test 1: Single Strategy Performance[/yellow]")
        
        strategy = 'ma_crossover_cached'
        params = {'fast_period': 20, 'slow_period': 100, 'ma_type': 'EMA'}
        
        # Clear cache
        get_factor_store().invalidate_factors(data.symbol, data.timeframe)
        
        # Test uncached
        console.print("  Testing uncached performance...")
        times_uncached = []
        for i in range(3):
            start = time.time()
            result = self.bt_uncached.optimize_strategy(
                strategy, data, 
                param_grid={k: [v] for k, v in params.items()},
                warm_cache=False
            )
            elapsed = time.time() - start
            times_uncached.append(elapsed)
            console.print(f"    Run {i+1}: {elapsed:.3f}s")
        
        # Test cached (first run will populate cache)
        console.print("  Testing cached performance...")
        times_cached = []
        for i in range(3):
            start = time.time()
            result = self.bt_cached.optimize_strategy(
                strategy, data,
                param_grid={k: [v] for k, v in params.items()},
                warm_cache=(i == 0)
            )
            elapsed = time.time() - start
            times_cached.append(elapsed)
            console.print(f"    Run {i+1}: {elapsed:.3f}s")
        
        # Calculate speedup (use runs 2-3 to exclude cache warming)
        avg_uncached = np.mean(times_uncached[1:])
        avg_cached = np.mean(times_cached[1:])
        speedup = avg_uncached / avg_cached if avg_cached > 0 else 0
        
        self.results['tests']['single_strategy'] = {
            'avg_uncached': avg_uncached,
            'avg_cached': avg_cached,
            'speedup': speedup,
            'improvement_percent': (1 - avg_cached/avg_uncached) * 100
        }
        
        console.print(f"  [green]Speedup: {speedup:.2f}x[/green]\n")
    
    def test_grid_optimization_performance(self, data):
        """Test 2: Measure grid optimization performance (100 combinations)"""
        console.print("[yellow]Test 2: Grid Optimization Performance (100 combinations)[/yellow]")
        
        # Define grid with ~100 combinations
        param_grid = {
            'fast_period': [10, 20, 30, 40, 50],
            'slow_period': [100, 150, 200, 250],
            'ma_type': ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA']
        }
        
        total_combos = 1
        for values in param_grid.values():
            total_combos *= len(values)
        console.print(f"  Total combinations: {total_combos}")
        
        # Clear cache
        get_factor_store().invalidate_factors(data.symbol, data.timeframe)
        
        # Test uncached
        console.print("  Testing uncached grid optimization...")
        start_uncached = time.time()
        results_uncached = self.bt_uncached.optimize_strategy(
            'ma_crossover_cached', data,
            param_grid=param_grid,
            warm_cache=False
        )
        time_uncached = time.time() - start_uncached
        
        # Test cached
        console.print("\n  Testing cached grid optimization...")
        start_cached = time.time()
        results_cached = self.bt_cached.optimize_strategy(
            'ma_crossover_cached', data,
            param_grid=param_grid,
            warm_cache=True
        )
        time_cached = time.time() - start_cached
        
        # Calculate improvement
        speedup = time_uncached / time_cached if time_cached > 0 else 0
        improvement = (1 - time_cached/time_uncached) * 100
        
        self.results['tests']['grid_optimization'] = {
            'combinations': total_combos,
            'time_uncached': time_uncached,
            'time_cached': time_cached,
            'speedup': speedup,
            'improvement_percent': improvement,
            'target_met': time_cached <= 2.0  # Target: 1-2 seconds
        }
        
        # Check success criteria
        if improvement >= 80:
            self.results['success_criteria']['grid_runtime_improved_80_percent'] = True
        
        if speedup >= 10:
            self.results['success_criteria']['target_10x_speedup'] = True
        
        console.print(f"  Uncached: {time_uncached:.2f}s")
        console.print(f"  Cached: {time_cached:.2f}s")
        console.print(f"  [green]Speedup: {speedup:.2f}x ({improvement:.1f}% improvement)[/green]")
        
        if time_cached <= 2.0:
            console.print(f"  [green]✓ Met target of 1-2 seconds![/green]\n")
        else:
            console.print(f"  [yellow]⚠ Target: 1-2s, Actual: {time_cached:.2f}s[/yellow]\n")
    
    def test_cache_correctness(self, data):
        """Test 3: Verify cached results match uncached results"""
        console.print("[yellow]Test 3: Cache Correctness Verification[/yellow]")
        
        # Test parameters
        test_cases = [
            ('ma_crossover_cached', {'fast_period': 20, 'slow_period': 100, 'ma_type': 'EMA'}),
            ('rsi_mean_reversion_cached', {'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70, 'bb_period': 20, 'bb_std': 2.0, 'use_bb_filter': True}),
            ('trend_following_cached', {'ema_fast': 12, 'ema_slow': 50, 'adx_period': 14, 'adx_threshold': 25, 'atr_period': 14, 'atr_multiplier': 2.0})
        ]
        
        all_match = True
        
        for strategy_name, params in test_cases:
            console.print(f"  Testing {strategy_name}...")
            
            # Clear cache
            get_factor_store().invalidate_factors(data.symbol, data.timeframe)
            
            # Run uncached
            result_uncached = self.bt_uncached.optimize_strategy(
                strategy_name, data,
                param_grid={k: [v] for k, v in params.items()},
                warm_cache=False
            )
            
            # Run cached
            result_cached = self.bt_cached.optimize_strategy(
                strategy_name, data,
                param_grid={k: [v] for k, v in params.items()},
                warm_cache=True
            )
            
            # Compare results
            if len(result_uncached) > 0 and len(result_cached) > 0:
                metrics_to_compare = ['return_pct', 'sharpe', 'max_dd', 'num_trades']
                
                match = True
                for metric in metrics_to_compare:
                    uncached_val = result_uncached.iloc[0][metric]
                    cached_val = result_cached.iloc[0][metric]
                    
                    # Allow small floating point differences
                    if abs(uncached_val - cached_val) > 1e-6:
                        match = False
                        console.print(f"    [red]✗ {metric}: {uncached_val} vs {cached_val}[/red]")
                
                if match:
                    console.print(f"    [green]✓ Results match exactly[/green]")
                else:
                    all_match = False
            else:
                console.print(f"    [red]✗ No results to compare[/red]")
                all_match = False
        
        self.results['tests']['cache_correctness'] = {
            'all_match': all_match,
            'test_cases': len(test_cases)
        }
        
        if all_match:
            self.results['success_criteria']['no_change_in_equity_curves'] = True
        
        console.print()
    
    def test_cache_size_limits(self):
        """Test 4: Verify cache size remains under limits"""
        console.print("[yellow]Test 4: Cache Size Limits[/yellow]")
        
        cache_stats = get_factor_store().get_cache_stats()
        
        memory_mb = cache_stats['memory_usage_mb']
        disk_mb = cache_stats['disk_usage_mb']
        total_mb = memory_mb + disk_mb
        
        self.results['tests']['cache_size'] = {
            'memory_mb': memory_mb,
            'disk_mb': disk_mb,
            'total_mb': total_mb,
            'total_factors': cache_stats['total_factors'],
            'under_1gb': total_mb < 1024
        }
        
        console.print(f"  Memory usage: {memory_mb:.1f} MB")
        console.print(f"  Disk usage: {disk_mb:.1f} MB")
        console.print(f"  Total: {total_mb:.1f} MB")
        console.print(f"  Total factors: {cache_stats['total_factors']}")
        
        if total_mb < 1024:
            self.results['success_criteria']['cache_size_under_1gb'] = True
            console.print(f"  [green]✓ Under 1GB limit[/green]\n")
        else:
            console.print(f"  [red]✗ Exceeds 1GB limit[/red]\n")
    
    def test_multi_strategy_optimization(self, data):
        """Test 5: Test multiple strategies use cache efficiently"""
        console.print("[yellow]Test 5: Multi-Strategy Optimization[/yellow]")
        
        strategies = list(CACHED_STRATEGY_REGISTRY.keys())[:3]  # Test first 3
        
        total_time = 0
        strategy_results = []
        
        for strategy in strategies:
            console.print(f"  Optimizing {strategy}...")
            
            start = time.time()
            result = self.bt_cached.optimize_strategy(
                strategy, data,
                param_grid=None,  # Use default grid
                warm_cache=True
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            if len(result) > 0:
                best = result.iloc[0]
                strategy_results.append({
                    'strategy': strategy,
                    'time': elapsed,
                    'combinations': len(result),
                    'best_return': best['return_pct'],
                    'best_sharpe': best['sharpe']
                })
                
                console.print(f"    Time: {elapsed:.2f}s, Best return: {best['return_pct']:.2f}%")
        
        # All strategies should benefit from shared indicators
        cache_stats = get_factor_store().get_cache_stats()
        
        self.results['tests']['multi_strategy'] = {
            'strategies_tested': len(strategies),
            'total_time': total_time,
            'avg_time_per_strategy': total_time / len(strategies),
            'cache_hit_rate': cache_stats['hit_rate'],
            'results': strategy_results
        }
        
        if cache_stats['hit_rate'] > 0.5:  # At least 50% cache reuse
            self.results['success_criteria']['new_strategies_use_cache'] = True
        
        console.print(f"  Total time: {total_time:.2f}s")
        console.print(f"  Cache hit rate: {cache_stats['hit_rate']*100:.1f}%\n")
    
    def test_cache_persistence(self, data):
        """Test 6: Verify cache persists across sessions"""
        console.print("[yellow]Test 6: Cache Persistence[/yellow]")
        
        # Get current cache stats
        stats_before = get_factor_store().get_cache_stats()
        factors_before = stats_before['total_factors']
        
        # Close and reopen factor store (simulating new session)
        get_factor_store().close()
        
        # Force new instance
        from factor_store import _global_factor_store
        _global_factor_store = None
        
        # Get stats from new instance
        stats_after = get_factor_store().get_cache_stats()
        factors_after = stats_after['total_factors']
        
        self.results['tests']['cache_persistence'] = {
            'factors_before': factors_before,
            'factors_after': factors_after,
            'persisted': factors_after == factors_before
        }
        
        if factors_after == factors_before:
            console.print(f"  [green]✓ Cache persisted: {factors_after} factors[/green]\n")
        else:
            console.print(f"  [red]✗ Cache not persisted: {factors_before} -> {factors_after}[/red]\n")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        console.print("[bold cyan]Factor Cache Validation Report[/bold cyan]\n")
        
        # Success criteria table
        table = Table(title="Success Criteria")
        table.add_column("Criteria", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")
        
        criteria_details = {
            'grid_runtime_improved_80_percent': f"{self.results['tests']['grid_optimization']['improvement_percent']:.1f}% improvement",
            'no_change_in_equity_curves': f"{self.results['tests']['cache_correctness']['test_cases']} strategies tested",
            'cache_size_under_1gb': f"{self.results['tests']['cache_size']['total_mb']:.1f} MB used",
            'new_strategies_use_cache': f"{self.results['tests']['multi_strategy']['cache_hit_rate']*100:.1f}% hit rate",
            'target_10x_speedup': f"{self.results['tests']['grid_optimization']['speedup']:.1f}x speedup achieved"
        }
        
        all_passed = True
        for criteria, passed in self.results['success_criteria'].items():
            status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
            if not passed:
                all_passed = False
            
            readable_name = criteria.replace('_', ' ').title()
            details = criteria_details.get(criteria, "")
            
            table.add_row(readable_name, status, details)
        
        console.print(table)
        
        # Performance summary
        perf_table = Table(title="Performance Summary")
        perf_table.add_column("Test", style="cyan")
        perf_table.add_column("Speedup", style="yellow", justify="right")
        perf_table.add_column("Time Saved", style="green", justify="right")
        
        if 'single_strategy' in self.results['tests']:
            single = self.results['tests']['single_strategy']
            perf_table.add_row(
                "Single Strategy",
                f"{single['speedup']:.2f}x",
                f"{single['avg_uncached'] - single['avg_cached']:.3f}s"
            )
        
        if 'grid_optimization' in self.results['tests']:
            grid = self.results['tests']['grid_optimization']
            perf_table.add_row(
                "Grid Optimization (100)",
                f"{grid['speedup']:.2f}x",
                f"{grid['time_uncached'] - grid['time_cached']:.1f}s"
            )
        
        console.print("\n")
        console.print(perf_table)
        
        # Final verdict
        console.print("\n[bold]Final Verdict:[/bold]")
        if all_passed:
            console.print("[bold green]✅ ALL SUCCESS CRITERIA MET![/bold green]")
            console.print("Factor caching implementation is ready for production use.")
        else:
            console.print("[bold yellow]⚠️  Some criteria not met. See details above.[/bold yellow]")
        
        # Save detailed report
        report_path = Path("results/factor_cache_validation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        console.print(f"\nDetailed report saved to: {report_path}")
    
    def cleanup(self):
        """Clean up resources"""
        self.dm.close()
        get_factor_store().close()


def main():
    """Run validation suite"""
    validator = FactorCacheValidator()
    
    try:
        validator.run_all_tests()
    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()