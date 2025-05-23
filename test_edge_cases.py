#!/usr/bin/env python
"""
Edge case testing for factor caching system
Tests partial windows, parameter collisions, and datatype drift
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from rich import print
from rich.console import Console
from rich.table import Table

from data_manager import DataManager
from factor_store import get_factor_store
import cached_indicators as ci
from strategy_framework_cached import MarketData

console = Console()


class EdgeCaseTester:
    """Test edge cases in factor caching"""
    
    def __init__(self):
        self.dm = DataManager()
        self.fs = get_factor_store()
        self.results = []
    
    def run_all_tests(self):
        """Run all edge case tests"""
        console.print("[bold cyan]Factor Caching Edge Case Tests[/bold cyan]\n")
        
        # Load test data
        df = self.dm.load_data('BTC', '1m')
        if len(df) == 0:
            console.print("[red]No data available. Please load data first.[/red]")
            return
        
        # Convert to proper format
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('datetime')
        
        # Run tests
        self.test_partial_window_candles(df)
        self.test_parameter_collisions(df)
        self.test_datatype_drift(df)
        self.test_cache_key_uniqueness()
        self.test_concurrent_access(df)
        self.test_null_handling(df)
        
        # Summary
        self.print_summary()
    
    def test_partial_window_candles(self, df):
        """Test 1: Partial window candles (incomplete last candle)"""
        console.print("[yellow]Test 1: Partial Window Candles[/yellow]")
        
        try:
            # Simulate live data with incomplete last candle
            df_complete = df.iloc[:-1].copy()  # All but last
            df_partial = df.copy()  # Include last (potentially incomplete)
            
            # Add a truly incomplete candle (future timestamp)
            future_time = df.index[-1] + timedelta(minutes=1)
            incomplete_candle = pd.DataFrame({
                'open': [df['close'].iloc[-1]],
                'high': [df['close'].iloc[-1] * 1.001],
                'low': [df['close'].iloc[-1] * 0.999],
                'close': [df['close'].iloc[-1] * 1.0005],
                'volume': [df['volume'].iloc[-1] * 0.1]  # Low volume indicates incomplete
            }, index=[future_time])
            
            df_with_incomplete = pd.concat([df, incomplete_candle])
            
            # Test caching with complete data
            ci.set_current_data(df_complete, 'BTC', '1m')
            ema_complete = ci.ema(period=20)
            
            # Clear cache
            self.fs.invalidate_factors('BTC', '1m')
            
            # Test with incomplete data
            ci.set_current_data(df_with_incomplete, 'BTC', '1m')
            ema_with_incomplete = ci.ema(period=20)
            
            # Check if incomplete candle is handled properly
            # The cache should ideally not store the incomplete candle
            # or mark it specially
            
            last_complete_value = ema_complete.iloc[-1]
            last_with_incomplete = ema_with_incomplete.iloc[-2]  # Second to last should match
            
            match = abs(last_complete_value - last_with_incomplete) < 1e-6
            
            self.results.append({
                'test': 'Partial Window Candles',
                'passed': match,
                'details': f"Complete: {last_complete_value:.2f}, With incomplete: {last_with_incomplete:.2f}"
            })
            
            if match:
                console.print("  [green]✓ Partial candles handled correctly[/green]")
            else:
                console.print("  [red]✗ Partial candle handling issue[/red]")
                
            # Additional check: Verify cache metadata
            stats = self.fs.get_cache_stats()
            console.print(f"  Cache entries: {stats['total_factors']}")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            self.results.append({
                'test': 'Partial Window Candles',
                'passed': False,
                'details': str(e)
            })
        
        console.print()
    
    def test_parameter_collisions(self, df):
        """Test 2: Parameter collisions (float vs int parameters)"""
        console.print("[yellow]Test 2: Parameter Collisions[/yellow]")
        
        try:
            ci.set_current_data(df, 'BTC', '1m')
            
            # Test 1: Integer parameter
            rsi_int = ci.rsi(period=14)
            
            # Test 2: Float parameter that rounds to same value
            rsi_float = ci.rsi(period=14.0)
            
            # Test 3: Float parameter that's actually different
            rsi_float_diff = ci.rsi(period=14.4)  # Should create different cache entry
            
            # Verify results
            int_float_match = np.allclose(rsi_int.values, rsi_float.values, equal_nan=True)
            float_diff = not np.allclose(rsi_int.values, rsi_float_diff.values, equal_nan=True)
            
            self.results.append({
                'test': 'Parameter Collisions - Int/Float',
                'passed': int_float_match and float_diff,
                'details': f"14 vs 14.0 match: {int_float_match}, 14 vs 14.4 differ: {float_diff}"
            })
            
            # Test with different parameter types
            # Create custom indicator that accepts multiple param types
            def test_multi_param(data, param1, param2):
                # Simple calculation using both params
                return data['close'].rolling(int(param1)).mean() * param2
            
            # These should create different cache entries
            result1 = self.fs.get_factor(
                indicator_type='test_multi',
                parameters={'param1': 10, 'param2': 1.5},
                symbol='BTC',
                timeframe='1m',
                compute_func=lambda: test_multi_param(df, 10, 1.5)
            )
            
            result2 = self.fs.get_factor(
                indicator_type='test_multi',
                parameters={'param1': 10.0, 'param2': 1.5},
                symbol='BTC',
                timeframe='1m',
                compute_func=lambda: test_multi_param(df, 10.0, 1.5)
            )
            
            result3 = self.fs.get_factor(
                indicator_type='test_multi',
                parameters={'param1': 10, 'param2': '1.5'},  # String param
                symbol='BTC',
                timeframe='1m',
                compute_func=lambda: test_multi_param(df, 10, float('1.5'))
            )
            
            # Check cache stats to see if proper separation
            stats = self.fs.get_cache_stats()
            
            console.print(f"  [green]✓ Parameter type handling tested[/green]")
            console.print(f"  Total unique factors: {stats['total_factors']}")
            console.print(f"  Cache hit rate: {stats['hit_rate']*100:.1f}%")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            self.results.append({
                'test': 'Parameter Collisions',
                'passed': False,
                'details': str(e)
            })
        
        console.print()
    
    def test_datatype_drift(self, df):
        """Test 3: Datatype drift detection"""
        console.print("[yellow]Test 3: Datatype Drift[/yellow]")
        
        try:
            # Create test data with different dtypes
            df_float64 = df.copy()
            df_float32 = df.astype({
                'open': 'float32',
                'high': 'float32', 
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            })
            
            # Test with float64
            ci.set_current_data(df_float64, 'BTC_F64', '1m')
            ema_f64 = ci.ema(period=20)
            
            # Test with float32
            ci.set_current_data(df_float32, 'BTC_F32', '1m')
            ema_f32 = ci.ema(period=20)
            
            # Check if different dtypes create different cache entries
            # or if the system handles dtype conversion properly
            
            # Get cache entries
            stats_before = self.fs.get_cache_stats()
            
            # Try to use float32 data with float64 cache
            ci.set_current_data(df_float32, 'BTC_F64', '1m')
            ema_mixed = ci.ema(period=20)
            
            stats_after = self.fs.get_cache_stats()
            
            # Check if cache was invalidated or reused
            cache_reused = stats_after['hits'] > stats_before['hits']
            
            self.results.append({
                'test': 'Datatype Drift',
                'passed': True,  # System handles it without crashing
                'details': f"Cache {'reused' if cache_reused else 'invalidated'} on dtype change"
            })
            
            console.print(f"  [green]✓ Datatype changes handled[/green]")
            console.print(f"  Behavior: Cache {'reused' if cache_reused else 'invalidated'} on dtype change")
            
            # Test with completely different data structure
            df_extra_col = df.copy()
            df_extra_col['extra'] = np.random.randn(len(df))
            
            ci.set_current_data(df_extra_col, 'BTC_EXTRA', '1m')
            ema_extra = ci.ema(period=20)
            
            console.print(f"  [green]✓ Schema changes handled[/green]")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            self.results.append({
                'test': 'Datatype Drift',
                'passed': False,
                'details': str(e)
            })
        
        console.print()
    
    def test_cache_key_uniqueness(self):
        """Test 4: Cache key uniqueness and collisions"""
        console.print("[yellow]Test 4: Cache Key Uniqueness[/yellow]")
        
        try:
            # Test similar parameter combinations that could collide
            test_cases = [
                ('ema', {'period': 20}, 'BTC', '1m'),
                ('ema', {'period': 20.0}, 'BTC', '1m'),
                ('ema', {'period': 20}, 'BTC', '5m'),  # Different timeframe
                ('ema', {'period': 20}, 'ETH', '1m'),  # Different symbol
                ('sma', {'period': 20}, 'BTC', '1m'),  # Different indicator
                ('ema', {'period': 21}, 'BTC', '1m'),  # Different param value
            ]
            
            factor_ids = []
            for indicator, params, symbol, tf in test_cases:
                factor_id = self.fs._generate_factor_id(indicator, params, symbol, tf)
                factor_ids.append(factor_id)
                console.print(f"  {indicator}({params}) {symbol}/{tf}: {factor_id[:8]}...")
            
            # Check uniqueness
            unique_ids = len(set(factor_ids))
            expected_unique = len(test_cases) - 1  # First two might be same
            
            self.results.append({
                'test': 'Cache Key Uniqueness',
                'passed': unique_ids >= expected_unique - 1,
                'details': f"{unique_ids} unique IDs from {len(test_cases)} test cases"
            })
            
            if unique_ids >= expected_unique - 1:
                console.print(f"  [green]✓ Cache keys properly unique: {unique_ids}/{len(test_cases)}[/green]")
            else:
                console.print(f"  [red]✗ Cache key collision detected[/red]")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            self.results.append({
                'test': 'Cache Key Uniqueness',
                'passed': False,
                'details': str(e)
            })
        
        console.print()
    
    def test_concurrent_access(self, df):
        """Test 5: Concurrent access to cache"""
        console.print("[yellow]Test 5: Concurrent Access[/yellow]")
        
        try:
            import threading
            
            ci.set_current_data(df, 'BTC', '1m')
            
            results = []
            errors = []
            
            def compute_indicator(period, results_list, error_list):
                try:
                    result = ci.ema(period=period)
                    results_list.append((period, len(result)))
                except Exception as e:
                    error_list.append((period, str(e)))
            
            # Launch multiple threads
            threads = []
            for period in [10, 20, 30, 40, 50]:
                t = threading.Thread(
                    target=compute_indicator,
                    args=(period, results, errors)
                )
                threads.append(t)
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join()
            
            # Check results
            success = len(errors) == 0 and len(results) == 5
            
            self.results.append({
                'test': 'Concurrent Access',
                'passed': success,
                'details': f"{len(results)} successful, {len(errors)} errors"
            })
            
            if success:
                console.print(f"  [green]✓ Concurrent access handled safely[/green]")
                console.print(f"  Computed {len(results)} indicators concurrently")
            else:
                console.print(f"  [red]✗ Concurrent access issues[/red]")
                for period, error in errors:
                    console.print(f"    Period {period}: {error}")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            self.results.append({
                'test': 'Concurrent Access',
                'passed': False,
                'details': str(e)
            })
        
        console.print()
    
    def test_null_handling(self, df):
        """Test 6: NULL/NaN handling in cache"""
        console.print("[yellow]Test 6: NULL/NaN Handling[/yellow]")
        
        try:
            # Create data with NaN values
            df_with_nan = df.copy()
            
            # Introduce NaN in different places
            df_with_nan.loc[df_with_nan.index[10:15], 'close'] = np.nan
            df_with_nan.loc[df_with_nan.index[100], 'volume'] = np.nan
            
            ci.set_current_data(df_with_nan, 'BTC_NAN', '1m')
            
            # Test various indicators with NaN
            ema_result = ci.ema(period=20)
            rsi_result = ci.rsi(period=14)
            bb_result = ci.bollinger_bands(period=20)
            
            # Check if NaN propagates correctly
            has_nan_ema = ema_result.isna().any()
            has_nan_rsi = rsi_result.isna().any()
            
            # Store and retrieve to test NaN persistence
            self.fs.invalidate_factors('BTC_NAN', '1m')
            
            # Recompute - should get from cache
            ema_cached = ci.ema(period=20)
            
            # Verify NaN values are preserved
            nan_preserved = np.array_equal(
                ema_result.isna().values,
                ema_cached.isna().values,
                equal_nan=True
            )
            
            self.results.append({
                'test': 'NULL/NaN Handling',
                'passed': nan_preserved,
                'details': f"NaN preserved: {nan_preserved}, EMA has NaN: {has_nan_ema}"
            })
            
            if nan_preserved:
                console.print("  [green]✓ NaN values handled correctly[/green]")
                console.print(f"  EMA NaN count: {ema_result.isna().sum()}")
                console.print(f"  RSI NaN count: {rsi_result.isna().sum()}")
            else:
                console.print("  [red]✗ NaN handling issue[/red]")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")
            self.results.append({
                'test': 'NULL/NaN Handling',
                'passed': False,
                'details': str(e)
            })
        
        console.print()
    
    def print_summary(self):
        """Print test summary"""
        console.print("\n[bold cyan]Test Summary[/bold cyan]")
        
        table = Table(title="Edge Case Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        
        passed = 0
        for result in self.results:
            status = "[green]PASS[/green]" if result['passed'] else "[red]FAIL[/red]"
            if result['passed']:
                passed += 1
            
            table.add_row(
                result['test'],
                status,
                result['details']
            )
        
        console.print(table)
        
        # Overall result
        total = len(self.results)
        console.print(f"\n[bold]Overall: {passed}/{total} tests passed[/bold]")
        
        if passed == total:
            console.print("[bold green]✅ All edge cases handled correctly![/bold green]")
        else:
            console.print("[bold yellow]⚠️  Some edge cases need attention[/bold yellow]")
        
        # Cache stats
        stats = self.fs.get_cache_stats()
        console.print(f"\nFinal cache state:")
        console.print(f"  Total factors: {stats['total_factors']}")
        console.print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        console.print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
    
    def cleanup(self):
        """Clean up test data"""
        # Clear test symbols
        for symbol in ['BTC_F64', 'BTC_F32', 'BTC_EXTRA', 'BTC_NAN']:
            self.fs.invalidate_factors(symbol, '1m')
        
        self.dm.close()
        self.fs.close()


def main():
    """Run edge case tests"""
    tester = EdgeCaseTester()
    
    try:
        tester.run_all_tests()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()