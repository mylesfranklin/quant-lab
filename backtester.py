#!/usr/bin/env python
"""
Enhanced backtester with DuckDB integration and improved performance
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
warnings.filterwarnings('ignore')

from data_manager import DataManager
from strategy_framework import BaseStrategy, MarketData, load_strategy, list_strategies, STRATEGY_REGISTRY


class Backtester:
    """Enhanced backtester with data management integration"""
    
    def __init__(self, data_manager: DataManager = None):
        self.dm = data_manager or DataManager()
        self.results = []
        
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
        
        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            open=df['open'].astype(float),
            high=df['high'].astype(float),
            low=df['low'].astype(float),
            close=df['close'].astype(float),
            volume=df['volume'].astype(float)
        )
    
    def run_single_backtest(self, strategy: BaseStrategy, data: MarketData,
                           init_cash: float = 10000, fees: float = 0.001) -> Dict:
        """Run backtest for a single parameter combination"""
        try:
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
                    'calmar': float(stats.get('Calmar Ratio', 0))
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
                         n_jobs: int = 1) -> pd.DataFrame:
        """Run optimization over parameter grid"""
        # Get strategy class
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = STRATEGY_REGISTRY[strategy_name]
        
        # Use default param grid if not provided
        if param_grid is None:
            param_grid = strategy_class.param_grid
        
        # Generate parameter combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        param_combinations = list(itertools.product(*values))
        
        print(f"Testing {len(param_combinations)} parameter combinations for {strategy_name}")
        
        results = []
        
        if n_jobs == 1:
            # Sequential execution
            for i, combo_values in enumerate(param_combinations):
                params = dict(zip(keys, combo_values))
                
                # Create strategy instance
                strategy = strategy_class(**params)
                
                # Run backtest
                result = self.run_single_backtest(strategy, data, init_cash, fees)
                
                if result['success']:
                    # Add parameters to result
                    result['metrics'].update(params)
                    result['metrics']['strategy'] = strategy_name
                    results.append(result['metrics'])
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(param_combinations)}")
        
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                futures = {}
                for combo_values in param_combinations:
                    params = dict(zip(keys, combo_values))
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
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{len(param_combinations)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by return
        if len(df) > 0:
            df = df.sort_values('return_pct', ascending=False)
        
        return df
    
    def compare_strategies(self, strategy_names: List[str], data: MarketData,
                          init_cash: float = 10000, fees: float = 0.001) -> pd.DataFrame:
        """Compare multiple strategies with their best parameters"""
        comparison_results = []
        
        for strategy_name in strategy_names:
            print(f"\nOptimizing {strategy_name}...")
            
            # Run optimization
            opt_results = self.optimize_strategy(
                strategy_name, data, init_cash=init_cash, fees=fees
            )
            
            if len(opt_results) > 0:
                # Get best result
                best = opt_results.iloc[0]
                
                # Extract parameters
                strategy_class = STRATEGY_REGISTRY[strategy_name]
                param_keys = list(strategy_class.param_grid.keys())
                best_params = {k: best[k] for k in param_keys if k in best}
                
                comparison_results.append({
                    'strategy': strategy_name,
                    'return_pct': best['return_pct'],
                    'sharpe': best['sharpe'],
                    'max_dd': best['max_dd'],
                    'num_trades': best['num_trades'],
                    'win_rate': best['win_rate'],
                    'parameters': json.dumps(best_params)
                })
        
        return pd.DataFrame(comparison_results)
    
    def save_results(self, results: pd.DataFrame, filepath: str):
        """Save results to JSON file"""
        output_path = Path(filepath)
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert DataFrame to records
        records = results.to_dict('records')
        
        # Add metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': records,
            'summary': {
                'total_tests': len(records),
                'best_return': float(results['return_pct'].max()) if len(results) > 0 else 0,
                'best_sharpe': float(results['sharpe'].max()) if len(results) > 0 else 0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Backtester')
    parser.add_argument('command', choices=['optimize', 'compare', 'list'])
    parser.add_argument('--symbol', default='BTC', help='Symbol to test')
    parser.add_argument('--timeframe', default='1m', help='Timeframe')
    parser.add_argument('--strategy', help='Strategy name for optimize command')
    parser.add_argument('--strategies', nargs='+', help='Strategy names for compare command')
    parser.add_argument('--fees', type=float, default=0.001, help='Trading fees')
    parser.add_argument('--cash', type=float, default=10000, help='Initial cash')
    parser.add_argument('--output', default='results/backtest_results.json', help='Output file')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel jobs')
    
    args = parser.parse_args()
    
    # Initialize backtester
    bt = Backtester()
    
    try:
        if args.command == 'list':
            # List available strategies
            print("\nAvailable Strategies:")
            for name, info in list_strategies().items():
                print(f"\n{name}:")
                print(f"  Name: {info['name']}")
                print(f"  Version: {info['version']}")
                print(f"  Description: {info['description']}")
        
        else:
            # Load data
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
                        print(results.head())
                        
                        bt.save_results(results, args.output)
                    else:
                        print("No successful results")
            
            elif args.command == 'compare':
                if not args.strategies:
                    # Use all available strategies
                    args.strategies = list(STRATEGY_REGISTRY.keys())
                
                results = bt.compare_strategies(
                    args.strategies, data,
                    init_cash=args.cash,
                    fees=args.fees
                )
                
                print("\nStrategy Comparison:")
                print(results)
                
                bt.save_results(results, args.output)
    
    finally:
        bt.dm.close()