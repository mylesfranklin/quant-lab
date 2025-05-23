#!/usr/bin/env python
"""Unified results analyzer with markdown output support"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path) as f:
        results = json.load(f)
    
    # Convert to DataFrame and handle Infinity/NaN
    df = pd.DataFrame(results)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def analyze_results(df):
    """Analyze backtest results and return summary statistics"""
    # Filter to only strategies that made trades
    df_with_trades = df[df['num_trades'] > 0]
    
    analysis = {
        'total_combinations': len(df),
        'combinations_with_trades': len(df_with_trades),
        'combinations_no_trades': len(df) - len(df_with_trades)
    }
    
    # Per-strategy analysis
    strategy_stats = []
    for strategy in df['strategy'].unique():
        strat_df = df[df['strategy'] == strategy]
        strat_with_trades = strat_df[strat_df['num_trades'] > 0]
        
        if len(strat_with_trades) > 0:
            stats = {
                'strategy': strategy,
                'combinations': len(strat_df),
                'with_trades': len(strat_with_trades),
                'best_return': strat_with_trades['return_pct'].max(),
                'avg_return': strat_with_trades['return_pct'].mean(),
                'best_sharpe': strat_with_trades['sharpe'].max(),
                'avg_trades': strat_with_trades['num_trades'].mean()
            }
            strategy_stats.append(stats)
    
    return analysis, pd.DataFrame(strategy_stats), df_with_trades

def get_top_results(df, metric, n=10):
    """Get top N results by a specific metric"""
    return df.nlargest(n, metric)

def format_parameters(row):
    """Extract and format strategy parameters from a result row"""
    exclude_cols = ['return_pct', 'sharpe', 'max_dd', 'win_rate', 'num_trades', 
                   'avg_trade_duration', 'profit_factor', 'final_value', 
                   'strategy', 'timestamp']
    
    params = {k: v for k, v in row.items() if k not in exclude_cols and pd.notna(v)}
    return ', '.join(f'{k}={v}' for k, v in params.items())

def generate_markdown_report(analysis, strategy_df, df_with_trades):
    """Generate a markdown report of the results"""
    report = []
    
    # Header
    report.append("# Backtest Results Analysis\n")
    
    # Summary
    report.append("## Summary")
    report.append(f"- Total combinations tested: {analysis['total_combinations']}")
    report.append(f"- Combinations with trades: {analysis['combinations_with_trades']}")
    report.append(f"- Combinations with no trades: {analysis['combinations_no_trades']}\n")
    
    # Strategy Performance Table
    report.append("## Strategy Performance")
    # Manual markdown table generation
    if len(strategy_df) > 0:
        report.append("| Strategy | Combinations | With Trades | Best Return | Avg Return | Best Sharpe | Avg Trades |")
        report.append("|----------|--------------|-------------|-------------|------------|-------------|------------|")
        for _, row in strategy_df.iterrows():
            report.append(f"| {row['strategy']} | {row['combinations']} | {row['with_trades']} | "
                         f"{row['best_return']:.2f} | {row['avg_return']:.2f} | "
                         f"{row['best_sharpe']:.2f} | {row['avg_trades']:.1f} |")
    report.append("")
    
    if len(df_with_trades) > 0:
        # Top 10 by Return
        report.append("## Top 10 Results by Return")
        top_return = get_top_results(df_with_trades, 'return_pct')
        
        results_table = []
        for _, row in top_return.iterrows():
            results_table.append({
                'Strategy': row['strategy'],
                'Return %': f"{row['return_pct']:.2f}",
                'Sharpe': f"{row['sharpe']:.2f}",
                'Trades': row['num_trades'],
                'Parameters': format_parameters(row)
            })
        
        # Manual markdown table for results
        report.append("| Strategy | Return % | Sharpe | Trades | Parameters |")
        report.append("|----------|----------|--------|--------|------------|")
        for item in results_table:
            report.append(f"| {item['Strategy']} | {item['Return %']} | {item['Sharpe']} | "
                         f"{item['Trades']} | {item['Parameters']} |")
        report.append("")
        
        # Top 10 by Sharpe
        report.append("## Top 10 Results by Sharpe Ratio")
        top_sharpe = get_top_results(df_with_trades, 'sharpe')
        
        sharpe_table = []
        for _, row in top_sharpe.iterrows():
            sharpe_table.append({
                'Strategy': row['strategy'],
                'Sharpe': f"{row['sharpe']:.2f}",
                'Return %': f"{row['return_pct']:.2f}",
                'Max DD %': f"{row['max_dd']:.2f}",
                'Parameters': format_parameters(row)
            })
        
        # Manual markdown table for sharpe
        report.append("| Strategy | Sharpe | Return % | Max DD % | Parameters |")
        report.append("|----------|--------|----------|----------|------------|")
        for item in sharpe_table:
            report.append(f"| {item['Strategy']} | {item['Sharpe']} | {item['Return %']} | "
                         f"{item['Max DD %']} | {item['Parameters']} |")
        report.append("")
        
        # Best Overall
        best_idx = df_with_trades['return_pct'].idxmax()
        best = df_with_trades.loc[best_idx]
        
        report.append("## Best Overall Result")
        report.append(f"- **Strategy**: {best['strategy']}")
        report.append(f"- **Return**: {best['return_pct']:.2f}%")
        report.append(f"- **Sharpe Ratio**: {best['sharpe']:.2f}")
        report.append(f"- **Max Drawdown**: {best['max_dd']:.2f}%")
        report.append(f"- **Win Rate**: {best['win_rate']:.1f}%")
        report.append(f"- **Number of Trades**: {best['num_trades']}")
        report.append(f"- **Parameters**: {format_parameters(best)}")
    
    return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description='Analyze backtest results')
    parser.add_argument('--input', default='results/metrics.json', 
                       help='Input results file (default: results/metrics.json)')
    parser.add_argument('--output', help='Output markdown file (optional)')
    parser.add_argument('--save-best', action='store_true', 
                       help='Save best parameters to JSON')
    args = parser.parse_args()
    
    # Check if results file exists
    results_file = Path(args.input)
    if not results_file.exists():
        print(f"No results file found at {results_file}. Run batch_backtest.py first.")
        return
    
    # Load and analyze results
    df = load_results(results_file)
    analysis, strategy_df, df_with_trades = analyze_results(df)
    
    # Generate report
    report = generate_markdown_report(analysis, strategy_df, df_with_trades)
    
    # Output report
    print(report)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\nReport saved to {output_path}")
    
    # Save best parameters if requested
    if args.save_best and len(df_with_trades) > 0:
        best_params = {}
        for strategy in df['strategy'].unique():
            strat_df = df[df['strategy'] == strategy]
            strat_df = strat_df[strat_df['num_trades'] > 0]
            
            if len(strat_df) > 0:
                best_row = strat_df.loc[strat_df['return_pct'].idxmax()]
                
                exclude_cols = ['return_pct', 'sharpe', 'max_dd', 'win_rate', 
                              'num_trades', 'avg_trade_duration', 'profit_factor', 
                              'final_value', 'strategy', 'timestamp']
                
                params = {k: v for k, v in best_row.items() 
                         if k not in exclude_cols and pd.notna(v)}
                
                best_params[strategy] = {
                    'parameters': params,
                    'return_pct': float(best_row['return_pct']),
                    'sharpe': float(best_row['sharpe']),
                    'num_trades': int(best_row['num_trades'])
                }
        
        # Save best parameters
        best_params_file = results_file.parent / 'best_parameters.json'
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\nBest parameters saved to {best_params_file}")

if __name__ == "__main__":
    main()