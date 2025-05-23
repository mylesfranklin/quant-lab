#!/usr/bin/env python
"""
QuantLab CLI - Unified interface for all operations
"""
import typer
from typing import Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Import our modules
from data_manager import DataManager
from fetch_data_incremental import IncrementalFetcher
from backtester import Backtester
from strategy_framework import list_strategies, STRATEGY_REGISTRY

# Create Typer app
app = typer.Typer(help="QuantLab - Cryptocurrency Backtesting Framework")
console = Console()

# Data management commands
data_app = typer.Typer(help="Data management commands")
app.add_typer(data_app, name="data")

# Strategy commands
strategy_app = typer.Typer(help="Strategy commands")
app.add_typer(strategy_app, name="strategy")

# Backtest commands
backtest_app = typer.Typer(help="Backtesting commands")
app.add_typer(backtest_app, name="backtest")

# Factor cache commands
cache_app = typer.Typer(help="Factor cache management")
app.add_typer(cache_app, name="cache")


@data_app.command("fetch")
def fetch_data(
    symbol: str = typer.Argument("BTC", help="Symbol to fetch"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe (1m, 5m, 1h, etc)"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days to maintain"),
    check_only: bool = typer.Option(False, "--check-only", help="Only check coverage without fetching")
):
    """Fetch cryptocurrency data with incremental updates"""
    with console.status(f"[bold green]Checking {symbol} {timeframe} data..."):
        dm = DataManager()
        
        try:
            if check_only:
                coverage = dm.get_coverage(symbol, timeframe)
                if len(coverage) > 0:
                    row = coverage.iloc[0]
                    console.print(f"\n[green]Current coverage for {symbol} {timeframe}:[/green]")
                    console.print(f"  Start: {row['start_date']}")
                    console.print(f"  End: {row['end_date']}")
                    console.print(f"  Last updated: {row['last_updated']}")
                    
                    stats = dm.analyze_data_quality(symbol, timeframe)
                    console.print(f"\n[yellow]Data quality:[/yellow]")
                    console.print(f"  Total rows: {stats['total_rows']:,}")
                    console.print(f"  Unique days: {stats['unique_days']}")
                else:
                    console.print(f"[red]No data found for {symbol} {timeframe}[/red]")
            else:
                fetcher = IncrementalFetcher(dm)
                fetcher.fetch_incremental(symbol, timeframe, days)
                
        finally:
            dm.close()


@data_app.command("coverage")
def show_coverage(
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe", "-t", help="Filter by timeframe")
):
    """Show data coverage summary"""
    dm = DataManager()
    
    try:
        coverage_df = dm.get_coverage(symbol, timeframe)
        
        if len(coverage_df) == 0:
            console.print("[yellow]No data coverage found[/yellow]")
            return
        
        # Create rich table
        table = Table(title="Data Coverage")
        table.add_column("Symbol", style="cyan")
        table.add_column("Timeframe", style="cyan")
        table.add_column("Start Date", style="green")
        table.add_column("End Date", style="green")
        table.add_column("Days", justify="right", style="yellow")
        table.add_column("Last Updated", style="dim")
        
        for _, row in coverage_df.iterrows():
            days = (row['end_date'] - row['start_date']).days + 1
            table.add_row(
                row['symbol'],
                row['timeframe'],
                str(row['start_date']),
                str(row['end_date']),
                str(days),
                str(row['last_updated'])
            )
        
        console.print(table)
        
    finally:
        dm.close()


@data_app.command("import")
def import_data(
    file: Path = typer.Argument(..., help="Parquet file to import"),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol name"),
    timeframe: str = typer.Option(..., "--timeframe", "-t", help="Timeframe")
):
    """Import existing parquet file into database"""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)
    
    dm = DataManager()
    
    try:
        with console.status(f"[bold green]Importing {file}..."):
            dm.import_parquet(file, symbol, timeframe)
        
        console.print(f"[green]✓ Successfully imported {file}[/green]")
        
        # Show coverage
        coverage = dm.get_coverage(symbol, timeframe)
        if len(coverage) > 0:
            row = coverage.iloc[0]
            console.print(f"  Date range: {row['start_date']} to {row['end_date']}")
            
            stats = dm.analyze_data_quality(symbol, timeframe)
            console.print(f"  Total rows: {stats['total_rows']:,}")
            
    finally:
        dm.close()


@strategy_app.command("list")
def list_strategies_cmd():
    """List all available strategies"""
    strategies = list_strategies()
    
    table = Table(title="Available Strategies")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Description")
    
    for strategy_id, info in strategies.items():
        table.add_row(
            strategy_id,
            info['name'],
            info['version'],
            info['description']
        )
    
    console.print(table)


@strategy_app.command("info")
def strategy_info(
    strategy_id: str = typer.Argument(..., help="Strategy ID")
):
    """Show detailed information about a strategy"""
    if strategy_id not in STRATEGY_REGISTRY:
        console.print(f"[red]Unknown strategy: {strategy_id}[/red]")
        raise typer.Exit(1)
    
    strategy_class = STRATEGY_REGISTRY[strategy_id]
    config = strategy_class.config
    
    console.print(f"\n[bold]{config.name}[/bold] (v{config.version})")
    console.print(f"{config.description}")
    
    # Show parameter grid
    console.print("\n[yellow]Parameters:[/yellow]")
    for param, values in strategy_class.param_grid.items():
        console.print(f"  {param}: {values}")


@backtest_app.command("run")
def run_backtest(
    strategy: str = typer.Argument(..., help="Strategy ID to test"),
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to test"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe"),
    fees: float = typer.Option(0.001, "--fees", "-f", help="Trading fees"),
    cash: float = typer.Option(10000, "--cash", "-c", help="Initial cash"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="Parallel jobs for optimization")
):
    """Run backtest optimization for a strategy"""
    bt = Backtester()
    
    try:
        # Load data
        with console.status(f"[bold green]Loading {symbol} {timeframe} data..."):
            data = bt.load_data(symbol, timeframe)
        
        console.print(f"[green]✓ Loaded {len(data)} data points[/green]")
        
        # Run optimization
        console.print(f"\n[yellow]Optimizing {strategy} strategy...[/yellow]")
        results = bt.optimize_strategy(
            strategy, data,
            init_cash=cash,
            fees=fees,
            n_jobs=jobs
        )
        
        if len(results) == 0:
            console.print("[red]No successful results[/red]")
            return
        
        # Show top results
        console.print(f"\n[green]Top 5 Results:[/green]")
        
        table = Table()
        table.add_column("Return %", justify="right", style="green")
        table.add_column("Sharpe", justify="right", style="yellow")
        table.add_column("Max DD %", justify="right", style="red")
        table.add_column("Trades", justify="right")
        table.add_column("Parameters")
        
        # Get parameter columns
        param_cols = [col for col in results.columns 
                     if col in STRATEGY_REGISTRY[strategy].param_grid]
        
        for _, row in results.head().iterrows():
            params = ", ".join(f"{k}={row[k]}" for k in param_cols)
            table.add_row(
                f"{row['return_pct']:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['max_dd']:.2f}",
                str(int(row['num_trades'])),
                params
            )
        
        console.print(table)
        
        # Save results if requested
        if output:
            bt.save_results(results, str(output))
            console.print(f"\n[green]✓ Results saved to {output}[/green]")
        
    finally:
        bt.dm.close()


@backtest_app.command("compare")
def compare_strategies(
    symbols: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to test"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe"),
    strategies: Optional[List[str]] = typer.Option(None, "--strategy", "-st", help="Specific strategies to test"),
    fees: float = typer.Option(0.001, "--fees", "-f", help="Trading fees"),
    cash: float = typer.Option(10000, "--cash", "-c", help="Initial cash"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file")
):
    """Compare multiple strategies"""
    bt = Backtester()
    
    try:
        # Load data
        with console.status(f"[bold green]Loading {symbols} {timeframe} data..."):
            data = bt.load_data(symbols, timeframe)
        
        console.print(f"[green]✓ Loaded {len(data)} data points[/green]")
        
        # Use all strategies if none specified
        if not strategies:
            strategies = list(STRATEGY_REGISTRY.keys())
        
        # Compare strategies
        console.print(f"\n[yellow]Comparing {len(strategies)} strategies...[/yellow]")
        results = bt.compare_strategies(
            strategies, data,
            init_cash=cash,
            fees=fees
        )
        
        # Show results table
        table = Table(title="Strategy Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Return %", justify="right", style="green")
        table.add_column("Sharpe", justify="right", style="yellow")
        table.add_column("Max DD %", justify="right", style="red")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate %", justify="right")
        table.add_column("Best Parameters")
        
        for _, row in results.iterrows():
            table.add_row(
                row['strategy'],
                f"{row['return_pct']:.2f}",
                f"{row['sharpe']:.2f}",
                f"{row['max_dd']:.2f}",
                str(int(row['num_trades'])),
                f"{row['win_rate']:.1f}",
                row['parameters']
            )
        
        console.print(table)
        
        # Save if requested
        if output:
            bt.save_results(results, str(output))
            console.print(f"\n[green]✓ Results saved to {output}[/green]")
        
    finally:
        bt.dm.close()


@cache_app.command("stats")
def cache_stats():
    """Show factor cache statistics"""
    from factor_store import get_factor_store
    
    fs = get_factor_store()
    stats = fs.get_cache_stats()
    
    # Create summary table
    table = Table(title="Factor Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    
    table.add_row("Hit Rate", f"{stats['hit_rate']*100:.1f}%")
    table.add_row("Total Requests", f"{stats['hits'] + stats['misses']:,}")
    table.add_row("Cache Hits", f"{stats['hits']:,}")
    table.add_row("Cache Misses", f"{stats['misses']:,}")
    table.add_row("Factors Computed", f"{stats['computes']:,}")
    table.add_row("Invalidations", f"{stats['invalidations']:,}")
    table.add_row("", "")  # Separator
    table.add_row("Total Factors", f"{stats['total_factors']:,}")
    table.add_row("Memory Usage", f"{stats['memory_usage_mb']:.1f} MB")
    table.add_row("Disk Usage", f"{stats['disk_usage_mb']:.1f} MB")
    table.add_row("Avg Compute Time", f"{stats['avg_compute_time_ms']:.1f} ms")
    
    console.print(table)
    
    # Popular indicators
    if stats['popular_indicators']:
        pop_table = Table(title="Most Used Indicators")
        pop_table.add_column("Indicator", style="cyan")
        pop_table.add_column("Count", style="yellow", justify="right")
        pop_table.add_column("Avg Accesses", style="green", justify="right")
        
        for ind in stats['popular_indicators']:
            pop_table.add_row(
                ind['type'],
                str(ind['count']),
                f"{ind['avg_accesses']:.1f}"
            )
        
        console.print(pop_table)
    
    fs.close()


@cache_app.command("warm")
def warm_cache(
    strategy: str = typer.Argument(..., help="Strategy to warm cache for"),
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe")
):
    """Pre-compute indicators for a strategy"""
    from backtester_cached import CachedBacktester
    from strategy_framework_cached import CACHED_STRATEGY_REGISTRY
    
    if strategy not in CACHED_STRATEGY_REGISTRY:
        console.print(f"[red]Unknown cached strategy: {strategy}[/red]")
        console.print(f"Available: {list(CACHED_STRATEGY_REGISTRY.keys())}")
        raise typer.Exit(1)
    
    bt = CachedBacktester()
    
    try:
        # Load data
        with console.status(f"[bold green]Loading {symbol} {timeframe} data..."):
            data = bt.load_data(symbol, timeframe)
        
        console.print(f"[green]✓ Loaded {len(data)} data points[/green]")
        
        # Warm cache
        bt.warm_cache(strategy, data)
        
        # Show cache stats
        cache_stats = bt.get_cache_report()
        console.print(f"\n[green]✓ Cache warmed successfully[/green]")
        console.print(f"  Hit rate: {cache_stats['overall_hit_rate']:.1f}%")
        console.print(f"  Total factors: {cache_stats['cache_stats']['total_factors']}")
        
    finally:
        bt.dm.close()


@cache_app.command("clear")
def clear_cache(
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Clear cache for specific symbol"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe", "-t", help="Clear cache for specific timeframe"),
    days: Optional[int] = typer.Option(None, "--older-than", help="Clear factors older than N days"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Clear factor cache"""
    from factor_store import get_factor_store
    
    fs = get_factor_store()
    
    try:
        if days:
            # Clear old factors
            if not confirm:
                confirm = typer.confirm(f"Clear factors older than {days} days?")
            
            if confirm:
                result = fs.cleanup_old_factors(days)
                console.print(f"[green]✓ Removed {result['removed_count']} factors[/green]")
                console.print(f"  Freed {result['freed_mb']:.1f} MB")
        
        elif symbol and timeframe:
            # Clear specific symbol/timeframe
            if not confirm:
                confirm = typer.confirm(f"Clear cache for {symbol} {timeframe}?")
            
            if confirm:
                fs.invalidate_factors(symbol, timeframe)
                console.print(f"[green]✓ Cache cleared for {symbol} {timeframe}[/green]")
        
        else:
            console.print("[yellow]Specify --older-than or both --symbol and --timeframe[/yellow]")
    
    finally:
        fs.close()


@cache_app.command("benchmark")
def benchmark_cache(
    strategy: str = typer.Argument(..., help="Strategy to benchmark"),
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol"),
    timeframe: str = typer.Option("1m", "--timeframe", "-t", help="Timeframe"),
    samples: int = typer.Option(10, "--samples", "-n", help="Number of parameter combinations to test")
):
    """Benchmark cache performance for a strategy"""
    from backtester_cached import CachedBacktester
    
    bt = CachedBacktester()
    
    try:
        # Load data
        with console.status(f"[bold green]Loading {symbol} {timeframe} data..."):
            data = bt.load_data(symbol, timeframe)
        
        console.print(f"[green]✓ Loaded {len(data)} data points[/green]")
        
        # Run benchmark
        results = bt.compare_cached_vs_uncached(strategy, data, sample_size=samples)
        
        # Create results table
        table = Table(title="Cache Performance Benchmark")
        table.add_column("Metric", style="cyan")
        table.add_column("Cached", style="green", justify="right")
        table.add_column("Uncached", style="yellow", justify="right")
        table.add_column("Improvement", style="bold green", justify="right")
        
        table.add_row(
            "Total Time",
            f"{results['total_cached_time']:.2f}s",
            f"{results['total_uncached_time']:.2f}s",
            f"{results['speedup']:.1f}x faster"
        )
        
        table.add_row(
            "Avg per Backtest",
            f"{results['avg_cached_time']:.3f}s",
            f"{results['avg_uncached_time']:.3f}s",
            f"{results['percent_improvement']:.1f}% faster"
        )
        
        table.add_row(
            "Time Saved",
            "-",
            "-",
            f"{results['time_saved']:.2f}s"
        )
        
        console.print(table)
        
        # Success indicator
        if results['speedup'] >= 10:
            console.print(f"\n[bold green]✅ SUCCESS: Achieved {results['speedup']:.1f}x speedup![/bold green]")
        else:
            console.print(f"\n[yellow]⚠️  Current speedup: {results['speedup']:.1f}x (target: 10x)[/yellow]")
    
    finally:
        bt.dm.close()


@app.command("version")
def version():
    """Show QuantLab version"""
    console.print("[bold]QuantLab[/bold] v3.0")
    console.print("High-performance cryptocurrency backtesting with factor caching")
    console.print("✨ Now with 10x faster optimization!")


if __name__ == "__main__":
    app()