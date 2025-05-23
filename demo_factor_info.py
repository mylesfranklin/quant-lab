#!/usr/bin/env python
"""
Demonstration of consolidated factor info and invalidation helpers
"""
from rich import print
from rich.console import Console
from rich.table import Table
from datetime import datetime

# Import the consolidated helpers
from factor_store_enhanced import (
    factor_info, 
    find_factors, 
    validate_factor,
    invalidate_stale_factors
)

# Also get the store for some operations
from factor_store_enhanced import get_enhanced_factor_store
from data_manager import DataManager
import cached_indicators as ci
import pandas as pd

console = Console()


def demo_factor_info():
    """Demonstrate factor info retrieval"""
    console.print("[bold cyan]Factor Info Demo[/bold cyan]\n")
    
    # First, ensure we have some cached factors
    dm = DataManager()
    df = dm.load_data('BTC', '1m')
    
    if len(df) > 0:
        # Setup data context
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df = df.set_index('datetime')
        ci.set_current_data(df, 'BTC', '1m')
        
        # Create some factors
        console.print("Creating test factors...")
        ema_20 = ci.ema(period=20)
        rsi_14 = ci.rsi(period=14)
        console.print("✓ Factors created\n")
    
    # 1. Find all cached factors
    console.print("[yellow]1. Finding all cached factors:[/yellow]")
    all_factors = find_factors()
    
    if all_factors:
        table = Table(title="Cached Factors")
        table.add_column("Factor ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Symbol", style="yellow")
        table.add_column("Parameters")
        table.add_column("Fingerprint", style="dim")
        table.add_column("Partial", style="red")
        
        for factor in all_factors[:5]:  # Show first 5
            table.add_row(
                factor['factor_id'][:8] + "...",
                factor['indicator_type'],
                f"{factor['symbol']}/{factor['timeframe']}",
                str(factor['parameters']),
                factor['fingerprint'][:8] + "...",
                "Yes" if factor['is_partial'] else "No"
            )
        
        console.print(table)
    else:
        console.print("No cached factors found")
    
    # 2. Get detailed info about a specific factor
    if all_factors:
        factor_id = all_factors[0]['factor_id']
        console.print(f"\n[yellow]2. Detailed info for factor {factor_id[:8]}...:[/yellow]")
        
        info = factor_info(factor_id)
        if info:
            console.print(f"  Indicator: [green]{info['indicator_type']}[/green]")
            console.print(f"  Parameters: {info['parameters']}")
            console.print(f"  Symbol/TF: {info['symbol']}/{info['timeframe']}")
            console.print(f"  Rows: {info['rows']:,}")
            console.print(f"  Dtype: {info['dtype']}")
            console.print(f"  Fingerprint: {info['fingerprint']}")
            console.print(f"  Is Partial: {info['is_partial']}")
            console.print(f"  Size: {info['size_bytes']:,} bytes")
            console.print(f"  Compute Time: {info['computation_time_ms']:.1f}ms")
            
            if info.get('data_range'):
                console.print(f"  Data Range: {info['data_range']['start']} to {info['data_range']['end']}")
            
            if info.get('value_stats'):
                stats = info['value_stats']
                console.print(f"  Value Stats: min={stats['min']:.2f}, avg={stats['avg']:.2f}, max={stats['max']:.2f}")
    
    # 3. Find factors by criteria
    console.print("\n[yellow]3. Finding RSI factors:[/yellow]")
    rsi_factors = find_factors(indicator_type='rsi')
    
    for factor in rsi_factors:
        console.print(f"  • {factor['factor_id'][:8]}... - {factor['symbol']}/{factor['timeframe']} - params: {factor['parameters']}")
    
    # 4. Validate a factor
    if all_factors:
        factor_id = all_factors[0]['factor_id']
        console.print(f"\n[yellow]4. Validating factor {factor_id[:8]}...:[/yellow]")
        
        validation = validate_factor(factor_id)
        
        if validation['valid']:
            console.print("  [green]✓ Factor is valid[/green]")
        else:
            console.print(f"  [red]✗ Factor is invalid: {validation['reason']}[/red]")
            if 'old_hash' in validation:
                console.print(f"    Old hash: {validation['old_hash']}")
                console.print(f"    New hash: {validation['new_hash']}")
    
    # 5. Find factors for specific symbol
    console.print("\n[yellow]5. Finding all BTC/1m factors:[/yellow]")
    btc_factors = find_factors(symbol='BTC', timeframe='1m')
    
    indicator_counts = {}
    for factor in btc_factors:
        ind_type = factor['indicator_type']
        indicator_counts[ind_type] = indicator_counts.get(ind_type, 0) + 1
    
    for ind_type, count in indicator_counts.items():
        console.print(f"  • {ind_type}: {count} cached variants")
    
    # 6. Check fingerprints for drift detection
    console.print("\n[yellow]6. Checking data fingerprints:[/yellow]")
    
    unique_fingerprints = set()
    for factor in all_factors:
        unique_fingerprints.add(factor['fingerprint'])
    
    console.print(f"  Unique data fingerprints: {len(unique_fingerprints)}")
    
    if len(unique_fingerprints) > 1:
        console.print("  [yellow]Multiple fingerprints detected - possible schema changes[/yellow]")
    else:
        console.print("  [green]All factors using same data structure[/green]")
    
    # 7. Invalidate old factors
    console.print("\n[yellow]7. Checking for stale factors:[/yellow]")
    
    # Just check, don't actually invalidate
    old_count = 0
    for factor in all_factors:
        created = datetime.fromisoformat(str(factor['created_at']))
        age_days = (datetime.now() - created).days
        if age_days > 7:
            old_count += 1
    
    console.print(f"  Factors older than 7 days: {old_count}")
    
    if old_count > 0:
        console.print("  Run `invalidate_stale_factors(7)` to clean up")
    
    # Clean up
    dm.close()


def demo_strategy_usage():
    """Show how strategies can use factor info"""
    console.print("\n[bold cyan]Strategy Usage Example[/bold cyan]\n")
    
    example_code = '''# In your strategy:
from factor_store_enhanced import factor_info, find_factors

# Check what's cached for your symbol
btc_factors = find_factors(symbol='BTC', timeframe='1m')
print(f"Found {len(btc_factors)} cached factors")

# Get info about a specific factor
for factor in btc_factors:
    if factor['indicator_type'] == 'rsi':
        info = factor_info(factor['factor_id'])
        print(f"RSI({info['parameters']['period']}) - {info['rows']} rows, dtype: {info['dtype']}")
        
        # Check if it matches your needs
        if info['parameters']['period'] == 14:
            print("Found cached RSI(14)!")
            print(f"Fingerprint: {info['fingerprint']}")
            print(f"Is partial: {info['is_partial']}")
'''
    
    console.print(example_code)


def main():
    """Run all demos"""
    try:
        demo_factor_info()
        demo_strategy_usage()
        
        # Show final summary
        fs = get_enhanced_factor_store()
        stats = fs.get_cache_stats()
        
        console.print("\n[bold green]Cache Summary:[/bold green]")
        console.print(f"  Total factors: {stats['total_factors']}")
        console.print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        console.print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
        
    finally:
        fs.close()


if __name__ == "__main__":
    main()