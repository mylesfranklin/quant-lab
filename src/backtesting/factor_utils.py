"""
Utility module for factor store access.
Import this in strategies to get easy access to factor information.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from src.backtesting.factor_store_enhanced import (
    get_enhanced_factor_store,
    factor_info,
    find_factors,
    validate_factor,
    invalidate_stale_factors
)


class FactorCache:
    """High-level interface for factor caching."""
    
    @staticmethod
    def info(factor_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a cached factor."""
        return factor_info(factor_id)
    
    @staticmethod
    def search(indicator_type: str, symbol: str = None, timeframe: str = None) -> List[str]:
        """Search for cached factors matching criteria."""
        return find_factors(indicator_type, symbol, timeframe)
    
    @staticmethod
    def validate(factor_id: str, current_data: pd.DataFrame) -> bool:
        """Check if a cached factor is still valid."""
        return validate_factor(factor_id, current_data)
    
    @staticmethod
    def cleanup(days: int = 7) -> int:
        """Remove factors older than specified days."""
        return invalidate_stale_factors(days)
    
    @staticmethod
    def get_factor_summary(symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get a summary of all cached factors for a symbol/timeframe pair."""
        summary = {
            'total_factors': 0,
            'by_indicator': {},
            'partial_factors': 0,
            'stale_factors': 0,
            'memory_cached': 0,
            'disk_cached': 0
        }
        
        # Get factor store instance
        fs = get_enhanced_factor_store()
        
        # Search all indicator types
        indicator_types = ['ma', 'ema', 'rsi', 'bb', 'macd', 'atr', 'adx', 
                          'stoch', 'cci', 'williams_r', 'mfi', 'obv']
        
        for indicator in indicator_types:
            factors = find_factors(indicator, symbol, timeframe)
            if factors:
                summary['by_indicator'][indicator] = len(factors)
                summary['total_factors'] += len(factors)
                
                for factor_id in factors:
                    info = factor_info(factor_id)
                    if info:
                        if info.get('is_partial'):
                            summary['partial_factors'] += 1
                        
                        # Check if in memory cache
                        if factor_id in fs.memory_cache:
                            summary['memory_cached'] += 1
                        else:
                            summary['disk_cached'] += 1
        
        return summary
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get overall cache statistics."""
        fs = get_enhanced_factor_store()
        
        # Get DuckDB stats
        conn = fs._get_connection()
        cursor = conn.execute("SELECT COUNT(*) as total, SUM(rows) as total_rows FROM factor_metadata")
        db_stats = cursor.fetchone()
        
        stats = {
            'memory_cache_size': len(fs.memory_cache),
            'memory_cache_hits': getattr(fs.memory_cache, 'hits', 0),
            'memory_cache_misses': getattr(fs.memory_cache, 'misses', 0),
            'db_total_factors': db_stats[0] if db_stats else 0,
            'db_total_rows': db_stats[1] if db_stats else 0,
            'cache_hit_rate': 0.0
        }
        
        # Calculate hit rate
        total_accesses = stats['memory_cache_hits'] + stats['memory_cache_misses']
        if total_accesses > 0:
            stats['cache_hit_rate'] = stats['memory_cache_hits'] / total_accesses
        
        return stats
    
    @staticmethod
    def warm_cache(symbol: str, timeframe: str, indicators: List[Tuple[str, Dict[str, Any]]]) -> int:
        """Pre-warm the cache with specific indicators."""
        fs = get_enhanced_factor_store()
        warmed = 0
        
        for indicator_type, params in indicators:
            factor_id = fs._get_factor_id(indicator_type, params, symbol, timeframe)
            
            # Check if already cached
            info = factor_info(factor_id)
            if not info:
                # Would need to compute - skip for now
                continue
                
            # Load into memory cache if not already there
            if factor_id not in fs.memory_cache:
                factor_data = fs._load_from_db(factor_id)
                if factor_data is not None:
                    fs.memory_cache[factor_id] = factor_data
                    warmed += 1
        
        return warmed


def print_factor_report(symbol: str, timeframe: str):
    """Print a detailed report about cached factors."""
    summary = FactorCache.get_factor_summary(symbol, timeframe)
    stats = FactorCache.get_cache_stats()
    
    print(f"\n{'='*50}")
    print(f"Factor Cache Report for {symbol} {timeframe}")
    print(f"{'='*50}")
    
    print(f"\nTotal Cached Factors: {summary['total_factors']}")
    print(f"Partial Factors: {summary['partial_factors']}")
    print(f"Memory Cached: {summary['memory_cached']}")
    print(f"Disk Cached: {summary['disk_cached']}")
    
    if summary['by_indicator']:
        print("\nFactors by Indicator:")
        for indicator, count in sorted(summary['by_indicator'].items()):
            print(f"  - {indicator}: {count}")
    
    print(f"\n{'='*50}")
    print("Overall Cache Statistics")
    print(f"{'='*50}")
    
    print(f"Memory Cache Size: {stats['memory_cache_size']}")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    print(f"Total DB Factors: {stats['db_total_factors']}")
    print(f"Total DB Rows: {stats['db_total_rows']:,}")


# Convenience functions for strategies
def get_factor_age(factor_id: str) -> Optional[timedelta]:
    """Get the age of a cached factor."""
    info = factor_info(factor_id)
    if info and 'created_at' in info:
        created = datetime.fromisoformat(info['created_at'])
        return datetime.now() - created
    return None


def is_factor_fresh(factor_id: str, max_age_hours: int = 24) -> bool:
    """Check if a factor is fresh enough to use."""
    age = get_factor_age(factor_id)
    if age is None:
        return False
    return age < timedelta(hours=max_age_hours)


def get_newest_factor(indicator_type: str, symbol: str, timeframe: str) -> Optional[str]:
    """Get the most recently created factor of a given type."""
    factors = find_factors(indicator_type, symbol, timeframe)
    if not factors:
        return None
    
    newest_id = None
    newest_time = None
    
    for factor_id in factors:
        info = factor_info(factor_id)
        if info and 'created_at' in info:
            created = datetime.fromisoformat(info['created_at'])
            if newest_time is None or created > newest_time:
                newest_time = created
                newest_id = factor_id
    
    return newest_id


# Export main interface
__all__ = [
    'FactorCache',
    'factor_info',
    'find_factors', 
    'validate_factor',
    'invalidate_stale_factors',
    'print_factor_report',
    'get_factor_age',
    'is_factor_fresh',
    'get_newest_factor'
]