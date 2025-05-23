#!/usr/bin/env python
"""
High-performance factor caching system with three-tier caching:
Memory (LRU) -> DuckDB -> Compute
"""
import hashlib
import json
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Callable, Any, Tuple, Union
import pandas as pd
import numpy as np
import duckdb
from dataclasses import dataclass, asdict
import pickle
import zlib


@dataclass
class FactorMetadata:
    """Metadata for cached factors"""
    factor_id: str
    indicator_type: str
    parameters: Dict[str, Any]
    symbol: str
    timeframe: str
    data_hash: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    computation_time_ms: float
    
    def to_dict(self):
        d = asdict(self)
        d['parameters'] = json.dumps(d['parameters'])
        return d


class FactorStore:
    """
    Thread-safe factor caching system with automatic invalidation
    """
    
    def __init__(self, db_path: Optional[str] = None, max_memory_mb: int = 500):
        """
        Initialize factor store
        
        Args:
            db_path: Path to DuckDB database (uses default if None)
            max_memory_mb: Maximum memory cache size in MB
        """
        if db_path is None:
            db_path = "data/quant_lab.duckdb"
        
        self.db_path = db_path
        self.max_memory_mb = max_memory_mb
        self.conn = duckdb.connect(db_path)
        
        # Thread safety
        self._lock = threading.RLock()
        self._memory_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'computes': 0,
            'invalidations': 0
        }
        
        # Setup schema
        self._setup_schema()
        
        # Memory tracking
        self._memory_usage = 0
        
    def _setup_schema(self):
        """Create factor caching schema in DuckDB"""
        with self._lock:
            # Create schema
            self.conn.execute("CREATE SCHEMA IF NOT EXISTS factors")
            
            # Factor metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS factors.metadata (
                    factor_id VARCHAR PRIMARY KEY,
                    indicator_type VARCHAR NOT NULL,
                    parameters JSON NOT NULL,
                    symbol VARCHAR NOT NULL,
                    timeframe VARCHAR NOT NULL,
                    data_hash VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    size_bytes BIGINT NOT NULL,
                    computation_time_ms DOUBLE NOT NULL
                )
            """)
            
            # Create indexes separately
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON factors.metadata(symbol, timeframe)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON factors.metadata(last_accessed)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_indicator_type ON factors.metadata(indicator_type)")
            
            # Cache configuration
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS factors.cache_config (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR NOT NULL
                )
            """)
            
            # Set default config
            self.conn.execute("""
                INSERT OR REPLACE INTO factors.cache_config VALUES
                ('max_memory_mb', ?),
                ('ttl_days', '30'),
                ('compression_enabled', 'true')
            """, [str(self.max_memory_mb)])
            
            # Create factor data tables will be done dynamically
            
    def _get_data_hash(self, symbol: str, timeframe: str) -> str:
        """Get hash of underlying data to detect changes"""
        try:
            result = self.conn.execute("""
                SELECT 
                    COUNT(*) as count,
                    MIN(ts) as min_ts,
                    MAX(ts) as max_ts,
                    SUM(close) as sum_close
                FROM prices
                WHERE symbol = ? AND timeframe = ?
            """, [symbol, timeframe]).fetchone()
            
            if result and result[0] > 0:
                # Create hash from data characteristics
                hash_input = f"{result[0]}_{result[1]}_{result[2]}_{result[3]:.6f}"
                return hashlib.md5(hash_input.encode()).hexdigest()[:16]
            return "no_data"
        except:
            return "error"
    
    def _generate_factor_id(self, indicator_type: str, parameters: Dict[str, Any], 
                          symbol: str, timeframe: str) -> str:
        """Generate unique factor ID"""
        # Sort parameters for consistent hashing
        param_str = json.dumps(parameters, sort_keys=True)
        factor_key = f"{indicator_type}_{param_str}_{symbol}_{timeframe}"
        return hashlib.md5(factor_key.encode()).hexdigest()[:16]
    
    def _compress_data(self, df: pd.DataFrame) -> bytes:
        """Compress DataFrame for storage"""
        return zlib.compress(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))
    
    def _decompress_data(self, data: bytes) -> pd.DataFrame:
        """Decompress stored DataFrame"""
        return pickle.loads(zlib.decompress(data))
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name]
            ).fetchone()
            return result[0] > 0
        except:
            return False
    
    def _create_factor_table(self, factor_id: str):
        """Create table for storing factor data"""
        table_name = f"factors.factor_{factor_id}"
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ts BIGINT PRIMARY KEY,
                value DOUBLE
            )
        """)
    
    def _store_factor_data(self, factor_id: str, df: pd.DataFrame):
        """Store factor data in DuckDB"""
        table_name = f"factors.factor_{factor_id}"
        
        # Create table if needed
        self._create_factor_table(factor_id)
        
        # Clear existing data
        self.conn.execute(f"DELETE FROM {table_name}")
        
        # Prepare data
        if isinstance(df, pd.Series):
            df = df.to_frame(name='value')
        else:
            df = df.copy()
        
        # Ensure we have ts index
        if 'ts' not in df.columns:
            if hasattr(df.index, 'astype'):
                # Convert datetime index to timestamp
                df['ts'] = df.index.astype(np.int64) // 10**6  # Convert to milliseconds
            else:
                raise ValueError("DataFrame must have timestamp index")
        
        # Select only ts and value columns, ensuring proper order
        if 'value' in df.columns:
            df_to_insert = df[['ts', 'value']]
        else:
            # If it's a Series converted to DataFrame, the column might have a different name
            df_to_insert = df.copy()
            df_to_insert.columns = ['value']
            df_to_insert['ts'] = df['ts'] if 'ts' in df.columns else df.index.astype(np.int64) // 10**6
            df_to_insert = df_to_insert[['ts', 'value']]
        
        # Insert data (DuckDB handles NaN as NULL)
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df_to_insert")
    
    def _load_factor_data(self, factor_id: str) -> Optional[pd.DataFrame]:
        """Load factor data from DuckDB"""
        table_name = f"factors.factor_{factor_id}"
        
        if not self._table_exists(f"factor_{factor_id}"):
            return None
        
        try:
            df = self.conn.execute(f"""
                SELECT ts, value 
                FROM {table_name}
                ORDER BY ts
            """).fetchdf()
            
            if len(df) == 0:
                return None
            
            # Convert ts back to datetime index
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.set_index('datetime')['value']
            return df
            
        except Exception as e:
            print(f"Error loading factor {factor_id}: {e}")
            return None
    
    def get_factor(self, 
                   indicator_type: str,
                   parameters: Dict[str, Any],
                   symbol: str,
                   timeframe: str,
                   compute_func: Optional[Callable] = None) -> pd.Series:
        """
        Get factor from cache or compute if needed
        
        Args:
            indicator_type: Type of indicator (e.g., 'ema', 'rsi')
            parameters: Indicator parameters
            symbol: Trading symbol
            timeframe: Data timeframe
            compute_func: Function to compute factor if not cached
            
        Returns:
            Factor data as pandas Series
        """
        factor_id = self._generate_factor_id(indicator_type, parameters, symbol, timeframe)
        
        with self._lock:
            # Check memory cache
            if factor_id in self._memory_cache:
                self._cache_stats['hits'] += 1
                self._update_access_stats(factor_id)
                return self._memory_cache[factor_id].copy()
            
            # Check DuckDB cache
            current_data_hash = self._get_data_hash(symbol, timeframe)
            metadata = self._get_metadata(factor_id)
            
            if metadata and metadata.data_hash == current_data_hash:
                # Load from DuckDB
                factor_data = self._load_factor_data(factor_id)
                if factor_data is not None:
                    self._cache_stats['hits'] += 1
                    self._update_access_stats(factor_id)
                    
                    # Add to memory cache
                    self._add_to_memory_cache(factor_id, factor_data)
                    
                    return factor_data.copy()
            
            # Need to compute
            self._cache_stats['misses'] += 1
            
            if compute_func is None:
                raise ValueError(f"No compute function provided for {indicator_type}")
            
            # Compute factor
            start_time = datetime.now()
            factor_data = compute_func()
            computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Store in cache
            self._store_factor(
                factor_id=factor_id,
                indicator_type=indicator_type,
                parameters=parameters,
                symbol=symbol,
                timeframe=timeframe,
                data_hash=current_data_hash,
                factor_data=factor_data,
                computation_time_ms=computation_time_ms
            )
            
            self._cache_stats['computes'] += 1
            
            return factor_data.copy()
    
    def _store_factor(self, factor_id: str, indicator_type: str, parameters: Dict[str, Any],
                     symbol: str, timeframe: str, data_hash: str, 
                     factor_data: pd.Series, computation_time_ms: float):
        """Store factor in both DuckDB and memory cache"""
        # Store data
        self._store_factor_data(factor_id, factor_data)
        
        # Calculate size
        size_bytes = factor_data.memory_usage(deep=True)
        
        # Store metadata
        metadata = FactorMetadata(
            factor_id=factor_id,
            indicator_type=indicator_type,
            parameters=parameters,
            symbol=symbol,
            timeframe=timeframe,
            data_hash=data_hash,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            size_bytes=int(size_bytes),
            computation_time_ms=computation_time_ms
        )
        
        self._store_metadata(metadata)
        
        # Add to memory cache
        self._add_to_memory_cache(factor_id, factor_data)
    
    def _store_metadata(self, metadata: FactorMetadata):
        """Store factor metadata"""
        self.conn.execute("""
            INSERT OR REPLACE INTO factors.metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            metadata.factor_id,
            metadata.indicator_type,
            json.dumps(metadata.parameters),
            metadata.symbol,
            metadata.timeframe,
            metadata.data_hash,
            metadata.created_at,
            metadata.last_accessed,
            metadata.access_count,
            metadata.size_bytes,
            metadata.computation_time_ms
        ])
    
    def _get_metadata(self, factor_id: str) -> Optional[FactorMetadata]:
        """Get factor metadata"""
        result = self.conn.execute("""
            SELECT * FROM factors.metadata WHERE factor_id = ?
        """, [factor_id]).fetchone()
        
        if result:
            return FactorMetadata(
                factor_id=result[0],
                indicator_type=result[1],
                parameters=json.loads(result[2]),
                symbol=result[3],
                timeframe=result[4],
                data_hash=result[5],
                created_at=result[6],
                last_accessed=result[7],
                access_count=result[8],
                size_bytes=result[9],
                computation_time_ms=result[10]
            )
        return None
    
    def _update_access_stats(self, factor_id: str):
        """Update access statistics for a factor"""
        self.conn.execute("""
            UPDATE factors.metadata 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE factor_id = ?
        """, [datetime.now(), factor_id])
    
    def _add_to_memory_cache(self, factor_id: str, factor_data: pd.Series):
        """Add factor to memory cache with size management"""
        # Calculate size
        size_mb = factor_data.memory_usage(deep=True) / (1024 * 1024)
        
        # Check if we need to evict
        while self._memory_usage + size_mb > self.max_memory_mb and self._memory_cache:
            # Evict least recently used
            self._evict_from_memory()
        
        # Add to cache
        self._memory_cache[factor_id] = factor_data.copy()
        self._memory_usage += size_mb
    
    def _evict_from_memory(self):
        """Evict least recently used factor from memory"""
        if not self._memory_cache:
            return
        
        # Get access times
        access_times = self.conn.execute("""
            SELECT factor_id, last_accessed 
            FROM factors.metadata 
            WHERE factor_id IN ({})
            ORDER BY last_accessed ASC
            LIMIT 1
        """.format(','.join(['?'] * len(self._memory_cache))), 
            list(self._memory_cache.keys())
        ).fetchone()
        
        if access_times:
            factor_id = access_times[0]
            if factor_id in self._memory_cache:
                factor_data = self._memory_cache.pop(factor_id)
                size_mb = factor_data.memory_usage(deep=True) / (1024 * 1024)
                self._memory_usage -= size_mb
    
    def invalidate_factors(self, symbol: str, timeframe: str):
        """Invalidate all factors for a symbol/timeframe when data updates"""
        with self._lock:
            # Get affected factors
            affected = self.conn.execute("""
                SELECT factor_id 
                FROM factors.metadata 
                WHERE symbol = ? AND timeframe = ?
            """, [symbol, timeframe]).fetchall()
            
            for (factor_id,) in affected:
                # Remove from memory cache
                if factor_id in self._memory_cache:
                    factor_data = self._memory_cache.pop(factor_id)
                    size_mb = factor_data.memory_usage(deep=True) / (1024 * 1024)
                    self._memory_usage -= size_mb
                
                # Mark as invalidated (by setting data_hash to 'invalidated')
                self.conn.execute("""
                    UPDATE factors.metadata 
                    SET data_hash = 'invalidated'
                    WHERE factor_id = ?
                """, [factor_id])
            
            self._cache_stats['invalidations'] += len(affected)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            # Basic stats
            stats = self._cache_stats.copy()
            
            # Hit rate
            total_requests = stats['hits'] + stats['misses']
            stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0
            
            # Memory usage
            stats['memory_usage_mb'] = self._memory_usage
            stats['memory_cache_size'] = len(self._memory_cache)
            
            # DuckDB stats
            db_stats = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_factors,
                    SUM(size_bytes) / 1024.0 / 1024.0 as total_size_mb,
                    AVG(computation_time_ms) as avg_compute_time_ms,
                    SUM(access_count) as total_accesses
                FROM factors.metadata
                WHERE data_hash != 'invalidated'
            """).fetchone()
            
            stats['total_factors'] = db_stats[0] or 0
            stats['disk_usage_mb'] = db_stats[1] or 0
            stats['avg_compute_time_ms'] = db_stats[2] or 0
            stats['total_accesses'] = db_stats[3] or 0
            
            # Popular factors
            popular = self.conn.execute("""
                SELECT indicator_type, COUNT(*) as count, AVG(access_count) as avg_accesses
                FROM factors.metadata
                WHERE data_hash != 'invalidated'
                GROUP BY indicator_type
                ORDER BY avg_accesses DESC
                LIMIT 5
            """).fetchall()
            
            stats['popular_indicators'] = [
                {'type': row[0], 'count': row[1], 'avg_accesses': row[2]}
                for row in popular
            ]
            
            return stats
    
    def cleanup_old_factors(self, days: int = 30):
        """Remove factors not accessed in the specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            # Get factors to remove
            old_factors = self.conn.execute("""
                SELECT factor_id, size_bytes
                FROM factors.metadata
                WHERE last_accessed < ?
            """, [cutoff_date]).fetchall()
            
            removed_count = 0
            removed_bytes = 0
            
            for factor_id, size_bytes in old_factors:
                # Remove from memory cache
                if factor_id in self._memory_cache:
                    self._memory_cache.pop(factor_id)
                
                # Drop factor table
                table_name = f"factors.factor_{factor_id}"
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Remove metadata
                self.conn.execute("""
                    DELETE FROM factors.metadata WHERE factor_id = ?
                """, [factor_id])
                
                removed_count += 1
                removed_bytes += size_bytes
            
            return {
                'removed_count': removed_count,
                'freed_mb': removed_bytes / 1024 / 1024
            }
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Global instance for convenience
_global_factor_store = None

def get_factor_store() -> FactorStore:
    """Get global factor store instance"""
    global _global_factor_store
    if _global_factor_store is None:
        _global_factor_store = FactorStore()
    return _global_factor_store