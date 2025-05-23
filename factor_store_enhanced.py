#!/usr/bin/env python
"""
Enhanced factor store with robust edge case handling
Addresses: partial candles, parameter collisions, datatype drift
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


def generate_data_fingerprint(df: pd.DataFrame) -> str:
    """Generate a fingerprint of the data structure and types"""
    fingerprint_parts = [
        f"shape:{df.shape}",
        f"index_type:{type(df.index).__name__}",
        f"index_dtype:{df.index.dtype if hasattr(df.index, 'dtype') else 'none'}",
    ]
    
    # Add column information if DataFrame
    if hasattr(df, 'columns'):
        for col in sorted(df.columns):
            fingerprint_parts.append(f"col_{col}:{df[col].dtype}")
    elif isinstance(df, pd.Series):
        fingerprint_parts.append(f"series_dtype:{df.dtype}")
    
    # Create hash of fingerprint
    fingerprint = "|".join(fingerprint_parts)
    return hashlib.md5(fingerprint.encode()).hexdigest()[:16]


def normalize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize parameters to ensure consistent caching"""
    normalized = {}
    
    for key, value in params.items():
        # Handle numeric parameters
        if isinstance(value, (int, float)):
            # Convert floats that are whole numbers to int
            if isinstance(value, float) and value.is_integer():
                normalized[key] = int(value)
            else:
                normalized[key] = value
        # Handle string parameters
        elif isinstance(value, str):
            normalized[key] = value.lower().strip()
        # Handle lists/tuples
        elif isinstance(value, (list, tuple)):
            normalized[key] = tuple(normalize_parameters({'v': v})['v'] for v in value)
        else:
            normalized[key] = value
    
    return normalized


class EnhancedFactorStore:
    """Factor store with enhanced edge case handling"""
    
    def __init__(self, db_path: Optional[str] = None, max_memory_mb: int = 500):
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
            'invalidations': 0,
            'collisions_avoided': 0,
            'partial_candles_handled': 0
        }
        
        # Data fingerprints for drift detection
        self._data_fingerprints = {}
        
        # Setup schema
        self._setup_schema()
        
        # Memory tracking
        self._memory_usage = 0
    
    def _setup_schema(self):
        """Create enhanced schema with additional metadata"""
        with self._lock:
            # Create schema
            self.conn.execute("CREATE SCHEMA IF NOT EXISTS factors")
            
            # Enhanced metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS factors.metadata (
                    factor_id VARCHAR PRIMARY KEY,
                    indicator_type VARCHAR NOT NULL,
                    parameters JSON NOT NULL,
                    normalized_params JSON NOT NULL,
                    symbol VARCHAR NOT NULL,
                    timeframe VARCHAR NOT NULL,
                    data_hash VARCHAR NOT NULL,
                    data_fingerprint VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    size_bytes BIGINT NOT NULL,
                    computation_time_ms DOUBLE NOT NULL,
                    is_partial BOOLEAN DEFAULT FALSE,
                    data_end_time TIMESTAMP
                )
            """)
            
            # Create indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON factors.metadata(symbol, timeframe)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_data_fingerprint ON factors.metadata(data_fingerprint)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_is_partial ON factors.metadata(is_partial)")
    
    def _generate_factor_id(self, indicator_type: str, parameters: Dict[str, Any], 
                          symbol: str, timeframe: str, data_fingerprint: str) -> str:
        """Generate unique factor ID including data fingerprint"""
        # Normalize parameters first
        normalized = normalize_parameters(parameters)
        
        # Include data fingerprint in ID to handle dtype changes
        factor_key = f"{indicator_type}_{json.dumps(normalized, sort_keys=True)}_{symbol}_{timeframe}_{data_fingerprint}"
        return hashlib.md5(factor_key.encode()).hexdigest()[:16]
    
    def _detect_partial_candle(self, df: pd.DataFrame, timeframe: str) -> bool:
        """Detect if the last candle is likely incomplete"""
        if len(df) < 2:
            return False
        
        # Get expected candle duration
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }.get(timeframe, 1)
        
        expected_duration = timedelta(minutes=timeframe_minutes)
        
        # Check last candle
        if hasattr(df.index, 'to_pydatetime'):
            last_time = df.index[-1]
            current_time = datetime.now()
            
            # If last candle is very recent (within the timeframe), it might be incomplete
            if (current_time - last_time) < expected_duration:
                # Additional check: volume comparison
                if 'volume' in df.columns:
                    avg_volume = df['volume'].iloc[-10:-1].mean()
                    last_volume = df['volume'].iloc[-1]
                    
                    # If volume is significantly lower, likely incomplete
                    if last_volume < avg_volume * 0.5:
                        return True
        
        return False
    
    def get_factor(self, 
                   indicator_type: str,
                   parameters: Dict[str, Any],
                   symbol: str,
                   timeframe: str,
                   compute_func: Optional[Callable] = None,
                   data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Enhanced get_factor with edge case handling"""
        
        # Normalize parameters
        normalized_params = normalize_parameters(parameters)
        
        # Generate data fingerprint if data provided
        data_fingerprint = generate_data_fingerprint(data) if data is not None else "unknown"
        
        # Check for partial candle
        is_partial = False
        if data is not None:
            is_partial = self._detect_partial_candle(data, timeframe)
            if is_partial:
                self._cache_stats['partial_candles_handled'] += 1
        
        # Generate factor ID
        factor_id = self._generate_factor_id(
            indicator_type, parameters, symbol, timeframe, data_fingerprint
        )
        
        with self._lock:
            # Check memory cache
            if factor_id in self._memory_cache:
                self._cache_stats['hits'] += 1
                self._update_access_stats(factor_id)
                return self._memory_cache[factor_id].copy()
            
            # Check DuckDB cache
            current_data_hash = self._get_data_hash(symbol, timeframe)
            metadata = self._get_metadata(factor_id)
            
            # Validate cache entry
            cache_valid = (
                metadata and 
                metadata['data_hash'] == current_data_hash and
                metadata['data_fingerprint'] == data_fingerprint and
                not (is_partial and not metadata.get('is_partial', False))
            )
            
            if cache_valid:
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
            
            # If partial candle detected, compute on complete data only
            if is_partial and data is not None:
                data_for_compute = data.iloc[:-1]  # Exclude last candle
                factor_data = compute_func(data_for_compute)
                
                # Pad with NaN for the last value
                if isinstance(factor_data, pd.Series):
                    last_index = data.index[-1]
                    factor_data = factor_data.reindex(data.index)
                    factor_data.loc[last_index] = np.nan
            else:
                factor_data = compute_func()
            
            computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Store in cache
            self._store_factor(
                factor_id=factor_id,
                indicator_type=indicator_type,
                parameters=parameters,
                normalized_params=normalized_params,
                symbol=symbol,
                timeframe=timeframe,
                data_hash=current_data_hash,
                data_fingerprint=data_fingerprint,
                factor_data=factor_data,
                computation_time_ms=computation_time_ms,
                is_partial=is_partial
            )
            
            self._cache_stats['computes'] += 1
            
            return factor_data.copy()
    
    def _store_factor(self, factor_id: str, indicator_type: str, parameters: Dict[str, Any],
                     normalized_params: Dict[str, Any], symbol: str, timeframe: str, 
                     data_hash: str, data_fingerprint: str, factor_data: pd.Series, 
                     computation_time_ms: float, is_partial: bool = False):
        """Store factor with enhanced metadata"""
        # Store data
        self._store_factor_data(factor_id, factor_data)
        
        # Calculate size
        size_bytes = factor_data.memory_usage(deep=True)
        
        # Get data end time
        data_end_time = None
        if hasattr(factor_data.index, 'max'):
            data_end_time = factor_data.index.max()
        
        # Store metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO factors.metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            factor_id,
            indicator_type,
            json.dumps(parameters),
            json.dumps(normalized_params),
            symbol,
            timeframe,
            data_hash,
            data_fingerprint,
            datetime.now(),
            datetime.now(),
            1,
            int(size_bytes),
            computation_time_ms,
            is_partial,
            data_end_time
        ])
        
        # Add to memory cache
        self._add_to_memory_cache(factor_id, factor_data)
    
    def _get_metadata(self, factor_id: str) -> Optional[Dict]:
        """Get enhanced metadata"""
        result = self.conn.execute("""
            SELECT * FROM factors.metadata WHERE factor_id = ?
        """, [factor_id]).fetchone()
        
        if result:
            return {
                'factor_id': result[0],
                'indicator_type': result[1],
                'parameters': json.loads(result[2]),
                'normalized_params': json.loads(result[3]),
                'symbol': result[4],
                'timeframe': result[5],
                'data_hash': result[6],
                'data_fingerprint': result[7],
                'created_at': result[8],
                'last_accessed': result[9],
                'access_count': result[10],
                'size_bytes': result[11],
                'computation_time_ms': result[12],
                'is_partial': result[13],
                'data_end_time': result[14]
            }
        return None
    
    def get_collision_report(self) -> Dict[str, Any]:
        """Report on parameter collision handling"""
        with self._lock:
            # Check for similar parameters
            similar_params = self.conn.execute("""
                SELECT indicator_type, COUNT(*) as variants,
                       COUNT(DISTINCT normalized_params) as unique_normalized
                FROM factors.metadata
                GROUP BY indicator_type
                HAVING variants > 1
            """).fetchall()
            
            return {
                'collisions_avoided': self._cache_stats['collisions_avoided'],
                'partial_candles_handled': self._cache_stats['partial_candles_handled'],
                'parameter_variants': [
                    {
                        'indicator': row[0],
                        'total_variants': row[1],
                        'unique_after_normalization': row[2]
                    }
                    for row in similar_params
                ]
            }
    
    # Include all other methods from original factor_store.py
    # (Not repeating to save space, but they would be included)
    
    def get_factor_info(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cached factor"""
        with self._lock:
            # Get metadata
            metadata = self._get_metadata(factor_id)
            if not metadata:
                return None
            
            # Get factor data stats
            table_name = f"factors.factor_{factor_id}"
            
            try:
                # Check if table exists
                if not self._table_exists(f"factor_{factor_id}"):
                    return None
                
                # Get data statistics
                stats = self.conn.execute(f"""
                    SELECT 
                        COUNT(*) as rows,
                        COUNT(*) - COUNT(value) as null_count,
                        MIN(ts) as min_ts,
                        MAX(ts) as max_ts,
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value
                    FROM {table_name}
                """).fetchone()
                
                # Load a sample to check dtype
                sample = self.conn.execute(f"""
                    SELECT value FROM {table_name} 
                    WHERE value IS NOT NULL 
                    LIMIT 1
                """).fetchone()
                
                dtype = 'unknown'
                if sample:
                    # Infer dtype from value
                    val = sample[0]
                    if isinstance(val, float):
                        dtype = 'float64'
                    elif isinstance(val, int):
                        dtype = 'int64'
                
                return {
                    'factor_id': factor_id,
                    'indicator_type': metadata['indicator_type'],
                    'parameters': metadata['parameters'],
                    'normalized_params': metadata['normalized_params'],
                    'symbol': metadata['symbol'],
                    'timeframe': metadata['timeframe'],
                    'rows': stats[0],
                    'null_count': stats[1],
                    'dtype': dtype,
                    'fingerprint': metadata['data_fingerprint'],
                    'data_hash': metadata['data_hash'],
                    'is_partial': metadata.get('is_partial', False),
                    'created_at': metadata['created_at'],
                    'last_accessed': metadata['last_accessed'],
                    'access_count': metadata['access_count'],
                    'size_bytes': metadata['size_bytes'],
                    'computation_time_ms': metadata['computation_time_ms'],
                    'data_range': {
                        'start': datetime.fromtimestamp(stats[2]/1000) if stats[2] else None,
                        'end': datetime.fromtimestamp(stats[3]/1000) if stats[3] else None
                    },
                    'value_stats': {
                        'avg': stats[4],
                        'min': stats[5],
                        'max': stats[6]
                    }
                }
                
            except Exception as e:
                # Return basic metadata if stats fail
                return {
                    'factor_id': factor_id,
                    'indicator_type': metadata['indicator_type'],
                    'parameters': metadata['parameters'],
                    'symbol': metadata['symbol'],
                    'timeframe': metadata['timeframe'],
                    'fingerprint': metadata['data_fingerprint'],
                    'is_partial': metadata.get('is_partial', False),
                    'error': str(e)
                }
    
    def find_factors(self, indicator_type: Optional[str] = None, 
                    symbol: Optional[str] = None,
                    timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find cached factors matching criteria"""
        with self._lock:
            conditions = []
            params = []
            
            if indicator_type:
                conditions.append("indicator_type = ?")
                params.append(indicator_type)
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if timeframe:
                conditions.append("timeframe = ?")
                params.append(timeframe)
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            results = self.conn.execute(f"""
                SELECT factor_id, indicator_type, parameters, symbol, timeframe,
                       data_fingerprint, is_partial, created_at, access_count
                FROM factors.metadata
                {where_clause}
                ORDER BY last_accessed DESC
            """, params).fetchall()
            
            return [
                {
                    'factor_id': row[0],
                    'indicator_type': row[1],
                    'parameters': json.loads(row[2]),
                    'symbol': row[3],
                    'timeframe': row[4],
                    'fingerprint': row[5],
                    'is_partial': row[6],
                    'created_at': row[7],
                    'access_count': row[8]
                }
                for row in results
            ]
    
    def validate_factor(self, factor_id: str, current_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Validate if a cached factor is still valid"""
        with self._lock:
            info = self.get_factor_info(factor_id)
            if not info:
                return {'valid': False, 'reason': 'Factor not found'}
            
            # Check data hash
            current_hash = self._get_data_hash(info['symbol'], info['timeframe'])
            if current_hash != info['data_hash']:
                return {
                    'valid': False,
                    'reason': 'Data has changed',
                    'old_hash': info['data_hash'],
                    'new_hash': current_hash
                }
            
            # Check fingerprint if data provided
            if current_data is not None:
                current_fingerprint = generate_data_fingerprint(current_data)
                if current_fingerprint != info['fingerprint']:
                    return {
                        'valid': False,
                        'reason': 'Data structure has changed',
                        'old_fingerprint': info['fingerprint'],
                        'new_fingerprint': current_fingerprint
                    }
            
            # Check if partial flag matches current state
            if current_data is not None and info['is_partial']:
                is_still_partial = self._detect_partial_candle(current_data, info['timeframe'])
                if not is_still_partial:
                    return {
                        'valid': False,
                        'reason': 'Partial candle is now complete'
                    }
            
            return {'valid': True, 'info': info}
    
    def invalidate_stale_factors(self, max_age_days: int = 30) -> int:
        """Invalidate factors older than specified age"""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=max_age_days)
            
            stale_factors = self.conn.execute("""
                SELECT factor_id FROM factors.metadata
                WHERE last_accessed < ?
            """, [cutoff]).fetchall()
            
            count = 0
            for (factor_id,) in stale_factors:
                self._invalidate_factor(factor_id)
                count += 1
            
            return count
    
    def _invalidate_factor(self, factor_id: str):
        """Invalidate a single factor"""
        # Remove from memory cache
        if factor_id in self._memory_cache:
            self._memory_cache.pop(factor_id)
        
        # Mark as invalidated
        self.conn.execute("""
            UPDATE factors.metadata 
            SET data_hash = 'invalidated'
            WHERE factor_id = ?
        """, [factor_id])
        
        self._cache_stats['invalidations'] += 1
    
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
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Wrapper to replace original factor store
def get_enhanced_factor_store() -> EnhancedFactorStore:
    """Get enhanced factor store instance"""
    global _global_factor_store
    if _global_factor_store is None or not isinstance(_global_factor_store, EnhancedFactorStore):
        _global_factor_store = EnhancedFactorStore()
    return _global_factor_store


# Convenience helper functions
def factor_info(factor_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a cached factor"""
    fs = get_enhanced_factor_store()
    return fs.get_factor_info(factor_id)


def find_factors(indicator_type: Optional[str] = None, 
                symbol: Optional[str] = None,
                timeframe: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find cached factors matching criteria"""
    fs = get_enhanced_factor_store()
    return fs.find_factors(indicator_type, symbol, timeframe)


def validate_factor(factor_id: str, current_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Validate if a cached factor is still valid"""
    fs = get_enhanced_factor_store()
    return fs.validate_factor(factor_id, current_data)


def invalidate_stale_factors(max_age_days: int = 30) -> int:
    """Invalidate factors older than specified age"""
    fs = get_enhanced_factor_store()
    return fs.invalidate_stale_factors(max_age_days)


# For backward compatibility
_global_factor_store = None