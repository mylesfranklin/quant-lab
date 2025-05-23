# QuantLab Project Memory

## Project Overview
QuantLab is a high-performance cryptocurrency backtesting framework focused on speed and scalability. The system features a three-tier factor caching architecture that achieves 10x+ performance improvements for grid optimization tasks.

## Architecture

### Core Components
1. **Factor Store** (`factor_store.py`, `factor_store_enhanced.py`)
   - Three-tier caching: Memory LRU → DuckDB → Compute
   - Thread-safe with RLock implementation
   - Automatic cache invalidation via data fingerprinting
   - Handles edge cases: partial candles, parameter normalization, datatype drift

2. **Cached Indicators** (`cached_indicators.py`)
   - 20+ pre-built technical indicators with automatic caching
   - Decorator-based implementation for easy extension
   - VectorBT integration with proper type handling

3. **Strategy Framework** (`src/strategies/`)
   - `BaseStrategy` abstract class for all strategies
   - Vectorized signal generation
   - Built-in performance metrics

4. **Data Management** (`src/data/`)
   - DuckDB-based persistent storage
   - Incremental data fetching
   - Efficient OHLCV data handling

## Coding Standards

### Python Style
- Use type hints for all function signatures
- Docstrings for all public methods and classes
- NO comments in code unless explicitly requested
- Prefer descriptive variable names over comments

### Import Organization
```python
# Standard library
import os
import sys
from datetime import datetime

# Third-party
import pandas as pd
import numpy as np

# Local imports
from src.backtesting.factor_store import FactorStore
```

### Error Handling
- Use try-except blocks for external operations (DB, file I/O)
- Log errors appropriately, don't print
- Raise specific exceptions with clear messages

### Performance Patterns
- Vectorize operations using pandas/numpy
- Batch database operations
- Use concurrent operations where possible
- Pre-compute indicators once, reuse across parameter sweeps

## Key Workflows

### 1. Adding a New Cached Indicator
```python
from src.backtesting.cached_indicators import cached_indicator

@cached_indicator('my_indicator')
def my_indicator(data: pd.DataFrame, param1: int = 10) -> pd.Series:
    # Ensure integer parameters for vectorbt
    param1 = int(param1)
    # Compute and return indicator
    return result
```

### 2. Creating a Strategy
```python
from src.strategies.base_strategy import BaseStrategy
from src.backtesting.cached_indicators import ma, rsi

class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Use cached indicators
        ma_values = ma(data, period=20)
        rsi_values = rsi(data, period=14)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        # ... signal logic ...
        return signals
```

### 3. Checking Factor Information
```python
from src.backtesting.factor_store_enhanced import factor_info

# Get factor details
info = factor_info("ma_BTC/USDT_1h_{'period': 20}")
if info and info.get('is_partial'):
    print("Warning: Factor has partial data!")
```

## Testing Commands
Always run these before committing:
```bash
# Run tests
python -m pytest tests/

# Type checking (if mypy is set up)
mypy src/

# Linting (if configured)
ruff check .
```

## Performance Targets
- Single backtest: < 100ms
- Grid optimization (100 parameters): 1-2 seconds (down from 10-20s)
- Factor cache hit rate: > 90% after warmup
- Memory usage: < 1GB for typical sessions

## Edge Cases to Consider
1. **Partial Window Candles**: Last candle might be incomplete in live data
2. **Parameter Collisions**: 14 vs 14.0 should map to same factor
3. **Datatype Drift**: Float64 vs Float32 differences
4. **Thread Safety**: Multiple strategies accessing cache concurrently
5. **Cache Invalidation**: Data updates should invalidate stale factors

## Database Schema

### DuckDB Tables
```sql
-- OHLCV data
CREATE TABLE ohlcv_data (
    symbol VARCHAR,
    timeframe VARCHAR,
    timestamp TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

-- Factor metadata
CREATE TABLE factor_metadata (
    factor_id VARCHAR PRIMARY KEY,
    indicator_type VARCHAR,
    parameters JSON,
    symbol VARCHAR,
    timeframe VARCHAR,
    created_at TIMESTAMP,
    rows INTEGER,
    dtype VARCHAR,
    data_fingerprint VARCHAR,
    is_partial BOOLEAN
);

-- Factor data
CREATE TABLE factor_data (
    factor_id VARCHAR,
    timestamp TIMESTAMP,
    value DOUBLE,
    PRIMARY KEY (factor_id, timestamp)
);
```

## File Structure
```
quant-lab/
├── src/
│   ├── backtesting/
│   │   ├── factor_store.py          # Core caching engine
│   │   ├── factor_store_enhanced.py # Enhanced with edge cases
│   │   ├── cached_indicators.py     # Pre-built indicators
│   │   └── factor_utils.py          # High-level utilities
│   ├── strategies/
│   │   ├── base_strategy.py         # Abstract base class
│   │   └── ma_cross_with_cache.py   # Example strategy
│   └── data/
│       ├── data_manager.py          # DuckDB interface
│       └── fetcher.py               # Data fetching logic
├── tests/
│   └── test_edge_cases.py           # Edge case validation
├── cache/                            # DuckDB storage
└── logs/                             # Application logs
```

## Common Issues & Solutions

### Issue: Float parameters causing cache misses
**Solution**: Always cast numeric parameters to int in indicators
```python
period = int(period)  # Ensure integer
```

### Issue: Stale factors after data update
**Solution**: Use data fingerprinting and validate before use
```python
if not validate_factor(factor_id, current_data):
    # Recompute factor
```

### Issue: Memory growth with large datasets
**Solution**: LRU cache automatically evicts old entries, configure max_size appropriately

## Future Enhancements
1. Redis integration for distributed caching
2. GPU acceleration for indicator computation
3. Real-time streaming data support
4. Multi-timeframe factor relationships
5. Automatic hyperparameter optimization

## Quick Reference

### Import Factor Info in Any Strategy
```python
from src.backtesting.factor_store_enhanced import factor_info
```

### Clean Up Old Factors
```python
from src.backtesting.factor_utils import FactorCache
removed = FactorCache.cleanup(days=7)
```

### Get Cache Statistics
```python
from src.backtesting.factor_utils import print_factor_report
print_factor_report('BTC/USDT', '1h')
```