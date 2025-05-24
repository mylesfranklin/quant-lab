QuantLab - High-Performance Backtesting Framework

A production-ready quantitative trading framework featuring intelligent factor caching, vectorized operations, and multi-strategy optimization.

Features

- **Factor Caching System**: DuckDB-powered caching with automatic invalidation and fingerprinting
- **Vectorized Backtesting**: High-performance backtesting with VectorBT
- **Multi-Strategy Support**: Batch testing and parameter optimization
- **CLI Interface**: Command-line tools for data fetching, backtesting, and analysis
- **Edge Case Handling**: Robust handling of partial windows, data gaps, and type mismatches

## Project Structure

```
quant-lab/
├── .venv/                    # Python virtual environment
├── data/                     # Market data storage
│   ├── BTC_1m.parquet       # Sample Bitcoin data
│   ├── quant_lab.duckdb     # Factor cache database
│   └── metadata.json        # Data metadata
├── seeds/                    # Strategy templates
│   ├── base_strategy.py     # Base strategy class
│   ├── bollinger_obv_breakout.py
│   ├── ema_atr_mean_reversion.py
│   └── macd_rsi_trend_follower.py
├── src/                      # Source modules
│   ├── backtesting/         # Backtesting components
│   └── strategies/          # Strategy implementations
├── results/                  # Backtest outputs
├── scripts/                  # Utility scripts
│
# Core Modules
├── backtester_cached.py     # Cached backtesting engine
├── factor_store_enhanced.py # Enhanced factor caching
├── data_manager.py          # Data management utilities
├── strategy_framework_cached.py # Cached strategy framework
├── batch_backtest_vectorized.py # Vectorized batch testing
└── cli.py                   # Command-line interface
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quant-lab.git
cd quant-lab
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Fetch Market Data
```bash
# Fetch 7 days of BTC data
python fetch_data_incremental.py --symbol BTC/USDT --days 7

# Or use the CLI
python cli.py fetch-data --symbol ETH/USDT --days 30
```

### 2. Run Example Strategy
```bash
# Run a single strategy
python example_strategy_clean.py

# Run batch backtesting
python batch_backtest_vectorized.py
```

### 3. Use the CLI
```bash
# Run backtest
python cli.py backtest --strategy MA_Cross --fast 10 --slow 30

# Analyze results
python cli.py analyze --input results/metrics_vectorized.json
```

## Factor Caching System

The framework includes an intelligent caching system that dramatically improves performance:

```python
from factor_store_enhanced import FactorStore

# Initialize factor store
fs = FactorStore()

# Cache computation automatically
df = fs.get_or_compute(
    "rsi_14",
    lambda: talib.RSI(price_data, timeperiod=14),
    dependencies={'symbol': 'BTC/USDT', 'timeperiod': 14}
)

# Check factor information
info = fs.factor_info("rsi_14")
print(f"Cached factor: {info['rows']} rows, fingerprint: {info['fingerprint']}")
```

## Strategy Development

### Creating a New Strategy

```python
from strategy_framework_cached import CachedStrategy

class MyStrategy(CachedStrategy):
    # Define parameters
    param_grid = {
        'rsi_period': [14, 21, 28],
        'rsi_oversold': [20, 25, 30],
        'rsi_overbought': [70, 75, 80]
    }
    
    @classmethod
    def entries(cls, price):
        rsi = cls.get_cached_indicator('rsi', price, cls.rsi_period)
        return rsi < cls.rsi_oversold
    
    @classmethod
    def exits(cls, price):
        rsi = cls.get_cached_indicator('rsi', price, cls.rsi_period)
        return rsi > cls.rsi_overbought
```

### Batch Testing Multiple Strategies

```bash
# Test all strategies in seeds/ folder
python batch_backtest_vectorized.py --out results/batch_results.json

# Analyze results
python analyze_results_unified.py
```

## Advanced Features

### Edge Case Handling
- Automatic handling of partial candles in live data
- Parameter collision detection and resolution
- Data type drift detection with schema checksums
- Robust error recovery and logging

### Performance Optimization
- Vectorized operations throughout
- Parallel processing for multi-strategy testing
- Efficient memory usage with chunked processing
- Smart caching with dependency tracking

### Analysis Tools
```bash
# Compare strategies
python benchmark_comparison.py

# Generate detailed reports
python analyze_results.py --format html --output results/report.html
```

## Documentation

- [FACTOR_CACHING_IMPLEMENTATION.md](FACTOR_CACHING_IMPLEMENTATION.md) - Detailed caching system docs
- [EDGE_CASE_SOLUTIONS.md](EDGE_CASE_SOLUTIONS.md) - Edge case handling guide
- [CLAUDE.md](CLAUDE.md) - AI assistant context and guidelines

## Testing

```bash
# Run all tests
pytest

# Test factor caching
python test_factor_caching.py

# Test edge cases
python test_edge_cases.py
```

## Performance Benchmarks

With factor caching enabled:
- 10x speedup for repeated backtests
- 5x reduction in memory usage
- Support for 1000+ parameter combinations
- Sub-second indicator calculations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VectorBT for the amazing backtesting framework
- DuckDB for the high-performance database
- TA-Lib for technical indicators
- The quantitative trading community

## Roadmap

Apply to new industries
