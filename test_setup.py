#!/usr/bin/env python3
"""Test script to verify QuantLab setup"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import duckdb
import polars as pl
import talib
from pathlib import Path

print("=== QuantLab Setup Test ===")
print()

# Test 1: Load data
print("1. Testing data loading...")
data_path = Path("data/BTC_1m.parquet")
if data_path.exists():
    df = pd.read_parquet(data_path)
    print(f"✓ Loaded {len(df)} rows of BTC data")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Date range: {df['ts'].min()} to {df['ts'].max()}")
else:
    print("✗ Data file not found")

# Test 2: VectorBT
print("\n2. Testing VectorBT...")
try:
    # Simple moving average crossover
    close = pd.Series([100, 101, 102, 101, 103, 104, 103, 105, 106, 105])
    fast_ma = vbt.MA.run(close, 3)
    slow_ma = vbt.MA.run(close, 5)
    print("✓ VectorBT working - calculated moving averages")
except Exception as e:
    print(f"✗ VectorBT error: {e}")

# Test 3: DuckDB
print("\n3. Testing DuckDB...")
try:
    conn = duckdb.connect()
    result = conn.execute("SELECT 'DuckDB' as db, 42 as answer").fetchall()
    print(f"✓ DuckDB working - query result: {result}")
except Exception as e:
    print(f"✗ DuckDB error: {e}")

# Test 4: Polars
print("\n4. Testing Polars...")
try:
    pl_df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })
    print(f"✓ Polars working - created DataFrame with shape {pl_df.shape}")
except Exception as e:
    print(f"✗ Polars error: {e}")

# Test 5: TA-Lib
print("\n5. Testing TA-Lib...")
try:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sma = talib.SMA(data, timeperiod=3)
    print(f"✓ TA-Lib working - SMA calculation: {sma}")
except Exception as e:
    print(f"✗ TA-Lib error: {e}")

print("\n=== Setup test complete ===")