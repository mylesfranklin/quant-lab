#!/usr/bin/env python
"""
Simple test to verify caching is working
"""
import time
import pandas as pd
import cached_indicators as ci
from data_manager import DataManager
from factor_store import get_factor_store

# Load data
dm = DataManager()
df = dm.load_data('BTC', '1m')
print(f"Loaded {len(df)} rows")

# Convert to proper format
df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
df = df.set_index('datetime')

# Set cache context
ci.set_current_data(df, 'BTC', '1m')

# Test caching
print("\nTest 1: Computing EMA(20) - should be slow")
start = time.time()
ema1 = ci.ema(period=20)
time1 = time.time() - start
print(f"Time: {time1:.3f}s")

print("\nTest 2: Computing EMA(20) again - should be fast (cached)")
start = time.time()
ema2 = ci.ema(period=20)
time2 = time.time() - start
print(f"Time: {time2:.3f}s")

print(f"\nSpeedup: {time1/time2:.1f}x")

# Check cache stats
stats = get_factor_store().get_cache_stats()
print(f"\nCache stats:")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
print(f"  Factors: {stats['total_factors']}")

# Verify results match
import numpy as np
if np.allclose(ema1.values, ema2.values, equal_nan=True):
    print("\n✓ Results match - cache working correctly!")
else:
    print("\n✗ Results don't match - cache issue!")

dm.close()