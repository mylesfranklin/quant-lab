#!/bin/bash
# Run factor caching validation

cd /Users/mylesfranklin/QuantLab/quant-lab
source .venv/bin/activate

echo "=== Factor Caching Validation ==="
echo ""
echo "1. Testing cache statistics..."
python cli.py cache stats

echo ""
echo "2. Testing strategy list..."
python cli.py strategy list | grep cached

echo ""
echo "3. Running validation test suite..."
python test_factor_caching.py

echo ""
echo "=== Validation Complete ==="