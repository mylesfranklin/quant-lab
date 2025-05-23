#!/usr/bin/env python
"""Quick benchmark comparison between original and vectorized batch backtest"""
import time
import subprocess
import json
from pathlib import Path

def run_backtest(script_name, output_file):
    """Run a backtest script and measure execution time"""
    start_time = time.time()
    
    # Run the script
    cmd = [".venv/bin/python", script_name, "--out", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Count results
    if Path(output_file).exists():
        with open(output_file) as f:
            results = json.load(f)
        num_results = len(results)
    else:
        num_results = 0
    
    return {
        'execution_time': execution_time,
        'num_results': num_results,
        'success': result.returncode == 0
    }

def main():
    print("=== QuantLab Performance Benchmark ===\n")
    
    # Test original batch backtest
    print("Running original batch backtest...")
    original_results = run_backtest(
        "batch_backtest.py", 
        "results/benchmark_original.json"
    )
    
    # Test vectorized batch backtest
    print("Running vectorized batch backtest...")
    vectorized_results = run_backtest(
        "batch_backtest_vectorized.py",
        "results/benchmark_vectorized.json"
    )
    
    # Display results
    print("\n=== Results ===")
    print(f"\nOriginal Implementation:")
    print(f"  Execution time: {original_results['execution_time']:.2f} seconds")
    print(f"  Results generated: {original_results['num_results']}")
    
    print(f"\nVectorized Implementation:")
    print(f"  Execution time: {vectorized_results['execution_time']:.2f} seconds")
    print(f"  Results generated: {vectorized_results['num_results']}")
    
    # Calculate speedup
    if original_results['execution_time'] > 0:
        speedup = original_results['execution_time'] / vectorized_results['execution_time']
        print(f"\n=== Performance Improvement ===")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Time saved: {original_results['execution_time'] - vectorized_results['execution_time']:.2f} seconds")
    
    # Clean up temporary files
    Path("results/benchmark_original.json").unlink(missing_ok=True)
    Path("results/benchmark_vectorized.json").unlink(missing_ok=True)

if __name__ == "__main__":
    main()