#!/usr/bin/env python
"""
Improved data fetcher with CLI arguments, proper pagination, and retry logic
"""
import requests
import pandas as pd
import datetime as dt
import pathlib
import argparse
import time
from typing import Optional

def fetch_with_retry(url: str, params: dict, max_retries: int = 3, timeout: int = 15) -> Optional[list]:
    """Fetch data with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Request failed, retrying in {wait_time}s... ({e})")
            time.sleep(wait_time)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch cryptocurrency data from Binance')
    parser.add_argument('--symbol', '--sym', default='BTC', help='Symbol to fetch (default: BTC)')
    parser.add_argument('--timeframe', '--tf', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--days', type=float, default=3.4, help='Number of days to fetch (default: 3.4)')
    parser.add_argument('--output', default='data/{symbol}_{timeframe}.parquet', 
                       help='Output file path (default: data/{symbol}_{timeframe}.parquet)')
    parser.add_argument('--partition', action='store_true', 
                       help='Partition output by date for incremental updates')
    args = parser.parse_args()
    
    # Calculate time range
    end = int(dt.datetime.utcnow().timestamp() * 1000)
    start = int((dt.datetime.utcnow() - dt.timedelta(days=args.days)).timestamp() * 1000)
    
    # API configuration
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': f'{args.symbol}USDT',
        'interval': args.timeframe,
        'startTime': start,
        'endTime': end,
        'limit': 1000
    }
    
    print(f"Fetching {args.symbol} {args.timeframe} data for {args.days} days...")
    print(f"Time range: {dt.datetime.fromtimestamp(start/1000)} to {dt.datetime.fromtimestamp(end/1000)}")
    
    rows = []
    current_start = start
    
    while current_start < end:
        params['startTime'] = current_start
        
        # Fetch with retry
        res = fetch_with_retry(url, params)
        if not res or len(res) == 0:
            break
            
        rows.extend(res)
        
        # Update start time for next request
        # Use the timestamp from the last candle + 1 interval
        last_candle_time = res[-1][0]
        
        # Calculate interval in milliseconds
        interval_ms = {
            '1m': 60_000,
            '5m': 300_000,
            '15m': 900_000,
            '1h': 3_600_000,
            '4h': 14_400_000,
            '1d': 86_400_000
        }.get(args.timeframe, 60_000)
        
        current_start = last_candle_time + interval_ms
        
        # Progress indicator
        progress = (current_start - start) / (end - start) * 100
        print(f"Progress: {progress:.1f}% - Downloaded {len(rows)} candles", end='\r')
        
        # If we got less than limit, we've reached the end
        if len(res) < 1000:
            break
    
    print(f"\nDownloaded {len(rows)} candles total")
    
    # Convert to DataFrame
    cols = ['ts', 'open', 'high', 'low', 'close', 'vol', 'close_time', 
            'quote_vol', 'trades', 'taker_base', 'taker_quote', 'ignore']
    df = pd.DataFrame(rows, columns=cols)
    
    # Keep only the main columns
    df = df[['ts', 'open', 'high', 'low', 'close', 'vol']]
    
    # Save data
    if args.partition:
        # Partition by date
        df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
        
        for date, group in df.groupby('date'):
            output_dir = pathlib.Path(f'data/{args.symbol}_{args.timeframe}/date={date}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f'{args.symbol}_{date}.parquet'
            group[['ts', 'open', 'high', 'low', 'close', 'vol']].to_parquet(output_file)
            print(f'Saved {len(group)} rows → {output_file}')
    else:
        # Single file output
        output_path = args.output.format(symbol=args.symbol, timeframe=args.timeframe)
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        print(f'Saved → {out}')

if __name__ == "__main__":
    main()