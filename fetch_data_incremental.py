#!/usr/bin/env python
"""
Incremental data fetcher with partitioned storage support
Only fetches missing data based on existing coverage
"""
import requests
import pandas as pd
import datetime as dt
import pathlib
import argparse
import time
from typing import Optional, List
from data_manager import DataManager

class IncrementalFetcher:
    """Fetches only missing data and updates partitioned storage"""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.base_url = 'https://api.binance.com/api/v3/klines'
        
    def fetch_with_retry(self, params: dict, max_retries: int = 3, timeout: int = 15) -> Optional[list]:
        """Fetch data with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return None
                wait_time = 2 ** attempt
                print(f"Request failed, retrying in {wait_time}s... ({e})")
                time.sleep(wait_time)
    
    def get_interval_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        intervals = {
            '1m': 60_000,
            '5m': 300_000,
            '15m': 900_000,
            '30m': 1_800_000,
            '1h': 3_600_000,
            '4h': 14_400_000,
            '1d': 86_400_000
        }
        return intervals.get(timeframe, 60_000)
    
    def fetch_range(self, symbol: str, timeframe: str, 
                   start_ms: int, end_ms: int) -> List[dict]:
        """Fetch data for a specific time range"""
        all_rows = []
        current_start = start_ms
        interval_ms = self.get_interval_ms(timeframe)
        
        while current_start < end_ms:
            params = {
                'symbol': f'{symbol}USDT',
                'interval': timeframe,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000
            }
            
            # Fetch with retry
            res = self.fetch_with_retry(params)
            if not res or len(res) == 0:
                break
            
            all_rows.extend(res)
            
            # Update start time for next request
            last_candle_time = res[-1][0]
            current_start = last_candle_time + interval_ms
            
            # Progress indicator
            progress = (current_start - start_ms) / (end_ms - start_ms) * 100
            print(f"  Progress: {progress:.1f}% - Downloaded {len(all_rows)} candles", end='\r')
            
            # If we got less than limit, we've reached the end
            if len(res) < 1000:
                break
            
            # Rate limit protection
            time.sleep(0.1)
        
        print()  # New line after progress
        return all_rows
    
    def fetch_incremental(self, symbol: str, timeframe: str, days_back: int = 30):
        """Fetch only missing data based on current coverage"""
        end_date = dt.datetime.utcnow()
        start_date = end_date - dt.timedelta(days=days_back)
        
        print(f"\nChecking data coverage for {symbol} {timeframe}...")
        
        # Get current coverage
        coverage_df = self.dm.get_coverage(symbol, timeframe)
        
        if len(coverage_df) > 0:
            # We have existing data
            existing_start = pd.to_datetime(coverage_df.iloc[0]['start_date'])
            existing_end = pd.to_datetime(coverage_df.iloc[0]['end_date'])
            
            print(f"Existing data: {existing_start.date()} to {existing_end.date()}")
            
            # Determine what to fetch
            fetch_ranges = []
            
            # Check if we need earlier data
            if start_date < existing_start:
                fetch_ranges.append((start_date, existing_start - dt.timedelta(minutes=1)))
                print(f"Will fetch earlier data: {start_date.date()} to {existing_start.date()}")
            
            # Check if we need recent data
            if existing_end < end_date - dt.timedelta(days=1):
                fetch_start = existing_end + dt.timedelta(days=1)
                fetch_ranges.append((fetch_start, end_date))
                print(f"Will fetch recent data: {fetch_start.date()} to {end_date.date()}")
            
            # Check for gaps
            missing_dates = self.dm.get_missing_dates(symbol, timeframe, start_date, end_date)
            if missing_dates:
                print(f"Found {len(missing_dates)} missing dates in coverage")
                for date in missing_dates[:5]:  # Show first 5
                    print(f"  - {date}")
                if len(missing_dates) > 5:
                    print(f"  ... and {len(missing_dates) - 5} more")
        else:
            # No existing data, fetch everything
            fetch_ranges = [(start_date, end_date)]
            print(f"No existing data found, will fetch full range")
        
        # Fetch each range
        total_new_rows = 0
        for start, end in fetch_ranges:
            print(f"\nFetching {symbol} {timeframe} from {start} to {end}")
            
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            
            rows = self.fetch_range(symbol, timeframe, start_ms, end_ms)
            
            if rows:
                # Convert to DataFrame
                cols = ['ts', 'open', 'high', 'low', 'close', 'vol', 'close_time', 
                       'quote_vol', 'trades', 'taker_base', 'taker_quote', 'ignore']
                df = pd.DataFrame(rows, columns=cols)
                
                # Keep only main columns and rename
                df = df[['ts', 'open', 'high', 'low', 'close', 'vol']]
                df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
                
                # Convert types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                df['ts'] = df['ts'].astype('int64')
                
                # Insert into data manager
                self.dm.insert_data(df, symbol, timeframe)
                total_new_rows += len(df)
                
                print(f"  Added {len(df)} rows to database")
        
        if total_new_rows > 0:
            print(f"\nTotal new data points: {total_new_rows}")
            
            # Save updated partitions
            print("Saving partitioned files...")
            self.dm.save_partitioned(symbol, timeframe, start_date, end_date)
            
            # Export updated metadata
            self.dm.export_metadata()
        else:
            print("\nNo new data to fetch - already up to date!")
        
        # Show final coverage
        coverage_df = self.dm.get_coverage(symbol, timeframe)
        if len(coverage_df) > 0:
            row = coverage_df.iloc[0]
            print(f"\nFinal coverage: {row['start_date']} to {row['end_date']}")
            
            # Analyze data quality
            stats = self.dm.analyze_data_quality(symbol, timeframe)
            print(f"Total rows: {stats['total_rows']:,}")
            print(f"Date range: {stats['unique_days']} days")
            print(f"Data quality: {100 - (stats['null_closes'] / stats['total_rows'] * 100):.2f}% complete")


def main():
    parser = argparse.ArgumentParser(description='Incremental cryptocurrency data fetcher')
    parser.add_argument('--symbol', '--sym', default='BTC', help='Symbol to fetch (default: BTC)')
    parser.add_argument('--timeframe', '--tf', default='1m', help='Timeframe (default: 1m)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to maintain (default: 30)')
    parser.add_argument('--check-only', action='store_true', help='Only check coverage without fetching')
    
    args = parser.parse_args()
    
    # Initialize data manager
    dm = DataManager()
    
    try:
        if args.check_only:
            # Just show current coverage
            coverage = dm.get_coverage(args.symbol, args.timeframe)
            if len(coverage) > 0:
                row = coverage.iloc[0]
                print(f"\nCurrent coverage for {args.symbol} {args.timeframe}:")
                print(f"  Start: {row['start_date']}")
                print(f"  End: {row['end_date']}")
                print(f"  Last updated: {row['last_updated']}")
                
                # Check quality
                stats = dm.analyze_data_quality(args.symbol, args.timeframe)
                print(f"\nData quality:")
                print(f"  Total rows: {stats['total_rows']:,}")
                print(f"  Unique days: {stats['unique_days']}")
                print(f"  Null values: {stats['null_closes']}")
                print(f"  Zero volume: {stats['zero_volume_rows']}")
            else:
                print(f"No data found for {args.symbol} {args.timeframe}")
        else:
            # Fetch incremental data
            fetcher = IncrementalFetcher(dm)
            fetcher.fetch_incremental(args.symbol, args.timeframe, args.days)
    
    finally:
        dm.close()


if __name__ == "__main__":
    main()