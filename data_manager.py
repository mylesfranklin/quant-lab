#!/usr/bin/env python
"""
DuckDB-based data manager for efficient partitioned storage and querying
"""
import duckdb
import pandas as pd
import pathlib
from datetime import datetime, timedelta
from typing import Optional, Union, List
import json

class DataManager:
    """Manages partitioned crypto data using DuckDB for efficient queries"""
    
    def __init__(self, data_dir: str = "data", db_path: str = None):
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Use persistent database by default
        if db_path is None:
            db_path = str(self.data_dir / "quant_lab.duckdb")
        
        self.conn = duckdb.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """Create tables and views in DuckDB"""
        # Create main price table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                symbol VARCHAR,
                timeframe VARCHAR,
                ts BIGINT,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,  
                close DOUBLE,
                volume DOUBLE,
                date DATE
            )
        """)
        
        # Create unique index instead of primary key
        self.conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_unique 
            ON prices(symbol, timeframe, ts)
        """)
        
        # Create metadata table for tracking data coverage
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_coverage (
                symbol VARCHAR,
                timeframe VARCHAR,
                start_date DATE,
                end_date DATE,
                last_updated TIMESTAMP,
                PRIMARY KEY (symbol, timeframe)
            )
        """)
    
    def import_parquet(self, file_path: Union[str, pathlib.Path], 
                      symbol: str = None, timeframe: str = None):
        """Import existing parquet file into DuckDB"""
        file_path = pathlib.Path(file_path)
        
        # Auto-detect symbol and timeframe from filename if not provided
        if symbol is None or timeframe is None:
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                symbol = symbol or parts[0]
                timeframe = timeframe or parts[1]
        
        # Read parquet and add metadata columns
        # Check if column is 'vol' or 'volume'
        temp_df = pd.read_parquet(file_path).head(1)
        volume_col = 'vol' if 'vol' in temp_df.columns else 'volume'
        
        # Execute import with better error handling
        try:
            # First, let's read into a temp table to debug
            self.conn.execute(f"""
                CREATE TEMP TABLE temp_import AS 
                SELECT 
                    '{symbol}' as symbol,
                    '{timeframe}' as timeframe,
                    CAST(ts AS BIGINT) as ts,
                    CAST(open AS DOUBLE) as open,
                    CAST(high AS DOUBLE) as high,
                    CAST(low AS DOUBLE) as low,
                    CAST(close AS DOUBLE) as close,
                    CAST({volume_col} AS DOUBLE) as volume,
                    DATE_TRUNC('day', to_timestamp(ts/1000)) as date
                FROM read_parquet('{file_path}')
            """)
            
            # Check temp table
            temp_count = self.conn.execute("SELECT COUNT(*) as cnt FROM temp_import").fetchdf()['cnt'][0]
            print(f"  Read {temp_count} rows from parquet")
            
            # Insert from temp table (DuckDB uses INSERT OR IGNORE)
            self.conn.execute("""
                INSERT OR IGNORE INTO prices 
                SELECT * FROM temp_import
            """)
            
            # Drop temp table
            self.conn.execute("DROP TABLE temp_import")
            
            # Check how many rows were inserted
            count_result = self.conn.execute(f"""
                SELECT COUNT(*) as count 
                FROM prices 
                WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
            """).fetchdf()
            
            rows_inserted = count_result['count'][0]
            print(f"  Inserted {rows_inserted} rows")
            
        except Exception as e:
            print(f"Error during import: {e}")
            raise
        
        # Update coverage metadata
        self._update_coverage(symbol, timeframe)
        
        print(f"Imported {file_path} for {symbol} {timeframe}")
    
    def insert_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Insert new data from DataFrame"""
        # Prepare dataframe
        df = df.copy()
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
        
        # Insert into DuckDB
        self.conn.execute("""
            INSERT INTO prices 
            SELECT * FROM df
            ON CONFLICT DO NOTHING
        """)
        
        # Update coverage
        self._update_coverage(symbol, timeframe)
    
    def save_partitioned(self, symbol: str, timeframe: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None):
        """Export data to partitioned parquet files"""
        # Build query
        conditions = [f"symbol = '{symbol}'", f"timeframe = '{timeframe}'"]
        
        if start_date:
            conditions.append(f"date >= '{start_date.date()}'")
        if end_date:
            conditions.append(f"date <= '{end_date.date()}'")
        
        where_clause = " AND ".join(conditions)
        
        # Get unique dates
        dates_df = self.conn.execute(f"""
            SELECT DISTINCT date 
            FROM prices 
            WHERE {where_clause}
            ORDER BY date
        """).fetchdf()
        
        # Save each date partition
        for date in dates_df['date']:
            partition_dir = self.data_dir / f"{symbol}_{timeframe}" / f"date={date}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = partition_dir / f"{symbol}_{date}.parquet"
            
            self.conn.execute(f"""
                COPY (
                    SELECT ts, open, high, low, close, volume
                    FROM prices
                    WHERE symbol = '{symbol}' 
                    AND timeframe = '{timeframe}'
                    AND date = '{date}'
                    ORDER BY ts
                ) TO '{output_file}' (FORMAT PARQUET)
            """)
            
        print(f"Saved {len(dates_df)} partitioned files for {symbol} {timeframe}")
    
    def load_data(self, symbol: str, timeframe: str,
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  columns: List[str] = None) -> pd.DataFrame:
        """Load data with optional date filtering and column selection"""
        # Default columns
        if columns is None:
            columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        
        # Build query
        select_cols = ', '.join(columns)
        conditions = [f"symbol = '{symbol}'", f"timeframe = '{timeframe}'"]
        
        if start_date:
            conditions.append(f"date >= '{start_date.date()}'")
        if end_date:
            conditions.append(f"date <= '{end_date.date()}'")
        
        where_clause = " AND ".join(conditions)
        
        # Execute query
        df = self.conn.execute(f"""
            SELECT {select_cols}
            FROM prices
            WHERE {where_clause}
            ORDER BY ts
        """).fetchdf()
        
        return df
    
    def get_coverage(self, symbol: str = None, timeframe: str = None) -> pd.DataFrame:
        """Get data coverage information"""
        conditions = []
        if symbol:
            conditions.append(f"symbol = '{symbol}'")
        if timeframe:
            conditions.append(f"timeframe = '{timeframe}'")
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        return self.conn.execute(f"""
            SELECT * FROM data_coverage
            {where_clause}
            ORDER BY symbol, timeframe
        """).fetchdf()
    
    def _update_coverage(self, symbol: str, timeframe: str):
        """Update coverage metadata for a symbol/timeframe pair"""
        self.conn.execute(f"""
            INSERT INTO data_coverage 
            SELECT 
                '{symbol}' as symbol,
                '{timeframe}' as timeframe,
                MIN(date) as start_date,
                MAX(date) as end_date,
                NOW() as last_updated
            FROM prices
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
            ON CONFLICT (symbol, timeframe) DO UPDATE SET
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                last_updated = EXCLUDED.last_updated
        """)
    
    def get_missing_dates(self, symbol: str, timeframe: str,
                         start_date: datetime, end_date: datetime) -> List[datetime]:
        """Find missing dates in the data coverage"""
        # Get existing dates
        existing_df = self.conn.execute(f"""
            SELECT DISTINCT date
            FROM prices
            WHERE symbol = '{symbol}' 
            AND timeframe = '{timeframe}'
            AND date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
        """).fetchdf()
        
        existing_dates = set(pd.to_datetime(existing_df['date']).dt.date)
        
        # Generate all dates in range
        all_dates = []
        current = start_date.date()
        while current <= end_date.date():
            if current not in existing_dates:
                all_dates.append(current)
            current += timedelta(days=1)
        
        return all_dates
    
    def analyze_data_quality(self, symbol: str, timeframe: str) -> dict:
        """Analyze data quality and statistics"""
        stats = self.conn.execute(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT date) as unique_days,
                MIN(ts) as first_timestamp,
                MAX(ts) as last_timestamp,
                AVG(close) as avg_close,
                STDDEV(close) as std_close,
                MIN(close) as min_close,
                MAX(close) as max_close,
                SUM(volume) as total_volume,
                COUNT(*) FILTER (WHERE close IS NULL) as null_closes,
                COUNT(*) FILTER (WHERE volume = 0) as zero_volume_rows
            FROM prices
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """).fetchdf()
        
        return stats.to_dict(orient='records')[0]
    
    def optimize_storage(self):
        """Optimize DuckDB storage and create indexes"""
        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_date 
            ON prices(symbol, timeframe, date)
        """)
        
        # Analyze tables for query optimization
        self.conn.execute("ANALYZE prices")
        self.conn.execute("ANALYZE data_coverage")
        
        print("Storage optimized and indexes created")
    
    def export_metadata(self, output_file: str = "data/metadata.json"):
        """Export metadata about available data"""
        coverage_df = self.get_coverage()
        
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'symbols': {}
        }
        
        for _, row in coverage_df.iterrows():
            symbol = row['symbol']
            if symbol not in metadata['symbols']:
                metadata['symbols'][symbol] = {}
            
            metadata['symbols'][symbol][row['timeframe']] = {
                'start_date': row['start_date'].isoformat(),
                'end_date': row['end_date'].isoformat(),
                'last_updated': row['last_updated'].isoformat()
            }
        
        # Add quality metrics
        for symbol in metadata['symbols']:
            for timeframe in metadata['symbols'][symbol]:
                quality = self.analyze_data_quality(symbol, timeframe)
                metadata['symbols'][symbol][timeframe]['quality'] = quality
        
        # Save metadata
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Metadata exported to {output_file}")
    
    def close(self):
        """Close DuckDB connection"""
        self.conn.close()


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DuckDB Data Manager')
    parser.add_argument('command', choices=['import', 'export', 'coverage', 'analyze'])
    parser.add_argument('--file', help='File to import')
    parser.add_argument('--symbol', help='Symbol (e.g., BTC)')
    parser.add_argument('--timeframe', help='Timeframe (e.g., 1m)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    dm = DataManager()
    
    try:
        if args.command == 'import':
            if args.file:
                dm.import_parquet(args.file, args.symbol, args.timeframe)
            else:
                print("Error: --file required for import")
        
        elif args.command == 'export':
            if args.symbol and args.timeframe:
                start = datetime.fromisoformat(args.start_date) if args.start_date else None
                end = datetime.fromisoformat(args.end_date) if args.end_date else None
                dm.save_partitioned(args.symbol, args.timeframe, start, end)
            else:
                print("Error: --symbol and --timeframe required for export")
        
        elif args.command == 'coverage':
            coverage = dm.get_coverage(args.symbol, args.timeframe)
            print(coverage)
        
        elif args.command == 'analyze':
            if args.symbol and args.timeframe:
                stats = dm.analyze_data_quality(args.symbol, args.timeframe)
                for key, value in stats.items():
                    print(f"{key}: {value}")
            else:
                print("Error: --symbol and --timeframe required for analyze")
        
        dm.export_metadata()
        
    finally:
        dm.close()