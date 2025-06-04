import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from google.cloud import bigquery
from google.oauth2 import service_account
import logging
from typing import List, Dict, Optional
import time
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataCrawler:
    def __init__(self, credentials_path: str, project_id: str):
        """Initialize the MarketDataCrawler with BigQuery credentials."""
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = bigquery.Client(
            credentials=self.credentials,
            project=project_id
        )
        self.project_id = project_id
        self.dataset_id = "market_data"
        self.market_breadth_table = "market_breadth"
        
        # Initialize dataset and tables if they don't exist
        self._initialize_bigquery_tables()

    def _initialize_bigquery_tables(self):
        """Create dataset and tables if they don't exist."""
        # Create dataset
        dataset_ref = f"{self.project_id}.{self.dataset_id}"
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        self.client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {self.dataset_id} created or already exists")

        # Create marketing table
        marketing_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.marketing` (
            date DATE,
            ticker STRING,
            open FLOAT64,
            high FLOAT64,
            low FLOAT64,
            close FLOAT64,
            volume FLOAT64,
            adj_close FLOAT64,
            ema5 FLOAT64,
            ema20 FLOAT64,
            ema60 FLOAT64,
            PRIMARY KEY(date, ticker) NOT ENFORCED
        )
        """
        self.client.query(marketing_query).result()
        logger.info("Table marketing created or already exists")

        # Create market breadth table
        breadth_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}` (
            date DATE,
            close FLOAT64,
            ema5_above INT64,
            ema5_below INT64,
            ema20_above INT64,
            ema20_below INT64,
            ema60_above INT64,
            ema60_below INT64,
            ema5_ratio FLOAT64,
            ema20_ratio FLOAT64,
            ema60_ratio FLOAT64,
            PRIMARY KEY(date) NOT ENFORCED
        )
        """
        self.client.query(breadth_query).result()
        logger.info(f"Table {self.market_breadth_table} created or already exists")

    def get_symbols_from_bigquery(self) -> List[str]:
        """Get all tickers from crawler_tickers table."""
        query = f"""
        SELECT DISTINCT ticker
        FROM `{self.project_id}.{self.dataset_id}.crawler_tickers`
        ORDER BY ticker
        """
        result = self.client.query(query).result()
        symbols = [row.ticker for row in result]
        if not symbols:
            raise ValueError("No symbols found in crawler_tickers table")
        logger.info(f"Loaded {len(symbols)} symbols from BigQuery")
        return symbols

    def get_sp500_symbols_from_bigquery(self) -> List[str]:
        """Get only S&P 500 tickers from crawler_tickers table."""
        query = f"""
        SELECT DISTINCT ticker
        FROM `{self.project_id}.{self.dataset_id}.crawler_tickers`
        WHERE is_sp500 = TRUE
        ORDER BY ticker
        """
        result = self.client.query(query).result()
        symbols = [row.ticker for row in result]
        if not symbols:
            raise ValueError("No S&P 500 symbols found in crawler_tickers table")
        logger.info(f"Loaded {len(symbols)} S&P 500 symbols from BigQuery")
        return symbols

    def get_symbols_from_file(self) -> List[str]:
        """Get S&P 500 symbols from local file."""
        with open('sp500_symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        if not symbols:
            raise ValueError("No symbols found in sp500_symbols.txt")
        logger.info(f"Loaded {len(symbols)} symbols from sp500_symbols.txt")
        return symbols

    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA5, EMA20, and EMA60 for a given DataFrame."""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        df.loc[:, 'ema5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df.loc[:, 'ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df.loc[:, 'ema60'] = df['Close'].ewm(span=60, adjust=False).mean()
        return df

    def check_marketing_data_exists(self, start_date: datetime.datetime, end_date: datetime.datetime, symbol: str) -> bool:
        """Check if data already exists for the given date range and symbol."""
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.marketing`
        WHERE date >= '{start_date.strftime('%Y-%m-%d')}'
        AND date <= '{end_date.strftime('%Y-%m-%d')}'
        AND ticker = '{symbol}'
        """
        result = self.client.query(query).result()
        count = next(result).count
        return count > 0

    def check_breadth_data_exists(self, date: datetime.datetime) -> bool:
        """Check if market breadth data already exists for the given date."""
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}`
        WHERE date = '{date.strftime('%Y-%m-%d')}'
        """
        result = self.client.query(query).result()
        count = next(result).count
        return count > 0

    def update_market_data(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None):
        """Update market data and calculate market breadth.
        
        Args:
            start_date: Optional start date for data update
            end_date: Optional end date for data update
        """
        # Get symbols
        if self.use_bigquery:
            symbols = self.get_symbols_from_bigquery()
        else:
            symbols = self.get_symbols_from_file()
        
        logger.info(f"Processing {len(symbols)} symbols")

        # Determine date range
        if start_date is None:
            # Get the last date in the marketing table
            query = f"""
            SELECT MAX(date) as last_date
            FROM `{self.project_id}.{self.dataset_id}.marketing`
            """
            result = self.client.query(query).result()
            last_date = next(result).last_date

            if last_date is None:
                # If no data exists, start from 90 days ago
                start_date = datetime.datetime.now() - datetime.timedelta(days=90)
            else:
                start_date = last_date + datetime.timedelta(days=1)

        if end_date is None:
            end_date = datetime.datetime.now()

        logger.info(f"Updating data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Download data for all symbols at once
        logger.info("Downloading data for all symbols...")
        df = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker',
            auto_adjust=True,
        )

        if df.empty:
            logger.error("No data downloaded for any symbols")
            return

        # Process each symbol's data
        for symbol in symbols:
            # Get data for this symbol
            symbol_data = df[symbol].copy()
            
            if symbol_data.empty:
                logger.warning(f"No data available for {symbol}")
                continue

            # Check if data already exists
            if self.check_marketing_data_exists(start_date, end_date, symbol):
                logger.info(f"Data already exists for {symbol}, skipping...")
                continue

            # Calculate EMAs
            symbol_data = self.calculate_emas(symbol_data)

            # Prepare data for BigQuery
            symbol_data.loc[:, 'ticker'] = symbol
            symbol_data = symbol_data.reset_index()
            symbol_data = symbol_data.rename(columns={'Date': 'date'})

            # Upload to BigQuery
            table_ref = f"{self.project_id}.{self.dataset_id}.marketing"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND
            )
            self.client.load_table_from_dataframe(
                symbol_data, table_ref, job_config=job_config
            )
            logger.info(f"Updated data for {symbol}")

        # Calculate and update market breadth
        self.update_market_breadth(start_date, end_date)

    def update_market_breadth(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None):
        """Calculate and update market breadth metrics using only S&P 500 stocks."""
        if start_date is None:
            # Get the last date in the market breadth table
            query = f"""
            SELECT MAX(date) as last_date
            FROM `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}`
            """
            result = self.client.query(query).result()
            last_date = next(result).last_date

            if last_date is None:
                # If no data exists, start from 90 days ago
                start_date = datetime.datetime.now() - datetime.timedelta(days=90)
            else:
                start_date = last_date + datetime.timedelta(days=1)

        if end_date is None:
            end_date = datetime.datetime.now()

        # Get S&P 500 symbols
        sp500_symbols = self.get_sp500_symbols_from_bigquery()
        symbols_str = "', '".join(sp500_symbols)

        # Calculate market breadth for each date using only S&P 500 stocks
        query = f"""
        WITH daily_stats AS (
            SELECT
                date,
                AVG(close) as close,
                COUNTIF(close > ema5) as ema5_above,
                COUNTIF(close <= ema5) as ema5_below,
                COUNTIF(close > ema20) as ema20_above,
                COUNTIF(close <= ema20) as ema20_below,
                COUNTIF(close > ema60) as ema60_above,
                COUNTIF(close <= ema60) as ema60_below
            FROM `{self.project_id}.{self.dataset_id}.marketing`
            WHERE date >= '{start_date.strftime('%Y-%m-%d')}'
            AND date <= '{end_date.strftime('%Y-%m-%d')}'
            AND ticker IN ('{symbols_str}')
            GROUP BY date
        )
        SELECT
            date,
            close,
            ema5_above,
            ema5_below,
            ema20_above,
            ema20_below,
            ema60_above,
            ema60_below,
            SAFE_DIVIDE(ema5_above, ema5_above + ema5_below) as ema5_ratio,
            SAFE_DIVIDE(ema20_above, ema20_above + ema20_below) as ema20_ratio,
            SAFE_DIVIDE(ema60_above, ema60_above + ema60_below) as ema60_ratio
        FROM daily_stats
        ORDER BY date
        """
        
        result = self.client.query(query).result()
        df = result.to_dataframe()

        if not df.empty:
            # Filter out dates that already exist in the table
            existing_dates = set()
            for date in df['date']:
                if self.check_breadth_data_exists(date):
                    existing_dates.add(date)
            
            if existing_dates:
                df = df[~df['date'].isin(existing_dates)]
                logger.info(f"Skipping {len(existing_dates)} existing dates")

            if not df.empty:
                # Upload to BigQuery
                table_ref = f"{self.project_id}.{self.dataset_id}.{self.market_breadth_table}"
                job_config = bigquery.LoadJobConfig(
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND
                )
                self.client.load_table_from_dataframe(
                    df, table_ref, job_config=job_config
                )
                logger.info(f"Updated market breadth data for {len(df)} dates")
            else:
                logger.info("No new market breadth data to update")

def main():
    """Main entry point for the crawler."""
    import argparse

    parser = argparse.ArgumentParser(description="Market Data Crawler")
    parser.add_argument(
        "--credentials",
        required=True,
        help="Path to GCP service account credentials JSON file"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--start-date",
        help="Start date for data update (YYYY-MM-DD). If not provided, uses last date in database or 90 days ago."
    )
    parser.add_argument(
        "--end-date",
        help="End date for data update (YYYY-MM-DD). If not provided, uses current date."
    )
    parser.add_argument(
        "--use-bigquery",
        action="store_true",
        default=True,
        help="Use symbols from BigQuery sp500_tickers table instead of local file"
    )

    args = parser.parse_args()

    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start_date:
        try:
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid start date format. Use YYYY-MM-DD")
            sys.exit(1)
    
    if args.end_date:
        try:
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid end date format. Use YYYY-MM-DD")
            sys.exit(1)

    crawler = MarketDataCrawler(args.credentials, args.project_id)
    crawler.use_bigquery = args.use_bigquery
    crawler.update_market_data(start_date, end_date)
    logger.info("Update completed successfully")

if __name__ == "__main__":
    main() 