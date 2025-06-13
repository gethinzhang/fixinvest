import pandas as pd
import logging
from typing import List, Dict
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
import datetime
import yfinance as yf
import time
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TickerUpdater:
    def __init__(self, credentials_path: str, project_id: str):
        """Initialize the TickerUpdater with BigQuery credentials."""
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = bigquery.Client(
            credentials=self.credentials,
            project=project_id
        )
        self.project_id = project_id
        self.dataset_id = "market_data"
        self.tickers_table = "crawler_tickers"
        self._initialize_bigquery_table()

    def _initialize_bigquery_table(self):
        """Create crawler_tickers table if it doesn't exist."""
        dataset_ref = f"{self.project_id}.{self.dataset_id}"
        
        # Create dataset if it doesn't exist
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        self.client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset {self.dataset_id} created or already exists")

        # Create crawler_tickers table
        query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.{self.tickers_table}` (
            ticker STRING,
            is_sp500 BOOL,
            sub_category STRING,
            symbol_type STRING,
            PRIMARY KEY(ticker) NOT ENFORCED
        )
        """
        self.client.query(query).result()
        logger.info(f"Table {self.tickers_table} created or already exists")

    def fetch_sp500_symbols(self) -> List[str]:
        """Fetch current S&P 500 symbols from Wikipedia."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            symbols = sorted(df['Symbol'].tolist())
            logger.info(f"Successfully fetched {len(symbols)} symbols from Wikipedia")
            return symbols
        except pd.errors.EmptyDataError:
            logger.error("No data found in Wikipedia table")
            sys.exit(1)
        except pd.errors.ParserError:
            logger.error("Failed to parse Wikipedia table")
            sys.exit(1)
        except ConnectionError:
            logger.error("Failed to connect to Wikipedia")
            sys.exit(1)

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get sub-category
            sub_category = info.get('industry', '')
            
            # Determine symbol type
            symbol_type = 'STOCK'  # Default type
            
            # Check for ETFs
            if info.get('quoteType') == 'ETF':
                symbol_type = 'ETF'
            # Check for Mutual Funds
            elif info.get('quoteType') == 'MUTUALFUND':
                symbol_type = 'MUTUALFUND'
            # Check for REITs
            elif info.get('quoteType') == 'EQUITY' and info.get('industry') == 'REIT':
                symbol_type = 'REIT'
            # Check for Preferred Stocks
            elif info.get('quoteType') == 'EQUITY' and 'Preferred' in info.get('longName', ''):
                symbol_type = 'PREFERRED'
            
            return {
                'sub_category': sub_category,
                'symbol_type': symbol_type
            }
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            return {
                'sub_category': '',
                'symbol_type': 'UNKNOWN'
            }

    def update_symbols_file(self, symbols: List[str]) -> None:
        """Update sp500_symbols.txt with new symbols."""
        try:
            with open('sp500_symbols.txt', 'w') as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            logger.info("Successfully updated sp500_symbols.txt")
        except IOError as e:
            logger.error(f"Failed to write to sp500_symbols.txt: {e}")
            sys.exit(1)

    def update_bigquery_tickers(self, symbols: List[str], is_sp500: bool = False) -> None:
        """Update crawler_tickers table in BigQuery."""
        try:
            # Get existing tickers
            query = f"""
            SELECT ticker, is_sp500, sub_category, symbol_type
            FROM `{self.project_id}.{self.dataset_id}.{self.tickers_table}`
            """
            result = self.client.query(query).result()
            existing_df = result.to_dataframe()
            existing_tickers = set(existing_df['ticker'].tolist()) if not existing_df.empty else set()

            # Create new DataFrame for symbols
            new_data = []
            for symbol in symbols:
                # Skip if ticker already exists
                if symbol in existing_tickers:
                    logger.info(f"Skipping existing ticker: {symbol}")
                    continue

                # Get symbol info
                info = self.get_symbol_info(symbol)
                new_data.append({
                    'ticker': symbol,
                    'is_sp500': is_sp500,
                    'sub_category': info['sub_category'],
                    'symbol_type': info['symbol_type']
                })
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            if not new_data:
                logger.info("No new tickers to add")
                return

            new_df = pd.DataFrame(new_data)

            # Combine with existing data
            if not existing_df.empty:
                df = pd.concat([new_df, existing_df], ignore_index=True)
            else:
                df = new_df

            # Upload to BigQuery
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.tickers_table}"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND
            )
            self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            logger.info(f"Successfully updated {self.tickers_table} in BigQuery with {len(new_df)} new tickers")
        except Exception as e:
            logger.error(f"Failed to update BigQuery table: {e}")
            sys.exit(1)

def main():
    """Main entry point for updating symbols."""
    import argparse

    parser = argparse.ArgumentParser(description="Symbols Updater")
    parser.add_argument(
        "--credentials",
        default="credentials/gcp-credentials.json",
        help="Path to GCP service account credentials JSON file"
    )
    parser.add_argument(
        "--project-id",
        default="hi5-strategy",
        help="GCP project ID"
    )
    parser.add_argument(
        "--additional-symbols",
        help="Comma-separated list of additional symbols to add",
        default="PDD,VUG,VO,MOAT,PFF,VNQ,RSP"
    )

    args = parser.parse_args()
    
    logger.info("Starting symbols update")
    
    # Initialize updater
    updater = TickerUpdater(args.credentials, args.project_id)
    
    # Fetch S&P 500 symbols
    sp500_symbols = updater.fetch_sp500_symbols()
    
    # Update local file with S&P 500 symbols
    updater.update_symbols_file(sp500_symbols)
    
    # Update BigQuery with S&P 500 symbols
    updater.update_bigquery_tickers(sp500_symbols, is_sp500=True)
    
    # Process additional symbols if provided
    if args.additional_symbols:
        additional_symbols = [s.strip() for s in args.additional_symbols.split(',')]
        logger.info(f"Processing {len(additional_symbols)} additional symbols")
        updater.update_bigquery_tickers(additional_symbols, is_sp500=False)
    
    logger.info("Update completed successfully")

if __name__ == "__main__":
    main() 