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
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataCrawler:
    def __init__(self, gcp_config_path: str, smtp_config_path: str):
        """Initialize the MarketDataCrawler with BigQuery credentials."""
        # Load GCP configuration
        with open(gcp_config_path, 'r') as f:
            gcp_config = json.load(f)
        
        self.credentials = service_account.Credentials.from_service_account_file(
            gcp_config['credentials_path']
        )
        self.client = bigquery.Client(
            credentials=self.credentials,
            project=gcp_config['project_id']
        )
        self.project_id = gcp_config['project_id']
        self.dataset_id = "market_data"
        self.market_breadth_table = "market_breadth"
        
        # Load SMTP configuration
        with open(smtp_config_path, 'r') as f:
            self.smtp_config = json.load(f)
        
        # Initialize dataset and tables if they don't exist
        self._initialize_bigquery_tables()

    def _initialize_bigquery_tables(self):
        """Create dataset and tables if they don't exist."""
        # Create marketing table
        marketing_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.marketing` (
            date DATE,
            ticker STRING,
            open FLOAT64,
            high FLOAT64,
            low FLOAT64,
            raw_close FLOAT64,
            close FLOAT64,
            adj_close FLOAT64,
            dividend FLOAT64,
            volume INT64,
            capital_gains FLOAT64,
            split_ratio FLOAT64,
            PRIMARY KEY(date, ticker) NOT ENFORCED
        )
        """
        self.client.query(marketing_query).result()
        logger.info("Table marketing created or already exists")

        # Create technical indicators table
        indicators_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.technical_indicators` (
            date DATE,
            ticker STRING,
            ma5 FLOAT64,
            ma20 FLOAT64,
            ma50 FLOAT64,
            ma200 FLOAT64,
            ema5 FLOAT64,
            ema20 FLOAT64,
            ema50 FLOAT64,
            ema200 FLOAT64,
            PRIMARY KEY(date, ticker) NOT ENFORCED
        )
        """
        self.client.query(indicators_query).result()
        logger.info("Table technical_indicators created or already exists")

        # Create market breadth table
        breadth_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}` (
            date DATE,
            ma20_above INT64,
            ma20_below INT64,
            ma50_above INT64,
            ma50_below INT64,
            ma200_above INT64,
            ma200_below INT64,
            ma20_ratio FLOAT64,
            ma50_ratio FLOAT64,
            ma200_ratio FLOAT64,
            ema20_above INT64,
            ema20_below INT64,
            ema50_above INT64,
            ema50_below INT64,
            ema200_above INT64,
            ema200_below INT64,
            ema20_ratio FLOAT64,
            ema50_ratio FLOAT64,
            ema200_ratio FLOAT64,
            PRIMARY KEY(date) NOT ENFORCED
        )
        """
        self.client.query(breadth_query).result()
        logger.info(f"Table {self.market_breadth_table} created or already exists")

    def get_symbols_from_bigquery(self, is_sp500: bool = False, specific_tickers: Optional[List[str]] = None) -> List[str]:
        """Get tickers from crawler_tickers table.
        
        Args:
            is_sp500: If True, only return S&P 500 tickers. If False, return all tickers.
            specific_tickers: Optional list of specific tickers to validate and return.
            
        Returns:
            List of ticker symbols
            
        Raises:
            RuntimeError: If any of the specific tickers don't exist in crawler_tickers table
        """
        if specific_tickers:
            # Validate that all specified tickers exist in crawler_tickers
            query = f"""
            SELECT ticker
            FROM `{self.project_id}.{self.dataset_id}.crawler_tickers`
            WHERE ticker IN UNNEST(@tickers)
            {f"AND is_sp500 = TRUE" if is_sp500 else ""}
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("tickers", "STRING", specific_tickers)
                ]
            )
            
            result = self.client.query(query, job_config=job_config).result()
            found_tickers = {row.ticker for row in result}
            
            # Check for missing tickers
            missing_tickers = set(specific_tickers) - found_tickers
            if missing_tickers:
                raise RuntimeError(
                    f"The following tickers do not exist in crawler_tickers table: {', '.join(sorted(missing_tickers))}"
                )
            
            symbols = list(found_tickers)
        else:
            # Get all tickers (existing behavior)
            query = f"""
            SELECT DISTINCT ticker
            FROM `{self.project_id}.{self.dataset_id}.crawler_tickers`
            {f"WHERE is_sp500 = TRUE" if is_sp500 else ""}
            ORDER BY ticker
            """
            result = self.client.query(query).result()
            symbols = [row.ticker for row in result]
        
        if not symbols:
            raise ValueError(f"No {'S&P 500 ' if is_sp500 else ''}symbols found in crawler_tickers table")
        
        logger.info(f"Loaded {len(symbols)} {'S&P 500 ' if is_sp500 else ''}symbols from BigQuery")
        return symbols

    def update_market_breadth(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None):
        """Directly calculate and update market breadth metrics in BigQuery for the given date range."""
        if start_date is None:
            # Get the last date in the market breadth table
            query = f"""
            SELECT MAX(date) as last_date
            FROM `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}`
            """
            result = self.client.query(query).result()
            last_date = next(result).last_date
            if last_date is None:
                start_date = datetime.datetime.now() - datetime.timedelta(days=90)
            else:
                start_date = last_date + datetime.timedelta(days=1)
        if end_date is None:
            end_date = datetime.datetime.now()

        print(f"update market breadth start_date: {start_date}, end_date: {end_date}")

        merge_sql = f"""
        MERGE `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}` target
        USING (
            SELECT
                m.date,
                COUNTIF(adj_close > ma20) as ma20_above,
                COUNTIF(adj_close <= ma20) as ma20_below,
                COUNTIF(adj_close > ma50) as ma50_above,
                COUNTIF(adj_close <= ma50) as ma50_below,
                COUNTIF(adj_close > ma200) as ma200_above,
                COUNTIF(adj_close <= ma200) as ma200_below,
                SAFE_DIVIDE(COUNTIF(adj_close > ma20), COUNT(*)) as ma20_ratio,
                SAFE_DIVIDE(COUNTIF(adj_close > ma50), COUNT(*)) as ma50_ratio,
                SAFE_DIVIDE(COUNTIF(adj_close > ma200), COUNT(*)) as ma200_ratio,
                COUNTIF(adj_close > ema20) as ema20_above,
                COUNTIF(adj_close <= ema20) as ema20_below,
                COUNTIF(adj_close > ema50) as ema50_above,
                COUNTIF(adj_close <= ema50) as ema50_below,
                COUNTIF(adj_close > ema200) as ema200_above,
                COUNTIF(adj_close <= ema200) as ema200_below,
                SAFE_DIVIDE(COUNTIF(adj_close > ema20), COUNT(*)) as ema20_ratio,
                SAFE_DIVIDE(COUNTIF(adj_close > ema50), COUNT(*)) as ema50_ratio,
                SAFE_DIVIDE(COUNTIF(adj_close > ema200), COUNT(*)) as ema200_ratio
            FROM `{self.project_id}.{self.dataset_id}.marketing` m
            JOIN `{self.project_id}.{self.dataset_id}.technical_indicators` t
                ON m.date = t.date AND m.ticker = t.ticker
            WHERE m.date >= '{start_date.strftime('%Y-%m-%d')}'
              AND m.date <= '{end_date.strftime('%Y-%m-%d')}'
              AND m.ticker IN (SELECT ticker from `{self.project_id}.{self.dataset_id}.crawler_tickers` WHERE is_sp500=TRUE)
            GROUP BY date
        ) source
        ON target.date = source.date
        WHEN MATCHED THEN
            UPDATE SET
                ma20_above = source.ma20_above,
                ma20_below = source.ma20_below,
                ma50_above = source.ma50_above,
                ma50_below = source.ma50_below,
                ma200_above = source.ma200_above,
                ma200_below = source.ma200_below,
                ma20_ratio = source.ma20_ratio,
                ma50_ratio = source.ma50_ratio,
                ma200_ratio = source.ma200_ratio,
                ema20_above = source.ema20_above,
                ema20_below = source.ema20_below,
                ema50_above = source.ema50_above,
                ema50_below = source.ema50_below,
                ema200_above = source.ema200_above,
                ema200_below = source.ema200_below,
                ema20_ratio = source.ema20_ratio,
                ema50_ratio = source.ema50_ratio,
                ema200_ratio = source.ema200_ratio
        WHEN NOT MATCHED THEN
            INSERT (date, ma20_above, ma20_below, ma50_above, ma50_below, ma200_above, ma200_below,
                    ma20_ratio, ma50_ratio, ma200_ratio, ema20_above, ema20_below, ema50_above, ema50_below,
                    ema200_above, ema200_below, ema20_ratio, ema50_ratio, ema200_ratio)
            VALUES (source.date, source.ma20_above, source.ma20_below, source.ma50_above, source.ma50_below,
                    source.ma200_above, source.ma200_below, source.ma20_ratio, source.ma50_ratio, source.ma200_ratio,
                    source.ema20_above, source.ema20_below, source.ema50_above, source.ema50_below,
                    source.ema200_above, source.ema200_below, source.ema20_ratio, source.ema50_ratio, source.ema200_ratio)
        """
        
        self.client.query(merge_sql).result()
        logger.info("Market breadth update completed successfully")

    def update_market_data(self, start_date: Optional[datetime.datetime] = None, 
                          end_date: Optional[datetime.datetime] = None,
                          specific_tickers: Optional[List[str]] = None):
        """Update market data and calculate market breadth.
        
        Args:
            start_date: Optional start date for data update
            end_date: Optional end date for data update
            specific_tickers: Optional list of specific tickers to process
        """
        # Get symbols
        symbols = self.get_symbols_from_bigquery(specific_tickers=specific_tickers)
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
            auto_adjust=False,
            actions=True,
            group_by='ticker',
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

            # Prepare data for BigQuery
            symbol_data.loc[:, 'ticker'] = symbol
            symbol_data = symbol_data.reset_index()
            symbol_data = symbol_data.rename(columns={'Date': 'date'})
            symbol_data = symbol_data.rename(columns={'Adj Close': 'adj_close'})
            symbol_data = symbol_data.rename(columns={'Dividends': 'dividend'})
            symbol_data = symbol_data.rename(columns={'Close': 'close'})
            symbol_data = symbol_data.rename(columns={'Stock Splits': 'split_ratio'})
            symbol_data = symbol_data.rename(columns={'Capital Gains': 'capital_gains'})
            symbol_data['Volume'] = symbol_data['Volume'].fillna(0).round(0).astype(pd.Int64Dtype())

            # Upload to BigQuery
            table_ref = f"{self.project_id}.{self.dataset_id}.marketing"
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND
            )
            self.client.load_table_from_dataframe(
                symbol_data, table_ref, job_config=job_config
            )
            logger.info(f"Updated data for {symbol}")

    def deduplicate_tables(self, categories: List[str]):
        """Deduplicate tables based on the categories that were processed.
        
        Args:
            categories: List of categories that were processed in this run
        """
        logger.info("Starting table deduplication...")
        
        # Deduplicate market_breadth table if breadth was processed
        if 'breadth' in categories:
            market_breadth_dedup_query = f"""
            CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}` AS
            SELECT * EXCEPT(row_num)
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER(PARTITION BY `date` ORDER BY `date` DESC) as row_num
                FROM `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}`
            )
            WHERE row_num = 1
            """
            self.client.query(market_breadth_dedup_query).result()
            logger.info("Market breadth table deduplication completed")

        # Deduplicate marketing table if market data was processed
        if 'market' in categories:
            marketing_dedup_query = f"""
            CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.marketing` AS
            SELECT * EXCEPT(row_num)
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER(PARTITION BY `date`, `ticker` ORDER BY `date` DESC) as row_num
                FROM `{self.project_id}.{self.dataset_id}.marketing`
            )
            WHERE row_num = 1
            """
            self.client.query(marketing_dedup_query).result()
            logger.info("Marketing table deduplication completed")

        # Deduplicate technical indicators table if indicators were processed
        if 'indicators' in categories:
            indicators_dedup_query = f"""
            CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.technical_indicators` AS
            SELECT * EXCEPT(row_num)
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER(PARTITION BY `date`, `ticker` ORDER BY `date` DESC) as row_num
                FROM `{self.project_id}.{self.dataset_id}.technical_indicators`
            )
            WHERE row_num = 1
            """
            self.client.query(indicators_dedup_query).result()
            logger.info("Technical indicators table deduplication completed")

    def send_notification(self, subject: str, body: str, recipient_email: str):
        """Send email notification using SMTP configuration.
        
        Args:
            subject: Email subject
            body: Email body content
            recipient_email: Email address to send notification to
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['username']
            msg['To'] = recipient_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            logger.info("Email notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")

    def send_crawler_notification(self, categories: List[str], recipient_email: str, success: bool = True):
        """Send crawler completion notification.
        
        Args:
            categories: List of categories that were processed
            recipient_email: Email address to send notification to
            success: Whether the crawler completed successfully
        """
        status = "Successfully" if success else "Failed to"
        subject = f"Market Data Crawler {status} Complete"
        
        body = f"""
Market Data Crawler has {status.lower()} completed the following operations:
{', '.join(categories)}

Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_notification(subject, body, recipient_email)

    def update_technical_indicators(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None):
        """Calculate and update technical indicators using BigQuery SQL.
        
        Args:
            start_date: Optional start date for calculation
            end_date: Optional end date for calculation
        """
        if start_date is None:
            # Get the last date in the technical indicators table
            query = f"""
            SELECT MAX(date) as last_date
            FROM `{self.project_id}.{self.dataset_id}.technical_indicators`
            """
            result = self.client.query(query).result()
            last_date = next(result).last_date

            if last_date is None:
                # If no data exists, start from 200 days ago to ensure enough data for MA200
                start_date = datetime.datetime.now() - datetime.timedelta(days=200)
            else:
                start_date = last_date + datetime.timedelta(days=1)

        if end_date is None:
            end_date = datetime.datetime.now()

        logger.info(f"Updating technical indicators from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Calculate technical indicators using window functions
        merge_query = f"""
        MERGE `{self.project_id}.{self.dataset_id}.technical_indicators` target
        USING (
            SELECT
                date,
                ticker,
                ma5,
                ma20,
                ma50,
                ma200,
                ema5,
                ema20,
                ema50,
                ema200
            FROM (
                SELECT
                    pd.date,
                    pd.ticker,
                    pd.adj_close,
                    pd.ma5,
                    pd.ma20,
                    pd.ma50,
                    pd.ma200,
                    pd.ma5 as ema5, 
                    pd.ma20 as ema20,
                    pd.ma50 as ema50,
                    pd.ma200 as ema200
                FROM (
                    SELECT
                        date,
                        ticker,
                        adj_close,
                        CASE
                            WHEN COUNT(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                            ) = 5 THEN AVG(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                            )
                        END as ma5,
                        CASE
                            WHEN COUNT(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                            ) = 20 THEN AVG(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                            )
                        END as ma20,
                        CASE
                            WHEN COUNT(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                            ) = 50 THEN AVG(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                            )
                        END as ma50,
                        CASE
                            WHEN COUNT(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                            ) = 200 THEN AVG(adj_close) OVER (
                                PARTITION BY ticker
                                ORDER BY date
                                ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                            )
                        END as ma200
                        FROM `{self.project_id}.{self.dataset_id}.marketing`
                        WHERE date >= '{start_date.strftime('%Y-%m-%d')}'
                          AND date <= '{end_date.strftime('%Y-%m-%d')}'
                ) pd
            ) ec
        ) source
        ON target.date = source.date AND target.ticker = source.ticker
        WHEN MATCHED THEN
            UPDATE SET
                ma5 = source.ma5,
                ma20 = source.ma20,
                ma50 = source.ma50,
                ma200 = source.ma200,
                ema5 = source.ema5,
                ema20 = source.ema20,
                ema50 = source.ema50,
                ema200 = source.ema200
        WHEN NOT MATCHED THEN
            INSERT (date, ticker, ma5, ma20, ma50, ma200, ema5, ema20, ema50, ema200)
            VALUES (source.date, source.ticker, source.ma5, source.ma20, source.ma50, source.ma200, 
                    source.ema5, source.ema20, source.ema50, source.ema200)
        """
        self.client.query(merge_query).result()
        logger.info("Technical indicators update completed successfully")

def main():
    """Main entry point for the crawler."""
    import argparse

    parser = argparse.ArgumentParser(description="Market Data Crawler")
    parser.add_argument(
        "--gcp-config",
        default="gcp-config.json",
        help="Path to GCP configuration JSON file"
    )
    parser.add_argument(
        "--smtp-config",
        default="smtp-config.json",
        help="Path to SMTP configuration JSON file"
    )
    parser.add_argument(
        "--email",
        default="zgxcassar@gmail.com",
        help="Email address to send notifications to"
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
        "--tickers",
        help="Comma-separated list of tickers to process. If not provided, processes all tickers from crawler_tickers table."
    )
    parser.add_argument(
        "--crawl",
        default="market,breadth,dedup",
        help="Comma-separated list of operations to perform. Options: market (crawl market data), indicators (calculate technical indicators), breadth (calculate market breadth), dedup (deduplicate tables). Default: market,breadth,dedup"
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

    # Parse tickers if provided
    specific_tickers = None
    if args.tickers:
        specific_tickers = [ticker.strip().upper() for ticker in args.tickers.split(',')]
        if not specific_tickers:
            logger.error("No valid tickers provided")
            sys.exit(1)

    # Parse crawler categories
    categories = [cat.strip().lower() for cat in args.crawl.split(',')]
    valid_categories = {'market', 'indicators', 'breadth', 'dedup'}
    invalid_categories = set(categories) - valid_categories
    
    if invalid_categories:
        logger.error(f"Invalid crawler categories: {', '.join(invalid_categories)}")
        logger.error(f"Valid categories are: {', '.join(valid_categories)}")
        sys.exit(1)

    crawler = MarketDataCrawler(args.gcp_config, args.smtp_config)

    try:
        # Execute requested operations
        if 'market' in categories:
            crawler.update_market_data(start_date, end_date, specific_tickers)
            logger.info("Market data update completed successfully")

        if 'indicators' in categories:
            crawler.update_technical_indicators(start_date, end_date)
            logger.info("Technical indicators update completed successfully")

        if 'breadth' in categories:
            crawler.update_market_breadth(start_date, end_date)
            logger.info("Market breadth update completed successfully")

        if 'dedup' in categories:
            crawler.deduplicate_tables(categories)
            logger.info("Deduplication completed successfully")

        # Send success notification
        crawler.send_crawler_notification(categories, args.email, success=True)

    except (ValueError, RuntimeError) as e:
        logger.error(f"Crawler failed: {str(e)}")
        #crawler.send_crawler_notification(categories, args.email, success=False)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        #crawler.send_crawler_notification(categories, args.email, success=False)
        sys.exit(1)

if __name__ == "__main__":
    main() 