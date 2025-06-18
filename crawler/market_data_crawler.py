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
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarketDataCrawler:
    def __init__(self, gcp_config_path: str, smtp_config_path: str):
        """Initialize the MarketDataCrawler with BigQuery credentials."""
        # Set timezone
        self.timezone = pytz.timezone("America/New_York")

        # Load GCP configuration
        with open(gcp_config_path, "r") as f:
            gcp_config = json.load(f)

        self.credentials = service_account.Credentials.from_service_account_file(
            gcp_config["credentials_path"]
        )
        self.client = bigquery.Client(
            credentials=self.credentials, project=gcp_config["project_id"]
        )
        self.project_id = gcp_config["project_id"]
        self.dataset_id = "market_data"
        self.market_breadth_table = "market_breadth"

        # Load SMTP configuration
        with open(smtp_config_path, "r") as f:
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
            update_time TIMESTAMP,
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
            update_time TIMESTAMP,
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
            update_time TIMESTAMP,
            PRIMARY KEY(date) NOT ENFORCED
        )
        """
        self.client.query(breadth_query).result()
        logger.info(f"Table {self.market_breadth_table} created or already exists")

    def get_symbols_from_bigquery(
        self, is_sp500: bool = False, specific_tickers: Optional[List[str]] = None
    ) -> List[str]:
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
            raise ValueError(
                f"No {'S&P 500 ' if is_sp500 else ''}symbols found in crawler_tickers table"
            )

        logger.info(
            f"Loaded {len(symbols)} {'S&P 500 ' if is_sp500 else ''}symbols from BigQuery"
        )
        return symbols

    def update_market_breadth(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ):
        """Directly calculate and update market breadth metrics using a MERGE operation."""
        ny_tz = pytz.timezone("America/New_York")
        if end_date is None:
            end_date = datetime.datetime.now(ny_tz)
        elif end_date.tzinfo is None:
            end_date = ny_tz.localize(end_date)

        if start_date is None:
            start_date = end_date - datetime.timedelta(days=30)
        elif start_date.tzinfo is None:
            start_date = ny_tz.localize(start_date)

        logger.info(
            f"Updating market breadth from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        merge_sql = f"""
        MERGE `{self.project_id}.{self.dataset_id}.{self.market_breadth_table}` T
        USING (
            SELECT
                m.date,
                COUNTIF(m.adj_close > ti.ma20) as ma20_above,
                COUNTIF(m.adj_close <= ti.ma20) as ma20_below,
                SAFE_DIVIDE(COUNTIF(m.adj_close > ti.ma20), COUNT(*)) as ma20_ratio,
                COUNTIF(m.adj_close > ti.ma50) as ma50_above,
                COUNTIF(m.adj_close <= ti.ma50) as ma50_below,
                SAFE_DIVIDE(COUNTIF(m.adj_close > ti.ma50), COUNT(*)) as ma50_ratio,
                COUNTIF(m.adj_close > ti.ma200) as ma200_above,
                COUNTIF(m.adj_close <= ti.ma200) as ma200_below,
                SAFE_DIVIDE(COUNTIF(m.adj_close > ti.ma200), COUNT(*)) as ma200_ratio,
                COUNTIF(m.adj_close > ti.ema20) as ema20_above,
                COUNTIF(m.adj_close <= ti.ema20) as ema20_below,
                SAFE_DIVIDE(COUNTIF(m.adj_close > ti.ema20), COUNT(*)) as ema20_ratio,
                COUNTIF(m.adj_close > ti.ema50) as ema50_above,
                COUNTIF(m.adj_close <= ti.ema50) as ema50_below,
                SAFE_DIVIDE(COUNTIF(m.adj_close > ti.ema50), COUNT(*)) as ema50_ratio,
                COUNTIF(m.adj_close > ti.ema200) as ema200_above,
                COUNTIF(m.adj_close <= ti.ema200) as ema200_below,
                SAFE_DIVIDE(COUNTIF(m.adj_close > ti.ema200), COUNT(*)) as ema200_ratio,
                CURRENT_TIMESTAMP() as update_time
            FROM `{self.project_id}.{self.dataset_id}.marketing` m
            JOIN `{self.project_id}.{self.dataset_id}.technical_indicators` ti
                ON m.date = ti.date AND m.ticker = ti.ticker
            WHERE m.date >= '{start_date.strftime('%Y-%m-%d')}'
                AND m.date <= '{end_date.strftime('%Y-%m-%d')}'
                AND m.ticker IN (SELECT ticker FROM `{self.project_id}.{self.dataset_id}.crawler_tickers` WHERE is_sp500=TRUE)
            GROUP BY m.date
        ) S
        ON T.date = S.date
        WHEN MATCHED THEN
            UPDATE SET
                T.ma20_above = S.ma20_above, T.ma20_below = S.ma20_below, T.ma20_ratio = S.ma20_ratio,
                T.ma50_above = S.ma50_above, T.ma50_below = S.ma50_below, T.ma50_ratio = S.ma50_ratio,
                T.ma200_above = S.ma200_above, T.ma200_below = S.ma200_below, T.ma200_ratio = S.ma200_ratio,
                T.ema20_above = S.ema20_above, T.ema20_below = S.ema20_below, T.ema20_ratio = S.ema20_ratio,
                T.ema50_above = S.ema50_above, T.ema50_below = S.ema50_below, T.ema50_ratio = S.ema50_ratio,
                T.ema200_above = S.ema200_above, T.ema200_below = S.ema200_below, T.ema200_ratio = S.ema200_ratio,
                T.update_time = S.update_time
        WHEN NOT MATCHED THEN
            INSERT (date, ma20_above, ma20_below, ma20_ratio, ma50_above, ma50_below, ma50_ratio, ma200_above, ma200_below, ma200_ratio, ema20_above, ema20_below, ema20_ratio, ema50_above, ema50_below, ema50_ratio, ema200_above, ema200_below, ema200_ratio, update_time)
            VALUES (S.date, S.ma20_above, S.ma20_below, S.ma20_ratio, S.ma50_above, S.ma50_below, S.ma50_ratio, S.ma200_above, S.ma200_below, S.ma200_ratio, S.ema20_above, S.ema20_below, S.ema20_ratio, S.ema50_above, S.ema50_below, S.ema50_ratio, S.ema200_above, S.ema200_below, S.ema200_ratio, S.update_time)
        """

        self.client.query(merge_sql).result()
        logger.info("Market breadth update completed successfully")

    def update_market_data(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        specific_tickers: Optional[List[str]] = None,
    ):
        """Update market data using a MERGE operation to ensure idempotency.

        Args:
            start_date: Optional start date for data update (interpreted as New York time). Defaults to 30 days ago.
            end_date: Optional end date for data update (interpreted as New York time). Defaults to now.
            specific_tickers: Optional list of specific tickers to process.
        """
        symbols = self.get_symbols_from_bigquery(specific_tickers=specific_tickers)
        logger.info(f"Processing {len(symbols)} symbols")

        ny_tz = pytz.timezone("America/New_York")
        if end_date is None:
            end_date = datetime.datetime.now(ny_tz)
        elif end_date.tzinfo is None:
            end_date = ny_tz.localize(end_date)

        if start_date is None:
            start_date = end_date - datetime.timedelta(days=3)
        elif start_date.tzinfo is None:
            start_date = ny_tz.localize(start_date)

        logger.info(
            f"Updating data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (New York time)"
        )

        start_utc, end_utc = start_date.astimezone(pytz.UTC), end_date.astimezone(pytz.UTC)
        logger.info(f"Downloading data from {start_utc.strftime('%Y-%m-%d')} to {end_utc.strftime('%Y-%m-%d')} UTC...")

        df = yf.download(
            symbols, start=start_utc, end=end_utc, progress=False, auto_adjust=False, actions=True, group_by="ticker"
        )
        if df.empty:
            logger.error("No data downloaded from yfinance.")
            return

        all_symbol_data = []
        for symbol in symbols:
            symbol_data = df.get(symbol)
            if symbol_data is None or symbol_data.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            symbol_data = symbol_data.copy()
            symbol_data.loc[:, "ticker"] = symbol
            all_symbol_data.append(symbol_data)

        if not all_symbol_data:
            logger.error("No valid data processed for any symbols.")
            return
            
        full_df = pd.concat(all_symbol_data).reset_index()
        full_df.rename(
            columns={
                "Date": "date", "Open": "open", "High": "high", "Low": "low", 
                "Close": "raw_close", "Adj Close": "adj_close", "Volume": "volume",
                "Dividends": "dividend", "Stock Splits": "split_ratio", "Capital Gains": "capital_gains"
            }, inplace=True
        )

        # Filter out rows with no trading volume to avoid inserting zero-volume days.
        full_df = full_df[full_df['volume'] > 0].copy()
        if full_df.empty:
            logger.warning("No data with trading volume > 0 found for any symbols.")
            return
        print(full_df.head())

        full_df["date"] = pd.to_datetime(full_df["date"]).dt.date
        full_df["volume"] = full_df["volume"].round(0).astype(pd.Int64Dtype())
        full_df["update_time"] = datetime.datetime.now(ny_tz)
        
        # Use MERGE for idempotent writes
        temp_table_id = f"{self.project_id}.{self.dataset_id}.temp_market_data_{int(time.time())}"
        target_table_id = f"{self.project_id}.{self.dataset_id}.marketing"

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.client.load_table_from_dataframe(full_df, temp_table_id, job_config=job_config).result()
        
        merge_sql = f"""
        MERGE `{target_table_id}` T
        USING `{temp_table_id}` S
        ON T.date = S.date AND T.ticker = S.ticker
        WHEN MATCHED THEN
            UPDATE SET
                T.open = S.open, T.high = S.high, T.low = S.low, T.raw_close = S.raw_close,
                T.close = S.adj_close, T.adj_close = S.adj_close, T.dividend = S.dividend,
                T.volume = S.volume, T.capital_gains = S.capital_gains, 
                T.split_ratio = S.split_ratio, T.update_time = S.update_time
        WHEN NOT MATCHED THEN
            INSERT (date, ticker, open, high, low, raw_close, close, adj_close, dividend, volume, capital_gains, split_ratio, update_time)
            VALUES (S.date, S.ticker, S.open, S.high, S.low, S.raw_close, S.adj_close, S.adj_close, S.dividend, S.volume, S.capital_gains, S.split_ratio, S.update_time)
        """
        self.client.query(merge_sql).result()
        self.client.delete_table(temp_table_id, not_found_ok=True)
        logger.info(f"Successfully merged market data into {target_table_id}")

    def send_notification(self, subject: str, body: str, recipient_email: str):
        """Send email notification using SMTP configuration.

        Args:
            subject: Email subject
            body: Email body content
            recipient_email: Email address to send notification to
        """
        try:
            msg = MIMEMultipart()
            msg["From"] = self.smtp_config["username"]
            msg["To"] = recipient_email
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(
                self.smtp_config["server"], self.smtp_config["port"]
            ) as server:
                if self.smtp_config.get("use_tls", True):
                    server.starttls()
                server.login(self.smtp_config["username"], self.smtp_config["password"])
                server.send_message(msg)

            logger.info("Email notification sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")

    def send_crawler_notification(
        self, categories: List[str], recipient_email: str, success: bool = True
    ):
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

    def update_technical_indicators(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ):
        """Calculate and update technical indicators using a MERGE operation.
        
        Defaults to a 30-day window but automatically handles a longer lookback period for accurate calculations.
        """
        ny_tz = pytz.timezone("America/New_York")
        if end_date is None:
            end_date = datetime.datetime.now(ny_tz)
        elif end_date.tzinfo is None:
            end_date = ny_tz.localize(end_date)

        if start_date is None:
            start_date = end_date - datetime.timedelta(days=30)
        elif start_date.tzinfo is None:
            start_date = ny_tz.localize(start_date)

        # Lookback period to ensure enough data for long-term MAs (e.g., MA200)
        lookback_start = start_date - datetime.timedelta(days=400)

        logger.info(
            f"Updating technical indicators from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        merge_query = f"""
        MERGE `{self.project_id}.{self.dataset_id}.technical_indicators` T
        USING (
            WITH price_data AS (
                SELECT date, ticker, adj_close
                FROM `{self.project_id}.{self.dataset_id}.marketing`
                WHERE date >= '{lookback_start.strftime('%Y-%m-%d')}' AND date <= '{end_date.strftime('%Y-%m-%d')}'
            ),
            indicator_calcs AS (
                SELECT
                    date,
                    ticker,
                    AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS ma5,
                    AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
                    AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma50,
                    AVG(adj_close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ma200,
                    (
                        adj_close * (2.0 / (5 + 1)) +
                        COALESCE(LAG(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date), adj_close) * (1 - (2.0 / (5 + 1)))
                    ) AS ema5,
                    (
                        adj_close * (2.0 / (20 + 1)) +
                        COALESCE(LAG(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date), adj_close) * (1 - (2.0 / (20 + 1)))
                    ) AS ema20,
                    (
                        adj_close * (2.0 / (50 + 1)) +
                        COALESCE(LAG(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date), adj_close) * (1 - (2.0 / (50 + 1)))
                    ) AS ema50,
                    (
                        adj_close * (2.0 / (200 + 1)) +
                        COALESCE(LAG(adj_close, 1) OVER (PARTITION BY ticker ORDER BY date), adj_close) * (1 - (2.0 / (200 + 1)))
                    ) AS ema200
                FROM price_data
            )
            SELECT *, CURRENT_TIMESTAMP() as update_time
            FROM indicator_calcs
            WHERE date >= '{start_date.strftime('%Y-%m-%d')}' AND date <= '{end_date.strftime('%Y-%m-%d')}'
        ) S
        ON T.date = S.date AND T.ticker = S.ticker
        WHEN MATCHED THEN
            UPDATE SET
                T.ma5 = S.ma5, T.ma20 = S.ma20, T.ma50 = S.ma50, T.ma200 = S.ma200,
                T.ema5 = S.ema5, T.ema20 = S.ema20, T.ema50 = S.ema50, T.ema200 = S.ema200,
                T.update_time = S.update_time
        WHEN NOT MATCHED THEN
            INSERT (date, ticker, ma5, ma20, ma50, ma200, ema5, ema20, ema50, ema200, update_time)
            VALUES (S.date, S.ticker, S.ma5, S.ma20, S.ma50, S.ma200, S.ema5, S.ema20, S.ema50, S.ema200, S.update_time)
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
        help="Path to GCP configuration JSON file",
    )
    parser.add_argument(
        "--smtp-config",
        default="smtp-config.json",
        help="Path to SMTP configuration JSON file",
    )
    parser.add_argument(
        "--email",
        default="zgxcassar@gmail.com",
        help="Email address to send notifications to",
    )
    parser.add_argument(
        "--start-date",
        help="Start date for data update (YYYY-MM-DD). If not provided, uses last 30 days.",
    )
    parser.add_argument(
        "--end-date",
        help="End date for data update (YYYY-MM-DD). If not provided, uses current date.",
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated list of tickers to process. If not provided, processes all tickers from crawler_tickers table.",
    )
    parser.add_argument(
        "--crawl",
        default="market,indicators,breadth",
        help="Comma-separated list of operations to perform. Options: market, indicators, breadth. Default: market,indicators,breadth",
    )

    args = parser.parse_args()

    # Parse dates if provided
    start_date = None
    end_date = None

    if args.start_date:
        try:
            # Parse the date and localize it to New York timezone
            start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid start date format. Use YYYY-MM-DD")
            sys.exit(1)

    if args.end_date:
        try:
            # Parse the date and localize it to New York timezone
            end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid end date format. Use YYYY-MM-DD")
            sys.exit(1)

    # Parse tickers if provided
    specific_tickers = None
    if args.tickers:
        specific_tickers = [
            ticker.strip().upper() for ticker in args.tickers.split(",")
        ]
        if not specific_tickers:
            logger.error("No valid tickers provided")
            sys.exit(1)

    # Parse crawler categories
    categories = [cat.strip().lower() for cat in args.crawl.split(",")]
    valid_categories = {"market", "indicators", "breadth"}
    invalid_categories = set(categories) - valid_categories

    if invalid_categories:
        logger.error(f"Invalid crawler categories: {', '.join(invalid_categories)}")
        logger.error(f"Valid categories are: {', '.join(valid_categories)}")
        sys.exit(1)

    crawler = MarketDataCrawler(args.gcp_config, args.smtp_config)

    try:
        # Execute requested operations
        if "market" in categories:
            crawler.update_market_data(start_date, end_date, specific_tickers)

        if "indicators" in categories:
            crawler.update_technical_indicators(start_date, end_date)

        if "breadth" in categories:
            crawler.update_market_breadth(start_date, end_date)

        # Send success notification
        crawler.send_crawler_notification(categories, args.email, success=True)

    except (ValueError, RuntimeError) as e:
        logger.error(f"Crawler failed: {str(e)}")
        # crawler.send_crawler_notification(categories, args.email, success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
