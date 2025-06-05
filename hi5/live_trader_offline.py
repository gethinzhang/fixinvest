import os
import datetime
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import icalendar
import sys
import math
from google.cloud import bigquery
from google.oauth2 import service_account

# Import our strategy components
from hi5.hi5 import Hi5State, InMemoryStorageEngine, LocalStorageEngine
from hi5.hi5 import GCPStorageEngine


@dataclass
class TradingPlan:
    """Represents a trading plan entry"""

    date: datetime.datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    reason: str
    current_price: float
    target_value: float
    execution_notes: str = ""


class EmailManager:
    """Manages email and calendar invitations via SMTP"""

    def __init__(
        self, smtp_server: str, smtp_port: int, smtp_username: str, smtp_password: str
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password

    def create_calendar_invitation(self, plan: TradingPlan) -> bytes:
        """Create an ICS file for the trading plan"""
        cal = icalendar.Calendar()
        cal.add('prodid', '-//Hi5 Trading Plan//EN')
        cal.add('version', '2.0')
        cal.add('method', 'REQUEST')  # Add method REQUEST for calendar invitations

        # Create event
        event = icalendar.Event()
        event.add('summary', f'Hi5 Trade: {plan.action} {plan.shares} {plan.ticker}')
        #event.add('status', 'CONFIRMED')  # Add status
        #event.add('transp', 'OPAQUE')  # Add transparency
        #event.add('sequence', 0)  # Add sequence number

        # Set time to 8:30 AM ET on next trading day
        et_tz = pytz.timezone('America/New_York')
        event_date = plan.date
        while event_date.weekday() >= 5:  # Skip weekends
            event_date += datetime.timedelta(days=1)
        
        start_time = et_tz.localize(
            datetime.datetime.combine(event_date.date(), datetime.time(8, 30))
        )

        end_time = start_time + datetime.timedelta(minutes=30)

        event.add('dtstart', start_time)
        event.add('dtend', end_time)
        event.add('dtstamp', datetime.datetime.now(pytz.UTC))  # Add timezone
        event.add('created', datetime.datetime.now(pytz.UTC))  # Add creation time
        event.add('last-modified', datetime.datetime.now(pytz.UTC))  # Add last modified

        # Add organizer
        event.add('organizer', f'MAILTO:{self.smtp_username}')
        event.add('attendee', f'MAILTO:{self.smtp_username}')

        # Add alarm/reminder 30 minutes before
        alarm = icalendar.Alarm()
        alarm.add('action', 'DISPLAY')
        alarm.add('description', f'Reminder: Hi5 Trade {plan.action} {plan.shares} {plan.ticker}')
        alarm.add('trigger', datetime.timedelta(minutes=-30))
        event.add_component(alarm)

        # Add description
        description = f"""Hi5 Trading Plan

Action: {plan.action}
Ticker: {plan.ticker}
Shares: {plan.shares}
Current Price: ${plan.current_price:.2f}
Target Value: ${plan.target_value:.2f}
Reason: {plan.reason}

Execution Notes:
{plan.execution_notes}

Pre-Trade Checklist:
□ Check pre-market movement
□ Verify no pending news/earnings
□ Confirm market conditions align
□ Set limit order if high volatility
□ Document actual execution price

Remember: This is a plan, not a commitment. Adjust based on market conditions.
"""
        event.add('description', description)

        # Add to calendar
        cal.add_component(event)

        return cal.to_ical()

    def send_calendar_invitation(self, recipient_email: str, plan: TradingPlan):
        """Send calendar invitation via email"""
        try:
            # Create message
            msg = MIMEMultipart('mixed')
            msg['Subject'] = f'Hi5 Trading Plan: {plan.action} {plan.ticker}'
            msg['From'] = self.smtp_username
            msg['To'] = recipient_email

            # Add text body
            body = f"""Hi5 Trading Plan

Action: {plan.action}
Ticker: {plan.ticker}
Shares: {plan.shares}
Current Price: ${plan.current_price:.2f}
Target Value: ${plan.target_value:.2f}
Reason: {plan.reason}

Execution Notes:
{plan.execution_notes}

Pre-Trade Checklist:
□ Check pre-market movement
□ Verify no pending news/earnings
□ Confirm market conditions align
□ Set limit order if high volatility
□ Document actual execution price

Remember: This is a plan, not a commitment. Adjust based on market conditions.
"""
            msg.attach(MIMEText(body, 'plain'))

            # Add calendar invitation
            ics_data = self.create_calendar_invitation(plan)
            part = MIMEBase('text', 'calendar', method='REQUEST')
            part.set_payload(ics_data)
            encoders.encode_base64(part)
            part.add_header('Content-Type', 'text/calendar; method=REQUEST; charset=UTF-8')
            part.add_header('Content-Disposition', 'attachment; filename=invite.ics')
            msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"Calendar invitation sent to {recipient_email}")
            return True

        except Exception as e:
            print(f"Error sending calendar invitation: {e}")
            return False

    def send_plan_summary_email(self, recipient_email: str, plans: List[TradingPlan]):
        """Send email summary of trading plans"""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = (
                f"Hi5 Trading Plan - {datetime.datetime.now().strftime('%Y-%m-%d')}"
            )
            msg["From"] = self.smtp_username
            msg["To"] = recipient_email

            # Build HTML body
            total_investment = sum(p.target_value for p in plans)

            body = f"""
            <html>
                <body>
                    <h2>Hi5 Trading Plan Summary</h2>
                    <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Total Investment:</strong> ${total_investment:,.2f}</p>
                    <hr>
                    <h3>Planned Trades:</h3>
                    <table border="1" cellpadding="5" cellspacing="0">
                        <tr>
                            <th>Ticker</th>
                            <th>Action</th>
                            <th>Shares</th>
                            <th>Price</th>
                            <th>Value</th>
                            <th>Reason</th>
                        </tr>
            """

            for plan in plans:
                body += f"""
                        <tr>
                            <td>{plan.ticker}</td>
                            <td>{plan.action}</td>
                            <td>{plan.shares}</td>
                            <td>${plan.current_price:.2f}</td>
                            <td>${plan.target_value:.2f}</td>
                            <td>{plan.reason}</td>
                        </tr>
                """

            body += """
                    </table>
                    <hr>
                    <p><strong>Next Steps:</strong></p>
                    <ol>
                        <li>Check your calendar for trade reminders</li>
                        <li>Execute trades during market hours</li>
                        <li>Update tracking spreadsheet after execution</li>
                    </ol>
                    <p style="color: red;"><strong>Remember:</strong> This is a plan, not a commitment. Adjust based on market conditions.</p>
                </body>
            </html>
            """

            html_part = MIMEText(body, "html")
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"Plan summary email sent to {recipient_email}")
            return True

        except Exception as e:
            print(f"Error sending email: {e}")
            return False

    def create_daily_calendar_invitation(
        self, plans: List[TradingPlan], trading_day: datetime.datetime
    ) -> bytes:
        """Create an ICS file for all trading plans for a day"""
        cal = icalendar.Calendar()
        cal.add('prodid', '-//Hi5 Trading Plan//EN')
        cal.add('version', '2.0')
        cal.add('method', 'REQUEST')  # Add method REQUEST for calendar invitations

        event = icalendar.Event()
        event.add('summary', f'Hi5 Trading Plan for {trading_day.strftime("%Y-%m-%d")}')
        #event.add('status', 'CONFIRMED')  # Add status
        #event.add('transp', 'OPAQUE')  # Add transparency
        #event.add('sequence', 0)  # Add sequence number

        et_tz = pytz.timezone('America/New_York')
        # Set time to 8:30 AM ET
        start_time = et_tz.localize(
            datetime.datetime.combine(trading_day.date(), datetime.time(8, 30))
        )
        end_time = start_time + datetime.timedelta(minutes=30)
        event.add('dtstart', start_time)
        event.add('dtend', end_time)
        event.add('dtstamp', datetime.datetime.now(pytz.UTC))  # Add timezone
        event.add('created', datetime.datetime.now(pytz.UTC))  # Add creation time
        event.add('last-modified', datetime.datetime.now(pytz.UTC))  # Add last modified

        # Add organizer
        event.add('organizer', f'MAILTO:{self.smtp_username}')
        event.add('attendee', f'MAILTO:{self.smtp_username}')

        # Add alarm/reminder 30 minutes before
        alarm = icalendar.Alarm()
        alarm.add('action', 'DISPLAY')
        alarm.add('description', f'Reminder: Hi5 Trading Plan for {trading_day.strftime("%Y-%m-%d")}')
        alarm.add('trigger', datetime.timedelta(minutes=-30))
        event.add_component(alarm)

        # Build description
        description = "Hi5 Trading Plan\n\n"
        for plan in plans:
            description += f"{plan.action} {plan.shares} {plan.ticker} @ ${plan.current_price:.2f} (Reason: {plan.reason})\n"
        description += (
            "\nPre-Trade Checklist:\n"
            "□ Check pre-market movement\n"
            "□ Verify no pending news/earnings\n"
            "□ Confirm market conditions align\n"
            "□ Set limit order if high volatility\n"
            "□ Document actual execution price\n"
        )
        event.add('description', description)
        cal.add_component(event)
        return cal.to_ical()

    def send_daily_calendar_invitation(
        self,
        recipient_email: str,
        plans: List[TradingPlan],
        trading_day: datetime.datetime,
        html_summary: str,
    ):
        """Send a single calendar invitation for all trades on a day, with HTML summary as the main body"""
        try:
            msg = MIMEMultipart('mixed')
            msg['Subject'] = f'Hi5 Trading Plan: {trading_day.strftime("%Y-%m-%d")}'
            msg['From'] = self.smtp_username
            msg['To'] = recipient_email

            # Add HTML summary as the main body
            msg.attach(MIMEText(html_summary, 'html'))

            # Add calendar invitation
            ics_data = self.create_daily_calendar_invitation(plans, trading_day)
            part = MIMEBase('text', 'calendar', method='REQUEST')
            part.set_payload(ics_data)
            encoders.encode_base64(part)
            part.add_header('Content-Type', 'text/calendar; method=REQUEST; charset=UTF-8')
            part.add_header('Content-Disposition', 'attachment; filename=invite.ics')
            msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"Daily calendar invitation (with summary) sent to {recipient_email}")
            return True
        except Exception as e:
            print(f"Error sending daily calendar invitation: {e}")
            return False


def parse_test_dates(test_date_arg):
    """Parse --test-date argument, supporting single date or range."""
    if not test_date_arg:
        return None
    if ":" in test_date_arg:
        start_str, end_str = test_date_arg.split(":", 1)
        start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d")
        if end_date < start_date:
            raise ValueError("End date must not be before start date")
        num_days = (end_date - start_date).days + 1
        return [start_date + datetime.timedelta(days=i) for i in range(num_days)]
    else:
        return [datetime.datetime.strptime(test_date_arg, "%Y-%m-%d")]


class OfflineTradingPlanner:
    """Trading planner that works without IBKR connection"""

    def __init__(
        self,
        recipient_email=None,
        smtp_server=None,
        smtp_port=None,
        smtp_username=None,
        smtp_password=None,
        test_date=None,
        engine="local",
        bucket_name=None,
        blob_name=None,
        credentials_path=None,
        use_bigquery=False,
        project_id=None,
    ):
        self.recipient_email = recipient_email
        # test_date can be None, a list of datetimes, or a single datetime
        if test_date is None:
            self.test_dates = None
        elif isinstance(test_date, list):
            self.test_dates = test_date
        else:
            self.test_dates = [test_date]
        self.engine = engine
        self.use_bigquery = use_bigquery
        self.project_id = project_id
        
        # Initialize BigQuery client if needed
        if use_bigquery:
            if not credentials_path or not project_id:
                raise ValueError("credentials_path and project_id are required when use_bigquery is True")
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.bq_client = bigquery.Client(credentials=credentials, project=project_id)
        
        # Email services
        if recipient_email:
            if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
                print(
                    "Error: SMTP settings are required when email notifications are enabled"
                )
                print("Please provide SMTP configuration in smtp_config.json")
                sys.exit(1)
            self.email_manager = EmailManager(
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                smtp_username=smtp_username,
                smtp_password=smtp_password,
            )
        else:
            self.email_manager = None
        # Strategy setup: choose storage engine
        if engine == "local":
            storage_engine = LocalStorageEngine("./hi5_offline_state.json")
        elif engine == "memory":
            storage_engine = InMemoryStorageEngine()
        elif engine == "gcp":
            if not bucket_name or not blob_name:
                print(
                    "Error: bucket_name and blob_name must be provided for GCP engine."
                )
                sys.exit(1)
            storage_engine = GCPStorageEngine(
                bucket_name=bucket_name,
                blob_name=blob_name,
                credentials_path=credentials_path,
            )
        else:
            raise ValueError(f"Unknown engine: {engine}")
        self.state = Hi5State(storage_engine=storage_engine)
        # Add month_start_date to state if not present
        if not hasattr(self.state, 'month_start_date'):
            self.state.month_start_date = None
        # Trading parameters (from Hi5Strategy)
        self.tickers = ["VUG", "VO", "MOAT", "PFF", "VNQ"]
        self.benchmark_ticker = "RSP"
        self.cash_per_contribution = 10000
        self.non_resident_tax_rate = 0.3
        # Market data cache
        self.price_cache = {}

    def _get_market_data_from_bigquery(self, trading_day: datetime.datetime) -> pd.DataFrame:
        """Fetch market data from BigQuery for the given trading day"""
        query = f"""
        WITH prev_day AS (
            SELECT 
                ticker,
                date,
                close,
                LAG(close) OVER (PARTITION BY ticker ORDER BY date) as prev_close
            FROM `{self.project_id}.market_data.marketing`
            WHERE date <= DATE('{trading_day.strftime('%Y-%m-%d')}')
        )
        SELECT 
            ticker,
            date,
            close,
            prev_close,
            (close - prev_close) as change,
            ((close - prev_close) / prev_close * 100) as change_percent
        FROM prev_day
        WHERE date = DATE('{trading_day.strftime('%Y-%m-%d')}')
        AND ticker IN UNNEST(@tickers)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("tickers", "STRING", self.tickers + [self.benchmark_ticker])
            ]
        )
        
        query_job = self.bq_client.query(query, job_config=job_config)
        results = query_job.result()
        
        # Convert to DataFrame and ensure correct types
        df = results.to_dataframe()
        if df.empty:
            raise RuntimeError(f"No data found in BigQuery for {trading_day.strftime('%Y-%m-%d')}")
        
        # Create a new DataFrame with only essential columns and proper types
        clean_df = pd.DataFrame({
            'ticker': df['ticker'].astype(str),
            'date': pd.to_datetime(df['date']).dt.date,
            'close': df['close'].astype(float),
            'prev_close': df['prev_close'].astype(float),
            'change': df['change'].astype(float),
            'change_percent': df['change_percent'].astype(float)
        })
            
        return clean_df

    def update_market_data(self, trading_day, preloaded_data=None):
        """
        Fetch market data for the previous trading day.
        If preloaded_data is provided (multi-ticker DataFrame), use it.
        If use_bigquery is True, fetch from BigQuery instead of yfinance.
        """
        all_tickers = self.tickers + [self.benchmark_ticker]
        self.price_cache = {}

        if preloaded_data is not None:
            print(f"Using preloaded data for {trading_day.strftime('%Y-%m-%d')}")
            # The DataFrame has columns like ('VUG', 'Close'), ('VO', 'Close'), etc.
            df = preloaded_data
            # Get all available dates
            available_dates = df.index
            if pd.Timestamp(trading_day) not in available_dates:
                raise RuntimeError(
                    f"No data for trading day {trading_day.strftime('%Y-%m-%d')}"
                )
            prev_idx = available_dates.get_loc(pd.Timestamp(trading_day)) - 1
            if prev_idx < 0:
                raise RuntimeError(
                    f"No previous trading day for {trading_day.strftime('%Y-%m-%d')}"
                )
            prev_trading_day = available_dates[prev_idx]
            prev_prev_idx = prev_idx - 1 if prev_idx - 1 >= 0 else None
            for ticker in all_tickers:
                try:
                    closes = df[(ticker, "Close")]
                    close_price = closes.loc[prev_trading_day]
                    prev_close = (
                        closes.iloc[prev_prev_idx]
                        if prev_prev_idx is not None
                        else close_price
                    )
                    change = close_price - prev_close
                    change_percent = (
                        (change / prev_close * 100)
                        if pd.notna(prev_close) and prev_close != 0
                        else 0
                    )
                    self.price_cache[ticker] = {
                        "current": close_price,
                        "previous_close": prev_close,
                        "change": change,
                        "change_percent": change_percent,
                    }
                    print(
                        f"{ticker}: ${close_price:.2f} (Prev: ${prev_close:.2f}, {change_percent:+.2f}%)"
                    )
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")
                    raise
        elif self.use_bigquery:
            print(f"Fetching data from BigQuery for {trading_day.strftime('%Y-%m-%d')}")
            try:
                df = self._get_market_data_from_bigquery(trading_day)
                for _, row in df.iterrows():
                    ticker = row['ticker']
                    self.price_cache[ticker] = {
                        "current": row['close'],
                        "previous_close": row['prev_close'],
                        "change": row['change'],
                        "change_percent": row['change_percent'],
                    }
                    print(
                        f"{ticker}: ${row['close']:.2f} (Prev: ${row['prev_close']:.2f}, {row['change_percent']:+.2f}%)"
                    )
            except Exception as e:
                print(f"Error fetching data from BigQuery: {e}")
                raise
        else:
            # Single-date or continuous mode: download just for this ticker
            prev_trading_day = trading_day - datetime.timedelta(days=1)
            for ticker in all_tickers:
                try:
                    df = yf.download(
                        ticker,
                        start=(prev_trading_day - datetime.timedelta(days=7)).strftime(
                            "%Y-%m-%d"
                        ),
                        end=(trading_day + datetime.timedelta(days=1)).strftime(
                            "%Y-%m-%d"
                        ),
                        progress=False,
                        auto_adjust=True,
                    )
                    # Create a clean DataFrame with only essential columns and proper types
                    clean_df = pd.DataFrame({
                        'date': pd.to_datetime(df.index).date,
                        'close': df['Close'].astype(float),
                        'volume': df['Volume'].round(0).astype(pd.Int64Dtype())
                    })
                    
                    closes = clean_df['close'].loc[: prev_trading_day.strftime("%Y-%m-%d")]
                    if len(closes) < 1:
                        raise RuntimeError(
                            f"No data for {ticker} on {prev_trading_day.strftime('%Y-%m-%d')}"
                        )
                    close_price = closes.iloc[-1]
                    prev_close = closes.iloc[-2] if len(closes) >= 2 else close_price
                    change = close_price - prev_close
                    change_percent = (
                        (change / prev_close * 100)
                        if pd.notna(prev_close) and prev_close != 0
                        else 0
                    )
                    self.price_cache[ticker] = {
                        "current": close_price,
                        "previous_close": prev_close,
                        "change": change,
                        "change_percent": change_percent,
                    }
                    print(
                        f"{ticker}: ${close_price:.2f} (Prev: ${prev_close:.2f}, {change_percent:+.2f}%)"
                    )
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")
                    raise

    def check_hi5_signals(self, trading_day, preloaded_data=None) -> List[TradingPlan]:
        """Check Hi5 strategy for trading signals"""
        plans = []
        self.update_market_data(trading_day, preloaded_data=preloaded_data)

        # Get RSP data for strategy logic
        rsp_data = self.price_cache.get(self.benchmark_ticker, {})
        rsp_price = rsp_data.get("current", 0)
        rsp_prev_close = rsp_data.get("previous_close", rsp_price)
        rsp_month_start = self.state.rsp_month_start_price
        month_start_date = self.state.month_start_date

        if not rsp_price:
            print("Warning: Could not get RSP price")
            return plans

        # Month refresh logic
        if (
            self.state.current_month is None
            or self.state.current_month != trading_day.month
        ):
            # Find the first natural trading day of the month in the data
            if preloaded_data is not None:
                # Multi-index DataFrame: (ticker, field)
                dates = preloaded_data.index
                # Find first date in this month with RSP data
                for dt in dates:
                    if dt.month == trading_day.month:
                        try:
                            price = preloaded_data[(self.benchmark_ticker, "Close")].loc[dt]
                            if not pd.isna(price):
                                self.state.rsp_month_start_price = price
                                self.state.month_start_date = dt.strftime('%Y-%m-%d')
                                break
                        except Exception:
                            continue
                else:
                    self.state.rsp_month_start_price = rsp_price
                    self.state.month_start_date = trading_day.strftime('%Y-%m-%d')
            else:
                # Fallback: use current day
                self.state.rsp_month_start_price = rsp_price
                self.state.month_start_date = trading_day.strftime('%Y-%m-%d')
            self.state.current_month = trading_day.month
            self.state.first_exec = False
            self.state.second_exec = False
            self.state.third_exec = False
            self.state.save_state()
            print(f"Month refreshed: {trading_day.month}, RSP start: ${self.state.rsp_month_start_price:.2f} on {self.state.month_start_date}")

        # Check for buy signals
        # 1. First monthly investment
        if not self.state.first_exec:
            daily_return = (rsp_price / rsp_prev_close - 1) if rsp_prev_close else 0
            if daily_return <= -0.01:
                reason = (
                    f"RSP daily drop <= -1%: Month start ${self.state.rsp_month_start_price:.2f} (on {self.state.month_start_date}), "
                    f"Current ${rsp_price:.2f}, Daily change {daily_return*100:+.2f}%"
                )
                plans.extend(self._create_trading_plans(reason))
                self.state.first_exec = True
            elif self._is_third_week_end():
                plans.extend(self._create_trading_plans("Third week end"))
                self.state.first_exec = True

        # 2. Second investment on larger drop
        if not self.state.second_exec and self.state.rsp_month_start_price:
            mtd_return = (rsp_price / self.state.rsp_month_start_price) - 1
            if mtd_return <= -0.05:
                reason = (
                    f"RSP MTD drop <= -5%: Month start ${self.state.rsp_month_start_price:.2f} (on {self.state.month_start_date}), "
                    f"Current ${rsp_price:.2f}, MTD change {mtd_return*100:+.2f}%"
                )
                plans.extend(self._create_trading_plans(reason, multiplier=3))
                self.state.second_exec = True

        # Save state after checking
        if plans:
            self.state.save_state()

        return plans

    def _create_trading_plans(
        self, reason: str, multiplier: int = 1
    ) -> List[TradingPlan]:
        """Create trading plans for all tickers"""
        plans = []
        total_investment = self.cash_per_contribution * multiplier
        investment_per_ticker = total_investment / len(self.tickers)

        for ticker in self.tickers:
            ticker_data = self.price_cache.get(ticker, {})
            price = ticker_data.get("current", 100)

            if price and price > 0:
                shares = int(investment_per_ticker / price)
                if shares > 0:
                    # Add execution notes based on ticker characteristics
                    notes = self._generate_execution_notes(ticker, ticker_data)

                    plan = TradingPlan(
                        date=datetime.datetime.now(),
                        ticker=ticker,
                        action="BUY",
                        shares=shares,
                        reason=reason,
                        current_price=price,
                        target_value=shares * price,
                        execution_notes=notes,
                    )
                    plans.append(plan)

        return plans

    def _generate_execution_notes(self, ticker: str, ticker_data: dict) -> str:
        """Generate execution notes based on ticker and market conditions"""
        notes = []

        change_pct = ticker_data.get("change_percent", 0)

        # Add notes based on price movement
        if abs(change_pct) > 2:
            notes.append(f"High volatility today ({change_pct:+.1f}%)")
            notes.append("Consider using limit order")

        if change_pct < -1:
            notes.append("Already down today - may see further decline")
            notes.append("Consider splitting order throughout the day")

        # Ticker-specific notes
        ticker_notes = {
            "VUG": "Growth ETF - check tech sector sentiment",
            "VO": "Mid-cap ETF - verify no major economic news",
            "MOAT": "Wide moat ETF - less volatile, market order OK",
            "PFF": "Preferred stock ETF - check interest rate news",
            "VNQ": "REIT ETF - sensitive to interest rates",
        }

        if ticker in ticker_notes:
            notes.append(ticker_notes[ticker])

        return "\n".join(notes)

    def _is_third_week_end(self) -> bool:
        """Check if today is the end of third week"""
        today = datetime.date.today()
        day = today.day

        # Between 15-21 and it's Thursday or Friday
        if 15 <= day <= 21 and today.weekday() in [3, 4]:
            return True
        return False

    def run_planning_cycle(self, decision_day, market_data_day, preloaded_data=None):
        """Run one planning cycle for a specific date (trading_day)"""
        print("\n" + "=" * 60)
        print(f"Running Hi5 Planning Cycle - {decision_day}")
        print("=" * 60)
        plans = self.check_hi5_signals(market_data_day, preloaded_data=preloaded_data)
        if plans:
            print(f"\nFound {len(plans)} trading signals!")
            if self.email_manager and self.recipient_email:
                print("\nSending daily calendar invitation (with summary)...")
                # Build HTML summary (same as send_plan_summary_email)
                total_investment = sum(p.target_value for p in plans)
                html_summary = f"""
                <html>
                    <body>
                        <h2>Hi5 Trading Plan Summary</h2>
                        <p><strong>Date:</strong> {decision_day.strftime('%Y-%m-%d')}</p>
                        <p><strong>Total Investment:</strong> ${total_investment:,.2f}</p>
                        <hr>
                        <h3>Planned Trades:</h3>
                        <table border="1" cellpadding="5" cellspacing="0">
                            <tr>
                                <th>Ticker</th>
                                <th>Action</th>
                                <th>Shares</th>
                                <th>Price</th>
                                <th>Value</th>
                                <th>Reason</th>
                            </tr>
                """
                for plan in plans:
                    html_summary += f"""
                            <tr>
                                <td>{plan.ticker}</td>
                                <td>{plan.action}</td>
                                <td>{plan.shares}</td>
                                <td>${plan.current_price:.2f}</td>
                                <td>${plan.target_value:.2f}</td>
                                <td>{plan.reason}</td>
                            </tr>
                    """
                html_summary += """
                        </table>
                        <hr>
                        <p><strong>Next Steps:</strong></p>
                        <ol>
                            <li>Check your calendar for trade reminders</li>
                            <li>Execute trades during market hours</li>
                            <li>Update tracking spreadsheet after execution</li>
                        </ol>
                        <p style="color: red;"><strong>Remember:</strong> This is a plan, not a commitment. Adjust based on market conditions.</p>
                    </body>
                </html>
                """
                self.email_manager.send_daily_calendar_invitation(
                    self.recipient_email, plans, decision_day, html_summary
                )
            print("\nPlan complete!")
        else:
            print("\nNo trading signals at this time.")
            print(
                f"Current state: Month={self.state.current_month}, Month_Start={self.state.rsp_month_start_price} (on {self.state.month_start_date}), First={self.state.first_exec}, Second={self.state.second_exec}"
            )
            # Send heartbeat email if email_manager is enabled
            if self.email_manager and self.recipient_email:
                subject = f"Hi5 Heartbeat: No trades for {decision_day.strftime('%Y-%m-%d')}"
                body = f"""
Hi5 Heartbeat Notification

No trading signals were detected for {decision_day.strftime('%Y-%m-%d')}.

This is an automated message to confirm the Hi5 Offline Trading Planner is running and monitoring the market as expected.

Current state:
- Month: {self.state.current_month}
- Month Start: {self.state.rsp_month_start_price} (on {self.state.month_start_date})
- First Exec: {self.state.first_exec}
- Second Exec: {self.state.second_exec}

If you have any questions or need to adjust the strategy, please check the logs or contact the system administrator.
"""
                try:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = subject
                    msg["From"] = self.email_manager.smtp_username
                    msg["To"] = self.recipient_email
                    msg.attach(MIMEText(body, "plain"))
                    with smtplib.SMTP(self.email_manager.smtp_server, self.email_manager.smtp_port) as server:
                        server.starttls()
                        server.login(self.email_manager.smtp_username, self.email_manager.smtp_password)
                        server.send_message(msg)
                    print(f"Heartbeat email sent to {self.recipient_email}")
                except Exception as e:
                    print(f"Error sending heartbeat email: {e}")

    def run_range(self):
        """Run planning cycle for each date in test_dates (range mode)"""
        if not self.test_dates:
            print("No test date range provided.")
            return
        # Efficient batch download for all tickers and all dates
        start_date = self.test_dates[0]
        end_date = self.test_dates[-1]
        # Download from (start_date - 1 day) to end_date
        download_start = start_date - datetime.timedelta(days=1)
        print(
            f"Batch downloading all ticker data from {download_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )
        all_tickers = self.tickers + [self.benchmark_ticker]
        data = yf.download(
            all_tickers,
            start=download_start.strftime("%Y-%m-%d"),
            end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            group_by="ticker",
            progress=False,
            auto_adjust=False,
        )
        # For single ticker, data is not multi-indexed
        if len(all_tickers) == 1:
            data = {all_tickers[0]: data}
        self._batch_data = data
        # After downloading data...
        available_dates = data.index

        for d in self.test_dates:
            d_ts = pd.Timestamp(d)
            prev_trading_days = available_dates[available_dates < d_ts]
            if len(prev_trading_days) == 0:
                print(
                    f"Skipping {d.strftime('%Y-%m-%d')}: no previous trading day available."
                )
                continue
            prev_trading_day = prev_trading_days[-1]
            self.run_planning_cycle(
                decision_day=d, market_data_day=prev_trading_day, preloaded_data=data
            )

    def run_continuous(self, check_interval_minutes=60):
        """Run continuously, checking periodically"""
        print("Starting Hi5 Offline Trading Planner...")
        print(f"Will check for signals every {check_interval_minutes} minutes")
        print("\nPress Ctrl+C to stop\n")
        while True:
            try:
                now = datetime.datetime.now()
                et_tz = pytz.timezone("America/New_York")
                et_now = et_tz.localize(now.replace(tzinfo=None))
                if et_now.weekday() < 5 and 8 <= et_now.hour < 17:
                    self.run_planning_cycle(
                        trading_day=now, market_data_day=now, preloaded_data=None
                    )
                else:
                    print(f"{now}: Market closed, skipping check")
                time.sleep(check_interval_minutes * 60)
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in planning cycle: {e}")
                time.sleep(60)  # Wait a minute before retrying


def load_smtp_config(config_file="smtp_config.json"):
    """Load SMTP configuration from JSON file"""
    try:
        if not os.path.exists(config_file):
            return None

        with open(config_file, "r") as f:
            config = json.load(f)

        required_fields = ["server", "port", "username", "password"]
        if not all(field in config for field in required_fields):
            print(f"Error: SMTP config file missing required fields: {required_fields}")
            return None

        return config
    except Exception as e:
        print(f"Error loading SMTP config: {e}")
        return None


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Hi5 Offline Trading Planner")
    parser.add_argument("--email", help="Email address for notifications")
    parser.add_argument(
        "--smtp-config",
        help="Path to SMTP config file (default: smtp-config.json)",
        default="smtp-config.json",
    )
    parser.add_argument(
        "--test-date", help="Test date in YYYY-MM-DD format (e.g., 2024-01-15)"
    )
    parser.add_argument(
        "--engine",
        choices=["local", "memory", "gcp"],
        default="local",
        help="State storage engine: local (default), memory, or gcp",
    )
    parser.add_argument(
        "--gcp-config",
        help="Path to GCP config file (default: gcp_config.json)",
        default="gcp-config.json",
    )
    parser.add_argument(
        "--use-bigquery",
        action="store_true",
        help="Use BigQuery instead of yfinance for market data",
    )
    parser.add_argument(
        "--project-id",
        help="GCP project ID (required when using BigQuery)",
    )

    args = parser.parse_args()

    # Parse test date if provided, else use today
    if args.test_date:
        try:
            test_dates = parse_test_dates(args.test_date)
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        # Default: today
        test_dates = [datetime.datetime.now()]

    # Load SMTP config if email is enabled
    smtp_config = None
    if args.email:
        smtp_config = load_smtp_config(args.smtp_config)
        if not smtp_config:
            print(f"Error: Could not load SMTP configuration from {args.smtp_config}")
            print("Please create a smtp_config.json file with the following format:")
            print(
                """
{
    "server": "smtp.gmail.com",
    "port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password"
}
                """
            )
            return

    # check gcp_config file exists
    with open(args.gcp_config, "r") as f:
        gcp_config = json.load(f)
    if not gcp_config:
        print(f"Error: GCP config file {args.gcp_config} is empty.")
        return
    if (
        not gcp_config.get("bucket_name")
        or not gcp_config.get("blob_name")
    ):
        print(f"Error: GCP config file {args.gcp_config} is missing required fields.")
        return

    # Configuration
    config = {
        "recipient_email": args.email,
        "smtp_server": smtp_config["server"] if smtp_config else None,
        "smtp_port": smtp_config["port"] if smtp_config else None,
        "smtp_username": smtp_config["username"] if smtp_config else None,
        "smtp_password": smtp_config["password"] if smtp_config else None,
        "test_date": test_dates,
        "engine": args.engine,
        "bucket_name": gcp_config.get("bucket_name") if args.engine == "gcp" else None,
        "blob_name": gcp_config.get("blob_name") if args.engine == "gcp" else None,
        "credentials_path": gcp_config.get("credentials_path"),
        "use_bigquery": args.use_bigquery,
        "project_id": args.project_id,
    }

    # Create planner
    planner = OfflineTradingPlanner(**config)

    # Efficient batch download for all tickers and all dates (even if only one date)
    start_date = test_dates[0]
    end_date = test_dates[-1]
    download_start = start_date - datetime.timedelta(days=40)
    print(
        f"Batch downloading all ticker data from {download_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    all_tickers = planner.tickers + [planner.benchmark_ticker]
    
    if not args.use_bigquery:
        data = yf.download(
            all_tickers,
            start=download_start.strftime("%Y-%m-%d"),
            end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            group_by="ticker",
            progress=False,
            auto_adjust=False,
        )
        if len(all_tickers) == 1:
            data = {all_tickers[0]: data}
        available_dates = data.index
    else:
        data = None
        available_dates = None

    for d in test_dates:
        d_ts = pd.Timestamp(d)
        if not args.use_bigquery:
            prev_trading_days = available_dates[available_dates < d_ts]
            if len(prev_trading_days) == 0:
                print(
                    f"Skipping {d.strftime('%Y-%m-%d')}: no previous trading day available."
                )
                continue
            prev_trading_day = prev_trading_days[-1]
        else:
            prev_trading_day = d
        planner.run_planning_cycle(
            decision_day=d, market_data_day=prev_trading_day, preloaded_data=data
        )


if __name__ == "__main__":
    main()
