import os
import datetime
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import icalendar
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
from googleapiclient.discovery import build
import argparse


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

    def create_daily_calendar_invitation(
        self, plans: List[TradingPlan], trading_day: datetime.datetime
    ) -> bytes:
        """Create an ICS file for all trading plans for a day"""
        cal = icalendar.Calendar()
        cal.add("prodid", "-//Hi5 Trading Plan//EN")
        cal.add("version", "2.0")
        cal.add("method", "REQUEST")

        event = icalendar.Event()
        event.add("summary", f'Hi5 Trading Plan for {trading_day.strftime("%Y-%m-%d")}')

        et_tz = pytz.timezone("America/New_York")
        start_time = et_tz.localize(
            datetime.datetime.combine(trading_day.date(), datetime.time(8, 30))
        )
        end_time = start_time + datetime.timedelta(minutes=30)

        event.add("dtstart", start_time)
        event.add("dtend", end_time)
        event.add("dtstamp", datetime.datetime.now(pytz.UTC))
        event.add("created", datetime.datetime.now(pytz.UTC))
        event.add("last-modified", datetime.datetime.now(pytz.UTC))
        event.add("organizer", f"MAILTO:{self.smtp_username}")
        event.add("attendee", f"MAILTO:{self.smtp_username}")

        alarm = icalendar.Alarm()
        alarm.add("action", "DISPLAY")
        alarm.add(
            "description",
            f'Reminder: Hi5 Trading Plan for {trading_day.strftime("%Y-%m-%d")}',
        )
        alarm.add("trigger", datetime.timedelta(minutes=-30))
        event.add_component(alarm)

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
        event.add("description", description)
        cal.add_component(event)
        return cal.to_ical()

    def send_daily_calendar_invitation(
        self,
        recipient_email: str,
        plans: List[TradingPlan],
        trading_day: datetime.datetime,
        html_summary: str,
    ):
        """Send a single calendar invitation for all trades on a day."""
        try:
            msg = MIMEMultipart("mixed")
            msg["Subject"] = f'Hi5 Trading Plan: {trading_day.strftime("%Y-%m-%d")}'
            msg["From"] = self.smtp_username
            msg["To"] = recipient_email

            msg.attach(MIMEText(html_summary, "html"))

            ics_data = self.create_daily_calendar_invitation(plans, trading_day)
            part = MIMEBase("text", "calendar", method="REQUEST")
            part.set_payload(ics_data)
            encoders.encode_base64(part)
            part.add_header(
                "Content-Type", "text/calendar; method=REQUEST; charset=UTF-8"
            )
            part.add_header("Content-Disposition", "attachment; filename=invite.ics")
            msg.attach(part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            print(f"Daily calendar invitation sent to {recipient_email}")
        except Exception as e:
            print(f"Error sending daily calendar invitation: {e}")


class GoogleDocsManager:
    """Manages writing data to a Google Doc."""

    def __init__(self, credentials_path: str, document_id: str):
        self.document_id = document_id
        try:
            scopes = ["https://www.googleapis.com/auth/documents"]
            creds = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
            self.service = build("docs", "v1", credentials=creds)
        except Exception as e:
            print(f"Error initializing Google Docs client: {e}")
            raise

    def clear_document_body(self):
        """Deletes all content from the document body."""
        try:
            # Get the document to find the end index of the body content
            document = (
                self.service.documents().get(documentId=self.document_id).execute()
            )
            content = document.get("body").get("content")

            # Find the total length of the content in the body.
            # We need to leave the first character to avoid deleting the body segment.
            end_index = 1
            if len(content) > 1:
                for element in content:
                    if "endIndex" in element:
                        end_index = max(end_index, element["endIndex"])

            if end_index > 2:  # Only delete if there is content
                requests = [
                    {
                        "deleteContentRange": {
                            "range": {
                                "startIndex": 1,
                                "endIndex": end_index
                                - 1,  # Delete up to the last character
                            }
                        }
                    }
                ]
                self.service.documents().batchUpdate(
                    documentId=self.document_id, body={"requests": requests}
                ).execute()
        except Exception as e:
            print(f"Error clearing Google Doc: {e}")

    def update_doc(
        self,
        portfolio_summary: dict,
        orders_df: pd.DataFrame,
        dividends_df: pd.DataFrame,
        irr: float,
    ):
        """Updates the Google Doc with the latest portfolio status, orders, and dividends."""
        self.clear_document_body()

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build the text content
        content = f"Hi5 Portfolio Status - Last Updated: {now_str}\n\n"
        content += "--- Portfolio Summary ---\n"
        for key, value in portfolio_summary.items():
            try:  # Try to format as currency
                content += f"{key}: ${float(value):,.2f}\n"
            except (ValueError, TypeError):
                content += f"{key}: {value}\n"
        content += f"Estimated IRR: {irr:.2%}\n\n"

        content += "--- Recent Orders ---\n"
        content += (
            orders_df.to_string(index=False)
            if not orders_df.empty
            else "No recent orders.\n"
        )
        content += "\n\n"

        content += "--- Recent Dividends ---\n"
        content += (
            dividends_df.to_string(index=False)
            if not dividends_df.empty
            else "No recent dividends.\n"
        )

        requests = [
            {
                "insertText": {
                    "location": {
                        "index": 1,
                    },
                    "text": content,
                }
            }
        ]

        try:
            self.service.documents().batchUpdate(
                documentId=self.document_id, body={"requests": requests}
            ).execute()
            print(
                f"Successfully updated Google Doc: https://docs.google.com/document/d/{self.document_id}/edit"
            )
        except Exception as e:
            print(f"Error updating Google Doc: {e}")


class OfflineTradingPlanner:
    """Trading planner that works without IBKR connection"""

    def __init__(
        self,
        recipient_email=None,
        smtp_server=None,
        smtp_port=None,
        smtp_username=None,
        smtp_password=None,
        test_dates=None,
        engine="local",
        bucket_name=None,
        blob_name=None,
        credentials_path=None,
        project_id=None,
        gdocs_document_id=None,
    ):
        self.recipient_email = recipient_email
        self.test_dates = (
            test_dates
            if test_dates
            else [datetime.datetime.now() - datetime.timedelta(days=1)]
        )
        self.engine = engine
        self.project_id = project_id

        self.gdocs_manager = None
        if gdocs_document_id and credentials_path:
            print(
                f"Google Docs integration enabled for document ID: {gdocs_document_id}"
            )
            self.gdocs_manager = GoogleDocsManager(
                credentials_path=credentials_path, document_id=gdocs_document_id
            )

        # Initialize BigQuery client
        if not credentials_path or not project_id:
            raise ValueError("credentials_path and project_id are required")
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.bq_client = bigquery.Client(credentials=credentials, project=project_id)

        # Email services
        if recipient_email:
            if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
                print(
                    "Error: SMTP settings are required when email notifications are enabled"
                )
                print("Please provide SMTP configuration in smtp-config.json")
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
        # Trading parameters
        self.tickers = ["VUG", "VO", "MOAT", "PFFD", "IDWP.L"]
        self.benchmark_ticker = "RSP"
        self.cash_per_contribution = 10000
        self.price_cache = {}

    def get_investment_amount(self, portfolio_value: float) -> float:
        """Determines the investment amount based on portfolio value."""
        if portfolio_value > 500000:
            print("Using percentage-based investment strategy (1% of portfolio value)")
            return portfolio_value * 0.01
        else:
            print(
                f"Using incremental investment strategy (base: ${self.cash_per_contribution:,.2f})"
            )
            return self.cash_per_contribution

    def update_market_data(self, trading_day):
        """Fetch market data from BigQuery for the given trading day."""
        all_tickers = self.tickers + [self.benchmark_ticker]
        self.price_cache = {}

        current_query = f"""
        WITH prev_day AS (
            SELECT 
                ticker,
                date,
                adj_close,
                LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) as prev_adj_close
            FROM `{self.project_id}.market_data.marketing`
            WHERE date <= DATE('{trading_day.strftime('%Y-%m-%d')}')
        )
        SELECT 
            ticker,
            date,
            adj_close as close,
            prev_adj_close as prev_close,
            (adj_close - prev_adj_close) as change,
            ((adj_close - prev_adj_close) / prev_adj_close * 100) as change_percent
        FROM prev_day
        WHERE date = DATE('{trading_day.strftime('%Y-%m-%d')}')
        AND ticker IN UNNEST(@tickers)
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("tickers", "STRING", all_tickers)
            ]
        )
        
        query_job = self.bq_client.query(current_query, job_config=job_config)
        df = query_job.result().to_dataframe()

        if df.empty:
            print("BigQuery query (for debugging):\n" + current_query)
            raise RuntimeError(f"No data found in BigQuery for {trading_day.strftime('%Y-%m-%d')}")
        
        for _, row in df.iterrows():
            ticker = row["ticker"]
            self.price_cache[ticker] = {
                "current": row["close"],
                "previous_close": row["prev_close"],
                "change": row["change"],
                "change_percent": row["change_percent"],
            }

        missing_tickers = [t for t in all_tickers if t not in self.price_cache]
        if missing_tickers:
            print("BigQuery query (for debugging):\n" + current_query)
            raise RuntimeError(f"Missing data for tickers: {missing_tickers} on {trading_day.strftime('%Y-%m-%d')}")

    def check_hi5_signals(self, trading_day) -> Tuple[List[TradingPlan], dict]:
        """Check Hi5 strategy for trading signals."""
        plans = []
        self.update_market_data(trading_day)

        # Always fail if price data is missing or invalid
        rsp_data = self.price_cache.get(self.benchmark_ticker)
        if not rsp_data or rsp_data.get("current") is None:
            raise RuntimeError(f"Missing or invalid RSP price data for {trading_day.strftime('%Y-%m-%d')}: {rsp_data}")
        rsp_price = rsp_data["current"]
        rsp_prev_close = rsp_data["previous_close"]
        if rsp_price is None or rsp_prev_close is None:
            raise RuntimeError(f"Missing RSP price or previous close for {trading_day.strftime('%Y-%m-%d')}: {rsp_data}")

        portfolio_value = self.get_investment_amount(0)
        print(f"Current portfolio value for target tickers: ${portfolio_value:,.2f}")

        base_investment_amount = self.get_investment_amount(portfolio_value)

        trigger_context = {
            "rsp_price": rsp_price,
            "rsp_daily_return": (
                (rsp_price / rsp_prev_close - 1) if rsp_prev_close else 0
            ),
            "rsp_mtd_return": None,
            "ma50_ratio": None,
            "first_exec_triggered": False,
            "second_exec_triggered": False,
            "third_exec_triggered": False,
        }

        if (
            self.state.current_month is None
            or self.state.current_month != trading_day.month
        ):
            self.state.current_month = trading_day.month
            self.state.first_exec = False
            self.state.second_exec = False
            self.state.third_exec = False
            self.state.save_state()
            print(f"Month refreshed: {trading_day.month}")

        if not self.state.first_exec:
            daily_return = trigger_context["rsp_daily_return"]
            if daily_return <= -0.01:
                reason = f"RSP daily drop <= -1%: Current ${rsp_price:.2f}, Daily change {daily_return*100:+.2f}%"
                multiplier = 2 if portfolio_value < 500000 else 1
                investment_amount = base_investment_amount * multiplier
                plans.extend(self._create_trading_plans(reason, investment_amount))
                self.state.first_exec = True
                trigger_context["first_exec_triggered"] = True
            elif self._is_third_week_end():
                reason = "Third week end"
                multiplier = 2 if portfolio_value < 500000 else 1
                investment_amount = base_investment_amount * multiplier
                plans.extend(self._create_trading_plans(reason, investment_amount))
                self.state.first_exec = True
                trigger_context["first_exec_triggered"] = True

        if not self.state.second_exec:
            query = f"""
            SELECT MIN(date) as first_date
            FROM `{self.project_id}.market_data.marketing`
            WHERE date >= DATE_TRUNC(DATE('{trading_day.strftime('%Y-%m-%d')}'), MONTH)
            AND ticker = '{self.benchmark_ticker}'
            """
            result = self.bq_client.query(query).result()
            first_date_row = next(result, None)
            if first_date_row and first_date_row.first_date:
                first_date = first_date_row.first_date
                price_query = f"""
                SELECT adj_close
                FROM `{self.project_id}.market_data.marketing`
                WHERE date = DATE('{first_date}')
                AND ticker = '{self.benchmark_ticker}'
                """
                price_result = self.bq_client.query(price_query).result()
                price_row = next(price_result, None)
                if price_row:
                    month_start_price = price_row.adj_close
                    mtd_return = (rsp_price / month_start_price) - 1
                    trigger_context["rsp_mtd_return"] = mtd_return
                    if mtd_return <= -0.05:
                        multiplier = 3 if portfolio_value < 500000 else 1
                        investment_amount = base_investment_amount * multiplier
                        reason = f"RSP MTD drop <= -5% ({mtd_return:.2%})."
                        print(
                            f"TRIGGER: {reason}. Investing ${investment_amount:,.2f}."
                        )
                        plans.extend(
                            self._create_trading_plans(reason, investment_amount)
                        )
                        self.state.second_exec = True
                        trigger_context["second_exec_triggered"] = True

        if not self.state.third_exec:
            extreme_query = f"""
            SELECT ma50_ratio
            FROM `{self.project_id}.market_data.market_breadth`
            WHERE date = DATE('{trading_day.strftime('%Y-%m-%d')}')
            """
            extreme_result = self.bq_client.query(extreme_query).result()
            for row in extreme_result:
                trigger_context["ma50_ratio"] = row.ma50_ratio
                if row.ma50_ratio <= 0.15:
                    reason = f"Human extreme condition triggered: MA50 ratio = {row.ma50_ratio:.2%}"
                    multiplier = 5 if portfolio_value < 500000 else 1
                    investment_amount = base_investment_amount * multiplier
                    plans.extend(self._create_trading_plans(reason, investment_amount))
                    self.state.third_exec = True
                    trigger_context["third_exec_triggered"] = True

        if plans:
            self.state.save_state()
            print(
                f"Current State: First={self.state.first_exec}, Second={self.state.second_exec}, Third={self.state.third_exec}"
            )

        return plans, trigger_context

    def _create_trading_plans(
        self, reason: str, investment_amount: float
    ) -> List[TradingPlan]:
        """Create trading plans for all tickers."""
        plans = []
        investment_per_ticker = investment_amount / len(self.tickers)

        for ticker in self.tickers:
            ticker_data = self.price_cache.get(ticker, {})
            price = ticker_data.get("current", 100)

            if price and price > 0:
                shares = int(investment_per_ticker / price)
                if shares > 0:
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
        """Generate execution notes based on ticker and market conditions."""
        notes = []
        change_pct = ticker_data.get("change_percent", 0)

        if abs(change_pct) > 2:
            notes.append(f"High volatility today ({change_pct:+.1f}%)")
            notes.append("Consider using limit order")
        if change_pct < -1:
            notes.append("Already down today - may see further decline")
            notes.append("Consider splitting order throughout the day")

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
        """Check if today is the end of third week of the month."""
        today = datetime.date.today()
        return 15 <= today.day <= 21 and today.weekday() in [3, 4]

    def run_planning_cycle(self, decision_day, market_data_day):
        """Run one planning cycle for a specific date."""
        print("\n" + "=" * 60)
        print(
            f"Running Hi5 Planning Cycle for {decision_day.strftime('%Y-%m-%d')} using data from {market_data_day.strftime('%Y-%m-%d')}"
        )
        print("=" * 60)

        if self.gdocs_manager:
            print("\nUpdating Google Docs with latest portfolio information...")
            self.gdocs_manager.update_doc(
                portfolio_summary={},
                orders_df=pd.DataFrame(),
                dividends_df=pd.DataFrame(),
                irr=0.0,
            )

        plans, trigger_context = self.check_hi5_signals(market_data_day)

        if plans:
            print(f"\nFound {len(plans)} trading signals!")
            total_investment = sum(p.target_value for p in plans)
            html_summary = f"""
            <html><body>
                <h2>Hi5 Trading Plan Summary</h2>
                <p><strong>Date:</strong> {decision_day.strftime('%Y-%m-%d')}</p>
                <p><strong>Total Investment:</strong> ${total_investment:,.2f}</p>
                <hr><h3>Planned Trades:</h3>
                <table border="1" cellpadding="5" cellspacing="0">
                    <tr><th>Ticker</th><th>Action</th><th>Shares</th><th>Price</th><th>Value</th><th>Reason</th></tr>
            """
            for p in plans:
                html_summary += f"<tr><td>{p.ticker}</td><td>{p.action}</td><td>{p.shares}</td><td>${p.current_price:.2f}</td><td>${p.target_value:.2f}</td><td>{p.reason}</td></tr>"
            html_summary += """
                </table><hr><p><strong>Next Steps:</strong></p><ol>
                    <li>Check your calendar for trade reminders</li><li>Execute trades during market hours</li>
                    <li>Update tracking spreadsheet after execution</li></ol>
                <p style="color: red;"><strong>Remember:</strong> This is a plan, not a commitment. Adjust based on market conditions.</p>
            </body></html>
            """

            # Print email content to console
            print("\n=== Email Content ===")
            print(html_summary)
            print("===================\n")

            if self.email_manager and self.recipient_email:
                self.email_manager.send_daily_calendar_invitation(
                    self.recipient_email, plans, decision_day, html_summary
                )
            print("\nPlan complete!")
        else:
            print("\nNo trading signals at this time.")
            print(
                f"Current state: Month={self.state.current_month}, First={self.state.first_exec}, Second={self.state.second_exec}, Third={self.state.third_exec}"
            )

            # Create and print heartbeat message
            subject = (
                f"Hi5 Heartbeat: No trades for {decision_day.strftime('%Y-%m-%d')}"
            )
            body = f"""
Hi5 Heartbeat Notification

No trading signals were detected for {decision_day.strftime('%Y-%m-%d')}.
This is an automated message to confirm the planner is running.

---
Trigger Status:
- RSP Price: ${trigger_context.get('rsp_price', 0):.2f}
- RSP Daily Return: {trigger_context.get('rsp_daily_return', 0)*100:+.2f}% (Triggered: {trigger_context.get('first_exec_triggered', False)})
- RSP MTD Return: {(trigger_context.get('rsp_mtd_return') or 0)*100:.2f}% (Triggered: {trigger_context.get('second_exec_triggered', False)})
- MA50 Ratio: {(trigger_context.get('ma50_ratio') or 0):.2%} (Triggered: {trigger_context.get('third_exec_triggered', False)})
---

Current state:
- Month: {self.state.current_month}
- First Exec: {self.state.first_exec}
- Second Exec: {self.state.second_exec}
- Third Exec: {self.state.third_exec}
"""
            # Print heartbeat message to console
            print("\n=== Heartbeat Message ===")
            print(f"Subject: {subject}")
            print(body)
            print("=======================\n")

            if self.email_manager and self.recipient_email:
                try:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = subject
                    msg["From"] = self.email_manager.smtp_username
                    msg["To"] = self.recipient_email
                    msg.attach(MIMEText(body, "plain"))
                    with smtplib.SMTP(
                        self.email_manager.smtp_server, self.email_manager.smtp_port
                    ) as server:
                        server.starttls()
                        server.login(
                            self.email_manager.smtp_username,
                            self.email_manager.smtp_password,
                        )
                        server.send_message(msg)
                    print(f"Heartbeat email sent to {self.recipient_email}")
                except Exception as e:
                    print(f"Error sending heartbeat email: {e}")

    def run(self):
        """Run planning cycle for each date in test_dates."""
        start_date = self.test_dates[0]
        end_date = self.test_dates[-1]

        query = f"""
        SELECT DISTINCT date FROM `{self.project_id}.market_data.marketing`
        WHERE date >= DATE('{start_date.strftime('%Y-%m-%d')}') AND date <= DATE('{end_date.strftime('%Y-%m-%d')}')
        ORDER BY date
        """
        all_trading_days = [row.date for row in self.bq_client.query(query).result()]

        if not all_trading_days:
            print("No trading days found in BigQuery for the specified date range.")
            return

        for decision_day in self.test_dates:
            # Find the latest available market data day for the given decision day
            market_data_day = next(
                (d for d in reversed(all_trading_days) if d <= decision_day.date()),
                None,
            )

            if market_data_day:
                self.run_planning_cycle(
                    decision_day=decision_day, market_data_day=market_data_day
                )
            else:
                print(
                    f"Skipping {decision_day.strftime('%Y-%m-%d')}: no previous or current trading day data available."
                )


def load_smtp_config(config_file="smtp-config.json"):
    """Load SMTP configuration from JSON file."""
    if not os.path.exists(config_file):
        print(f"Warning: SMTP config file {config_file} not found.")
        return None
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        required = ["server", "port", "username", "password"]
        if not all(field in config for field in required):
            print(f"Error: SMTP config file missing one of: {required}")
            return None
        return config
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading SMTP config: {e}")
        return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hi5 Offline Trading Planner")
    parser.add_argument("--email", help="Email address for notifications")
    parser.add_argument(
        "--smtp-config", default="smtp-config.json", help="Path to SMTP config file"
    )
    parser.add_argument(
        "--test-date",
        help="Test date (YYYY-MM-DD) or range (YYYY-MM-DD:YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--engine",
        choices=["local", "memory", "gcp"],
        default="local",
        help="State storage engine",
    )
    parser.add_argument(
        "--gcp-config", default="gcp-config.json", help="Path to GCP config file"
    )
    parser.add_argument("--gdocs-document-id", help="ID of the Google Doc to update.")
    args = parser.parse_args()

    test_dates = []
    if args.test_date:
        try:
            if ":" in args.test_date:
                start_str, end_str = args.test_date.split(":", 1)
                start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d")
                if end_date < start_date:
                    raise ValueError("End date must not be before start date")
                test_dates = [
                    start_date + datetime.timedelta(days=i)
                    for i in range((end_date - start_date).days + 1)
                ]
            else:
                test_dates = [datetime.datetime.strptime(args.test_date, "%Y-%m-%d")]
        except ValueError as e:
            print(f"Error parsing --test-date: {e}")
            return

    smtp_config = load_smtp_config(args.smtp_config) if args.email else {}

    try:
        with open(args.gcp_config, "r") as f:
            gcp_config = json.load(f)
        if not gcp_config.get("project_id") or not gcp_config.get("credentials_path"):
            raise ValueError(
                "GCP config must contain 'project_id' and 'credentials_path'"
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error with GCP config file {args.gcp_config}: {e}")
        return

    config = {
        "recipient_email": args.email,
        "smtp_server": smtp_config.get("server"),
        "smtp_port": smtp_config.get("port"),
        "smtp_username": smtp_config.get("username"),
        "smtp_password": smtp_config.get("password"),
        "test_dates": test_dates,
        "engine": args.engine,
        "bucket_name": gcp_config.get("bucket_name"),
        "blob_name": gcp_config.get("blob_name"),
        "credentials_path": gcp_config.get("credentials_path"),
        "project_id": gcp_config.get("project_id"),
        "gdocs_document_id": args.gdocs_document_id,
    }

    planner = OfflineTradingPlanner(**config)
    planner.run()


if __name__ == "__main__":
    main()
