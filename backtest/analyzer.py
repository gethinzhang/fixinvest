import backtrader as bt
import pandas as pd
import numpy as np
import numpy_financial as npf
from datetime import date
import csv


class BacktestAnalyzer(bt.Analyzer):
    """
    A backtrader analyzer to provide detailed performance metrics,
    supporting both fixed-investment and variable-investment strategies.
    Params:
        - risk_free_rate (float): The annual risk-free rate for Sharpe ratio calculation.
        - fix_investment (bool): If True, enables logic for fixed periodic investments,
          including a monthly IRR calculation. If False, uses standard portfolio analysis.
    """

    params = (
        ("risk_free_rate", 0.04),
        ("fix_investment", False),
        ("non_resident_tax_rate", 0.3),
        ("print_drawdown_history", False),
    )

    def __init__(self):
        super().__init__()
        self.strategy = self.strategy
        self.data_feeds = {data._name: data for data in self.datas}

        # Data stores
        self.portfolio_values = []
        self.cash_values = []
        self.stock_values = []
        self.dates = []
        self.trading_log = []
        self.dividend_log = []
        self.investment_log = []
        self.final_positions = []
        self.final_stock_value = 0.0

        # Metrics tracking
        self.total_investor_deposits = 0.0
        self.total_dividend_cash = 0.0
        self.total_gross_dividends = 0.0
        self.total_dividend_tax = 0.0

        # For monthly IRR in fix_investment mode
        self.monthly_portfolio_values = []
        self.monthly_cashflows = []
        self.current_month_cashflow = 0.0
        self.last_month = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        order_date = (
            bt.num2date(order.executed.dt)
            if order.executed
            else self.strategy.datetime.date(0)
        )

        if order.status == order.Completed:
            order_info = {
                "date": order_date.date(),
                "ticker": order.data._name,
                "type": "BUY" if order.isbuy() else "SELL",
                "price": order.executed.price,
                "size": order.executed.size * (1 if order.isbuy() else -1),
                "value": order.executed.value,
                "commission": order.executed.comm,
                "reason": getattr(order, "reason", "N/A"),
            }
            self.trading_log.append(order_info)

    def record_investment(self, date, amount, reason):
        """Records an investment/deposit transaction."""
        investment_record = {"date": date, "amount": amount, "reason": reason}
        self.investment_log.append(investment_record)
        self.total_investor_deposits += amount
        if self.p.fix_investment:
            self.current_month_cashflow += amount

    def add_dividend_event(
        self, date, ticker, gross_amount, shares_held
    ):
        """Records a dividend event and its impact on cash."""
        tax_rate = self.p.non_resident_tax_rate
        calculated_tax = gross_amount * tax_rate
        net_dividend = gross_amount - calculated_tax

        dividend_info = {
            "date": date,
            "ticker": ticker,
            "type": "DIVIDEND",
            "price": gross_amount / shares_held if shares_held else 0,
            "size": shares_held,
            "value": gross_amount,
            "net_value": net_dividend,
            "commission": calculated_tax,
            "reason": f"Net: ${net_dividend:.2f} (Tax: {tax_rate*100:.0f}%)",
        }
        self.dividend_log.append(dividend_info)
        self.total_dividend_cash += net_dividend
        self.total_gross_dividends += gross_amount
        self.total_dividend_tax += calculated_tax

        # For IRR purposes, dividends are treated as a cash withdrawal.
        self.record_investment(
            date,
            -net_dividend,
            f"Dividend from {ticker} Gross: ${gross_amount:.2f}, Tax: ${calculated_tax:.2f}",
        )

    def next(self):
        current_date = self.strategy.datetime.date(0)
        self.dates.append(current_date)

        cash_value = self.strategy.broker.get_cash()
        stock_value = 0.0
        for data in self.strategy.datas:
            position = self.strategy.getposition(data)
            if position.size != 0:
                price = data.close[0]
                if price and not np.isnan(price):
                    stock_value += position.size * price
        
        portfolio_value = cash_value + stock_value
        self.portfolio_values.append(portfolio_value)
        self.cash_values.append(cash_value)
        self.stock_values.append(stock_value)

        if self.p.fix_investment:
            current_month = current_date.month
            if self.last_month is None:
                self.last_month = current_month

            if current_month != self.last_month:
                self.monthly_cashflows.append(self.current_month_cashflow)
                self.current_month_cashflow = 0.0
                self.last_month = current_month
    
    def stop(self):
        if self.p.fix_investment:
            # Append the last month's cashflow
            self.monthly_cashflows.append(self.current_month_cashflow)

        self.capture_final_positions(self.strategy)

    def capture_final_positions(self, strat):
        self.final_stock_value = 0.0
        positions = []

        # Create a summary of total dividends received per ticker
        ticker_dividends = {}
        for div_event in self.dividend_log:
            ticker = div_event["ticker"]
            amount = div_event["net_value"]
            ticker_dividends[ticker] = ticker_dividends.get(ticker, 0) + amount

        # Iterate through all data feeds, which is more reliable than strat.positions
        for data in strat.datas:
            pos = strat.getposition(data)
            if pos.size == 0:
                continue

            ticker = data._name
            # Skip FX feeds if they are present
            if ticker.endswith("=X"):
                continue

            price = data.close[0]
            value = 0.0
            unrealized_pnl = 0.0
            
            dividends_received = ticker_dividends.get(ticker, 0.0)
            total_pnl = 0.0

            if price and not np.isnan(price):
                value = pos.size * price
                unrealized_pnl = (price - pos.price) * pos.size
            
            total_pnl = unrealized_pnl + dividends_received
            self.final_stock_value += value

            positions.append(
                {
                    "Ticker": ticker,
                    "Shares": pos.size,
                    "Avg Cost Price": pos.price,
                    "Market Price": price,
                    "Market Value": value,
                    "Unrealized PnL": unrealized_pnl,
                    "Dividends Received": dividends_received,
                    "Total PnL": total_pnl,
                }
            )
        self.final_positions = positions

    def get_analysis(self):
        if not self.dates or not self.portfolio_values:
            return {}

        (
            max_dd,
            peak_date,
            peak_price,
            trough_date,
            trough_price,
            drawdown_history_df,
        ) = self._calculate_max_drawdown()
        monthly_irr, annual_irr = self._calculate_ir()

        final_value = self.portfolio_values[-1]
        pnl = final_value - self.total_investor_deposits

        # Non-fixed investment ROI
        roi_annualized = 0.0
        if not self.p.fix_investment and self.total_investor_deposits > 0:
            total_return = pnl / self.total_investor_deposits
            duration_years = (self.dates[-1] - self.dates[0]).days / 365.25
            if duration_years > 0:
                roi_annualized = (1 + total_return) ** (1 / duration_years) - 1

        return {
            "summary_metrics": {
                "Start Date": self.dates[0],
                "End Date": self.dates[-1],
                "Initial Portfolio Value": self.portfolio_values[0],
                "Final Portfolio Value": final_value,
                "Total Net Deposits": self.total_investor_deposits,
                "Total Gross Dividends": self.total_gross_dividends,
                "Total Dividend Tax": self.total_dividend_tax,
                "Net P&L": pnl,
                "Sharpe Ratio": self._calculate_sharpe_ratio(),
                "Max Drawdown": {
                    "Max DD": max_dd,
                    "Peak Date": peak_date,
                    "Peak Price": peak_price,
                    "Trough Date": trough_date,
                    "Trough Price": trough_price,
                },
                "Monthly IRR": monthly_irr,
                "Annual IRR": annual_irr,
                "Annualized ROI": roi_annualized,
            },
            "final_positions": self.final_positions,
            "trade_history": pd.DataFrame(self.trading_log),
            "dividend_history": pd.DataFrame(self.dividend_log),
            "investment_history": pd.DataFrame(self.investment_log),
            "drawdown_history": drawdown_history_df,
        }
    
    def _calculate_sharpe_ratio(self):
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return 0.0

        returns_df = pd.DataFrame(
            {"value": self.portfolio_values}, index=pd.to_datetime(self.dates)
        )
        monthly_returns = (
            returns_df["value"].resample("ME").last().pct_change(fill_method=None).dropna()
        )

        if monthly_returns.empty or monthly_returns.std() == 0:
            return 0.0

        monthly_risk_free = (1 + self.p.risk_free_rate) ** (1 / 12) - 1
        excess_returns = monthly_returns - monthly_risk_free
        std_dev = excess_returns.std()

        if std_dev > 0:
            return np.sqrt(12) * excess_returns.mean() / std_dev
        return 0.0

    def _calculate_max_drawdown(self):
        if not self.portfolio_values:
            return 0.0, None, None, None, None, None

        portfolio_values_np = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values_np)

        non_zero_peak_indices = peak != 0
        drawdown = np.zeros_like(portfolio_values_np, dtype=float)

        if np.any(non_zero_peak_indices):
            drawdown[non_zero_peak_indices] = (
                peak[non_zero_peak_indices]
                - portfolio_values_np[non_zero_peak_indices]
            ) / peak[non_zero_peak_indices]

            max_dd = np.max(drawdown)
            if max_dd > 0:
                trough_idx = np.argmax(drawdown)
                peak_value = peak[trough_idx]
                peak_idx = np.where(portfolio_values_np[: trough_idx + 1] == peak_value)[
                    0
                ][0]

                peak_date = self.dates[peak_idx]
                trough_date = self.dates[trough_idx]
                trough_price = portfolio_values_np[trough_idx]

                drawdown_dates = self.dates[peak_idx : trough_idx + 1]
                drawdown_values = self.portfolio_values[peak_idx : trough_idx + 1]
                drawdown_df = pd.DataFrame(
                    {"Date": drawdown_dates, "Portfolio Value": drawdown_values}
                )

                return (
                    max_dd,
                    peak_date,
                    peak_value,
                    trough_date,
                    trough_price,
                    drawdown_df,
                )

        return 0.0, None, None, None, None, None

    def _calculate_ir(self):
        if self.p.fix_investment:
            if not self.monthly_cashflows or len(self.monthly_cashflows) < 2:
                return 0.0, 0.0

            cash_flows = [-cf for cf in self.monthly_cashflows]
            cash_flows[-1] += self.portfolio_values[-1]

            try:
                # np.irr requires both positive and negative cash flows
                if any(cf > 0 for cf in cash_flows) and any(
                    cf < 0 for cf in cash_flows
                ):
                    monthly_irr = npf.irr(cash_flows)
                    if np.isnan(monthly_irr) or np.isinf(monthly_irr):
                        return 0.0, 0.0
                    annual_irr = (1 + monthly_irr) ** 12 - 1
                    return monthly_irr, annual_irr
            except Exception:
                pass
            return 0.0, 0.0
        else:
            # For non-fixed investments, calculate an annualized Return on Investment.
            if self.total_investor_deposits <= 0 or not self.dates:
                return 0.0, 0.0

            final_value = self.portfolio_values[-1]
            total_return = (
                final_value - self.total_investor_deposits
            ) / self.total_investor_deposits

            days = (self.dates[-1] - self.dates[0]).days
            if days <= 0:
                return 0.0, 0.0

            annual_roi = (1 + total_return) ** (365.25 / days) - 1
            monthly_roi = (1 + annual_roi) ** (1 / 12) - 1  # Just for reference
            return monthly_roi, annual_roi

    def print_summary(self, print_drawdown_history=None):
        """Prints a summary of the backtest results to the console."""
        analysis = self.get_analysis()
        if not analysis:
            print("No analysis data to generate summary.")
            return

        print("\n" + "=" * 50)
        print("Backtest Results Summary")
        print("=" * 50)

        metrics = analysis["summary_metrics"]
        for key, value in metrics.items():
            if key == "Max Drawdown":
                dd_info = value
                if dd_info and dd_info.get("Peak Date"):
                    print(
                        f"{'Max Drawdown':<25}: {dd_info['Max DD']:.2%} "
                        f"(Peak: {dd_info['Peak Date'].strftime('%Y-%m-%d')} ${dd_info['Peak Price']:,.2f}, "
                        f"Trough: {dd_info['Trough Date'].strftime('%Y-%m-%d')} ${dd_info['Trough Price']:,.2f})"
                    )
                elif dd_info:
                    print(f"{'Max Drawdown':<25}: {dd_info.get('Max DD', 0.0):.2%}")
            elif isinstance(value, float):
                if "IRR" in key or "ROI" in key:
                     print(f"{key:25}: {value:.2%}")
                else:
                    print(f"{key:25}: {value:,.2f}")
            elif isinstance(value, date):
                print(f"{key:25}: {value.strftime('%Y-%m-%d')}")
            else:
                print(f"{key:25}: {value}")

        print("\n" + "-" * 50)
        print("Final Positions")
        print("-" * 50)
        positions = analysis.get("final_positions", [])
        if not positions:
            print("No open positions at the end of the backtest.")
        else:
            for pos in positions:
                print(f"  - Ticker: {pos['Ticker']}")
                print(f"    Shares: {pos['Shares']}")
                print(f"    Market Value: {pos.get('Market Value', 0.0):,.2f}")
                print(f"    Unrealized PnL:   {pos.get('Unrealized PnL', 0.0):,.2f}")
                print(f"    Dividends Rcvd: {pos.get('Dividends Received', 0.0):,.2f}")
                print(f"    Total PnL:        {pos.get('Total PnL', 0.0):,.2f}")

        if self.p.print_drawdown_history or (
            print_drawdown_history is not None and print_drawdown_history
        ):
            print("\n" + "=" * 50)
            print("Portfolio Value During Max Drawdown")
            print("=" * 50)
            dd_history = analysis.get("drawdown_history")
            if dd_history is not None and not dd_history.empty:
                print(dd_history.to_string())
            else:
                print("No drawdown period to display.")
                
    def export_to_excel(self, filename="backtest_results.xlsx"):
        """Exports the backtest results to an Excel file with colored transaction history."""
        analysis = self.get_analysis()
        if not analysis:
            print("No analysis data to export.")
            return

        # Combine trade and dividend history into a single DataFrame
        trades_df = analysis["trade_history"]
        dividends_df = analysis["dividend_history"]
        all_transactions_df = (
            pd.concat([trades_df, dividends_df], sort=False)
            .sort_values(by="date")
            .reset_index(drop=True)
        )

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            # --- Summary Sheet ---
            summary_df = pd.DataFrame.from_dict(
                analysis["summary_metrics"], orient="index", columns=["Value"]
            )
            summary_df.index.name = "Metric"
            summary_df.to_excel(writer, sheet_name="Summary")

            # --- All Transactions Sheet (Trades + Dividends) with Colors ---
            if not all_transactions_df.empty:
                all_transactions_df.to_excel(
                    writer, sheet_name="Transactions", index=False
                )
                workbook = writer.book
                worksheet = writer.sheets["Transactions"]
                # Add formats
                buy_format = workbook.add_format({"bg_color": "#C6EFCE"})  # Green
                sell_format = workbook.add_format({"bg_color": "#FFC7CE"})  # Red
                div_format = workbook.add_format({"bg_color": "#FFEB9C"})  # Yellow
                # Color rows based on type
                for i, row in all_transactions_df.iterrows():
                    if row["type"] == "BUY":
                        worksheet.set_row(i + 1, cell_format=buy_format)
                    elif row["type"] == "SELL":
                        worksheet.set_row(i + 1, cell_format=sell_format)
                    elif row["type"] == "DIVIDEND":
                        worksheet.set_row(i + 1, cell_format=div_format)

            # --- Final Positions Sheet ---
            if analysis["final_positions"]:
                pd.DataFrame(analysis["final_positions"]).to_excel(
                    writer, sheet_name="Final Positions", index=False
                )
        print(f"Backtest results successfully exported to {filename}")
