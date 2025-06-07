import backtrader as bt
import pandas as pd
import numpy as np
import numpy_financial as npf
from datetime import date


class BacktestAnalyzer(bt.Analyzer):
    """
    A backtrader analyzer to provide detailed performance metrics,
    supporting both fixed-investment and variable-investment strategies.
    Params:
        - risk_free_rate (float): The annual risk-free rate for Sharpe ratio calculation.
        - fix_investment (bool): If True, enables logic for fixed periodic investments,
          including a monthly IRR calculation. If False, uses standard portfolio analysis.
    """

    params = (("risk_free_rate", 0.04), ("fix_investment", False), ("non_resident_tax_rate", 0.3))

    def __init__(self):
        super().__init__()
        # Data storage
        self.trade_history = []
        self.portfolio_values = []
        self.dates = []
        self.cash_values = []
        self.stock_values = []
        self.investment_schedule = []
        self.final_positions = []

        # Metrics tracking
        self.total_investor_deposits = 0.0
        self.total_dividend_cash = 0.0
        self.total_gross_dividends = 0.0
        self.total_dividend_tax = 0.0

        # For monthly IRR in fix_investment mode
        self.monthly_portfolio_values = []
        self.monthly_dates = []
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
            self.trade_history.append(order_info)

    def record_investment(self, date, amount, reason):
        """Records an investment/deposit transaction."""
        investment_record = {"date": date, "amount": amount, "reason": reason}
        self.investment_schedule.append(investment_record)
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
            "commission": calculated_tax,
            "reason": f"Net: ${net_dividend:.2f} (Tax: {tax_rate*100:.0f}%)",
        }
        self.trade_history.append(dividend_info)
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
        if not hasattr(self, "strategy") or self.strategy is None:
            return

        current_date = self.strategy.datetime.date(0)
        cash_value = self.strategy.broker.get_cash() + self.total_dividend_cash
        stock_value = sum(
            self.strategy.getposition(data).size * data.close[0]
            for data in self.strategy.datas
            if self.strategy.getposition(data).size > 0
        )

        portfolio_value = cash_value + stock_value
        self.portfolio_values.append(portfolio_value)
        self.cash_values.append(cash_value)
        self.stock_values.append(stock_value)
        self.dates.append(current_date)

        if self.p.fix_investment:
            current_month = (current_date.year, current_date.month)
            if self.last_month is None:
                self.last_month = current_month
                self.monthly_portfolio_values.append(portfolio_value)
                self.monthly_dates.append(current_date)
            elif current_month != self.last_month:
                self.monthly_cashflows.append(self.current_month_cashflow)
                self.current_month_cashflow = 0.0

                self.monthly_portfolio_values.append(portfolio_value)
                self.monthly_dates.append(current_date)
                self.last_month = current_month
            else:  # Same month
                if self.monthly_portfolio_values:
                    self.monthly_portfolio_values[-1] = portfolio_value
                    self.monthly_dates[-1] = current_date

    def stop(self):
        """Finalize calculations at the end of the backtest."""
        if self.p.fix_investment:
            # Append the last month's cashflow
            self.monthly_cashflows.append(self.current_month_cashflow)

        self.capture_final_positions(self.strategy)

    def get_analysis(self):
        if not self.dates or not self.portfolio_values:
            return {}

        metrics = self._calculate_metrics()

        self.trade_history.sort(key=lambda x: x["date"])
        trades_df = pd.DataFrame(self.trade_history)

        portfolio_history_df = pd.DataFrame(
            {
                "Date": self.dates,
                "Total Value": self.portfolio_values,
                "Cash": self.cash_values,
                "Stocks": self.stock_values,
                "Cash %": [
                    (c / t * 100) if t > 0 else 0
                    for c, t in zip(self.cash_values, self.portfolio_values)
                ],
                "Stocks %": [
                    (s / t * 100) if t > 0 else 0
                    for s, t in zip(self.stock_values, self.portfolio_values)
                ],
            }
        )

        analysis = {
            "summary_metrics": metrics,
            "trades": trades_df,
            "portfolio_history": portfolio_history_df,
            "final_positions": pd.DataFrame(self.final_positions),
        }
        if self.p.fix_investment:
            analysis["investment_schedule"] = self.investment_schedule

        return analysis

    def _calculate_metrics(self):
        sharpe = self._calculate_sharpe_ratio()
        max_dd = self._calculate_max_drawdown()
        monthly_irr, annual_irr = self._calculate_irr()

        final_value = self.portfolio_values[-1]
        pnl = final_value - self.total_investor_deposits

        return {
            "Start Date": self.dates[0],
            "End Date": self.dates[-1],
            "Initial Portfolio Value": self.portfolio_values[0],
            "Final Portfolio Value": final_value,
            "Total Net Deposits": self.total_investor_deposits,
            "Total Gross Dividends": self.total_gross_dividends,
            "Total Dividend Tax": self.total_dividend_tax,
            "Net P&L": pnl,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Monthly IRR": monthly_irr if self.p.fix_investment else "N/A",
            "Annual IRR": annual_irr,
        }

    def _calculate_irr(self):
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

    def _calculate_sharpe_ratio(self):
        # Using monthly returns for Sharpe Ratio calculation
        # This part of the logic is complex for fix_investment with cashflows,
        # using simple portfolio returns for now.
        returns_df = pd.DataFrame(
            self.portfolio_values,
            index=pd.to_datetime(self.dates),
            columns=["value"],
        )
        monthly_returns = returns_df["value"].resample("ME").last().pct_change().dropna()

        if len(monthly_returns) < 2:
            return 0.0

        monthly_risk_free = (1 + self.p.risk_free_rate) ** (1 / 12) - 1
        excess_returns = monthly_returns - monthly_risk_free
        std_dev = excess_returns.std()

        if std_dev > 0:
            return np.sqrt(12) * excess_returns.mean() / std_dev
        return 0.0

    def _calculate_max_drawdown(self):
        if not self.portfolio_values:
            return 0.0

        portfolio_values_np = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values_np)

        non_zero_peak_indices = peak != 0
        drawdown = np.zeros_like(portfolio_values_np, dtype=float)

        if np.any(non_zero_peak_indices):
            drawdown[non_zero_peak_indices] = (
                peak[non_zero_peak_indices]
                - portfolio_values_np[non_zero_peak_indices]
            ) / peak[non_zero_peak_indices]
            return np.max(drawdown)
        return 0.0

    def capture_final_positions(self, strategy):
        self.final_positions = []
        ticker_dividends = {}
        ticker_dividend_tax = {}

        for trade in self.trade_history:
            if trade["type"] == "DIVIDEND":
                ticker = trade["ticker"]
                ticker_dividends.setdefault(ticker, 0)
                ticker_dividend_tax.setdefault(ticker, 0)
                ticker_dividends[ticker] += trade["value"]
                ticker_dividend_tax[ticker] += trade["commission"]

        for data in strategy.datas:
            position = strategy.getposition(data)
            if position.size > 0:
                ticker = data._name
                current_price = data.close[0]
                market_value = position.size * current_price
                avg_cost = abs(position.price)
                total_cost = position.size * avg_cost
                unrealized_pnl = market_value - total_cost
                unrealized_pnl_pct = (
                    (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
                )

                gross_dividends = ticker_dividends.get(ticker, 0)
                dividend_tax = ticker_dividend_tax.get(ticker, 0)
                net_dividends = gross_dividends - dividend_tax
                total_pnl = unrealized_pnl + net_dividends
                total_pnl_pct = (
                    (total_pnl / total_cost * 100) if total_cost > 0 else 0
                )

                self.final_positions.append(
                    {
                        "Ticker": ticker,
                        "Shares": position.size,
                        "Current Price": current_price,
                        "Market Value": market_value,
                        "Average Cost": avg_cost,
                        "Total Cost": total_cost,
                        "Unrealized P&L": unrealized_pnl,
                        "Unrealized P&L %": unrealized_pnl_pct,
                        "Gross Dividends": gross_dividends,
                        "Dividend Tax": dividend_tax,
                        "Net Dividends": net_dividends,
                        "Total P&L (incl. Div)": total_pnl,
                        "Total P&L %": total_pnl_pct,
                    }
                )

        cash_amount = strategy.broker.get_cash()
        if cash_amount > 0:
            self.final_positions.append(
                {
                    "Ticker": "CASH",
                    "Shares": 1,
                    "Current Price": cash_amount,
                    "Market Value": cash_amount,
                    "Average Cost": cash_amount,
                    "Total Cost": cash_amount,
                    "Unrealized P&L": 0,
                    "Unrealized P&L %": 0,
                    "Gross Dividends": 0,
                    "Dividend Tax": 0,
                    "Net Dividends": 0,
                    "Total P&L (incl. Div)": 0,
                    "Total P&L %": 0,
                }
            )

    def print_summary(self, final_positions=True):
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
            if isinstance(value, float):
                if key in ("Max Drawdown", "Monthly IRR", "Annual IRR"):
                    print(f"{key:25}: {value*100:,.2f}%")
                else:
                    print(f"{key:25}: {value:,.2f}")
            elif isinstance(value, date):
                print(f"{key:25}: {value.strftime('%Y-%m-%d')}")
            else:
                print(f"{key:25}: {value}")

        if final_positions:
            print("\n" + "=" * 50)
            print("Final Positions")
            print("=" * 50)
            final_positions_df = analysis.get("final_positions")
            if final_positions_df is not None and not final_positions_df.empty:
                for _index, position in final_positions_df.iterrows():
                    print("-" * 40)
                    for col, val in position.items():
                        if isinstance(val, float):
                            if "%" in col:
                                print(f"  {col:<25}: {val:,.2f}%")
                            else:
                                print(f"  {col:<25}: {val:,.2f}")
                        else:
                            print(f"  {col:<25}: {val}")
                print("-" * 40)
            else:
                print("No final positions.")

    def export_to_excel(self, filename="backtest_results.xlsx"):
        """
        Exports the backtest analysis to an Excel file.
        Requires `openpyxl` to be installed (`pip install openpyxl`).
        """
        analysis = self.get_analysis()
        if not analysis:
            print("No analysis data to export.")
            return

        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Summary Sheet
                summary_df = pd.DataFrame.from_dict(
                    analysis["summary_metrics"], orient="index", columns=["Value"]
                )
                summary_df.to_excel(writer, sheet_name="Summary")

                # Final Positions Sheet
                if not analysis["final_positions"].empty:
                    analysis["final_positions"].to_excel(
                        writer, sheet_name="Final Positions", index=False
                    )

                # Trades Sheet with colors
                if not analysis["trades"].empty:
                    trades_df = analysis["trades"]

                    def style_trades(row):
                        color = ""
                        if row["type"] == "BUY":
                            color = "background-color: #c6efce"  # light green
                        elif row["type"] == "SELL":
                            color = "background-color: #ffc7ce"  # light red
                        return [color] * len(row)

                    trades_df.style.apply(style_trades, axis=1).to_excel(
                        writer, sheet_name="Trading History", index=False
                    )

                # Portfolio History Sheet
                if not analysis["portfolio_history"].empty:
                    analysis["portfolio_history"].to_excel(
                        writer, sheet_name="Portfolio History", index=False
                    )

            print(f"Successfully exported backtest results to {filename}")
        except ImportError:
            print("Please install 'openpyxl' to export to Excel: pip install openpyxl")
        except Exception as e:
            print(f"An error occurred while exporting to Excel: {e}")
