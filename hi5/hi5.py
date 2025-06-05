import backtrader as bt


import pandas as pd
import numpy as np

import json
from abc import abstractmethod
import os  # 用于LocalStorageEngine检查文件路径

# Add Google Cloud Storage
try:
    from google.cloud import storage
except ImportError:
    storage = None


class Hi5PandasDataWithDividends(bt.feeds.PandasData):
    lines = ("dividends",)
    params = (
        ("dividends", None),
    )


class Hi5StateStorageEngine:
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def get(self, key, default=None):  # 添加default参数
        pass

    @abstractmethod
    def set(self, key, value):
        pass


class LocalStorageEngine(Hi5StateStorageEngine):
    def __init__(self, local_file_path):
        self.local_file_path = local_file_path
        self.data = {}
        # 确保目录存在
        os.makedirs(os.path.dirname(self.local_file_path), exist_ok=True)

    def load(self):
        try:
            if os.path.exists(self.local_file_path):
                with open(self.local_file_path, "r") as f:
                    content = f.read()
                    if content:  # 确保文件内容不为空
                        self.data = json.load(f)
                    else:
                        self.data = {}  # 文件为空，初始化为空字典
            else:
                self.data = {}  # 文件不存在，初始化为空字典
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from {self.local_file_path}. Initializing with empty state."
            )
            self.data = {}  # JSON解析错误，初始化为空字典
        except Exception as e:
            print(
                f"Error loading state from {self.local_file_path}: {e}. Initializing with empty state."
            )
            self.data = {}

    def save(self):
        try:
            with open(self.local_file_path, "w") as f:
                json.dump(self.data, f, indent=4)  # indent for readability
        except Exception as e:
            print(f"Error saving state to {self.local_file_path}: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class InMemoryStorageEngine(Hi5StateStorageEngine):
    def __init__(self):
        self.data = {}

    def load(self):
        pass  # 内存引擎不需要从外部加载

    def save(self):
        pass  # 内存引擎的"保存"是即时的，不需要显式操作

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class GCPStorageEngine(Hi5StateStorageEngine):
    def __init__(self, bucket_name, blob_name, credentials_path=None):
        if storage is None:
            raise RuntimeError("google-cloud-storage is not installed. GCPStorageEngine cannot be used. Please install google-cloud-storage or choose a different engine.")

        try:
            # Initialize the GCP client
            if credentials_path and os.path.exists(credentials_path):
                self.client = storage.Client.from_service_account_json(credentials_path)
            else:
                self.client = storage.Client()

            # Get bucket and blob
            self.bucket = self.client.bucket(bucket_name)
            self.blob = self.bucket.blob(blob_name)

            # Verify bucket exists
            if not self.bucket.exists():
                raise RuntimeError(f"Bucket {bucket_name} does not exist. Please create it first.")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCP client: {e}. Please check your GCP credentials and configuration.")

    def load(self):
        try:
            if self.blob.exists():
                content = self.blob.download_as_string().decode('utf-8')
                if content:
                    self.data = json.loads(content)
                else:
                    self.data = {}
            else:
                print(f"GCP blob {self.blob.name} does not exist in bucket {self.bucket.name}. Initializing empty state.")
                self.data = {}
        except Exception as e:
            print(f"Error loading state from GCP: {e}. Initializing with empty state.")
            self.data = {}

    def save(self):
        try:
            self.blob.upload_from_string(
                json.dumps(self.data, indent=4),
                content_type='application/json'
            )
        except Exception as e:
            print(f"Error saving state to GCP: {e}")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


class Hi5State:
    def __init__(self, storage_engine: Hi5StateStorageEngine):
        self.storage_engine = storage_engine
        # 从存储引擎加载初始值或使用默认值
        self.restore_state()  # 在初始化时尝试恢复状态

    def _initialize_defaults(self):
        """Helper to set default values for attributes if not loaded."""
        self.current_month = getattr(self, "current_month", None)
        self.first_exec = getattr(self, "first_exec", False)
        self.second_exec = getattr(self, "second_exec", False)
        self.third_exec = getattr(self, "third_exec", False)
        self.orders = getattr(
            self, "orders", []
        )  # 注意：这里的orders是Hi5State内部跟踪的，可能与Analyzer的orders不同
        self.current_month_cash_flow = getattr(self, "current_month_cash_flow", 0)
        self.invested_amount = getattr(self, "invested_amount", 0)  # 总投入资本
        self.rsp_month_start_price = getattr(self, "rsp_month_start_price", None)
        self.total_dividends_gross = getattr(
            self, "total_dividends_gross", 0.0
        )  # 税前总股息
        self.total_tax_paid_on_dividends = getattr(
            self, "total_tax_paid_on_dividends", 0.0
        )  # 总股息税
        self.rebalanced_this_year_august = getattr(
            self, "rebalanced_this_year_august", False
        )  # 用于年度再平衡的标志

    def restore_state(self):
        self.storage_engine.load()
        self.current_month = self.storage_engine.get(
            "current_month"
        )  # 默认None由get处理
        self.first_exec = self.storage_engine.get("first_exec", False)
        self.second_exec = self.storage_engine.get("second_exec", False)
        self.third_exec = self.storage_engine.get("third_exec", False)
        self.orders = self.storage_engine.get("orders", [])
        self.rsp_month_start_price = self.storage_engine.get("rsp_month_start_price")
        self.current_month_cash_flow = self.storage_engine.get(
            "current_month_cash_flow", 0
        )
        self.invested_amount = self.storage_engine.get("invested_amount", 0)
        self.total_dividends_gross = self.storage_engine.get(
            "total_dividends_gross", 0.0
        )
        self.total_tax_paid_on_dividends = self.storage_engine.get(
            "total_tax_paid_on_dividends", 0.0
        )
        self.rebalanced_this_year_august = self.storage_engine.get(
            "rebalanced_this_year_august", False
        )
        self._initialize_defaults()  # 确保所有属性都被初始化

    def save_state(self):
        self.storage_engine.set("current_month", self.current_month)
        self.storage_engine.set("first_exec", self.first_exec)
        self.storage_engine.set("second_exec", self.second_exec)
        self.storage_engine.set("third_exec", self.third_exec)
        self.storage_engine.set("current_month_cash_flow", self.current_month_cash_flow)
        self.storage_engine.set(
            "orders", self.orders
        )  # 注意这会覆盖分析器中的交易，需要区分
        self.storage_engine.set("rsp_month_start_price", self.rsp_month_start_price)
        self.storage_engine.set("invested_amount", self.invested_amount)
        self.storage_engine.set("total_dividends_gross", self.total_dividends_gross)
        self.storage_engine.set(
            "total_tax_paid_on_dividends", self.total_tax_paid_on_dividends
        )
        self.storage_engine.set(
            "rebalanced_this_year_august", self.rebalanced_this_year_august
        )
        self.storage_engine.save()

    def refresh_to_new_month(self, rsp_data_feed):  # 需要传入RSP的data feed
        current_date = rsp_data_feed.datetime.date(0)
        self.current_month = current_date.month
        self.first_exec = False
        self.second_exec = False
        self.third_exec = False
        # self.orders = [] # 这个orders是Hi5State内部的，如果每月重置，会丢失长期记录
        self.current_month_cash_flow = 0
        if len(rsp_data_feed.close) > 0:
            self.rsp_month_start_price = rsp_data_feed.close[0]
        else:
            self.rsp_month_start_price = None  # 处理数据不足的情况
            print(f"Warning: RSP data feed empty at month start: {current_date}")

        # 如果当前月份是1月，重置年度再平衡标志
        if self.current_month == 1:
            self.rebalanced_this_year_august = False

        self.save_state()


# 主要用来存储回测数据，live trading 不需要
class BacktestAnalyzer(bt.Analyzer):
    params = (
        ('risk_free_rate', 0.04),
    )

    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.trade_history = []  # 用于存储所有交易和股息事件
        self.portfolio_values = []
        self.dates = []
        
        # Track cash and stock values separately
        self.cash_values = []  # Track cash portion
        self.stock_values = []  # Track stock portion
        self.portfolio_history = []  # Track detailed portfolio breakdown
        
        self.monthly_returns = []  # Track monthly returns instead of daily
        self.monthly_portfolio_values = []  # Track end-of-month portfolio values
        self.monthly_dates = []  # Track end-of-month dates
        self.last_month = None  # Track current month for monthly calculations
        
        # Track REAL investment schedule (what investor needs to deposit)
        self.investment_schedule = []  # Track when investments are made and how much
        self.total_investor_deposits = 0.0  # Total deposits investor needs to make
        self.total_dividend_cash = 0.0  # Total net dividend cash received
        
        # Track final positions
        self.final_positions = []

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(
            0
        )  # self.datas[0] 可能在analyzer中不可靠，最好从strategy传递
        # Safer: dt = dt or self.strategy.datetime.date(0)
        print(f"[{dt.strftime('%Y-%m-%d')}] {txt}")

    def notify_order(self, order):
        """Track order execution for trade history only"""
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
                "size": order.executed.size
                * (1 if order.isbuy() else -1),  # size为正代表买入，为负代表卖出
                "value": order.executed.value,  # 总是正值
                "commission": order.executed.comm,
                "reason": getattr(order, "reason", "N/A"),
            }
            self.trade_history.append(order_info)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.Status[order.status]}: {order.data._name}")

    def record_investment(self, date, amount, reason):
        """Record when strategy makes an investment - this represents required external deposit"""
        investment_record = {
            "date": date,
            "amount": amount,
            "reason": reason
        }
        self.investment_schedule.append(investment_record)
        self.total_investor_deposits += amount
        
        self.log(f"Investment recorded: ${amount:.2f} on {date} for {reason} (Total deposits: ${self.total_investor_deposits:.2f})")

    def add_dividend_event(
        self, date, ticker, gross_amount, tax_amount, net_amount, shares_held
    ):
        """由策略调用以记录股息事件"""
        dividend_info = {
            "date": date,
            "ticker": ticker,
            "type": "DIVIDEND",
            "price": gross_amount / shares_held if shares_held else 0,  # 每股股息
            "size": shares_held,  # 持有股数
            "value": gross_amount,  # 税前总额
            "commission": tax_amount,  # 税费记录在佣金字段
            "reason": f"Net: ${net_amount:.2f}",
        }
        self.trade_history.append(dividend_info)
        
        # Track dividend cash for reference
        self.total_dividend_cash += net_amount
        self.record_investment(date, -net_amount, "Dividend")

    def next(self):
        # 确保 self.strategy 存在
        if not hasattr(self, "strategy") or self.strategy is None:
            return

        current_date = self.strategy.datetime.date(0)  # 使用strategy的datetime
        
        # Use cumulative dividend cash for correct cash value
        cash_value = self.strategy.broker.get_cash() + self.total_dividend_cash
        
        # Calculate total stock value by summing all positions
        stock_value = 0.0
        position_details = {}
        
        for data in self.strategy.datas:
            position = self.strategy.getposition(data)
            if position.size > 0:
                current_price = data.close[0]
                position_value = position.size * current_price
                stock_value += position_value
                
                position_details[data._name] = {
                    'shares': position.size,
                    'price': current_price,
                    'value': position_value
                }
        
        # Total portfolio value
        portfolio_value = cash_value + stock_value
        
        # Store all values
        self.portfolio_values.append(portfolio_value)
        self.cash_values.append(cash_value)  # This now includes dividends from _check_dividends
        self.stock_values.append(stock_value)
        self.dates.append(current_date)

        # Store detailed portfolio breakdown
        portfolio_detail = {
            'date': current_date,
            'cash': cash_value,
            'stocks': stock_value,
            'total': portfolio_value,
            'cash_pct': (cash_value / portfolio_value * 100) if portfolio_value > 0 else 0,
            'stocks_pct': (stock_value / portfolio_value * 100) if portfolio_value > 0 else 0,
            'positions': position_details
        }
        self.portfolio_history.append(portfolio_detail)

        # Track monthly returns instead of daily returns
        current_month = (current_date.year, current_date.month)
        
        # If this is a new month or the first data point
        if self.last_month is None:
            self.last_month = current_month
            self.monthly_portfolio_values.append(portfolio_value)
            self.monthly_dates.append(current_date)
        elif current_month != self.last_month:
            # New month detected - calculate monthly return
            if len(self.monthly_portfolio_values) > 0:
                prev_value = self.monthly_portfolio_values[-1]
                if prev_value > 0:
                    monthly_return = (portfolio_value / prev_value) - 1
                    self.monthly_returns.append(monthly_return)
            
            # Update monthly tracking
            self.monthly_portfolio_values.append(portfolio_value)
            self.monthly_dates.append(current_date)
            self.last_month = current_month
        else:
            # Same month - update the current month's portfolio value
            if self.monthly_portfolio_values:
                self.monthly_portfolio_values[-1] = portfolio_value
                self.monthly_dates[-1] = current_date

    def get_analysis(self):
        if not self.dates or not self.portfolio_values:
            return {
                "sharpe_ratio": 0.0,
                "monthly_irr": 0.0,
                "annual_irr": 0.0,
                "max_drawdown": 0.0,
                "total_investor_deposits": 0.0,
                "trades": pd.DataFrame(),
                "portfolio_values": pd.DataFrame(),
                "portfolio_history": pd.DataFrame(),
            }

        # Calculate monthly and annual IRR based on real investor deposits
        monthly_irr, annual_irr = self.calculate_monthly_irr()
        if monthly_irr is None:
            monthly_irr = 0.0
        if annual_irr is None:
            annual_irr = 0.0

        # Sharpe Ratio based on monthly returns
        sharpe_ratio = 0.0
        if len(self.monthly_returns) > 1:
            returns_np = np.array(self.monthly_returns)
            monthly_risk_free = (1 + self.p.risk_free_rate) ** (1/12) - 1
            
            excess_returns = returns_np - monthly_risk_free
            sharpe_ratio = np.sqrt(12) * (np.mean(excess_returns) / np.std(excess_returns))

        # Max Drawdown
        max_drawdown = 0.0
        if len(self.portfolio_values) > 0:
            portfolio_values_np = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(portfolio_values_np)
            non_zero_peak_indices = peak != 0
            drawdown = np.zeros_like(portfolio_values_np, dtype=float)
            drawdown[non_zero_peak_indices] = (
                peak[non_zero_peak_indices] - portfolio_values_np[non_zero_peak_indices]
            ) / peak[non_zero_peak_indices]
            if len(drawdown) > 0:
                max_drawdown = np.max(drawdown)

        # Trades DataFrame
        self.trade_history.sort(key=lambda x: x["date"])
        trades_df = pd.DataFrame(self.trade_history)

        return {
            "sharpe_ratio": sharpe_ratio,
            "monthly_irr": monthly_irr,
            "annual_irr": annual_irr,
            "max_drawdown": max_drawdown,
            "total_investor_deposits": self.total_investor_deposits,
            "investment_schedule": self.investment_schedule,
            "trades": trades_df,
            "portfolio_values_df": pd.DataFrame(
                {"Date": self.dates, "Portfolio Value": self.portfolio_values}
            ),
            "portfolio_history_df": pd.DataFrame({
                "Date": self.dates,
                "Total Value": self.portfolio_values,
                "Cash": self.cash_values,
                "Stocks": self.stock_values,
                "Cash %": [(cash / total * 100) if total > 0 else 0 for cash, total in zip(self.cash_values, self.portfolio_values)],
                "Stocks %": [(stock / total * 100) if total > 0 else 0 for stock, total in zip(self.stock_values, self.portfolio_values)]
            }),
            "monthly_returns": self.monthly_returns,
            "monthly_dates": self.monthly_dates[1:] if len(self.monthly_dates) > 1 else [],  # Skip first date since no return calculated
        }

    def calculate_irr_cash_flows(self):
        """Calculate cash flows for IRR using REAL investor deposits"""
        if not self.investment_schedule:
            return [], []
        
        # Group deposits by month
        monthly_deposits = {}
        
        for investment in self.investment_schedule:
            # Use year-month as key for grouping
            investment_date = investment["date"]
            month_key = (investment_date.year, investment_date.month)
            if month_key not in monthly_deposits:
                monthly_deposits[month_key] = {
                    "date": investment_date,
                    "total_amount": 0.0
                }
            monthly_deposits[month_key]["total_amount"] += investment["amount"]
        
        # Create chronological cash flows
        cash_flows = []
        cash_flow_dates = []
        
        # Sort monthly deposits by date
        sorted_months = sorted(monthly_deposits.items(), key=lambda x: x[0])
        
        for (year, month), data in sorted_months:
            cash_flows.append(-data["total_amount"])  # Negative for investor deposits (outflows)
            cash_flow_dates.append(data["date"])
        
        # Add final portfolio value as positive cash flow
        if self.portfolio_values and self.dates:
            final_value = self.portfolio_values[-1]
            cash_flows.append(final_value)  # Positive for final portfolio value
            cash_flow_dates.append(self.dates[-1])
        
        return cash_flows, cash_flow_dates

    def calculate_monthly_irr(self):
        """Calculate monthly IRR based on actual investor deposits"""
        try:
            import numpy_financial as npf
            
            cash_flows, dates = self.calculate_irr_cash_flows()
            
            if len(cash_flows) < 2:  # Need at least deposit and final value
                return None, None
            
            # Convert to monthly periods
            start_date = dates[0]
            monthly_periods = []
            
            for date in dates:
                months_diff = (date.year - start_date.year) * 12 + (date.month - start_date.month)
                monthly_periods.append(months_diff)
            
            # Calculate monthly IRR using numpy financial
            try:
                # Create monthly cash flow array
                max_period = max(monthly_periods)
                monthly_cf_array = [0] * (max_period + 1)
                
                for i, period in enumerate(monthly_periods):
                    monthly_cf_array[period] += cash_flows[i]
                
                monthly_irr = npf.irr(monthly_cf_array)
                if monthly_irr is not None and not np.isnan(monthly_irr):
                    annual_irr = (1 + monthly_irr) ** 12 - 1
                    return monthly_irr, annual_irr
                
            except:
                pass
            
            # Fallback calculation
            if len(cash_flows) >= 2:
                total_deposits = sum(abs(cf) for cf in cash_flows[:-1])
                final_value = cash_flows[-1]
                total_months = monthly_periods[-1]
                
                if total_months > 0 and total_deposits > 0:
                    total_return = (final_value / total_deposits) - 1
                    monthly_irr = (1 + total_return) ** (1 / total_months) - 1
                    annual_irr = (1 + monthly_irr) ** 12 - 1
                    return monthly_irr, annual_irr
            
            return None, None
            
        except ImportError:
            # Fallback calculation without numpy_financial
            if len(cash_flows) >= 2:
                total_deposits = sum(abs(cf) for cf in cash_flows[:-1])
                final_value = cash_flows[-1]
                total_months = (dates[-1].year - dates[0].year) * 12 + (dates[-1].month - dates[0].month)
                
                if total_months > 0 and total_deposits > 0:
                    total_return = (final_value / total_deposits) - 1
                    monthly_irr = (1 + total_return) ** (1 / total_months) - 1
                    annual_irr = (1 + monthly_irr) ** 12 - 1
                    return monthly_irr, annual_irr
            
            return None, None
        except:
            return None, None

    def capture_final_positions(self, strategy):
        """Capture final positions from the strategy including dividend tracking"""
        self.final_positions = []
        
        # Calculate total dividends per ticker from trade history
        ticker_dividends = {}
        ticker_dividend_tax = {}
        
        for trade in self.trade_history:
            if trade['type'] == 'DIVIDEND':
                ticker = trade['ticker']
                if ticker not in ticker_dividends:
                    ticker_dividends[ticker] = 0
                    ticker_dividend_tax[ticker] = 0
                ticker_dividends[ticker] += trade['value']  # Gross dividends
                ticker_dividend_tax[ticker] += trade['commission']  # Tax paid
        
        # Track positions
        for data in strategy.datas:
            position = strategy.getposition(data)
            ticker = data._name
            
            if position.size > 0:  # Only include positions with holdings
                current_price = data.close[0]
                market_value = position.size * current_price
                avg_cost = abs(position.price) if position.price != 0 else 0
                total_cost = position.size * avg_cost
                unrealized_pnl = market_value - total_cost
                unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
                
                # Get dividends for this ticker
                gross_dividends = ticker_dividends.get(ticker, 0)
                dividend_tax = ticker_dividend_tax.get(ticker, 0)
                net_dividends = gross_dividends - dividend_tax
                
                # Calculate total P&L including dividends
                total_pnl = unrealized_pnl + net_dividends
                total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
                
                position_info = {
                    'Ticker': ticker,
                    'Shares': position.size,
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Average Cost': avg_cost,
                    'Total Cost': total_cost,
                    'Unrealized P&L': unrealized_pnl,
                    'Unrealized P&L %': unrealized_pnl_pct,
                    'Gross Dividends': gross_dividends,
                    'Dividend Tax': dividend_tax,
                    'Net Dividends': net_dividends,
                    'Total P&L (incl. Div)': total_pnl,
                    'Total P&L %': total_pnl_pct,
                }
                self.final_positions.append(position_info)
        
        # Add cash position
        cash_amount = strategy.broker.get_cash()
        if cash_amount > 0:
            self.final_positions.append({
                'Ticker': 'CASH',
                'Shares': 1,
                'Current Price': cash_amount,
                'Market Value': cash_amount,
                'Average Cost': cash_amount,
                'Total Cost': cash_amount,
                'Unrealized P&L': 0,
                'Unrealized P&L %': 0,
                'Gross Dividends': 0,
                'Dividend Tax': 0,
                'Net Dividends': 0,
                'Total P&L (incl. Div)': 0,
                'Total P&L %': 0,
            })

    def export_to_excel(self, tickers, start_date, end_date, filename=None):
        """Enhanced Excel export using REAL investor deposits for all calculations"""
        
        # Generate filename if not provided
        if filename is None:
            filename = f"QUANT_FOR_{'-'.join(tickers)}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.xlsx"
        
        # Create Excel writer
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        buy_format = workbook.add_format({'bg_color': '#C6EFCE'})  # Light green
        sell_format = workbook.add_format({'bg_color': '#FFC7CE'})  # Light red
        dividend_format = workbook.add_format({'bg_color': '#FFEB9C'})  # Light yellow
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        currency_format = workbook.add_format({'num_format': '$#,##0.00'})
        percentage_format = workbook.add_format({'num_format': '0.00%'})
        
        # Export final positions as the first sheet
        if self.final_positions:
            positions_df = pd.DataFrame(self.final_positions)
            positions_df.to_excel(writer, sheet_name='Final Positions', index=False)
            
            worksheet = writer.sheets['Final Positions']
            
            # Format headers
            for col_num, value in enumerate(positions_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Apply formatting to data
            for idx, row in positions_df.iterrows():
                worksheet.write(idx + 1, 0, row['Ticker'])  # Ticker
                worksheet.write(idx + 1, 1, row['Shares'])  # Shares
                worksheet.write(idx + 1, 2, row['Current Price'], currency_format)  # Current Price
                worksheet.write(idx + 1, 3, row['Market Value'], currency_format)  # Market Value
                worksheet.write(idx + 1, 4, row['Average Cost'], currency_format)  # Average Cost
                worksheet.write(idx + 1, 5, row['Total Cost'], currency_format)  # Total Cost
                worksheet.write(idx + 1, 6, row['Unrealized P&L'], currency_format)  # Unrealized P&L
                worksheet.write(idx + 1, 7, row['Unrealized P&L %'] / 100, percentage_format)  # Unrealized P&L %
                worksheet.write(idx + 1, 8, row['Gross Dividends'], currency_format)  # Gross Dividends
                worksheet.write(idx + 1, 9, row['Dividend Tax'], currency_format)  # Dividend Tax
                worksheet.write(idx + 1, 10, row['Net Dividends'], currency_format)  # Net Dividends
                worksheet.write(idx + 1, 11, row['Total P&L (incl. Div)'], currency_format)  # Total P&L
                worksheet.write(idx + 1, 12, row['Total P&L %'] / 100, percentage_format)  # Total P&L %
            
            # Adjust column widths
            worksheet.set_column('A:A', 10)  # Ticker
            worksheet.set_column('B:B', 12)  # Shares
            worksheet.set_column('C:M', 15)  # All other columns
            
            # Add totals row
            total_row = len(positions_df) + 2
            worksheet.write(total_row, 0, 'TOTAL', header_format)
            worksheet.write(total_row, 3, f'=SUM(D2:D{len(positions_df)+1})', currency_format)  # Total Market Value
            worksheet.write(total_row, 5, f'=SUM(F2:F{len(positions_df)+1})', currency_format)  # Total Cost
            worksheet.write(total_row, 6, f'=SUM(G2:G{len(positions_df)+1})', currency_format)  # Total Unrealized P&L
            worksheet.write(total_row, 8, f'=SUM(I2:I{len(positions_df)+1})', currency_format)  # Total Gross Dividends
            worksheet.write(total_row, 9, f'=SUM(J2:J{len(positions_df)+1})', currency_format)  # Total Dividend Tax
            worksheet.write(total_row, 10, f'=SUM(K2:K{len(positions_df)+1})', currency_format)  # Total Net Dividends
            worksheet.write(total_row, 11, f'=SUM(L2:L{len(positions_df)+1})', currency_format)  # Total P&L (incl. Div)

        # Export order history with formatting
        trades_df = pd.DataFrame(self.trade_history)
        if not trades_df.empty:
            trades_df = trades_df.sort_values('date')
            trades_df.to_excel(writer, sheet_name='Trade History', startrow=0, index=False)
            
            worksheet = writer.sheets['Trade History']
            
            # Format headers
            for col_num, value in enumerate(trades_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Apply formatting based on trade type
            for idx, row in trades_df.iterrows():
                # Format date column
                worksheet.write(idx + 1, 0, row['date'], date_format)
                
                if row['type'] == 'BUY':
                    for col in range(1, len(trades_df.columns)):
                        worksheet.write(idx + 1, col, row[trades_df.columns[col]], buy_format)
                elif row['type'] == 'SELL':
                    for col in range(1, len(trades_df.columns)):
                        worksheet.write(idx + 1, col, row[trades_df.columns[col]], sell_format)
                elif row['type'] == 'DIVIDEND':
                    for col in range(1, len(trades_df.columns)):
                        worksheet.write(idx + 1, col, row[trades_df.columns[col]], dividend_format)
            
            # Adjust column widths
            worksheet.set_column('A:A', 15)  # Date column
            worksheet.set_column('B:B', 10)  # Ticker column
            worksheet.set_column('C:C', 10)  # Type column
            worksheet.set_column('D:H', 12)  # Numeric columns
        
        # Export portfolio history with cash and stock breakdown
        analysis_results = self.get_analysis()
        portfolio_history_df = analysis_results['portfolio_history_df']
        
        if not portfolio_history_df.empty:
            # Select only the four columns we want
            portfolio_history_df = portfolio_history_df[['Date', 'Total Value', 'Cash', 'Stocks']]
            portfolio_history_df.to_excel(writer, sheet_name='Portfolio History', index=False)
            worksheet = writer.sheets['Portfolio History']
            
            # Format headers
            for col_num, value in enumerate(portfolio_history_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Apply formatting to data
            for idx, row in portfolio_history_df.iterrows():
                worksheet.write(idx + 1, 0, row['Date'], date_format)  # Date
                worksheet.write(idx + 1, 1, row['Total Value'], currency_format)  # Total Value
                worksheet.write(idx + 1, 2, row['Cash'], currency_format)  # Cash
                worksheet.write(idx + 1, 3, row['Stocks'], currency_format)  # Stocks
            
            # Adjust column widths
            worksheet.set_column('A:A', 15)  # Date column
            worksheet.set_column('B:D', 15)  # Value columns

        # Calculate metrics using REAL investor deposits
        annual_irr = analysis_results['annual_irr']
        max_drawdown = analysis_results['max_drawdown']
        sharpe_ratio = analysis_results['sharpe_ratio']
        total_investor_deposits = analysis_results['total_investor_deposits']
        
        # Calculate additional metrics based on real deposits
        final_value = self.portfolio_values[-1] if self.portfolio_values else 0
        total_return = (final_value - total_investor_deposits) / total_investor_deposits if total_investor_deposits > 0 else 0
        
        # Count trade types
        if not trades_df.empty:
            buy_orders = len([t for t in trades_df.to_dict('records') if t['type'] == 'BUY'])
            sell_orders = len([t for t in trades_df.to_dict('records') if t['type'] == 'SELL'])
            dividend_events = len([t for t in trades_df.to_dict('records') if t['type'] == 'DIVIDEND'])
            total_commission = trades_df[trades_df['type'].isin(['BUY', 'SELL'])]['commission'].sum()
            total_dividends_gross = trades_df[trades_df['type'] == 'DIVIDEND']['value'].sum()
            total_dividend_tax = trades_df[trades_df['type'] == 'DIVIDEND']['commission'].sum()
            avg_order_size = trades_df['size'].abs().mean() if not trades_df.empty else 0
        else:
            buy_orders = sell_orders = dividend_events = 0
            total_commission = total_dividends_gross = total_dividend_tax = avg_order_size = 0

        # Prepare metrics data with corrected labels
        metrics_list = [
            'Total Investor Deposits Required',
            'Final Portfolio Value',
            'Total Return (on Deposits)',
            'Annual IRR (on Deposits)',
            'Maximum Drawdown',
            'Sharpe Ratio',
            'Total Orders',
            'Total Buy Orders',
            'Total Sell Orders',
            'Total Dividend Events',
            'Average Order Size',
            'Total Commission Paid',
            'Total Dividends (Gross)',
            'Total Dividend Tax',
            'Net Dividends Received',
            'Net Profit (Final Value - Deposits)'
        ]
        
        net_profit = final_value - total_investor_deposits
        
        hi5_values = [
            total_investor_deposits,
            final_value,
            total_return,
            annual_irr,
            max_drawdown,
            sharpe_ratio,
            len(trades_df) if not trades_df.empty else 0,
            buy_orders,
            sell_orders,
            dividend_events,
            avg_order_size,
            total_commission,
            total_dividends_gross,
            total_dividend_tax,
            total_dividends_gross - total_dividend_tax,
            net_profit
        ]
             
        # Export comprehensive metrics
        metrics_data = {
            'Metric': metrics_list,
            'Value': hi5_values
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
        worksheet = writer.sheets['Summary Metrics']
        for col_num, value in enumerate(metrics_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:A', 25)  # Metric names
        worksheet.set_column('B:B', 20)  # Values
        
        # Export cash injection history for verification
        investment_schedule = getattr(self.strategy, 'investment_schedule', [])
        if investment_schedule:
            investment_df = pd.DataFrame(investment_schedule)
            investment_df.to_excel(writer, sheet_name='Investment Schedule', index=False)
            worksheet = writer.sheets['Investment Schedule']
            for col_num, value in enumerate(investment_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
        
        # Export cash flows for IRR calculation (based on real deposits)
        cash_flows, cash_flow_dates = self.calculate_irr_cash_flows()
        if cash_flows:
            cash_flow_df = pd.DataFrame({
                'Date': cash_flow_dates,
                'Cash Flow': cash_flows,
                'Type': ['Investment' if cf < 0 else 'Final Value' for cf in cash_flows]
            })
            cash_flow_df.to_excel(writer, sheet_name='IRR Cash Flows', index=False)
            worksheet = writer.sheets['IRR Cash Flows']
            for col_num, value in enumerate(cash_flow_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
        
        # Save the Excel file
        writer.close()
        print(f"Results exported to {filename}")
        print(f"Total investor deposits required: ${total_investor_deposits:,.2f}")
        print(f"Final portfolio value: ${final_value:,.2f}")
        print(f"Final cash: ${self.cash_values[-1]:,.2f} ({self.cash_values[-1]/final_value*100:.1f}%)")
        print(f"Final stocks: ${self.stock_values[-1]:,.2f} ({self.stock_values[-1]/final_value*100:.1f}%)")
        print(f"Annual IRR (on deposits): {annual_irr:.2%}")
        print(f"Net profit: ${net_profit:,.2f}")
        
        return filename


class Hi5Strategy(bt.Strategy):
    params = (
        ("cash_per_contribution", 10000),
        ("extreme_market_breadth_threshold", 0.2),  # 未使用
        ("market_breadth_days", 120),  # 未使用
        ("non_resident_tax_rate", 0.3),
        ("extreme_condition_invest_factor", 2),  # 未使用
        ("tickers", ["VUG", "VO", "MOAT", "PFF", "VNQ"]),  # <--- UPDATED
        ("benchmark_tickers", ["RSP"]),  # <--- NEW: for calculation only
        ("min_period", 14),  # K线最小周期
        ("state_storage_path", "hi5_strategy_state.json"),
        ("enable_cash_injection", True),
        ("cash_injection_threshold", 3),
    )

    def __init__(self, state: Hi5State):  # state 通过外部注入
        self.state = state
        self.rsp_data = None  # 用于存储RSP的data feed
        self.dividend_events_for_analyzer = []  # 临时存储股息事件给analyzer
        self.total_cash_injected = 0.0  # Track total cash injected for reporting
        self.cash_injection_history = []  # Track when cash was injected
        self.total_dividend_cash_received = 0.0  # Track total net dividend cash
        self.last_cash_injection_check = None  # Track last time we checked for injection

    def log(self, txt, dt=None):
        """Custom log method to handle logging"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

    def start(self):
        # Find RSP data feed for benchmark logic
        self.rsp_data = None
        for data in self.datas:
            if data._name == "RSP":
                self.rsp_data = data
                break
        # If RSP is not found, crash immediately
        assert self.rsp_data is not None, "RSP data feed (benchmark_ticker) is required but not found!"

        # 尝试恢复状态（如果Hi5State的__init__没有自动做的话）
        # self.state.restore_state() # 已在Hi5State的__init__中调用

        self.log(f"Strategy started. Initial cash: {self.broker.get_cash():.2f}")
        if self.p.enable_cash_injection:
            self.log(f"Cash injection enabled. Minimum cash threshold: {self.p.cash_injection_threshold * self.p.cash_per_contribution:.2f}")
        self.log(
            f"Initial state: Current Month: {self.state.current_month}, First Exec: {self.state.first_exec}, RSP Month Start: {self.state.rsp_month_start_price}"
        )

    def _check_and_inject_cash(self):
        """Check if cash injection is needed and inject if necessary (backtest mode only)"""
        if not self.p.enable_cash_injection:
            return
            
        current_date = self.datetime.date(0)
        current_cash = self.broker.get_cash()
        minimum_cash_required = self.p.cash_injection_threshold * self.p.cash_per_contribution
        
        # Only check once per day to avoid multiple injections
        if self.last_cash_injection_check == current_date:
            return
        self.last_cash_injection_check = current_date
        
        if current_cash < minimum_cash_required:
            # Calculate how much cash to inject to reach the minimum threshold
            cash_to_inject = minimum_cash_required - current_cash
            
            # Add cash to broker
            self.broker.add_cash(cash_to_inject)
            self.total_cash_injected += cash_to_inject
            
            # Record injection for tracking
            injection_record = {
                "date": current_date,
                "amount": cash_to_inject,
                "cash_before": current_cash,
                "cash_after": self.broker.get_cash(),
                "total_dividend_cash": self.total_dividend_cash_received
            }
            self.cash_injection_history.append(injection_record)
            
            self.log(
                f"Cash injection: ${cash_to_inject:.2f} (Before: ${current_cash:.2f}, After: ${self.broker.get_cash():.2f}, Total injected: ${self.total_cash_injected:.2f}, Dividend cash: ${self.total_dividend_cash_received:.2f})"
        )

    def _check_dividends(self):
        current_date = self.datetime.date(0)
        for data in self.datas: # data.dividends[0] 可能为 None
            pos = self.getposition(data)
            if pos.size > 0:  # 只有持仓时才处理股息
                gross_dividend_per_share = data.dividends[0]
            if gross_dividend_per_share <= 0:
                continue
            
            gross_total_dividend = pos.size * gross_dividend_per_share
            tax_amount = gross_total_dividend * self.p.non_resident_tax_rate
            net_dividend = gross_total_dividend - tax_amount

            # Backtrader 自动将税前股息添加到现金。我们需要手动扣除税款。
            self.broker.add_cash(net_dividend)
                
            # Track net dividend cash received
            self.total_dividend_cash_received += net_dividend

            # 更新 Hi5State 中的累计股息和税款
            self.state.total_dividends_gross += gross_total_dividend
            self.state.total_tax_paid_on_dividends += tax_amount

            self.log(
                f"Dividend for {data._name}: {pos.size} shares * ${gross_dividend_per_share:.4f}/share = "
                f"Gross ${gross_total_dividend:.2f}, Tax ${tax_amount:.2f}, Net ${net_dividend:.2f}. Cash: {self.broker.get_cash():.2f}, Total div cash: ${self.total_dividend_cash_received:.2f}"
            )

            # 通知分析器
            # 检查是否有附加的分析器并且它有 add_dividend_event 方法
            for analyzer in self.analyzers:
                if hasattr(analyzer, "add_dividend_event"):
                    analyzer.add_dividend_event(
                        current_date,
                        data._name,
                        gross_total_dividend,
                        tax_amount,
                        net_dividend,
                        pos.size,
                    )

    def buy_etfs(self, reason, multiplier=1):
        self.log(
            f"Attempting contribution: {reason}. Cash before: ${self.broker.get_cash():.2f}, Dividend cash: ${self.total_dividend_cash_received:.2f}"
        )

        # Record this as an investment that requires external deposit
        current_date = self.datetime.date(0)
        this_time_invest_amout = self.p.cash_per_contribution * multiplier
        
        # Notify analyzer about the investment requirement
        for analyzer in self.analyzers:
            if hasattr(analyzer, "record_investment"):
                analyzer.record_investment(
                    current_date,
                    this_time_invest_amout,
                    reason
                )

        target_investment_per_ticker = this_time_invest_amout / len(
            self.p.tickers
        )

        cash_needed_for_full_contribution = this_time_invest_amout
        available_cash = self.broker.get_cash()

        if (
            available_cash < cash_needed_for_full_contribution * 0.1
        ):  # 如果现金连10%都买不起，就不买了
            self.log(
                f"Insufficient cash ({available_cash:.2f}) for significant contribution ({cash_needed_for_full_contribution:.2f}). Skipping buy for '{reason}'."
            )
            return

        actual_total_spent = 0
        for data in self.datas:
            if data._name not in self.p.tickers:  # 确保只购买参数中指定的tickers
                continue

            price = data.close[0]
            if price <= 0:
                self.log(f"Skipping buy for {data._name} due to invalid price: {price}")
                continue

            size_to_buy = int(target_investment_per_ticker / price)

            if (
                size_to_buy * price > available_cash - actual_total_spent
            ):  # 确保单笔购买不超过剩余现金
                size_to_buy = int((available_cash - actual_total_spent) / price)

            if size_to_buy > 0:
                order = self.buy(data=data, size=size_to_buy)
                if order:
                    order.reason = reason
                    self.log(
                        f"BUY Order Created for {data._name}: Size {size_to_buy} @ ~{price:.2f} for '{reason}'"
                    )
                    actual_total_spent += (
                        size_to_buy * price
                    )  # 估算花费，实际以执行为准
                else:
                    self.log(f"Could not create BUY order for {data._name}")
            else:
                self.log(
                    f"Calculated size for {data._name} is 0 or less. Skipping buy."
                )

        if actual_total_spent > 0:
            self.state.invested_amount += (
                self.p.cash_per_contribution
            )  # 记录计划投入的金额
        self.log(
            f"Contribution '{reason}' completed. Estimated spend: {actual_total_spent:.2f}. Cash after (approx): {available_cash - actual_total_spent:.2f}"
        )

    def rebalance(self):
        self.log(
            f"Attempting annual rebalance. Portfolio value: {self.broker.getvalue():.2f}"
        )
        active_datas = [d for d in self.datas if d._name in self.p.tickers]

        if not active_datas:
            self.log(
                "No active data feeds found for rebalancing based on self.p.tickers."
            )
            return

        # Calculate total stock value (excluding cash)
        total_stock_value = sum(self.getposition(data).size * data.close[0] for data in active_datas)
        target_value_per_ticker = total_stock_value / len(active_datas)

        for data in active_datas:
            current_value = self.getposition(data).size * data.close[0]
            if current_value < target_value_per_ticker:
                # Need to buy more
                additional_value = target_value_per_ticker - current_value
                size_to_buy = int(additional_value / data.close[0])
                if size_to_buy > 0:
                    order = self.buy(data=data, size=size_to_buy)
            if order:
                order.reason = "annual_rebalance"
                self.log(
                            f"Rebalance BUY Order: {size_to_buy} shares of {data._name} to reach target value of ${target_value_per_ticker:.2f}"
                        )
            elif current_value > target_value_per_ticker:
                # Need to sell some
                excess_value = current_value - target_value_per_ticker
                size_to_sell = int(excess_value / data.close[0])
                if size_to_sell > 0:
                    order = self.sell(data=data, size=size_to_sell)
                    if order:
                        order.reason = "annual_rebalance"
                        self.log(
                            f"Rebalance SELL Order: {size_to_sell} shares of {data._name} to reach target value of ${target_value_per_ticker:.2f}"
                        )

        self.state.rebalanced_this_year_august = True

    def is_third_week_end(self):
        dt_obj = self.datetime.date(0)  # 当前K线日期
        day_of_month = dt_obj.day

        if not (15 <= day_of_month <= 21):  # 检查是否在第三周（15号到21号）
            return False

        # 检查是否是当月的最后一个交易日（近似）
        try:
            next_day_dt_obj = self.datetime.date(1)  # 下一个交易日的日期
            # 如果下一天是22号或之后，则当前天是第三周的"最后交易日"的一种情况
            if 22 <= next_day_dt_obj.day <= 28:
                return True
            # 如果下一天的月份变了，说明今天是本月最后一个交易日
            if next_day_dt_obj.month != dt_obj.month:
                return True
        except IndexError:  # 如果没有下一个bar，说明是数据末尾
            return True
        return False

    def next(self):
        current_date = self.datetime.date(0)

        # 0. Check and inject cash if needed (backtest mode only)
        self._check_and_inject_cash()

        # 1. 检查股息 (通常在所有逻辑之前处理现金变动)
        self._check_dividends()

        # 2. 确保有足够的历史数据
        if len(self.datas[0]) < self.p.min_period or len(self.rsp_data) < self.p.min_period:
                return

        # 3. 月度状态刷新
        if self.state.current_month is None or self.state.current_month != current_date.month:
            self.state.refresh_to_new_month(self.rsp_data)
            self.log(
                f"Month refreshed using RSP. New month: {self.state.current_month}. RSP Month Start Price: {self.state.rsp_month_start_price}"
            )

        # 4. 获取RSP价格
        rsp_prev_close = None
        rsp_today_close = None
        if len(self.rsp_data) >= 2:
            rsp_prev_close = self.rsp_data.close[-1]
            rsp_today_close = self.rsp_data.close[0]

        # 5. 执行定投逻辑 (依赖RSP的逻辑)
        if rsp_today_close is not None:
            # 5.1. 第一次定投: RSP日跌幅 ≤ -1% 或 第三周末尾
            if not self.state.first_exec:
                if rsp_prev_close is not None and rsp_prev_close > 0:
                    drawback = (rsp_today_close / rsp_prev_close) - 1
                    if drawback <= -0.01:
                        self.buy_etfs("RSP daily drop <= -1%")
                        self.state.first_exec = True
                    elif self.is_third_week_end():
                        self.buy_etfs("Fallback on 3rd week end")
                        self.state.first_exec = True
                elif self.is_third_week_end():
                    self.buy_etfs("Fallback on 3rd week end (no prev_close)")
                    self.state.first_exec = True

            # 5.2. 第二次定投: RSP月内跌幅 ≤ -5%
            if not self.state.second_exec:
                if (
                    self.state.rsp_month_start_price is not None
                    and self.state.rsp_month_start_price > 0
                ):
                    mtd_ret = (rsp_today_close / self.state.rsp_month_start_price) - 1
                    if mtd_ret <= -0.05:
                        self.buy_etfs("RSP MTD drop <= -5%", multiplier=3)
                        self.state.second_exec = True

        # 6. 第三次定投: 极端事件 (TODO)
        if not self.state.third_exec:
            # Human extreme event: market breadth < 0.15
            mb_df = getattr(self.state, 'market_breadth_df', None)
            if mb_df is not None:
                mb_row = mb_df.loc[mb_df.index == current_date]
                if not mb_row.empty:
                    mb_value = mb_row.iloc[0]['market_breadth']
                    if mb_value < 0.15:
                        self.buy_etfs("Human Extreme (Breadth < 0.15)", multiplier=3)
                        self.state.third_exec = True

        # 7. 年度再平衡: 8月第一个交易日
        if current_date.month == 8 and not self.state.rebalanced_this_year_august:
            if self.datetime.date(-1).month == 7:
                self.rebalance()

        # 8. 保存状态
        self.state.save_state()

    def stop(self):
        self.log(
            f"Strategy stopped. Final portfolio value: {self.broker.getvalue():.2f}"
        )
        self.log(
            f"Final State: Current Month: {self.state.current_month}, First Exec: {self.state.first_exec}, Second Exec: {self.state.second_exec}, RSP Month Start: {self.state.rsp_month_start_price}"
        )
        self.log(
            f"Total Gross Dividends: {self.state.total_dividends_gross:.2f}, Total Tax on Dividends: {self.state.total_tax_paid_on_dividends:.2f}"
        )
        self.log(
            f"Total Net Dividend Cash Received: ${self.total_dividend_cash_received:.2f}"
        )
        self.log(
            f"Total Planned Investment Contributions: {self.state.invested_amount:.2f}"
        )  # 这是计划投入的，不一定是实际成交
        
        # Report cash injection statistics
        if self.p.enable_cash_injection:
            self.log(f"Total Cash Injected (Backtest): ${self.total_cash_injected:.2f}")
            self.log(f"Number of Cash Injections: {len(self.cash_injection_history)}")
            net_external_cash = self.total_cash_injected - self.total_dividend_cash_received
            self.log(f"Net External Cash (Injected - Dividends): ${net_external_cash:.2f}")
        
        self.state.save_state()