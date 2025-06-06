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
        self.restore_state()

    def _initialize_defaults(self):
        """Helper to set default values for attributes if not loaded."""
        self.current_month = getattr(self, "current_month", None)
        self.first_exec = getattr(self, "first_exec", False)
        self.second_exec = getattr(self, "second_exec", False)
        self.third_exec = getattr(self, "third_exec", False)
        self.rebalanced_this_year_august = getattr(self, "rebalanced_this_year_august", False)

    def restore_state(self):
        self.storage_engine.load()
        self.current_month = self.storage_engine.get("current_month")
        self.first_exec = self.storage_engine.get("first_exec", False)
        self.second_exec = self.storage_engine.get("second_exec", False)
        self.third_exec = self.storage_engine.get("third_exec", False)
        self.rebalanced_this_year_august = self.storage_engine.get("rebalanced_this_year_august", False)
        self._initialize_defaults()

    def save_state(self):
        self.storage_engine.set("current_month", self.current_month)
        self.storage_engine.set("first_exec", self.first_exec)
        self.storage_engine.set("second_exec", self.second_exec)
        self.storage_engine.set("third_exec", self.third_exec)
        self.storage_engine.set("rebalanced_this_year_august", self.rebalanced_this_year_august)
        self.storage_engine.save()

    def refresh_to_new_month(self, rsp_data_feed):
        current_date = rsp_data_feed.datetime.date(0)
        self.current_month = current_date.month
        self.first_exec = False
        self.second_exec = False
        self.third_exec = False

        if self.current_month == 1:
            self.rebalanced_this_year_august = False

        self.save_state()


class Hi5Strategy(bt.Strategy):
    params = (
        ('tickers', []),
        ('enable_cash_injection', True),
        ('cash_injection_threshold', 5),
        ('cash_per_contribution', 10000),
        ('non_resident_tax_rate', 0.3),
        ('market_breadth_df', None),
    )

    def __init__(self):
        super().__init__()
        self.state = Hi5State(storage_engine=InMemoryStorageEngine())
        self.data_feeds = {data._name: data for data in self.datas}
        self.benchmark_ticker = "RSP"
        self.tickers = self.p.tickers
        
        if self.p.market_breadth_df is not None:
            self.market_breadth = MarketBreadthIndicator(
                self.data_feeds[self.benchmark_ticker],
                market_breadth_df=self.p.market_breadth_df
            )

    def next(self):
        current_date = self.data.datetime.date(0)
        
        # Check for August 1st rebalancing
        if current_date.month == 8 and current_date.day == 1:
            self._rebalance_portfolio("Annual rebalancing on August 1st")
        
        if self.state.current_month is None or self.state.current_month != current_date.month:
            self.state.current_month = current_date.month
            self.state.first_exec = False
            self.state.second_exec = False
            self.state.third_exec = False
            self.state.save_state()

        rsp_data = self.data_feeds[self.benchmark_ticker]
        rsp_price = rsp_data.close[0]
        rsp_prev_close = rsp_data.close[-1]

        if not self.state.first_exec:
            daily_return = (rsp_price / rsp_prev_close - 1) if rsp_prev_close else 0
            if daily_return <= -0.01:
                reason = f"RSP daily drop <= -1%: Current ${rsp_price:.2f}, Daily change {daily_return*100:+.2f}%"
                self._execute_trades(reason)
                self.state.first_exec = True
            elif self._is_third_week_end(current_date):
                self._execute_trades("Third week end")
                self.state.first_exec = True

        if not self.state.second_exec:
            month_start = current_date.replace(day=1)
            month_start_price = None
            for i in range(len(rsp_data)):
                if rsp_data.datetime.date(i) >= month_start:
                    month_start_price = rsp_data.close[i]
                    break
            
            if month_start_price:
                mtd_return = (rsp_price / month_start_price) - 1
                if mtd_return <= -0.05:
                    reason = f"RSP MTD drop <= -5%: Month start ${month_start_price:.2f}, Current ${rsp_price:.2f}, MTD change {mtd_return*100:+.2f}%"
                    self._execute_trades(reason, multiplier=3)
                    self.state.second_exec = True

        if not self.state.third_exec and hasattr(self, 'market_breadth'):
            current_breadth = self.market_breadth.lines.ema20_ratio[0]
            market_breadth = self.market_breadth.lines.market_breadth[0]
            if current_breadth <= 0.2 and market_breadth < 0:
                reason = f"Human extreme condition triggered: EMA20 ratio = {current_breadth:.2%}, Market Breadth = {market_breadth:.2%}"
                self._execute_trades(reason, multiplier=5)
                self.state.third_exec = True

        if any([self.state.first_exec, self.state.second_exec, self.state.third_exec]):
            self.state.save_state()

    def _rebalance_portfolio(self, reason):
        """Rebalance portfolio to equal weights"""
        # Calculate total portfolio value
        total_value = self.broker.getvalue()
        target_per_position = total_value / len(self.tickers)
        
        # First, sell all existing positions
        for ticker in self.tickers:
            data = self.data_feeds[ticker]
            position = self.getposition(data)
            if position.size > 0:
                self.close(data=data, reason=f"Rebalancing: {reason}")
        
        # Then, buy new positions with equal weights
        for ticker in self.tickers:
            data = self.data_feeds[ticker]
            price = data.close[0]
            if price and price > 0:
                shares = int(target_per_position / price)
                if shares > 0:
                    self.buy(data=data, size=shares, reason=f"Rebalancing: {reason}")

    def _execute_trades(self, reason, multiplier=1):
        total_investment = self.p.cash_per_contribution * multiplier
        investment_per_ticker = total_investment / len(self.tickers)

        if hasattr(self, 'analyzers') and self.analyzers.hi5analyzer:
            self.analyzers.hi5analyzer.record_investment(
                self.data.datetime.date(0),
                total_investment,
                reason
            )

        for ticker in self.tickers:
            data = self.data_feeds[ticker]
            price = data.close[0]
            
            if price and price > 0:
                shares = int(investment_per_ticker / price)
                if shares > 0:
                    order = self.buy(data=data, size=shares)
                    order.reason = reason

    def _is_third_week_end(self, current_date):
        day = current_date.day
        if 15 <= day <= 21 and current_date.weekday() in [3, 4]:
            return True
        return False


class MarketBreadthIndicator(bt.Indicator):
    lines = ('ema20_ratio', 'market_breadth')
    params = (('market_breadth_df', None),)

    def __init__(self):
        super(MarketBreadthIndicator, self).__init__()
        self.market_breadth_df = self.p.market_breadth_df
        if self.market_breadth_df is None:
            raise ValueError("market_breadth_df must be provided")

    def next(self):
        current_date = self.data.datetime.date(0)
        try:
            self.lines.ema20_ratio[0] = self.market_breadth_df.loc[current_date, 'ema20_ratio']
            self.lines.market_breadth[0] = self.market_breadth_df.loc[current_date, 'market_breadth']
        except KeyError:
            self.lines.ema20_ratio[0] = 0.0
            self.lines.market_breadth[0] = 0.0