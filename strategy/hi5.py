import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PandasDataWithDividends(bt.feeds.PandasData):
    lines = ('dividends',)
    params = (
        ('dividends', 'dividends'),
    )


class IBKRUSFixedCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.005),     # $0.005 per share
        ('min_commission', 1.00),  # $1.00 minimum
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )

    def getcommission(self, size, price, pseudoexec):
        cost = abs(size) * self.p.commission
        return max(cost, self.p.min_commission)
    
class Hi5State:
    def __init__(self):
        self.current_month = None
        self.current_month_cash_flow = 0
        self.first_exec = False
        self.second_exec = False
        self.third_exec = False
        self.month_start_price = None
        
class Hi5Analyzer(bt.Analyzer):
    def __init__(self):
        

        
        

class Hi5Strategy(bt.Strategy):
    params = (
        ('ticker', ('IWY', 'RSP', 'MOAT', 'PFF', 'VNQ')),
        ('initial_cash', 100000),
        ('cash_per_contribution', 10000),
        ('start_date', '2024-01-01'),
        ('end_date', '2024-12-31'),
        ('non_resident_tax_rate', 0.30),
    )

    def __init__(self):
        self.dividends = {}
        self.cash_per_contribution = self.params.cash_per_contribution
        self.start_date = self.params.start_date
        self.end_date = self.params.end_date

        # used for tracking orders and cashflow
        self.orders = {}

        # used for backtesting
        self.monthly_cash_flows = {}
        self.monthly_portfolio_values = {}
        self.monthly_dates = []

               # Track dividends and taxes
        self.dividend_history = []  # List to store dividend payments
        self.total_dividends = 0
        self.total_tax_paid = 0
        self.non_resident_tax_rate = self.params.non_resident_tax_rate

        self.broker.setcash(self.params.initial_cash)
        self.broker.setcommission(IBKRUSFixedCommission())

    def log_cash_flow(self, date, cash_flow):
        if date not in self.monthly_cash_flows:
            self.monthly_cash_flows[date] = 0
        self.monthly_cash_flows[date] += cash_flow

    def log_portfolio_value(self, date, value):
        if date not in self.monthly_portfolio_values:
            self.monthly_portfolio_values[date] = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.strftime('%Y-%m-%d')} {txt}")

    def is_third_week_end(self):
        # Determine the last valid trading day (Mon-Fri) of the 3rd calendar week
        dt = self.datas[0].datetime.date(0)
        try:
            next_dt = self.datas[0].datetime.date(1)
        except IndexError:
            next_dt = None

        # if next_dt is in 4th week and dt is in 3rd week, then it is the last trading day of the 3rd week
        if (next_dt is None or 22 <= next_dt.day <= 28) and 15 <= dt.day <= 21:
            return True
        return False
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            # Record order details
            order_info = {
                'date': self.datas[0].datetime.date(0),
                'ticker': order.data._name,
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'reason': getattr(order, 'reason', 'N/A')  # Get the reason if available
            }
            self.order_history.append(order_info)
            
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, {order.data._name}, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                    f'Size: {order.executed.size}'
                )
            else:
                self.log(
                    f'SELL EXECUTED, {order.data._name}, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                    f'Size: {order.executed.size}'
                )
        elif order.status in [order.Canceled, order.Rejected]:
            self.log(f'Order Canceled/Rejected: {order.data._name}, reason: {str(order.info)}')
        elif order.status in [order.Margin]:
            self.log(f'Order Margin not enough {order.data._name}')

        if order.ref in self.orders:
            del self.orders[order.ref]

    def check_human_extreme(self):
        # Placeholder: integrate real 60-day SPX breadth data
        return False
