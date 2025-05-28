import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf

from data.data_provider import BacktraderDataProvider

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
        self.monthly_cash_flows = []
        self.monthly_portfolio_values = []
        self.monthly_dates = []
        self.dividend_history = []
        self.total_dividends = 0
        self.total_tax_paid = 0
        self.order_history = []
        self.non_resident_tax_rate = 0.30

    def add_monthly_data(self, date, portfolio_value, cash_flow):
        self.monthly_portfolio_values.append(portfolio_value)
        self.monthly_dates.append(date)
        self.monthly_cash_flows.append(cash_flow)

    def add_dividend(self, date, ticker, amount, tax, net):
        self.dividend_history.append({
            'date': date,
            'ticker': ticker,
            'amount': amount,
            'tax': tax,
            'net': net
        })
        self.total_dividends += amount
        self.total_tax_paid += tax

    def add_order(self, order_info):
        self.order_history.append(order_info)

    def calculate_metrics(self):
        total_invested = -sum(self.monthly_cash_flows)
        final_value = self.monthly_portfolio_values[-1]
        total_return = (final_value - total_invested) / total_invested
        
        try:
            cash_flows = self.monthly_cash_flows.copy()
            cash_flows.append(final_value)
            irr = npf.irr(cash_flows)
            annual_irr = (1 + irr) ** 12 - 1
        except:
            irr = None
            annual_irr = None
        
        portfolio_values = np.array(self.monthly_portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        return {
            'total_invested': total_invested,
            'final_value': final_value,
            'total_return': total_return,
            'annual_irr': annual_irr,
            'max_drawdown': max_drawdown
        }

class Hi5Strategy(bt.Strategy):
    params = (
        ('ticker', ('IWY', 'RSP', 'MOAT', 'PFF', 'VNQ')),
        ('initial_cash', 100000),
        ('cash_per_contribution', 10000),
        ('non_resident_tax_rate', 0.30),
        ('start_date', '2024-01-01'),
        ('end_date', '2024-12-31'),
    )

    def __init__(self):
        self.broker.setcash(self.p.initial_cash)
        self.broker.setcommission(IBKRUSFixedCommission())

        self.state = Hi5State()
        self.data_provider = BacktraderDataProvider()
        self.analyzer = Hi5Analyzer()

        for ticker in self.p.tickers:
            self.data_provider.get_data(ticker, self.p.start_date, self.p.end_date)

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

    def _log_completed_order(self, order):
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
            self.state.add_order(order_info)
            self._log_completed_order(order)
        elif order.status in [order.Canceled, order.Rejected]:
            self.log(f'Order Canceled/Rejected: {order.data._name}, reason: {str(order.info)}')
        elif order.status in [order.Margin]:
            self.log(f'Order Margin not enough {order.data._name}')

    def make_decision(self, reason = "N/A"):
        alloc = self.p.cash_per_contribution / len(self.p.tickers)

        for _, data in self.datas_by_name.items():
            order = self.buy(data, int(alloc / data.close[0]))
            if order:
                order.reason = reason
    
    def rebalance(self):
        for _, data in self.datas_by_name.items():
            order = self.order_target_percent(data, 1 / len(self.p.tickers))
            if order:
                order.reason = "Rebalance"
    
    def next(self):
        if len(self) < self.p.min_period:
            return
    
        
        
