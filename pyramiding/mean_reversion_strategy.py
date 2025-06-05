import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np # For potential NaN handling if needed, though yf.download usually handles this
import datetime
import pytz # For timezone handling if you extend to live trading or more complex date logic
from dataclasses import dataclass, field # Added field for default_factory
from typing import List, Dict, Optional # For type hinting, added Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for interactive plotting
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

@dataclass
class TradingSignal:
    """Represents a trading signal"""
    date: datetime.datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    price: float
    reason: str
    zscore: float
    ma: float
    std: float

# Custom ZScore Indicator
class ZScoreIndicator(bt.Indicator):
    lines = ('zscore',)
    params = (('period', 20),)  # Default period, can be linked to strategy's lookback

    def __init__(self):
        # Calculate Z-Score using the input data
        self.lines.zscore = (self.data - self.data1) / bt.indicators.Max(self.data2, 0.0000001)


class MeanReversionStrategy(bt.Strategy):
    """
    Mean Reversion Strategy using Z-Score
    - Calculates moving average and standard deviation for each data feed
    - Generates signals when price deviates significantly from mean
    - Uses Z-Score to determine entry/exit points
    """
    
    params = (
        ('lookback', 20),
        ('zscore_threshold', 2.0),
        ('risk_free_rate', 0.04),  # Used by SharpeRatio analyzer
        ('log_signals', True),  # New parameter to control signal logging
    )

    def __init__(self):
        # Dictionaries to hold indicators, orders, and positions per data feed (ticker)
        self.inds = {}
        self.orders_pending = {}  # Tracks pending orders for each ticker
        self.current_positions = {}  # Tracks current position size for each ticker
        self.signals: List[TradingSignal] = []
        
        for i, d in enumerate(self.datas):
            ticker_name = d._name  # Get the name of the data feed (ticker)
            self.inds[ticker_name] = {}  # Create a sub-dictionary for this ticker's indicators
            
            # Calculate moving average
            self.inds[ticker_name]['ma'] = bt.indicators.SMA(d.close, period=self.p.lookback)
            # Calculate standard deviation
            self.inds[ticker_name]['std'] = bt.indicators.StandardDeviation(d.close, period=self.p.lookback)
            
            # Calculate Z-Score using the custom indicator
            self.inds[ticker_name]['zscore'] = ZScoreIndicator(
                d.close,  # Price data
                self.inds[ticker_name]['ma'],  # Moving average
                self.inds[ticker_name]['std'],  # Standard deviation
                period=self.p.lookback
            )

            self.orders_pending[ticker_name] = None
            self.current_positions[ticker_name] = 0
            
            # Configure plot settings
            d.plotinfo.plot = True
            self.inds[ticker_name]['ma'].plotinfo.plot = True
            self.inds[ticker_name]['ma'].plotinfo.subplot = False
            self.inds[ticker_name]['zscore'].plotinfo.plot = True
            self.inds[ticker_name]['zscore'].plotinfo.subplot = True
            self.inds[ticker_name]['zscore'].plotinfo.plotname = f'Z-Score ({ticker_name})'
            self.inds[ticker_name]['zscore'].plotinfo.plotyhlines = [-self.p.zscore_threshold, self.p.zscore_threshold]


    def log(self, txt, dt=None, doprint=False):
        if doprint: 
            dt = dt or self.datas[0].datetime.date(0) 
            print(f'{dt.isoformat()} {txt}')

    def next(self):
        current_date = self.datas[0].datetime.datetime(0) 

        for i, d in enumerate(self.datas):
            ticker_name = d._name
            
            # Ensure all parts of the zscore indicator have enough length
            if len(self.inds[ticker_name]['ma']) < self.p.lookback or \
               len(self.inds[ticker_name]['std']) < self.p.lookback or \
               len(self.inds[ticker_name]['zscore'].lines.zscore) == 0: 
                continue
                
            current_zscore = self.inds[ticker_name]['zscore'].lines.zscore[0] 
            current_price = d.close[0]
            current_ma = self.inds[ticker_name]['ma'][0]
            current_std = self.inds[ticker_name]['std'][0]

            if current_std < 0.00001: 
                continue

            if current_zscore <= -self.p.zscore_threshold and self.current_positions[ticker_name] <= 0:
                if self.orders_pending[ticker_name] is None: 
                    size = self.calculate_position_size(d)
                    if size > 0:
                        self.log(f'BUY CREATE ({ticker_name}): Price {current_price:.2f}, Size {size}, Z-Score {current_zscore:.2f}', doprint=self.p.log_signals)
                        self.orders_pending[ticker_name] = self.buy(data=d, size=size)
                        self.signals.append(TradingSignal(
                            date=current_date, 
                            ticker=ticker_name, action='BUY', price=current_price,
                            reason=f'Z-Score oversold: {current_zscore:.2f}',
                            zscore=current_zscore, ma=current_ma, std=current_std ))
                        
            elif current_zscore >= self.p.zscore_threshold and self.current_positions[ticker_name] > 0: # Only sell if long
                if self.orders_pending[ticker_name] is None:
                    size_to_trade = self.current_positions[ticker_name]
                    self.log(f'SELL CREATE (Close Long) ({ticker_name}): Price {current_price:.2f}, Size {size_to_trade}, Z-Score {current_zscore:.2f}', doprint=self.p.log_signals)
                    self.orders_pending[ticker_name] = self.sell(data=d, size=size_to_trade)
                    self.signals.append(TradingSignal(
                        date=current_date, ticker=ticker_name, action='SELL (Close Long)', price=current_price,
                        reason=f'Z-Score overbought: {current_zscore:.2f}',
                        zscore=current_zscore, ma=current_ma, std=current_std ))
            
            elif abs(current_zscore) < 0.5 and self.current_positions[ticker_name] > 0 : # Only close if long
                 if self.orders_pending[ticker_name] is None:
                    self.log(f'CLOSE LONG (Mean Revert) ({ticker_name}): Price {current_price:.2f}, Z-Score {current_zscore:.2f}', doprint=self.p.log_signals)
                    self.orders_pending[ticker_name] = self.close(data=d, size=self.current_positions[ticker_name]) 
                    self.signals.append(TradingSignal(
                        date=current_date, ticker=ticker_name, action='CLOSE LONG (Reversion)',
                        price=current_price, reason=f'Z-Score reverted: {current_zscore:.2f}',
                        zscore=current_zscore, ma=current_ma, std=current_std ))


    def calculate_position_size(self, data_feed):
        cash = self.broker.getcash()
        price = data_feed.close[0]
        if price == 0: return 0
        target_value = cash * 0.05 
        size = int(target_value / price)
        return size

    def notify_order(self, order):
        ticker_name = order.data._name
        if order.status in [order.Submitted, order.Accepted]:
            return 
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED ({ticker_name}): Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, Size: {order.executed.size}', doprint=self.p.log_signals)
                self.current_positions[ticker_name] += order.executed.size
            else:  # Sell
                self.log(
                    f'SELL EXECUTED ({ticker_name}): Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, Size: {order.executed.size}', doprint=self.p.log_signals)
                # For sells (closing longs or opening shorts), executed.size is negative
                self.current_positions[ticker_name] += order.executed.size 
                if self.current_positions[ticker_name] < 0.000001 and self.current_positions[ticker_name] > -0.000001: # Float comparison
                    self.current_positions[ticker_name] = 0 # Clean up potential float inaccuracies
            # self.bar_executed = len(self) # This variable is not used elsewhere, consider removing or using.
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'ORDER {order.getStatusName()} ({ticker_name}): Ref {order.ref}', doprint=True)
        self.orders_pending[ticker_name] = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        ticker_name = trade.data._name
        self.log(f'TRADE PROFIT ({ticker_name}): Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}', doprint=True)

    def stop(self):
        self.log(f'(Lookback {self.p.lookback}, Z-Threshold {self.p.zscore_threshold}) Ending Value {self.broker.getvalue():.2f}', doprint=True)
        if self.p.log_signals and self.signals: # Check if signals list is not empty
            print("\n--- All Generated Trading Signals ---")
            for signal in self.signals:
                print(f"{signal.date.strftime('%Y-%m-%d %H:%M:%S')} - {signal.ticker}: {signal.action} @ ${signal.price:.2f} "
                      f"(Z: {signal.zscore:.2f}, MA: {signal.ma:.2f}, Std: {signal.std:.2f}) - {signal.reason}")
        elif self.p.log_signals:
            print("\nNo trading signals were generated during this backtest.")


class EmailNotifier:
    """Handles email notifications for trading signals"""
    
    def __init__(self, smtp_config_path='smtp_config.json'):
        self.config = self.load_smtp_config(smtp_config_path)
        if not self.config and smtp_config_path == 'smtp_config.json': 
             script_dir = os.path.dirname(os.path.abspath(__file__))
             alt_config_path = os.path.join(script_dir, smtp_config_path)
             if os.path.exists(alt_config_path): # Check if alternative path exists
                self.config = self.load_smtp_config(alt_config_path)
             else:
                print(f"SMTP config file not found at default or alternative path: {alt_config_path}")

    def load_smtp_config(self, config_path):
        try:
            if not os.path.exists(config_path):
                print(f"SMTP config file not found at: {config_path}")
                return None
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading SMTP config from {config_path}: {e}")
            return None

    def send_signal_email(self, signals: List[TradingSignal], recipient_email: str, strategy_params: dict):
        if not self.config:
            print("SMTP configuration not loaded. Cannot send email.")
            return False
        if not signals:
            print("No signals to send for email.")
            return False
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f'Trading Signals - {datetime.datetime.now().strftime("%Y-%m-%d")}'
            msg['From'] = self.config['username']
            msg['To'] = recipient_email
            html = f"""
            <html><head><style>
                table {{ font-family: Arial, sans-serif; border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                .buy {{ color: green; font-weight: bold; }} 
                .sell {{ color: red; font-weight: bold; }} 
                .close {{ color: orange; font-weight: bold; }}
            </style></head><body>
                <h2>Trading Signals Summary</h2>
                <p><strong>Report Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr> <h3>Signals ({len(signals)}):</h3>
                <table><tr><th>Date</th><th>Ticker</th><th>Action</th><th>Price</th><th>Z-Score</th><th>MA</th><th>Std</th><th>Reason</th></tr>"""
            for signal in signals:
                action_class = ""
                if "BUY" in signal.action.upper(): action_class = "buy"
                elif "SELL" in signal.action.upper(): action_class = "sell"
                elif "CLOSE" in signal.action.upper(): action_class = "close"
                html += f"""
                        <tr>
                            <td>{signal.date.strftime('%Y-%m-%d %H:%M:%S')}</td>
                            <td>{signal.ticker}</td>
                            <td class="{action_class}">{signal.action}</td>
                            <td>${signal.price:.2f}</td>
                            <td>{signal.zscore:.2f}</td>
                            <td>${signal.ma:.2f}</td>
                            <td>${signal.std:.2f}</td>
                            <td>{signal.reason}</td>
                        </tr>"""
            html += "</table><hr>"
            html += "<p><strong>Strategy Parameters Used:</strong></p><ul>"
            for k, v in strategy_params.items():
                html += f"<li>{k.replace('_', ' ').title()}: {v}</li>"
            html += "</ul>"
            html += '<p style="color: #888888; font-size: 0.9em;"><strong>Disclaimer:</strong> This is an automated signal for informational purposes only. Please verify all information before making any trading decisions.</p>'
            html += "</body></html>"
            msg.attach(MIMEText(html, 'html'))
            with smtplib.SMTP(self.config['server'], self.config['port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            print(f"Signal email sent to {recipient_email}")
            return True
        except Exception as e:
            print(f"Error sending signal email: {e}")
            return False

def get_sp500_symbols():
    """Fetches S&P 500 symbols from Wikipedia. More robust sources are recommended for production."""
    try:
        # Using a more specific table index if Wikipedia page structure changes
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df_sp500 = payload[0] # Usually the first table
        symbols = df_sp500['Symbol'].tolist()
        # Clean symbols: some symbols from Wikipedia might have issues (e.g. 'BRK.B' vs 'BRK-B')
        # yfinance generally prefers '-' for sub-classes if '.' causes issues.
        symbols = [s.replace('.', '-') if '.' in s else s for s in symbols]
        # Further cleaning for symbols like 'BF.B' which yfinance might expect as 'BF-B'
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}. Using a default small list.")
        return ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'] 

def run_backtest(tickers: List[str] = ['GOOG'], 
                 start_date_str: Optional[str] = None, 
                 end_date_str: Optional[str] = None, 
                 initial_cash: float = 100000.0,
                 lookback_period: int = 20,
                 zscore_thresh: float = 2.0,
                 log_signals_to_console: bool = True,
                 email_recipient: Optional[str] = None): 
    
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else datetime.datetime.now() - datetime.timedelta(days=3*365) 
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else datetime.datetime.now()

    cerebro = bt.Cerebro(stdstats=True) 
    
    strategy_params_dict = dict(
        lookback=lookback_period,
        zscore_threshold=zscore_thresh,
        log_signals=log_signals_to_console
    )
    cerebro.addstrategy(MeanReversionStrategy, **strategy_params_dict)
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001) 

    print(f"\nFetching data for {len(tickers)} tickers...")
    for ticker in tickers:
        print(f"Attempting to fetch data for {ticker}...")
        try:
            data_df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  Error downloading data for {ticker}: {e}. Skipping.")
            continue
            
        if data_df.empty:
            print(f"  Warning: No data for {ticker} in the given range. Skipping.")
            continue

        if isinstance(data_df.columns, pd.MultiIndex):
            print(f"  {ticker} data has MultiIndex columns. Flattening to first level.")
            data_df.columns = data_df.columns.get_level_values(0)
        
        data_df.columns = [col.capitalize() if isinstance(col, str) else col for col in data_df.columns]
        rename_map = {}
        for col in data_df.columns:
            if str(col).lower() == 'open': rename_map[col] = 'Open'
            elif str(col).lower() == 'high': rename_map[col] = 'High'
            elif str(col).lower() == 'low': rename_map[col] = 'Low'
            elif str(col).lower() == 'close': rename_map[col] = 'Close'
            elif str(col).lower() == 'volume': rename_map[col] = 'Volume'
        data_df.rename(columns=rename_map, inplace=True)

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(data_df.columns):
            print(f"  Warning: {ticker} missing one or more required columns ({required_cols - set(data_df.columns)}). Skipping.")
            continue
        
        if data_df.index.tz is not None:
            data_df.index = data_df.index.tz_localize(None)

        data_feed = bt.feeds.PandasData(
            dataname=data_df,
            name=ticker,
            datetime=None, 
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=None 
        )
        cerebro.adddata(data_feed)
        print(f"  Successfully added data for {ticker}")
    
    if not cerebro.datas:
        print("No data feeds were successfully added to Cerebro. Exiting backtest.")
        return None

    # Add analyzers with fixed risk-free rate
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', riskfreerate=0.04, timeframe=bt.TimeFrame.Days, compression=252)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    print(f"\nStarting backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} with {len(cerebro.datas)} data feeds.")
    results = cerebro.run()
    strategy_instance = results[0]
    
    final_value = cerebro.broker.getvalue()
    print(f"\n--- Backtest Results ---")
    print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    total_return_pct = ((final_value - initial_cash) / initial_cash * 100) if initial_cash != 0 else 0
    print(f"Total Return: {total_return_pct:,.2f}%")
    
    print("\n--- Performance Metrics ---")
    sharpe = strategy_instance.analyzers.sharpe_ratio.get_analysis()
    annual_ret = strategy_instance.analyzers.annual_return.get_analysis()
    drawdown = strategy_instance.analyzers.drawdown.get_analysis()
    trades = strategy_instance.analyzers.trades.get_analysis()
    sqn = strategy_instance.analyzers.sqn.get_analysis()

    print(f"Annualized Sharpe Ratio: {sharpe.get('sharperatio', 'N/A') if sharpe else 'N/A':.2f}")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%" if drawdown and drawdown.max.drawdown is not None else "Max Drawdown: N/A")
    avg_annual_return = np.mean([ret for ret in annual_ret.values() if ret is not None]) * 100 if annual_ret and any(annual_ret.values()) else 'N/A'
    print(f"Average Annual Return: {avg_annual_return if isinstance(avg_annual_return, str) else f'{avg_annual_return:.2f}%'}")
    print(f"System Quality Number (SQN): {sqn.get('sqn', 'N/A') if sqn else 'N/A':.2f}")
    
    if trades and trades.get('total', {}).get('total', 0) > 0:
        print("\n--- Trade Statistics ---")
        total_trades = trades.total.total
        won_trades = trades.won.total
        lost_trades = trades.lost.total
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades} ({(won_trades / total_trades * 100) if total_trades > 0 else 0:.2f}%)")
        print(f"Losing Trades: {lost_trades} ({(lost_trades / total_trades * 100) if total_trades > 0 else 0:.2f}%)")
        
        if trades.long.total > 0 : print(f"Long PnL: Total ${trades.long.pnl.total:.2f}, Average ${trades.long.pnl.average:.2f}")
        if trades.short.total > 0 : print(f"Short PnL: Total ${trades.short.pnl.total:.2f}, Average ${trades.short.pnl.average:.2f}") 
        
        profit_factor = 'Inf'
        if trades.lost.pnl.total != 0:
            profit_factor_val = trades.won.pnl.total / abs(trades.lost.pnl.total)
            profit_factor = f"{profit_factor_val:.2f}"
        print(f"Profit Factor: {profit_factor}")
    else:
        print("\nNo trades were executed.")
    
    if email_recipient and strategy_instance.signals:
        print(f"\nAttempting to send email to {email_recipient}...")
        notifier = EmailNotifier() 
        email_strat_params = {
            'Lookback Period': lookback_period,
            'Z-Score Threshold': zscore_thresh,
            'Initial Cash': f"${initial_cash:,.2f}",
            'Tickers': ", ".join(tickers) if tickers else "N/A",
            'Start Date': start_date.strftime('%Y-%m-%d'),
            'End Date': end_date.strftime('%Y-%m-%d')
        }
        notifier.send_signal_email(strategy_instance.signals, email_recipient, email_strat_params)
    elif email_recipient and not strategy_instance.signals:
        print(f"\nNo signals generated, so no email sent to {email_recipient}.")

    if len(cerebro.datas) > 0:
        print("\nGenerating plot...")
        try:            
            # Plot using cerebro
            cerebro.plot(
                style='candlestick',
                barup='green',
                bardown='red',
                volume=True,
                iplot=False,
                numfigs=len(cerebro.datas),
                plotdist=0.1,
                subtxtsize=7,
                valuetags=True,
                fmt_x_ticks='%Y-%m-%d',
                fmt_x_data='%Y-%m-%d',
                grid=True,
                plot_mode=1,
                barupfill=True,
                bardownfill=True,
                width=16,
                height=9,
                dpi=100
            )
            
        except Exception as e:
            print(f"Error during plotting: {e}")
            print("Available matplotlib styles:", plt.style.available)
            print("Available backends:", matplotlib.rcsetup.all_backends)
    else:
        print("Skipping plot generation as no data was loaded.")
        
    return strategy_instance

if __name__ == '__main__':
    # Test with a smaller set of tickers for faster execution
    #tickers_to_test = ['GOOG', 'AAPL']  # Reduced from 5 to 2 tickers
    tickers_to_test = ['GOOG']
    
    start_date_config = "2010-01-01"  # Shorter period for faster testing
    end_date_config = "2023-12-31"
    initial_cash_config = 100000.0
    lookback_config = 20
    zscore_threshold_config = 2.0
    log_signals_console_config = True
    email_recipient_config = None

    print("=== Starting Mean Reversion Backtest ===")
    strategy_run = run_backtest(
        tickers=tickers_to_test,
        start_date_str=start_date_config,
        end_date_str=end_date_config,
        initial_cash=initial_cash_config,
        lookback_period=lookback_config,
        zscore_thresh=zscore_threshold_config,
        log_signals_to_console=log_signals_console_config,
        email_recipient=email_recipient_config
    )
    print("\n=== Backtest Finished ===")
