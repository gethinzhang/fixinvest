import backtrader as bt
import os
import pandas as pd
import datetime
import numpy as np
from hi5.hi5 import (
    Hi5PandasDataWithDividends,
    Hi5Strategy,
    Hi5State,
    InMemoryStorageEngine,
)

import argparse
import json
from google.cloud import bigquery
from google.oauth2 import service_account

START_DATE = datetime.datetime(2012, 5, 1)
END_DATE = datetime.datetime.now()
INITIAL_CASH = 10000000.0
COMMISSION_RATE = 0.001
RISK_FREE_RATE = 0.04


class BacktestAnalyzer(bt.Analyzer):
    params = (
        ('risk_free_rate', 0.04),
    )

    def __init__(self):
        super().__init__()
        self.trade_history = []
        self.portfolio_values = []
        self.dates = []
        self.cash_values = []
        self.stock_values = []
        self.monthly_returns = []
        self.monthly_portfolio_values = []
        self.monthly_dates = []
        self.last_month = None
        self.investment_schedule = []
        self.total_investor_deposits = 0.0
        self.total_dividend_cash = 0.0
        self.final_positions = []

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
        investment_record = {
            "date": date,
            "amount": amount,
            "reason": reason
        }
        self.investment_schedule.append(investment_record)
        self.total_investor_deposits += amount

    def add_dividend_event(self, date, ticker, gross_amount, tax_amount, net_amount, shares_held):
        tax_rate = getattr(self.strategy, 'p', {}).get('non_resident_tax_rate', 0.3)
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
        self.record_investment(date, -net_dividend, f"Dividend from {ticker} Gross: ${gross_amount:.2f}, Tax: ${calculated_tax:.2f}")

    def next(self):
        if not hasattr(self, "strategy") or self.strategy is None:
            return

        current_date = self.strategy.datetime.date(0)
        cash_value = self.strategy.broker.get_cash() + self.total_dividend_cash
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
        
        portfolio_value = cash_value + stock_value
        self.portfolio_values.append(portfolio_value)
        self.cash_values.append(cash_value)
        self.stock_values.append(stock_value)
        self.dates.append(current_date)

        current_month = (current_date.year, current_date.month)
        if self.last_month is None:
            self.last_month = current_month
            self.monthly_portfolio_values.append(portfolio_value)
            self.monthly_dates.append(current_date)
        elif current_month != self.last_month:
            if len(self.monthly_portfolio_values) > 0:
                prev_value = self.monthly_portfolio_values[-1]
                if prev_value > 0:
                    monthly_return = (portfolio_value / prev_value) - 1
                    self.monthly_returns.append(monthly_return)
            self.monthly_portfolio_values.append(portfolio_value)
            self.monthly_dates.append(current_date)
            self.last_month = current_month
        else:
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

        monthly_irr, annual_irr = self.calculate_monthly_irr()
        if monthly_irr is None:
            monthly_irr = 0.0
        if annual_irr is None:
            annual_irr = 0.0

        sharpe_ratio = 0.0
        if len(self.monthly_returns) > 1:
            returns_np = np.array(self.monthly_returns)
            monthly_risk_free = (1 + self.p.risk_free_rate) ** (1/12) - 1
            excess_returns = returns_np - monthly_risk_free
            std_dev = np.std(excess_returns)
            if std_dev > 0:
                sharpe_ratio = np.sqrt(12) * (np.mean(excess_returns) / std_dev)

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
            "monthly_dates": self.monthly_dates[1:] if len(self.monthly_dates) > 1 else [],
        }

    def calculate_monthly_irr(self):
        if not self.portfolio_values or not self.dates:
            return 0.0, 0.0
            
        total_deposits = self.total_investor_deposits
        final_value = self.portfolio_values[-1]
        initial_cash = self.portfolio_values[0]
        net_profit = final_value - initial_cash
        
        if total_deposits > 0:
            total_return = (net_profit / total_deposits)
            start_date = self.dates[0]
            end_date = self.dates[-1]
            total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            
            if total_months > 0:
                monthly_irr = (1 + total_return) ** (1 / total_months) - 1
                annual_irr = (1 + monthly_irr) ** 12 - 1
                return monthly_irr, annual_irr
        
        return 0.0, 0.0

    def capture_final_positions(self, strategy):
        self.final_positions = []
        ticker_dividends = {}
        ticker_dividend_tax = {}
        
        for trade in self.trade_history:
            if trade['type'] == 'DIVIDEND':
                ticker = trade['ticker']
                if ticker not in ticker_dividends:
                    ticker_dividends[ticker] = 0
                    ticker_dividend_tax[ticker] = 0
                ticker_dividends[ticker] += trade['value']
                ticker_dividend_tax[ticker] += trade['commission']
        
        for data in strategy.datas:
            position = strategy.getposition(data)
            ticker = data._name
            
            if position.size > 0:
                current_price = data.close[0]
                market_value = position.size * current_price
                avg_cost = abs(position.price) if position.price != 0 else 0
                total_cost = position.size * avg_cost
                unrealized_pnl = market_value - total_cost
                unrealized_pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
                
                gross_dividends = ticker_dividends.get(ticker, 0)
                dividend_tax = ticker_dividend_tax.get(ticker, 0)
                net_dividends = gross_dividends - dividend_tax
                
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

    def export_to_excel(self, tickers, start_date, end_date):
        """Export backtest results to Excel file"""
        analysis = self.get_analysis()
        
        # Create output directory if it doesn't exist
        output_dir = "backtest_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename with date range
        filename = f"hi5_backtest_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Portfolio Summary
            summary_data = {
                'Metric': [
                    'Sharpe Ratio',
                    'Annual IRR',
                    'Max Drawdown',
                    'Total Investor Deposits',
                    'Total Dividend Cash',
                ],
                'Value': [
                    analysis['sharpe_ratio'],
                    analysis['annual_irr'] * 100,
                    analysis['max_drawdown'] * 100,
                    analysis['total_investor_deposits'],
                    self.total_dividend_cash,
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Portfolio History
            analysis['portfolio_history_df'].to_excel(writer, sheet_name='Portfolio History', index=False)
            
            # Trade History
            if not analysis['trades'].empty:
                analysis['trades'].to_excel(writer, sheet_name='Trade History', index=False)
            
            # Investment Schedule
            if self.investment_schedule:
                pd.DataFrame(self.investment_schedule).to_excel(writer, sheet_name='Investment Schedule', index=False)
            
            # Final Positions
            if self.final_positions:
                pd.DataFrame(self.final_positions).to_excel(writer, sheet_name='Final Positions', index=False)
            
            # Monthly Returns
            if analysis['monthly_returns'] and analysis['monthly_dates']:
                monthly_data = {
                    'Date': analysis['monthly_dates'],
                    'Return': [r * 100 for r in analysis['monthly_returns']]
                }
                pd.DataFrame(monthly_data).to_excel(writer, sheet_name='Monthly Returns', index=False)
        
        return filepath


def setup_cerebro(strategy_tickers, market_breadth_df=None):
    """Setup cerebro with strategy and analyzers"""
    cerebro = bt.Cerebro(stdstats=False)
    
    # Setup Hi5 strategy state
    storage_engine = InMemoryStorageEngine()
    hi5_state = Hi5State(storage_engine=storage_engine)

    market_breadth_df = market_breadth_df.copy()
    market_breadth_df['market_breadth'] = market_breadth_indicator(market_breadth_df)

    # Add Hi5 strategy
    cerebro.addstrategy(
        Hi5Strategy,
        state=hi5_state, 
        tickers=strategy_tickers,
        enable_cash_injection=True,  # Enable for backtesting
        cash_injection_threshold=5,  # Maintain 3x cash_per_contribution
        cash_per_contribution=10000,
        non_resident_tax_rate=0.3,  # Set to 0 for US residents, 0.3 for non-residents
        market_breadth_df=market_breadth_df,  # Pass market breadth data as parameter
    )

    # Setup broker
    cerebro.broker.set_cash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION_RATE)
    
    # Add analyzers
    cerebro.addanalyzer(
        BacktestAnalyzer, _name="hi5analyzer", risk_free_rate=RISK_FREE_RATE
    )
    
    return cerebro


def add_data_feeds(cerebro, tickers_to_download, data_dict):
    """Add data feeds to cerebro"""
    feeds_added = 0
    for ticker_name in tickers_to_download:
        if ticker_name in data_dict and not data_dict[ticker_name].empty:
            df = data_dict[ticker_name]
            # Validate required columns
            required_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: {ticker_name} missing columns: {missing_cols}. Skipping.")
                continue

            # Create and add data feed
            data_feed = Hi5PandasDataWithDividends(
                dataname=df,
                name=ticker_name,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                dividends="Dividends",
                openinterest=-1,
            )
            data_feed.plotinfo.plot = False
            cerebro.adddata(data_feed)
            feeds_added += 1
    print(f"Added {feeds_added} data feeds to cerebro")
    return feeds_added


def print_results(strategy_instance):
    """Print comprehensive backtest results"""
    # Get analysis results
    hi5_analysis = strategy_instance.analyzers.hi5analyzer.get_analysis()
    final_positions = strategy_instance.analyzers.hi5analyzer.final_positions

    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)

    # Hi5 Strategy Results
    print(f"\n--- Hi5 Strategy Performance ---")
    print(f"Sharpe Ratio: {hi5_analysis.get('sharpe_ratio', 0):.2f}")
    print(f"Annual IRR: {hi5_analysis.get('annual_irr', 0)*100:.2f}%")
    print(f"Max Drawdown: {hi5_analysis.get('max_drawdown', 0)*100:.2f}%")

    # Portfolio Summary
    if final_positions:
        print(f"\n--- Portfolio Summary ---")
        total_market_value = sum(pos['Market Value'] for pos in final_positions)
        total_cost = sum(pos['Total Cost'] for pos in final_positions if pos['Ticker'] != 'CASH')
        total_gross_dividends = sum(pos['Gross Dividends'] for pos in final_positions)
        total_dividend_tax = sum(pos['Dividend Tax'] for pos in final_positions)
        total_net_dividends = total_gross_dividends - total_dividend_tax
        total_pnl_incl_div = sum(pos['Total P&L (incl. Div)'] for pos in final_positions if pos['Ticker'] != 'CASH')
        
        print(f"Final Portfolio Value: ${total_market_value:,.2f}")
        print(f"Total Cost Basis: ${total_cost:,.2f}")
        print(f"Total Net Dividends: ${total_net_dividends:,.2f}")
        print(f"Total P&L (incl. Dividends): ${total_pnl_incl_div:,.2f}")

        # Investment Analysis
        total_investor_deposits = hi5_analysis.get('total_investor_deposits', 0)
        dividend_cash = getattr(strategy_instance, 'total_dividend_cash_received', 0)
        
        print(f"\n--- Investment Analysis ---")
        print(f"Total Deposits Required: ${total_investor_deposits:,.2f}")
        print(f"Total Dividend Cash Received: ${dividend_cash:,.2f}")
        
        if total_investor_deposits > 0:
            total_return_on_deposits = (total_pnl_incl_div / total_investor_deposits) * 100
            dividend_funding_rate = (dividend_cash / total_investor_deposits * 100) if total_investor_deposits > 0 else 0
            
            print(f"Total Return on Deposits: {total_return_on_deposits:.1f}%")
            print(f"Dividend Self-Funding Rate: {dividend_funding_rate:.1f}%")
            
            # Net external cash analysis
            net_external_cash = total_investor_deposits - dividend_cash
            if net_external_cash > 0:
                return_on_external_cash = (total_pnl_incl_div / net_external_cash) * 100
                print(f"Net External Cash Needed: ${net_external_cash:,.2f}")
                print(f"Return on Net External Cash: {return_on_external_cash:.1f}%")

        # Cash injection details (backtest mechanics)
        if hasattr(strategy_instance, 'total_cash_injected') and strategy_instance.total_cash_injected > 0:
            cash_injection_history = getattr(strategy_instance, 'cash_injection_history', [])
            print(f"\n--- Backtest Cash Management ---")
            print(f"Total Cash Injected by System: ${strategy_instance.total_cash_injected:,.2f}")
            print(f"Number of Cash Injections: {len(cash_injection_history)}")
            if cash_injection_history:
                avg_injection = strategy_instance.total_cash_injected / len(cash_injection_history)
                print(f"Average Injection Size: ${avg_injection:,.2f}")

    print("\n" + "="*60)


def run_backtest():
    """Main backtest execution function"""
    print("Initializing backtest...")
    
    # Strategy configuration
    strategy_tickers = ["VUG", "VO", "MOAT", "PFF", "VNQ"]
    benchmark_ticker = "RSP"

    # Download data for all tickers (trading + benchmark)
    all_tickers = strategy_tickers + [benchmark_ticker]
    
    # Load data from BigQuery
    data_dict = load_bigquery_data(all_tickers, START_DATE, END_DATE, "gcp-config.json", "hi5-strategy")
    market_breadth_df = load_market_breadth(START_DATE, END_DATE, "gcp-config.json", "hi5-strategy")
    
    if not data_dict:
        print("Error: No data downloaded. Exiting.")
        return
    
    # Setup cerebro with market breadth data
    cerebro = setup_cerebro(strategy_tickers, market_breadth_df)
    
    # Add data feeds
    feeds_added = add_data_feeds(cerebro, all_tickers, data_dict)
    if feeds_added == 0:
        print("Error: No data feeds added. Exiting.")
        return
    
    print(f"\nStarting backtest from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}...")
    
    # Run backtest
    results = cerebro.run()
    print("Backtest completed successfully!")
    
    # Get strategy instance and capture final positions
    strategy_instance = results[0]
    strategy_instance.analyzers.hi5analyzer.capture_final_positions(strategy_instance)
    
    # Print results
    print_results(strategy_instance)
    
    # Export to Excel
    hi5_analysis = strategy_instance.analyzers.hi5analyzer.get_analysis()
    
    excel_file = strategy_instance.analyzers.hi5analyzer.export_to_excel(
        tickers=strategy_tickers,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    print(f"\nDetailed results exported to: {excel_file}")
    
    # Optional plotting
    try_plot_results(cerebro)
    
    return strategy_instance


def try_plot_results(cerebro):
    """Attempt to plot results with error handling"""
    try:
        if "RSP" in cerebro.datasbyname:
            cerebro.plot(
                style="candlestick", 
                barup="green", 
                bardown="red", 
            )
        else:
            cerebro.plot(style="candlestick", barup="green", bardown="red")
    except Exception as e:
        print(f"Note: Chart plotting failed ({str(e)}), but backtest completed successfully")


def load_gcp_config():
    """Load GCP configuration from gcp-config.json"""
    try:
        with open("gcp-config.json", 'r') as f:
            config = json.load(f)
        if not config.get('project_id'):
            raise ValueError("project_id not found in gcp-config.json")
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading GCP config: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Hi5 Backtester with BigQuery support")
    parser.add_argument('--tickers', type=str, default='VUG,VO,MOAT,PFF,VNQ', help='Comma-separated list of tickers')
    parser.add_argument('--benchmark-ticker', type=str, default='RSP', help='Benchmark ticker (default: RSP)')
    parser.add_argument('--start-date', type=str, default='2012-05-01', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='Backtest end date (YYYY-MM-DD, default: today)')
    parser.add_argument('--gcp-config', type=str, default='hi5/gcp-config.json', help='Path to GCP config JSON')
    return parser.parse_args()


def load_bigquery_data(tickers, start_date, end_date, gcp_config):
    """Load market data from BigQuery"""
    # Load GCP credentials
    credentials = service_account.Credentials.from_service_account_file(gcp_config['credentials_path'])
    client = bigquery.Client(credentials=credentials, project=gcp_config['project_id'])

    # Query BigQuery for all tickers and date range
    tickers_list = ','.join([f"'{t}'" for t in tickers])
    query = f'''
        SELECT date, ticker, open, high, low, adj_close as close, volume, dividend
        FROM `{gcp_config['project_id']}.market_data.marketing`
        WHERE ticker IN ({tickers_list})
          AND date >= '{start_date.strftime('%Y-%m-%d')}'
          AND date <= '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY ticker, date
    '''
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError('No data returned from BigQuery for the given tickers and date range.')
    
    # Prepare a dict of DataFrames, one per ticker
    data_dict = {}
    for ticker in tickers:
        tdf = df[df['ticker'] == ticker].copy()
        tdf = tdf.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'dividend': 'Dividends'
        })
        tdf = tdf[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']]
        tdf['date'] = pd.to_datetime(tdf['date'])
        tdf.set_index('date', inplace=True)
        data_dict[ticker] = tdf
    return data_dict


def load_market_breadth(start_date, end_date, gcp_config):
    """Load market breadth data from BigQuery"""
    credentials = service_account.Credentials.from_service_account_file(gcp_config['credentials_path'])
    client = bigquery.Client(credentials=credentials, project=gcp_config['project_id'])

    query = f'''
        SELECT date, ema20_ratio
        FROM `{gcp_config['project_id']}.market_data.market_breadth`
        WHERE date >= '{start_date.strftime('%Y-%m-%d')}'
          AND date <= '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY date
    '''
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError('No market breadth data returned from BigQuery for the given date range.')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def market_breadth_indicator(market_breadth_df):
    """Compute market breadth indicator from market_breadth_df.
    
    This function takes the market_breadth_df (which contains the ema20_ratio) and computes a market breadth indicator.
    The indicator is computed as the difference between the current ema20_ratio and its 10-day moving average.
    
    Args:
        market_breadth_df (pandas.DataFrame): DataFrame containing the ema20_ratio column.
    
    Returns:
        pandas.Series: The computed market breadth indicator.
    """
    if 'ema20_ratio' not in market_breadth_df.columns:
        raise ValueError("market_breadth_df must contain an 'ema20_ratio' column.")
    
    # Compute the 10-day moving average of ema20_ratio
    ma10 = market_breadth_df['ema20_ratio'].rolling(window=10).mean()
    
    # Compute the market breadth indicator as the difference between current ema20_ratio and its 10-day moving average
    indicator = market_breadth_df['ema20_ratio'] - ma10
    
    return indicator


if __name__ == "__main__":
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(',')]
    benchmark_ticker = args.benchmark_ticker
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date) if args.end_date else datetime.datetime.now()
    
    # Load GCP configuration
    gcp_config = load_gcp_config()

    print(f"Tickers: {tickers}")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    print(f"GCP Config: {args.gcp_config}")
    print(f"Project ID: {gcp_config['project_id']}")

    # Load data from BigQuery
    all_tickers = tickers + [benchmark_ticker]
    data_dict = load_bigquery_data(all_tickers, start_date, end_date, gcp_config)
    market_breadth_df = load_market_breadth(start_date, end_date, gcp_config)

    # Setup and run backtest
    cerebro = setup_cerebro(tickers, market_breadth_df)
    feeds_added = add_data_feeds(cerebro, all_tickers, data_dict)
    if feeds_added == 0:
        print("Error: No data feeds added. Exiting.")
        exit(1)

    print("\nRunning backtest...")
    results = cerebro.run()
    strategy = results[0]
    
    analysis = strategy.analyzers.hi5analyzer.get_analysis()
    
    print("\nBacktest Results:")
    print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
    print(f"Annual IRR: {analysis['annual_irr']*100:.2f}%")
    print(f"Max Drawdown: {analysis['max_drawdown']*100:.2f}%")
    print(f"Total Investor Deposits: ${analysis['total_investor_deposits']:,.2f}")
    
    excel_file = strategy.analyzers.hi5analyzer.export_to_excel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    print(f"\nDetailed results exported to: {excel_file}")
    
    try:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    except Exception as e:
        print(f"Note: Chart plotting failed ({str(e)}), but backtest completed successfully")
