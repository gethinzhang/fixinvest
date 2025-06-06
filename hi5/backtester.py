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
    LocalStorageEngine,
    BacktestAnalyzer,
)
import pytz
import argparse
import json
from google.cloud import bigquery
from google.oauth2 import service_account

START_DATE = datetime.datetime(2012, 5, 1)
#END_DATE = datetime.datetime(2023, 12, 31)
END_DATE = datetime.datetime.now()
INITIAL_CASH = 100000.0
COMMISSION_RATE = 0.001
RISK_FREE_RATE = 0.04


def setup_cerebro(strategy_tickers, market_breadth_df=None):
    """Setup cerebro with strategy and analyzers"""
    cerebro = bt.Cerebro(stdstats=False)
    
    # Setup Hi5 strategy state
    storage_engine = InMemoryStorageEngine()
    hi5_state = Hi5State(storage_engine=storage_engine)

    # Calculate market breadth indicator if data is provided
    if market_breadth_df is not None:
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
        SELECT date, ticker, open, high, low, adj_close as close, volume
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
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        tdf['Dividends'] = 0.0  # No dividends data in table
        tdf = tdf[['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends']]
        # Convert date to datetime
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
        # Run the backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Get analysis results
    analysis = strategy.analyzers.hi5analyzer.get_analysis()
    
    # Print key metrics
    print("\nBacktest Results:")
    print(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
    print(f"Annual IRR: {analysis['annual_irr']*100:.2f}%")
    print(f"Max Drawdown: {analysis['max_drawdown']*100:.2f}%")
    print(f"Total Investor Deposits: ${analysis['total_investor_deposits']:,.2f}")
    
    # Export detailed results to Excel
    excel_file = strategy.analyzers.hi5analyzer.export_to_excel(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    print(f"\nDetailed results exported to: {excel_file}")
    
    # Try to plot results
    try:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    except Exception as e:
        print(f"Note: Chart plotting failed ({str(e)}), but backtest completed successfully")
