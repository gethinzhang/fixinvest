import backtrader as bt
import yfinance as yf
import os
import pandas as pd
import datetime
import numpy as np
from hi5 import (
    Hi5PandasDataWithDividends,
    Hi5Strategy,
    Hi5State,
    InMemoryStorageEngine,
    LocalStorageEngine,
    BacktestAnalyzer,
    BenchmarkAnalyzer,
)
import pytz

START_DATE = datetime.datetime(2012, 5, 1)
#END_DATE = datetime.datetime(2023, 12, 31)
END_DATE = datetime.datetime.now()
INITIAL_CASH = 100000.0
COMMISSION_RATE = 0.001
RISK_FREE_RATE = 0.04


def download_multiple_tickers(tickers, start, end, cache_dir="cache"):
    """Download and cache ticker data with dividends"""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    data_dict = {}
    for ticker in tickers:
        safe_ticker_name = "".join(c if c.isalnum() else "_" for c in ticker)
        cache_file = os.path.join(
            cache_dir,
            f"{safe_ticker_name}-{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}.csv",
        )
        
        if os.path.exists(cache_file):
            # Load cached data
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if "Dividends" not in df.columns:
                df["Dividends"] = 0.0
            data_dict[ticker] = df
        else:
            # Download new data
            try:
                ticker_obj = yf.Ticker(ticker)
                df_prices = ticker_obj.history(
                    start=start, end=end, auto_adjust=False, back_adjust=False
                )
                if df_prices.empty:
                    print(f"Warning: No price data for {ticker}")
                    continue
                
                # Handle dividends with timezone conversion
                us_tz = pytz.timezone("US/Eastern")
                df_dividends_ts = ticker_obj.dividends.tz_convert(us_tz)
                start_ts_us = pd.Timestamp(start).tz_localize(us_tz)
                end_ts_us = pd.Timestamp(end).tz_localize(us_tz)
                df_dividends = df_dividends_ts[
                    (df_dividends_ts.index >= start_ts_us)
                    & (df_dividends_ts.index <= end_ts_us)
                ]
                
                # Initialize dividends column
                df_prices["Dividends"] = 0.0
                if not df_dividends.empty:
                    temp_div_series = pd.Series(df_dividends.values, index=df_dividends.index)
                    aligned_dividends = temp_div_series.reindex(df_prices.index).fillna(0.0)
                    df_prices["Dividends"] = aligned_dividends

                # Standardize column names
                df_prices.rename(
                    columns={
                        "Open": "Open",
                        "High": "High", 
                        "Low": "Low",
                        "Close": "Close",
                        "Volume": "Volume",
                    },
                    inplace=True,
                )
                
                # Cache the data
                df_prices.to_csv(cache_file)
                data_dict[ticker] = df_prices
                
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
    
    return data_dict


def setup_cerebro(strategy_tickers):
    """Setup cerebro with strategy and analyzers"""
    cerebro = bt.Cerebro(stdstats=False)
    
    # Setup Hi5 strategy state
    storage_engine = LocalStorageEngine(local_file_path="./hi5_strategy_state.json")
    hi5_state = Hi5State(storage_engine=storage_engine)

    # Add Hi5 strategy
    cerebro.addstrategy(
        Hi5Strategy, 
        state=hi5_state, 
        tickers=strategy_tickers,
        enable_cash_injection=True,  # Enable for backtesting
        cash_injection_threshold=3,  # Maintain 3x cash_per_contribution
        cash_per_contribution=10000,
        non_resident_tax_rate=0,  # Set to 0 for US residents, 0.3 for non-residents
    )

    # Setup broker
    cerebro.broker.set_cash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION_RATE)
    
    # Add analyzers
    cerebro.addanalyzer(
        BacktestAnalyzer, _name="hi5analyzer", risk_free_rate=RISK_FREE_RATE
    )
    cerebro.addanalyzer(
        BenchmarkAnalyzer, _name="benchmark", risk_free_rate=RISK_FREE_RATE
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
    benchmark_analysis = strategy_instance.analyzers.benchmark.get_benchmark_analysis()
    final_positions = strategy_instance.analyzers.hi5analyzer.final_positions

    print("\n" + "="*60)
    print("BACKTEST RESULTS SUMMARY")
    print("="*60)

    # Hi5 Strategy Results
    print(f"\n--- Hi5 Strategy Performance ---")
    print(f"Sharpe Ratio: {hi5_analysis.get('sharpe_ratio', 0):.2f}")
    print(f"Annual IRR: {hi5_analysis.get('annual_irr', 0)*100:.2f}%")
    print(f"Max Drawdown: {hi5_analysis.get('max_drawdown', 0)*100:.2f}%")

    # Benchmark Results
    print(f"\n--- Benchmark (Buy & Hold) Performance ---")
    print(f"Annual IRR: {benchmark_analysis.get('benchmark_annual_irr', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {benchmark_analysis.get('benchmark_sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {benchmark_analysis.get('benchmark_max_drawdown', 0)*100:.2f}%")

    # Performance Comparison
    excess_return = hi5_analysis.get('annual_irr', 0) - benchmark_analysis.get('benchmark_annual_irr', 0)
    print(f"\n--- Performance Comparison ---")
    print(f"Excess Return vs Benchmark: {excess_return*100:.2f}%")

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
    strategy_tickers = ["IWY", "RSP", "MOAT", "PFF", "VNQ"]
    
    # Setup cerebro
    cerebro = setup_cerebro(strategy_tickers)
    
    # Download data
    print(f"Downloading data for {len(strategy_tickers)} tickers...")
    data_dict = download_multiple_tickers(strategy_tickers, START_DATE, END_DATE)
    
    if not data_dict:
        print("Error: No data downloaded. Exiting.")
        return
    
    # Add data feeds
    feeds_added = add_data_feeds(cerebro, strategy_tickers, data_dict)
    if feeds_added == 0:
        print("Error: No data feeds added. Exiting.")
        return
    
    print(f"\nStarting backtest from {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}...")
    
    # Run backtest
    try:
        results = cerebro.run()
        print("Backtest completed successfully!")
        
        # Get strategy instance and capture final positions
        strategy_instance = results[0]
        strategy_instance.analyzers.hi5analyzer.capture_final_positions(strategy_instance)
        
        # Print results
        print_results(strategy_instance)
        
        # Export to Excel
        hi5_analysis = strategy_instance.analyzers.hi5analyzer.get_analysis()
        benchmark_analysis = strategy_instance.analyzers.benchmark.get_benchmark_analysis()
        
        excel_file = strategy_instance.analyzers.hi5analyzer.export_to_excel(
            tickers=strategy_tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            benchmark_analysis=benchmark_analysis,
        )
        print(f"\nDetailed results exported to: {excel_file}")
        
        # Optional plotting
        try_plot_results(cerebro)
        
        return strategy_instance
        
    except Exception as e:
        print(f"Error during backtest execution: {e}")
        return None


def try_plot_results(cerebro):
    """Attempt to plot results with error handling"""
    try:
        if "RSP" in cerebro.datasbyname:
            cerebro.plot(
                style="candlestick", 
                barup="green", 
                bardown="red", 
                benchmarkname="RSP"
            )
        else:
            cerebro.plot(style="candlestick", barup="green", bardown="red")
    except Exception as e:
        print(f"Note: Chart plotting failed ({str(e)}), but backtest completed successfully")


if __name__ == "__main__":
    strategy_instance = run_backtest()
    
    #if strategy_instance:
    #    print("\nBacktest completed successfully!")
    #else:
   #     print("\nBacktest failed!")
