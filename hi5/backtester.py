import argparse
import datetime
import pandas as pd
import backtrader as bt
from backtest.analyzer import BacktestAnalyzer
from hi5.hi5 import (
    Hi5PandasDataWithDividends,
    Hi5Strategy,
    Hi5State,
    InMemoryStorageEngine,
)

from google.cloud import bigquery
from google.oauth2 import service_account


def load_gcp_config():
    import json

    with open("gcp-config.json", "r") as f:
        config = json.load(f)
    if not config.get("project_id"):
        raise ValueError("project_id not found in gcp-config.json")
    return config


def add_data_feeds(cerebro, tickers_to_download, data_dict):
    feeds_added = 0
    for ticker_name in tickers_to_download:
        if ticker_name in data_dict and not data_dict[ticker_name].empty:
            df = data_dict[ticker_name]
            required_cols = ["Open", "High", "Low", "Close", "Volume", "Dividends"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                continue
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
    return feeds_added


def load_bigquery_data(tickers, start_date, end_date, gcp_config):
    credentials = service_account.Credentials.from_service_account_file(
        gcp_config["credentials_path"]
    )
    client = bigquery.Client(credentials=credentials, project=gcp_config["project_id"])
    tickers_list = ",".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT date, ticker, open, high, low, close, volume, dividend
        FROM `{gcp_config["project_id"]}.market_data.marketing`
        WHERE ticker IN ({tickers_list})
          AND date >= '{start_date.strftime('%Y-%m-%d')}'
          AND date <= '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY ticker, date
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError(
            "No data returned from BigQuery for the given tickers and date range."
        )

    # Create a full date range for the backtest to align all data feeds
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    data_dict = {}
    for ticker in tickers:
        tdf = df[df["ticker"] == ticker].copy()
        if tdf.empty:
            continue
        tdf["date"] = pd.to_datetime(tdf["date"])
        tdf.set_index("date", inplace=True)
        tdf.sort_index(inplace=True)

        # Reindex to the full date range, forward-fill holidays/missing days, then backfill any leading NaNs
        tdf = tdf.reindex(full_date_range).ffill().bfill()

        if "split_ratio" in tdf.columns and not tdf[tdf["split_ratio"] != 1.0].empty:
            price_adj_factor = (
                tdf["split_ratio"].iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1)
            )
            tdf["open"] = tdf["open"] / price_adj_factor
            tdf["high"] = tdf["high"] / price_adj_factor
            tdf["low"] = tdf["low"] / price_adj_factor
            tdf["close"] = tdf["close"] / price_adj_factor
            tdf["volume"] = tdf["volume"] * price_adj_factor

        tdf = tdf.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "dividend": "Dividends",
            }
        )
        tdf = tdf[["Open", "High", "Low", "Close", "Volume", "Dividends"]]
        data_dict[ticker] = tdf
    return data_dict


def load_market_breadth(start_date, end_date, gcp_config):
    credentials = service_account.Credentials.from_service_account_file(
        gcp_config["credentials_path"]
    )
    client = bigquery.Client(credentials=credentials, project=gcp_config["project_id"])
    query = f"""
        SELECT date, ma50_ratio
        FROM `{gcp_config["project_id"]}.market_data.market_breadth`
        WHERE date >= '{start_date.strftime('%Y-%m-%d')}'
          AND date <= '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY date
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError(
            "No market breadth data returned from BigQuery for the given date range."
        )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


# --- Helper functions ---
def setup_cerebro(
    strategy_tickers,
    benchmark_ticker,
    market_breadth_df=None,
    contribution_mode="fixed",
    monthly_increment=500.0,
    contribution_percentage=0.2,
    initial_contribution=10000.0,
):
    cerebro = bt.Cerebro(stdstats=False)
    storage_engine = InMemoryStorageEngine()
    hi5_state = Hi5State(storage_engine=storage_engine)
    market_breadth_df = market_breadth_df.copy()
    market_breadth_df["market_breadth"] = market_breadth_df["ma50_ratio"]
    cerebro.addstrategy(
        Hi5Strategy,
        state=hi5_state,
        tickers=strategy_tickers,
        benchmark_ticker=benchmark_ticker,
        enable_cash_injection=True,
        cash_injection_threshold=5,
        cash_per_contribution=initial_contribution,
        market_breadth_df=market_breadth_df,
        contribution_mode=contribution_mode,
        monthly_increment=monthly_increment,
        contribution_percentage=contribution_percentage,
    )

    cerebro.addanalyzer(
        BacktestAnalyzer,
        _name="hi5analyzer",
        risk_free_rate=0.04,
        non_resident_tax_rate=0.15,
        fix_investment=True,
    )
    return cerebro


def main():
    parser = argparse.ArgumentParser(description="Hi5 Backtest Runner")
    parser.add_argument(
        "--tickers",
        type=str,
        #default="VUG,VO,MOAT,PFF,VNQ",
        default="VUG,VO,MOAT,PFFD,IDWP.L",
        help="Comma-separated list of tickers",
    )
    parser.add_argument(
        "--benchmark-ticker",
        type=str,
        default="RSP",
        help="Benchmark ticker (default: RSP)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--gcp-config",
        type=str,
        default="gcp-config.json",
        help="Path to GCP config JSON",
    )
    parser.add_argument(
        "--show-drawdown-history",
        action="store_true",
        help="Show portfolio values during the maximum drawdown period.",
    )
    # New arguments for contribution logic
    parser.add_argument(
        "--contribution-mode",
        type=str,
        default="fixed",
        choices=["fixed", "incremental", "percentage"],
        help="Contribution strategy: fixed amount, incremental, or percentage of portfolio.",
    )
    parser.add_argument(
        "--monthly-increment",
        type=float,
        default=500.0,
        help="The amount to increment the contribution by each month in 'incremental' mode.",
    )
    parser.add_argument(
        "--contribution-percentage",
        type=float,
        default=0.01,
        help="The portfolio percentage to contribute in 'percentage' mode (e.g., 0.01 for 1%).",
    )
    parser.add_argument(
        "--initial-contribution",
        type=float,
        default=10000.0,
        help="The initial contribution amount for 'fixed' and 'incremental' modes.",
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    benchmark_ticker = args.benchmark_ticker
    start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = (
        datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.end_date
        else datetime.datetime.now()
    )
    gcp_config = load_gcp_config()

    print(f"Tickers: {tickers}")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Start: {start_date}")
    print(f"End: {end_date}")
    print(f"GCP Config: {args.gcp_config}")
    print(f"Project ID: {gcp_config['project_id']}")
    print("--------------------------------")

    all_tickers = tickers + [benchmark_ticker]

    market_breadth_df = load_market_breadth(start_date, end_date, gcp_config)
    data_df = load_bigquery_data(all_tickers, start_date, end_date, gcp_config)
    cerebro = setup_cerebro(
        tickers,
        benchmark_ticker,
        market_breadth_df,
        contribution_mode=args.contribution_mode,
        monthly_increment=args.monthly_increment,
        contribution_percentage=args.contribution_percentage,
        initial_contribution=args.initial_contribution,
    )
    add_data_feeds(cerebro, all_tickers, data_df)

    cerebro.broker.setcash(1.0)
    cerebro.broker.setcommission(commission=0.003)
    results = cerebro.run()
    hi5_strategy = results[0]
    analyzer = hi5_strategy.analyzers.hi5analyzer


    analyzer.print_summary(print_drawdown_history=args.show_drawdown_history)
    analyzer.export_to_excel()


if __name__ == "__main__":
    main()
