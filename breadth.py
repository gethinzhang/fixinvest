import backtrader as bt
import numpy as np
import yfinance as yf
import pandas as pd
import requests
import os
import datetime


# --- Helper Function for S&P 500 Tickers (using yahoo_fin for convenience) ---
def get_sp500_tickers():
    """
    Retrieves the S&P 500 tickers directly from a raw CSV file on GitHub.
    This function reads the data into memory without saving a local file.

    Returns:
        list: A list of S&P 500 ticker symbols.
        None: If there's an error retrieving the data.
    """
    # The raw URL of the sp500.csv file on GitHub
    github_raw_url = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
    try:
        # Use pandas to directly read the CSV from the URL
        # pandas can handle URLs directly for common file types
        df = pd.read_csv(github_raw_url)

        # Assuming the ticker symbol is in a column named 'Symbol' or 'symbol'
        # You might need to inspect the CSV structure if this doesn't work
        if "Symbol" in df.columns:
            tickers = df["Symbol"].tolist()
        elif "symbol" in df.columns:  # Sometimes column names are lowercase
            tickers = df["symbol"].tolist()
        elif "Ticker" in df.columns:
            tickers = df["Ticker"].tolist()
        else:
            print(
                "Error: Could not find a 'Symbol', 'symbol', or 'Ticker' column in the GitHub CSV."
            )
            print(f"Available columns: {df.columns.tolist()}")
            return None

        # Clean tickers (e.g., replace periods with hyphens for yfinance compatibility)
        clean_tickers = [ticker.replace(".", "-") for ticker in tickers]
        return clean_tickers

    except requests.exceptions.RequestException as e:
        print(f"Network error or invalid URL: {e}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty or malformed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# --- Custom SP500 Breadth Indicator ---
class SP500Breadth(bt.Indicator):
    """
    Calculates S&P 500 breadth indicators:
    - S5TW: Percentage of S&P 500 stocks above their 20-day SMA
    - S5FI: Percentage of S&P 500 stocks above their 50-day SMA
    - S5TH: Percentage of S&P 500 stocks above their 100-day SMA

    This indicator expects to be passed a tuple of all data feeds (self.datas from strategy).
    """

    lines = (
        "s5tw",
        "s5fi",
        "s5th",
    )
    params = (("ma_periods", [20, 50, 100]),)  # Default MA periods for the indicators

    def __init__(self):
        # Dictionary to hold SMA instances for each component and each period
        self.smas = {period: [] for period in self.p.ma_periods}

        # Create SMA indicators for each component data feed for all periods
        for d in self.datas[0]:
            for period in self.p.ma_periods:
                self.smas[period].append(bt.indicators.SMA(d.close, period=period))

        # Plotting info
        self.plotinfo.plotname = "S&P 500 Breadth"
        self.plotinfo.plot = True
        self.plotinfo.subplot = True
        self.plotinfo.plotupper = 100
        self.plotinfo.plotlower = 0

    def next(self):
        # Calculate for each period
        for i, period in enumerate(self.p.ma_periods):
            num_above_ma = 0
            total_stocks_with_valid_ma = 0

            # Iterate through each component stock's data and its corresponding SMA
            for j, d in enumerate(self.datas):
                current_sma = self.smas[period][
                    j
                ]  # Get the specific SMA for this data and period

                # Check if current close and SMA are valid (i.e., enough historical data for SMA calculation)
                if (
                    len(d) > period
                    and d.close[0] is not None
                    and current_sma[0] is not None
                ):
                    total_stocks_with_valid_ma += 1
                    if d.close[0] > current_sma[0]:
                        num_above_ma += 1

            # Set the line value based on the index of the period in self.p.ma_periods
            if total_stocks_with_valid_ma > 0:
                percentage = (num_above_ma / total_stocks_with_valid_ma) * 100
                if period == 20:
                    self.lines.s5tw[0] = percentage
                elif period == 50:
                    self.lines.s5fi[0] = percentage
                elif period == 100:
                    self.lines.s5th[0] = percentage
            else:
                # If no valid stocks, set to NaN or carry previous value (NaN for initial period)
                # Using nz (null-to-zero) for initial fill, or propagate previous value
                # This ensures the indicator starts properly once data is available.
                prev_val = self.lines[i][-1] if len(self) > 1 else np.nan
                self.lines[i][0] = prev_val


def download_multiple_tickers(tickers, start, end, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_dict = {}
    for ticker in tickers:
        cache_file = os.path.join(
            cache_dir,
            f"{ticker}-{start.strftime('%Y-%m-%d')}-{end.strftime('%Y-%m-%d')}.csv",
        )
        if os.path.exists(cache_file):
            data_dict[ticker] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            try:
                ticker_obj = yf.Ticker(ticker)
                # Download both price and dividend data
                df = ticker_obj.history(start=start, end=end)
                # Get dividend data
                dividends = ticker_obj.dividends
                if not dividends.empty:
                    # Align dividend data with price data
                    df["Dividends"] = dividends.reindex(df.index).fillna(0)
                else:
                    df["Dividends"] = 0
                df.to_csv(cache_file)
                data_dict[ticker] = df
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
    return data_dict


class PandasDataWithDividends(bt.feeds.PandasData):
    params = (("dividends", None),)
    lines = ("dividends",)


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    if tickers:
        data_dict = download_multiple_tickers(
            tickers, start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2024, 1, 1)
        )
        # Create a Cerebro instance
        cerebro = bt.Cerebro()
        for ticker in tickers:
            if ticker in data_dict:
                data = PandasDataWithDividends(
                    dataname=data_dict[ticker],
                    datetime=None,  # 使用索引作为日期
                    open="Open",
                    high="High",
                    low="Low",
                    close="Close",
                    volume="Volume",
                    dividends="Dividends",
                    openinterest=-1,
                )
                cerebro.adddata(data, name=ticker)

        # Add the SP500Breadth indicator
        cerebro.addindicator(SP500Breadth, cerebro.datas)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        # Set initial cash
        cerebro.broker.setcash(100000.0)

        # Run the backtest
        print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
        results = cerebro.run()
        print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())

        # Plot the results
        cerebro.plot(style="candlestick")
