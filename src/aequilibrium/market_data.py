"""
Market Data Module

This module provides tools for loading, processing, and analyzing financial
market data, specifically historical stock prices and returns. It abstracts
common operations like fetching data from external sources and preparing it
for portfolio analysis or quantitative modeling.
"""

import yfinance as yf
import pandas as pd


class MarketData:
    """
    Handles operations related to market data acquisition, storage, and processing.

    This class serves as a container for methods that manage financial data,
    such as loading stock data, calculating volatility, or managing portfolio
    metrics.
    """

    def returns(self, tickers, start_date, end_date):
        """
        Load historical stock price data and compute daily returns.

        Parameters
        ----------
        tickers : list of str
            Stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format

        Returns
        -------
        pd.DataFrame
            Daily returns for each stock (columns = tickers, rows = dates)
        """
        # Fetch historical data froa specific period
        # It gets 'Adj Close' prices (adjusted for splits/dividends)
        # tThis operation downloads data into a pandas DatFrame
        data = yf.download(tickers, start=start_date,
                           end=end_date, auto_adjust=True)

        # Calculate daily fractional change (daily returns)
        rets = data['Close'].pct_change()
        # Drop the first row (NaN from pct_change)
        rets = rets.dropna()

        return rets
