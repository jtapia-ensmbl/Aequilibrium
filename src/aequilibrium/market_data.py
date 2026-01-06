"""
Market Data Module

This module provides tools for loading, processing, and analyzing financial
market data, specifically historical stock prices and returns. It abstracts
common operations like fetching data from external sources and preparing it
for portfolio analysis or quantitative modeling.
"""

import pandas as pd
import yfinance as yf

class MarketData:
    """
    Handles operations related to market data acquisition, storage, and processing.

    This class is designed for use in trading simulators and backtesting
    environments where robustness and deterministic behavior are critical.
    """

    @staticmethod
    def _empty_returns(tickers):
        """
        Create an empty returns DataFrame with the expected schema.

        Parameters
        ----------
        tickers : list[str]

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with ticker columns
        """
        return pd.DataFrame(columns=tickers)

    def fetch_returns(self, tickers, start_date, end_date):
        """
        Fetch historical stock prices and compute daily returns.

        Fail-soft behavior:
        - Any error or missing data returns an empty DataFrame.
        - No exceptions are raised.
        - Columns always match requested tickers.
        - Missing tickers appear as all-NaN columns.
        - Forward-fills available prices before computing returns.

        Parameters
        ----------
        tickers : list of str
            Stock ticker symbols.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame
            Daily returns indexed by date with columns = tickers.
            May be empty if data is unavailable, invalid, or tickers have no data.
        """
        # Start with an empty DataFrame that matches the expected schema
        returns = self._empty_returns(tickers).copy()

        # Attempt to parse start and end dates, returning empty if invalid
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if pd.isna(start) or pd.isna(end) or start > end:
                return returns
        except (ValueError, TypeError):
            return returns

        # Attempt to download historical data from yfinance
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
            )
        except (ValueError, ConnectionError, TimeoutError, OSError):
            return returns

        # If download fails or returns empty data, keep the DataFrame empty
        if data is None or data.empty:
            return returns

        # Extract the 'Close' prices; if missing, returns remain empty
        prices = data["Close"] if "Close" in data.columns else None

        if prices is not None:
            # Ensure consistent DataFrame shape for a single ticker
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])

            # Fill forward missing prices to avoid gaps in returns
            prices = prices.ffill()

            # Compute daily percentage change; drop the first row which is NaN
            tmp_returns = prices.pct_change(fill_method=None).dropna()

            # Only update the returns DataFrame if there is valid data
            if not tmp_returns.empty:
                # Reindex to guarantee columns match requested tickers
                returns = tmp_returns.reindex(columns=tickers)

        # Return the fail-soft DataFrame, either empty or with valid returns
        return returns
