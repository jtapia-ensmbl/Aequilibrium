# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

"""Unit tests for MarketData."""

import pytest
import pandas as pd
import yfinance.shared as shared

from aequilibrium.market_data import MarketData


@pytest.fixture
def md():
    """
    Provide a fresh marketData instance for each test.

    Ensures tests are isolated and do not share state.
    """
    return MarketData()

# Helper function to clear errors before and after tests


@pytest.fixture(autouse=True)
def clear_yf_errors():
    """Fixture to ensure shared._ERRORS is clear before and after each test."""
    shared._ERRORS.clear()
    yield
    shared._ERRORS.clear()


def test_fetch_returns_structure(md):
    """
    Test that the fetch method returns a DataFrame with correct structure.
    """
    tickers = ['AAPL', 'MSFT']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    data = md.fetch_returns(tickers, start_date, end_date)

    # 1. Check if the result is a pandas DataFrame
    assert isinstance(data, pd.DataFrame)

    # 2. Check if the columns match the expected return names
    expected_cols = ['AAPL', 'MSFT']
    assert list(data.columns) == expected_cols

    # 3. Check if there are a reasonable number of trading days (around 250 in a year)
    assert len(data) > 200 and len(data) < 300

    # 4. Check that the first few entries are not empty
    assert not data.head().empty


def test_fetch_no_nan_values(md):
    """
    Test that the returned DataFrame contains no NaN values after dropping the first row.
    """
    tickers = ['AAPL', 'MSFT']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    data = md.fetch_returns(tickers, start_date, end_date)

    # Verify there are no missing values anywhere in the DataFrame
    assert not data.isna().any().any()


def test_fetch_single_ticker(md):
    """
    Ensure the method works correctly when fetching data for a single stock.
    """
    tickers = ['AAPL']
    data = md.fetch_returns(tickers, '2023-01-01', '2023-01-31')

    # Check that the single expected column is present
    assert list(data.columns) == ['AAPL']
    # Check shape is reasonable for 1 month of data
    assert len(data) > 15


def test_returns_date_alignment(md):
    """
    Testing a range with a weekend (Jan 1, 2023 was a Sunday)
    """
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-10')
    # First trading day was Jan 3rd, pct_change makes first return Jan 4th
    assert data.index[0] == pd.Timestamp('2023-01-04')


@pytest.mark.parametrize("tickers", [['AAPL'], ['AAPL', 'MSFT', 'GOOGL']])
def test_returns_column_count(md, tickers):
    """
    Verify that the returned data frame contains one column for each requested ticker.
    """
    data = md.fetch_returns(tickers, '2023-01-01', '2023-01-15')
    assert len(data.columns) == len(tickers)


@pytest.mark.parametrize("invalid_tickers", [
    (['NONEXISTENT']),               # Single invalid ticker
    (['AAPL', 'NOTAREALTICKER']),    # Mixed valid and invalid
    (['INVALID1', 'INVALID2'])       # Multiple invalid tickers
])
def test_detects_invalid_tickers(md, invalid_tickers):
    """
    Test that invalid ticker symbols are correctly identified
    """
    md.fetch_returns(invalid_tickers, '2023-01-01', '2023-01-15')

    # Check if the specific invalid tickers were captured in _ERRORS
    for ticker in invalid_tickers:
        if ticker == 'AAPL':  # Skip valid ones if testing mixed sets
            continue

        # Verify the ticker exists as a key in the errors dictionary
        assert ticker in shared._ERRORS

        # Verify the error message contains '404' or 'not found'
        error_msg = str(shared._ERRORS[ticker])
        assert any(x in error_msg.lower()
            for x in ["404", "not found", "no timezone"])
