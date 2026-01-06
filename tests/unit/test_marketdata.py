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


def test_fetch_returns_structure(md: MarketData):
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


def test_fetch_no_nan_values(md: MarketData):
    """
    Test that the returned DataFrame contains no NaN values after dropping the first row.
    """
    tickers = ['AAPL', 'MSFT']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    data = md.fetch_returns(tickers, start_date, end_date)

    # Verify there are no missing values anywhere in the DataFrame
    assert not data.isna().any().any()


def test_fetch_single_ticker(md: MarketData):
    """
    Ensure the method works correctly when fetching data for a single stock.
    """
    tickers = ['AAPL']
    data = md.fetch_returns(tickers, '2023-01-01', '2023-01-31')

    # Check that the single expected column is present
    assert list(data.columns) == ['AAPL']
    # Check shape is reasonable for 1 month of data
    assert len(data) > 15


def test_returns_date_alignment(md: MarketData):
    """
    Testing a range with a weekend (Jan 1, 2023 was a Sunday)
    """
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-10')
    # First trading day was Jan 3rd, pct_change makes first return Jan 4th
    assert data.index[0] == pd.Timestamp('2023-01-04')


@pytest.mark.parametrize("tickers", [['AAPL'], ['AAPL', 'MSFT', 'GOOGL']])
def test_returns_column_count(md: MarketData, tickers: list[str]):
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
def test_detects_invalid_tickers(md: MarketData, invalid_tickers: list[str]):
    """
    Test that invalid ticker symbols are correctly identified and the function returns None.
    """
    # Capture the return value and assert it is None ***
    result = md.fetch_returns(invalid_tickers, '2023-01-01', '2023-01-15')
    assert isinstance(result, pd.DataFrame)
    assert result.empty

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


def test_fetch_failed_download(md: MarketData, monkeypatch: pytest.MonkeyPatch):
    """Test that function returns an empty DataFrame if yf.download fails/returns None."""
    monkeypatch.setattr("yfinance.download", lambda *a, **k: None)

    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')

    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_fetch_empty_data(md: MarketData, monkeypatch: pytest.MonkeyPatch):
    """Test that function returns an empty DataFrame if no rows are found."""
    monkeypatch.setattr("yfinance.download", lambda *a, **k: pd.DataFrame())

    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')

    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_fetch_missing_columns(md: MarketData, monkeypatch: pytest.MonkeyPatch):
    """Test that function returns an empty DataFrame if 'Close' column is missing."""
    df = pd.DataFrame(
        {'Open': [1, 2]},
        index=pd.to_datetime(['2023-01-01', '2023-01-02'])
    )
    monkeypatch.setattr("yfinance.download", lambda *a, **k: df)

    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')

    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_fetch_invalid_dates(md: MarketData):
    """Test that impossible date ranges return an empty DataFrame."""
    data = md.fetch_returns(['AAPL'], '2026-01-01', '2025-01-01')

    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_fetch_returns_is_deterministic(md: MarketData):
    """
    Verify that fetch_returns produces identical DataFrames for the same input.
    """
    tickers = ['AAPL', 'MSFT']
    start = '2023-01-01'
    end = '2023-02-01'

    data1 = md.fetch_returns(tickers, start, end)
    data2 = md.fetch_returns(tickers, start, end)

    pd.testing.assert_frame_equal(data1, data2)


def test_column_order_matches_requested_tickers(md: MarketData):
    """
    Ensure that the order of columns in the returned DataFrame matches the
    order of the requested tickers.
    """
    tickers = ['MSFT', 'AAPL']
    data = md.fetch_returns(tickers, '2023-01-01', '2023-01-31')

    assert list(data.columns) == tickers


def test_zero_variance_ticker_handling(md: MarketData, monkeypatch):
    """
    Verify that tickers with zero price variance return zero daily returns.
    """
    dates = pd.date_range('2023-01-01', periods=5, freq='B')
    df = pd.DataFrame(
        {'Close': [100, 100, 100, 100, 100]},
        index=dates
    )
    monkeypatch.setattr("yfinance.download", lambda *a, **k: df)

    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-10')

    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert (data['AAPL'] == 0.0).all()


def test_single_trading_day_window_returns_empty(md: MarketData):
    """
    Confirm that a one-day date range returns an empty DataFrame.
    Since pct_change requires at least two data points, a single trading day
    cannot produce any returns and should yield an empty DataFrame with the
    correct columns.
    """
    data = md.fetch_returns(['AAPL'], '2023-01-03', '2023-01-03')

    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_returns_index_properties(md: MarketData):
    """
    Validate properties of the DataFrame index.
    Ensures that the index is a DatetimeIndex, monotonic increasing, and unique,
    which are essential properties for time series analysis and backtesting.
    """
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-03-01')

    assert data.index.is_monotonic_increasing
    assert data.index.is_unique
    assert isinstance(data.index, pd.DatetimeIndex)


def test_fetch_returns_invalid_date_string(md: MarketData):
    """Test that invalid start or end dates return empty DataFrame."""
    data = md.fetch_returns(['AAPL'], 'not-a-date', '2023-01-01')
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']

    data = md.fetch_returns(['AAPL'], '2023-01-01', None)
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_fetch_returns_yf_download_exception(md: MarketData, monkeypatch):
    """Test that yf.download exceptions return empty DataFrame."""
    # Simulate ValueError
    monkeypatch.setattr("yfinance.download", lambda *a, **
                        k: (_ for _ in ()).throw(ValueError("bad")))
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']

    # Simulate ConnectionError
    monkeypatch.setattr("yfinance.download", lambda *a, **
                        k: (_ for _ in ()).throw(ConnectionError("fail")))
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']

    # Simulate TimeoutError
    monkeypatch.setattr("yfinance.download", lambda *a, **
                        k: (_ for _ in ()).throw(TimeoutError("timeout")))
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']

    # Simulate OSError
    monkeypatch.setattr("yfinance.download", lambda *a, **
                        k: (_ for _ in ()).throw(OSError("os error")))
    data = md.fetch_returns(['AAPL'], '2023-01-01', '2023-01-05')
    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']


def test_fetch_returns_start_after_end(md: MarketData):
    """Test that a start date after the end date returns an empty DataFrame."""
    start_date = '2023-12-31'
    end_date = '2023-01-01'  # earlier than start_date

    data = md.fetch_returns(['AAPL'], start_date, end_date)

    assert isinstance(data, pd.DataFrame)
    assert data.empty
    assert list(data.columns) == ['AAPL']
