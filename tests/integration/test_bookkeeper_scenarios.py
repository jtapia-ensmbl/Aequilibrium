# pylint: disable=redefined-outer-name

"""Integration tests for BookKeeper scenarios."""

import pytest
import numpy as np
from aequilibrium.bookkeeper import BookKeeper


@pytest.fixture
def bk():
    """
    Provide a fresh BookKeeper instance for each test.

    Ensures tests are isolated and do not share state.
    """
    return BookKeeper()


def test_multi_day_scenario(bk: BookKeeper):
    """
    Test a realistic multi-day portfolio scenario.
    This test simulates a portfolio over two days, checking portfolio value,
    weights, leverage, holdings update, and return calculation.
    """
    # Day 0 Data
    holdings_0 = np.array([40000.0, 30000.0, 20000.0, 10000.0])

    # Day 1 Returns
    returns_d1 = np.array([0.02, 0.01, -0.01, 0.00])

    # 1. Initial portfolio value
    initial_value = bk.portfolio_value(holdings_0)
    assert initial_value == 100000.0

    # 2. Initial weights
    initial_weights = bk.compute_weights(holdings_0)
    np.testing.assert_allclose(initial_weights, np.array([0.4, 0.3, 0.2, 0.1]))

    # 3. Initial leverage
    initial_leverage = bk.compute_leverage(initial_weights)
    assert initial_leverage == pytest.approx(0.9)

    # 4. Holdings after Day 1
    holdings_1 = bk.update_portfolio(holdings_0, returns_d1)
    np.testing.assert_allclose(holdings_1, np.array(
        [40800.0, 30300.0, 19800.0, 10000.0]))

    # 5. Portfolio return for Day 1
    day1_return = bk.portfolio_return(holdings_0, holdings_1)
    assert pytest.approx(day1_return) == 0.009

    # 6. New weights and New leverage
    new_weights = bk.compute_weights(holdings_1)
    new_leverage = bk.compute_leverage(new_weights)

    np.testing.assert_almost_equal(new_weights, np.array(
        [0.404360, 0.300297, 0.196233, 0.099108]), decimal=5)
    assert new_leverage == pytest.approx(0.900883, abs=1e-5)


def test_single_asset_portfolio(bk: BookKeeper):
    """Test portfolio with only one non-cash asset."""
    # Portfolio: $80k in one stock, $20k cash
    holdings = np.array([80000.0, 20000.0])
    returns = np.array([0.05, 0.0])  # Stock up 5%, cash 0%

    # The leverage should be 0.8
    weights = bk.compute_weights(holdings)
    leverage = bk.compute_leverage(weights)
    assert leverage == pytest.approx(0.8, abs=1e-5)

    # Update holdings
    new_holdings = bk.update_portfolio(holdings, returns)
    np.testing.assert_allclose(new_holdings, np.array([84000.0, 20000.0]))

    # Portfolio return
    port_return = bk.portfolio_return(holdings, new_holdings)
    assert port_return == pytest.approx(0.04, abs=1e-5)  # 4% return overall


def test_all_cash_portfolio(bk: BookKeeper):
    """Test portfolio with no asset exposure (all cash)."""
    holdings = np.array([100000.0])  # Just cash
    returns = np.array([0.0])

    # calculate the leverage
    weights = bk.compute_weights(holdings)
    leverage = bk.compute_leverage(weights)
    assert leverage == pytest.approx(0.0)

    # Update holdings
    new_holdings = bk.update_portfolio(holdings, returns)
    np.testing.assert_allclose(new_holdings, np.array([100000.0]))

    # Portfolio return
    port_return = bk.portfolio_return(holdings, new_holdings)
    assert port_return == pytest.approx(0.0)


def test_short_positions(bk: BookKeeper):
    """Test portfolio with short positions."""
    # Long $60k, Short $20k, Cash $60k
    holdings = np.array([60000.0, -20000.0, 60000.0])

    # expected leverage is |0.6| + |-0.2| = 0.8
    weights = bk.compute_weights(holdings)
    leverage = bk.compute_leverage(weights)
    assert leverage == pytest.approx(0.8, abs=1e-5)

    # If shorted asset goes UP 10%, what happens to holdings?
    returns = np.array([0.0, 0.10, 0.0])  # Cash 0%, Shorted asset +10%
    new_holdings = bk.update_portfolio(holdings, returns)
    np.testing.assert_allclose(
        new_holdings, np.array([60000.0, -22000.0, 60000.0]))

    # Portfolio return
    port_return = bk.portfolio_return(holdings, new_holdings)
    # Return = (98000 - 100000) / 100000 = -0.02
    assert port_return == pytest.approx(-0.02, abs=1e-5)


def test_zero_portfolio_value(bk: BookKeeper):
    """Test edge case of zero portfolio value."""
    holdings_zero = np.array([0.0, 0.0, 0.0])

    # Should return np.nan for portfolio_return with zero initial value
    holdings_next = np.array([100.0, 100.0, 100.0])
    ret = bk.portfolio_return(holdings_zero, holdings_next)
    assert np.isnan(ret)


def test_negative_portfolio_value(bk: BookKeeper):
    """Test portfolio that goes negative (bankruptcy scenario)."""
    # Highly leveraged short position
    holdings = np.array([50000.0, -100000.0, 30000.0])

    # We allow negative portfolio values
    vt = bk.portfolio_value(holdings)
    assert vt == -20000

    # Compute_weights should stillwork?
    wt = bk.compute_weights(holdings)
    np.testing.assert_allclose(wt, np.array([-2.5, 5.0, -1.5]))
