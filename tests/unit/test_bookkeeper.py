
# pylint: disable=redefined-outer-name

"""Unit tests for the BookKeeper class methods."""

import numpy as np
import pytest

from aequilibrium.bookkeeper import BookKeeper


@pytest.fixture
def bk():
    """
    Provide a fresh BookKeeper instance for each test.

    Ensures tests are isolated and do not share state.
    """
    return BookKeeper()


# -------------------------
# portfolio_value tests
# -------------------------
def test_portfolio_value_basic_sum(bk):
    """
    Portfolio value equals the sum of all holdings.

    Verifies the basic accounting identity v_t = 1^T h_t.
    """
    h = np.array([40000, 30000, 20000, 10000])
    assert bk.portfolio_value(h) == 100000


def test_portfolio_value_with_floats(bk):
    """
    Portfolio value is correctly computed for floating-point holdings.
    """
    h = np.array([1.5, 2.5, 3.0])
    assert bk.portfolio_value(h) == pytest.approx(7.0)


def test_portfolio_value_allows_negative_holdings(bk):
    """
    Portfolio value includes short positions (negative holdings).

    Shorts reduce total portfolio value via simple summation.
    """
    h = np.array([100.0, -40.0, 10.0])
    assert bk.portfolio_value(h) == pytest.approx(70.0)


def test_portfolio_value_empty_array(bk):
    """
    An empty holdings vector has zero portfolio value.

    This behavior follows NumPy's sum convention.
    """
    h = np.array([])
    assert bk.portfolio_value(h) == 0.0


# -------------------------
# compute_weights tests
# -------------------------
def test_compute_weights_matches_expected(bk):
    """
    Weights equal holdings divided by total portfolio value.

    Validates the example given in the method docstring.
    """
    h = np.array([40000, 30000, 20000, 10000], dtype=float)
    w = bk.compute_weights(h)
    np.testing.assert_allclose(w, np.array([0.4, 0.3, 0.2, 0.1]))


def test_compute_weights_sum_to_one_when_value_nonzero(bk):
    """
    Weights must sum to 1.0 when portfolio value is non-zero.

    This is a defining property of portfolio weights.
    """
    h = np.array([10.0, 20.0, -5.0, 5.0])  # sum = 30
    w = bk.compute_weights(h)
    assert w.sum() == pytest.approx(1.0)


def test_compute_weights_with_shorts(bk):
    """
    Weights correctly reflect long and short positions.

    Short holdings produce negative weights but the total still sums to 1.
    """
    h = np.array([120.0, -20.0, 0.0])  # sum = 100
    w = bk.compute_weights(h)
    np.testing.assert_allclose(w, np.array([1.2, -0.2, 0.0]))
    assert w.sum() == pytest.approx(1.0)


def test_compute_weights_zero_total_value_raises(bk):
    """
     Computing weights with zero portfolio value should raise an error.
     """
    h = np.array([10.0, -10.0])
    with pytest.raises(ValueError):
        bk.compute_weights(h)


# -------------------------
# compute_leverage tests
# -------------------------
def test_compute_leverage_excludes_cash_last_element(bk):
    """
    Cash must be excluded from leverage calculation.

    The last weight represents cash and should not contribute
    to portfolio leverage.
    """
    w = np.array([0.4, 0.3, 0.2, 0.1])
    assert bk.compute_leverage(w) == pytest.approx(0.9)


def test_compute_leverage_uses_absolute_values(bk):
    """
    Leverage must count short positions via absolute values.

    Negative weights increase leverage just like positive ones.
    """
    w = np.array([0.6, -0.3, 0.4, 0.3])  # cash is last element
    assert bk.compute_leverage(w) == pytest.approx(1.3)


def test_compute_leverage_long_only_fully_invested_cash_zero(bk):
    """
    A long-only, fully invested portfolio has leverage equal to 1.0.

    This is a key financial sanity check.
    """
    w = np.array([0.5, 0.5, 0.0])  # last element cash=0
    assert bk.compute_leverage(w) == pytest.approx(1.0)


def test_compute_leverage_only_cash_returns_zero(bk):
    """
    A portfolio holding only cash has zero leverage.

    With no risky assets, leverage must be zero.
    """
    w = np.array([1.0])  # only cash element
    assert bk.compute_leverage(w) == pytest.approx(0.0)


def test_compute_leverage_is_non_negative(bk):
    """
    Leverage is always non-negative by definition.

    The L1 norm of asset weights cannot be negative.
    """
    w = np.array([-1.0, 2.0, -3.0, 10.0])  # cash last
    assert bk.compute_leverage(w) >= 0.0


# -------------------------
# update_portfolio tests
# -------------------------
def test_update_portfolio_basic_adjustment(bk):
    """
    Portfolio holdings are updated correctly with return.

    Validates the basic addition of returns to current holdings.
    """
    current_holdings = np.array([40000, 30000, 20000, 10000])
    returns = np.array([0.02, 0.01, -0.01, 0.0])
    new_holdings = bk.update_portfolio(current_holdings, returns)
    np.testing.assert_allclose(
        new_holdings, np.array([40800, 30300, 19800, 10000]))


def test_update_portfolio_with_negative_returns(bk):
    """
    Portfolio holdings are updated correctly with negative returns.

    Ensures that losses are subtracted from current holdings.
    """
    current_holdings = np.array([50000, 25000, 15000, 10000])
    returns = np.array([-0.02, -0.01, -0.03, 0.0])
    new_holdings = bk.update_portfolio(current_holdings, returns)
    np.testing.assert_allclose(
        new_holdings, np.array([49000, 24750, 14550, 10000]))


def test_update_portfolio_zero_returns(bk):
    """
    Portfolio holdings remain unchanged with zero returns.

    Validates that zero returns do not affect holdings.
    """
    current_holdings = np.array([60000, 20000, 15000, 5000])
    returns = np.array([0.0, 0.0, 0.0, 0.0])
    new_holdings = bk.update_portfolio(current_holdings, returns)
    np.testing.assert_allclose(new_holdings, current_holdings)


def test_update_portfolio_with_short_positions(bk):
    """
    Portfolio holdings are updated correctly with short positions.
    Ensures that both long and short holdings are adjusted properly.
    """
    current_holdings = np.array([40000, -10000, 20000, 5000])
    returns = np.array([0.01, -0.02, 0.03, 0.0])
    new_holdings = bk.update_portfolio(current_holdings, returns)
    np.testing.assert_allclose(
        new_holdings, np.array([40400, -9800, 20600, 5000]))

# -------------------------
# portfolio_return tests
# -------------------------


def test_portfolio_return_basic_case(bk):
    """
    Portfolio return is calculated correctly for a basic case.

    Validates the formula R^p_t = (v_{t+1} - v_t) / v_t.
    """
    h_t = np.array([40000, 30000, 20000, 10000])
    h_t1 = np.array([40800, 30300, 19800, 10000])
    ret = bk.portfolio_return(h_t, h_t1)
    assert ret == pytest.approx(0.009)


def test_portfolio_return_with_negative_returns(bk):
    """
    Portfolio return is calculated correctly with negative returns.

    Ensures that losses are reflected in the portfolio return.
    """
    h_t = np.array([50000, 25000, 15000, 10000])
    h_t1 = np.array([49000, 24750, 14550, 10000])
    ret = bk.portfolio_return(h_t, h_t1)
    assert ret == pytest.approx(-0.017)


def test_portfolio_return_zero_change(bk):
    """ 
    Portfolio return is zero when holdings do not change.
    Validates that no change in holdings results in zero return.
    """
    h_t = np.array([60000, 20000, 15000, 5000])
    h_t1 = np.array([60000, 20000, 15000, 5000])
    ret = bk.portfolio_return(h_t, h_t1)
    assert ret == pytest.approx(0.0)


def test_portfolio_return_with_short_positions(bk):
    """
    Portfolio return is calculated correctly with short positions.
    Ensures that both long and short holdings affect the portfolio return.
    """
    h_t = np.array([40000, -10000, 20000, 5000])
    h_t1 = np.array([40400, -9800, 20600, 5000])
    ret = bk.portfolio_return(h_t, h_t1)
    assert ret == pytest.approx(0.02181818)


def test_portfolio_return_with_zero_portfolio_value(bk):
    """
    Portfolio return is NaN when portfolio value is zero.

    Validates that a zero portfolio value results in NaN return.
    """
    h_t = np.array([0, 0, 0, 0])
    h_t1 = np.array([0, 0, 0, 0])
    ret = bk.portfolio_return(h_t, h_t1)
    assert np.isnan(ret)
