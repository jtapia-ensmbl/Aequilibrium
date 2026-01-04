import numpy as np


class BookKeeper:
    """
    BookKeeper for portfolio accounting.
    Provides methods to compute portfolio value, weights, leverage,
    update holdings with returns, and calculate portfolio returns.
    """

    def portfolio_value(self, holdings) -> float:
        """
        Calculate total portfolio value.

        Parameters
        ----------
        holdings : np.array, shape (n+1,)
            Dollar holdings in n assets plus cash

        Returns
        -------
        float
            Total portfolio value v_t = 1^T h_t
        """
        vt = holdings.sum()
        return vt

    def compute_weights(self, holdings) -> np.array:
        """
        Convert holdings to weights.

        Parameters
        ----------
        holdings : np.array, shape (n+1,)
            Dollar holdings

        Returns
        -------
        np.array, shape (n+1,)
            Weights w_t = h_t / v_t

        Notes
        -----
        - Weights must sum to 1.0
        - Use portfolio_value() function
        """
        vt = self.portfolio_value(holdings)
        if vt == 0:
            raise ValueError(
                "Cannot compute weights: portfolio value is zero."
            )
        wt = holdings/vt
        return wt

    def compute_leverage(self, weights) -> float:
        """
        Calculate portfolio leverage.

        Parameters
        ----------
        weights : np.array, shape (n+1,)
            Portfolio weights

        Returns
        -------
        float
            Leverage ||w_{1:n}||_1

        Notes
        -----
        - Leverage is sum of absolute values of ASSET weights
        - Cash (last element) is excluded
        - Long-only fully-invested portfolio has leverage = 1.0
        """
        leverage = np.abs(weights[:-1]).sum()
        return leverage

    def update_portfolio(self, holdings, returns):
        """
        Apply one period of returns to holdings (no trading).

        Parameters
        ----------
        holdings : np.array, shape (n+1,)
            Current holdings h_t
        returns : np.array, shape (n+1,)
            Returns for the period r_t

        Returns
        -------
        np.array, shape (n+1,)
            Next period holdings h_{t+1}

        Formula
        -------
        h_{t+1} = (1 + r_t) ⊙ h_t
        """
        new_holdings = holdings * (1 + returns)
        return new_holdings

    def portfolio_return(self, holdings_t, holdings_t_plus_1) -> float:
        """
        Calculate realized portfolio return for the period.

        Parameters
        ----------
        holdings_t : np.array, shape (n+1,)
            Holdings at start of period
        holdings_t_plus_1 : np.array, shape (n+1,)
            Holdings at end of period

        Returns
        -------
        float
            Portfolio return R^p_t = (v_{t+1} - v_t) / v_t
        """
        v_t = self.portfolio_value(holdings_t)
        v_t_plus_1 = self.portfolio_value(holdings_t_plus_1)
        if v_t == 0:
            return np.nan
        return (v_t_plus_1 - v_t) / v_t
