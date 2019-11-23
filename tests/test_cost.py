# =========================================================================== #
#                              TEST COST                                      #
# =========================================================================== #
#%%%
"""Tests for Cost classes."""
import numpy as np
import pytest
from pytest import mark
from gradient_descent.v1_linear_regression.cost import RegressionCostFactory
 
class CostTests:

    @mark.cost
    def test_cost(self, get_cost_test_data):
        df = get_cost_test_data
        X = df[['X0', 'X1']].values
        y = df['Y']
        y_pred = df['YPRED']
        thetas = np.atleast_2d(np.mean(df[['Theta0', 'Theta1']]).values).reshape(-1,1)
        cost_func = RegressionCostFactory()(cost='quadratic')
        cost = cost_func(y, y_pred)
        gradient = cost_func.gradient(X, y, y_pred)
        assert np.isclose(cost,686.35,rtol=1e-2), "Cost is not close"
        assert np.allclose(np.array(gradient), np.array(thetas),1e-2), "Gradient not correct."


# %%
