# =========================================================================== #
#                            TEST METRICS                                     #
# =========================================================================== #
#%%%
"""Tests for Metrics classes."""
import numpy as np
import pytest
from pytest import mark
from gradient_descent.v1_linear_regression import metrics
from gradient_descent.v1_linear_regression.metrics import RegressionMetricFactory 
 
class MetricsTests:

    @mark.metrics
    def test_metrics_mae(self, get_metrics_test_data):
        y, y_pred = get_metrics_test_data
        scorer = RegressionMetricFactory()(metric='mean_absolute_error')
        score = scorer(y, y_pred)        
        assert np.isclose(score,22.2,rtol=1e-2), "MAE not close"

    @mark.metrics
    def test_metrics_mse(self, get_metrics_test_data):
        y, y_pred = get_metrics_test_data
        scorer = RegressionMetricFactory()(metric='mean_squared_error')
        score = scorer(y, y_pred)        
        assert np.isclose(score,705.4,rtol=1e-2), "MSE not close"        

    @mark.metrics
    def test_metrics_rmse(self, get_metrics_test_data):
        y, y_pred = get_metrics_test_data
        scorer = RegressionMetricFactory()(metric='root_mean_squared_error')
        score = scorer(y, y_pred)        
        assert np.isclose(score,26.55937,rtol=1e-2), "RMSE not close"        


# %%
