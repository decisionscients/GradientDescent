# =========================================================================== #
#                            PYTEST FIXTURES                                  #
# =========================================================================== #
import numpy as np
import pandas as pd
from pytest import fixture
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
# --------------------------------------------------------------------------- #
@fixture(scope='session')
def get_metrics_test_data():
    y = np.array([17,1,63,33,67,43,31,76,2,77])
    yhat = np.array([43,33,37,22,18,58,11,77,5,38])
    return y, yhat

@fixture(scope='session')
def get_cost_test_data():
    filename = "tests/test_cost_data.xlsx"
    df = pd.read_excel(io=filename, sheet_name='cost',
                       usecols=[0,1,2,3,4,5,6])    
    return df
    
@fixture(scope='session')
def get_batch_data():
    filename = "tests/test_monitor.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[0,1,2,3,4])    
    return df

@fixture(scope='session')
def get_epoch_data():
    filename = "tests/test_monitor.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',nrows=10,
                       usecols=[6,7,8,9,10,11])    
    return df    

