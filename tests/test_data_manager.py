# =========================================================================== #
#                              TEST COST                                      #
# =========================================================================== #
#%%%
"""Tests for Data Management classes and functions."""
import numpy as np
import pytest
from pytest import mark
from gradient_descent.v1_linear_regression.data_manager import StandardScaler
from gradient_descent.v1_linear_regression.data_manager import shuffle_data
from gradient_descent.v1_linear_regression.data_manager import data_split
from gradient_descent.v1_linear_regression.data_manager import batch_iterator
 
class DataManagerTests:

    @mark.scaler
    def test_scaler(self):
        X = np.random.random_integers(10,1000, size=(100,10))
        scaler = StandardScaler()
        scaler.fit(X)
        X_new = scaler.transform(X)
        X_mean = np.mean(X_new, axis=0)
        X_var = np.var(X_new, axis=0)
        print(X_mean)
        print(X_var)        
        assert X_mean.shape == (10,), "X_mean shape not correct."
        assert X_var.shape == (10,), "X_var shape not correct."
        assert np.allclose(np.ones(shape=(10,)),X_var), "X not scaled." 
        assert np.allclose(np.zeros(shape=(10,)),X_mean), "X not centered."    

    @mark.shuffle
    def test_shuffle_data(self):
        X = np.random.random_integers(10,100, size=(10,2))
        y = np.random.random_integers(10,100, size=(10,))
        X_new, y_new = shuffle_data(X,y)
        assert not np.array_equal(X, X_new), "Data not shuffled."
        assert not np.array_equal(y, y_new), "Data not shuffled."

    @mark.split
    def test_split_data_wo_shuffle(self):
        X,y = np.arange(20).reshape((10,2)), np.arange(10)        
        X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.3,shuffle=False)        
        assert X_train.shape == (7,2), "X_train shape not correct."
        assert X_test.shape == (3,2), "X_test shape not correct."        
        assert y_train.shape == (7,), "y_new shape not correct."
        assert y_test.shape == (3,), "y_new shape not correct."
        assert np.allclose(X[:7,:2],X_train), "X_train doesn't match same portion of X"
        assert np.allclose(X[7:,:2],X_test), "X_test doesn't match same portion of X"
        assert np.allclose(y[:7],y_train), "y_train doesn't match same portion of y"
        assert np.allclose(y[7:],y_test), "y_test doesn't match same portion of y"

    @mark.split
    def test_split_data_w_shuffle(self):
        X,y = np.arange(20).reshape((10,2)), np.arange(10)        
        X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.3,shuffle=True)        
        assert X_train.shape == (7,2), "X_train shape not correct."
        assert X_test.shape == (3,2), "X_test shape not correct."        
        assert y_train.shape == (7,), "y_new shape not correct."
        assert y_test.shape == (3,), "y_new shape not correct."
        assert not np.allclose(X[:7,:2],X_train), "X_train doesn't match same portion of X"
        assert not np.allclose(X[7:,:2],X_test), "X_test doesn't match same portion of X"
        assert not np.allclose(y[:7],y_train), "y_train doesn't match same portion of y"
        assert not np.allclose(y[7:],y_test), "y_test doesn't match same portion of y"

    @mark.split
    def test_split_data_w_stratify(self):
        classes = np.arange(5)
        X,y = np.arange(200).reshape((100,2)), np.repeat(classes,20)
        X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.3,stratify=True)  
        for c in classes:
            assert np.sum(y_train == c) == len(y)/5*.7, "y_train not stratifying class %d " % c
            assert np.sum(y_test == c) == len(y)/5*.3, "y_test not stratifying class %d " % c
            assert not np.allclose(X[:70,:2],X_train), "X_train doesn't match same portion of X"
            assert not np.allclose(X[70:,:2],X_test), "X_test doesn't match same portion of X"


    @mark.batch_iterator
    def test_batch_iterator(self):
        X,y = np.arange(20).reshape((10,2)), np.arange(10)        
        batch=0
        for X_batch, y_batch in batch_iterator(X, y, batch_size=3):            
            if batch < 3:
                assert X_batch.shape == (3,2), "X_batch shape incorrect"
                assert y_batch.shape == (3,), "y_batch shape incorrect"
            else:
                assert X_batch.shape == (1,2), "X_batch shape incorrect"
                assert y_batch.shape == (1,), "y_batch shape incorrect"
            batch += 1



