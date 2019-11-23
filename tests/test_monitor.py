# =========================================================================== #
#                              TEST MONITOR                                   #
# =========================================================================== #
#%%%
"""Tests for the History and Progress classes."""
import datetime
import numpy as np
import pytest
from pytest import mark
from gradient_descent.v1_linear_regression.monitor import History
 
class HistoryTests:

    @mark.history
    def test_history(self, get_batch_data, get_epoch_data):        
        # Begin training
        history = History()
        history.on_train_begin()
        assert history.total_epochs == 0, "History epoch not initialized in on_train_begin"
        assert history.total_batches == 0, "History batches not initialized in on_train_begin"
        assert isinstance(history.start, datetime.datetime), "History batches not initialized in on_train_begin"
        assert len(history.epoch_log) == 0, "History epoch log length not zero in on_train_begin"
        assert len(history.batch_log) == 0, "History epoch log length not zero in on_train_begin"

        # Get Batch Data        
        batch_data = get_batch_data
        batch = batch_data['BATCH']
        cost = batch_data['COST']
        theta_batch = batch_data[['theta0', 'theta1']].values
        batch_size = batch_data['batch_size']
        # Get Epoch Data
        epoch_data = get_epoch_data
        epoch = epoch_data['EPOCH']        
        learning_rate = epoch_data['LEARNING_RATE']
        theta_epoch = epoch_data[['THETA0', 'THETA1']].values
        train_cost = epoch_data['TRAIN_COST']
        train_score = epoch_data['TRAIN_SCORE']
        # Iterate through epochs
        for e in np.arange(len(epoch)):            
            epoch_log = {}
            epoch_log['learning_rate']=learning_rate[e]
            epoch_log['theta'] = np.atleast_1d(theta_epoch[e])
            epoch_log['train_cost'] = train_cost[e]
            epoch_log['train_score'] = train_score[e]
            history.on_epoch_end(epoch=epoch[e], logs=epoch_log)
        for b in np.arange(len(batch)):
            batch_log = {}
            batch_log['cost'] = cost[b]
            batch_log['theta'] = theta_batch[b]
            batch_log['batch_size'] = batch_size[b]
            history.on_batch_end(b+1, logs=batch_log)

        # End training
        history.on_train_end()
        # Evaluation
        assert history.total_epochs == 10, "Total epochs incorrect"
        assert history.total_batches == 100, "Total batches incorrect"
        assert isinstance(history.end, datetime.datetime), "End time is not datetime object"
        assert isinstance(history.duration, float), "Duration not an float"
        assert len(history.epoch_log['learning_rate']) == 10, "Epoch log is corrupt"
        assert len(history.epoch_log['theta']) == 10, "Theta is corrupt"
        assert len(history.epoch_log['train_cost']) == 10, "Train cost is corrupt"
        assert len(history.epoch_log['train_score']) == 10, "Epoch log is corrupt"
        assert isinstance(history.epoch_log['learning_rate'][0],float), "Epoch learning rate is not float"
        assert isinstance(history.epoch_log['theta'][0][0],float), "Theta is not  float"
        assert len(history.batch_log['cost']) == 100, "Batch log is not correct length"
        assert len(history.batch_log['theta']) == 100, "Batch theta not correct"
        assert len(history.batch_log['batch_size']) == 100, "Batch size not correct type"
