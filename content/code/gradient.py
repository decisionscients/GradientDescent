# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
from abc import ABC, abstractmethod
import datetime
import math
import numpy as np
from numpy import array, newaxis
from numpy.random import RandomState
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from itertools import zip_longest
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
#                        Gradient Search Base Class                           #
# --------------------------------------------------------------------------- #


class GradientDescent(ABC, BaseEstimator, RegressorMixin):
    '''Abstract base class for Gradient Descent

    GradientDescent is the abstract base class for the Batch Gradient 
    Descent (BGD), Stochastic Gradient Descent (SGD), and Minibatch Gradient
    Descent (MBGD) subclasses.
    '''

    def __init__(self, theta_init=None, learning_rate=0.01,  learning_rate_sched='c',  
                decay_rate=0, step_epochs=10, maxiter=2500, early_stop=True,  
                precision=0.01, epochs_stable=5):

        self.alg = "Gradient Descent Base Class"
        
        # Parameters
        self._theta_init = theta_init
        self._learning_rate = learning_rate
        self._learning_rate_sched = learning_rate_sched
        self._early_stop = early_stop
        self._precision = precision
        self._decay_rate = decay_rate
        self._step_epochs = step_epochs
        self._maxiter = maxiter
        self._epochs_stable = epochs_stable
        
        # State variables
        self._iter_no_improvement = 0
        self._best_error= 10**10

        # Logs
        self._epochs_history = []
        self._iterations_history = []    
        self._theta_history = [] 
        self._learning_rate_history = []
        self._J_history = []
        
        # Attribute
        self.theta=None

        # Timing
        self._start = None
        self._end =None     

    def get_params(self, deep=True):
        return{'theta_init' : self._theta_init,
               'learning_rate': self._learning_rate,
               'learning_rate_sched': self._learning_rate_sched, 
               'early_stop': self._early_stop,
               'precision': self._precision,
               'decay_rate': self._decay_rate,
               'step_epochs': self._step_epochs,
               'maxiter': self._maxiter,
               'epochs_stable': self._epochs_stable}

    def set_params(self, **parameters):
        self._theta_init = parameters.get('theta_init')
        self._learning_rate = parameters.get('learning_rate', 0.001)
        self._learning_rate_sched = parameters.get('learning_rate_sched', 'c')
        self._early_stop = parameters.get('early_stop', True)
        self._decay_rate = parameters.get('decay_rate', 0)
        self._step_epochs = parameters.get('step_epochs', 10)
        self._precision = parameters.get('precision', 0.001)
        self._maxiter = parameters.get('maxiter', 5000)
        self._epochs_stable = parameters.get('epochs_stable', 5)
        return self         

    def _get_params(self):
        params = pd.DataFrame({'alg': self.alg,                  
                  'learning_rate_sched': self._learning_rate_sched,
                  'learning_rate_sched_label': self._get_label(self._learning_rate_sched),                  
                  'learning_rate': self._learning_rate,
                  'early_stop': self._early_stop,
                  'precision': self._precision,
                  "decay_rate": self._decay_rate,
                  "step_epochs": self._step_epochs,
                  "maxiter": self._maxiter,
                  "precision": self._precision,
                  "epochs_stable": self._epochs_stable}, index=[0])
        thetas = self._get_thetas(initial=True)
        params = pd.concat([params, thetas], axis=1)                  
        return(params)

    def _get_thetas(self, initial=False, final=False):

        if initial:
            thetas = self._todf(self._theta_history, stub='theta_init_')
            thetas=thetas.head(1)
        elif final:
            thetas = self._todf(self._theta_history, stub='theta_final_')
            thetas=thetas.tail(1)
        else:
            thetas = self._todf(self._theta_history, stub='theta_')
        return(thetas)

    def _get_detail(self):
        detail = {'alg': self.alg,
                  'epoch': self._epochs_history,
                  'iteration': self._iterations_history,
                  'learning_rate': self._learning_rate_history,
                  'cost': self._J_history}
        detail = pd.DataFrame.from_dict(detail)      
        thetas = self._get_thetas()
        detail = pd.concat([detail, thetas], axis=1)        
        return(detail)

    def _get_summary(self):        
        summary = pd.DataFrame({'start' : self._start,
                                'end' : self._end,
                                'duration':(self._end-self._start).total_seconds(),                   
                                'epochs': self._epochs_history[-1],
                                'iterations': self._iterations_history[-1],
                                'initial_cost': self._J_history[0],
                                'final_cost': self._J_history[-1]}, index=[0])
        params = self._get_params()
        thetas = self._get_thetas(final=True)
        thetas = thetas.reset_index(drop=True)
        summary = pd.concat([params, summary, thetas], axis=1)        
        return(summary)

    def results(self):
        if self.theta is None:
            raise Exception("Model has not been fit. No training errors to report.")        
        summary = self._get_summary()
        detail = self._get_detail()        
        return({'summary': summary, 'detail': detail})                

    def _get_label(self, x):
        labels = {'c': 'Constant Learning Rate',
                  't': 'Time Decay Learning Rate',
                  's': 'Step Decay Learning Rate',
                  'e': 'Exponential Decay Learning Rate'}
        return(labels.get(x,x))        

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df)        

    def _init_thetas(self, X):
        seed = RandomState(50)
        self.theta = None
        if self._theta_init is None:
            theta = seed.normal(size=X.shape[1])
        else:
            theta = self._theta_init
        return(theta)   

    def _split_data(self, X, y):
        if self._early_stop:
            X, X_dev, y, y_dev = train_test_split(X,y, test_size=0.3,
                                              random_state=50)  
        else:
            X_dev = y_dev = None         
        return(X, X_dev, y, y_dev)

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e=None, **kwargs):
        if e is not None:
            return(1/2 * np.mean(e**2))
        else:
            X = kwargs.get('X', None)
            y = kwargs.get('y', None)
            theta = kwargs.get('theta', None)
            h = self._hypothesis(X, theta)
            e = self._error(h, y)
            return(1/2 * np.mean(e**2))

    def _rmse(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h, y)
        return(np.sqrt(np.mean(e**2)))
    
    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, theta, learning_rate, gradient):
        return(theta-(learning_rate * gradient))

    def _update_learning_rate(self, learning_rate, epoch):
        
        if self._learning_rate_sched == 'c':
            learning_rate_new = learning_rate
        elif self._learning_rate_sched == 't':
            k = self._decay_rate
            learning_rate_new = self._learning_rate/(1+k*epoch)            
        elif self._learning_rate_sched == 's':
            drop = self._decay_rate
            epochs_drop = self._step_epochs
            learning_rate_new = self._learning_rate*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        elif self._learning_rate_sched == 'e':            
            k = self._decay_rate
            learning_rate_new = self._learning_rate * math.exp(-k*epoch)

        return(learning_rate_new)

    def _improvement(self, E):

        if E <= self._best_error:
            if abs(self._best_error - E)/self._best_error > self._precision:                       
                self._iter_no_improvement = 0
                self._best_error = E 
                return(True)
            self._iter_no_improvement += 1
        else:
            self._iter_no_improvement += 1
            return(False)

    def _finished(self, X, y, theta, epoch):

        if self._early_stop:
            E = self._rmse(X, y, theta)
            if self._improvement(E):                
                self.theta = theta
            elif self._iter_no_improvement == self._epochs_stable:
                return(True)
    
        self.theta = theta
        if self._maxiter:
            if epoch > self._maxiter:
                return(True)       
        return(False)

    def _init_history(self):
        self._epochs_history = []
        self._iterations_history = []    
        self._theta_history = [] 
        self._learning_rate_history = []
        self._J_history = []                    

    def _update_history(self, J, theta, epoch, iteration, learning_rate):

        self._learning_rate_history.append(learning_rate)
        self._J_history.append(J)            
        self._theta_history.append(theta)            
        self._iterations_history.append(iteration)
        self._epochs_history.append(epoch)

    def rmse(self, y_truth, y_predicted):
        e = y_truth - y_predicted
        return(np.sqrt(np.mean(e**2)))

    def predict(self, X):      
        return(X.dot(self.theta))        

    @abstractmethod
    def fit(self, X, y):
        pass

# --------------------------------------------------------------------------- #
#                           Batch Gradient Descent                            #
# --------------------------------------------------------------------------- #            
class BGD(GradientDescent, BaseEstimator, RegressorMixin):
    '''Batch Gradient Descent'''

    def __init__(self, *args, **kwargs):        
        super(BGD, self).__init__(*args, **kwargs)
        self.alg = "Batch Gradient Descent"

    def fit(self,  X, y): 

        self._start = datetime.datetime.now()

        # Initialize iterations, logs, thetas, lerning rate and development sets
        epoch = iteration = 1
        self._init_history()
        theta = self._init_thetas(X)
        learning_rate = self._learning_rate
        X, X_dev, y, y_dev = self._split_data(X, y)

        while not self._finished(X_dev, y_dev, theta, epoch):

            # Compute costs and update history 
            h = self._hypothesis(X, theta)
            e = self._error(h, y)
            J = self._cost(e=e)
            
            self._update_history(J, theta, epoch, iteration, learning_rate)      

            # Compute gradient and update thetas
            g = self._gradient(X, e)
            theta = self._update(theta, learning_rate, g)

            # Update learning rate 
            learning_rate = self._update_learning_rate(learning_rate, epoch)

            iteration += 1
            epoch += 1

        self._end = datetime.datetime.now()
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self

# --------------------------------------------------------------------------- #
#                      Stochastic Gradient Descent                            #
# --------------------------------------------------------------------------- #            
class SGD(GradientDescent, BaseEstimator, RegressorMixin):
    '''Stochastic Gradient Descent'''

    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        self.alg = "Stochastic Gradient Descent"

    def fit(self,  X, y): 

        self._start = datetime.datetime.now()

        # Initialize iterations, logs, thetas, lerning rate and development sets
        epoch = iteration = 1
        self._init_history()
        theta = self._init_thetas(X)
        learning_rate = self._learning_rate
        X, X_dev, y, y_dev = self._split_data(X, y)

        while not self._finished(X_dev, y_dev, theta, epoch):

            # Compute costs on training set, then update history 
            J = self._cost(X=X, y=y, theta=theta)
            self._update_history(J, theta, epoch, iteration, learning_rate)    

            X, y = shuffle(X, y)

            for x_i, y_i in zip(X,y):              
                # Compute costs, gradient and perform update
                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                g = self._gradient(x_i, e)
                theta = self._update(theta, learning_rate, g)
                iteration += 1
            
            learning_rate = self._update_learning_rate(learning_rate, epoch)

            epoch += 1        
            
        self._end = datetime.datetime.now()
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self

# --------------------------------------------------------------------------- #
#                      MINI-BATCH Gradient Descent                            #
# --------------------------------------------------------------------------- #            
class MBGD(GradientDescent, BaseEstimator, RegressorMixin):
    '''Minibatch Gradient Descent'''

    def __init__(self, batch_size=32, *args, **kwargs):
        super(MBGD, self).__init__(*args, **kwargs)
        self.alg = "Minibatch Gradient Descent"
        self._batch_size = batch_size

    def get_params(self, deep=True):
        return{'learning_rate': self._learning_rate,
               'learning_rate_sched': self._learning_rate_sched, 
               'batch_size': self._batch_size,
               'precision': self._precision,
               'decay_rate': self._decay_rate,
               'step_epochs': self._step_epochs,
               'maxiter': self._maxiter,
               'epochs_stable': self._epochs_stable}

    def set_params(self, **parameters):
        self._learning_rate = parameters.get('learning_rate', 0.001)
        self._learning_rate_sched = parameters.get('learning_rate_sched', 'c')
        self._batch_size = parameters.get('batch_size', 32)
        self._decay_rate = parameters.get('decay_rate', 0)
        self._step_epochs = parameters.get('steps_epochs', 10)
        self._precision = parameters.get('precision', 0.001)
        self._maxiter = parameters.get('maxiter', 5000)
        self._epochs_stable = parameters.get('epochs_stable', 5)
        return self         

    def _get_params(self):
        params = pd.DataFrame({'alg': self.alg,
                  'learning_rate_sched': self._learning_rate_sched,
                  'learning_rate_sched_label': self._get_label(self._learning_rate_sched),                  
                  'learning_rate': self._learning_rate,
                  'batch_size': self._batch_size,
                  'precision': self._precision,
                  "decay_rate": self._decay_rate,
                  "step_epochs": self._step_epochs,
                  "maxiter": self._maxiter,
                  "precision": self._precision,
                  "epochs_stable": self._epochs_stable}, index=[0])
        thetas = self._get_thetas(initial=True)
        params = pd.concat([params, thetas], axis=1)
        return(params)        

    def _batch(self, x, n):
        for i in range(0, len(x), n):
            yield x[i:i+n]     

    def fit(self,  X, y): 

        self._start = datetime.datetime.now()

        # Initialize iterations, logs, thetas, lerning rate and development sets
        epoch = iteration = 1
        self._init_history()
        theta = self._init_thetas(X)
        learning_rate = self._learning_rate
        X, X_dev, y, y_dev = self._split_data(X, y)

        while not self._finished(X_dev, y_dev, theta, epoch):

            # Update history
            J = self._cost(X=X, y=y, theta=theta)
            self._update_history(J, theta, epoch, iteration, learning_rate) 

            # Shuffle data and creaet batches 
            X, y = shuffle(X, y)
            X_batches = self._batch(X, n=self._batch_size)
            y_batches = self._batch(y, n=self._batch_size)

            for x_mb, y_mb in zip(X_batches,y_batches):

                # Compute costs, and error on development set
                h = self._hypothesis(x_mb, theta)
                e = self._error(h, y_mb)
                g = self._gradient(x_mb, e)
                theta = self._update(theta, learning_rate, g)
                iteration += 1

            learning_rate = self._update_learning_rate(learning_rate, epoch)

            epoch += 1   

        self._end = datetime.datetime.now()
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self

