# =========================================================================== #
#                             GRADIENT SEARCH                                 #
# =========================================================================== #

# --------------------------------------------------------------------------- #
import numpy as np
from numpy import array
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from gradient import BGD
from filemanager import save_fig, save_csv

# --------------------------------------------------------------------------- #
class GradientSearch:
    '''
    Gradient Search
    '''

    def __init__(self, algorithm):
        self._algorithm = algorithm
        self._gd = None
        self._summary = None
        self._train = None
        self._validation = None
        # Parameters
        self._theta = []
        self._learning_rate_init = []
        self._learning_rate_sched = []
        self._precision = []
        self._decay_rate = []
        self._step_epochs = []
        self._maxiter = None
        self._i_s = []
        self._batch_size = []
        self._compute_scores = False
        self._X_val = None
        self._y_val = None
            
    def summary(self, nbest=0, directory=None, filename=None):
        if self._summary is None:
            raise Exception("No summary to report")
        else:
            if directory is not None:
                if filename is None:
                    filename = self._gd.alg + ' Search Summary.csv'
                save_csv(self._summary, directory, filename) 
            if nbest:
                t = self._summary.sort_values(by=['final_train_error', 'duration'])
                v = self._summary.sort_values(by=['final_validation_error', 'duration'])
                return(t.head(nbest), v.head(nbest))
            return(self._summary)
         

    def train(self, nbest=0, directory=None, filename=None):
        if self._train is None:
            raise Exception("No train to report")
        else:
            if directory is not None:
                if filename is None:
                    filename = self._gd.alg + ' Training Set.csv'
                save_csv(self._train, directory, filename)             
            if nbest:
                s = self.summary(nbest=nbest)
                d = self._train
                d = d.loc[d['experiment'].isin(s['experiment'])]
                return(d)
            return(self._train)    

    def validation(self, nbest=0, directory=None, filename=None):
        if self._validation is None:
            raise Exception("No evaluation to report")
        else:
            if directory is not None:
                if filename is None:
                    filename = self._gd.alg + ' Validation Set.csv'
                save_csv(self._train, directory, filename)             
            if nbest:
                s = self.summary(nbest=nbest)
                d = self._validation
                d = d.loc[d['experiment'].isin(s['experiment'])]
                return(d)
            return(self._validation)                

    def _set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter,value)
        return self            

    def scores(self):
        if self._compute_scores:
            if self._scores.shape[0] > 0:
                return(self._scores)
        raise Exception("No scores to report.")

    def _score(self, X,y,theta):
        e = X.dot(theta) - y
        return(np.mean(e**2))

    def gridsearch(self, X, y, **parameters):
        experiment = 1
        self._summary = pd.DataFrame()
        self._validation = pd.DataFrame()
        self._train = pd.DataFrame()
        self._scores = pd.DataFrame()

        # Unpack parameters
        self._theta = parameters.get('theta', None)
        self._learning_rate_init = parameters.get('learning_rate', [0.01])
        self._learning_rate_sched = parameters.get('learning_rate_sched', ['c'])
        self._decay_rate = parameters.get('decay_rate', [0])
        self._step_epochs = parameters.get('step_epochs', [10])
        self._precision = parameters.get('precision', [0.001])
        self._maxiter = parameters.get('maxiter', 5000)
        self._i_s = parameters.get('i_s', [5])
        self._compute_scores = parameters.get('compute_scores', False)
        self._X_val = parameters.get('X_val', None)
        self._y_val = parameters.get('y_val', None)

        # Constant learning rates
        if 'c' in self._learning_rate_sched:            
            for n in self._i_s:
                for p in self._precision:
                    for a in self._learning_rate_init:  
                        self._gd = self._algorithm(theta=self._theta, learning_rate=a, 
                                learning_rate_sched='c', maxiter=self._maxiter, 
                                precision=p, i_s=n,)
                        self._gd.fit(X,y)

                        results = self._gd.results()
                        summary = results['summary']
                        summary['experiment'] = experiment
                        train = results['train_errors']
                        train['experiment'] = experiment
                        evals = results['validation_errors']
                        evals['experiment'] = experiment

                        self._summary = pd.concat([self._summary, summary], axis=0)    
                        self._train = pd.concat([self._train, train], axis=0)    
                        self._validation = pd.concat([self._validation, evals], axis=0)                           

                        if self._compute_scores:                            
                            train_score = pd.DataFrame({'score_type': 'Training Set Score',
                                                        'score': self._score(X,y, self._gd.theta)},
                                                        index=[0])
                            val_score = pd.DataFrame({'score_type': 'Validation Set Score',
                                                      'score': self._score(self._X_val, self._y_val, self._gd.theta)},
                                                      index=[0])
                            scores = pd.concat([train_score, val_score], axis=0)                                                      
                            scores = pd.concat([summary, scores], axis=1)
                            self._scores = pd.concat([self._scores, scores], axis=0)
                        
                        experiment += 1               

        # Time Decay Learning Rates
        if 't' in self._learning_rate_sched:
            for n in self._i_s:
                for p in self._precision:
                    for a in self._learning_rate_init:
                        for d in self._decay_rate:                    
                            self._gd = self._algorithm(theta=self._theta, learning_rate=a, 
                                    learning_rate_sched='t',decay_rate=d, 
                                    maxiter=self._maxiter, 
                                    precision=p, i_s=n,)
                            self._gd.fit(X,y)

                            results = self._gd.results()
                            summary = results['summary']
                            summary['experiment'] = experiment
                            train = results['train_errors']
                            train['experiment'] = experiment
                            evals = results['validation_errors']
                            evals['experiment'] = experiment


                            self._summary = pd.concat([self._summary, summary], axis=0)    
                            self._train = pd.concat([self._train, train], axis=0)    
                            self._validation = pd.concat([self._validation, evals], axis=0)    

                            if self._compute_scores:                            
                                train_score = pd.DataFrame({'score_type': 'Training Set Score',
                                                            'score': self._score(X,y, self._gd.theta)},
                                                            index=[0])
                                val_score = pd.DataFrame({'score_type': 'Validation Set Score',
                                                        'score': self._score(self._X_val, self._y_val, self._gd.theta)},
                                                        index=[0])
                                scores = pd.concat([train_score, val_score], axis=0)                                                      
                                scores = pd.concat([summary, scores], axis=1)
                                self._scores = pd.concat([self._scores, scores], axis=0)                         
                            
                            experiment += 1    

        # Step Decay Learning Rates
        if 's' in self._learning_rate_sched:
            for n in self._i_s:
                for p in self._precision:
                    for a in self._learning_rate_init:
                        for d in self._decay_rate:                    
                            for e in self._step_epochs:
                                self._gd = self._algorithm(theta=self._theta, learning_rate=a, 
                                        learning_rate_sched='s', decay_rate=d, step_epochs=e,
                                        maxiter=self._maxiter, precision=p, i_s=n,)
                                self._gd.fit(X,y)

                                results = self._gd.results()
                                summary = results['summary']
                                summary['experiment'] = experiment
                                train = results['train_errors']
                                train['experiment'] = experiment
                                evals = results['validation_errors']
                                evals['experiment'] = experiment

                                self._summary = pd.concat([self._summary, summary], axis=0)    
                                self._train = pd.concat([self._train, train], axis=0)    
                                self._validation = pd.concat([self._validation, evals], axis=0)    
                                
                                if self._compute_scores:                            
                                    train_score = pd.DataFrame({'score_type': 'Training Set Score',
                                                                'score': self._score(X,y, self._gd.theta)},
                                                                index=[0])
                                    val_score = pd.DataFrame({'score_type': 'Validation Set Score',
                                                            'score': self._score(self._X_val, self._y_val, self._gd.theta)},
                                                            index=[0])
                                    scores = pd.concat([train_score, val_score], axis=0)                                                      
                                    scores = pd.concat([summary, scores], axis=1)
                                    self._scores = pd.concat([self._scores, scores], axis=0)

                                experiment += 1   

        # Exponential Decay Learning Rates
        if 'e' in self._learning_rate_sched:
            for n in self._i_s:
                for p in self._precision:
                    for a in self._learning_rate_init:
                        for d in self._decay_rate:                    
                            self._gd = self._algorithm(theta=self._theta, learning_rate=a, 
                                    learning_rate_sched='e',decay_rate=d, 
                                    maxiter=self._maxiter, 
                                    precision=p, i_s=n,)
                            self._gd.fit(X,y)

                            results = self._gd.results()
                            summary = results['summary']
                            summary['experiment'] = experiment
                            train = results['train_errors']
                            train['experiment'] = experiment
                            evals = results['validation_errors']
                            evals['experiment'] = experiment


                            self._summary = pd.concat([self._summary, summary], axis=0)    
                            self._train = pd.concat([self._train, train], axis=0)    
                            self._validation = pd.concat([self._validation, evals], axis=0)    

                            if self._compute_scores:                            
                                train_score = pd.DataFrame({'score_type': 'Training Set Score',
                                                            'score': self._score(X,y, self._gd.theta)},
                                                            index=[0])
                                val_score = pd.DataFrame({'score_type': 'Validation Set Score',
                                                        'score': self._score(self._X_val, self._y_val, self._gd.theta)},
                                                        index=[0])
                                scores = pd.concat([train_score, val_score], axis=0)                                                      
                                scores = pd.concat([summary, scores], axis=1)
                                self._scores = pd.concat([self._scores, scores], axis=0)                    
                            
                            experiment += 1                                                


