# %%
# =========================================================================== #
#                                DATASETS                                     #
# =========================================================================== #
''' Collection of datasets for use in regression and classification. '''
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

# %%
# --------------------------------------------------------------------------- #
class Ames():
    def __init__(self):
        self._feature_names = None
        self._target_name = 'SalePrice'

    def load(self, n=None, seed=None, vars=None, drop_na_columns=False):
        d = {}
        df = pd.read_csv("./data/raw/train.csv",
                     encoding="Latin-1", low_memory=False)

        if drop_na_columns:
            s, d = self.info()
            df = df.drop(d['idx_cols_missing'], axis=1)                     

        if n is not None:
            df = df.sample(n=n, random_state=seed)
        
        all_vars = list(df.columns)
        if vars is not None:
            if set(vars) <= set(all_vars):
                df = df[vars]                
                self._feature_names = vars
            else:
                raise Exception("Feature(s) is/are not valid.")
        else:
            self._feature_names = all_vars
        self._feature_names.remove('SalePrice')
        
        # Format returned object
        d['ames'] = df
        d['feature_names'] = self._feature_names
        d['target_name'] = 'SalePrice'
        d['features'] = df[self._feature_names]
        d['target'] = df['SalePrice'].values
        return(d)

    def load_df(self, n=None, vars=None, seed=None, drop_na_columns=False):
        df = pd.read_csv("./data/raw/train.csv",
                     encoding="Latin-1", low_memory=False)

        if drop_na_columns:
            s, d = self.info()
            df = df.drop(d['idx_cols_missing'], axis=1)                     
        if n is not None:
            df = df.sample(n=n, random_state=seed)
        if vars is not None:
            if isinstance(df[vars], pd.Series):
                df = df[vars].to_series(name=vars)
            else:
                df = df[vars]
        return(df) 
        
    def feature_names(self, drop_na_columns=None):
        if self._feature_names is None:
            df = pd.read_csv("./data/raw/train.csv",
                     encoding="Latin-1", low_memory=False)
        
        if drop_na_columns:
            s, d = self.info()
            df = df.drop(d['idx_cols_missing'], axis=1)
        self._feature_names = list(df.columns)
        self._feature_names.remove('SalePrice')
            
        return(self._feature_names)


    def target_name(self):
        return(self._target_name)

    def features(self, format='a', vars=None, dropna=False, drop_na_columns=False):
        '''Returns features in an array or dataframe format. 

        Args:
            format (str, optional): Indicates format of output - 'a' for
                array, 'd' for dataframe
            vars (str): String or list containing names of features to return

        Returns:
            Array or dataframe containing requested features.
        '''
        data = self.load()
        df = data['features']
        if drop_na_columns:
            s, d = self.info()
            df = df.drop(d['idx_cols_missing'], axis=1)
                    
        if vars is not None:
            df = df[vars] 
        
        if dropna:
            df = df.dropna()
        if format == 'a':
            return(df.values)
        else:
            return(df)

    def target(self, format='a'):
        '''Returns the target of the data in an array or dataframe format. 

        Args:
            format (str, optional): Indicates format of output - 'a' for
                array, 'd' for dataframe

        Returns:
            Array or dataframe containing the target.
        '''
        
        df = pd.read_csv("./data/raw/train.csv", encoding="Latin-1",
                         low_memory=False)
        if format == 'a':
            return(df[self._target_name].values)
        else:
            return(df[self._target_name])

    def describe(self):
        """ Returns summary and diagnostic information on the DataFrame

        Returns:
        --------
        Dictionary with summary information and descriptive statistics

        """
        import sys, os
        eda_studio_path = "../eda_studio"
        eda_studio_path = os.path.abspath(eda_studio_path)
        sys.path.append(eda_studio_path)
        from eda_studio.describe import describe
        # ------------------------------------------------------------------- #
        df = self.load_df()
        desc = describe(df)
        return(desc)

    def info(self):
        """ Returns summary and diagnostic information on the DataFrame

        Returns:
        --------
        Dictionary with summary information and descriptive statistics

        """
        import sys, os
        eda_studio_path = "../eda_studio"
        eda_studio_path = os.path.abspath(eda_studio_path)
        sys.path.append(eda_studio_path)
        from eda_studio.describe import info
        # ------------------------------------------------------------------- #
        df = self.load_df()
        s, d = info(df)
        return(s, d)        
