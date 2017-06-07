'''
Library for preprocessing techniques that are not included elsewhere.
'''
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotSubset(BaseEstimator, TransformerMixin):
    '''One hot encoder for all datatypes (string, int, float, etc), 
    where we can specify subset of values that are encoded.
    Values that are no included, are coded as the zero vector.
    If both encode and drop are None, we use all variable names.
    
    encode: Iterable with variables that should be encoded. If None, we use all.
    
    handle_unknown : str, 'error' or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform.
    
    ignore_nan: If true we ignore nans, if False we raise an error.
    '''
    def __init__(self, encode=None, handle_unknown='error', ignore_nan=True):
        self.encode = encode
        self.handle_unknown = handle_unknown
        self.ignore_nan = ignore_nan
    
    def _check_nan(self, x):
        if not self.ignore_nan:
            if np.isnan(x).any():
                raise Exception('Array contains nans!')
    
    def fit(self, x, y=None):
        self._check_nan(x)
        self.allVars = self.enc = set(x)
        if hasattr(self.encode, '__iter__'):
            self.enc = set(self.encode)
        elif self.encode is not None:
            raise ValueError("Need encode to be iterable or None.")
        return self
        

    def transform(self, x):
        if hasattr(self, 'enc') == False:
            raise Error('Need to fit method before we can transform.')
        
        self._check_nan(x)
        values = set(x)
        if self.handle_unknown == 'error' and len(values.difference(self.allVars)) > 0:
            raise Error('Unknown values!')
        elif self.handle_unknown not in ['error', 'ignore']:
            raise ValueError('handle_unknown needs to be "error" or "ignore".')
            
        drop = values.difference(self.enc)
        if len(drop) == 0:
            return pd.get_dummies(x)
        x = pd.Series(x)
        return pd.get_dummies(x.replace(list(drop), np.nan))
