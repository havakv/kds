'''
Library for preprocessing techniques that are not included elsewhere.
'''
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

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
    def __init__(self, encode=None, handle_unknown='error', ignore_nan=False):
        warnings.warn('Should maybe have a categorical implementation, see https://tomaugspurger.github.io/categorical-pipelines.html')
        self.encode = encode
        self.handle_unknown = handle_unknown
        self.ignore_nan = ignore_nan

    def _check_nan(self, x):
        if not self.ignore_nan:
            if x.isnull().any():
                raise Exception('Array contains nans!')
        else:
            raise Exception('This method is currently not able to use np.nan as a separate category,\
                as we use np.nan to encode all variables that was not fitted.')

    def fit(self, x, y=None):
        if hasattr(self.encode, '__iter__'):
            self.encSet = set(self.encode)
            if len(self.encSet - set(x)) != 0:
                raise ValueError('encode varialbe contains values not found in x.')
        elif self.encode is not None:
            raise ValueError("Need encode to be iterable or None.")
        else:
            self.encSet = set(x)

        self.encMap = {k: i+1 for i, k in enumerate(self.encSet)}

        x = pd.Series(x)
        self._check_nan(x)
        xEmb = x.map(self.encMap).fillna(0)
        xfit = xEmb[xEmb != 0].values.reshape(-1, 1)

        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(xfit)

        return self


    def transform(self, x):
        if hasattr(self, 'encoder') == False:
            raise Exception('Need to fit method before we can transform.')
        if self.handle_unknown not in['error', 'ignore']:
            raise ValueError('handle_unknown needs to be "error" or "ignore"')

        x = pd.Series(x)
        self._check_nan(x)
        xEmb = x.map(self.encMap)
        if self.handle_unknown == 'ignore':
            xEmb.fillna(0, inplace=True)
        return self.encoder.transform(xEmb.values.reshape(-1, 1))


class QuantStandardizer(BaseEstimator, TransformerMixin):
    '''Make data follow a standard normal distribution, using quantiles.
    TODO:
     - Use correct quantiles (1/2).
     - Center quantiles (not use left).
     - Hanle muliple columns at a time (now it only handles one)
     - Hanle many identical values.
     - Interpolation rather than lookup???
     - Give option to only use unique values (remove duplicates). 
        Violates normal assumption, but might be good when very many idential values
    '''
    def fit(self, x, y=None):
        '''x: covariates.
        y: doesn't do anything.
        '''
        self.maxval = x.max()
        self.n = len(x)
        self.xSorted = np.sort(x)
        return self

    def transform(self, x):
        '''x: covariates'''
        order = self.xSorted.searchsorted(x)
        order[order >= self.n] = self.n - 1
        return norm.ppf((order + 1) / (self.n + 1))
