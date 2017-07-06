import pandas as pd
from functools import reduce

"""
TODO:
    DataFrame:
    - Make wrapper for saving and loading objects to disk.
    - Make general print methods used for logging training of models.
    - Make functions for creating cross-validations with repetitions.

    groupby obj:
    - Make nest function.
        - Should also be able to nest groups containing data frames (already nested)
            This could be done by instead changing pipe or apply or something...
"""

class DataFrame(pd.DataFrame):
    '''DataFrame for something like tidyr.'''
    
    @property
    def _constructor(self):
        '''Many pandas methods use copy which return a pd.DataFrame. 
        By over writing this, we keep our current type (not pandas).
        '''
        return DataFrame
    

    def applyRow(self, func, *args, **kwargs):
        '''Like pandas.apply(func, axis=1), but can return all types of objects.
        If pandas.apply later allow for returning arbitrary objects, remove this function.
        '''
        new = [row.pipe(func, *args, **kwargs) for idx, row in self.iterrows()]
        return pd.Series(new, index=self.index)
    

    def asapRow(self, **kwargs):
        '''Assign and apply to row. So like assign, but works on rows rather than full df.
        Shorthand to call sels.applylRow(func), and assign to variable name.
        '''
        # Don't know why this doesn't work...
#         return self.assign(**{name: lambda x: x.applyRow(func) for name, func in kwargs.items()})
        return self.assign(**{name: self.applyRow(func) for name, func in kwargs.items()})


    def unnest(self, column, dropIndex=True, dropColumn=True, checkNestedColumns=True):
        '''Like tidyr unnest. 
        Doesn't work for multiindex.
        column: column name to unnest.
        dropIndex: if we should keep the old index as a columns
        dropColumns: if we should drop the column that we unnest.
        checkNestedColumns: if True, make sure that all dataframes in 'column' have the same columns.
        '''
        if isinstance(self.index, pd.core.index.MultiIndex):
            # raise ValueError("Function unnest doesn't work wtih Multiindex. Use reset_index() before unnest.")
            self = self.reset_index()
        indexName = 'index_nest' if self.index.name is None else self.index.name
        def checkIndexName(indexName):
            if self.columns.contains(indexName) or self[column].iloc[0].columns.contains(indexName):
                indexName = indexName + '_nest'
                checkIndexName(indexName)
            return indexName
        indexName = checkIndexName(indexName)

        if checkNestedColumns:
            allCols = self[column].apply(lambda x: set(x.columns)).pipe(lambda s: reduce(lambda x,y: x|y, s))
            assert allCols == set(self[column].iloc[0].columns),\
                    "Column of dataframes in '"+column+"' differ. Need to be equal to unnest."
        
        df = self.reset_index().rename(columns={'index': indexName})
        mergeCol = (df
                    .applyRow(lambda x: x[column].assign(**{indexName: x[indexName]}))
                    .pipe(lambda x: pd.concat(list(x))))
        if dropColumn:
            df = df.drop(column, axis=1)
        df = df.merge(mergeCol, 'left', on=indexName)
        if dropIndex:
            df = df.drop(indexName, axis=1)
        return df


    #------------------------------------------
    # Suggested functions
    def assignUnzip(self, names, col):
        '''When a column contains tuples, this will assigne the tuples in 'col' to columns 'names'.
        names: list of new column names.
        col: names of column that contain tuples.
        '''
        unzip = list(zip(*self[col]))
        return self.assign(**{name: list(series) for name, series in zip(names, unzip)})
    
    def nest(self):
        '''Like tidyr nest. Should be part of groupby object.
        Don't know if it will ever be implemented.
        '''
        raise NotImplementedError
    
