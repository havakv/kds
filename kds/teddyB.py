import pandas as pd

class DataFrame(pd.DataFrame):
    '''DataFrame for something like tidyr.'''
    
    @property
    def _constructor(self):
        '''Many pandas methods use copy which return a pd.DataFrame. 
        By over writing this, we keep our current type (not pandas).
        '''
        return DataFrame
    

    def applyRow(self, func, *args, **kwargs):
        new = [row.pipe(func, *args, **kwargs) for idx, row in self.iterrows()]
        return pd.Series(new, index=self.index)
    

    def asapRow(self, **kwargs):
        '''Assign and apply to row. So like assign, but works on rows rather than full df.
        Shorthand to call sels.applylRow(func), and assign to variable name.
        '''
        # Don't know why this doesn't work...
#         return self.assign(**{name: lambda x: x.applyRow(func) for name, func in kwargs.items()})
        return self.assign(**{name: self.applyRow(func) for name, func in kwargs.items()})


    def unnest(self, column, dropOldIndex=False, dropColumn=True):
        '''Like tidyr unnest.
        column: column name to unnest.
        dropOldIndex: if we should keep the old index as a columns
        dropColumns: if we should drop the column that we unnest.
        '''
        indexName = 'index_old' if self.index.name is None else self.index.name
        def checkIndexName(indexName):
            if self.columns.contains(indexName) or self[column].iloc[0].columns.contains(indexName):
                indexName = indexName + '_old'
                checkIndexName(indexName)
            return indexName
        indexName = checkIndexName(indexName)
        
        df = self.reset_index().rename(columns={'index': indexName})
        mergeCol = (df
                    .applyRow(lambda x: x[column].assign(**{indexName: x[indexName]}))
                    .pipe(lambda x: pd.concat(list(x))))
        if dropColumn:
            df = df.drop(column, axis=1)
        df = df.merge(mergeCol, 'left', on=indexName)
        if dropOldIndex:
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
        return self.assign(**{name: series for name, series in zip(names, unzip)})
    
    def nest(self):
        '''Like tidyr nest. Don't know if it will ever be implemented.'''
        raise NotImplementedError
    
