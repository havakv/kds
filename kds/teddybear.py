"""
TODO:
    DataFrame:
    - Make wrapper for saving and loading objects to disk.
    - Make general print methods used for logging training of models. Maybe not in this library??
    - Make functions for creating cross-validations with repetitions.
"""

import pandas as pd
from functools import reduce


class Series(pd.Series):
    @property
    def _constructor(self):
        '''Many pandas methods use copy which return a pd.Series.
        By over writing this, we keep our current type (not pandas).
        '''
        return Series


    @property
    def _constructor_expanddim(self):
        '''Used to reset index and make a DataFrame. Override to keep teddy.'''
        return DataFrame


class DataFrame(pd.DataFrame):
    '''DataFrame for something like tidyr.'''

    @property
    def _constructor(self):
        '''Many pandas methods use copy which return a pd.DataFrame.
        By over writing this, we keep our current type (not pandas).
        '''
        return DataFrame


    _constructor_sliced = Series  # Keep teddy


    def rapply(self, func, *args, **kwargs):
        '''Like pandas.apply(func, axis=1), but can return all types of objects.
        If pandas.apply later allow for returning arbitrary objects, remove this function.
        '''
        new = [row.pipe(func, *args, **kwargs) for idx, row in self.iterrows()]
        return Series(new, index=self.index)


    def asap(self, **kwargs):
        '''Assign and apply to row. So like assign, but works on rows rather than full df.
        Shorthand to call sels.applylRow(func), and assign to variable name.
        '''
        # Don't know why this doesn't work...
#         return self.assign(**{name: lambda x: x.rapply(func) for name, func in kwargs.items()})
        return self.assign(**{name: self.rapply(func) for name, func in kwargs.items()})


    def unnest(self, column, dropNestIdx=True,
               dropIndex=False, dropColumn=True, checkNestedColumns=True):
        '''Like tidyr unnest.
        Doesn't work for multiindex.
        column: column name to unnest.
        dropNestIdx: if we should drop the index in the nested data frames.
        dropIndex: if we should keep the old index as a columns.
        dropColumns: if we should drop the column that we unnest.
        checkNestedColumns: if True, make sure that all dataframes in
            'column' have the same columns.
        '''
        new = self
        if isinstance(self.index, pd.core.index.MultiIndex):
            new = new.reset_index()

        if not dropNestIdx:
            new = new.asap(**{column: lambda x: x[column].reset_index()})

        indexName = 'index_nest' if new.index.name is None else new.index.name
        def checkIndexName(indexName):
            if new.columns.contains(indexName) or new[column].iloc[0].columns.contains(indexName):
                indexName = indexName + '_nest'
                checkIndexName(indexName)
            return indexName
        indexName = checkIndexName(indexName)

        if checkNestedColumns:
            allCols = (
                new[column]
                .apply(lambda x: set(x.columns))
                .pipe(lambda s: reduce(lambda x, y: x|y, s))
                )
            assert allCols == set(new[column].iloc[0].columns),\
                    "Column of dataframes in '"+column+"' differ. Need to be equal to unnest."

        new = new.reset_index().rename(columns={'index': indexName})
        mergeCol = (
            new.rapply(lambda x: x[column].assign(**{indexName: x[indexName]}))
            .pipe(lambda x: pd.concat(list(x)))
            )
        if dropColumn:
            new = new.drop(column, axis=1)
        new = new.merge(mergeCol, 'left', on=indexName)
        if dropIndex:
            new = new.drop(indexName, axis=1)
        return new

    def unzip(self, col, into, drop=True):
        '''Unzip column 'col' containing tuples.
        col: column to unzip.
        into: list of names of new columns.
        drop: if we should drop col from new data frame.
        '''
        raise NotImplementedError


    def assign_unzip(self, col, into, drop=True):
        '''When a column contains tuples, this will assigne the tuples in 'col' to columns 'into'.
        col: into of column that contain tuples.
        into: list of new column into.
        '''
        unzip = list(zip(*self[col]))
        new = self.assign(**{name: list(series) for name, series in zip(into, unzip)})
        if drop and (col not in into):
            return new.drop(col, axis=1)
        return new


    def flatten_colnames(self, inplace=False, bindChar='_'):
        '''For MultiIndex columns (multiple levels), flatten the columns, and concatenate names.
        '''
        newNames = [bindChar.join(i) for i in self.columns.get_values()]
        newNames = [i[:-1] if i[-1] == bindChar else i for i in newNames]
        if len(newNames) != len(set(newNames)):
            raise ValueError('The new column names are not unique. Maybe use another bindChar?')
        new = self if inplace else self.copy()
        new.columns = newNames
        return new

    def dropcol(self, columns):
        '''Shorthand for drop(columns, axis=1)'''
        return self.drop(columns, axis=1)

    applyRow = rapply  # legacy
    asapRow = asap
    assignUnzip = assign_unzip
    flatten_col_multi_index = flatten_colnames
    flattenColMultiIndex = flatten_col_multi_index


    def nest(self):
        '''Like tidyr nest. Should be part of groupby object.
        Don't know if it will ever be implemented.
        '''
        raise NotImplementedError


    def groupby(self, by=None, axis=0, level=None, as_index=True, sort=True,
                group_keys=True, squeeze=False, **kwargs):
        """
        Group series using mapper (dict or key function, apply given function
        to group, return result as series) or by a series of columns.
        Parameters
        ----------
        by : mapping, function, str, or iterable
            Used to determine the groups for the groupby.
            If ``by`` is a function, it's called on each value of the object's
            index. If a dict or Series is passed, the Series or dict VALUES
            will be used to determine the groups (the Series' values are first
            aligned; see ``.align()`` method). If an ndarray is passed, the
            values are used as-is determine the groups. A str or list of strs
            may be passed to group by the columns in ``self``
        axis : int, default 0
        level : int, level name, or sequence of such, default None
            If the axis is a MultiIndex (hierarchical), group by a particular
            level or levels
        as_index : boolean, default True
            For aggregated output, return object with group labels as the
            index. Only relevant for DataFrame input. as_index=False is
            effectively "SQL-style" grouped output
        sort : boolean, default True
            Sort group keys. Get better performance by turning this off.
            Note this does not influence the order of observations within each
            group.  groupby preserves the order of rows within each group.
        group_keys : boolean, default True
            When calling apply, add group keys to index to identify pieces
        squeeze : boolean, default False
            reduce the dimensionality of the return type if possible,
            otherwise return a consistent type
        Examples
        --------
        DataFrame results
        >>> data.groupby(func, axis=0).mean()
        >>> data.groupby(['col1', 'col2'])['col3'].mean()
        DataFrame with hierarchical index
        >>> data.groupby(['col1', 'col2']).mean()
        Returns
        -------
        GroupBy object
        """
        #from pandas.core.groupby import groupby # we use DataFrameGrouBy instead.
        groupby = _teddy_groupby

        if level is None and by is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        axis = self._get_axis_number(axis)
        return groupby(self, by=by, axis=axis, level=level, as_index=as_index,
                       sort=sort, group_keys=group_keys, squeeze=squeeze,
                       **kwargs)


def _teddy_groupby(obj, by, **kwsd):
    '''Function replacing pandas.core.groupby, so we can use inherited version
    of DataFrameGroupBy.
    '''
    return DataFrameGroupBy(obj, by, **kwsd)


class DataFrameGroupBy(pd.core.groupby.DataFrameGroupBy):
    def nest(self, grAsIndex=True, dropGrColsInNested=True):
        '''Returns a DataFrame with group data nested into DataFrames (column: data[_nested*x].
        Like nest in tidyr.
        grAsIndex: If the groups should be the index or columns of the returned DataFrame.
        dropGrColsInNested: If the group variables should be dropped from the nested data.
        '''
        def checkName(dataName):
            if dataName in self.keys:
                dataName = checkName(dataName + '_nested')
            return dataName
        dataName = checkName('data')
        keys = [self.keys] if self.keys.__class__ is str else self.keys
        new = DataFrame(list(self), columns=[keys[0], dataName])
        if len(keys) > 1:
            new = new.assign_unzip(keys, keys[0])
        if dropGrColsInNested:
            new = new.asap(**{dataName: lambda x: x[dataName].drop(keys, axis=1)})
        if grAsIndex:
            return new.set_index(keys)
        return new[keys + [dataName]]

    # def _wrap_aggregated_output(self, output, names=None):
    #     print('agg')
    #     return DataFrame(super()._wrap_aggregated_output(output, name))

    # def _wrap_transformed_output(self, output, names=None):
    #     print('trans')
    #     return DataFrame(super()._wrap_transformed_output(output, names))

    # def _wrap_agged_blocks(self, items, blocks):
    #     print('block')
    #     return DataFrame(super()._wrap_agged_blocks(items, blocks))

    # def _wrap_generic_output(self, result, obj):
    #     print('generic')
    #     return DataFrame(super()._wrap_generic_output(result, obj))

    def _gotitem(self, key, ndim, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : string / list of selections
        ndim : 1,2
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return DataFrameGroupBy(subset, self.grouper, selection=key,
                                    grouper=self.grouper,
                                    exclusions=self.exclusions,
                                    as_index=self.as_index)
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return SeriesGroupBy(subset, selection=key,
                                 grouper=self.grouper)

        raise AssertionError("invalid ndim for _gotitem")


class SeriesGroupBy(pd.core.groupby.SeriesGroupBy):
    pass
    # def aggregate(self, func_or_funcs, *args, **kwargs):
    #     print('aggreg')
    #     return Series(super().aggregate(func_or_funcs, *args, **kwargs))
# SeriesGroupBy = pd.core.groupby.SeriesGroupBy

concat = pd.concat # Works out of the box
