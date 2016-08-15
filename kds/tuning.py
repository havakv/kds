"""
Some helper functions when tuning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


class Grid_scores(object):
    """Class for working with grid_scores from sklearn.grid_search.
    clf: object obtained from sklearn.grid_serach.GridSearchCV.fit method.
    """
    def __init__(self, clf):
        self.clf = clf
        self.nb_parameters = len(self.clf.param_grid)
        self.parameter_names = self.clf.param_grid.keys()

    def get_mean_validation_score(self):
        """Return scores as an np.array."""
        return np.array([score.mean_validation_score for score in self.clf.grid_scores_])

    def plot_scores(self):
        """Plot all scores."""
        x = plt.plot(self.get_mean_validation_score())
        plt.xlabel('task')
        plt.ylabel('mean_validation_score')
        return x

    def get_parameter_df(self):
        """Return a pd.DataFrame with parameters and scores."""
        par = defaultdict(list)
        for x in self.clf.grid_scores_:
            for k, v in x.parameters.iteritems():
                par[k].append(v)
            par['mean_validation_score'].append(x.mean_validation_score)
        self.parameter_df = pd.DataFrame(par)
        return self.parameter_df

    def violinplot(self, parameter_name, **kwargs):
        """Plot seaborn.violinplot of mean_validation_scores against parameter_name."""
        if not hasattr(self, 'parameter_df'):
            self.get_parameter_df()
        return sns.violinplot(x=parameter_name, y='mean_validation_score', data=self.parameter_df, **kwargs)

    def violinplot_all(self, cols=2, figsize=None, **kwargs):
        """Plot seaborn.violinplot of mean_validation_scores against all parameters."""
        rows = self.nb_parameters / cols + (self.nb_parameters % cols > 0)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        for name, ax in zip(self.parameter_names, axs.flatten()):
            self.violinplot(name, ax=ax, **kwargs)

    def pairgrid(self, vars='all', size=2.5):
        """Plot seaborn.PairGrid for the tuning parameters.
        !!!! Does not work for categorical variables!!!!!!!
        vars: list of variable names. If 'all' then all are plotted.
        **kwargs: passed to sns.pariplot.
        """
        if not hasattr(self, 'parameter_df'):
            self.get_parameter_df()
        if vars == 'all': vars = self.parameter_names
        parameter_df_notnull = self.parameter_df.loc[self.parameter_df.notnull().all(axis=1)]

        g = sns.PairGrid(parameter_df_notnull, vars=vars, size=size)
        cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
        c = parameter_df_notnull.mean_validation_score
        # g.map(plt.scatter, c=c, cmap=cmap)
        g.map_upper(plt.scatter, c=c, cmap=cmap)
        g.map_lower(sns.kdeplot, c=c, cmap=cmap)
        g.map_diag(plt.hist)
        # g.map_diag(plt.scatter, c=c, cmap=cmap)
        cax = g.fig.add_axes([.98, .4, .01, .2])
        plt.colorbar(cax=cax)




grig_parameters_random_forests = {
        'n_estimators': [120, 300, 500, 800, 1200],
        'max_depth': [5, 8, 15, 25, 30, None],
        'min_samples_split': [1, 2, 5, 10, 15, 100],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['log2', 'sqrt', None]
        }

def plot_feature_importances_(rf, names=None, title='Feature importance', **kwargs):
    """Plot RandomForeset features_importances_.
    names: list of names of the features.
    **kwargs: given to pandas.seres.plot()
    """
    ret = pd.DataFrame(rf.feature_importances_.reshape(1, -1), columns=names).ix[0].plot(kind='bar', **kwargs)
    plt.title(title)
    return ret
    

