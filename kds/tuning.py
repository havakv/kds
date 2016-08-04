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




grig_parameters_random_forests = {
        'n_estimators': [120, 300, 500, 800, 1200],
        'max_depth': [5, 8, 15, 25, 30, None],
        'min_samples_split': [1, 2, 5, 10, 15, 100],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['log2', 'sqrt', None]
        }
