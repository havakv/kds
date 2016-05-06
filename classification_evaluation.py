"""
Some helper functions for evaluating classification performance.
"""

from sklearn.metrics import classification_report, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Class_eval(object):
    """
    Class for evaluating a classifier.
    
    TODO:
     - Consider inherit from pandas
     - Include multi-class classifiers.
     - Include option for classifiers that don't give probabilities.
     - Handle named vectors as well as int vectors.
     - Add better colors.
     - Add child class that have multiple predictions, so we can evaluate uncertainty of probabilities.
    """
    def __init__(self, true, probs, labels=None, ids=None):
        self.true = true
        assert probs.shape[1] == 2 # Currently only works for binary classification
        self.probs = probs
        self.labels = labels
        self.ids = ids
        self.description = None
        self._to_dataframe()
        
    def _to_dataframe(self):
        """Make pd.DataFrame."""
        df_true = pd.DataFrame(self.true, columns=['true'])
        df_probs = pd.DataFrame(self.probs)
        df_probs.columns = self.labels if self.labels is not None else np.unique(self.true)
        self.df = pd.concat([df_true, df_probs], axis=1)
        if self.ids is not None:
            self.df['id'] = self.ids
    
    @classmethod
    def from_df(cls, df):
        """
        Constructor with dataframe.
        
        Should be implemented nicer... Without transforming to vectors
        """
        assert df.__class__ == pd.core.frame.DataFrame
        labels = df.columns[1:3]
        ids = df.id if 'id' in df.columns else None
        return cls(df.true.as_matrix(), df.iloc[:, 1:3].as_matrix(), labels,
                   ids.as_matrix())
        
    def classification_report(self, treshold=0.5):
        """
        sklearn.metrics.classification_report
        
        TODO:
         - Include option to remove f1-score (and replace with other metrics?).
         - Include option to use proportions instaed of support.
         - Include option to choose treshold that balances recall.
         - Include option to write latex table.
         - Add *args and **kwargs.
         - Use dataframe.
        """
        print classification_report(self.true, self.probs[:, 0] < (1-treshold), 
                                     target_names=self.labels)
    def hist_prob(self, labels=None, bins=20, **kwargs):
        """Give histograms of probabilities in labels."""
        ax = self.df.hist(labels, bins=bins, **kwargs)
        plt.xlabel('prob est')
        plt.ylabel('nr instances')
        return ax
    
    def confusion_matrix(self):
        raise NotImplemented
    
    def roc_area(self):
        fp, tp, _ = roc_curve(self.true, self.probs[:, 1])
        return auc(fp, tp)
    
    def roc_curve(self, legend_lab='', **kwargs):
        """
        Plotting roc_curve.
        
        TODO:
         - Multi class: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html 
         - Choose which class (label) we are looking at. Don't just choose probs[:, 1]!!!
         - Use DF instead.
        """
        fp, tp, _ = roc_curve(self.true, self.probs[:, 1])
        plt.plot(fp, tp, label=legend_lab + ' (area = %0.2f)' % auc(fp, tp), **kwargs)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('true positive rate')
        plt.xlabel('false positive rate')
        plt.legend(loc="lower right")
    
    def bar_prob(self, label, nb_bins=20, ref_line=True, counts=True, ax=None):
        """
        Bar chart with estimated probabilities vs. class proportions.
        """
        if self.labels is None: raise ValueError('Need to set labels in __init__ to use this function')
        bins = np.linspace(0, 1, nb_bins+1)
        df = self.df[['true', label]].copy()
        df['true_name'] = np.array(self.labels)[df['true'].astype(int)]
        df['mean_prob'] = 0
        for l, u in zip(bins[:-1], bins[1:]):
            df.loc[df[label] >= l, ['mean_prob']] = np.mean([l, u])
        count_bin = df.groupby(['mean_prob', 'true_name'])[label].count().unstack('true_name')
        props = (count_bin[label]/count_bin.sum(axis=1))
        ax = ax if ax else plt.subplots()[1]
        ax.bar(props.index, props, width=bins[1], align='center', color='grey', 
               label='prop', zorder=3)
        plt.title(label)
        if ref_line:
            ax.plot([0, 1], [0, 1], 'k--', label='reference', zorder=3)
        ax.set_xlabel('prob est')
        ax.set_ylabel('proportions')
        if not counts: return ax
        ax2 = ax.twinx()
        ax2.grid(None)
        ax2.plot(props.index, count_bin.sum(axis=1), 'blue')
        for tl in ax2.get_yticklabels(): tl.set_color('b')
        ax2.set_ylabel('nr instances', color='b')
        return (ax, ax2)
    
    def add_description(description):
        """Add text description to object"""
        self.description = description
    
    def save_to_file(filename):
        raise NotImplemented
    
    def query(self, q, *args, **kwargs):
        """
        Use pandas query to subset data.
        Returns Class_eval object
        """
        return Class_eval.from_df(self.df.query(q, *args, **kwargs))
    
    def subset_by_ids(self, ids):
        return Class_eval.from_df(self.df[self.df.id.isin(ids)])
        
        
        
        
class Class_ensamble(object):
    def __init__(self):
        raise NotImplemented