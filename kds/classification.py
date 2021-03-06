"""
Some helper functions for evaluating classification performance.
"""

from __future__ import print_function
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc,\
    precision_recall_fscore_support, brier_score_loss, accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings



class BinaryEval(object):
    """
    Class for evaluating a classifier.
    
    TODO:
     - Make construction more general. Should be able to take shape
            (samples, 2), (samples, 1), (samples,), and (samples)
     - Include multi-class classifiers. 
     - Handle named vectors as well as int vectors.
     - Add better colors.
     - Add child class that have multiple predictions, so we can evaluate uncertainty of probabilities.
    """
    def __init__(self, true, probs, labels=None, ids=None):
        if len(true.shape) > 1:
            assert len(true.shape) == 2, 'Need 1 or 2 dim array'
            assert true.shape[1] == 2, 'Currently only works for binary classification'
            self.true = true[:, 1]
        else:
            self.true = true
        if len(probs.shape) > 1:
            assert len(probs.shape) == 2, 'Need 1 or 2 dim array'
            assert probs.shape[1] == 2, 'Currently only works for binary classification'
        else:
            probs = probs.reshape(probs.shape[0], 1)
            probs = np.concatenate((1-probs, probs), axis=1)
        self.probs = probs
        self.labels = labels
        if self.labels is None:
            self.labels = ['0', '1']
        self.target_name = self.labels[1]
        self.ids = np.array(ids)
        self.description = None
        self._to_dataframe()
        
    def _to_dataframe(self):
        """Make pd.DataFrame."""
        df_true = pd.DataFrame(self.true, columns=['true'])
        df_probs = pd.DataFrame(self.probs)
        df_probs.columns = self.labels if self.labels is not None else np.unique(self.true)
        self.df = pd.concat([df_true, df_probs], axis=1)
        if self.ids is not None:
            self.df['id'] = self.ids # If ids are a pd.Series, this goes wrong...
    
    @classmethod
    def from_df(cls, df):
        """
        Constructor with dataframe.
        
        Should be implemented nicer... Without transforming to vectors
        """
        assert df.__class__ == pd.core.frame.DataFrame
        labels = df.columns[1:3]
        ids = df.id.as_matrix() if 'id' in df.columns else None
        return cls(df.true.as_matrix(), df.iloc[:, 1:3].as_matrix(), labels, ids)
        
    def classification_report(self, treshold=0.5, digits=4, **kwargs):
        """
        sklearn.metrics.classification_report
        
        TODO:
         - Include option to remove f1-score (and replace with other metrics?).
         - Include option to use proportions instead of support.
         - Include option to choose threshold that balances recall.
         - Include option to write latex table.
         - Add *args and **kwargs.
         - Use dataframe.
        """
        print(classification_report(self.true, self.probs[:, 0] < (1-treshold), target_names=self.labels, digits=digits, **kwargs))

    def precision_recall_fscore_support(self, treshold=0.5, **kwargs):
        """See sklearn.metrics.precision_recall_fscore_support.
        **kwargs: passed to sklearn.metrics.precision_recall_fscore_support.
        """
        pred = (self.df[self.target_name] > treshold).astype(int)
        p, r, f, s =  precision_recall_fscore_support(self.df.true, pred, **kwargs)
        results = pd.DataFrame({'precision': p, 'recall': r, 'f1_score': f, 'support': s},
                index=self.labels, columns=['precision', 'recall', 'f1_score', 'support'])
        return results

    def hist_prob(self, labels=None, bins=20, **kwargs):
        """Give histograms of probabilities in labels."""
        ax = self.df.hist(labels, bins=bins, **kwargs)
        plt.xlabel('prob est')
        plt.ylabel('nr instances')
        return ax
    
    def confusion_matrix(self):
        raise NotImplemented

    # def recall_score(self, treshold=0.5, pos_label=1, average='binary', sample_weight=None):
    #     return metrics.recall_score(self.true, self.probs[:, 1] > treshold, self.labels,
    #                                 pos_label, average, sample_weight)

    # def precision_score(self, treshold=0.5, pos_label=1, average='binary', sample_weight=None):
    #     '''Sklearn precision_score'''
    #     return metrics.precision_score(self.true, self.probs[:, 1] > treshold, self.labels,
    #                                    pos_label, average, sample_weight)

    def log_loss(self, eps=1e-15, normalize=True, sample_weight=None):
        '''Sklearn log_loss'''
        return metrics.log_loss(self.true, self.probs[:, 1], eps, normalize, sample_weight, 
                                labels=None)

    def roc_auc_score(self, average='macro', sample_weight=None):
        '''Sklearn roc_auc_score.
        '''
        return metrics.roc_auc_score(self.true, self.probs[:, 1])

    def roc_area(self):
        warnings.warn('use roc_auc_score instead')
        return self.roc_auc_score()

    def roc_curve(self, legend_lab='', decimals='%0.3f', **kwargs):
        """
        Plotting roc_curve.
        Positive class is class 1, or the class that would be 1 if no labels.
        
        legend_lab: Name in start of legend.
        decimals: Gives how the auc (roc area under curve) is printed.
        
        TODO:
         - Multi class: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html 
         - Choose which class (label) we are looking at. Don't just choose probs[:, 1]!!!
         - Use DF instead.
        """
        fp, tp, _ = roc_curve(self.true, self.probs[:, 1])
        # plt.plot(fp, tp, label=legend_lab + ' (area = %0.3f)' % auc(fp, tp), **kwargs)
        plt.plot(fp, tp, label=legend_lab + ' (area = '+decimals % auc(fp, tp) +')', **kwargs)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.ylabel('true positive rate')
        plt.xlabel('false positive rate')
        plt.legend(loc="lower right")

    def accuracy_score(self, treshold=0.5, normalize=True, sample_weight=None):
        '''sklearn.metrics.accuracy_score
        Accuracy classification score.

        Parameters:
            normalize : bool, optional (default=True)
                If False, return the number of correctly classified samples.
                Otherwise, return the fraction of correctly classified samples.
            sample_weight : array-like of shape = [n_samples], optional.
                Sample weights.
        '''
        return metrics.accuracy_score(self.true, self.probs[:, 0] < (1-treshold), normalize, sample_weight)

    
    def bar_prob(self, label, nb_bins=20, ref_line=True, counts=True, ax=None, title=True):
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
        if title:
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

    def calibration_plot(self, ax=None, legend_lab=None, bins=10):
        """Similar to bar_prob, but this is the most common way to plot quality of the probabilities.
        label: Name of column to use as predictions. Defaults to class 1.
        ax: For plotting multiple lines in the same plot, let ax be the returned values from the previous plot.
        bins: Number of bins to use.
        legend_lab: Name to use in the legend.
        --------
        return:
        ax1, ax2: handles for the two plots in the figure. Pass to another .calibration_plot(ax=(ax1, ax2))
                  to plot multiple lines in the same figure.
        """
        preds = self.probs[:, 1]
        fraction_of_positives, mean_predicted_value = \
                calibration_curve(self.df.true.values, preds, n_bins=bins)
        if ax is None:
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
#             ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax1.plot([0, 1], [0, 1], "k:")
        else:
            ax1, ax2 = ax
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label=legend_lab)
        ax2.hist(preds, range=(0, 1), bins=bins, label=legend_lab,
                     histtype="step", lw=2)
        ax1.set_ylabel("fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
#         ax1.set_title('calibration plots  (reliability curve)')
        ax2.set_xlabel("mean predicted value")
        ax2.set_ylabel("count")
        ax2.legend(loc="upper center", ncol=2)
        return ax1, ax2

    def brier_score(self, **kwargs):
        """Brier score. See sklearn.metrics.brier_score_loss for full documentation."""
        return brier_score_loss(self.df.true, self.probs[:, 1])
    
    def add_description(self, description):
        """Add text description to object"""
        self.description = description

    def to_pickle(self, filename):
        """Pickle the dataframe."""
        self.df.to_pickle(filename)

    @staticmethod
    def read_pickle(filename):
        """Read pickled object."""
        return Class_eval.from_df(pd.read_pickle(filename))
    
    def to_hdf(self, path_or_buf, key='df', **kwargs):
        """
        Store using hdf5. 
        path_or_buf : the path (string) of HDFStore object.
        key : string identifier for the group in the store.
        See pandas documentation.
        """
        self.df.to_hdf(path_or_buf, key, **kwargs)

    @staticmethod
    def read_hdf(path_or_buf, key=None, **kwargs):
        """Read hdf5 file. 
        path_or_buf : path (string), or buffer to read from.
        key : group identifier in the store. Can be omitted a HDF file contains a single pandas object.
        See pandas documentation."""
        return Class_eval.from_df(pd.read_hdf(path_or_buf, key, **kwargs))

    def query(self, q, *args, **kwargs):
        """
        Use pandas query to subset data.
        Returns Class_eval object
        """
        return Class_eval.from_df(self.df.query(q, *args, **kwargs))
    
    def subset_by_ids(self, ids):
        """Return Class_eval object only containing relevant ids."""
        return Class_eval.from_df(self.df[self.df.id.isin(ids)])
        
        
        
        
class Class_eval_ensemble(object):
    """
    Class for working with multiple binary classifiers.
    Each classifiers should only contain the probability for class 1.
    
    true: true labels.
    classifiers: list of classifiers (probabilities for class 1).
    labels: labels of classes.
    names: names of classifiers.
    ids: list of ids for each datapoint.
    """
    def __init__(self, true, classifiers, labels=None, names=None, ids=None):
        self.true = true
        # Implement test to see if classifiers are of correct format
        self.classifiers = classifiers
        self.labels = labels
        self.names = names
        self.ids = ids
        self._to_dataframe()
        self._start_class = self.df.shape[1] - len(self.classifiers)
    def _to_dataframe(self):
        """Make pd.DataFrame"""
        df_true = pd.DataFrame(self.true, columns=['true'])
        cl_list = [pd.DataFrame(cl) for cl in self.classifiers]
        self.df = pd.concat([df_true] + cl_list, axis=1)
        if self.names is None:
            self.df.columns = ['true'] + ['cl'+str(i) for i in range(len(cl_list))]
        else:
            self.df.columns = ['true'] + self.names
        if self.ids is not None:
            self.df['id'] = self.ids
            cols = self.df.columns.tolist()
            self.df = self.df[cols[-1:] + cols[:-1]]
    def _series_to_Class_eval(self, series):
        """Returns Class_eval object from results."""
        df = pd.concat([1-series, series], axis=1)
        return Class_eval(self.true, df.as_matrix(), self.labels, self.ids)
    def _predict_average(self):
        """Predict using average over ensamble."""
        average = self.df.iloc[:, self._start_class:].mean(axis=1)
        return self._series_to_Class_eval(average)
    def _predict_max(self):
        """Predict using max over ensamble.
        Not implemented. Now we only take max, which will greatly favor the 1 class!!!!!!"""
        raise NotImplemented
        # max = self.df.iloc[:, self._start_class:].max(axis=1)
        # return self._series_to_Class_eval(max)
    def predict(self, rule='average'):
        """
        Do predictions of ensemble.
        rule: average, max, multiplication, vote, etc...
        """
        if rule == 'average': return self._predict_average()
        if rule == 'max': return self._predict_max()

def calibration_plots(results, ax=None, bins=10):
    """Function for plotting a calibration plot for a dict/list of Class_eval objects.
    ax: If you want to add the results to an existing figure. See docs for Class_eval.calibration_plot.
    bins: Number of bins.
    """
    if results.__class__ is dict:
        for name, res in results.iteritems():
            ax = res.calibration_plot(ax, name, bins)
    else:
        for res in results:
            ax = res.calibration_plot(ax, bins=bins)
    return ax


def calibration_from_undersampling(p, beta):
    """Correction of probabilities obtained by undersampling one class. 
    See eq (4) in https://www3.nd.edu/~rjohns15/content/papers/ssci2015_calibrating.pdf
    ---------------
    p: Probability of undersampled class.
    beta: Correction variable P(including a point of the undersampled class in the training set).
    ---------------
    returns: corrected probabilities.
    """
    p_correct = beta*p / (beta*p - p + 1)
    return p_correct


def odds(p):
    return (p / (1-p))


def logodds(p):
    return np.log(odds(p))


class Platt_scaling(object):
    """A class for doinc Platt scaling.
    logodds: If you want to fit the logodds instead of the preds.
    C: Regularizer to LogistiRegression
    """
    def __init__(self, transform=False, C=1000, **kwargs):
        self.transform = transform
        self.lr = LogisticRegression(C=C, **kwargs)
    def fit(self, preds, true):
        if self.transform == False:
            self.lr.fit(preds, true)
        else:
            self.lr.fit(self.transform(preds), true)
    def predict(self, preds):
        if self.transform == False:
            return self.lr.predict_proba(preds)
        return self.lr.predict_proba(self.transform(preds))

class Class_eval(BinaryEval):
    '''Legasy'''
