"""
Helper functions for keras.
"""

from __future__ import print_function

import itertools
from copy import deepcopy

import keras
import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from PIL import Image
from sklearn.model_selection import KFold, LeaveOneGroupOut, ParameterGrid

from .classification import Class_eval, Class_eval_ensemble
from .teddybear import DataFrame


def load_and_convert_images(filenames, size, color='L'):
    """Loading and converting images so they will fit in Keras net.
    -----------
    filenames: iterable of filenames.
    size: touple of image dimensions.
    color: is spesified accoring to PIL.Image.Image.convert.
    -----------
    return: np.array of scaled images
    """
    assert color == 'L' # We have currently only implemented for grayscale
    img = [np.asarray(Image.open(f).convert(color).resize(size)) for f in filenames]
    img = np.asarray(img).reshape(len(img), 1, size[0], size[1]).astype('float32')
    return img / 255

def retreive_converted_image(img, size=(100, 100)):
    """Get original image from converted image.
    """
    return Image.fromarray((img*255).astype('uint8')).resize(size)



class Model_activations(object):
    """Class for investigating activations of a keras model.

    @@@@@@@@@@@@@@
    @@@@@@@@@@@@@@
    This should be done by just one forward pass and not one per every layer!!!!!!!!!!!!!!!!!!!!1
    ########
    ########
    """
    def __init__(self, model):
        self.model = model
        self.model_input = model.layers[0].input
        self.layer_models = None
        self.layer_models_dict = None

    def build_layer_submodels(self):
        """
        Builds one model for each layer in the model, so we can get output in each layer.
        """
        self.layer_models = [Model(input=self.model_input, output=layer.output) for layer in self.model.layers]
        self.layer_models_dict = {model.layers[-1].name: model for model in self.layer_models}

    def get_output(self, X, layer_names=None):
        """
        Get activation for layer <layer_name>.
        X: input features.
        layer_name: Name of layer we want activations at. If None, all activations are computed.
        """
        if layer_names is None:
            return [model.predict(X) for model in self.layer_models]
        assert hasattr(layer_names, '__iter__')
        return [self.layer_models_dict[layer_name].predict(X) for layer_name in layer_names]



class Weights_each_batch(keras.callbacks.Callback):
    """Store relevant trainable weights for each batch, when fitting the keras model.

    TODO:
     - Make option for not storing all weights but rather calculate the weights stats on the go.

    Example:
    weights = Weights_each_batch()
    model.fit(X, Y, nb_epoch=1, callbacks=[weights])
    weights.hist_scale_stats('ratio')
    print(weights.scale_stats.groupby('weight_name').ratio.mean.max())
    """
    def __init__(self, layer_names=None, store_all=False):
        if layer_names is not None: assert hasattr(layer_names, '__iter__')
        self.layer_names = layer_names
        self.store_all = store_all
        self.layer_dict = None
        self.weights = []
        self.weights_diff = None
        self.scale_stats = None
        super(Weights_each_batch, self).__init__()

    def _append_weights(self):
        """Get weights from current model.
        """
        if self.layer_names is None: # get all trainable weights
            weights = []
            for layer in self.model.layers:
                w_list = [keras.backend.get_value(w) for w in layer.trainable_weights]
                weights.append((layer, w_list))
            # self.weights.append(weights)
        else:
            weights = []
            for layer in self.layer_names:
                w_list = [keras.backend.get_value(w) for w in self.layer_dict[layer].trainable_weights]
                weights.append((layer, w_list))
            # self.weights.append(weights)
        self.weights.append(weights)

    def on_train_begin(self, logs={}):
        """Function called by kera's fit function.
        """
        self.weights = []
        self.weights_diff = None
        self.scale_stats = None
        super(Weights_each_batch, self).__init__()

        if self.layer_names is not None:
            self.layer_dict = {layer.name: layer for layer in self.layers}
        if not self.store_all:
            self._scale_stats_list = []
            self.weights_diff = []
        self._append_weights()

    def on_batch_end(self, batch, logs={}):
        """Function called by kera's fit function.
        """
        self._append_weights()
        if not self.store_all:
            self.weights_diff.append(self._make_diff_batch(self.weights[0], self.weights[1]))
            scale_stats = self._get_scale_stats_batch(self.weights[0], self.weights_diff[0])
            self._scale_stats_list.append(scale_stats)
            self._delete_prev_iteration()

    def _make_diff_batch(self, weights_prev, weights_curr):
        """Based on the previous and current weights, compute the difference between them.
        """
        weights_diff = []
        for lay_p, lay_c in zip(weights_prev, weights_curr):
            weights_diff.append((lay_p[0], [wc - wp for wp, wc in zip(lay_p[1], lay_c[1])]))
        # self.weights_diff.append(weights_diff)
        return weights_diff

    def _delete_prev_iteration(self):
        """Delete weights no longer needed.
        """
        assert self.store_all == False, 'Should only be used when store_all=False'
        self.weights.pop(0)
        self.weights_diff.pop(0)


    def make_diff(self):
        """Make self.weights_diff to be  the difference between weights in each iterations.
        """
        assert self.store_all, 'Need to be run with store_all=True'
        self.weights_diff = []
        for prev, curr in zip(self.weights[:-1], self.weights[1:]):
            # weights_diff = []
            # for lay_p, lay_c in zip(prev, curr):
                # weights_diff.append((lay_p[0], [wc - wp for wp, wc in zip(lay_p[1], lay_c[1])]))
            # self.weights_diff.append(weights_diff)
            self.weights_diff.append(self._make_diff_batch(prev, curr))

    def _get_scale_stats_list(self):
        """Returns the 2-norm of the unraveled weights, their update and the ratio norm(update)/norm(weights).
        """
        if self.weights_diff is None:
            self.make_diff()
        stats = []
        for weights, diffs in zip(self.weights[:-1], self.weights_diff):
            stats.append(self._get_scale_stats_batch(weights, diffs))
        return stats

    def _get_scale_stats_batch(self, weights, diffs):
        """Returns the 2-norm of the unraveled weights, their update and the ratio norm(update)/norm(weights).
        weights : list of weights for a model.
        diffs : list of weights_diffs for a model.
        """
        stats = []
        for lay_w, lay_d in zip(weights, diffs):
            stats_iter = []
            for w, d in zip(lay_w[1], lay_d[1]):
                norms = (np.linalg.norm(w), np.linalg.norm(d))
                stats_iter.append(norms + (norms[1]/norms[0],))
            stats.append(stats_iter)
        return stats

    def get_scale_stats(self):
        """Returns dataframe with the 2-norm of the unraveled weights, their update and 
        the ratio norm(update)/norm(weights).

        In df:
        norm_w : 2-norm of weights before iteration.
        norm_diff : 2-norm of difference between weighs before and after iteration.
        ratio : norm_diff / norm_w
        """
        if self.store_all:
            stats = self._get_scale_stats_list()
        else:
            stats = self._scale_stats_list
        dfs = []
        for it, stats_iter in enumerate(stats):
            dfs_iter = []
            for layer_stats, layer in zip(stats_iter, self.model.layers):
                df = pd.DataFrame(layer_stats, columns=('norm_w', 'norm_diff', 'ratio'))
                df['weight_name'] = [str(w) for w in layer.trainable_weights]
                df['layer'] = layer.name
                df['iteration'] = it+1
                dfs_iter.append(df)
            dfs.extend(dfs_iter)
        self.scale_stats = pd.concat(dfs).reset_index().iloc[:, 1:]
        return self.scale_stats

    def hist_scale_stats(self, column=['ratio'], query=None, **kwargs):
        """Histograms for displaying scale stats over all batches.
        *args, and **kwargs are passed to the histogram function.
        By passing only the name 'norm_w', 'norm_diff' or 'ratio', only the relevant plots are created.

        query: A query to self.scale_stats (pd.DataFrame). Useful for displaying subset of iterations.
            Example: query='iteration == 5'.
        """
        if self.scale_stats is None:
            self.get_scale_stats()
        if query is not None:
            return self.scale_stats.query(query).hist(column=column, **kwargs)
        return self.scale_stats.hist(column=column, **kwargs)

    def bar_mean_scale_stats(self, y='ratio', query=None, title='set_title', **kwargs):
        """Bar plot over average scale stats.
        The stats are the 2-norm over the weights (norm_w), the difference in the updated weights (norm_diff),
        and the ration between the two (ratio) (norm_diff/norm_w).
        See function get_scale_stats for further explanation.

        y : columns in self.scale_stats we want to plot.
        query: A query to self.scale_stats (pd.DataFrame). Useful for displaying subset of iterations.
            Example: query='iteration == 5'.
        """
        if title == 'set_title':
            title = 'Average ' + y
        if self.scale_stats is None:
            self.get_scale_stats()

        stats = self.scale_stats
        if query is not None:
            stats = stats.query(query)
        return stats.groupby('weight_name').mean().reset_index()\
                .plot(x='weight_name', y=y, kind='bar', title=title, **kwargs)

    def look_for_small_and_large_updates(self, relative=True):
        """A function for investigating small and large updates in the weights. 
        relative : if True, the update/weight is used instead of just the update.
        """
        pass


class Store_predictions(keras.callbacks.Callback):
    """Store predictions every epoch.
    """
    # def __init__(self):
        # self.nb_inputs = len(self.model.input_shape)
        # super(Store_predictions, self).__init__()

    def on_train_begin(self, logs={}):
        self.nb_inputs = len(self.model.input_shape) if self.model.input_shape.__class__ == list else 1
        self.predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.predictions.append(self.model.predict(self.model.validation_data[:self.nb_inputs]))

def _roc_auc(true, predictions):
    """Calculate ROC AUC.
    """
    return Class_eval(true, predictions).roc_area()


class ModelCheckpointExt(keras.callbacks.ModelCheckpoint):
    """An extension of Keras' ModelCheckpoint. Here we can also pass a function instead of a string.
    Currently only works for 'val_auc' in addition to regular ModelCheckpoint arguments.
    You can also pass a function to monitor: monitor(true, predictions), but this only works for the validation data.
    """
    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, 
                save_weights_only=False, mode='auto'):
        super(ModelCheckpointExt, self).__init__(filepath, monitor, verbose, save_best_only, 
                save_weights_only, mode)
        if monitor == 'val_auc':
            self.monitor = _roc_auc 
            if mode == 'auto':
                self.monitor_op = np.greater
                self.best = -np.Inf

    def on_train_begin(self, logs={}):
        self.nb_inputs = len(self.model.input_shape) if self.model.input_shape.__class__ == list else 1

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if (current is not None) or (self.save_best_only == False):
            super(ModelCheckpointExt, self).on_epoch_end(epoch, logs)
        else:
            filepath = self.filepath.format(epoch=epoch, **logs)
            val_predictions = self.model.predict(self.model.validation_data[:self.nb_inputs])
            val_true = self.model.validation_data[self.nb_inputs]
            current = self.monitor(val_true, val_predictions)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    if self.monitor.__class__ is str:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s'
                                % (epoch, self.monitor, self.best, current, filepath))
                    else:
                        print('Epoch %05d: measure improved from %0.5f to %0.5f,'
                                ' saving model to %s'
                                % (epoch, self.best, current, filepath))
                self.best = current
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve' %
                            (epoch, self.monitor))
                


def show_model_weight_shapes(model):
    """Function for printing a keras model, and its weight shapes.
    """
    # for layer in model.layers:
    for layer in model.layers:
        print('----', layer.name)
        for w in layer.trainable_weights:
            print(w, ':', keras.backend.get_value(w).shape)
        print('')

def show_model_layer_input_output_shapes(model):
    """Function for printing a keras model, and each layer's input and output shape.
    """
    print('Should rather use model.summary() function!!!!!!!!!!!!!')
    print('')
    for layer in model.layers:
        print('----', layer.name)
        print('input_shape  :', layer.input_shape)
        print('output_shape :', layer.output_shape)
        print('')


class Uncertainty_estimates(object):
    """Get uncertainty estimates from keras model using dropout as an bayesian approximation 
    See https://arxiv.org/abs/1506.02142

    !!! WARNING !!! This will effect all layers that behave differently under training and testing, 
    e.g. batchnorm

    !!! WARNING !!! Might need to partition data in batches.
    """
    def __init__(self, model):
        self.predict_fun = K.function([model.input, K.learning_phase()], [model.output])

    def predict_on_batch(self, X_batch, nb_samples=1):
        samples = [self.predict_fun([X_batch, 1])[0] for _ in range(nb_samples)]
        return samples

    def make_predictions(self, X, nb_samples=1, batch_size=256, overwrite=True):
        """Generates nb_samples prediction from the model, using the learning phase.
        X: data passed to model.
        batch_size: batche size for the model.
        np_samples: number of samples to make.
        overwrite: Overwrite previous results (True) or add new results to previous results (False).
        --------------
        returns: self
        """
        batches = np.arange(0, len(X)+1, batch_size)
        if batches[-1] < (len(X)+1):
            batches = np.concatenate([batches, [len(X)+1]])

        # Get all predictions
        estimates = []
        for start, end in zip(batches[:-1], batches[1:]):
            X_batch = X[start:end]
            estimates.append(self.predict_on_batch(X_batch, nb_samples))

        # Reorder output
        est = []
        for i in range(nb_samples):
            x = []
            for j in range(len(batches)-1):
                x.append(estimates[j][i][:, 1])
            est.append(np.concatenate(x))
        df = pd.DataFrame(est)
        if hasattr(self, 'df') and (overwrite == False):
            self.df = self.df.append(df, ignore_index=True)
        else:
            self.df = df
        return self

    def to_Class_eval(self, true, *args, **kwargs):
        """Return Class_eval object.
        See documentation for Class_eval_ensemble
        """
        return Class_eval(true, self.df.mean(), *args, **kwargs)

    def to_Class_eval_ensemble(self, true, *args, **kwargs):
        """Return Class_eval_ensemble object.
        See documentation for Class_eval_ensemble
        """
        raise NotImplementedError()


class KerasClassifier_lossScore(KerasClassifier):
    '''Same as keras.wrappers.scikit_learn.KerasClassifier, but
    with scorer that use loss instead of accuracy.
    TODO:
    Should be implemeted to use both...
    '''
    def score(self, x, y, **kwargs):
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
    #         if name == 'acc':
            if name == 'loss':
                return output
        raise ValueError("Don't know what went wrong. See source code...")


class KerasClassifierGSCV(object):
    def __init__(self, build_fun, **fitParameters):
        self.build_fun = build_fun
        self.fitParameters = fitParameters

    def build(self, **kwargs):
        self.model = self.build_fun(**kwargs)
        return self

    def fit(self, Xtr, ytr, Xte, yte, **kwargs):
        '''
        **kwargs can be used to overwrite arguments in __init__ **fitParameters.
        '''
        self.fitParameters.update(kwargs)
        log = self.model.fit(Xtr, ytr, validation_data=[Xte, yte], **self.fitParameters)
        return log, self

    def copy(self):
        return deepcopy(self)


class _KFoldWrapperForIterables(object):
    '''Wrapper to make a list of CV indexes behave like sklern KFold.
    splits: Iterable of indexes for the folds. Each [(train0, val0), (train1, val1), ...].
        Something like list(KFold(3).split(X))
    '''
    def __init__(self, splits):
        self._splits = list(splits)
        self.n_splits = len(self._splits)

    def split(self, X=None, y=None, groups=None):
        '''X, y and groups does not do anything.
        They are only there for compatability.
        '''
        for split in self._splits:
            yield split


class GridSearchCVKeras(object):
    '''
    estimator: KerasClassifierGSCV. Want to extend this to sklearn!!!!!!!!!!!!!!
    param_grid: dictionary of parameters (as in in sklearn GridSearchCV).
    cv: Number of cv folds, or KFold object.

    #-----------------
    Examples:
    #-----------------
    # Regular
    estimator = KerasClassifierGSCV(build_fun, verbose=1, batchsize=512, epochs=10)
    gr = GridSearchCVKeras(estimator, param_grid, cv=3)
    gr.fit(X, y)

    # Leave-one-group-out:
    logo = sklearn.model_selection.LeaveOneGroupOut().split(df, groups=train['group'].values)
    gr = GridSearchCVKeras(estimator, param_grid, cv=logo)
    gr.fit(df.drop(['y', 'group'], axis=1), df['y'].values)
    #-----------------


    TODO:
    - Make version that with validation set instead of cv folds.
    - Make version with random search over parameters.
    - Make fit method verbose.
    - Change name to GridSearchCVKeras
    - Should be able to call fit multiple times.
    - Fit methods should take 'epochs' as parameter???
    - Should have a constructor that can take a subset of the grid and continue training on that???
    '''
    def __init__(self, estimator, param_grid, cv=3):
        self.estimator = estimator
        self.param_grid = param_grid
        if type(cv) is int:
            self.cv = KFold(cv, shuffle=True)
        elif type(cv) is KFold:
            self.cv = cv
        elif hasattr(cv, '__iter__'):
            self.cv = _KFoldWrapperForIterables(cv)
        else:
            raise ValueError('Need cv to be int, iterable or KFold')

        self.cvParameters = list(self.param_grid.keys())

        self.grid_setup = DataFrame(list(ParameterGrid(dict(**self.param_grid, cv_fold=range(self.cv.n_splits)))))
        self._addParameterConfigId()

    def _addParameterConfigId(self):
        '''For each group of parameter, add parConfigId, which makes it easier to group later.'''
        parConfigId = (self.grid_setup
                       .drop('cv_fold', axis=1)
                       .drop_duplicates()
                       .assign(parConfigId=lambda x: np.arange(len(x))))
        self.grid_setup = self.grid_setup.merge(parConfigId, 'left', on=self.cvParameters)
        
    
    def setupFolds(self, X, y, mapper=None):
        '''Make indices for cv-folds
        '''
        if X.__class__ in [pd.DataFrame, DataFrame]:
            return self._setupFoldsDataFrame(X, y, mapper)

        raise NotImplementedError('need to implement mapper functionality')
        splits = list(self.cv.split(X))
        train, test = list(zip(*splits))
        splits = (DataFrame(dict(trainIdx=train, testIdx=test, cv_fold=range(self.cv.n_splits)))
                  .asapRow(Xtr = lambda x: X[x['trainIdx']], 
                           Xte = lambda x: X[x['testIdx']], 
                           ytr = lambda x: y[x['trainIdx']], 
                           yte = lambda x: y[x['testIdx']])
                 )
        self.grid = (self.grid_setup.merge(splits, 'left', on='cv_fold')
                     .asapRow(estimator= lambda x: self.estimator.copy()))
        return self
        
    def _setupFoldsDataFrame(self, X, y, mapper):
        '''Make indices for cv-folds
        mapper: sklearn_pandas.DataFrameMapper
        '''
        splits = list(self.cv.split(X))
        trainIdx, testIdx = list(zip(*splits))
        splits = (
            DataFrame(dict(trainIdx=trainIdx, testIdx=testIdx, cv_fold=range(self.cv.n_splits)))
            .asap(mapper=lambda x: deepcopy(mapper).fit(X.iloc[x['trainIdx']]))
            .asap(
                Xtr=lambda x: x['mapper'].transform(X.iloc[x['trainIdx']]),
                Xte=lambda x: x['mapper'].transform(X.iloc[x['testIdx']]),
                ytr=lambda x: y[x['trainIdx']],
                yte=lambda x: y[x['testIdx']],
            )
        )
        self.grid = (self.grid_setup.merge(splits, 'left', on='cv_fold')
                     .asapRow(estimator=lambda x: self.estimator.copy()))
        return self
    
    def fit(self, X, y, mapper=None, cvVerbose=1, **kwargs):
        '''Fit cv fold (full experiment).
        mapper: sklearn_pandas.DataFrameMapper
        cvVerbose: Verbose for each fold, or something.
        **kwargs are passed to the keras fit method to overwrite such things as epochs and batch size.
        TODO: 
        - Give option epoch maybe.
        - Should be able to call fit again to do more epochs.
        '''
        self.setupFolds(X, y, mapper)

        def fit(x):
            fit.it += 1
            if cvVerbose == 1:
                print(fit.it, 'of', len(self.grid))
            return x['estimator'].fit(x['Xtr'], x['ytr'], x['Xte'], x['yte'], **kwargs)
        fit.it = 0

        self.grid = (self.grid
                     .asap(estimator=lambda x: x['estimator'].build(**self._getParameterDictForRow(x)))
                    #  .asapRow(estimator=lambda x: x['estimator'].fit(x['Xtr'], x['ytr'], x['Xte'], x['yte'], **kwargs))
                     .asap(estimator=fit)
                     .assign_unzip('estimator', ['log', 'estimator'])
                    )
        return self


    def _getParameterDictForRow(self, row):
        '''Returns a dictionary of model parameters for one row in self.grid.
        row: a pd.Series object (e.g. self.grid.iloc[0]).
        '''
        return {col: row[col] for col in self.param_grid.keys()}

    
    def getHistory(self, indexName='epoch'):
        '''Only for Keras.
        Returns unnested history of all models.
        indexName: Name of index in nested history objects.
        '''
        history = (self.grid
                   .asapRow(log=lambda x: (DataFrame(x['log'].history)
                                           .reset_index()
                                           .rename(columns={'index': indexName})))
                   [['log']+list(self.grid_setup.columns)]
                   .unnest('log'))
        return history
    
    
    def getHistoryCvAvg(self, metricName='val_loss', indexName='epoch'):
        '''Only for Keras.
        Returns history of all models, averaged over each cv fold.
        metricName: Metric in history objects that should be used.
        indexName: Name of index in nested history ojbects.
        '''
        scores = (self.getHistory(indexName)
                  .groupby(['parConfigId']+self.cvParameters + ['epoch'])
                  [[metricName]]
                  .agg(['mean', 'std'])
                  .reset_index()
                  .sort_values([(metricName, 'mean')])
                 )
        return scores
