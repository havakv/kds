"""
Helper functions for keras.
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import Model
import keras
import itertools


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
    def __init__(self, layer_names=None):
        if layer_names is not None: assert hasattr(layer_names, '__iter__')
        self.layer_names = layer_names
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
            self.weights.append(weights)
        else:
            weights = []
            for layer in self.layer_names:
                w_list = [keras.backend.get_value(w) for w in self.layer_dict[layer].trainable_weights]
                weights.append((layer, w_list))
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
        self._append_weights()

    def on_batch_end(self, batch, logs={}):
        """Function called by kera's fit function.
        """
        self._append_weights()

    def make_diff(self):
        """Make self.weights_diff to be  the difference between weights in each iterations.
        """
        self.weights_diff = []
        for prev, curr in zip(self.weights[:-1], self.weights[1:]):
            weights_diff = []
            for lay_p, lay_c in zip(prev, curr):
                weights_diff.append((lay_p[0], [wc - wp for wp, wc in zip(lay_p[1], lay_c[1])]))
            self.weights_diff.append(weights_diff)

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
        stats = self._get_scale_stats_list()
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

    def hist_scale_stats(self, column=['ratio'], *args, **kwargs):
        """Histograms for displaying scale stats over all batches.
        *args, and **kwargs are passed to the histogram function.
        By passing only the name 'norm_w', 'norm_diff' or 'ratio', only the relevant plots are created.
        """
        if self.scale_stats is None:
            self.get_scale_stats()
        return self.scale_stats.hist(column=column, *args, **kwargs)

    def bar_mean_scale_stats(self, y='ratio', title='set_title', **kwargs):
        """Bar plot over average scale stats.
        The stats are the 2-norm over the weights (norm_w), the difference in the updated weights (norm_diff),
        and the ration between the two (ratio) (norm_diff/norm_w).
        See function get_scale_stats for further explanation.

        y : columns in self.scale_stats we want to plot.
        """
        if title == 'set_title':
            title = 'Average ' + y
        if self.scale_stats is None:
            self.get_scale_stats()
        return self.scale_stats.groupby('weight_name').mean().reset_index()\
                .plot(x='weight_name', y=y, kind='bar', title=title, **kwargs)

    def look_for_small_and_large_updates(self, relative=True):
        """A function for investigating small and large updates in the weights. 
        relative : if True, the update/weight is used instead of just the update.
        """
        pass


                


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
    for layer in model.layers:
        print('----', layer.name)
        print('input_shape  :', layer.input_shape)
        print('output_shape :', layer.output_shape)
        print('')
        

