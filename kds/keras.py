"""
Helper functions for keras.
"""

import numpy as np
from PIL import Image


def load_and_convert_images(filenames, size, color='L'):
    """
    Loading and converting images so they will fit in Keras net.
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
    """
    Get original image from converted image.
    """
    return Image.fromarray((img*255).astype('uint8')).resize(size)