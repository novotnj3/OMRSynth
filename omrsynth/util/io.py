"""
This module implements several shortcuts for loading and saving images.
"""

import skimage.io
import skimage.util
import skimage.color


def load_binary_float(filename):
    """
    Load a binary image from a given file.
    :param filename: Filename of a binarized image.
    :return: Binary image as a numpy array of floats with values
    between 0.0 and 1.0.
    """

    image = skimage.io.imread(filename)
    image = skimage.color.rgb2gray(image)

    # Ensure binary values:
    binary = image >= 0.5 * (image.max() - image.min())

    return skimage.util.img_as_float(binary)


def load_rgb_float(filename):
    """
    Load an RGB image from a given file.
    :param filename: Filename of an image.
    :return: RGB image as a numpy array of floats with values
    between 0.0 and 1.0.
    """

    image = skimage.util.img_as_float(skimage.io.imread(filename))

    # Ensure only RGB is loaded:
    return image[:, :, 0:3]


def load_float(filename):
    """
    Load an multichannel image from a given file.
    :param filename: Filename of an image.
    :return: Image with all channels preserved loaded as a numpy array
    of floats with values between 0.0 and 1.0.
    """

    return skimage.util.img_as_float(skimage.io.imread(filename))
