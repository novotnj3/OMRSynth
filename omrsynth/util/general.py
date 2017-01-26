"""
This module contains general utils that are not worth to separate into
individual modules (at the time of their creation, may be changed in future).
"""

import numpy as np


def clip_number(number, min_value=0, max_value=1):
    """
    Clips a number between given values.
    :param number: Number to be clipped.
    :param min_value: Min value (lower values will be set to this value).
    :param max_value: Max value (higher values will be set to this value).
    :return: Number clipped between min and max values.
    """
    return max(min(number, max_value), min_value)


def clip_image(image):
    """
    Clips values of an image to be between 0.0 and 1.0.
    :param image: Image that will be clamped in place.
    """
    np.clip(image, 0.0, 1.0, out=image)


def repeat_to_3_channels(image):
    """
    Converts an image to three channels by copying its values along dimensions.
    :param image: An image to be converted.
    :return: 3-channel image constructed by copying values along 3rd axis.
    """
    return np.repeat(image[:, :, np.newaxis], 3, axis=2)


def linear_transform(image, x_min=None, x_max=None, y_min=0.0, y_max=1.0):
    """
    Performs a linear transform of image values according to the equation:
    y_min + (image - x_min) * ((y_max - y_min) / (x_max - x_min))
    :param image: An image to be transformed.
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :return: The linearly transformed image.
    Please see TODO (href to a graph with explanation).
    """

    # Handle default cases
    if x_min is None:
        x_min = image.min()

    if x_max is None:
        x_max = image.max()

    # Transform the image and ensure the valid values range
    transformed = y_min + (image - x_min) * ((y_max - y_min) / (x_max - x_min))
    clip_image(transformed)

    return transformed
