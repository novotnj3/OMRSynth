"""
This module implements several enhancement operations on images.
All the images are expected to be Numpy arrays of floats between 0.0 and 1.0
and of the same dimensions. The output is an image of the same size.
"""

import numpy as np
import skimage.color

def white_balance(image, percentage=0.006):
    """
    White balance the image with the same algorithm as used in Gimp.
    See https://docs.gimp.org/en/gimp-layer-white-balance.html
    """

    image_uint8 = (255.0 * image).astype(np.uint8)

    pixels_total = image.shape[0] * image.shape[1]
    threshold = percentage * pixels_total

    _stretch_values(image_uint8, 0, threshold)
    _stretch_values(image_uint8, 1, threshold)
    _stretch_values(image_uint8, 2, threshold)

    return image_uint8 / 255.0


def _stretch_values(image, axis, threshold):
    channel = image[:, :, axis]

    hist, _ = np.histogram(channel, bins=256, range=(0, 256))

    cs = hist.cumsum()
    constraint = np.logical_and(cs > threshold, _gt_previous(cs))
    min_val = np.argmax(constraint)

    cs_rev = hist[::-1].cumsum()
    constraint = np.logical_and(cs_rev > threshold, _gt_previous(cs_rev))
    max_val = 255 - np.argmax(constraint)

    # Discard out of range pixels
    channel[channel < min_val] = min_val
    channel[channel > max_val] = max_val

    if max_val != min_val:
        val_range = float(max_val - min_val)
        image[:, :, axis] = 255.0 * (channel - min_val) / val_range


def _gt_previous(array):
    return np.r_[False, array[1:] > array[:-1]]


def hsv_scale(image, sat_scale=1.0, hue_shift=0.0):
    image_hsv = skimage.color.rgb2hsv(image)
    image_hsv[:, :, 1] += hue_shift
    image_hsv[:, :, 1] *= sat_scale
    return skimage.color.hsv2rgb(image_hsv)
