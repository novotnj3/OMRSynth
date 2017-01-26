"""
This module implements several shortcuts for loading and saving images.
"""

import numpy as np
import PIL.Image
import skimage.io
import skimage.util
import skimage.color

# Pillow quantization methods
MEDIAN_CUT = 0
MAX_COVERAGE = 1


def load_binary_float(filename):
    """
    Loads a binary image from a given file.
    :param filename: Filename of a binarized image.
    :return: Binary image as a NumPy array of floats with values
    between 0.0 and 1.0.
    """

    image = skimage.io.imread(filename)
    image = skimage.color.rgb2gray(image)

    # Ensure binary values
    binary = image >= 0.5 * (image.max() - image.min())

    return skimage.util.img_as_float(binary)


def load_rgb_float(filename):
    """
    Loads an RGB image from a given file.
    :param filename: Filename of an image.
    :return: RGB image as a NumPy array of floats with values
    between 0.0 and 1.0.
    """

    image = skimage.util.img_as_float(skimage.io.imread(filename))

    # Ensure only RGB is loaded
    return image[:, :, 0:3]


def load_float(filename):
    """
    Loads an multichannel image from a given file.
    :param filename: Filename of an image.
    :return: Image with all channels preserved loaded as a NumPy array
    of floats with values between 0.0 and 1.0.
    """

    return skimage.util.img_as_float(skimage.io.imread(filename))


def load_uint8(filename):
    """
    Loads an image from a given file.
    :param filename: Filename of an image.
    :return: The image as a NumPy array of uint8 with values between 0 and 255.
    """

    image = skimage.io.imread(filename)
    return image


def load_rgb_uint8(filename):
    """
    Loads an RGB image from a given file.
    :param filename: Filename of an image.
    :return: RGB image as a NumPy array of uint8 with values between 0 and 255.
    """

    image = skimage.io.imread(filename)

    # Ensure only RGB is loaded
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = image[:, :, 0:3]

    return image


def load_and_quantize(filename, num_colors=256, method=MEDIAN_CUT):
    """
    Loads an RGB image and performs quantization of its colors.
    :param filename: Filename of an image.
    :param num_colors: Number of quantized colors (max. 256).
    :param method: Method number of the Pillow quantization algorithm.
    (0 = median cut; 1 = maximum coverage; 2 = fast octree)
    :return: Triplet of NumPy arrays: (original, quantized, labels) where
    original contains an array of uint8 - the original image,
    quantized contains the quantized version of the image (indexed one) and
    labels contains uint8 colors as represented by the corresponding indices.
    """

    # Load an image as unsigned 8-bit integers
    im_original = load_rgb_uint8(filename)

    # Create an extra dimension if the input image is in grayscale
    if len(im_original.shape) < 3:
        im_original = im_original[:, :, np.newaxis]

    # Convert the image into Pillow format and quantize it using the median
    # cut method (empirically works best from the pillow options)
    im_pillow = PIL.Image.fromarray(im_original)
    im_pillow = im_pillow.quantize(colors=num_colors, method=method)

    im_quantized = np.asarray(im_pillow)
    labels = np.asarray(im_pillow.getpalette(), dtype=np.uint8).reshape(-1, 3)

    # If the input image contains less colors then demanded, crop the labels
    # to have a right size
    im_colors = im_quantized.max() + 1
    if im_colors < num_colors:
        labels = labels[:im_colors, :]

    return im_original, im_quantized, labels


def save_image(filename, image_array):
    skimage.io.imsave(filename, image_array)


def save_image_gray(filename, image_array):
    image_gray = skimage.color.rgb2gray(image_array)
    skimage.io.imsave(filename, image_gray)


def image_dims(filename):
    image = load_float(filename)

    if len(image.shape) < 3:
        return image.shape
    else:
        return image.shape[:3]
