"""
This module should simulate ink layer of music sheets...
TODO: Print vs. handwritten ink; degradation parameters and simulation.
"""

import numpy as np
import cv2
import util


def basic_print(image, level=0.1, blur_size=1):
    """
    Initial try of a very simple print simulation.
    To be changed...
    :param image: The input binary image of the music score:
    white background and black foreground are assumed.
    :param level: The amount of degradation.
    :param blur_size: Size of the blur kernel.
    :return: Grayscale image of the simulated print.
    """
    shape = image.shape

    # Salt noise:
    noise = np.random.rand(shape[0], shape[1])
    noise[noise < (1 - level)] = 0.0
    image = noise + image
    util.general.clip_image(image)

    # Pepper noise:
    noise = np.random.rand(shape[0], shape[1])
    image[noise < 0.0005 * level] = 0.0

    # Dilate with box structuring element:
    # TODO: Revise, not working as expected!
    # struct_elem = np.ones((2, 2), dtype=np.uint8)
    # image = cv2.dilate(image, struct_elem)

    # Ensure odd kernel size and blur the result:
    if blur_size % 2 == 0:
        blur_size += 1

    image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    return image
