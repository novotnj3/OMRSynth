"""
In this module, Perlin's original and Simplex noise are
encapsulated to simplify their usage.
"""

import noise
import numpy as np


def simplex_noise(shape,
                  shift_x=0.0, shift_y=0.0,
                  frequency_x=1.0, frequency_y=1.0,
                  octaves=1, persistence=0.5, lacunarity=2.0):
    """
    Returns Perlin's Simplex noise array of a given shape.
    :param shape: 2D tuple indicating image dimensions: (height, width)
    :param shift_x: Shift of the noise in the x-axis.
    :param shift_y: Shift of the noise in the y-axis.
    :param frequency_x: Frequency of the noise in the x-axis.
    :param frequency_y: Frequency of the noise in the y-axis.
    :param octaves: Number of coherent noise functions used (called octaves).
    :param persistence: A multiplier that determines how quickly the amplitudes
    diminish for each successive octave in the noise.
    :param lacunarity: A multiplier that determines how quickly the frequency
    increases for each successive octave in a the noise.
    :return: Numpy array of given shape filled with Perlin's noise of given
    parameters and values between 0.0 and 1.0.
    """

    noise_image = np.empty(shape, dtype='float')

    f_x = frequency_x * octaves
    f_y = frequency_y * octaves

    for y in range(shape[0]):
        for x in range(shape[1]):
            noise_core = noise.snoise2(
                x=(x + shift_x) / f_x,
                y=(y + shift_y) / f_y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity)

            # Mapping [-1.0; 1.0] -> [0.0; 1.0]
            noise_image[y, x] = 0.5 * (noise_core + 1)

    return noise_image


def perlin_noise(shape,
                 shift_x=0.0, shift_y=0.0,
                 frequency_x=1.0, frequency_y=1.0,
                 octaves=1, persistence=0.5, lacunarity=2.0):
    """
    Returns Perlin's original noise array of a given shape.
    :param shape: 2D tuple indicating image dimensions: (height, width)
    :param shift_x: Shift of the noise in the x-axis.
    :param shift_y: Shift of the noise in the y-axis.
    :param frequency_x: Frequency of the noise in the x-axis.
    :param frequency_y: Frequency of the noise in the y-axis.
    :param octaves: Number of coherent noise functions used (called octaves).
    :param persistence: A multiplier that determines how quickly the amplitudes
    diminish for each successive octave in the noise.
    :param lacunarity: A multiplier that determines how quickly the frequency
    increases for each successive octave in a the noise.
    :return: Numpy array of given shape filled with Perlin's noise of given
    parameters and values between 0.0 and 1.0.
    """

    noise_image = np.empty(shape, dtype='float')

    f_x = frequency_x * octaves
    f_y = frequency_y * octaves

    for y in range(shape[0]):
        for x in range(shape[1]):
            noise_core = noise.pnoise2(
                x=(x + shift_x) / f_x,
                y=(y + shift_y) / f_y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity)

            # Mapping [-1.0; 1.0] -> [0.0; 1.0]
            noise_image[y, x] = 0.5 * (noise_core + 1)

    return noise_image
