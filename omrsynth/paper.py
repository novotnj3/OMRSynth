"""
The module implements procedural simulation of paper background textures.
"""

import numpy as np
import colorsys
import cv2
import skimage.util
import skimage.color
import util.blendops
import util.general
import util.noises


def _base_color_rgb(tint='random'):
    """
    Get a base paper texture color.
    :param tint: String determining what color will be returned,
    whether one of the predefined or a random one.
    :return: A chosen or random paper base color in RGB.
    """

    t = tint.lower()

    # Predefined colors based on measurements of four real sheet music papers:
    if t == 'lightest':
        return [0.95686275,  0.94901961,  0.95294118]
    elif t == 'light_yellow':
        return [0.98039216,  0.96862745,  0.8627451]
    elif t == 'light_brown':
        return [0.9372549,  0.89411765,  0.85098039]
    elif t == 'dark_brown':
        return [0.88627451,  0.76862745,  0.65098039]

    # Random color of a reasonable paper tint,
    # generated in HSV, returned in RGB:
    elif t == 'random':
        h = np.random.randint(20, 60) / 360.
        max_saturation = 30

        # Desaturate unwanted reddish colors:
        if h < 0.25:
            max_saturation = 15

        s = np.random.randint(1, max_saturation) / 100.
        v = np.random.randint(80, 100) / 100.

        return colorsys.hsv_to_rgb(h, s, v)

    # Unknown parameter given:
    else:
        raise(ValueError("Unknown tint '{tint}'!".format(tint=tint)))


def _blurred_noise(shape, kernel_size, y_min=0.0, y_max=1.0):
    """
    Generates a blurred uniform noise image, potentially linearly transformed.
    :param shape: 2D tuple containing image size: (height, width)
    :param kernel_size: Size of the kernel used for blurring.
    :param y_min: Parameter of the linear_transform function (see util.general).
    :param y_max: Parameter of the linear_transform function (see util.general).
    :return: A blurred uniform noise image of given shape with 3 identical channels.
    """

    # Ensure kernel_size is an odd number:
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Generate noise and blur it:
    noise = np.random.rand(shape[0], shape[1])
    noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)
    noise = util.general.linear_transform(noise, y_min=y_min, y_max=y_max)

    # Copy grayscale to three channels:
    return util.general.grayscale_to_3_channels(noise)


def _random_add(scale=1.0):
    """
    Generates a random value between -1.0 and 1.0 and scale it by given amount.
    :param scale: Multiplier of a generated random value.
    :return: Uniformly distributed random value in the range of scale * [-1.0; 1.0]
    """
    return float(scale * (2 * np.random.rand() - 1))


def _high_freq_hsv_noise(shape, shift_x=0.0, shift_y=0.0, rnd_scale=0.0):
    """
    Generates high frequency noise using HSV color space.
    :param shape: 2D tuple of a desired image size: (height, width)
    :param shift_x: Noise shift in x-axis.
    :param shift_y: Noise shift in y-axis.
    :param rnd_scale: Randomness scale.
    :return: RGB image of HF noise with small amplitudes.
    Useful for addition or subtraction with an image.
    """

    # WARNING: All the hardcoded constants are empirical values only!

    # Make mask from the simplex noise:
    mask = util.noises.simplex_noise(shape=shape, octaves=2,
                                     shift_x=shift_x, shift_y=shift_y,
                                     frequency_x=4.0, frequency_y=4.0)
    # Randomly choose threshold (affecting amount of masked areas):
    t = 0.5 + _random_add(rnd_scale)
    mask[mask > t] = 1.0
    mask[mask <= t] = 0.0

    # Fill mask with a random noise of desired properties:
    noise = np.random.rand(shape[0], shape[1])

    salt_amount = 0.985 + _random_add(0.1 * rnd_scale)
    salt = skimage.util.img_as_float(noise > salt_amount)

    y_max_randomized = 0.75 + _random_add(rnd_scale)
    noise = util.general.linear_transform(noise, x_min=0, x_max=1, y_max=y_max_randomized)
    noise = noise + salt
    util.general.clip_image(noise)

    # TODO: Flatten midrange...

    # Hue channel comes from the generated noise but only in masked areas:
    hue = np.multiply(mask, noise)

    # Generate a new mask from hue channel:
    mask = skimage.util.img_as_float(hue > 0)

    # Perturb new mask a little by adding Salt and Pepper noise:
    salt_amount = 0.94 + _random_add(rnd_scale * 0.5)
    salt = skimage.util.img_as_float(np.random.rand(shape[0], shape[1]) > salt_amount)
    mask += salt
    util.general.clip_image(mask)
    pepper_amount = 0.04 * _random_add(rnd_scale * 0.1)
    pepper = 1 - skimage.util.img_as_float(np.random.rand(shape[0], shape[1]) < pepper_amount)
    mask *= pepper

    # Value channel results from the mask.
    # The mask itself will be further used, must copy!
    value = np.copy(mask)
    value = cv2.GaussianBlur(value, (3, 3), 0)
    x_min_randomized = 0.25 + _random_add(rnd_scale * 0.2)
    value = util.general.linear_transform(value, y_min=0, y_max=1, x_min=x_min_randomized, x_max=1)
    value = np.multiply(np.random.rand(shape[0], shape[1]), value)
    # Copy value image (for further usage) and transform original as desired
    value_copy = np.copy(value)
    value = util.general.linear_transform(value, y_min=0, y_max=0.1)

    # Bright spots:
    bs = np.random.rand(shape[0], shape[1])
    bs = util.general.linear_transform(bs, y_min=0, y_max=1, x_min=0.995, x_max=1)
    kernel_blur = np.matrix([[1, 0.5, 0], [0, 2, 0.5], [0, 0.5, 1]])
    bs = cv2.filter2D(bs, -1, kernel_blur)
    util.general.clip_image(bs)
    bs = cv2.GaussianBlur(bs, (3, 3), 0)

    mixed = util.blendops.darken_only(bs, value)
    mixed = util.general.linear_transform(mixed, y_min=0, y_max=1, x_min=0, x_max=0.55)
    value = util.blendops.screen(mixed, value)
    # Finally, value channel is prepared now...

    # Model saturation channel from copied images:
    y_max_randomized = 0.85 + _random_add(rnd_scale * 0.4)
    x_min_randomized = 0.45 + _random_add(rnd_scale * 0.5)
    value_copy = util.general.linear_transform(value_copy, y_min=0, y_max=y_max_randomized, x_min=x_min_randomized, x_max=1)
    saturation = util.blendops.difference(value_copy, mask)

    # Form the final HSV noise image:
    hsv_noise = np.empty(shape + (3, ), dtype='float')
    hsv_noise[..., 0] = hue
    hsv_noise[..., 1] = saturation
    hsv_noise[..., 2] = value

    return skimage.color.hsv2rgb(hsv_noise)


def basic_texture(shape, tint='random', seed=0, scale=1.0, bright_scale=1.0, random_scale=0.2):
    """
    Generates a random background texture.
    :param shape: 2D tuple of a desired texture size: (height, width)
    :param tint: String parameter of a color tint.
    :param seed: Numpy random seed.
    :param scale: Spatial scale of the generated texture.
    :param bright_scale: Brightness scale of generated noise.
    :param random_scale: Scale of "randomness".
    :return: Random RGB paper background texture image.
    """

    # WARNING: All the hardcoded constants are empirical values only!

    # Set the random seed (to ensure replication)
    np.random.seed(seed)

    # Introduce randomness to scale values:
    scale += _random_add(random_scale)
    bright_scale += _random_add(random_scale)
    hf_random_scale = random_scale + _random_add(0.25)
    hf_random_scale = util.general.clip_number(hf_random_scale, min_value=0.05, max_value=0.4)

    # At first, fill texture with one base color only:
    texture = np.zeros(shape + (3, ), dtype=float)
    texture[:, :] = _base_color_rgb(tint)

    # Define blur kernel size:
    kernel_size = int(scale * 0.04 * min(shape[0], shape[1]))

    # Low frequency noise to darken some places:
    y_min = 0.65 + 0.2 * _random_add(random_scale)
    low_f_noise = _blurred_noise(shape, kernel_size, y_min=y_min)
    texture = util.blendops.burn(low_f_noise, texture, alpha=0.6 * bright_scale)

    # Then again low frequency noise to darken in a different way
    y_min = 0.45 + 0.1 * _random_add(random_scale)
    low_f_noise = _blurred_noise(shape, kernel_size, y_min=y_min)
    texture = util.blendops.multiply(low_f_noise, texture, alpha=0.12 * bright_scale)

    # Medium frequency noise to brighten some places a little bit
    kernel_size //= 2
    y_max = 0.1 * 0.05 * _random_add(random_scale)
    med_f_noise = _blurred_noise(shape, kernel_size, y_max=y_max)
    texture = util.blendops.addition(med_f_noise, texture, alpha=0.85 * bright_scale)
    util.general.clip_image(texture)

    # High frequency additive noises:
    additive_hf = _high_freq_hsv_noise(shape, rnd_scale=hf_random_scale)
    subtractive_hf = _high_freq_hsv_noise(shape, shift_x=shape[1], shift_y=shape[0], rnd_scale=hf_random_scale)
    texture = texture + additive_hf - subtractive_hf
    util.general.clip_image(texture)

    return texture
