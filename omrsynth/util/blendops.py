"""
This module implements several blending operations of two image layers.
All the images are expected to be Numpy arrays of floats between 0.0 and 1.0
and of the same dimensions. The output is an image of the same size.
See https://en.wikipedia.org/wiki/Blend_modes for more information.
"""

import numpy as np
import skimage.color
import general


def normal(im_a, im_b, alpha=0.5):
    """
    Perform a classical alpha blending between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: (1 - alpha) * im_b + alpha * im_a
    """

    # Alpha must be in the range between 0.0 and 1.0:
    alpha = general.clip_number(alpha)
    return (1 - alpha) * im_b + alpha * im_a


def multiply(im_a, im_b, alpha=1.0):
    """
    Perform a multiplication operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: im_a .* im_b
    """

    im_mul = np.multiply(im_a, im_b)
    return normal(im_mul, im_b, alpha)


def screen(im_a, im_b, alpha=1.0):
    """
    Perform a screen operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: 1 - (1 - im_a) .* (1 - im_b)
    """

    im_scr = 1 - np.multiply(1 - im_a, 1 - im_b)
    return normal(im_scr, im_b, alpha)


def overlay(im_a, im_b, alpha=1.0):
    """
    Perform an overlay operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: 2 * im_a .* im_b where im_b < 0.5 and
    1 - 2 * (1 - im_a) .* (1 - im_b) otherwise
    """

    mask = im_b >= 0.5
    im_over = 2 * np.multiply(im_a, im_b)
    lighter = 1 - 2 * np.multiply(1 - im_a, 1 - im_b)
    im_over[mask] = lighter[mask]

    return normal(im_over, im_b, alpha)


def hard_light(im_a, im_b, alpha=1.0):
    """
    Perform a hard light operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: 2 * im_a .* im_b where im_a < 0.5 and
    1 - 2 * (1 - im_a) .* (1 - im_b) otherwise
    """

    mask = im_a >= 0.5
    im_over = 2 * np.multiply(im_a, im_b)
    lighter = 1 - 2 * np.multiply(1 - im_a, 1 - im_b)
    im_over[mask] = lighter[mask]

    return normal(im_over, im_b, alpha)


def soft_light(im_a, im_b, alpha=1.0):
    """
    Perform a soft light operator between two images.
    Pegtop's definition is used, see:
    http://www.pegtop.net/delphi/articles/blendmodes/softlight.htm
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: (1 - im_b) .* multiply(im_a, im_b) + im_b .* screen(im_a, im_b)
    """

    im_soft = (1 - im_b) * multiply(im_a, im_b) + np.multiply(im_b, screen(im_a, im_b))
    return normal(im_soft, im_b, alpha)


def darken_only(im_a, im_b, alpha=1.0):
    """
    Perform a darken only operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: element_wise_min(im_a, im_b)
    """

    im_darken = np.minimum(im_a, im_b)
    return normal(im_darken, im_b, alpha)


def lighten_only(im_a, im_b, alpha=1.0):
    """
    Perform a lighten only operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: element_wise_max(im_a, im_b)
    """

    im_lighten = np.maximum(im_a, im_b)
    return normal(im_lighten, im_b, alpha)


def addition(im_a, im_b, alpha=1.0):
    """
    Perform an addition blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: im_a + im_b, out of range values are clipped
    """

    # Add values and ensure the valid range:
    im_add = im_a + im_b
    general.clip_image(im_add)

    return normal(im_add, im_b, alpha)


def subtraction(im_a, im_b, alpha=1.0):
    """
    Perform an subtraction blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: im_b - im_a, out of range values are clipped
    """

    # Subtract values and ensure the valid range:
    im_sub = im_b - im_a
    general.clip_image(im_sub)

    return normal(im_sub, im_b, alpha)


def difference(im_a, im_b, alpha=1.0):
    """
    Perform an difference blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: abs(im_b - im_a)
    """

    im_diff = np.absolute(im_b - im_a)
    return normal(im_diff, im_b, alpha)


def divide(im_a, im_b, alpha=1.0):
    """
    Perform a divide blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: im_b ./ im_a, out of range values are clipped
    """

    # Ignore divisions by zero and divide the images:
    with np.errstate(divide='ignore'):
        im_mul = np.divide(im_b, im_a)

    # Correct NaN values:
    im_mul[np.isnan(im_mul)] = 0
    general.clip_image(im_mul)

    return normal(im_mul, im_b, alpha)


def dodge(im_a, im_b, alpha=1.0):
    """
    Perform a color dodge operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: im_b ./ (1 - im_a), out of range values are clipped
    """

    # Ignore divisions by zero and divide the images:
    with np.errstate(divide='ignore'):
        im_dodge = np.divide(im_b, 1 - im_a)

    # Correct NaN values:
    im_dodge[np.isnan(im_dodge)] = 0
    general.clip_image(im_dodge)

    return normal(im_dodge, im_b, alpha)


def burn(im_a, im_b, alpha=1.0):
    """
    Perform a color burn operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: 1 - (1 - im_b) ./ im_a, out of range values are clipped
    """

    # Ignore divisions by zero and divide the images:
    with np.errstate(divide='ignore'):
        im_burn = 1 - np.divide(1 - im_b, im_a)

    # Correct NaN values:
    im_burn[np.isnan(im_burn)] = 0
    general.clip_image(im_burn)

    return normal(im_burn, im_b, alpha)


def hue(im_a, im_b, alpha=1.0):
    """
    Perform a hue blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: Hue taken from the active layer, saturation and value from the background.
    """

    im_hue_hsv = skimage.color.rgb2hsv(im_b)
    im_hue_hsv[:, :, 0] = skimage.color.rgb2hsv(im_a)[:, :, 0]
    im_hue = skimage.color.hsv2rgb(im_hue_hsv)

    return normal(im_hue, im_b, alpha)


def saturation(im_a, im_b, alpha=1.0):
    """
    Perform a saturation blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: Saturation taken from the active layer, hue and value from the background.
    """

    im_sat_hsv = skimage.color.rgb2hsv(im_b)
    im_sat_hsv[:, :, 1] = skimage.color.rgb2hsv(im_a)[:, :, 1]
    im_sat = skimage.color.hsv2rgb(im_sat_hsv)

    return normal(im_sat, im_b, alpha)


def value(im_a, im_b, alpha=1.0):
    """
    Perform a saturation blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: Value (HSV) taken from the active layer, hue and saturation from the background.
    """

    im_val_hsv = skimage.color.rgb2hsv(im_b)
    im_val_hsv[:, :, 2] = skimage.color.rgb2hsv(im_a)[:, :, 2]
    im_val = skimage.color.hsv2rgb(im_val_hsv)

    return normal(im_val, im_b, alpha)


def color(im_a, im_b, alpha=1.0):
    """
    Perform a color blend operator between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :return: Hue and saturation taken from the active layer, value from the background.
    """

    im_col_hsv = skimage.color.rgb2hsv(im_a)
    im_col_hsv[:, :, 2] = skimage.color.rgb2hsv(im_b)[:, :, 2]
    im_col = skimage.color.hsv2rgb(im_col_hsv)

    return normal(im_col, im_b, alpha)


def blend(im_a, im_b, alpha, operator):
    """
    Perform a blending between two images.
    :param im_a: Active image layer (i.e. the foreground layer).
    :param im_b: Background image layer (i.e. the underlying layer).
    :param alpha: Parameter affecting the opacity of active layer.
    :param operator: String name of an operator to be used.
    :return: Blended image according to the chosen operator.
    """

    op = operator.lower()

    if op == 'normal':
        return normal(im_a, im_b, alpha)
    elif op == 'multiply':
        return multiply(im_a, im_b, alpha)
    elif op == 'screen':
        return screen(im_a, im_b, alpha)
    elif op == 'overlay':
        return overlay(im_a, im_b, alpha)
    elif op == 'hard_light':
        return hard_light(im_a, im_b, alpha)
    elif op == 'soft_light':
        return soft_light(im_a, im_b, alpha)
    elif op == 'darken_only':
        return darken_only(im_a, im_b, alpha)
    elif op == 'lighten_only':
        return lighten_only(im_a, im_b, alpha)
    elif op == 'addition':
        return addition(im_a, im_b, alpha)
    elif op == 'subtraction':
        return subtraction(im_a, im_b, alpha)
    elif op == 'difference':
        return difference(im_a, im_b, alpha)
    elif op == 'divide':
        return divide(im_a, im_b, alpha)
    elif op == 'dodge':
        return dodge(im_a, im_b, alpha)
    elif op == 'burn':
        return burn(im_a, im_b, alpha)
    elif op == 'hue':
        return hue(im_a, im_b, alpha)
    elif op == 'saturation':
        return saturation(im_a, im_b, alpha)
    elif op == 'value':
        return value(im_a, im_b, alpha)
    elif op == 'color':
        return color(im_a, im_b, alpha)
    else:
        raise ValueError("Unknown operator name '{op}'!".format(op=operator))
