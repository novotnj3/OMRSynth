"""
This module is determined for combining the individual layers into
the resulting synthetic image.
"""

from .util import blendops
import util.general


def combine(degraded_sheet, background, alpha=1.0, alpha_multiply=0.9):
    """
    Experimental combination of a degraded sheet layer with a background layer.
    :param degraded_sheet:
    :param background:
    :param alpha:
    :param alpha_multiply:
    :return:
    """

    # Enable blending with RGB by converting grayscale to 3 channels:
    sheet = util.general.grayscale_to_3_channels(degraded_sheet)

    # Perform experimental blending:
    mult = blendops.multiply(sheet, background, alpha=alpha_multiply)
    hard = blendops.hard_light(sheet, background)
    image = blendops.multiply(mult, hard)

    return blendops.normal(image, background, alpha=alpha)
