"""
Implementation of the classical median cut algorithm for color quantization.
"""

import numpy as np

import omrsynth._sandbox.nputils as nputils


class ColorBox(object):
    _colors_min = np.array([0, 0, 0])
    _colors_max = np.array([255, 255, 255])

    def __init__(self, colors):
        self._colors = colors
        self.resize()

    @property
    def size(self):
        return self._colors_max - self._colors_min

    @property
    def average(self):
        # Return the average color converted to floats in range [0.0; 1.0]:
        return self._colors.mean(axis=0) / 255.0

    def resize(self):
        self._colors_min = self._colors.min(axis=0)
        self._colors_max = self._colors.max(axis=0)

    def fill_values(self, color_ints, color_inds, index, start_index):
        num_colors = self._colors.shape[0]
        end_index = start_index + num_colors

        color_ints[start_index:end_index] = nputils.rgb2int(self._colors)
        color_inds[start_index:end_index] = index

        return end_index

    def fill_colors(self, colors, last_start):
        num_colors = self._colors.shape[0]
        end = last_start + num_colors
        colors[last_start:end, :] = self._colors
        return end

    def split(self, axis):
        assert axis < self._colors.shape[1], "Not a valid axis."

        # Sort colors according to the axis:
        self._colors = self._colors[self._colors[:, axis].argsort()]

        # Find median index:
        med_idx = self._colors.shape[0] // 2

        # Create and return split parts:
        cube_lt = ColorBox(self._colors[:med_idx])
        cube_gt = ColorBox(self._colors[med_idx:])
        return cube_lt, cube_gt


def median_cut(image, num_colors=256):
    nputils.assert_ndarray(image)
    nputils.assert_array_dims(image, 3)
    nputils.assert_subdtype(image, np.uint8)

    im_h, im_w, im_dims = image.shape
    image_array = image.reshape(-1, im_dims)
    #unique_colors = nputils.np_unique_rows(image_array)
    unique_colors = image_array

    # Ensure the number of indexed colors is not greater than the number
    # of colors contained in the image:
    image_distinct_colors = len(unique_colors)
    if image_distinct_colors < num_colors:
        num_colors = image_distinct_colors

    # Initialize the algorithm with one ColorBox containing all the colors:
    boxes = [ColorBox(unique_colors)]

    # While there is not enough indexed colors (represented by the boxes),
    # continue by splitting of the box with maximal extent.
    while len(boxes) < num_colors:
        boxes_max_size = 0
        max_dim = 0

        for index, box in enumerate(boxes):
            size = box.size
            max_size, max_dim = size.max(), size.argmax()

            if max_size > boxes_max_size:
                boxes_max_size = max_size
                max_box = index

        split_box = boxes[max_box]
        box_a, box_b = split_box.split(max_dim)
        boxes = boxes[:max_box] + [box_a, box_b] + boxes[max_box + 1:]

    # Construct two arrays: the first with all the distinct colors transformed
    # into their integer representation and the second with corresponding
    # quantization label indices.
    color_ints = np.empty(shape=(image_distinct_colors, ), dtype=np.int32)
    color_inds = np.empty(shape=(image_distinct_colors, ), dtype=np.int32)
    start_idx = 0
    for index, box in enumerate(boxes):
        start_idx = box.fill_values(color_ints, color_inds, index, start_idx)

    # Transform the input array to the quantized color indices:
    image_array_ints = nputils.rgb2int(image_array)
    image_array_indexed = np.zeros(shape=image_array_ints.shape, dtype=np.uint8)

    for idx, rgb_int in enumerate(color_ints):
        image_array_indexed[image_array_ints == rgb_int] = color_inds[idx]

    image_indexed = image_array_indexed.reshape(im_h, im_w)

    # Compute labels (palette) for quantized colors:
    labels = np.vstack([box.average for box in boxes])

    return image_indexed, labels
