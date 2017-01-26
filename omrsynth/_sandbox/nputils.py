"""
Implements several utils designed for NumPy arrays.
"""

import numpy as np


def assert_ndarray(array):
    assert type(array) is np.ndarray, "Not a valid NumPy array given!"


def assert_array_dims(array, dims):
    assert array.ndim == dims, "Not a {0}-dimensional array!".format(dims)


def assert_subdtype(array, dtype):
    assert np.issubdtype(array.dtype, dtype), \
        "Array values are not in {0}!".format(dtype)


def np_unique_rows(array):
    """
    Returns an array created from the rows of an input array, where all
    duplicate rows are eliminated.
    :param array: An array from which the rows are selected.
    :return: A NumPy array with distinct rows.
    """

    assert_ndarray(array)
    assert_array_dims(array, 2)

    array_contiguous = np.ascontiguousarray(array).\
        view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))

    array_unique = np.unique(array_contiguous).view(array.dtype).\
        reshape(-1, array.shape[1])

    return array_unique


def rgb2int(rgb_array):
    """
    Converts an array filled with RGB color values into integers.
    :param rgb_array: An array to be converted.
    :return: One dimensional array with integer-packed color values.
    """

    assert_ndarray(rgb_array)
    assert_array_dims(rgb_array, 2)
    assert rgb_array.shape[1] == 3, "Not a valid RGB array!"

    return rgb_array.dot(np.array([65536, 256, 1], dtype=np.int32))
