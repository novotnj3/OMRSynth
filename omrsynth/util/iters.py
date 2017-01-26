"""
Implements several iterators over pixels of images.
"""

import itertools
import math
import numba
import numpy as np


def hilbert(h, w, start=0):
    # Take greater dimension
    size = max(h, w)

    # Nearest successive power of two
    n = 2 ** int(math.log(size - 1, 2) + 1)

    d = start
    while d < n ** 2:
        col, row = _d2xy(n, d)
        if col < w and row < h:
            yield row, col

        d += 1


def scanline(h, w, start=0):
    order = np.ndindex((h, w))
    next(itertools.islice(order, start - 1, start))
    return order


def serpentine(h, w, start=0):
    for row, col in scanline(h, w, start):
        if row % 2 == 0:
            yield row, col
        else:
            yield row, w - 1 - col


def random(h, w, start=0):
    raise(NotImplementedError, 'Conflict with current _find_nearest function.')

    size = h * w
    random_indices = np.random.permutation(size)
    i = start
    while i < size:
        index = random_indices[i]
        i += 1
        row, col = index // w, index % w
        yield row, col


@numba.jit(nopython=True)
def _d2xy(n, d):
    t = d
    x = 0
    y = 0
    s = 1

    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2

    return x, y


@numba.jit(nopython=True)
def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y

        return y, x

    return x, y
