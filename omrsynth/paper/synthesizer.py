"""
Implements a base class with basic functionality for texture synthesis
algorithms.
"""

import logging
import numpy as np
import os.path

import omrsynth.util.iters as iters
import omrsynth.util.io as io


class Synthesizer(object):
    _quantized_colors = 256
    _pixel_orders = ['hilbert', 'scanline', 'serpentine', 'random']

    def __init__(self):
        self._random_seed = 0
        np.random.seed(self._random_seed)

        self._pixel_order = Synthesizer._pixel_orders[0]

    def random_seed(self, value):
        self._random_seed = value
        np.random.seed(value)
        return self

    def pixel_order(self, order):
        if order not in Synthesizer._pixel_orders:
            logging.warning('Unknown pixel order value {}!'.format(order))
        else:
            self._pixel_order = order

    def _pixel_generator(self, height, width, start):
        order_index = Synthesizer._pixel_orders.index(self._pixel_order)

        if order_index == 0:
            # Hilbert ordering
            return iters.hilbert(height, width, start)
        elif order_index == 1:
            # Scanline ordering
            return iters.scanline(height, width, start)
        elif order_index == 2:
            # Serpentine ordering
            return iters.serpentine(height, width, start)
        elif order_index == 3:
            # Random permutation
            return iters.random(height, width, start)

    def generate_texture(self, sample_filename, out_filename, out_shape):
        self._load_sample(sample_filename)
        out_origins = self._generate_texture(out_shape)
        self._save_results(out_origins, out_filename)

    def load_sample(self, sample_filename):
        self._load_sample(sample_filename)

    def generate_loaded_texture(self, out_filename, out_shape):
        out_origins = self._generate_texture(out_shape)
        self._save_results(out_origins, out_filename)

    def _load_sample(self, filename):
        # Load and quantize sample image
        num_colors = Synthesizer._quantized_colors
        sample, indexed, labels = io.load_and_quantize(filename, num_colors)
        self._sample = sample
        self._sample_indexed = indexed
        self._sample_indexed_ravel = indexed.ravel()
        self._index_labels = labels
        logging.info('Sample image loaded.')

        # Save quantized sample image for visual comparison
        quantized_sample = np.take(labels, indexed, axis=0)
        colors = 'indexed_{0}_colors'.format(Synthesizer._quantized_colors)
        filename_quantized = Synthesizer._add_to_filename(filename, colors)
        io.save_image(filename_quantized, quantized_sample)

        # Color cube part
        self._create_ind2rgb(filename)
        logging.info('Sample indices to RGB image created.')

    def _create_ind2rgb(self, filename):
        h, w = self._sample.shape[0:2]
        r, g = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        b = np.zeros_like(r)
        rgb_face = np.dstack((r, g, b))
        self._sample_ind2rgb = rgb_face.reshape(h * w, 3)
        filename = Synthesizer._add_to_filename(filename, 'rgb_map')
        io.save_image(filename, rgb_face)

    def _save_results(self, out_origins, filename):
        texture, indices = self._transform_origins(out_origins)
        io.save_image(filename, texture)
        indices_filename = Synthesizer._add_to_filename(filename, 'origins')
        io.save_image(indices_filename, indices)

    @staticmethod
    def _add_to_filename(filename, text):
        pre, ext = os.path.splitext(filename)
        return '{0}_{1}{2}'.format(pre, text, ext)

    def _generate_texture(self, out_shape):
        raise NotImplementedError

    def _transform_origins(self, out_origins):
        sample = self._sample
        sample_height, sample_width = sample.shape[:2]
        sample_size = sample_height * sample_width
        sample = sample.reshape(sample_size, -1)
        out_image = np.take(sample, out_origins, axis=0)
        out_rgb_inds = np.take(self._sample_ind2rgb, out_origins, axis=0)

        return out_image, out_rgb_inds
