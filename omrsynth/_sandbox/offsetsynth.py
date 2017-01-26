import logging
import numba
import numpy as np
import skimage.color
import skimage.io
import skimage.util

import omrsynth.util.hilbert as hilbert
import omrsynth.util.io as io


class OffsetSynth(object):
    _quantized_colors = 256
    _sim_colors = 16

    def __init__(self):
        # Fill default synthesizer parameters:
        self._half_window = 2
        self._num_nearest = 4
        self._random_seed = 0
        np.random.seed(self._random_seed)

    def random_seed(self, value):
        self._random_seed = value
        # Initialize NumPy's random:
        np.random.seed(value)
        return self

    def half_window(self, value):
        self._half_window = value
        return self

    def num_nearest(self, value):
        self._num_nearest = value
        return self

    def generate_texture(self, sample_filename, output_filename, out_shape):
        self._load_sample(sample_filename)
        texture = self._generate_texture(out_shape)
        io.save_image(output_filename, texture)

    def _load_sample(self, filename):
        logging.info('Loading sample image.')
        num_colors = OffsetSynth._quantized_colors
        sample, indexed, labels = io.load_and_quantize(filename, num_colors)

        self._sample = sample
        self._sample_indexed = indexed
        self._sample_indexed_ravel = indexed.ravel()
        self._index_labels = labels

        logging.info('Loading image for the equivalence map.')
        max_coverage = 1
        _, equiv, _ = io.load_and_quantize(filename,
                                           num_colors=OffsetSynth._sim_colors,
                                           method=max_coverage)

        # Equivalence map:
        equiv = equiv.ravel()
        equiv_map = []

        logging.info('Computing equivalence map.')
        for i in range(OffsetSynth._sim_colors):
            indices = np.where(equiv == i)[0]
            equiv_map.append(indices)
        logging.info('Equivalence map computed.')

        self._equiv_map = equiv_map
        self._sample_equiv = equiv
        self._prepare_metric()
        logging.info('Metric prepared.')
        self._prepare_sample_neighborhoods()
        logging.info('Sample neighborhoods prepared')
        self._prepare_jump_map()
        logging.info('Jump map prepared.')

    def _prepare_metric(self):
        colors = self._index_labels
        assert colors is not None, 'Indexed color labels must be filled!'
        logging.debug('Preparing color comparison matrix.')

        num_colors = colors.shape[0]

        # Transform the colors:
        # TODO: Try Lab color space?
        #colors_transformed = skimage.util.img_as_float(colors)
        colors_transformed = skimage.color.rgb2lab([colors]).squeeze()

        # Take elements row and column wise to operate on them as if they were
        # just two color vectors
        rows, cols = np.ogrid[:num_colors, :num_colors]
        colors_rows = np.take(colors_transformed, rows, axis=0)
        colors_cols = np.take(colors_transformed, cols, axis=0)

        matrix = OffsetSynth._metric_core(colors_rows, colors_cols)
        self._comparison_matrix = matrix

    @staticmethod
    def _metric_core(a, b):
        #return np.sum((a - b) ** 2, axis=2)
        return np.sum(np.abs(a - b), axis=2)

    def _prepare_sample_neighborhoods(self):
        sample_shape = self._sample_indexed.shape
        sample_size = sample_shape[0] * sample_shape[1]
        half_window = self._half_window
        nbrhoods = OffsetSynth._prepare_neighbor_indices(sample_shape,
                                                         half_window)
        nbrhoods = nbrhoods.reshape(sample_size, -1)
        self._sample_neighborhoods = nbrhoods

    @staticmethod
    def _prepare_neighbor_indices(shape, half_window):
        height, width = shape
        elements_total = height * width

        # Make a lookup array in order to handle with window border overlaps
        # (desired behavior is ensured by padding in 'reflect' mode):
        lookup_array = np.arange(elements_total).reshape(shape)
        lookup_array = np.pad(lookup_array, half_window, mode='reflect')

        # Take every possible block indices in the lookup array:
        window = 2 * half_window + 1
        indices = OffsetSynth._block_indices(lookup_array.shape, window)
        neighbor_indices = lookup_array.take(indices)

        return neighbor_indices

    @staticmethod
    def _block_indices(shape, block_size):
        """
        Returns list of indices of elements from all blocks created by sliding
        a (block_size x block_size) window across the input 2D matrix of given
        shape. Sliding is performed in the left-to-right and up-to-down manner
        with striding always by one element.
        :param shape: Shape of a two-dimensional matrix used for sliding.
        :param block_size: Size of a square sliding window.
        :return: 2D Array of raveled lists containing the array indices of all
        neighbors from a square block of given size centered at the
        corresponding 2D array position.
        """

        h, w = shape[0:2]
        col_extent = w - block_size + 1
        row_extent = h - block_size + 1

        # Prepare starting block indices:
        start_idx = (w * np.arange(block_size)[:, np.newaxis]
                     + np.arange(block_size))

        # Prepare offset indices:
        offset_idx = (w * np.arange(row_extent)[:, np.newaxis]
                      + np.arange(col_extent))

        # Compute indices of individual blocks, which will be of shape
        # (row_extent * col_extent, block_size ** 2):
        block_indices = start_idx.ravel() + offset_idx.ravel()[:, np.newaxis]

        # Reshape to the right size:
        block_indices = block_indices.reshape(row_extent, col_extent, -1)

        return block_indices

    def _prepare_jump_map(self):
        jump_map = []

        sample_height, sample_width = self._sample_indexed.shape
        sample_size = sample_height * sample_width

        for i in range(sample_size):
            color_class = self._sample_equiv[i]
            cand_inds = self._equiv_map[color_class]

            cand_vals = self._transform_candidates(cand_inds,
                                                   self._sample_neighborhoods,
                                                   self._sample_indexed_ravel)

            pixel_neighbors = self._sample_neighborhoods[i]
            best_inds = self._best_candidates(cand_vals, pixel_neighbors,
                                              self._num_nearest,
                                              self._sample_indexed_ravel,
                                              self._comparison_matrix)

            best_cands = cand_inds[best_inds]
            jump_map.append(best_cands)

        self._jump_map = jump_map

    @staticmethod
    def _transform_candidates(candidate_indices, sample_neighborhoods,
                              sample_indexed_raveled):
        neighbors_flat = sample_neighborhoods.take(candidate_indices,
                                                   axis=0).T.ravel()

        values_shape = (sample_neighborhoods.shape[1], len(candidate_indices))
        return sample_indexed_raveled[neighbors_flat].reshape(values_shape)

    @staticmethod
    def _best_candidates(candidate_values, pixel_neighbors, num_nearest,
                         sample_indexed_raveled, comparison_matrix):

        # Copy current pixel neighborhood values to be of the same shape as
        # the candidate values (enables vectorized computations):
        sample_elements = sample_indexed_raveled[pixel_neighbors]
        neighborhood_values = sample_elements[:, np.newaxis]

        # Compare candidate neighborhood pixel values with current pixel
        # neighborhood values (using precomputed matrix):
        num_colors = comparison_matrix.shape[0]
        matrix_indices = neighborhood_values + num_colors * candidate_values
        comparison = comparison_matrix.take(matrix_indices)
        candidate_distances = comparison.sum(axis=0)

        # Choose the best candidate as the one with minimal distance computed:
        num_candidates = min(num_nearest, len(candidate_distances))
        best_indices = candidate_distances.argsort()[:num_candidates]
        return best_indices

    def _generate_texture(self, out_shape):
        # Explicitly store the output image sizes:
        out_height, out_width = out_shape

        # Output array where indices to the input pixel values are stored:
        out_origins = np.zeros(out_shape, dtype=np.int64)
        out_indices = np.arange(out_height * out_width).reshape(out_shape)
        out_origins_mask = np.zeros(out_shape, dtype=np.bool)

        # Prepare input and output neighbor indices:
        sample_2d_shape = self._sample.shape[:2]
        sample_height, sample_width = sample_2d_shape[0], sample_2d_shape[1]
        sample_size = sample_height * sample_width

        # Fill the first pixel:
        out_origins[0, 0] = np.random.randint(sample_size)
        out_origins_mask[0, 0] = True

        """
        out_origins = np.arange(sample_size).reshape(sample_2d_shape)
        out_origins[12, 9] = 0
        arr = self._jump_map[887]
        for i in range(len(arr)):
            index = arr[i]
            r, c = index // 64, index % 64
            out_origins[r, c] = 771

        """
        for pix_row, pix_col in hilbert.generator(out_height, out_width,
                                                  start=1):

            last_row, last_col = self._find_nearest(pix_row, pix_col,
                                                    out_indices,
                                                    out_origins_mask)

            last_origin = out_origins[last_row, last_col]

            jump_index = self._choose_equiv_rnd(last_origin, 10)

            chosen_index = jump_index
            sample_row = chosen_index // sample_width
            sample_col = chosen_index % sample_width

            ccc_row = (pix_row - last_row + sample_row)
            ccc_col = (pix_col - last_col + sample_col)
            c_row = ccc_row % sample_height
            c_col = ccc_col % sample_width

            c_index = c_col + c_row * sample_width

            if ccc_row != c_row or ccc_col != c_col:
                jump_index = self._choose_equiv_rnd(last_origin, 0)

                chosen_index = jump_index
                sample_row = chosen_index // sample_width
                sample_col = chosen_index % sample_width

                ccc_row = (pix_row - last_row + sample_row)
                ccc_col = (pix_col - last_col + sample_col)
                c_row = ccc_row % sample_height
                c_col = ccc_col % sample_width

                c_index = c_col + c_row * sample_width

            out_origins[pix_row, pix_col] = c_index
            out_origins_mask[pix_row, pix_col] = True

        return self._transform_origins(out_origins)

    def _find_nearest(self, pix_row, pix_col, out_indices, out_origins_mask):
        hw = 1

        h, w = out_indices.shape

        min_row = max(0, pix_row - hw)
        max_row = min(h, pix_row + hw + 1)

        min_col = max(0, pix_col - hw)
        max_col = min(w, pix_col + hw + 1)

        sub_indices = out_indices[min_row:max_row, min_col:max_col]
        sub_mask = out_origins_mask[min_row:max_row, min_col:max_col]

        nearest_filled = sub_indices[sub_mask]
        maximum_index = nearest_filled.shape[0]
        random_index = np.random.randint(maximum_index)
        nearest_index = nearest_filled[random_index]
        row, col = nearest_index // w, nearest_index % w
        return row, col

    def _choose_equiv_rnd(self, sample_index, threshold=10.95):
        jump_map = self._jump_map

        if np.random.rand() < threshold:
            return sample_index

        indices = jump_map[sample_index]
        rnd_index = np.random.randint(len(indices))
        return indices[rnd_index]

    def _transform_origins(self, out_origins):
        sample = self._sample
        sample_height, sample_width = sample.shape[:2]
        sample_size = sample_height * sample_width
        sample = sample.reshape(sample_size, -1)
        out_image = np.take(sample, out_origins, axis=0)

        return out_image


def main():
    file_in = '../../imgs/textures/samples/classical1noper.png'
    file_out = '../../imgs/textures/synthesized/output.png'

    synthesizer = OffsetSynth().random_seed(0).half_window(12).num_nearest(4)
    out_shape = (256, 256)
    synthesizer.generate_texture(file_in, file_out, out_shape)

if __name__ == '__main__':
    import json
    import logging.config
    json_file = open('../logging.json')
    config = json.load(json_file)
    logging.config.dictConfig(config)
    logging.info('Welcome to Offset Texture Synthesizer')

    main()
