import logging
import numpy as np
import skimage.color
import skimage.morphology

import omrsynth.util.io as io
from omrsynth.paper.synthesizer import Synthesizer


class TextureSynth(Synthesizer):
    _sim_colors = 128
    _border_awareness = 0.1
    _max_candidates = 64

    def __init__(self):
        super(TextureSynth, self).__init__()
        self._half_window = 1
        self._impurities = 1.0
        self._verbatim_size = 10 ** 2

    def half_window(self, value):
        self._half_window = value
        return self

    def impurities(self, value):
        value = min(1, max(value, 0))
        self._impurities = value
        return self

    def _load_sample(self, filename):
        super(TextureSynth, self)._load_sample(filename)
        self._impurities_init(filename)
        self._prepare_equivalence_map(filename)
        self._prepare_metric()
        self._prepare_sample_neighborhoods()

    def _prepare_equivalence_map(self, filename):
        logging.debug('Loading image for the equivalence map.')
        _, equiv, l = io.load_and_quantize(filename,
                                           num_colors=TextureSynth._sim_colors,
                                           method=io.MAX_COVERAGE)

        # Save equivalence image for visual comparison
        equiv_image = np.take(l, equiv, axis=0)
        equiv_text = 'equivalence_{0}_colors'.format(TextureSynth._sim_colors)
        filename_equiv = Synthesizer._add_to_filename(filename, equiv_text)
        io.save_image(filename_equiv, equiv_image)

        equiv = equiv.ravel()
        equiv_map = []

        logging.info('Computing equivalence map.')
        for i in range(TextureSynth._sim_colors):
            indices = np.where(equiv == i)[0]
            equiv_map.append(indices)

        self._equiv_map = equiv_map
        self._sample_equiv = equiv

        logging.info('Equivalence map computed.')

    def _prepare_metric(self):
        colors = self._index_labels
        assert colors is not None, 'Indexed color labels must be filled!'
        logging.debug('Preparing color comparison matrix.')

        num_colors = colors.shape[0]

        # Transform the colors
        colors_transformed = skimage.color.rgb2lab([colors]).squeeze()

        # Take elements row and column wise to operate on them as if they were
        # just two color vectors
        rows, cols = np.ogrid[:num_colors, :num_colors]
        colors_rows = np.take(colors_transformed, rows, axis=0)
        colors_cols = np.take(colors_transformed, cols, axis=0)

        matrix = TextureSynth._metric_core(colors_rows, colors_cols)
        self._comparison_matrix = matrix
        logging.info('Color comparison matrix prepared.')

    @staticmethod
    def _metric_core(a, b):
        return np.sum((a - b) ** 2, axis=2)

    def _prepare_sample_neighborhoods(self):
        sample_shape = self._sample_indexed.shape
        sample_size = sample_shape[0] * sample_shape[1]
        half_window = self._half_window
        nbrhoods = TextureSynth._neighbor_indices(sample_shape,
                                                  half_window)
        nbrhoods = nbrhoods.reshape(sample_size, -1)
        self._sample_neighborhoods = nbrhoods

    @staticmethod
    def _neighbor_indices(shape, half_window):
        height, width = shape
        elements_total = height * width

        # Make a lookup array in order to handle with window border overlaps
        # (desired behavior is ensured by padding in 'reflect' mode)
        lookup_array = np.arange(elements_total).reshape(shape)
        lookup_array = np.pad(lookup_array, half_window, mode='reflect')

        # Take every possible block indices in the lookup array
        window = 2 * half_window + 1
        indices = TextureSynth._block_indices(lookup_array.shape, window)
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

        # Prepare starting block indices
        start_idx = (w * np.arange(block_size)[:, np.newaxis]
                     + np.arange(block_size))

        # Prepare offset indices
        offset_idx = (w * np.arange(row_extent)[:, np.newaxis]
                      + np.arange(col_extent))

        # Compute indices of individual blocks, which will be of shape
        # (row_extent * col_extent, block_size ** 2)
        block_indices = start_idx.ravel() + offset_idx.ravel()[:, np.newaxis]

        # Reshape to the right size:
        block_indices = block_indices.reshape(row_extent, col_extent, -1)

        return block_indices

    def _impurities_init(self, filename):
        # Load sample quantized in two colors by maximum coverage algorithm and
        # determine (and store) where the impurities are located
        _, bitonal, _ = io.load_and_quantize(filename, num_colors=2,
                                             method=io.MAX_COVERAGE)

        # Assuming count of the impurity pixels ("foreground") is less than
        # count of the clean paper pixels ("background")
        num_zeros, num_ones = (bitonal == 0).sum(), (bitonal == 1).sum()
        impurities_index = 0 if num_zeros < num_ones else 1

        # Create impurity map
        impurities_map = np.zeros_like(self._sample_indexed, dtype=np.bool)
        impurities_map[bitonal == impurities_index] = True

        # Dilate a little bit to better encompass the impurities
        impurities_map = skimage.morphology.binary_dilation(impurities_map)
        self._impurities_map_ravel = impurities_map.ravel()

        # Store foreground and background indices
        self._fg_indices = np.where(impurities_map.ravel())[0]
        self._bg_indices = np.where(~impurities_map.ravel())[0]

        # Save impurities map
        filename = TextureSynth._add_to_filename(filename, 'impurities')
        io.save_image(filename, 255 * impurities_map)

        logging.info('Map of impurities computed.')

    def _generate_texture(self, out_shape):
        # Explicitly store the output image sizes
        out_height, out_width = out_shape

        # Output array where indices to the input pixel values are stored
        out_origins = np.zeros(out_shape, dtype=np.int64)
        out_indices = np.arange(out_height * out_width).reshape(out_shape)
        out_origins_mask = np.zeros(out_shape, dtype=np.bool)

        out_neighbors = TextureSynth._neighbor_indices(out_shape,
                                                       self._half_window)
        logging.info('The output neighborhood array prepared.')

        sample_2d_shape = self._sample.shape[:2]
        sample_height, sample_width = sample_2d_shape[0], sample_2d_shape[1]

        # Fill the first pixel
        self._first_pixel(out_origins, out_origins_mask, 0, 0)
        last_out_index = 0

        # Show progress info approx. each ten percents
        traversed_pixels = -1
        out_size = out_height * out_width
        progress_each = out_size // 10

        pixel_order = self._pixel_generator(out_height, out_width, 1)
        for pix_row, pix_col in pixel_order:
            # Progress message
            traversed_pixels += 1
            if traversed_pixels % progress_each == 0:
                percent = 100.0 * traversed_pixels / out_size
                message = 'Generated {:.2f} % of pixels.'
                logging.info(message.format(percent))

            near_row, near_col = self._find_nearest(pix_row, pix_col,
                                                    out_indices,
                                                    out_origins_mask,
                                                    last_out_index)
            # Mark visited pixel of output array
            last_out_index = pix_row * out_width + pix_col

            near_origin = out_origins[near_row, near_col]

            if self._should_jump(near_origin, sample_width, sample_height):
                chosen_index = self._choose_equiv_best(near_row, near_col,
                                                       out_origins,
                                                       out_origins_mask,
                                                       out_neighbors)
            else:
                chosen_index = near_origin

            # Compute index of possible new origin
            sample_row = chosen_index // sample_width
            sample_col = chosen_index % sample_width
            n_row = (pix_row - near_row + sample_row) % sample_height
            n_col = (pix_col - near_col + sample_col) % sample_width
            n_index = n_col + n_row * sample_width

            was_paper = self._is_background(near_origin)
            is_impurity = self._is_foreground(n_index)

            if self._impurities < 1.0:
                # Restrict background -> foreground pixel transitions
                if was_paper and is_impurity:
                    if np.random.rand() > self._impurities:
                        n_index = self._choose_background(near_origin)

            out_origins[pix_row, pix_col] = n_index
            out_origins_mask[pix_row, pix_col] = True

        return out_origins

    def _first_pixel(self, out_origins, out_origins_mask, row, col):
        sample = self._sample_pixel()
        out_origins[row, col] = sample
        out_origins_mask[row, col] = True

    def _find_nearest(self, row, col, out_indices, out_origins_mask,
                      last_out_index):
        sub_indices = self._neighborhood(out_indices, row, col)
        sub_mask = self._neighborhood(out_origins_mask, row, col)

        nearest_filled = sub_indices[sub_mask]
        if nearest_filled.size == 0:
            # If the chosen pixel order is non-contiguous (e.g. Hilbert, iff
            # both output dimensions are not a power of two), the nearest
            # filled pixel may be beyond the one pixel neighborhood. In that
            # case, heuristically choose the last filled pixel index
            nearest_index = last_out_index
            pass
        else:
            # Choose randomly from the nearest pixels
            nearest_index = np.random.choice(nearest_filled)

        w = out_indices.shape[1]
        return nearest_index // w, nearest_index % w

    @staticmethod
    def _neighborhood(array, row, col, half_window=1):
        h, w = array.shape

        min_row = max(0, row - half_window)
        max_row = min(h, row + half_window + 1)

        min_col = max(0, col - half_window)
        max_col = min(w, col + half_window + 1)

        return array[min_row:max_row, min_col:max_col]

    def _should_jump(self, near_origin, sample_width, sample_height):
        sample_row = near_origin // sample_width
        sample_col = near_origin % sample_width

        min_row_dist = min(sample_row, sample_height - 1 - sample_row)
        min_col_dist = min(sample_col, sample_width - 1 - sample_col)
        min_border_dist = min(min_row_dist, min_col_dist)

        border = self._border_awareness * min(sample_width, sample_height)
        # Increase jump probability if approaching the border
        if min_border_dist <= border:
            if min_border_dist * np.random.rand() < 1:
                return True

        # Regular jumping to prevent large verbatim copies
        if self._verbatim_size * np.random.rand() < 1:
            return True

        return False

    def _is_background(self, origin):
        return ~self._impurities_map_ravel[origin]

    def _is_foreground(self, origin):
        return self._impurities_map_ravel[origin]

    def _sample_pixel(self):
        threshold = self._impurities

        if np.random.rand() <= threshold:
            # Sample from all the pixels
            max_index = len(self._sample_indexed_ravel)
            random_origin = np.random.randint(max_index)
            return random_origin
        else:
            # Sample from the background pixels only
            max_index = len(self._bg_indices)
            random_index = np.random.randint(max_index)
            return self._bg_indices[random_index]

    def _choose_background(self, sample_origin):
        return self._choose_equiv_rnd(sample_origin)

    def _choose_equiv_rnd(self, sample_index):
        sample_color = self._sample_equiv[sample_index]
        indices = self._equiv_map[sample_color]
        return np.random.choice(indices)

    def _choose_equiv_best(self, row, col, out_origins, out_origins_mask,
                           out_neighbors):

        sample_index = out_origins[row, col]
        sample_color = self._sample_equiv[sample_index]
        indices = self._equiv_map[sample_color]

        max_candidates = TextureSynth._max_candidates
        num_to_choose = min(max_candidates, len(indices))
        candidates = np.random.choice(indices, num_to_choose)
        values = TextureSynth._transform_candidates(candidates,
                                                    self._sample_neighborhoods,
                                                    self._sample_indexed_ravel)

        px_neighbors = out_neighbors[row, col]
        px_mask = out_origins_mask.take(px_neighbors)[:, np.newaxis]
        best = TextureSynth._best_candidates(values, px_neighbors, px_mask,
                                             out_origins,
                                             self._sample_indexed_ravel,
                                             self._comparison_matrix)
        return candidates[best]

    @staticmethod
    def _transform_candidates(candidate_indices, sample_neighborhoods,
                              sample_indexed_raveled):
        neighbors_flat = sample_neighborhoods.take(candidate_indices,
                                                   axis=0).T.ravel()

        values_shape = (sample_neighborhoods.shape[1], len(candidate_indices))
        return sample_indexed_raveled[neighbors_flat].reshape(values_shape)

    @staticmethod
    def _best_candidates(candidate_values, pixel_neighbors, pixel_mask,
                         out_origins, sample_indexed_raveled,
                         comparison_matrix):

        # Copy current pixel neighborhood values to be of the same shape as
        # the candidate values (enables vectorized computations)
        sample_indices = out_origins.take(pixel_neighbors)
        sample_elements = sample_indexed_raveled[sample_indices]
        neighborhood_values = sample_elements[:, np.newaxis]

        # Compare candidate neighborhood pixel values with current pixel
        # neighborhood values (using precomputed matrix)
        num_colors = comparison_matrix.shape[0]
        matrix_indices = neighborhood_values + num_colors * candidate_values
        comparison = comparison_matrix.take(matrix_indices)
        candidate_distances = (comparison * pixel_mask).sum(axis=0)

        # Choose the best candidate as the one with minimal distance computed
        return candidate_distances.argmin()


def single():
    file_in = '../../imgs/sandbox/samples/lq02.png'
    file_out = '../../imgs/sandbox/synthesized/output.png'

    synthesizer = TextureSynth().random_seed(10).impurities(1.0)
    out_shape = (1809, 3495)
    synthesizer.generate_texture(file_in, file_out, out_shape)


def multiple():
    import os.path

    files = ['lq02', 'lq04', 'lq05', 'lq01', 'hq04', 'hq06', 'janacek',
             'lombard', 'grieg', 'hq01']

    impurities = [0.0, 0.25, 0.5, 0.75, 1.0]
    path = '../../imgs/backgrounds/'
    out_shape = (1024, 1024)

    for f in files:
        directory = os.path.join(path, f)
        if not os.path.isdir(directory):
            os.mkdir(directory)

        file_in = os.path.join(path, f + '.png')
        synthesizer = TextureSynth()
        synthesizer.load_sample(file_in)

        for imp in impurities:
            logging.info('File: {0}, impurity level: {1}'.format(f, imp))

            name_out = '{0}_impurity_level_{1}.png'.format(f, imp)
            file_out = os.path.join(directory, name_out)
            synthesizer.impurities(imp)
            synthesizer.generate_loaded_texture(file_out, out_shape)


def pixel_cnn_backgrounds():
    import os.path

    path_samples = '/media/jirka/DATA/MFF/OMR Data/PXCNN/samples'
    path_resources = '/media/jirka/DATA/MFF/OMR Data/PXCNN/dalitz'
    path_output = '/media/jirka/DATA/MFF/OMR Data/PXCNN/backgrounds'

    samples = ['dvorak', 'empty', 'fibich_poem', 'fibich_vodnik', 'gershwin',
               'satie']

    resources = ['bach', 'bellinzani', 'brahms02', 'bruckner01', 'buxtehude',
                 'carcassi01', 'dalitz03', 'diabelli', 'mahler', 'pmw01',
                 'pmw03', 'pmw04', 'rameau', 'schumann', 'tye', 'victoria09',
                 'wagner', 'williams']

    extension = '.png'

    counter = 0
    for sample in samples:
        file_sample = os.path.join(path_samples, sample + extension)
        synthesizer = TextureSynth()
        synthesizer.load_sample(file_sample)

        for res in resources:
            percent = 100.0 * counter / (len(samples) * len(resources))
            message = '{0:.2f} %: sample {1}, resource {2}'
            logging.info(message.format(percent, sample, res))

            file_resource = os.path.join(path_resources, res + extension)
            out_shape = io.image_dims(file_resource)

            name_out = '{0}_{1}.png'.format(res, sample)
            file_out = os.path.join(path_output, name_out)
            synthesizer.random_seed(counter)
            synthesizer.generate_loaded_texture(file_out, out_shape)

            counter += 1


def main():
    pixel_cnn_backgrounds()

if __name__ == '__main__':
    import json
    import logging.config
    json_file = open('../logging.json')
    config = json.load(json_file)
    logging.config.dictConfig(config)
    logging.info('Welcome to Texture Synthesizer')

    main()
