import logging
import numba
import numpy as np
import skimage.color
import skimage.io
import skimage.util

import omrsynth.util.io as io


class TextureSynth(object):
    _quantized_colors = 256

    def __init__(self):
        # Fill default synthesizer parameters:
        self._half_window = 2
        self._initial_random_points = 8
        self._nearest_candidates = 8
        self._random_candidates = 20
        self._refinement_level = 3
        self._impurities = 0.5
        self._random_seed = 0
        np.random.seed(self._random_seed)
        
    def half_window(self, value):
        self._half_window = value
        return self
    
    def initial_random_points(self, value):
        self._initial_random_points = value
        return self
    
    def nearest_candidates(self, value):
        self._nearest_candidates = value
        return self
    
    def random_candidates(self, value):
        self._random_candidates = value
        return self

    def refinement_level(self, value):
        self._refinement_level = value
        return self

    def impurities(self, value):
        self._impurities = value
        return self

    def random_seed(self, value):
        self._random_seed = value
        # Initialize NumPy's random:
        np.random.seed(value)
        return self

    def generate_texture(self, sample_filename, output_filename, out_shape):
        self._load_sample(sample_filename)
        texture = self._generate_texture(out_shape)
        io.save_image(output_filename, texture)

    def _load_sample(self, filename):
        num_colors = TextureSynth._quantized_colors
        sample, indexed, labels = io.load_and_quantize(filename, num_colors)

        self._sample = sample
        self._sample_indexed = indexed
        self._index_labels = labels

        # Load sample quantized in two colors by maximum coverage algorithm and
        # determine (and store) where the impurities are located:
        max_coverage = 1
        _, bitonal, _ = io.load_and_quantize(filename, num_colors=2,
                                             method=max_coverage)

        # Assuming count of the impurity pixels ("foreground") is < count of
        # the clean paper pixels ("background"):
        num_zeros, num_ones = (bitonal == 0).sum(), (bitonal == 1).sum()
        impurities_index = 0 if num_zeros < num_ones else 1
        self._fg_indices = np.where(bitonal.ravel() == impurities_index)[0]
        self._bg_indices = np.where(bitonal.ravel() != impurities_index)[0]

    def _generate_texture(self, out_shape):
        # Explicitly store the output image sizes:
        out_height, out_width = out_shape
        out_size = out_height * out_width

        # Generate an order in which individual pixels will be iterated:
        pixel_order = np.random.permutation(out_size)
        import omrsynth.util.hilbert as hilbert
        pixel_order = [(row * out_width + col) for row, col in hilbert.generator(out_height, out_width, self._initial_random_points)]

        # Showing the progress about ten times in one iteration:
        progress_each = out_size // 10

        # Allocating variables out of the main loop for best performance.

        # Prepare candidate array:
        candidates_total = self._nearest_candidates + self._random_candidates
        candidate_indices = np.empty((candidates_total,), dtype=np.int64)

        # Output array where indices to the input pixel values are stored:
        out_origins = np.zeros(out_shape, dtype=np.int64)
        # Mask indicating where the output pixels are already filled:
        out_origins_mask = np.zeros(out_shape, dtype=bool)
        self._fill_initial_pixels(out_origins, out_origins_mask, pixel_order)

        # Array where candidates pixels will be stored:
        half_window = self._half_window
        window_size = 2 * half_window + 1
        window_elements = window_size ** 2
        candidate_values_shape = (window_elements, candidates_total)
        candidate_values = np.empty(candidate_values_shape, dtype=np.int64)
        neighborhood_values = np.empty(candidate_values_shape, dtype=np.int64)

        # Prepare input and output neighbor indices:
        sample_2d_shape = self._sample.shape[:2]
        sample_neighbors = self._prepare_neighbor_indices(sample_2d_shape,
                                                          half_window)
        sample_size = sample_2d_shape[0] * sample_2d_shape[1]
        sample_neighbors = sample_neighbors.reshape(sample_size, -1)
        out_neighbors = self._prepare_neighbor_indices(out_shape, half_window)
        logging.info('Neighborhood arrays prepared.')

        # Quantized sample image raveled for easier indexing:
        sample_indexed_raveled = self._sample_indexed.ravel()

        comparison_matrix = self._prepare_metric(self._index_labels)
        logging.info('Color comparison matrix prepared.')

        impurities_threshold = self._impurities_threshold()
        logging.info('Impurities threshold computed.')

        # Perform initial (zeroth) filling iteration and then the refinements:
        num_iterations = self._refinement_level + 1
        for iteration in range(num_iterations):
            logging.info('Iteration No. {0}'.format(iteration))
            traversed_pixels = 0

            # Loop through all the output pixels in a random order:
            min_index = self._initial_random_points if iteration == 0 else 0
            for index in pixel_order[min_index:]:
                # Log the progress (about ten times in one iteration):
                if traversed_pixels % progress_each == 0:
                    percent = 100.0 * traversed_pixels / out_size
                    message = 'Processed {:.2f} % of pixels in this iteration.'
                    logging.info(message.format(percent))

                # Prepare pixel indices and neighbors:
                pix_row, pix_col = index // out_width, index % out_width
                pixel_neighbors = out_neighbors[pix_row, pix_col]
                pixel_mask = TextureSynth._pixel_mask(out_origins_mask,
                                                      pixel_neighbors)

                # If the nearest neighborhood is not already filled, choose
                # a random sample according the impurities distribution:
                if False: # (~pixel_mask).all():
                    pixel_origin = self._choose_randomly(impurities_threshold)

                # If some of the neighborhood pixel is already filled, choose
                # the candidate by "best-fit" method:
                else:
                    # Find suitable candidate pixels (their indices pointing to
                    # the raveled sample image array):
                    self._fill_candidate_indices(pix_row, pix_col,
                                                 candidate_indices,
                                                 out_origins,
                                                 out_origins_mask)

                    # Transform the indices into 2D array of their pixel values
                    # including the pixels in their neighborhood. Shape of the
                    # values array is (number of pixels in the neighborhood,
                    # number of the candidate pixels), each pixel value is
                    # represented by its quantized color index.
                    self._transform_candidates(candidate_values,
                                               candidate_indices,
                                               sample_neighbors,
                                               sample_indexed_raveled)

                    # Choose the best pixel candidate according to the
                    # precomputed metric for indexed colors:
                    chosen = self._best_candidate(candidate_values,
                                                  neighborhood_values,
                                                  pixel_neighbors, out_origins,
                                                  pixel_mask,
                                                  sample_indexed_raveled,
                                                  comparison_matrix)

                    pixel_origin = candidate_indices[chosen]

                    #if True:#(~pixel_mask).all():
                        #pixel_origin = candidate_indices[iteration % 8]
                    #    pixel_origin = candidate_indices[0]

                # In both cases, update currently processed pixel:
                out_origins[pix_row, pix_col] = pixel_origin
                out_origins_mask[pix_row, pix_col] = True

                traversed_pixels += 1

            logging.debug('Iteration No. {0} done!'.format(iteration))
            ############################################################
            # TEMPORARY PEEKING THE INTERMEDIATE RESULTS
            file_out = '../../imgs/textures/synthesized/out_iter{}.png'\
                .format(iteration)

            image = self._transform_origins(out_origins)
            skimage.io.imsave(file_out, image)
            ############################################################

        return self._transform_origins(out_origins)

    def _fill_initial_pixels(self, out_origins, out_origins_mask, pixel_order):
        sample_height, sample_width = self._sample_indexed.shape
        sample_size = sample_height * sample_width
        num_pixels = self._initial_random_points

        initial_indices = pixel_order[:num_pixels]
        initial_indices = np.unravel_index(initial_indices, out_origins.shape)

        random_pixels = np.random.permutation(sample_size)[:num_pixels]
        out_origins[initial_indices] = random_pixels
        out_origins_mask[initial_indices] = True

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

    def _impurities_threshold(self):
        fg_pixels = len(self._fg_indices)
        bg_pixels = len(self._bg_indices)

        pixels_total = float(fg_pixels + bg_pixels)
        fg_prob = fg_pixels / pixels_total
        bg_prob = bg_pixels / pixels_total

        # Piecewise linear interpolation:
        # TODO: More comments.
        x = self._impurities
        if x < 0.5:
            return 2 * fg_prob * x
        else:
            return bg_prob * (2 * x - 1) + fg_prob

    def _choose_randomly(self, threshold):

        if np.random.rand() <= threshold:
            # Sample from the foreground pixels:
            max_index = len(self._fg_indices)
            random_index = np.random.randint(max_index)
            return self._fg_indices[random_index]
        else:
            # Sample from the background pixels:
            max_index = len(self._bg_indices)
            random_index = np.random.randint(max_index)
            return self._bg_indices[random_index]

    @staticmethod
    def _prepare_metric(colors):
        assert colors is not None, 'Indexed color labels must be filled!'
        logging.debug('Preparing color comparison matrix.')

        num_colors = colors.shape[0]

        # Transform the colors:
        # TODO: Try Lab color space?
        colors_transformed = skimage.util.img_as_float(colors)
        #colors_transformed = skimage.color.rgb2lab([colors]).squeeze()

        # Take elements row and column wise to operate on them as if they were
        # just two color vectors
        rows, cols = np.ogrid[:num_colors, :num_colors]
        colors_rows = np.take(colors_transformed, rows, axis=0)
        colors_cols = np.take(colors_transformed, cols, axis=0)

        matrix = TextureSynth._metric_core(colors_rows, colors_cols)
        return matrix

    @staticmethod
    def _metric_core(a, b):
        #return np.sum((a - b) ** 2, axis=2)
        sigma_inv = 1.0 / 20.0
        return np.log(np.prod(1 + sigma_inv * (a - b) ** 2, axis=2))

    def _fill_candidate_indices(self, pix_row, pix_col, candidate_indices,
                                out_origins, out_origins_mask):

        # Extract input sizes:
        sample_height, sample_width = self._sample_indexed.shape
        sample_size = sample_height * sample_width

        # Find n nearest pixels that are already filled:
        num_nearest = self._nearest_candidates
        TextureSynth._find_nearest(candidate_indices, pix_row, pix_col,
                                   out_origins, out_origins_mask,
                                   sample_height, sample_width, num_nearest)

        # Additional random candidates:
        num_random = self._random_candidates
        random_indices = np.random.randint(0, sample_size, num_random)
        candidate_indices[num_nearest:] = random_indices

    @staticmethod
    @numba.jit(numba.void(numba.int64[:], numba.int64, numba.int64,
                          numba.int64[:, :], numba.boolean[:, :], numba.int64,
                          numba.int64, numba.int64), nopython=True)
    def _find_nearest(candidates, pix_row, pix_col, out_origins,
                      out_origins_mask, sample_height, sample_width,
                      num_nearest):

        # Extract output image sizes:
        out_height, out_width = out_origins.shape

        neighbors_found = 0
        radius = 1

        x_pos, y_pos = np.empty(4, dtype=np.int64), np.empty(4, dtype=np.int64)

        # Search the neighborhood while there are not enough filled pixels:
        while neighbors_found < num_nearest:
            # Initialize actual pixel position:
            x_pos[:] = pix_col
            x_pos[0:2] -= radius
            x_pos[2:4] += radius

            y_pos[:] = pix_row
            y_pos[0] -= radius
            y_pos[3] -= radius
            y_pos[1] += radius
            y_pos[2] += radius

            # Travers all the pixels in given radius from the origin pixel:
            for k in range(2 * radius):
                x_pos[x_pos >= out_width] = out_width - 1
                x_pos[x_pos < 0] = 0

                y_pos[y_pos >= out_height] = out_height - 1
                y_pos[y_pos < 0] = 0

                #x_pos %= out_width
                #y_pos %= out_height

                for d in range(4):
                    col, row = x_pos[d], y_pos[d]

                    # If the pixel is already filled
                    if out_origins_mask[row, col]:
                        # Compute correct candidate index:
                        sample_index = out_origins[row, col]
                        sample_row = sample_index // sample_width
                        sample_col = sample_index % sample_width

                        c_row = (sample_row - row + pix_row) % sample_height
                        c_col = (sample_col - col + pix_col) % sample_width
                        c_index = c_col + c_row * sample_width

                        # Store the candidate index:
                        candidates[neighbors_found] = c_index
                        neighbors_found += 1

                    if neighbors_found >= num_nearest:
                        return

                x_pos[1] += 1
                x_pos[3] -= 1
                y_pos[0] += 1
                y_pos[2] -= 1

            radius += 1

    @staticmethod
    def _transform_candidates(candidate_values, candidate_indices,
                              sample_neighborhoods, sample_indexed_raveled):

        neighbors_flat = sample_neighborhoods.take(candidate_indices,
                                                   axis=0).T.ravel()

        candidate_values[:] = sample_indexed_raveled[neighbors_flat]\
            .reshape(candidate_values.shape)

    @staticmethod
    def _pixel_mask(out_origins_mask, pixel_neighbors):
        # Prepare boolean mask of neighbors of the current pixel and do not
        # include the pixel itself:
        px_mask = out_origins_mask.take(pixel_neighbors)
        central_index = (len(px_mask) - 1) // 2
        px_mask[central_index] = False
        px_mask = px_mask[:, np.newaxis]
        return px_mask

    @staticmethod
    def _best_candidate(candidate_values, neighborhood_values, pixel_neighbors,
                        out_origins, pixel_mask, sample_indexed_raveled,
                        comparison_matrix):

        # Copy current pixel neighborhood values to be of the same shape as
        # the candidate values (enables vectorized computations):
        sample_indices = out_origins.take(pixel_neighbors)
        sample_elements = sample_indexed_raveled[sample_indices]
        neighborhood_values[:, :] = sample_elements[:, np.newaxis]

        # Compare candidate neighborhood pixel values with current pixel
        # neighborhood values (using precomputed matrix):
        num_colors = comparison_matrix.shape[0]
        matrix_indices = neighborhood_values + num_colors * candidate_values
        comparison = comparison_matrix.take(matrix_indices)

        # Multiply with boolean mask (filled pixels) in order not to include
        # non-filled pixels in distance computations and sum the contributions:
        candidate_distances = (comparison * pixel_mask).sum(axis=0)

        # TODO DONT FORGET
        candidate_distances += 0.001 * np.random.rand(len(candidate_distances))

        # Choose the best candidate as the one with minimal distance computed:
        best_candidate = np.argmin(candidate_distances)
        return best_candidate

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

    synthesizer = TextureSynth()\
        .initial_random_points(1)\
        .nearest_candidates(8)\
        .random_seed(0)\
        .half_window(12)\
        .random_candidates(8)\
        .impurities(0.5)\
        .refinement_level(0)

    out_shape = (128, 128)
    synthesizer.generate_texture(file_in, file_out, out_shape)

if __name__ == '__main__':
    import json
    import logging.config
    json_file = open('../logging.json')
    config = json.load(json_file)
    logging.config.dictConfig(config)

    main()
